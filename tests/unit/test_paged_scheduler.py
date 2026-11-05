"""Paged KV Cache Continuous Batching 的端到端正确性测试（CPU，tiny 模型，float32，贪心解码）。

核心判据与 test_continuous.py 一致：每个请求的输出 Token 必须与该请求单独串行 generate 逐位一致，
并与 Dense 槽位路径逐位一致；此外校验块池的生命周期不变量。
"""

import pytest
import torch

from mini_infra.core import PagedCacheConfig, make_request_specs, required_pool_blocks
from mini_infra.engine import (
    BatchingMethod,
    FeatureFlag,
    GenerationOptions,
    Mode,
    SchedulePhase,
    generate,
    run_scheduler,
)

CPU = torch.device("cpu")
KV_MODE = Mode({FeatureFlag.KV_CACHE})
PAGED_MODE = Mode({FeatureFlag.KV_CACHE, FeatureFlag.PAGED_KV_CACHE}, batching_method=BatchingMethod.CONTINUOUS)


@pytest.fixture
def model(diverse_model):
    return diverse_model


def make_prompts(lengths: list[int]) -> list[list[int]]:
    return [[(i * 7 + j * 3) % 90 + 1 for j in range(length)] for i, length in enumerate(lengths)]


def serial_outputs(model, prompts, budgets, eos_token_id=-1, stop_on_eos=False) -> list[list[int]]:
    opts = GenerationOptions(eos_token_id=eos_token_id, device=CPU, stop_on_eos=stop_on_eos)
    return [
        generate(KV_MODE, model, [prompt], budget, options=opts).sequences[0].token_ids
        for prompt, budget in zip(prompts, budgets)
    ]


def run_paged(model, specs, capacity, eos=-1, stop_on_eos=False, **config):
    return run_scheduler(
        BatchingMethod.CONTINUOUS,
        model,
        specs,
        capacity,
        eos,
        CPU,
        stop_on_eos=stop_on_eos,
        paged_cache_config=PagedCacheConfig(**config),
    )


CASES = {
    "mixed_budgets": ([4, 9, 6, 3, 7], [2, 6, 3, 5, 1], [0.0] * 5, 2),
    "staggered": ([5, 5, 5, 5], [4, 6, 4, 6], [0.0, 1.0, 2.0, 5000.0], 3),
    "longer_prompt_joins_late": ([3, 12, 6], [8, 4, 3], [0.0, 0.0001, 0.0002], 2),
    "budget_one": ([4, 6, 5], [1, 1, 3], [0.0, 0.0, 0.0], 3),
    "capacity_not_binding": ([4, 7], [5, 3], [0.0, 0.0], 8),
    "single_request": ([6], [5], [0.0], 1),
}


@pytest.mark.parametrize("block_size", [1, 3, 4, 16])
@pytest.mark.parametrize("case", list(CASES))
def test_outputs_match_serial_generate_and_dense(model, case, block_size):
    lengths, budgets, arrivals, capacity = CASES[case]
    prompts = make_prompts(lengths)
    specs = make_request_specs(prompts, budgets, arrivals)

    paged = run_paged(model, specs, capacity, block_size=block_size)
    dense = run_scheduler(BatchingMethod.CONTINUOUS, model, specs, capacity, -1, CPU, stop_on_eos=False)

    assert paged.is_paged_kv_cache and not dense.is_paged_kv_cache
    assert [s.token_ids for s in paged.sequences] == serial_outputs(model, prompts, budgets)
    assert [s.token_ids for s in paged.sequences] == [s.token_ids for s in dense.sequences]
    assert [s.output_len for s in paged.sequences] == budgets
    assert all(s.finish_reason == "max_tokens" for s in paged.sequences)


def test_eos_stops_request_and_releases_its_blocks(model):
    prompts = make_prompts([4, 6, 5])
    budgets = [8, 8, 8]
    eos = serial_outputs(model, prompts, budgets)[0][2]
    specs = make_request_specs(prompts, budgets)

    result = run_paged(model, specs, 2, eos, stop_on_eos=True, block_size=4)

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, budgets, eos, stop_on_eos=True)
    assert result.sequences[0].finish_reason == "stop"


def test_timeline_and_event_invariants(model):
    lengths, budgets, arrivals, capacity = CASES["mixed_budgets"]
    specs = make_request_specs(make_prompts(lengths), budgets, arrivals)

    result = run_paged(model, specs, capacity, block_size=4)
    decode_events = [e for e in result.events if e.phase is SchedulePhase.DECODE]

    for request, budget in zip(result.request_metrics, budgets):
        assert request.arrival_ms <= request.admitted_ms <= request.first_token_ms <= request.completion_ms
        assert request.num_output_tokens == budget == len(request.token_times_ms)
    assert sum(e.num_generated_tokens for e in result.events) == sum(budgets)
    assert sorted(i for e in result.events for i in e.admitted_request_ids) == list(range(len(specs)))
    assert sorted(i for e in result.events for i in e.finished_request_ids) == list(range(len(specs)))
    # Paged Decode 只执行活跃请求：无 Padding 行，执行槽位利用率恒为 1
    assert all(e.physical_batch_size == e.num_active_requests for e in decode_events)
    assert result.batch_info.execution_slot_utilization == pytest.approx(1.0)
    assert result.schedule_metrics.busy_ms == pytest.approx(sum(e.step_ms for e in result.events))
    assert result.memory.kv_cache_mb > 0


def test_live_kv_bytes_track_used_blocks_and_stay_within_pool(model):
    block_size = 4
    specs = make_request_specs(make_prompts([5, 5]), [6, 6], [0.0, 0.0])
    config = model.config
    block_bytes = 2 * config.num_hidden_layers * block_size * config.num_key_value_heads * config.head_dim * 4

    result = run_paged(model, specs, 2, block_size=block_size)

    assert all(e.live_kv_cache_bytes % block_bytes == 0 for e in result.events)
    pool_blocks = required_pool_blocks(specs, 2, block_size)
    assert max(e.live_kv_cache_bytes for e in result.events) <= (pool_blocks - 1) * block_bytes
    # 两个请求各 10 个 Token（5 + 6 - 1）-> 各 3 块；同时在池的峰值为 6 块
    assert max(e.live_kv_cache_bytes for e in result.events) == 6 * block_bytes


def test_small_pool_gates_admission_and_still_matches_serial(model):
    """块池小于「全并发」所需：入场受限于空闲块，请求排队等待块被归还，输出仍与串行一致。"""
    prompts = make_prompts([6, 6, 6])
    budgets = [3, 3, 3]
    specs = make_request_specs(prompts, budgets)
    # 块大小 4：每个请求峰值 8 Token = 2 块；池 = 1 个 null block + 4 块，只够 2 个请求并发（并发上限 3）
    result = run_paged(model, specs, 3, block_size=4, num_blocks=5)

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, budgets)
    prefill_events = [e for e in result.events if e.phase is SchedulePhase.PREFILL]
    assert [e.admitted_request_ids for e in prefill_events] == [(0, 1), (2,)]
    assert result.request_metrics[2].admitted_ms >= result.request_metrics[0].completion_ms


def test_pool_that_cannot_hold_one_request_is_rejected(model):
    specs = make_request_specs(make_prompts([10]), [5])  # 峰值 14 Token -> 4 块（块大小 4）
    with pytest.raises(ValueError, match="无法容纳单个请求"):
        run_paged(model, specs, 1, block_size=4, num_blocks=4)


def test_concurrent_growth_exhaustion_fails_loudly(model):
    """Prompt 刚好能入场、但并发增长耗尽块池：显式报错（未实现抢占），而不是静默产出错误结果。"""
    specs = make_request_specs(make_prompts([4, 4]), [6, 6])  # 块大小 4：Prompt 各 1 块，增长至 9 Token 需 3 块
    with pytest.raises(RuntimeError, match="物理块已耗尽"):
        run_paged(model, specs, 2, block_size=4, num_blocks=5)


def test_paged_scheduling_only_supports_continuous_batching(model):
    specs = make_request_specs(make_prompts([4]), [2])
    for method in (BatchingMethod.FIXED, BatchingMethod.NONE):
        with pytest.raises(ValueError, match="仅支持"):
            run_scheduler(method, model, specs, 1, -1, CPU, paged_cache_config=PagedCacheConfig())


def test_generate_dispatches_paged_flag_and_result_workload_reports_it(model):
    prompts = make_prompts([4, 7, 5])
    opts = GenerationOptions(
        eos_token_id=-1, device=CPU, stop_on_eos=False, paged_cache_config=PagedCacheConfig(block_size=4)
    )

    result = generate(PAGED_MODE, model, prompts, [5, 3, 4], options=opts)

    assert result.is_paged_kv_cache
    assert result.workload.mode == PAGED_MODE
    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, [5, 3, 4])


def test_generate_validates_paged_flag_dependencies(model):
    opts = GenerationOptions(eos_token_id=-1, device=CPU)
    with pytest.raises(ValueError, match="continuous-batching"):
        generate(Mode({FeatureFlag.KV_CACHE, FeatureFlag.PAGED_KV_CACHE}), model, [[1, 2, 3]], 2, options=opts)
    with pytest.raises(ValueError, match="依赖 KV Cache"):
        generate(
            Mode({FeatureFlag.PAGED_KV_CACHE}, batching_method=BatchingMethod.CONTINUOUS),
            model,
            [[1, 2, 3]],
            2,
            options=opts,
        )
