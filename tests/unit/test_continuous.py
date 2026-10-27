"""固定批次 baseline 与 Continuous Batching 的核心正确性测试（CPU，tiny 模型，贪心解码）。

核心判据：无论调度策略如何，每个请求的输出 Token 必须与该请求单独串行 generate 的结果逐位一致。
"""

import pytest
import torch

from mini_infra.core import (
    RequestSpec,
    SchedulePhase,
    make_request_specs,
)
from mini_infra.engine import (
    BatchingMethod,
    FeatureFlag,
    Mode,
    generate,
    run_scheduler,
)

CPU = torch.device("cpu")
KV_MODE = Mode({FeatureFlag.KV_CACHE})
RUNNERS = {
    BatchingMethod.FIXED: lambda *args, **kwargs: run_scheduler(BatchingMethod.FIXED, *args, **kwargs),
    BatchingMethod.CONTINUOUS: lambda *args, **kwargs: run_scheduler(BatchingMethod.CONTINUOUS, *args, **kwargs),
}


@pytest.fixture
def model(diverse_model):
    return diverse_model


def test_fixture_outputs_are_diverse(model):
    """守卫：若输出退化为重复 Token，下面所有 parity 测试都会失去检出能力。"""
    tokens = generate(KV_MODE, model, [make_prompts([6])[0]], 10, eos_token_id=-1, device=CPU, stop_on_eos=False)
    assert len(set(tokens.sequences[0].token_ids)) >= 5


def make_prompts(lengths: list[int]) -> list[list[int]]:
    return [[(i * 7 + j * 3) % 90 + 1 for j in range(length)] for i, length in enumerate(lengths)]


def serial_outputs(model, prompts, budgets, eos_token_id=-1, stop_on_eos=False) -> list[list[int]]:
    return [
        generate(KV_MODE, model, [prompt], budget, eos_token_id=eos_token_id, device=CPU, stop_on_eos=stop_on_eos)
        .sequences[0]
        .token_ids
        for prompt, budget in zip(prompts, budgets)
    ]


CASES = {
    # 混合预算，全部同时到达，并发上限小于请求数：槽位会被反复回收复用
    "mixed_budgets": ([4, 9, 6, 3, 7], [2, 6, 3, 5, 1], [0.0] * 5, 2),
    # 错峰到达，含长时间空档（时钟跳转）
    "staggered": ([5, 5, 5, 5], [4, 6, 4, 6], [0.0, 1.0, 2.0, 5000.0], 3),
    # 后到达的 Prompt 更长：Decode 中途入场会触发 Cache 左侧扩容
    "longer_prompt_joins_late": ([3, 12, 6], [8, 4, 3], [0.0, 0.0001, 0.0002], 2),
    # 预算为 1：请求在 Prefill 当步即完成，不进入 Decode
    "budget_one": ([4, 6, 5], [1, 1, 3], [0.0, 0.0, 0.0], 3),
    # 并发上限不小于请求数
    "capacity_not_binding": ([4, 7], [5, 3], [0.0, 0.0], 8),
    "single_request": ([6], [5], [0.0], 1),
}


@pytest.mark.parametrize("method", list(RUNNERS))
@pytest.mark.parametrize("case", list(CASES))
def test_outputs_match_serial_generate(model, method, case):
    lengths, budgets, arrivals, capacity = CASES[case]
    prompts = make_prompts(lengths)
    specs = make_request_specs(prompts, budgets, arrivals)

    result = RUNNERS[method](model, specs, capacity, -1, CPU, stop_on_eos=False)

    assert result.batching_method == method
    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, budgets)
    assert [s.output_len for s in result.sequences] == budgets
    assert [s.prompt_len for s in result.sequences] == lengths
    assert all(s.finish_reason == "max_tokens" for s in result.sequences)


@pytest.mark.parametrize("method", list(RUNNERS))
def test_eos_stops_request_and_matches_serial(model, method):
    prompts = make_prompts([4, 6, 5])
    budgets = [8, 8, 8]
    # 取第一个请求的第 3 个输出 Token 作为 EOS：该请求应在第 3 步以 "stop" 结束
    eos = serial_outputs(model, prompts, budgets)[0][2]
    specs = make_request_specs(prompts, budgets)

    result = RUNNERS[method](model, specs, 2, eos, CPU, stop_on_eos=True)

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, budgets, eos, stop_on_eos=True)
    assert result.sequences[0].finish_reason == "stop"
    assert result.sequences[0].output_len <= 3
    assert result.sequences[0].token_ids[-1] == eos


@pytest.mark.parametrize("method", list(RUNNERS))
@pytest.mark.parametrize("case", ["mixed_budgets", "staggered", "longer_prompt_joins_late"])
def test_timeline_and_metric_invariants(model, method, case):
    lengths, budgets, arrivals, capacity = CASES[case]
    specs = make_request_specs(make_prompts(lengths), budgets, arrivals)

    result = RUNNERS[method](model, specs, capacity, -1, CPU, stop_on_eos=False)
    metrics = result.metrics

    for request, budget in zip(result.requests, budgets):
        assert request.arrival_ms <= request.admitted_ms <= request.first_token_ms <= request.completion_ms
        assert request.queue_ms >= 0 and request.ttft_ms > 0 and request.end_to_end_ms >= request.ttft_ms
        assert request.output_tokens == budget == len(request.token_times_ms)
        assert request.completion_ms == request.token_times_ms[-1]
        assert list(request.token_times_ms) == sorted(request.token_times_ms)

    assert metrics.request_count == len(specs) and metrics.max_running_requests == capacity
    assert metrics.makespan_ms == max(r.completion_ms for r in result.requests) - min(arrivals)
    assert metrics.busy_ms == pytest.approx(sum(e.total_ms for e in result.events))
    assert metrics.busy_ms == pytest.approx(metrics.model_ms + metrics.cache_management_ms)
    assert 0 < metrics.execution_slot_utilization <= 1
    assert 0 < metrics.running_capacity_utilization <= 1
    assert 0 <= metrics.cache_management_fraction < 1
    assert metrics.ttft_ms_p50 <= metrics.ttft_ms_p95
    assert sum(e.generated_tokens for e in result.events) == sum(budgets)
    assert sorted(i for e in result.events for i in e.admitted) == list(range(len(specs)))
    assert sorted(i for e in result.events for i in e.completed) == list(range(len(specs)))
    assert all(e.executed_batch_size <= capacity for e in result.events)


def test_fixed_batching_does_not_admit_new_requests_until_the_wave_finishes(model):
    specs = make_request_specs(make_prompts([4, 4, 4]), [2, 6, 3], [0.0, 0.0, 0.0])

    result = run_scheduler(BatchingMethod.FIXED, model, specs, 2, -1, CPU, stop_on_eos=False)

    prefill_events = [e for e in result.events if e.phase is SchedulePhase.PREFILL]
    assert [e.admitted for e in prefill_events] == [(0, 1), (2,)]
    # 第二波只在第一波（预算最大的请求 1，共 6 步）全部结束后才开始
    assert result.requests[2].admitted_ms >= result.requests[1].completion_ms
    # 已结束的短请求槽位空转：第一波 6 步均以 2 行执行
    assert [e.executed_batch_size for e in result.events[:6]] == [2] * 6
    assert [e.active_requests for e in result.events[:6]] == [2, 2, 1, 1, 1, 1]
    assert result.metrics.cache_management_ms == 0.0


def test_continuous_batching_refills_freed_slots_while_others_keep_running(model):
    specs = make_request_specs(make_prompts([4, 4, 4]), [2, 8, 3], [0.0, 0.0, 0.0])

    result = run_scheduler(BatchingMethod.CONTINUOUS, model, specs, 2, -1, CPU, stop_on_eos=False)

    prefill_events = [e for e in result.events if e.phase is SchedulePhase.PREFILL]
    assert [e.admitted for e in prefill_events] == [(0, 1), (2,)]
    # 请求 2 在请求 0 结束后立即补位，早于请求 1 完成
    assert result.requests[2].admitted_ms >= result.requests[0].completion_ms
    assert result.requests[2].admitted_ms < result.requests[1].completion_ms
    # Decode 总是以 capacity 行执行
    assert all(e.executed_batch_size == 2 for e in result.events if e.phase is SchedulePhase.DECODE)
    assert result.metrics.cache_management_ms > 0
    assert result.metrics.peak_live_cache_mb > 0


def test_idle_gap_jumps_logical_clock_to_next_arrival(model):
    specs = make_request_specs(make_prompts([4, 4]), [2, 2], [10.0, 90000.0])

    result = run_scheduler(BatchingMethod.CONTINUOUS, model, specs, 2, -1, CPU, stop_on_eos=False)

    assert result.requests[0].admitted_ms == 10.0  # 初始时钟取最早到达时间
    assert result.requests[1].admitted_ms == 90000.0  # 空档期时钟直接跳转，没有真实 sleep
    assert result.requests[1].queue_ms == 0.0


@pytest.mark.parametrize("runner", list(RUNNERS.values()))
def test_runners_reject_invalid_arguments(model, runner):
    specs = make_request_specs(make_prompts([4]), [2])
    with pytest.raises(ValueError, match="max_running_requests"):
        runner(model, specs, 0, -1, CPU)
    with pytest.raises(ValueError, match="不能为空"):
        runner(model, [], 2, -1, CPU)
    duplicate = [RequestSpec(0, (1, 2), 2), RequestSpec(0, (3, 4), 2)]
    with pytest.raises(ValueError, match="唯一"):
        runner(model, duplicate, 2, -1, CPU)


def test_slots_are_reused_after_all_requests_finish_and_cache_is_released(model):
    # 第二个请求在第一个完全结束（Cache 被整块回收）之后才到达
    specs = make_request_specs(make_prompts([4, 6]), [2, 3], [0.0, 90000.0])

    result = run_scheduler(BatchingMethod.CONTINUOUS, model, specs, 1, -1, CPU, stop_on_eos=False)

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, make_prompts([4, 6]), [2, 3])


class TestGenerateDispatch:
    """generate(mode=CONTINUOUS) 不再静默回退到固定批次路径。"""

    def test_continuous_mode_matches_serial_and_fixed(self, model):
        prompts = make_prompts([3, 6, 4])
        budgets = [5, 2, 4]
        continuous = Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.CONTINUOUS)
        fixed = Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.FIXED)

        result = generate(continuous, model, prompts, budgets, eos_token_id=-1, device=CPU, stop_on_eos=False)
        fixed_result = generate(fixed, model, prompts, budgets, eos_token_id=-1, device=CPU, stop_on_eos=False)

        expected = serial_outputs(model, prompts, budgets)
        assert [s.token_ids for s in result.sequences] == expected
        assert [s.token_ids for s in fixed_result.sequences] == expected
        assert result.batch_size == 3
        assert result.batch_info is None
        assert result.trace.active_tokens_per_step[0] == 3  # 首个 Prefill 事件产出 3 个 Token
        assert sum(result.trace.active_tokens_per_step) == sum(budgets)
        assert result.metrics.throughput_tok_s > 0
        assert result.padded_output_tokens(0).tolist() == expected[0]

    def test_continuous_mode_requires_kv_cache(self, model):
        mode = Mode(batching_method=BatchingMethod.CONTINUOUS)
        with pytest.raises(ValueError, match="KV Cache"):
            generate(mode, model, make_prompts([3, 4]), 2, device=CPU)

    @pytest.mark.parametrize("kwargs", [{"capture_logits": True}, {"profile": True}])
    def test_continuous_mode_rejects_unsupported_options(self, model, kwargs):
        mode = Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.CONTINUOUS)
        with pytest.raises(ValueError, match="暂不支持"):
            generate(mode, model, make_prompts([3, 4]), 2, device=CPU, **kwargs)

    def test_continuous_mode_respects_max_running_requests(self, model):
        prompts = make_prompts([3, 5, 4, 6])
        budgets = [3, 2, 4, 1]
        continuous = Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.CONTINUOUS)
        # 4 个离线请求，并发上限限制为 2，触发槽位排队复用
        result = generate(
            continuous,
            model,
            prompts,
            budgets,
            eos_token_id=-1,
            device=CPU,
            stop_on_eos=False,
            max_running_requests=2,
        )
        expected = serial_outputs(model, prompts, budgets)
        assert [s.token_ids for s in result.sequences] == expected
        assert result.batch_size == 4

    def test_continuous_mode_rejects_invalid_max_running_requests(self, model):
        continuous = Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.CONTINUOUS)
        with pytest.raises(ValueError, match="必须大于 0"):
            generate(continuous, model, make_prompts([3]), 2, device=CPU, max_running_requests=0)


def test_run_scheduler_dispatch(model):
    specs = make_request_specs(make_prompts([3, 4]), max_new_tokens=2)

    # 1. 验证枚举分发 CONTINUOUS
    res_cont = run_scheduler(BatchingMethod.CONTINUOUS, model, specs, 2, eos_token_id=-1, device=CPU, stop_on_eos=False)
    assert res_cont.batching_method == BatchingMethod.CONTINUOUS
    assert len(res_cont.requests) == 2

    # 2. 验证字符串分发 continuous
    res_cont_str = run_scheduler("continuous", model, specs, 2, eos_token_id=-1, device=CPU, stop_on_eos=False)
    assert res_cont_str.batching_method == BatchingMethod.CONTINUOUS

    # 3. 验证枚举分发 FIXED
    res_fixed = run_scheduler(BatchingMethod.FIXED, model, specs, 2, eos_token_id=-1, device=CPU, stop_on_eos=False)
    assert res_fixed.batching_method == BatchingMethod.FIXED
    assert len(res_fixed.requests) == 2

    # 4. 验证字符串分发 fixed
    res_fixed_str = run_scheduler("fixed", model, specs, 2, eos_token_id=-1, device=CPU, stop_on_eos=False)
    assert res_fixed_str.batching_method == BatchingMethod.FIXED

    # 5. 验证非法策略抛出异常
    with pytest.raises(ValueError, match="不支持的调度批处理策略"):
        run_scheduler(BatchingMethod.NONE, model, specs, 2, eos_token_id=-1, device=CPU)
    with pytest.raises(ValueError, match="未知批处理策略"):
        run_scheduler("unknown-strategy", model, specs, 2, eos_token_id=-1, device=CPU)
