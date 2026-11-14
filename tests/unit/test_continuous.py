"""固定批次 baseline 与 Continuous Batching 的核心正确性测试（CPU，tiny 模型，贪心解码）。

核心判据：无论调度策略如何，每个请求的输出 Token 必须与该请求单独串行 generate 的结果逐位一致。
"""

import pytest

from mini_infra.core import (
    DenseCacheConfig,
    Request,
    SchedulePhase,
    SchedulerConfig,
    TokenBudgetConfig,
    make_requests,
)
from mini_infra.engine import (
    BatchingMethod,
    EngineConfig,
    GenerationResult,
    ScheduleResult,
    run_scheduler,
)
from mini_infra.entrypoints import GenerationOptions, generate

KV_CONFIG = EngineConfig(DenseCacheConfig())
RUNNERS = {
    BatchingMethod.FIXED: lambda model, requests, capacity, eos=-1, **kwargs: run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED, max_running_requests=capacity),
        model,
        requests,
        eos,
        **kwargs,
    ),
    BatchingMethod.CONTINUOUS: lambda model, requests, capacity, eos=-1, **kwargs: run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=capacity),
        model,
        requests,
        eos,
        **kwargs,
    ),
}


@pytest.fixture
def model(diverse_model):
    return diverse_model


def test_fixture_outputs_are_diverse(model):
    """守卫：若输出退化为重复 Token，下面所有 parity 测试都会失去检出能力。"""
    opts = GenerationOptions(eos_token_id=-1, stop_on_eos=False)
    tokens = generate(KV_CONFIG, model, [make_prompts([6])[0]], 10, options=opts)
    assert len(set(tokens.sequences[0].token_ids)) >= 5


def make_prompts(lengths: list[int]) -> list[list[int]]:
    return [[(i * 7 + j * 3) % 90 + 1 for j in range(length)] for i, length in enumerate(lengths)]


def serial_outputs(model, prompts, budgets, eos_token_id=-1, stop_on_eos=False) -> list[list[int]]:
    opts = GenerationOptions(eos_token_id=eos_token_id, stop_on_eos=stop_on_eos)
    return [
        generate(KV_CONFIG, model, [prompt], budget, options=opts).sequences[0].token_ids
        for prompt, budget in zip(prompts, budgets)
    ]


CASES = {
    # 混合预算，全部同时到达，并发上限小于请求数：batch slot 会被反复回收复用
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
    requests = make_requests(prompts, budgets, arrivals)

    result = RUNNERS[method](model, requests, capacity, -1, stop_on_eos=False)

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
    requests = make_requests(prompts, budgets)

    result = RUNNERS[method](model, requests, 2, eos, stop_on_eos=True)

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, budgets, eos, stop_on_eos=True)
    assert result.sequences[0].finish_reason == "stop"
    assert result.sequences[0].output_len <= 3
    assert result.sequences[0].token_ids[-1] == eos


@pytest.mark.parametrize("method", list(RUNNERS))
@pytest.mark.parametrize("case", ["mixed_budgets", "staggered", "longer_prompt_joins_late"])
def test_timeline_and_metric_invariants(model, method, case):
    lengths, budgets, arrivals, capacity = CASES[case]
    requests = make_requests(make_prompts(lengths), budgets, arrivals)

    result = RUNNERS[method](model, requests, capacity, -1, stop_on_eos=False)
    metrics = result.schedule_metrics

    for request, budget in zip(result.request_metrics, budgets):
        assert request.arrival_ms <= request.admitted_ms <= request.first_token_ms <= request.completion_ms
        assert request.queue_ms >= 0 and request.ttft_ms > 0 and request.end_to_end_ms >= request.ttft_ms
        assert request.num_output_tokens == budget == len(request.token_times_ms)
        assert request.completion_ms == request.token_times_ms[-1]
        assert list(request.token_times_ms) == sorted(request.token_times_ms)

    assert metrics.request_count == len(requests) and metrics.max_running_requests == capacity
    assert metrics.makespan_ms == max(r.completion_ms for r in result.request_metrics) - min(arrivals)
    assert result.metrics.e2e_wall_ms >= result.metrics.e2e_model_ms
    assert result.metrics.throughput_req_s > 0
    assert result.metrics.throughput_tok_s > 0
    assert (result.metrics.effective_prompt_tok_s or 0) > 0
    assert (result.metrics.padded_prompt_tok_s or 0) > 0
    assert result.metrics.ttft_ms is None
    assert result.metrics.mean_tpot_ms is None
    assert result.memory.prefill_mb is None
    assert result.memory.kv_cache.peak_live_kv_cache_mb > 0
    assert result.batch_info is not None
    assert result.batch_info.running_capacity_utilization is not None
    assert max(s.completion_step for s in result.sequences if s.completion_step is not None) == len(result.events)
    assert result.padded_output_tokens(0).tolist() == result.sequences[0].token_ids
    assert metrics.busy_ms == pytest.approx(sum(e.step_ms for e in result.events))
    assert metrics.busy_ms == pytest.approx(result.metrics.e2e_model_ms + metrics.cache_management_ms)
    assert 0 < result.batch_info.slot_utilization <= 1
    assert 0 < result.batch_info.running_capacity_utilization <= 1
    assert 0 <= metrics.cache_management_fraction < 1
    assert metrics.ttft_ms_p50 <= metrics.ttft_ms_p95
    assert sum(e.num_generated_tokens for e in result.events) == sum(budgets)
    assert sorted(i for e in result.events for i in e.admitted_request_ids) == list(range(len(requests)))
    assert sorted(i for e in result.events for i in e.finished_request_ids) == list(range(len(requests)))
    assert all(e.physical_batch_size <= capacity for e in result.events)


def test_fixed_batching_does_not_admit_new_requests_until_the_wave_finishes(model):
    requests = make_requests(make_prompts([4, 4, 4]), [2, 6, 3], [0.0, 0.0, 0.0])

    result = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED, max_running_requests=2),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )

    prefill_events = [e for e in result.events if e.phase is SchedulePhase.PREFILL]
    assert [e.admitted_request_ids for e in prefill_events] == [(0, 1), (2,)]
    # 第二波只在第一波（预算最大的请求 1，共 6 步）全部结束后才开始
    assert result.request_metrics[2].admitted_ms >= result.request_metrics[1].completion_ms
    # 已结束的短请求 batch slot 空转：第一波 6 步均以 2 行执行
    assert [e.physical_batch_size for e in result.events[:6]] == [2] * 6
    assert [e.num_active_requests for e in result.events[:6]] == [2, 2, 1, 1, 1, 1]
    assert result.schedule_metrics.cache_management_ms == 0.0


def test_continuous_batching_refills_freed_slots_while_others_keep_running(model):
    requests = make_requests(make_prompts([4, 4, 4]), [2, 8, 3], [0.0, 0.0, 0.0])

    result = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )

    prefill_events = [e for e in result.events if e.phase is SchedulePhase.PREFILL]
    assert [e.admitted_request_ids for e in prefill_events] == [(0, 1), (2,)]
    # 请求 2 在请求 0 结束后立即补位，早于请求 1 完成
    assert result.request_metrics[2].admitted_ms >= result.request_metrics[0].completion_ms
    assert result.request_metrics[2].admitted_ms < result.request_metrics[1].completion_ms
    # Decode 总是以 capacity 行执行
    assert all(e.physical_batch_size == 2 for e in result.events if e.phase is SchedulePhase.DECODE)
    assert result.schedule_metrics.cache_management_ms > 0
    assert result.memory.kv_cache.peak_live_kv_cache_mb > 0


def test_idle_gap_jumps_logical_clock_to_next_arrival(model):
    requests = make_requests(make_prompts([4, 4]), [2, 2], [10.0, 90000.0])

    result = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )

    assert result.request_metrics[0].admitted_ms == 10.0  # 初始时钟取最早到达时间
    assert result.request_metrics[1].admitted_ms == 90000.0  # 空档期时钟直接跳转，没有真实 sleep
    assert result.request_metrics[1].queue_ms == 0.0
    assert result.schedule_metrics.makespan_ms > 89990.0  # 逻辑跨度包含空档期
    assert result.metrics.e2e_wall_ms < 5000.0  # 真实墙钟耗时不包含虚拟跳钟空档
    assert result.metrics.throughput_tok_s == pytest.approx(4 * 1000 / result.schedule_metrics.busy_ms)


@pytest.mark.parametrize("runner", list(RUNNERS.values()))
def test_runners_reject_invalid_arguments(model, runner):
    requests = make_requests(make_prompts([4]), [2])
    with pytest.raises(ValueError, match="max_running_requests"):
        runner(model, requests, 0, -1)
    with pytest.raises(ValueError, match="不能为空"):
        runner(model, [], 2, -1)
    duplicate = [Request(0, (1, 2), 2), Request(0, (3, 4), 2)]
    with pytest.raises(ValueError, match="唯一"):
        runner(model, duplicate, 2, -1)


def test_slots_are_reused_after_all_requests_finish_and_cache_is_released(model):
    # 第二个请求在第一个完全结束（Cache 被整块回收）之后才到达
    requests = make_requests(make_prompts([4, 6]), [2, 3], [0.0, 90000.0])

    result = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=1),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, make_prompts([4, 6]), [2, 3])


def test_prefill_first_event_trajectories(model):
    """锁定 prefill-first 基线在槽位复用、错峰到达与 EOS 提前退场下的完整事件轨迹。"""

    def event_trace(events):
        return [(e.phase, e.admitted_request_ids, e.finished_request_ids) for e in events]

    # 1. 混合预算与 batch slot 复用（含 budget=1 在 Prefill 当步直接完成）
    lengths, budgets, arrivals, capacity = CASES["mixed_budgets"]
    res_mixed = RUNNERS[BatchingMethod.CONTINUOUS](
        model, make_requests(make_prompts(lengths), budgets, arrivals), capacity, -1, stop_on_eos=False
    )
    assert event_trace(res_mixed.events) == [
        (SchedulePhase.PREFILL, (0, 1), ()),
        (SchedulePhase.DECODE, (), (0,)),
        (SchedulePhase.PREFILL, (2,), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), (2,)),
        (SchedulePhase.PREFILL, (3,), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), (1,)),
        (SchedulePhase.PREFILL, (4,), (4,)),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), (3,)),
    ]

    # 2. 错峰到达（t=0 先入场 0，步 0 后 1、2 均到达，按空余 batch slot 先入场 1，1 结束后再入场 2）
    lengths, budgets, arrivals, capacity = CASES["longer_prompt_joins_late"]
    res_late = RUNNERS[BatchingMethod.CONTINUOUS](
        model, make_requests(make_prompts(lengths), budgets, arrivals), capacity, -1, stop_on_eos=False
    )
    assert event_trace(res_late.events) == [
        (SchedulePhase.PREFILL, (0,), ()),
        (SchedulePhase.PREFILL, (1,), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), (1,)),
        (SchedulePhase.PREFILL, (2,), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), (2,)),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), (0,)),
    ]

    # 3. EOS 提前退场释放 batch slot
    prompts_eos = make_prompts([4, 6, 5])
    budgets_eos = [8, 8, 8]
    eos = serial_outputs(model, prompts_eos, budgets_eos)[0][2]
    res_eos = RUNNERS[BatchingMethod.CONTINUOUS](
        model, make_requests(prompts_eos, budgets_eos), 2, eos, stop_on_eos=True
    )
    assert event_trace(res_eos.events) == [
        (SchedulePhase.PREFILL, (0, 1), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), (0,)),
        (SchedulePhase.PREFILL, (2,), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), (1,)),
        (SchedulePhase.DECODE, (), ()),
        (SchedulePhase.DECODE, (), (2,)),
    ]


def run_token_budget(model, requests, capacity, token_budget, eos=-1, **kwargs):
    return run_scheduler(
        EngineConfig(
            DenseCacheConfig(),
            batching_method=BatchingMethod.CONTINUOUS,
            max_running_requests=capacity,
            scheduler_config=SchedulerConfig(TokenBudgetConfig(token_budget)),
        ),
        model,
        requests,
        eos,
        **kwargs,
    )


@pytest.mark.parametrize("budget_floor", [1, 12, 1000])  # 1 取到下限 token_budget = capacity，必然触发 oversize prefill
@pytest.mark.parametrize("case", list(CASES))
def test_token_budget_outputs_match_serial_and_keep_invariants(
    model, case, budget_floor, assert_token_budget_invariants
):
    lengths, budgets, arrivals, capacity = CASES[case]
    prompts = make_prompts(lengths)
    token_budget = max(capacity, budget_floor)

    result = run_token_budget(
        model, make_requests(prompts, budgets, arrivals), capacity, token_budget, stop_on_eos=False
    )

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, budgets)
    assert sum(e.num_generated_tokens for e in result.events) == sum(budgets)
    assert_token_budget_invariants(result, token_budget)


def test_token_budget_eos_stops_request_and_matches_serial(model, assert_token_budget_invariants):
    prompts = make_prompts([4, 6, 5])
    budgets = [8, 8, 8]
    eos = serial_outputs(model, prompts, budgets)[0][2]

    result = run_token_budget(model, make_requests(prompts, budgets), 2, 6, eos, stop_on_eos=True)

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, budgets, eos, stop_on_eos=True)
    assert result.sequences[0].finish_reason == "stop"
    assert_token_budget_invariants(result, 6)


def test_token_budget_event_trajectories(model):
    """锁定 token-budget 在 Prefill burst 拆分、decode protection 与 batch slot 复用下的完整事件轨迹。"""

    def event_trace(events):
        return [(e.phase, e.admitted_request_ids, e.finished_request_ids, e.num_scheduled_tokens) for e in events]

    # 1. 同时到达的 6 个 3-token 请求：预算 6 每次只放 2 个，且每次 Prefill 后强制一步 Decode
    burst = run_token_budget(model, make_requests(make_prompts([3] * 6), 4), 4, 6, stop_on_eos=False)
    assert event_trace(burst.events) == [
        (SchedulePhase.PREFILL, (0, 1), (), 6),
        (SchedulePhase.DECODE, (), (), 2),
        (SchedulePhase.PREFILL, (2, 3), (), 6),
        (SchedulePhase.DECODE, (), (), 4),
        (SchedulePhase.DECODE, (), (0, 1), 4),
        (SchedulePhase.PREFILL, (4, 5), (), 6),
        (SchedulePhase.DECODE, (), (2, 3), 4),
        (SchedulePhase.DECODE, (), (), 2),
        (SchedulePhase.DECODE, (), (4, 5), 2),
    ]

    # 2. 混合长度：[4, 9] 计 2 × 9 = 18 > 10，只入场队头；请求 4 在 Prefill 当步结束（RunningQueue 仍非空）
    lengths, budgets, arrivals, capacity = CASES["mixed_budgets"]
    mixed = run_token_budget(
        model, make_requests(make_prompts(lengths), budgets, arrivals), capacity, 10, stop_on_eos=False
    )
    assert event_trace(mixed.events) == [
        (SchedulePhase.PREFILL, (0,), (), 4),
        (SchedulePhase.DECODE, (), (0,), 1),
        (SchedulePhase.PREFILL, (1,), (), 9),
        (SchedulePhase.DECODE, (), (), 1),
        (SchedulePhase.PREFILL, (2,), (), 6),
        (SchedulePhase.DECODE, (), (), 2),
        (SchedulePhase.DECODE, (), (2,), 2),
        (SchedulePhase.PREFILL, (3,), (), 3),
        (SchedulePhase.DECODE, (), (), 2),
        (SchedulePhase.DECODE, (), (1,), 2),
        (SchedulePhase.PREFILL, (4,), (4,), 7),
        (SchedulePhase.DECODE, (), (), 1),
        (SchedulePhase.DECODE, (), (3,), 1),
    ]


def test_token_budget_padding_efficiency_uses_scheduled_tokens(model):
    """summarize_schedule 的 padded prompt token 数来自 PREFILL 事件的 num_scheduled_tokens。"""
    lengths, budgets, arrivals, capacity = CASES["mixed_budgets"]
    requests = make_requests(make_prompts(lengths), budgets, arrivals)
    prefill_first = RUNNERS[BatchingMethod.CONTINUOUS](model, requests, capacity, -1, stop_on_eos=False)
    budgeted = run_token_budget(model, requests, capacity, 10, stop_on_eos=False)

    for result in (prefill_first, budgeted):
        num_padded = sum(e.num_scheduled_tokens for e in result.events if e.phase is SchedulePhase.PREFILL)
        assert result.batch_info.padding_efficiency == pytest.approx(sum(lengths) / num_padded)
    # 预算把 [4, 9] 拆成两次单请求 Prefill，消除了 4-token 请求的左 Padding
    assert budgeted.batch_info.padding_efficiency > prefill_first.batch_info.padding_efficiency


def test_run_scheduler_rejects_invalid_token_budget(model):
    requests = make_requests(make_prompts([3, 4, 5]), 2)
    with pytest.raises(ValueError, match="仅支持 continuous-batching"):
        run_scheduler(
            EngineConfig(
                DenseCacheConfig(),
                batching_method=BatchingMethod.FIXED,
                scheduler_config=SchedulerConfig(TokenBudgetConfig(8)),
            ),
            model,
            requests,
        )
    # max_running_requests=None 时按请求总数 3 推导，预算 2 小于实际并发上限
    with pytest.raises(ValueError, match="不能小于 max_running_requests=3"):
        run_token_budget(model, requests, None, 2)


class TestGenerateDispatch:
    """generate(config=CONTINUOUS) 不再静默回退到固定批次路径。"""

    def test_continuous_mode_matches_serial_and_fixed(self, model):
        prompts = make_prompts([3, 6, 4])
        budgets = [5, 2, 4]
        continuous = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS)
        fixed = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED)

        opts = GenerationOptions(eos_token_id=-1, stop_on_eos=False)
        result = generate(continuous, model, prompts, budgets, options=opts)
        fixed_result = generate(fixed, model, prompts, budgets, options=opts)

        expected = serial_outputs(model, prompts, budgets)
        assert [s.token_ids for s in result.sequences] == expected
        assert [s.token_ids for s in fixed_result.sequences] == expected
        assert result.batch_size == 3
        assert result.batch_info is not None
        assert result.batch_info.running_capacity_utilization is not None
        assert fixed_result.batch_info is not None
        assert fixed_result.batch_info.running_capacity_utilization is None
        assert isinstance(result, GenerationResult)
        assert isinstance(result, ScheduleResult)
        assert isinstance(fixed_result, GenerationResult)
        assert not isinstance(fixed_result, ScheduleResult)
        assert result.schedule_metrics.request_count == 3
        assert len(result.request_metrics) == 3
        assert len(result.events) > 0
        assert result.trace.num_active_tokens_per_step[0] == 3  # 首个 Prefill 事件产出 3 个 Token
        assert sum(result.trace.num_active_tokens_per_step) == sum(budgets)
        assert result.metrics.throughput_tok_s > 0
        assert result.padded_output_tokens(0).tolist() == expected[0]

    def test_continuous_mode_requires_kv_cache(self, model):
        config = EngineConfig(batching_method=BatchingMethod.CONTINUOUS)
        with pytest.raises(ValueError, match="KV Cache"):
            generate(config, model, make_prompts([3, 4]), 2)

    @pytest.mark.parametrize("kwargs", [{"capture_logits": True}, {"profile": True}])
    def test_continuous_mode_rejects_unsupported_options(self, model, kwargs):
        config = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS)
        with pytest.raises(ValueError, match="暂不支持"):
            generate(config, model, make_prompts([3, 4]), 2, options=GenerationOptions(**kwargs))

    def test_continuous_mode_respects_max_running_requests(self, model):
        prompts = make_prompts([3, 5, 4, 6])
        budgets = [3, 2, 4, 1]
        continuous = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2)
        # 4 个离线请求，并发上限限制为 2，触发 batch slot 排队复用
        result = generate(
            continuous,
            model,
            prompts,
            budgets,
            options=GenerationOptions(
                eos_token_id=-1,
                stop_on_eos=False,
            ),
        )
        expected = serial_outputs(model, prompts, budgets)
        assert [s.token_ids for s in result.sequences] == expected
        assert result.batch_size == 4

    def test_continuous_mode_rejects_invalid_max_running_requests(self, model):
        with pytest.raises(ValueError, match="必须大于 0"):
            EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=0)


def test_run_scheduler_dispatch(model):
    requests = make_requests(make_prompts([3, 4]), max_new_tokens=2)

    # 1. 验证枚举分发 CONTINUOUS
    res_cont = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2),
        model,
        requests,
        eos_token_id=-1,
        stop_on_eos=False,
    )
    assert res_cont.batching_method == BatchingMethod.CONTINUOUS
    assert len(res_cont.request_metrics) == 2

    # 2. 验证枚举分发 FIXED
    res_fixed = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED, max_running_requests=2),
        model,
        requests,
        eos_token_id=-1,
        stop_on_eos=False,
    )
    assert res_fixed.batching_method == BatchingMethod.FIXED
    assert len(res_fixed.request_metrics) == 2

    # 3. 验证枚举分发 NONE（自动约束并发为 1，逐请求串行调度）
    res_none = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.NONE, max_running_requests=2),
        model,
        requests,
        eos_token_id=-1,
        stop_on_eos=False,
    )
    assert res_none.batching_method == BatchingMethod.NONE
    assert res_none.schedule_metrics.max_running_requests == 1
    assert [s.token_ids for s in res_none.sequences] == [s.token_ids for s in res_cont.sequences]
