"""ArrivalQueue（FCFS 到达队列）、调度策略、record_tokens 与请求规格校验的单元测试（纯 Python，无需模型）。"""

from collections.abc import Sequence

import pytest

from mini_infra.core import (
    ArrivalQueue,
    BatchingMethod,
    ChunkedPrefillConfig,
    IterationScheduler,
    KVCacheManager,
    PrefillChunk,
    PrefillFirstConfig,
    Request,
    RequestState,
    RequestStatus,
    RunningQueue,
    SchedulePhase,
    SchedulerConfig,
    SchedulerOutput,
    TokenBudgetConfig,
    WaitingQueue,
    make_requests,
    prepare_schedule,
    validate_requests,
)


def make_states(arrivals: list[float]) -> list[RequestState]:
    return [RequestState(Request(i, (1, 2, 3), 4, arrival)) for i, arrival in enumerate(arrivals)]


def make_states_with_prompt_lens(prompt_lens: list[int]) -> list[RequestState]:
    return [RequestState(Request(i, tuple(range(1, n + 1)), 4)) for i, n in enumerate(prompt_lens)]


def indices(states: Sequence[RequestState]) -> list[int]:
    return [state.request.request_idx for state in states]


def test_next_arrival_and_len():
    queue = ArrivalQueue(make_states([50.0, 20.0]))
    assert len(queue) == 2
    assert queue.next_arrival_ms() == 20.0
    queue.pop_arrived(clock_ms=20.0)
    assert queue.next_arrival_ms() == 50.0
    queue.pop_arrived(clock_ms=50.0)
    with pytest.raises(IndexError):
        queue.next_arrival_ms()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_idx": -1, "token_ids": (1,), "max_new_tokens": 1},
        {"request_idx": 0, "token_ids": (), "max_new_tokens": 1},
        {"request_idx": 0, "token_ids": (1,), "max_new_tokens": 0},
        {"request_idx": 0, "token_ids": (1,), "max_new_tokens": 1, "arrival_ms": -1.0},
    ],
)
def test_request_rejects_invalid_fields(kwargs):
    with pytest.raises(ValueError):
        Request(**kwargs)


def test_make_requests_broadcasts_budget_and_defaults_arrival():
    requests = make_requests([[1, 2], [3]], 5)
    assert [(r.request_idx, r.token_ids, r.max_new_tokens, r.arrival_ms) for r in requests] == [
        (0, (1, 2), 5, 0.0),
        (1, (3,), 5, 0.0),
    ]
    requests = make_requests([[1, 2], [3]], [2, 4], [0.0, 7.5])
    assert [(r.max_new_tokens, r.arrival_ms) for r in requests] == [(2, 0.0), (4, 7.5)]


def test_make_requests_rejects_length_mismatch_and_duplicates():
    with pytest.raises(ValueError, match="数量必须一致"):
        make_requests([[1], [2]], [1, 2, 3])
    with pytest.raises(ValueError, match="数量必须一致"):
        make_requests([[1], [2]], 1, [0.0])
    with pytest.raises(ValueError, match="不能为空"):
        validate_requests([])
    req = Request(0, (1,), 1)
    with pytest.raises(ValueError, match="唯一"):
        validate_requests([req, req])


def test_prepare_schedule_validates_and_returns_initial_clock():
    requests = make_requests([[1], [2], [3]], 2, [30.0, 10.0, 20.0])
    states, queue, clock_ms = prepare_schedule(requests, max_running_requests=2)
    assert len(states) == 3 and len(queue) == 3
    assert clock_ms == 10.0
    with pytest.raises(ValueError, match="max_running_requests"):
        prepare_schedule(requests, max_running_requests=0)
    with pytest.raises(ValueError, match="不能为空"):
        prepare_schedule([], max_running_requests=1)


def test_running_queue_process_step_output_prefers_eos_over_budget_and_tracks_timestamps():
    rq = RunningQueue(capacity=3)
    states = [RequestState(Request(i, (1,), 2)) for i in range(3)]
    rq.admit(states, clock_ms=0.0)

    # 产出第 1 批 token
    finished = rq.process_step_output([5, 5, 7], 1.0, eos_token_id=7, stop_on_eos=True)
    assert finished == [states[2]]
    assert states[2].finish_reason == "stop"
    rq.retire(finished)
    assert states[2].status is RequestStatus.FINISHED
    assert all(s.first_token_ms == 1.0 and s.token_times_ms == [1.0] for s in states)

    # 产出第 2 批 token（仅剩的前两项）
    finished = rq.process_step_output([5, 7], 2.0, eos_token_id=7, stop_on_eos=False)
    assert finished == states[:2]  # 预算（2 个 Token）耗尽；stop_on_eos=False 时 EOS 不触发 stop
    assert [s.finish_reason for s in states[:2]] == ["max_tokens", "max_tokens"]
    assert states[0].first_token_ms == 1.0 and states[0].completion_ms == 2.0
    rq.retire(finished)
    assert all(s.status is RequestStatus.FINISHED for s in states[:2])

    # 单请求 append_token 单元逻辑测试
    single = RequestState(Request(0, (1,), 2))
    assert not single.append_token(9, 10.0, eos_token_id=9, stop_on_eos=False)  # 仅产出第 1 个 token，未达预算
    assert single.append_token(9, 11.0, eos_token_id=9, stop_on_eos=False)  # 产出第 2 个 token，达到预算 2
    assert single.finish_reason == "max_tokens"
    single_eos = RequestState(Request(1, (1,), 10))
    assert single_eos.append_token(9, 10.0, eos_token_id=9, stop_on_eos=True)  # eos 停止
    assert single_eos.finish_reason == "stop"


def test_arrival_queue_pop_arrived_moves_all_ready_requests():
    queue = ArrivalQueue(make_states([10.0, 50.0, 10.0, 20.0]))
    arrived_0 = queue.pop_arrived(clock_ms=5.0)
    assert arrived_0 == []
    arrived_10 = queue.pop_arrived(clock_ms=10.0)
    assert indices(arrived_10) == [0, 2]
    arrived_30 = queue.pop_arrived(clock_ms=30.0)
    assert indices(arrived_30) == [3]
    assert len(queue) == 1
    assert queue.next_arrival_ms() == 50.0


def test_waiting_queue_fcfs_ordering_and_batch_pop():
    wq = WaitingQueue()
    assert len(wq) == 0 and not wq
    states = make_states([0.0, 0.0, 0.0])
    wq.extend(states)
    assert len(wq) == 3 and bool(wq)
    assert all(s.status is RequestStatus.WAITING for s in wq.requests)
    popped = wq.pop_batch(limit=2)
    assert indices(popped) == [0, 1]
    assert len(wq) == 1
    single = wq.pop_batch(limit=5)
    assert indices(single) == [2]
    assert not wq


def test_running_queue_lifecycle_and_batch_slot_management():
    rq = RunningQueue(capacity=3)
    assert rq.capacity == 3
    assert rq.num_free_batch_slots == 3
    assert rq.can_admit()
    with pytest.raises(ValueError, match="必须大于 0"):
        RunningQueue(capacity=0)

    states = make_states([0.0, 0.0, 0.0])
    admitted = rq.admit(states[:2], clock_ms=5.0)
    assert indices(admitted) == [0, 1]
    assert [s.batch_slot_idx for s in admitted] == [0, 1]
    assert all(s.status is RequestStatus.RUNNING for s in admitted)
    assert all(s.admitted_ms == 5.0 for s in admitted)
    assert rq.num_free_batch_slots == 1
    assert rq.free_batch_slots == (2,)

    # 尝试超量入场抛出异常
    with pytest.raises(ValueError, match="可用 batch slot 不足"):
        rq.admit(states, clock_ms=5.0)

    # 释放已结束请求
    rq.retire([states[0]])
    assert states[0].status is RequestStatus.FINISHED
    assert rq.num_free_batch_slots == 2
    assert rq.free_batch_slots == (0, 2)
    assert indices(rq.requests) == [1]


CONTINUOUS = BatchingMethod.CONTINUOUS


def test_scheduler_requires_resolved_max_running_requests():
    with pytest.raises(ValueError, match="max_running_requests"):
        IterationScheduler(SchedulerConfig(max_running_requests=None), CONTINUOUS)


def test_scheduler_prefill_first_decisions():
    scheduler = IterationScheduler(
        SchedulerConfig(max_running_requests=2, batching_policy=PrefillFirstConfig()),
        CONTINUOUS,
    )
    states = make_states([0.0, 0.0, 0.0])
    scheduler.add_requests(states)
    assert scheduler.has_unfinished_requests

    # 1. 有就绪请求且有 free batch slot：按 FCFS 前缀（受 batch slot 数截断）生成 PREFILL 计划并原子入场
    out = scheduler.schedule(clock_ms=0.0)
    assert out == SchedulerOutput(
        phase=SchedulePhase.PREFILL,
        scheduled_requests=tuple(states[:2]),
        num_scheduled_tokens=2 * 3,
        prefill_chunks=(PrefillChunk(states[0], 0, 3), PrefillChunk(states[1], 0, 3)),
    )
    assert len(scheduler.waiting_queue) == 1
    assert len(scheduler.running_queue) == 2
    assert [s.batch_slot_idx for s in states[:2]] == [0, 1]
    scheduler.update_from_output(out, [5, 5], clock_ms=0.5)

    # 2. batch slot 用尽但存在运行中请求：生成 DECODE 计划，每个运行中请求计 1 个 Token
    out_decode = scheduler.schedule(clock_ms=1.0)
    assert out_decode == SchedulerOutput(
        phase=SchedulePhase.DECODE,
        scheduled_requests=scheduler.running_queue.requests,
        num_scheduled_tokens=2,
    )

    # 3. 既无就绪请求也无运行中请求：返回 None
    scheduler.waiting_queue.pop_batch(1)
    scheduler.running_queue.retire(list(scheduler.running_queue.requests))
    assert not scheduler.has_unfinished_requests
    assert scheduler.schedule(clock_ms=2.0) is None


@pytest.mark.parametrize("batching_method", [BatchingMethod.FIXED, BatchingMethod.NONE])
def test_scheduler_wave_barrier_blocks_admission_until_running_queue_drains(batching_method):
    scheduler = IterationScheduler(
        SchedulerConfig(max_running_requests=2, batching_policy=PrefillFirstConfig()),
        batching_method,
    )
    states = make_states([0.0, 0.0, 0.0])
    scheduler.add_requests(states[:1])
    first = scheduler.schedule(clock_ms=0.0)
    assert first is not None and first.phase is SchedulePhase.PREFILL
    scheduler.update_from_output(first, [5], clock_ms=0.5)
    scheduler.add_requests(states[1:])

    # 仍有 free batch slot，但 wave barrier 要求 RunningQueue 排空后才入场下一波
    out = scheduler.schedule(clock_ms=1.0)
    assert out is not None and out.phase is SchedulePhase.DECODE
    scheduler.running_queue.retire(list(scheduler.running_queue.requests))
    out = scheduler.schedule(clock_ms=2.0)
    assert out is not None and out.phase is SchedulePhase.PREFILL
    assert indices(out.scheduled_requests) == [1, 2]


def test_scheduler_paged_admission_limited_by_free_blocks():
    # 1. block_size=4，num_blocks=4：1 个 null block + 3 个可分配块
    # Prompt 6/6/2 各需 2/2/1 块：只够队头 1 个；FCFS 不跳过队头去放 2-token 请求，且 schedule 内完成块分配
    manager_1 = KVCacheManager(block_size=4, num_blocks=4)
    scheduler_1 = IterationScheduler(
        SchedulerConfig(max_running_requests=8, batching_policy=PrefillFirstConfig()),
        CONTINUOUS,
        manager_1,
    )
    scheduler_1.add_requests(make_states_with_prompt_lens([6, 6, 2]))
    out = scheduler_1.schedule(clock_ms=0.0)
    assert out is not None and indices(out.scheduled_requests) == [0]
    assert manager_1.num_used_blocks == 2
    assert len(scheduler_1.waiting_queue) == 2 and len(scheduler_1.running_queue) == 1

    # 2. Prompt 6/2/6 各需 2/1/2 块：前 2 个共 3 块恰好放下
    manager_2 = KVCacheManager(block_size=4, num_blocks=4)
    scheduler_2 = IterationScheduler(
        SchedulerConfig(max_running_requests=8, batching_policy=PrefillFirstConfig()),
        CONTINUOUS,
        manager_2,
    )
    scheduler_2.add_requests(make_states_with_prompt_lens([6, 2, 6]))
    out = scheduler_2.schedule(clock_ms=0.0)
    assert out is not None and indices(out.scheduled_requests) == [0, 1]
    assert manager_2.num_used_blocks == 3

    # 3. 空闲块被占用后不可入场，运行队列为空时空闲返回 None
    manager_3 = KVCacheManager(block_size=4, num_blocks=4)
    assert manager_3.allocate_slots(99, 12)
    scheduler_3 = IterationScheduler(
        SchedulerConfig(max_running_requests=8, batching_policy=PrefillFirstConfig()),
        CONTINUOUS,
        manager_3,
    )
    scheduler_3.add_requests(make_states_with_prompt_lens([1]))
    assert scheduler_3.schedule(clock_ms=0.0) is None


def test_scheduler_update_from_output_advances_cache_length_and_frees_paged_blocks():
    manager = KVCacheManager(block_size=4, num_blocks=6)
    scheduler = IterationScheduler(
        SchedulerConfig(max_running_requests=2, batching_policy=PrefillFirstConfig()),
        CONTINUOUS,
        manager,
    )
    states = [
        RequestState(Request(0, (1, 2, 3, 4), 2)),
        RequestState(Request(1, (5, 6), 1)),
    ]
    scheduler.add_requests(states)

    # Prefill 步：请求 1 预算为 1，在 Prefill 后立即完成并归还物理块与 batch slot
    prefill_out = scheduler.schedule(clock_ms=0.0)
    assert prefill_out is not None and prefill_out.phase is SchedulePhase.PREFILL
    assert states[0].cache_length == 0  # 前向完成前 cache_length 尚未推进
    assert manager.num_used_blocks == 2
    finished_prefill = scheduler.update_from_output(
        prefill_out, [10, 11], clock_ms=1.0, eos_token_id=None, stop_on_eos=False
    )
    assert indices(finished_prefill) == [1]
    assert states[0].cache_length == 4  # update_from_output 将 cache_length 推进至 chunk.end
    assert manager.num_used_blocks == 1  # 请求 1 的物理块已归还
    assert indices(scheduler.running_queue.requests) == [0]

    # Decode 步：请求 0 从 4 Token 跨越块边界分配第 2 块，update_from_output 递增 cache_length 并完成回收
    decode_out = scheduler.schedule(clock_ms=1.0)
    assert decode_out is not None and decode_out.phase is SchedulePhase.DECODE
    assert manager.num_used_blocks == 2
    finished_decode = scheduler.update_from_output(decode_out, [12], clock_ms=2.0, eos_token_id=None, stop_on_eos=False)
    assert indices(finished_decode) == [0]
    assert states[0].cache_length == 5
    assert manager.num_used_blocks == 0
    assert not scheduler.has_unfinished_requests


def test_scheduler_prefill_first_has_no_decode_protection():
    scheduler = IterationScheduler(
        SchedulerConfig(max_running_requests=4, batching_policy=PrefillFirstConfig()),
        CONTINUOUS,
    )
    states = make_states_with_prompt_lens([3, 5, 9])
    scheduler.add_requests(states[:1])

    first = scheduler.schedule(clock_ms=0.0)
    assert first is not None and first.phase is SchedulePhase.PREFILL
    scheduler.update_from_output(first, [5], clock_ms=0.5)
    scheduler.add_requests(states[1:])
    second = scheduler.schedule(clock_ms=1.0)
    assert second is not None and second.phase is SchedulePhase.PREFILL
    assert indices(second.scheduled_requests) == [1, 2]
    assert second.num_scheduled_tokens == 2 * 9


def test_scheduler_token_budget_decisions():
    """移植 chapter_06 check_scheduler_decisions：padded token 预算、decode protection 与 oversize prefill。"""
    # prefill-first：free batch slot 内整批入场
    pf_scheduler = IterationScheduler(
        SchedulerConfig(max_running_requests=4, batching_policy=PrefillFirstConfig()),
        CONTINUOUS,
    )
    pf_scheduler.running_queue.admit([RequestState(Request(99, (11, 12), 4), cache_length=2)], clock_ms=0.0)
    pf_scheduler.add_requests(make_states_with_prompt_lens([3, 5, 9]))
    decision = pf_scheduler.schedule(clock_ms=0.0)
    assert decision is not None and decision.phase is SchedulePhase.PREFILL
    assert indices(decision.scheduled_requests) == [0, 1, 2]
    assert not decision.is_oversize_prefill

    # token-budget=6：[3] 计 3 ≤ 6；[3, 5] 计 2 × 5 = 10 > 6，只取队头
    budgeted = IterationScheduler(
        SchedulerConfig(max_running_requests=4, batching_policy=TokenBudgetConfig(6)),
        CONTINUOUS,
    )
    budgeted.running_queue.admit([RequestState(Request(99, (11, 12), 4), cache_length=2)], clock_ms=0.0)
    budgeted.add_requests(make_states_with_prompt_lens([3, 5, 9]))
    decision = budgeted.schedule(clock_ms=0.0)
    assert decision is not None and decision.phase is SchedulePhase.PREFILL
    assert indices(decision.scheduled_requests) == [0]
    assert decision.num_scheduled_tokens == 3
    assert not decision.is_oversize_prefill
    budgeted.update_from_output(decision, [5], clock_ms=0.5)

    # 上一步是 PREFILL 且 RunningQueue 非空：decode protection 强制 DECODE
    decision = budgeted.schedule(clock_ms=1.0)
    assert decision is not None and decision.phase is SchedulePhase.DECODE
    assert decision.num_scheduled_tokens == len(budgeted.running_queue)
    assert not decision.is_oversize_prefill

    # 队头单独超预算：oversize prefill 独自执行，避免永久阻塞 FCFS 队列
    oversize = IterationScheduler(
        SchedulerConfig(max_running_requests=4, batching_policy=TokenBudgetConfig(6)),
        CONTINUOUS,
    )
    oversize.add_requests(make_states_with_prompt_lens([3, 5, 9])[2:])
    decision = oversize.schedule(clock_ms=0.0)
    assert decision is not None and decision.phase is SchedulePhase.PREFILL
    assert indices(decision.scheduled_requests) == [2]
    assert decision.num_scheduled_tokens == 9
    assert decision.is_oversize_prefill


@pytest.mark.parametrize(
    ("prompt_lens", "token_budget", "capacity", "expected", "is_oversize"),
    [
        # padded 计费：2、2×3、3×3 = 9 恰好等于预算（含边界），第 4 个请求 4×3 = 12 超预算
        ([2, 3, 3, 1], 9, 4, [0, 1, 2], False),
        # FCFS 不跳过队头：[4, 8] 计 16 > 8 即停，即使后面的 1 Token 请求放得下
        ([4, 8, 1], 8, 3, [0], False),
        # 与 batch slot 约束取交集：预算充足时由 free batch slot 数截断
        ([1, 1, 1, 1], 100, 2, [0, 1], False),
        # 队头单独恰好等于预算：不算 oversize
        ([5, 1], 5, 2, [0], False),
        # oversize prefill：队头 7 > 预算 5，单独放行
        ([7, 1], 5, 2, [0], True),
    ],
)
def test_scheduler_token_budget_selects_fcfs_prefix(prompt_lens, token_budget, capacity, expected, is_oversize):
    scheduler = IterationScheduler(
        SchedulerConfig(max_running_requests=capacity, batching_policy=TokenBudgetConfig(token_budget)),
        CONTINUOUS,
    )
    scheduler.add_requests(make_states_with_prompt_lens(prompt_lens))
    out = scheduler.schedule(clock_ms=0.0)

    assert out is not None and out.phase is SchedulePhase.PREFILL
    assert indices(out.scheduled_requests) == expected
    assert out.num_scheduled_tokens == len(expected) * max(prompt_lens[i] for i in expected)
    assert out.is_oversize_prefill is is_oversize


def test_scheduler_token_budget_with_paged_admission_takes_intersection():
    # 3 个可分配块：Prompt 4/4/4 各需 1 块 -> KV 显存允许 3 个；预算 8 只允许 2 个（3 × 4 = 12 > 8）
    manager = KVCacheManager(block_size=4, num_blocks=4)
    scheduler = IterationScheduler(
        SchedulerConfig(max_running_requests=4, batching_policy=TokenBudgetConfig(8)),
        CONTINUOUS,
        manager,
    )
    scheduler.add_requests(make_states_with_prompt_lens([4, 4, 4]))
    out = scheduler.schedule(clock_ms=0.0)
    assert out is not None and indices(out.scheduled_requests) == [0, 1]


def test_scheduler_token_budget_decode_protection_lifecycle():
    # 预算 2：每次 Prefill 只放 1 个 2-token 请求
    scheduler = IterationScheduler(
        SchedulerConfig(max_running_requests=2, batching_policy=TokenBudgetConfig(2)),
        CONTINUOUS,
    )
    scheduler.add_requests(make_states_with_prompt_lens([2, 2, 2]))

    # PREFILL -> (running 非空) DECODE -> PREFILL：decode protection 只阻止紧邻的第二个 PREFILL
    first = step(scheduler, 0.0)
    assert first.phase is SchedulePhase.PREFILL
    second = step(scheduler, 1.0)
    assert second.phase is SchedulePhase.DECODE
    third = step(scheduler, 2.0)
    assert third.phase is SchedulePhase.PREFILL

    # 刚入场的请求在 Prefill 当步全部结束（RunningQueue 为空）：允许紧接着再 PREFILL
    scheduler.running_queue.retire(list(scheduler.running_queue.requests))
    fourth = step(scheduler, 3.0)
    assert fourth.phase is SchedulePhase.PREFILL

    # 两队列均为空：空闲返回 None
    scheduler.running_queue.retire(list(scheduler.running_queue.requests))
    assert scheduler.schedule(clock_ms=4.0) is None


# ---------------------------------------------------------------------------
# chunked-prefill：Prompt 按 token budget 硬预算切 chunk，跨步推进
# ---------------------------------------------------------------------------


def chunked_scheduler(token_budget: int, capacity: int, manager: KVCacheManager | None = None) -> IterationScheduler:
    return IterationScheduler(
        SchedulerConfig(max_running_requests=capacity, batching_policy=ChunkedPrefillConfig(token_budget)),
        CONTINUOUS,
        manager,
    )


def chunk_ranges(out: SchedulerOutput) -> list[tuple[int, int, int]]:
    return [(c.state.request.request_idx, c.start, c.end) for c in out.prefill_chunks]


def step(scheduler: IterationScheduler, clock_ms: float) -> SchedulerOutput:
    """调度一步并以占位 Token 推进状态（chunked 中间 chunk 不产出 Token）。"""
    out = scheduler.schedule(clock_ms)
    assert out is not None
    if out.phase is SchedulePhase.PREFILL:
        num_tokens = sum(c.is_last for c in out.prefill_chunks)
    else:
        num_tokens = len(out.scheduled_requests)
    scheduler.update_from_output(out, [7] * num_tokens, clock_ms, eos_token_id=None, stop_on_eos=False)
    return out


def test_chunked_prefill_config_validation_and_str():
    with pytest.raises(ValueError, match="token_budget"):
        ChunkedPrefillConfig(0)
    assert str(ChunkedPrefillConfig(64)) == "chunked-prefill(token_budget=64)"


def test_chunked_prefill_splits_prompt_across_steps_with_shared_budget():
    """chunked 计划：候选数 n 时每请求 chunk 上限 token_budget // n，尾 chunk 更短；n 随请求进入 Decode 而减小。"""
    scheduler = chunked_scheduler(token_budget=8, capacity=4)
    scheduler.add_requests(make_states_with_prompt_lens([10, 10, 3]))

    first = step(scheduler, 0.0)  # n=3 -> 上限 2
    assert chunk_ranges(first) == [(0, 0, 2), (1, 0, 2), (2, 0, 2)]
    assert first.num_scheduled_tokens == 3 * 2
    assert not first.is_oversize_prefill
    assert len(scheduler.waiting_queue) == 0 and len(scheduler.running_queue) == 3

    second = step(scheduler, 1.0)  # 请求 2 仅剩 1 Token（尾 chunk 更短），写满后立即产出首 Token 进入 Decode
    assert chunk_ranges(second) == [(0, 2, 4), (1, 2, 4), (2, 2, 3)]
    assert second.num_scheduled_tokens == 3 * 2

    # 请求 2 进入 Decode：decode protection 强制插入一步仅含请求 2 的 DECODE（在途 chunked prefill 请求不参与）
    decode = step(scheduler, 2.0)
    assert decode.phase is SchedulePhase.DECODE
    assert indices(decode.scheduled_requests) == [2]
    assert decode.num_scheduled_tokens == 1

    third = step(scheduler, 3.0)  # n=2 -> 上限 4
    assert chunk_ranges(third) == [(0, 4, 8), (1, 4, 8)]
    assert third.num_scheduled_tokens == 2 * 4


def test_chunked_prefill_decode_protection_alternates_prefill_and_decode():
    scheduler = chunked_scheduler(token_budget=4, capacity=2)
    scheduler.add_requests(make_states_with_prompt_lens([2, 12]))

    phases = []
    for clock_ms in range(6):
        out = step(scheduler, float(clock_ms))
        phases.append(out.phase)
    # 请求 0 首步写满进入 Decode；此后请求 1 的 chunk 与 Decode 严格交替，不连续两步 PREFILL
    assert phases[0] is SchedulePhase.PREFILL
    assert phases[1:] == [
        SchedulePhase.DECODE,
        SchedulePhase.PREFILL,
        SchedulePhase.DECODE,
        SchedulePhase.PREFILL,
        SchedulePhase.DECODE,
    ]


def test_chunked_prefill_without_decoders_runs_consecutive_prefill_steps():
    scheduler = chunked_scheduler(token_budget=3, capacity=1)
    scheduler.add_requests(make_states_with_prompt_lens([7]))

    out = step(scheduler, 0.0)
    assert chunk_ranges(out) == [(0, 0, 3)]
    # 在途 chunked prefill 请求不是 Decode 请求：不触发 decode protection，可连续 PREFILL 直到写满
    second = step(scheduler, 1.0)
    assert second.phase is SchedulePhase.PREFILL
    assert chunk_ranges(second) == [(0, 3, 6)]
    third = step(scheduler, 2.0)
    assert chunk_ranges(third) == [(0, 6, 7)]
    assert third.prefill_chunks[0].is_last


def test_chunked_prefill_never_exceeds_hard_budget_and_covers_every_prompt_once():
    prompt_lens = [17, 3, 9, 1, 25, 6, 12]
    budget = 7
    scheduler = chunked_scheduler(token_budget=budget, capacity=4)
    states = make_states_with_prompt_lens(prompt_lens)
    scheduler.add_requests(states)

    covered: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(prompt_lens))}
    for clock_ms in range(200):
        if not scheduler.has_unfinished_requests:
            break
        out = step(scheduler, float(clock_ms))
        if out.phase is SchedulePhase.PREFILL:
            assert out.num_scheduled_tokens <= budget
            assert out.num_scheduled_tokens == len(out.prefill_chunks) * max(c.num_tokens for c in out.prefill_chunks)
            assert not out.is_oversize_prefill
            assert indices(out.scheduled_requests) == [c.state.request.request_idx for c in out.prefill_chunks]
            for chunk in out.prefill_chunks:
                covered[chunk.state.request.request_idx].append((chunk.start, chunk.end))
    assert not scheduler.has_unfinished_requests
    # 每个请求的 chunk 首尾相接、恰好覆盖 [0, prompt_len)
    for request_idx, ranges in covered.items():
        assert ranges[0][0] == 0 and ranges[-1][1] == prompt_lens[request_idx]
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


def test_chunked_prefill_partial_chunk_produces_no_token_and_keeps_request_prefilling():
    scheduler = chunked_scheduler(token_budget=2, capacity=2)
    states = make_states_with_prompt_lens([5])
    scheduler.add_requests(states)

    out = scheduler.schedule(0.0)
    assert out is not None and chunk_ranges(out) == [(0, 0, 2)]
    assert states[0].cache_length == 0 and states[0].is_prefilling  # chunk 处理完前 cache_length 不推进
    assert scheduler.update_from_output(out, [], 1.0) == []
    assert states[0].cache_length == 2 and states[0].is_prefilling
    assert states[0].generated == [] and states[0].first_token_ms is None
    assert scheduler.running_queue.prefilling_requests == (states[0],)
    assert scheduler.running_queue.decoding_requests == ()

    step(scheduler, 1.0)  # [2, 4)
    last = scheduler.schedule(2.0)  # [4, 5)
    assert last is not None and last.prefill_chunks[0].is_last
    assert states[0].cache_length == 4 and states[0].is_prefilling
    scheduler.update_from_output(last, [9], 3.0, eos_token_id=None, stop_on_eos=False)
    assert not states[0].is_prefilling and states[0].generated == [9] and states[0].first_token_ms == 3.0
    assert states[0].cache_length == 5  # 首个 Token 的 K/V 由下一步 Decode 写入，cache_length 仍为 prompt_len


def test_chunked_prefill_allocates_blocks_per_chunk():
    manager = KVCacheManager(block_size=4, num_blocks=8)
    scheduler = chunked_scheduler(token_budget=6, capacity=2, manager=manager)
    scheduler.add_requests(make_states_with_prompt_lens([10]))

    step(scheduler, 0.0)  # [0, 6)：2 块
    assert manager.num_used_blocks == 2
    step(scheduler, 1.0)  # [6, 10)：累计 10 Token -> 3 块
    assert manager.num_used_blocks == 3
    assert len(manager.block_table(0)) == 3


def test_chunked_prefill_admission_reserves_blocks_of_in_flight_prompts():
    """在途请求尚未写入的 Prompt 块先行预留：新请求不能抢占，否则两个半截 Prefill 互相等块而卡死。"""
    manager = KVCacheManager(block_size=4, num_blocks=6)  # 5 个可分配块
    scheduler = chunked_scheduler(token_budget=4, capacity=4, manager=manager)
    scheduler.add_requests(make_states_with_prompt_lens([12, 12]))  # 各需 3 块

    # 步 1：请求 0 入场（3 <= 5），请求 1 在扣除请求 0 之外仅剩 2 块 -> 不入场；chunk 上限随候选数 1 为 4
    first = step(scheduler, 0.0)
    assert chunk_ranges(first) == [(0, 0, 4)]
    assert manager.num_used_blocks == 1

    # 步 2：空闲 4 块，但请求 0 仍需追加 2 块 -> 可用 2 块，请求 1（需 3 块）仍不能入场
    second = step(scheduler, 1.0)
    assert chunk_ranges(second) == [(0, 4, 8)]
    assert len(scheduler.waiting_queue) == 1

    third = step(scheduler, 2.0)
    assert chunk_ranges(third) == [(0, 8, 12)]
    assert manager.num_used_blocks == 3
    # 请求 0 写满后释放前，剩余 2 块不足以容纳请求 1
    assert len(scheduler.waiting_queue) == 1


def test_chunked_prefill_chunk_allocation_failure_fails_loudly():
    """Decode 增长吞掉了为 chunked prefill 预留的块：显式报错（未实现抢占），而不是静默写坏 block table。"""
    manager = KVCacheManager(block_size=4, num_blocks=4)  # 3 个可分配块
    scheduler = chunked_scheduler(token_budget=4, capacity=2, manager=manager)
    scheduler.add_requests(make_states_with_prompt_lens([4, 8]))
    first = step(scheduler, 0.0)  # 请求 0 [0, 4) 写满（1 块）；请求 1 (8 Token, 2 块) 受预留约束 3-0 >= 2 入场
    assert chunk_ranges(first) == [(0, 0, 2), (1, 0, 2)]
    assert manager.allocate_slots(99, 4)  # 外部占走剩余的块，模拟 Decode 增长吞掉预留
    assert manager.free_block_queue.num_free_blocks == 0
    step(scheduler, 1.0)  # 请求 0 写满进入 Decode；请求 1 [2, 4) 仍在已持有的 1 块内
    with pytest.raises(RuntimeError, match="物理块已耗尽"):
        for clock_ms in range(2, 6):
            step(scheduler, float(clock_ms))
