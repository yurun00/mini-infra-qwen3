"""ArrivalQueue（FCFS 到达队列）、调度策略、record_tokens 与请求规格校验的单元测试（纯 Python，无需模型）。"""

from collections.abc import Sequence

import pytest

from mini_infra.core import (
    ArrivalQueue,
    BatchingMethod,
    IterationScheduler,
    KVCacheManager,
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


def test_scheduler_prefill_first_decisions():
    scheduler = IterationScheduler(SchedulerConfig(PrefillFirstConfig()), CONTINUOUS)
    wq = WaitingQueue(make_states([0.0, 0.0, 0.0]))
    rq = RunningQueue(capacity=2)

    # 1. 有就绪请求且有 free batch slot：按 FCFS 前缀（受 batch slot 数截断）生成 PREFILL 计划
    out = scheduler.schedule(wq, rq)
    assert out == SchedulerOutput(
        phase=SchedulePhase.PREFILL,
        scheduled_requests=wq.requests[:2],
        num_scheduled_tokens=2 * 3,
    )

    # 2. batch slot 用尽但存在运行中请求：生成 DECODE 计划，每个运行中请求计 1 个 Token
    rq.admit(wq.pop_batch(2), clock_ms=0.0)
    out_decode = scheduler.schedule(wq, rq)
    assert out_decode == SchedulerOutput(
        phase=SchedulePhase.DECODE,
        scheduled_requests=rq.requests,
        num_scheduled_tokens=2,
    )

    # 3. 既无就绪请求也无运行中请求：返回 None
    rq.retire(list(rq.requests))
    assert scheduler.schedule(WaitingQueue(), rq) is None


@pytest.mark.parametrize("batching_method", [BatchingMethod.FIXED, BatchingMethod.NONE])
def test_scheduler_wave_barrier_blocks_admission_until_running_queue_drains(batching_method):
    scheduler = IterationScheduler(SchedulerConfig(PrefillFirstConfig()), batching_method)
    wq = WaitingQueue(make_states([0.0, 0.0, 0.0]))
    rq = RunningQueue(capacity=2)
    rq.admit(wq.pop_batch(1), clock_ms=0.0)

    # 仍有 free batch slot，但 wave barrier 要求 RunningQueue 排空后才入场下一波
    out = scheduler.schedule(wq, rq)
    assert out is not None and out.phase is SchedulePhase.DECODE
    rq.retire(list(rq.requests))
    out = scheduler.schedule(wq, rq)
    assert out is not None and out.phase is SchedulePhase.PREFILL
    assert indices(out.scheduled_requests) == [1, 2]


def test_scheduler_paged_admission_limited_by_free_blocks():
    # block_size=4，num_blocks=4：1 个 null block + 3 个可分配块
    manager = KVCacheManager(block_size=4, num_blocks=4)
    scheduler = IterationScheduler(SchedulerConfig(PrefillFirstConfig()), CONTINUOUS, manager)
    rq = RunningQueue(capacity=8)

    # Prompt 6/6/2 各需 2/2/1 块：只够队头 1 个；FCFS 不跳过队头去放 2-token 请求
    out = scheduler.schedule(WaitingQueue(make_states_with_prompt_lens([6, 6, 2])), rq)
    assert out is not None and indices(out.scheduled_requests) == [0]
    # Prompt 6/2/6 各需 2/1/2 块：前 2 个共 3 块恰好放下
    out = scheduler.schedule(WaitingQueue(make_states_with_prompt_lens([6, 2, 6])), rq)
    assert out is not None and indices(out.scheduled_requests) == [0, 1]
    # 空闲块被占用后不可入场，运行队列为空时空闲返回 None
    assert manager.allocate_slots(99, 12)
    assert scheduler.schedule(WaitingQueue(make_states_with_prompt_lens([1])), rq) is None


def test_scheduler_prefill_first_has_no_decode_protection():
    scheduler = IterationScheduler(SchedulerConfig(PrefillFirstConfig()), CONTINUOUS)
    states = make_states_with_prompt_lens([3, 5, 9])
    wq = WaitingQueue(states[:1])
    rq = RunningQueue(capacity=4)

    first = scheduler.schedule(wq, rq)
    assert first is not None and first.phase is SchedulePhase.PREFILL
    rq.admit(wq.pop_batch(len(first.scheduled_requests)), clock_ms=0.0)
    wq.extend(states[1:])
    second = scheduler.schedule(wq, rq)
    assert second is not None and second.phase is SchedulePhase.PREFILL
    assert indices(second.scheduled_requests) == [1, 2]
    assert second.num_scheduled_tokens == 2 * 9


def test_scheduler_token_budget_decisions():
    """移植 chapter_06 check_scheduler_decisions：padded token 预算、decode protection 与 oversize prefill。"""
    waiting = make_states_with_prompt_lens([3, 5, 9])
    running_state = RequestState(Request(99, (11, 12), 4))

    def queues(waiting_states: list[RequestState]) -> tuple[WaitingQueue, RunningQueue]:
        rq = RunningQueue(capacity=4)
        rq.admit([running_state], clock_ms=0.0)
        return WaitingQueue(waiting_states), rq

    # prefill-first：free batch slot 内整批入场
    wq, rq = queues(waiting)
    decision = IterationScheduler(SchedulerConfig(PrefillFirstConfig()), CONTINUOUS).schedule(wq, rq)
    assert decision is not None and decision.phase is SchedulePhase.PREFILL
    assert indices(decision.scheduled_requests) == [0, 1, 2]
    assert not decision.is_oversize_prefill

    # token-budget=6：[3] 计 3 ≤ 6；[3, 5] 计 2 × 5 = 10 > 6，只取队头
    budgeted = IterationScheduler(SchedulerConfig(TokenBudgetConfig(6)), CONTINUOUS)
    decision = budgeted.schedule(wq, rq)
    assert decision is not None and decision.phase is SchedulePhase.PREFILL
    assert indices(decision.scheduled_requests) == [0]
    assert decision.num_scheduled_tokens == 3
    assert not decision.is_oversize_prefill

    # 上一步是 PREFILL 且 RunningQueue 非空：decode protection 强制 DECODE
    wq, rq = queues(waiting[1:])
    decision = budgeted.schedule(wq, rq)
    assert decision is not None and decision.phase is SchedulePhase.DECODE
    assert decision.num_scheduled_tokens == len(rq)
    assert not decision.is_oversize_prefill

    # 队头单独超预算：oversize prefill 独自执行，避免永久阻塞 FCFS 队列
    oversize = IterationScheduler(SchedulerConfig(TokenBudgetConfig(6)), CONTINUOUS)
    decision = oversize.schedule(WaitingQueue(waiting[2:]), RunningQueue(capacity=4))
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
    wq = WaitingQueue(make_states_with_prompt_lens(prompt_lens))
    out = IterationScheduler(SchedulerConfig(TokenBudgetConfig(token_budget)), CONTINUOUS).schedule(
        wq, RunningQueue(capacity)
    )

    assert out is not None and out.phase is SchedulePhase.PREFILL
    assert indices(out.scheduled_requests) == expected
    assert out.num_scheduled_tokens == len(expected) * max(prompt_lens[i] for i in expected)
    assert out.is_oversize_prefill is is_oversize


def test_scheduler_token_budget_with_paged_admission_takes_intersection():
    # 3 个可分配块：Prompt 4/4/4 各需 1 块 -> KV 显存允许 3 个；预算 8 只允许 2 个（3 × 4 = 12 > 8）
    manager = KVCacheManager(block_size=4, num_blocks=4)
    out = IterationScheduler(SchedulerConfig(TokenBudgetConfig(8)), CONTINUOUS, manager).schedule(
        WaitingQueue(make_states_with_prompt_lens([4, 4, 4])), RunningQueue(capacity=4)
    )
    assert out is not None and indices(out.scheduled_requests) == [0, 1]


def test_scheduler_token_budget_decode_protection_lifecycle():
    # 预算 2：每次 Prefill 只放 1 个 2-token 请求
    scheduler = IterationScheduler(SchedulerConfig(TokenBudgetConfig(2)), CONTINUOUS)
    wq = WaitingQueue(make_states_with_prompt_lens([2, 2, 2]))
    rq = RunningQueue(capacity=2)

    # PREFILL -> (running 非空) DECODE -> PREFILL：decode protection 只阻止紧邻的第二个 PREFILL
    first = scheduler.schedule(wq, rq)
    assert first is not None and first.phase is SchedulePhase.PREFILL
    rq.admit(wq.pop_batch(1), clock_ms=0.0)
    second = scheduler.schedule(wq, rq)
    assert second is not None and second.phase is SchedulePhase.DECODE
    third = scheduler.schedule(wq, rq)
    assert third is not None and third.phase is SchedulePhase.PREFILL
    rq.admit(wq.pop_batch(1), clock_ms=0.0)

    # 刚入场的请求在 Prefill 当步全部结束（RunningQueue 为空）：允许紧接着再 PREFILL
    rq.retire(list(rq.requests))
    fourth = scheduler.schedule(wq, rq)
    assert fourth is not None and fourth.phase is SchedulePhase.PREFILL

    # 两队列均为空：空闲返回 None
    assert scheduler.schedule(WaitingQueue(), RunningQueue(capacity=2)) is None
