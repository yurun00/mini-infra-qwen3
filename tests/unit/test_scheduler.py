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
    )
    assert len(scheduler.waiting_queue) == 1
    assert len(scheduler.running_queue) == 2
    assert [s.batch_slot_idx for s in states[:2]] == [0, 1]

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
    assert manager.num_used_blocks == 2
    finished_prefill = scheduler.update_from_output(
        prefill_out, [10, 11], clock_ms=1.0, eos_token_id=None, stop_on_eos=False
    )
    assert indices(finished_prefill) == [1]
    assert states[0].cache_length == 4  # Prefill 步不递增 cache_length
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
    pf_scheduler.running_queue.admit([RequestState(Request(99, (11, 12), 4))], clock_ms=0.0)
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
    budgeted.running_queue.admit([RequestState(Request(99, (11, 12), 4))], clock_ms=0.0)
    budgeted.add_requests(make_states_with_prompt_lens([3, 5, 9]))
    decision = budgeted.schedule(clock_ms=0.0)
    assert decision is not None and decision.phase is SchedulePhase.PREFILL
    assert indices(decision.scheduled_requests) == [0]
    assert decision.num_scheduled_tokens == 3
    assert not decision.is_oversize_prefill

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
    first = scheduler.schedule(clock_ms=0.0)
    assert first is not None and first.phase is SchedulePhase.PREFILL
    second = scheduler.schedule(clock_ms=1.0)
    assert second is not None and second.phase is SchedulePhase.DECODE
    third = scheduler.schedule(clock_ms=2.0)
    assert third is not None and third.phase is SchedulePhase.PREFILL

    # 刚入场的请求在 Prefill 当步全部结束（RunningQueue 为空）：允许紧接着再 PREFILL
    scheduler.running_queue.retire(list(scheduler.running_queue.requests))
    fourth = scheduler.schedule(clock_ms=3.0)
    assert fourth is not None and fourth.phase is SchedulePhase.PREFILL

    # 两队列均为空：空闲返回 None
    scheduler.running_queue.retire(list(scheduler.running_queue.requests))
    assert scheduler.schedule(clock_ms=4.0) is None
