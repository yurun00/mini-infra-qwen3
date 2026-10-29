"""ArrivalQueue（FCFS 到达队列）、调度策略、record_tokens 与请求规格校验的单元测试（纯 Python，无需模型）。"""

import pytest

from mini_infra.core import (
    ArrivalQueue,
    RequestSpec,
    RequestState,
    RequestStatus,
    RunningQueue,
    WaitingQueue,
    make_request_specs,
    prepare_schedule,
    validate_request_specs,
)


def make_states(arrivals: list[float]) -> list[RequestState]:
    return [RequestState(RequestSpec(i, (1, 2, 3), 4, arrival)) for i, arrival in enumerate(arrivals)]


def indices(states: list[RequestState]) -> list[int]:
    return [state.spec.request_index for state in states]


def test_pop_ready_only_returns_arrived_requests_in_arrival_order():
    queue = ArrivalQueue(make_states([30.0, 0.0, 10.0]))
    assert indices(queue.pop_ready(clock_ms=5.0, limit=8)) == [1]
    assert indices(queue.pop_ready(clock_ms=10.0, limit=8)) == [2]
    assert indices(queue.pop_ready(clock_ms=100.0, limit=8)) == [0]
    assert not queue


def test_pop_ready_respects_limit_and_keeps_submission_order_for_ties():
    queue = ArrivalQueue(make_states([0.0, 0.0, 0.0, 0.0]))
    assert indices(queue.pop_ready(0.0, limit=2)) == [0, 1]
    assert indices(queue.pop_ready(0.0, limit=0)) == []
    assert indices(queue.pop_ready(0.0, limit=8)) == [2, 3]


def test_next_arrival_and_len():
    queue = ArrivalQueue(make_states([50.0, 20.0]))
    assert len(queue) == 2
    assert queue.next_arrival_ms() == 20.0
    queue.pop_ready(20.0, limit=1)
    assert queue.next_arrival_ms() == 50.0
    queue.pop_ready(50.0, limit=1)
    with pytest.raises(IndexError):
        queue.next_arrival_ms()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_index": -1, "token_ids": (1,), "max_new_tokens": 1},
        {"request_index": 0, "token_ids": (), "max_new_tokens": 1},
        {"request_index": 0, "token_ids": (1,), "max_new_tokens": 0},
        {"request_index": 0, "token_ids": (1,), "max_new_tokens": 1, "arrival_ms": -1.0},
    ],
)
def test_request_spec_rejects_invalid_fields(kwargs):
    with pytest.raises(ValueError):
        RequestSpec(**kwargs)


def test_make_request_specs_broadcasts_budget_and_defaults_arrival():
    specs = make_request_specs([[1, 2], [3]], 5)
    assert [(s.request_index, s.token_ids, s.max_new_tokens, s.arrival_ms) for s in specs] == [
        (0, (1, 2), 5, 0.0),
        (1, (3,), 5, 0.0),
    ]
    specs = make_request_specs([[1, 2], [3]], [2, 4], [0.0, 7.5])
    assert [(s.max_new_tokens, s.arrival_ms) for s in specs] == [(2, 0.0), (4, 7.5)]


def test_make_request_specs_rejects_length_mismatch_and_duplicates():
    with pytest.raises(ValueError, match="数量必须一致"):
        make_request_specs([[1], [2]], [1, 2, 3])
    with pytest.raises(ValueError, match="数量必须一致"):
        make_request_specs([[1], [2]], 1, [0.0])
    with pytest.raises(ValueError, match="不能为空"):
        validate_request_specs([])
    spec = RequestSpec(0, (1,), 1)
    with pytest.raises(ValueError, match="唯一"):
        validate_request_specs([spec, spec])


def test_prepare_schedule_validates_and_returns_initial_clock():
    specs = make_request_specs([[1], [2], [3]], 2, [30.0, 10.0, 20.0])
    states, queue, clock_ms = prepare_schedule(specs, max_running_requests=2)
    assert len(states) == 3 and len(queue) == 3
    assert clock_ms == 10.0
    with pytest.raises(ValueError, match="max_running_requests"):
        prepare_schedule(specs, max_running_requests=0)
    with pytest.raises(ValueError, match="不能为空"):
        prepare_schedule([], max_running_requests=1)


def test_running_queue_step_prefers_eos_over_budget_and_tracks_timestamps():
    rq = RunningQueue(capacity=3)
    states = [RequestState(RequestSpec(i, (1,), 2)) for i in range(3)]
    rq.admit(states, clock_ms=0.0)

    # 产出第 1 批 token
    finished = rq.step([5, 5, 7], 1.0, eos_token_id=7, stop_on_eos=True)
    assert finished == [states[2]]
    assert states[2].finish_reason == "stop"
    rq.retire(finished)
    assert states[2].status is RequestStatus.FINISHED
    assert all(s.first_token_ms == 1.0 and s.token_times_ms == [1.0] for s in states)

    # 产出第 2 批 token（仅剩的前两项）
    finished = rq.step([5, 7], 2.0, eos_token_id=7, stop_on_eos=False)
    assert finished == states[:2]  # 预算（2 个 Token）耗尽；stop_on_eos=False 时 EOS 不触发 stop
    assert [s.finish_reason for s in states[:2]] == ["max_tokens", "max_tokens"]
    assert states[0].first_token_ms == 1.0 and states[0].completion_ms == 2.0
    rq.retire(finished)
    assert all(s.status is RequestStatus.FINISHED for s in states[:2])

    # 单请求 append_token 单元逻辑测试
    single = RequestState(RequestSpec(0, (1,), 2))
    assert not single.append_token(9, 10.0, eos_token_id=9, stop_on_eos=False)  # 仅产出第 1 个 token，未达预算
    assert single.append_token(9, 11.0, eos_token_id=9, stop_on_eos=False)  # 产出第 2 个 token，达到预算 2
    assert single.finish_reason == "max_tokens"
    single_eos = RequestState(RequestSpec(1, (1,), 10))
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


def test_running_queue_lifecycle_and_slot_management():
    rq = RunningQueue(capacity=3)
    assert rq.capacity == 3
    assert rq.num_free_slots == 3
    assert rq.can_admit()
    with pytest.raises(ValueError, match="必须大于 0"):
        RunningQueue(capacity=0)

    states = make_states([0.0, 0.0, 0.0])
    admitted = rq.admit(states[:2], clock_ms=5.0)
    assert indices(admitted) == [0, 1]
    assert [s.slot_index for s in admitted] == [0, 1]
    assert all(s.status is RequestStatus.RUNNING for s in admitted)
    assert all(s.admitted_ms == 5.0 for s in admitted)
    assert rq.num_free_slots == 1
    assert rq.free_slots == [2]

    # 尝试超量入场抛出异常
    with pytest.raises(ValueError, match="可用槽位不足"):
        rq.admit(states, clock_ms=5.0)

    # 释放已结束请求
    rq.retire([states[0]])
    assert states[0].status is RequestStatus.FINISHED
    assert rq.num_free_slots == 2
    assert rq.free_slots == [0, 2]
    assert indices(rq.requests) == [1]
