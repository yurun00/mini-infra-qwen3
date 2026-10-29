"""FCFS 调度策略：三级队列（ArrivalQueue -> WaitingQueue -> RunningQueue）与入场退场管理。

本模块只负责调度状态机的纯逻辑转移，不执行模型前向，也不管理显存：
- ArrivalQueue：外部到达流（arrival_ms > clock_ms 的未来请求，按到达时间排序）；
- WaitingQueue：已到达但未分配槽位的就绪等待队列（RequestStatus.WAITING）；
- RunningQueue：已分配槽位并在计算中的活跃队列（RequestStatus.RUNNING，槽位池上限为 capacity）。
"""

import heapq
from collections import deque
from collections.abc import Iterable, Sequence

from mini_infra.core.request import (
    RequestSpec,
    RequestState,
    RequestStatus,
    validate_request_specs,
)


class ArrivalQueue:
    """按 arrival_ms 稳定排序的外部请求到达队列。

    Python 排序是稳定的：到达时间相同的请求保留调用方的提交顺序，作为 FCFS 次序。
    由于队列按到达时间有序，“已到达”的请求恒为队首前缀，出队为 O(1)，无需每轮全量扫描。
    """

    def __init__(self, states: Iterable[RequestState]):
        self._requests: deque[RequestState] = deque(sorted(states, key=lambda state: state.spec.arrival_ms))

    def __len__(self) -> int:
        return len(self._requests)

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __iter__(self):
        return iter(self._requests)

    @property
    def requests(self) -> list[RequestState]:
        return list(self._requests)

    def pop_arrived(self, clock_ms: float) -> list[RequestState]:
        """取出所有 arrival_ms <= clock_ms 的已到达请求，转入 WaitingQueue。"""
        arrived: list[RequestState] = []
        while self._requests and self._requests[0].spec.arrival_ms <= clock_ms:
            arrived.append(self._requests.popleft())
        return arrived

    def pop_ready(self, clock_ms: float, limit: int) -> list[RequestState]:
        """取出至多 limit 个 arrival_ms <= clock_ms 的请求（兼容旧接口）。"""
        ready: list[RequestState] = []
        while self._requests and len(ready) < limit and self._requests[0].spec.arrival_ms <= clock_ms:
            ready.append(self._requests.popleft())
        return ready

    def next_arrival_ms(self) -> float:
        """下一个请求的到达时间；队列为空时抛出 IndexError。"""
        return self._requests[0].spec.arrival_ms


class WaitingQueue:
    """已到达但尚未分配槽位的就绪等待队列（FCFS 次序）。"""

    def __init__(self, states: Iterable[RequestState] | None = None):
        self._requests: deque[RequestState] = deque()
        if states:
            self.extend(states)

    def __len__(self) -> int:
        return len(self._requests)

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __iter__(self):
        return iter(self._requests)

    @property
    def requests(self) -> list[RequestState]:
        return list(self._requests)

    def append(self, state: RequestState) -> None:
        state.status = RequestStatus.WAITING
        self._requests.append(state)

    def extend(self, states: Iterable[RequestState]) -> None:
        for state in states:
            self.append(state)

    def pop_batch(self, limit: int) -> list[RequestState]:
        """按就绪次序取出至多 limit 个就绪请求用于调度入场。"""
        poped: list[RequestState] = []
        while self._requests and len(poped) < limit:
            poped.append(self._requests.popleft())
        return poped


class RunningQueue:
    """正在占用槽位进行计算的活跃请求队列。"""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity 必须大于 0")
        self.capacity = capacity
        self._requests: list[RequestState] = []
        self._free_slots: list[int] = list(range(capacity))
        heapq.heapify(self._free_slots)

    def __len__(self) -> int:
        return len(self._requests)

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __iter__(self):
        return iter(self._requests)

    @property
    def free_slots(self) -> list[int]:
        return self._free_slots

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def requests(self) -> list[RequestState]:
        return self._requests

    def can_admit(self) -> bool:
        return bool(self._free_slots)

    def admit(self, states: Sequence[RequestState], clock_ms: float) -> list[RequestState]:
        """将就绪请求分配最小空闲槽位并标记 RUNNING 状态，加入运行中队列。"""
        if len(states) > len(self._free_slots):
            raise ValueError(f"可用槽位不足：需要 {len(states)}，剩余 {len(self._free_slots)}")
        for state in states:
            state.status = RequestStatus.RUNNING
            state.admitted_ms = clock_ms
            state.slot_index = heapq.heappop(self._free_slots)
            state.cache_length = state.spec.prompt_len
            self._requests.append(state)
        return list(states)

    def step(
        self,
        token_ids: Sequence[int],
        clock_ms: float,
        eos_token_id: int | None = None,
        stop_on_eos: bool = True,
        states: Sequence[RequestState] | None = None,
    ) -> list[RequestState]:
        """记录本步产出的 Token，返回达到结束条件的请求列表。

        默认推进全部活跃请求（self._requests）；也可通过 states 指定只推进部分请求（如刚入场的批次）。
        """
        targets = self._requests if states is None else states
        finished: list[RequestState] = []
        for state, token_id in zip(targets, token_ids):
            if state.append_token(token_id, clock_ms, eos_token_id, stop_on_eos):
                finished.append(state)
        return finished

    def retire(self, finished: Sequence[RequestState]) -> None:
        """释放已结束请求的槽位，标记 FINISHED 状态并从活跃队列中移出。"""
        for state in finished:
            state.status = RequestStatus.FINISHED
            if state.slot_index is not None:
                heapq.heappush(self._free_slots, state.slot_index)
        self._requests = [s for s in self._requests if s.status is RequestStatus.RUNNING]


def prepare_schedule(
    request_specs: Sequence[RequestSpec], max_running_requests: int
) -> tuple[list[RequestState], ArrivalQueue, float]:
    """校验输入并建立运行态与到达队列，返回 (states, queue, 初始逻辑时钟)。"""
    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    validate_request_specs(request_specs)
    states = [RequestState(spec) for spec in request_specs]
    return states, ArrivalQueue(states), min(state.spec.arrival_ms for state in states)
