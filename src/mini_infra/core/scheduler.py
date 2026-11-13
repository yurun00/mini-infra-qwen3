"""FCFS 调度策略：三级队列（ArrivalQueue -> WaitingQueue -> RunningQueue）、入场退场管理与迭代级 Scheduler。

本模块只负责调度状态机的纯逻辑转移，不执行模型前向，也不管理显存：
- ArrivalQueue：外部到达流（arrival_ms > clock_ms 的未来请求，按到达时间排序）；
- WaitingQueue：已到达但未分配 batch slot 的就绪等待队列（RequestStatus.WAITING）；
- RunningQueue：已分配 batch slot 并在计算中的活跃队列（RequestStatus.RUNNING，batch slot 池上限为 capacity）；
- IterationScheduler：按 SchedulerConfig（prefill-first / token-budget）逐步产出 SchedulerOutput。
"""

import heapq
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from mini_infra.core.block_manager import KVCacheManager
from mini_infra.core.request import (
    Request,
    RequestState,
    RequestStatus,
    validate_requests,
)
from mini_infra.core.types import BatchingMethod, PrefillFirstConfig, SchedulerConfig


class ArrivalQueue:
    """按 arrival_ms 稳定排序的外部请求到达队列。

    Python 排序是稳定的：到达时间相同的请求保留调用方的提交顺序，作为 FCFS 次序。
    由于队列按到达时间有序，“已到达”的请求恒为队首前缀，出队为 O(1)，无需每轮全量扫描。
    """

    def __init__(self, states: Iterable[RequestState]):
        self._requests: deque[RequestState] = deque(sorted(states, key=lambda state: state.request.arrival_ms))

    def __len__(self) -> int:
        return len(self._requests)

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __iter__(self):
        return iter(self._requests)

    def pop_arrived(self, clock_ms: float) -> list[RequestState]:
        """取出所有 arrival_ms <= clock_ms 的已到达请求，转入 WaitingQueue。"""
        arrived: list[RequestState] = []
        while self._requests and self._requests[0].request.arrival_ms <= clock_ms:
            arrived.append(self._requests.popleft())
        return arrived

    def next_arrival_ms(self) -> float:
        """下一个请求的到达时间；队列为空时抛出 IndexError。"""
        return self._requests[0].request.arrival_ms


class WaitingQueue:
    """已到达但尚未分配 batch slot 的就绪等待队列（FCFS 次序）。"""

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
    def requests(self) -> tuple[RequestState, ...]:
        """当前等待队列的只读快照（FCFS 次序）。"""
        return tuple(self._requests)

    def append(self, state: RequestState) -> None:
        state.status = RequestStatus.WAITING
        self._requests.append(state)

    def extend(self, states: Iterable[RequestState]) -> None:
        for state in states:
            self.append(state)

    def pop_batch(self, limit: int) -> list[RequestState]:
        """按就绪次序取出至多 limit 个就绪请求用于调度入场。"""
        popped: list[RequestState] = []
        while self._requests and len(popped) < limit:
            popped.append(self._requests.popleft())
        return popped


class RunningQueue:
    """正在占用 batch slot 进行计算的活跃请求队列。"""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity 必须大于 0")
        self.capacity = capacity
        self._requests: list[RequestState] = []
        self._free_batch_slots: list[int] = list(range(capacity))
        heapq.heapify(self._free_batch_slots)

    def __len__(self) -> int:
        return len(self._requests)

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __iter__(self):
        return iter(self._requests)

    @property
    def free_batch_slots(self) -> tuple[int, ...]:
        """空闲 batch slot 号的只读快照（按最小堆次序，非排序结果）。专供测试断言使用（Test-only）。"""
        return tuple(self._free_batch_slots)

    @property
    def num_free_batch_slots(self) -> int:
        return len(self._free_batch_slots)

    @property
    def requests(self) -> tuple[RequestState, ...]:
        """当前活跃队列的只读快照。"""
        return tuple(self._requests)

    def can_admit(self) -> bool:
        return bool(self._free_batch_slots)

    def admit(self, states: Sequence[RequestState], clock_ms: float) -> list[RequestState]:
        """将就绪请求分配最小空闲 batch slot 并标记 RUNNING 状态，加入运行中队列。"""
        if len(states) > len(self._free_batch_slots):
            raise ValueError(f"可用 batch slot 不足：需要 {len(states)}，剩余 {len(self._free_batch_slots)}")
        for state in states:
            state.status = RequestStatus.RUNNING
            state.admitted_ms = clock_ms
            state.batch_slot_idx = heapq.heappop(self._free_batch_slots)
            state.cache_length = state.request.prompt_len
            self._requests.append(state)
        return list(states)

    def process_step_output(
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
        """释放已结束请求的 batch slot，标记 FINISHED 状态并从活跃队列中移出。"""
        for state in finished:
            state.status = RequestStatus.FINISHED
            assert state.batch_slot_idx is not None
            heapq.heappush(self._free_batch_slots, state.batch_slot_idx)
        self._requests = [s for s in self._requests if s.status is RequestStatus.RUNNING]


def prepare_schedule(
    requests: Sequence[Request], max_running_requests: int
) -> tuple[list[RequestState], ArrivalQueue, float]:
    """校验输入并建立运行态与到达队列，返回 (states, queue, 初始逻辑时钟)。"""
    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    validate_requests(requests)
    states = [RequestState(req) for req in requests]
    return states, ArrivalQueue(states), min(state.request.arrival_ms for state in states)


class SchedulePhase(str, Enum):
    """一次调度事件所处阶段。"""

    PREFILL = "prefill"
    DECODE = "decode"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SchedulerOutput:
    """单步迭代调度的决策产出：承载本步执行阶段、被选中请求快照与本步计费 Token 数。

    num_scheduled_tokens：PREFILL 为左 Padding 后实际送入模型的 padded prompt token 数
    （len(scheduled_requests) × max(prompt_len)）；DECODE 为运行中请求数（每请求 1 个 Token）。
    """

    phase: SchedulePhase
    scheduled_requests: tuple[RequestState, ...]
    num_scheduled_tokens: int


def _num_padded_prompt_tokens(states: Sequence[RequestState]) -> int:
    """左 Padding Batched Prefill 实际送入模型的 Token 数：batch 行数 × 最长 Prompt 长度。"""
    return len(states) * max(state.request.prompt_len for state in states)


def _select_within_budget(candidates: tuple[RequestState, ...], token_budget: int) -> tuple[RequestState, ...]:
    """按 FCFS 取 padded prompt token 数不超过 token_budget 的最长前缀；队头放不下即停，不向后跳选。

    无 Chunked Prefill 时长 Prompt 无法拆分：若队头请求单独已超预算，则让它独自执行（oversize prefill），
    避免其永久阻塞 FCFS 队列。
    """
    num_selected = 0
    max_prompt_len = 0
    for state in candidates:
        next_max_prompt_len = max(max_prompt_len, state.request.prompt_len)
        if (num_selected + 1) * next_max_prompt_len > token_budget:
            break
        num_selected += 1
        max_prompt_len = next_max_prompt_len
    return candidates[: max(num_selected, 1)]


class IterationScheduler:
    """迭代级请求调度器：在每个调度步决策执行一次 Prefill batch 或一次 Decode batch。

    入场决策全部收敛于此，依次取 WaitingQueue FCFS 前缀的交集（均为前缀截断，可交换、可组合）：
    1. batching 约束：continuous-batching 有 free batch slot 即可入场；fixed-batching / none 须等
       RunningQueue 完全排空（wave barrier）；
    2. batch slot 约束：至多 RunningQueue.num_free_batch_slots 个；
    3. KV 显存约束（仅 Paged）：KVCacheManager 空闲 physical block 可容纳的前缀；
    4. 策略约束：PrefillFirstConfig 不再截断；TokenBudgetConfig 按 token budget 截断，且 RunningQueue
       非空时禁止连续两步 PREFILL（decode protection），以限制 Prefill burst 对在途请求 ITL 的冲击。
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        batching_method: BatchingMethod,
        kv_cache_manager: KVCacheManager | None = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.batching_method = batching_method
        # Paged 路径的逻辑块管理器（只读查询空闲块）；Dense 路径为 None，不受 KV 显存约束
        self.kv_cache_manager = kv_cache_manager
        self._last_phase: SchedulePhase | None = None

    def schedule(self, waiting_queue: WaitingQueue, running_queue: RunningQueue) -> SchedulerOutput | None:
        """根据就绪队列与运行队列生成本步调度计划；空闲（两队列均为空）时返回 None。

        waiting_queue 只含已到达请求（到达过滤由 engine 主循环按逻辑时钟完成），故无需传入 clock_ms。
        """
        output = self._decide(waiting_queue, running_queue)
        if output is not None:
            self._last_phase = output.phase
        return output

    def _admission_candidates(
        self, waiting_queue: WaitingQueue, running_queue: RunningQueue
    ) -> tuple[RequestState, ...]:
        """batching、batch slot 与 KV 显存约束下可入场的 WaitingQueue FCFS 前缀。"""
        is_continuous = self.batching_method is BatchingMethod.CONTINUOUS
        can_admit = running_queue.can_admit() if is_continuous else not running_queue
        if not can_admit:
            return ()
        candidates = waiting_queue.requests[: running_queue.num_free_batch_slots]
        if self.kv_cache_manager is not None:
            num_admissible = self.kv_cache_manager.num_admissible([s.request.prompt_len for s in candidates])
            candidates = candidates[:num_admissible]
        return candidates

    def _prefill_output(self, selected: tuple[RequestState, ...]) -> SchedulerOutput:
        return SchedulerOutput(
            phase=SchedulePhase.PREFILL,
            scheduled_requests=selected,
            num_scheduled_tokens=_num_padded_prompt_tokens(selected),
        )

    def _decode_output(self, running_queue: RunningQueue) -> SchedulerOutput:
        return SchedulerOutput(
            phase=SchedulePhase.DECODE,
            scheduled_requests=running_queue.requests,
            num_scheduled_tokens=len(running_queue),
        )

    def _decide(self, waiting_queue: WaitingQueue, running_queue: RunningQueue) -> SchedulerOutput | None:
        config = self.scheduler_config
        admission_candidates = self._admission_candidates(waiting_queue, running_queue)
        if isinstance(config, PrefillFirstConfig):
            if admission_candidates:
                return self._prefill_output(admission_candidates)
            if running_queue:
                return self._decode_output(running_queue)
            return None

        if running_queue and self._last_phase is SchedulePhase.PREFILL:
            return self._decode_output(running_queue)
        if admission_candidates:
            return self._prefill_output(_select_within_budget(admission_candidates, config.token_budget))
        if running_queue:
            return self._decode_output(running_queue)
        return None
