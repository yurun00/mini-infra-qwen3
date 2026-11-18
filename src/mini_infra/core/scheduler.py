"""三级调度队列（ArrivalQueue -> WaitingQueue -> RunningQueue）、入场退场管理与迭代级 Scheduler。

本模块只负责调度状态机的纯逻辑转移，不执行模型前向，也不管理物理显存：
- ArrivalQueue：外部到达流（arrival_ms > clock_ms 的未来请求，按到达时间排序）；
- WaitingQueue：已到达但未分配 batch slot 的就绪等待队列（RequestStatus.WAITING）；
- RunningQueue：已分配 batch slot 并在计算中的活跃队列（RequestStatus.RUNNING，batch slot 池上限为 capacity），
  含 Prompt 尚未写满 KV Cache 的 chunked prefill 请求（对标 vLLM V1 running 列表含部分 Prefill 请求）；
- IterationScheduler：按 SchedulerConfig 的 batching policy（prefill-first / token-budget / chunked-prefill）
  逐步产出 SchedulerOutput。

正交调度维度演进路线（详见 `core/types.py`）：
1. `queue_policy`（排队序）：当前固定为 FCFS；TODO 支持 `priority`（优先级）与 `srtf`（最短剩余时间优先）。
2. `batching_policy`（单步组批与阶段准入）：当前支持 `prefill-first`、`token-budget` 与 `chunked-prefill`；
   TODO 支持 chunk 与 Decode 同一次前向混批。
3. `preemption_mode`（抢占驱逐模式）：TODO 支持 Decode 阶段物理块耗尽时的 `recompute`（重算）与 `swap`（CPU 换入换出）。
4. `watermark`（准入容量水印）：TODO 支持 Prefill 准入时预留空闲块安全垫。
"""

import heapq
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from mini_infra.core.block_manager import KVCacheManager, num_blocks_for_tokens
from mini_infra.core.request import (
    Request,
    RequestState,
    RequestStatus,
    validate_requests,
)
from mini_infra.core.types import (
    BatchingMethod,
    ChunkedPrefillConfig,
    PrefillFirstConfig,
    SchedulerConfig,
)


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
    """已到达但尚未分配 batch slot 的就绪等待队列。

    TODO(queue_policy): 当前仅实现 FCFS 次序（deque FIFO）；后续接入 `queue_policy`（"fcfs" | "priority" | "srtf"）
    时，在此处按策略比较器维护出队优先级，以及支持抢占回退请求的队首/按序重插（prepend_preempted）。
    """

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
    """正在占用 batch slot 进行计算的活跃请求队列。

    TODO(preemption_mode / queue_policy): 后续当 Decode 步物理块耗尽触发抢占时，
    按 `queue_policy` 的逆序（FCFS 下选最晚入场请求，priority/srtf 下选最低优请求）从本队列尾部挑出 Victim，
    并配合 `preemption_mode`（"recompute" | "swap"）释放 batch slot 与物理块。
    """

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

    @property
    def prefilling_requests(self) -> tuple[RequestState, ...]:
        """Prompt 尚未全部写入 KV Cache 的 chunked prefill 请求（按入场 FCFS 次序）。"""
        return tuple(s for s in self._requests if s.is_prefilling)

    @property
    def decoding_requests(self) -> tuple[RequestState, ...]:
        """Prompt 已全部写入 KV Cache、逐步 Decode 的请求。"""
        return tuple(s for s in self._requests if not s.is_prefilling)

    def can_admit(self) -> bool:
        return bool(self._free_batch_slots)

    def admit(self, states: Sequence[RequestState], clock_ms: float) -> list[RequestState]:
        """将就绪请求分配最小空闲 batch slot 并标记 RUNNING 状态，加入运行中队列。

        cache_length 不在此设置：由调度步按 PrefillChunk 推进。
        """
        if len(states) > len(self._free_batch_slots):
            raise ValueError(f"可用 batch slot 不足：需要 {len(states)}，剩余 {len(self._free_batch_slots)}")
        for state in states:
            state.status = RequestStatus.RUNNING
            state.admitted_ms = clock_ms
            state.batch_slot_idx = heapq.heappop(self._free_batch_slots)
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
class PrefillChunk:
    """一次 Prefill 步为单个请求划定的 Prompt 区间 [start, end)（chunked prefill 的最小调度单元）。

    start 即本 chunk 之前已写入 KV Cache 的 Token 数（同时是 chunk 首 Token 的 position_ids）；
    非 chunked-prefill 策略下恒为整段 Prompt：start == 0 且 end == prompt_len。
    """

    state: RequestState
    start: int
    end: int

    @property
    def num_tokens(self) -> int:
        return self.end - self.start

    @property
    def is_last(self) -> bool:
        """本 chunk 写满 Prompt：本步 logits 末位即首个生成 Token。"""
        return self.end == self.state.request.prompt_len


@dataclass(frozen=True)
class SchedulerOutput:
    """单步迭代调度的决策产出：承载本步执行阶段、被选中请求快照与本步计费 Token 数。

    num_scheduled_tokens：PREFILL 为左 Padding 后实际送入模型的 padded chunk token 数
    （len(prefill_chunks) × max(chunk.num_tokens)）；DECODE 为 Decode 请求数（每请求 1 个 Token）。
    prefill_chunks：仅 PREFILL 非空，与 scheduled_requests 一一对应；chunked-prefill 下可只覆盖 Prompt 的一段。
    is_oversize_prefill：token-budget 下队头请求单独已超预算、被单独放行的 PREFILL（oversize prefill）。
    """

    phase: SchedulePhase
    scheduled_requests: tuple[RequestState, ...]
    num_scheduled_tokens: int
    is_oversize_prefill: bool = False
    prefill_chunks: tuple[PrefillChunk, ...] = ()


def _whole_prompt_chunks(states: Sequence[RequestState]) -> tuple[PrefillChunk, ...]:
    """每个请求整段 Prompt 一个 chunk（prefill-first / token-budget 不切分 Prompt）。"""
    return tuple(PrefillChunk(state, 0, state.request.prompt_len) for state in states)


def _select_within_budget(
    candidates: tuple[RequestState, ...], token_budget: int
) -> tuple[tuple[RequestState, ...], bool]:
    """按 FCFS 取 padded prompt token 数不超过 token_budget 的最长前缀；队头放不下即停，不向后跳选。

    无 Chunked Prefill 时长 Prompt 无法拆分：若队头请求单独已超预算，则让它独自执行（oversize prefill），
    避免其永久阻塞 FCFS 队列。返回 (入选前缀, 是否为 oversize prefill)。
    """
    num_selected = 0
    max_prompt_len = 0
    for state in candidates:
        next_max_prompt_len = max(max_prompt_len, state.request.prompt_len)
        if (num_selected + 1) * next_max_prompt_len > token_budget:
            break
        num_selected += 1
        max_prompt_len = next_max_prompt_len
    if num_selected == 0:
        return candidates[:1], True
    return candidates[:num_selected], False


class IterationScheduler:
    """迭代级请求调度器（对标 vLLM V1 Scheduler）：持有等待/运行队列与逻辑块管理器，驱动单步调度与状态闭环。

    入场决策全部收敛于此，依次取 WaitingQueue 就绪前缀的交集（均为前缀截断，可交换、可组合）：
    1. batching method 约束：continuous-batching 有 free batch slot 即可入场；fixed-batching / none 须等
       RunningQueue 完全排空（wave barrier）；
    2. batch slot 约束：至多 RunningQueue.num_free_batch_slots 个；
    3. KV 显存约束（仅 Paged）：KVCacheManager 空闲 physical block 可容纳的前缀（须整段 Prompt 放得下，
       并预留 chunked prefill 在途请求尚未写入的 Prompt 块；
       TODO(watermark): 后续在此支持预留 watermark 比例的空闲块防抢占抖动）；
    4. batching policy 约束：PrefillFirstConfig 不再截断；TokenBudgetConfig 按 token budget 截断，且 RunningQueue
       非空时禁止连续两步 PREFILL（decode protection），以限制 Prefill burst 对在途请求 ITL 的冲击；
       ChunkedPrefillConfig 不截断请求数而是切分 Prompt：在途 chunked prefill 请求与新入场请求共享 token budget
       （每请求 chunk 上限 token_budget // 候选数），同样启用 decode protection。

    chunked prefill 在 update_from_output 内按完成 chunk 推进 RequestState.cache_length（已写 KV Cache 的 Token 数）；
    Prompt 写满的当步产出首个 Token，之后与 prefill-first 一致逐步 Decode。P/D 仍为相位互斥。
    TODO(mixed-batching): 演进至统一 Token-Budget 范式（解除 P/D 相位互斥，单步先扣减在途 Decode token 预算，
    剩余预算将 chunk 与 Decode 混批执行，需 varlen Attention 支持）。
    TODO(preemption_mode): 当 Decode 步在途请求增长越过 block 边界且无空闲 physical block 时，
    触发 Victim 抢占（"recompute" 重算退回 WaitingQueue，或 "swap" 换出至 CPU Host Swap Pool）。
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        batching_method: BatchingMethod,
        kv_cache_manager: KVCacheManager | None = None,
    ) -> None:
        if scheduler_config.max_running_requests is None:
            raise ValueError("scheduler_config.max_running_requests 必须为已解析的正整数")
        self.scheduler_config = scheduler_config
        self.batching_method = batching_method
        self.waiting_queue = WaitingQueue()
        self.running_queue = RunningQueue(scheduler_config.max_running_requests)
        # Paged 路径的逻辑块管理器（由本调度器统一执行 allocate_slots / free）；Dense 路径为 None
        self.kv_cache_manager = kv_cache_manager
        self._last_phase: SchedulePhase | None = None

    @property
    def has_unfinished_requests(self) -> bool:
        """等待队列或运行队列中是否仍有未完成请求。"""
        return bool(self.waiting_queue or self.running_queue)

    def add_requests(self, states: Iterable[RequestState]) -> None:
        """将已到达请求推入等待队列。"""
        self.waiting_queue.extend(states)

    def schedule(self, clock_ms: float) -> SchedulerOutput | None:
        """生成本步调度计划并原子执行队列转移、batch slot 分配与 Paged 逻辑块分配；空闲时返回 None。"""
        output = self._decide()
        if output is None:
            return None
        if output.phase is SchedulePhase.PREFILL:
            num_new = sum(1 for chunk in output.prefill_chunks if chunk.state.status is RequestStatus.WAITING)
            self.running_queue.admit(self.waiting_queue.pop_batch(num_new), clock_ms)
            for chunk in output.prefill_chunks:
                self._allocate_slots(chunk.state, chunk.num_tokens)
        else:
            for state in output.scheduled_requests:
                self._allocate_slots(state, 1)
        self._last_phase = output.phase
        return output

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        token_ids: Sequence[int],
        clock_ms: float,
        eos_token_id: int | None = None,
        stop_on_eos: bool = True,
    ) -> list[RequestState]:
        """根据本步前向产出的 Token 推进请求状态，回收已完成请求的逻辑块与 batch slot。

        token_ids 与产出 Token 的请求一一对应：DECODE 为全部被调度请求；PREFILL 仅为本步写满 Prompt 的请求
        （chunked prefill 中间 chunk 不产出 Token）。
        """
        if scheduler_output.phase is SchedulePhase.DECODE:
            for state in scheduler_output.scheduled_requests:
                state.cache_length += 1
            producers = scheduler_output.scheduled_requests
        else:
            for chunk in scheduler_output.prefill_chunks:
                chunk.state.cache_length = chunk.end
            producers = tuple(chunk.state for chunk in scheduler_output.prefill_chunks if chunk.is_last)
        finished = self.running_queue.process_step_output(
            token_ids,
            clock_ms,
            eos_token_id,
            stop_on_eos,
            states=producers,
        )
        if finished:
            if self.kv_cache_manager is not None:
                for state in finished:
                    self.kv_cache_manager.free(state.request.request_idx)
            self.running_queue.retire(finished)
        return finished

    def _allocate_slots(self, state: RequestState, num_new_tokens: int) -> None:
        """Paged 路径为请求追加 num_new_tokens 个 Token 的缓存位；Dense 路径无逻辑块，无操作。"""
        if self.kv_cache_manager is None:
            return
        if not self.kv_cache_manager.allocate_slots(state.request.request_idx, num_new_tokens):
            raise RuntimeError("Paged KV Cache 物理块已耗尽（当前未实现抢占），请增大 num_blocks")

    def _num_reserved_blocks(self) -> int:
        """chunked prefill 在途请求尚需追加的 Prompt 块数（须在新入场前预留）。"""
        assert self.kv_cache_manager is not None
        block_size = self.kv_cache_manager.block_size
        return sum(
            num_blocks_for_tokens(s.request.prompt_len, block_size) - num_blocks_for_tokens(s.cache_length, block_size)
            for s in self.running_queue.prefilling_requests
        )

    def _admission_candidates(self) -> tuple[RequestState, ...]:
        """batching、batch slot 与 KV 显存约束下可入场的 WaitingQueue FCFS 前缀。"""
        is_continuous = self.batching_method is BatchingMethod.CONTINUOUS
        can_admit = self.running_queue.can_admit() if is_continuous else not self.running_queue
        if not can_admit:
            return ()
        candidates = self.waiting_queue.requests[: self.running_queue.num_free_batch_slots]
        if self.kv_cache_manager is not None:
            num_admissible = self.kv_cache_manager.num_admissible(
                [s.request.prompt_len for s in candidates], self._num_reserved_blocks()
            )
            candidates = candidates[:num_admissible]
        return candidates

    def _prefill_output(self, chunks: tuple[PrefillChunk, ...], is_oversize_prefill: bool = False) -> SchedulerOutput:
        return SchedulerOutput(
            phase=SchedulePhase.PREFILL,
            scheduled_requests=tuple(chunk.state for chunk in chunks),
            num_scheduled_tokens=len(chunks) * max(chunk.num_tokens for chunk in chunks),
            is_oversize_prefill=is_oversize_prefill,
            prefill_chunks=chunks,
        )

    def _decode_output(self) -> SchedulerOutput:
        decoding = self.running_queue.decoding_requests
        return SchedulerOutput(
            phase=SchedulePhase.DECODE,
            scheduled_requests=decoding,
            num_scheduled_tokens=len(decoding),
        )

    def _plan_chunks(self, token_budget: int) -> tuple[PrefillChunk, ...]:
        """在途 chunked prefill 请求与新入场请求按 FCFS 共享 token budget，各划一段 Prompt 区间。

        候选数为 n 时每请求 chunk 上限为 token_budget // n（尾 chunk 可更短），故 padded 成本
        n × max(chunk) ≤ token_budget 恒成立；token_budget ≥ max_running_requests 保证上限至少为 1。
        """
        candidates = (*self.running_queue.prefilling_requests, *self._admission_candidates())
        if not candidates:
            return ()
        chunk_cap = token_budget // len(candidates)
        return tuple(
            PrefillChunk(
                state,
                state.cache_length,
                state.cache_length + min(state.request.prompt_len - state.cache_length, chunk_cap),
            )
            for state in candidates
        )

    def _decide(self) -> SchedulerOutput | None:
        policy = self.scheduler_config.batching_policy
        if isinstance(policy, ChunkedPrefillConfig):
            if self.running_queue.decoding_requests and self._last_phase is SchedulePhase.PREFILL:
                return self._decode_output()
            chunks = self._plan_chunks(policy.token_budget)
            if chunks:
                return self._prefill_output(chunks)
            if self.running_queue.decoding_requests:
                return self._decode_output()
            return None

        admission_candidates = self._admission_candidates()
        if isinstance(policy, PrefillFirstConfig):
            if admission_candidates:
                return self._prefill_output(_whole_prompt_chunks(admission_candidates))
            if self.running_queue:
                return self._decode_output()
            return None

        if self.running_queue and self._last_phase is SchedulePhase.PREFILL:
            return self._decode_output()
        if admission_candidates:
            selected, is_oversize_prefill = _select_within_budget(admission_candidates, policy.token_budget)
            return self._prefill_output(_whole_prompt_chunks(selected), is_oversize_prefill)
        if self.running_queue:
            return self._decode_output()
        return None
