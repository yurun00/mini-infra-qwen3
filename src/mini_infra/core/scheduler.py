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

与上述维度正交的 Paged 存储特性 prefix caching（`PagedCacheConfig.is_prefix_caching_enabled`）：新请求的 Prefill
区间从 prefix cache 命中长度起算，K/V 落盘后再登记新写满的 cached block（见 `core/block_manager.py`）。
"""

import heapq
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from mini_infra.core.block_manager import NULL_KV_CACHE_MANAGER, BaseKVCacheManager
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
    非 chunked-prefill 策略下恒覆盖 Prompt 剩余部分：start == prefix_cache_hit_tokens（prefix cache 命中长度，
    未启用 prefix caching 时为 0）且 end == prompt_len。
    """

    state: RequestState
    start: int
    end: int

    @property
    def num_tokens(self) -> int:
        return self.end - self.start

    @property
    def is_first(self) -> bool:
        """本 chunk 为请求入场步的首个 chunk（起点恰为 prefix cache 命中长度）。"""
        return self.start == self.state.prefix_cache_hit_tokens

    @property
    def is_last(self) -> bool:
        """本 chunk 写满 Prompt：本步 logits 末位即首个生成 Token。"""
        return self.end == self.state.request.prompt_len


@dataclass(frozen=True)
class SchedulerOutput:
    """单步迭代调度的决策产出：承载本步执行阶段、被选中请求快照与本步计费 Token 数。

    num_scheduled_tokens：PREFILL 为左 Padding 后实际送入模型的 padded chunk token 数
    （len(prefill_chunks) × max(chunk.num_tokens)，prefix cache 命中部分不计费）；DECODE 为 Decode 请求数。
    prefill_chunks：仅 PREFILL 非空，与 scheduled_requests 一一对应；chunked-prefill 下可只覆盖 Prompt 的一段。
    is_oversize_prefill：token-budget 下队头请求单独已超预算、被单独放行的 PREFILL（oversize prefill）。
    """

    phase: SchedulePhase
    scheduled_requests: tuple[RequestState, ...]
    num_scheduled_tokens: int
    is_oversize_prefill: bool = False
    prefill_chunks: tuple[PrefillChunk, ...] = ()


def _select_within_budget(chunks: tuple[PrefillChunk, ...], token_budget: int) -> tuple[tuple[PrefillChunk, ...], bool]:
    """按 FCFS 取 padded chunk token 数不超过 token_budget 的最长前缀；队头放不下即停，不向后跳选。

    无 Chunked Prefill 时长 Prompt 无法拆分：若队头请求单独已超预算，则让它独自执行（oversize prefill），
    避免其永久阻塞 FCFS 队列。返回 (入选前缀, 是否为 oversize prefill)。
    """
    num_selected = 0
    max_chunk_len = 0
    for chunk in chunks:
        next_max_chunk_len = max(max_chunk_len, chunk.num_tokens)
        if (num_selected + 1) * next_max_chunk_len > token_budget:
            break
        num_selected += 1
        max_chunk_len = next_max_chunk_len
    if num_selected == 0:
        return chunks[:1], True
    return chunks[:num_selected], False


class IterationScheduler:
    """迭代级请求调度器（对标 vLLM V1 Scheduler）：持有等待/运行队列与逻辑块管理器，驱动单步调度与状态闭环。

    入场决策全部收敛于此，依次取 WaitingQueue 就绪前缀的交集（均为前缀截断，可交换、可组合）：
    1. batching method 约束：continuous-batching 有 free batch slot 即可入场；fixed-batching / none 须等
       RunningQueue 完全排空（wave barrier）；
    2. batch slot 约束：至多 RunningQueue.num_free_batch_slots 个；
    3. KV 显存约束（仅 Paged）：KVCacheManager 空闲 physical block 可容纳的前缀（须整段 Prompt 放得下，
       并预留 chunked prefill 在途请求尚未写入的 Prompt 块；prefix cache 命中的块无需新分配；
       TODO(watermark): 后续在此支持预留 watermark 比例的空闲块防抢占抖动）；
    4. batching policy 约束：PrefillFirstConfig 不再截断；TokenBudgetConfig 按 token budget 截断，且 RunningQueue
       非空时禁止连续两步 PREFILL（decode protection），以限制 Prefill burst 对在途请求 ITL 的冲击；
       ChunkedPrefillConfig 不截断请求数而是切分 Prompt：在途 chunked prefill 请求与新入场请求共享 token budget
       （每请求 chunk 上限 token_budget // 候选数），同样启用 decode protection。

    prefix caching（仅 Paged）：新请求的 Prefill 区间从 prefix cache 命中长度起算（命中部分不计入
    token budget），入场时引用命中的 cached block；每步 K/V 落盘后（update_from_output）再把新写满的块登记为
    cached block，保证同批请求只会命中 K/V 已写入的块。

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
        kv_cache_manager: BaseKVCacheManager = NULL_KV_CACHE_MANAGER,
    ) -> None:
        if scheduler_config.max_running_requests is None:
            raise ValueError("scheduler_config.max_running_requests 必须为已解析的正整数")
        self.scheduler_config = scheduler_config
        self.batching_method = batching_method
        self.waiting_queue = WaitingQueue()
        self.running_queue = RunningQueue(scheduler_config.max_running_requests)
        # 逻辑块管理器（Paged 为 KVCacheManager，Dense 默认为无操作的 NullKVCacheManager）
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
        """生成本步调度计划并原子执行队列转移、batch slot 分配与逻辑块分配；空闲时返回 None。"""
        output = self._decide()
        if output is None:
            return None
        if output.phase is SchedulePhase.PREFILL:
            num_new = sum(1 for chunk in output.prefill_chunks if chunk.state.status is RequestStatus.WAITING)
            admitted = self.running_queue.admit(self.waiting_queue.pop_batch(num_new), clock_ms)
            # 先为全部新入场请求引用 prefix cache 命中块，再统一追加新块：
            # 新块分配可能从 free block queue 队首淘汰空闲 cached block，先引用可保证已规划的命中不被淘汰
            for state in admitted:
                hit_tokens = self.kv_cache_manager.allocate_computed_blocks(state.request)
                state.prefix_cache_hit_tokens = hit_tokens
                state.cache_length = hit_tokens
            allocations = [(chunk.state, chunk.num_tokens) for chunk in output.prefill_chunks]
        else:
            allocations = [(state, 1) for state in output.scheduled_requests]
        for state, num_new_tokens in allocations:
            if not self.kv_cache_manager.allocate_slots(state.request.request_idx, num_new_tokens):
                raise RuntimeError("Paged KV Cache 物理块已耗尽（当前未实现抢占），请增大 num_blocks")
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
        """根据本步前向产出的 Token 推进请求状态，登记新写满的 cached block，回收已完成请求的逻辑块与 batch slot。

        token_ids 与产出 Token 的请求一一对应：DECODE 为全部被调度请求；PREFILL 仅为本步写满 Prompt 的请求
        （chunked prefill 中间 chunk 不产出 Token）。
        """
        if scheduler_output.phase is SchedulePhase.DECODE:
            for state in scheduler_output.scheduled_requests:
                state.cache_length += 1
                self.kv_cache_manager.cache_blocks(state)
            producers = scheduler_output.scheduled_requests
        else:
            for chunk in scheduler_output.prefill_chunks:
                chunk.state.cache_length = chunk.end
                self.kv_cache_manager.cache_blocks(chunk.state)
            producers = tuple(chunk.state for chunk in scheduler_output.prefill_chunks if chunk.is_last)
        finished = self.running_queue.process_step_output(
            token_ids,
            clock_ms,
            eos_token_id,
            stop_on_eos,
            states=producers,
        )
        if finished:
            for state in finished:
                self.kv_cache_manager.free(state.request.request_idx)
            self.running_queue.retire(finished)
        return finished

    def _num_computed_tokens(self, state: RequestState) -> int:
        """请求本步 Prefill 区间的起点：在途请求为已写入 KV Cache 的 Token 数；新请求为 prefix cache 命中 Token 数。"""
        if state.status is RequestStatus.RUNNING:
            return state.cache_length
        return self.kv_cache_manager.num_computed_tokens(state.request)

    def _admission_candidates(self) -> tuple[RequestState, ...]:
        """batching、batch slot 与 KV 显存约束下可入场的 WaitingQueue FCFS 前缀。"""
        is_continuous = self.batching_method is BatchingMethod.CONTINUOUS
        can_admit = self.running_queue.can_admit() if is_continuous else not self.running_queue
        if not can_admit:
            return ()
        candidates = self.waiting_queue.requests[: self.running_queue.num_free_batch_slots]
        num_admissible = self.kv_cache_manager.num_admissible(
            [s.request for s in candidates], self.running_queue.prefilling_requests
        )
        return candidates[:num_admissible]

    def _remaining_prompt_chunks(self, states: Sequence[RequestState]) -> tuple[PrefillChunk, ...]:
        """每个请求一个 chunk 覆盖 Prompt 剩余部分（prefill-first / token-budget 不切分 Prompt，跳过前缀命中）。"""
        return tuple(
            PrefillChunk(state, self._num_computed_tokens(state), state.request.prompt_len) for state in states
        )

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
        新请求的区间从 prefix cache 命中长度起算（命中长度 ≤ prompt_len - 1，至少推进 1 个 Token）。
        """
        candidates = (*self.running_queue.prefilling_requests, *self._admission_candidates())
        if not candidates:
            return ()
        chunk_cap = token_budget // len(candidates)
        chunks = []
        for state in candidates:
            start = self._num_computed_tokens(state)
            chunks.append(PrefillChunk(state, start, start + min(state.request.prompt_len - start, chunk_cap)))
        return tuple(chunks)

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
                return self._prefill_output(self._remaining_prompt_chunks(admission_candidates))
            if self.running_queue:
                return self._decode_output()
            return None

        if self.running_queue and self._last_phase is SchedulePhase.PREFILL:
            return self._decode_output()
        if admission_candidates:
            selected, is_oversize_prefill = _select_within_budget(
                self._remaining_prompt_chunks(admission_candidates), policy.token_budget
            )
            return self._prefill_output(selected, is_oversize_prefill)
        if self.running_queue:
            return self._decode_output()
        return None
