"""核心批处理调度循环（Core Scheduler Loop）。

作为底座引擎核心，串联三类组件驱动“调度 → 执行 → 槽位/块表更新”的迭代循环：
- 调度策略（入场、槽位/块表分配、退场与 EOS/预算判定）：core/scheduler.py 与 core/block_manager.py
- 模型执行（单步 Prefill / Decode 前向与计时）：worker/slot_runner.py 与 worker/paged_runner.py
- KV Cache 物理显存管理：worker/dense_slot_cache.py (DenseSlotCache) 与 worker/paged_kv_cache.py (PagedKVCachePool)

对外通过 run_scheduler 统一驱动三类批处理调度策略：
- CONTINUOUS（连续批处理 Dense 路径）：有空闲 batch slot 即准入，并将 Prefill Cache 拷入持久化 batch slot；
- FIXED（固定批次/波次基准）：仅当 RunningQueue 完全排空时准入下一波请求，零拷贝直接接管波次 Cache；
- PAGED（连续批处理 Paged 路径）：以物理块池与 Block Table 映射驱动，按需按块分配，消除最长序列预留碎片。

通过统一的 ExecutionStrategy 抽象策略基类与 ScheduleStepOutput 单步产出契约，实现调度主循环单份收敛。
"""

import abc
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mini_infra.core.block_manager import (
    KVCacheManager,
    PagedCacheConfig,
    num_blocks_for_tokens,
    required_pool_blocks,
)
from mini_infra.core.request import (
    RequestSpec,
    RequestState,
)
from mini_infra.core.scheduler import (
    RunningQueue,
    WaitingQueue,
    prepare_schedule,
)
from mini_infra.device import allocated_memory_mb, peak_memory_mb, reset_peak_memory, synchronize
from mini_infra.engine.metrics import (
    ScheduleEvent,
    SchedulePhase,
    summarize_schedule,
)
from mini_infra.engine.mode import BatchingMethod
from mini_infra.engine.results import ScheduleResult
from mini_infra.worker import DenseSlotCache, PagedKVCachePool, PagedRunner, SlotRunner

__all__ = [
    "DenseExecutionStrategy",
    "ExecutionStrategy",
    "PagedExecutionStrategy",
    "ScheduleStepOutput",
    "run_scheduler",
]


@dataclass(frozen=True)
class ScheduleStepOutput:
    """策略执行一次 Prefill 或 Decode 调度步后的度量与结果数据契约。"""

    next_tokens: list[int]
    model_ms: float
    cache_ms: float
    physical_batch_size: int
    live_kv_cache_bytes: int


class ExecutionStrategy(abc.ABC):
    """抽象调度单步执行策略基类（Strategy Pattern：承载通用硬件初始化与单步执行契约）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_running_requests: int,
        method: BatchingMethod,
    ) -> None:
        self.model = model
        self.device = device
        self.max_running_requests = max_running_requests
        self.method = method
        synchronize(device)
        self.baseline_mb = allocated_memory_mb(device)
        reset_peak_memory(device)

    @abc.abstractmethod
    def determine_admission(self, waiting_queue: WaitingQueue, running_queue: RunningQueue) -> int:
        """评估当前时刻允许从 waiting_queue 准入到 running_queue 的请求数量。"""
        ...

    @abc.abstractmethod
    def execute_prefill(self, admitted: list[RequestState]) -> ScheduleStepOutput:
        """执行一批新请求的 Prefill 前向，将 KV 写入缓存并返回首个 Token 与耗时。"""
        ...

    @abc.abstractmethod
    def execute_decode(self, running_queue: RunningQueue) -> ScheduleStepOutput:
        """为当前运行中请求组装输入，执行一步增量 Decode 并写回缓存。"""
        ...

    @abc.abstractmethod
    def retire(self, finished: Sequence[RequestState], running_queue: RunningQueue) -> None:
        """回收已结束请求的槽位或物理块资源，并在空闲时执行重置。"""
        ...


class DenseExecutionStrategy(ExecutionStrategy):
    """连续批处理与固定批次的稠密 batch slot Cache 执行策略（基于 DenseSlotCache 与 SlotRunner）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_running_requests: int,
        method: BatchingMethod,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__(model, device, max_running_requests, method)
        self.is_continuous = method is BatchingMethod.CONTINUOUS
        self.runner = SlotRunner(model, device, pad_token_id)
        self.dense_slot_cache = DenseSlotCache(max_running_requests, device)

    def determine_admission(self, waiting_queue: WaitingQueue, running_queue: RunningQueue) -> int:
        can_admit = running_queue.can_admit() if self.is_continuous else not running_queue
        if can_admit and waiting_queue:
            return min(len(waiting_queue), running_queue.num_free_batch_slots)
        return 0

    def execute_prefill(self, admitted: list[RequestState]) -> ScheduleStepOutput:
        prefill_out = self.runner.run_prefill_step([s.spec.token_ids for s in admitted])
        batch_slot_indices = [s.batch_slot_idx for s in admitted if s.batch_slot_idx is not None]
        cache_lengths = [s.cache_length for s in admitted]
        if self.is_continuous:
            synchronize(self.device)
            cache_start = time.perf_counter()
            self.dense_slot_cache.absorb_prefill(batch_slot_indices, cache_lengths, prefill_out.prefill_cache)
            synchronize(self.device)
            cache_ms = (time.perf_counter() - cache_start) * 1000
        else:
            self.dense_slot_cache.adopt_prefill(batch_slot_indices, cache_lengths, prefill_out.prefill_cache)
            cache_ms = 0.0
        return ScheduleStepOutput(
            next_tokens=prefill_out.next_tokens,
            model_ms=prefill_out.model_ms,
            cache_ms=cache_ms,
            physical_batch_size=len(admitted),
            live_kv_cache_bytes=self.dense_slot_cache.total_bytes,
        )

    def execute_decode(self, running_queue: RunningQueue) -> ScheduleStepOutput:
        batch_slot_indices = [s.batch_slot_idx for s in running_queue if s.batch_slot_idx is not None]
        last_tokens = [s.generated[-1] for s in running_queue]
        positions = [s.cache_length for s in running_queue]
        synchronize(self.device)
        prepare_start = time.perf_counter()
        input_ids, position_ids, attention_mask = self.runner.prepare_decode_inputs(
            self.dense_slot_cache,
            running_batch_slot_indices=batch_slot_indices,
            last_tokens=last_tokens,
            positions=positions,
        )
        synchronize(self.device)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000
        decode_out = self.runner.run_decode_step(input_ids, attention_mask, position_ids, self.dense_slot_cache)
        cache_ms = prepare_ms if self.is_continuous else 0.0
        next_tokens = [decode_out.batch_slot_tokens[s.batch_slot_idx] for s in running_queue]
        return ScheduleStepOutput(
            next_tokens=next_tokens,
            model_ms=decode_out.model_ms,
            cache_ms=cache_ms,
            physical_batch_size=self.dense_slot_cache.capacity,
            live_kv_cache_bytes=self.dense_slot_cache.total_bytes,
        )

    def retire(self, finished: Sequence[RequestState], running_queue: RunningQueue) -> None:
        for state in finished:
            if state.batch_slot_idx is not None:
                self.dense_slot_cache.release(state.batch_slot_idx)
        running_queue.retire(finished)
        if not running_queue:
            self.dense_slot_cache.reset()


class PagedExecutionStrategy(ExecutionStrategy):
    """连续批处理的分页 KV Cache 执行策略（基于 PagedRunner、PagedKVCachePool 与 KVCacheManager）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        request_specs: Sequence[RequestSpec],
        max_running_requests: int,
        paged_cache_config: PagedCacheConfig,
        pad_token_id: int = 0,
    ) -> None:
        self.block_size = paged_cache_config.block_size
        num_blocks = paged_cache_config.num_blocks or required_pool_blocks(
            request_specs, max_running_requests, self.block_size
        )
        largest_request_blocks = max(
            num_blocks_for_tokens(spec.prompt_len + spec.max_new_tokens - 1, self.block_size) for spec in request_specs
        )
        if largest_request_blocks > num_blocks - 1:
            raise ValueError(
                f"num_blocks={num_blocks} 无法容纳单个请求所需的 {largest_request_blocks} "
                "个物理块（另需 1 个 null block）"
            )
        super().__init__(model, device, max_running_requests, BatchingMethod.CONTINUOUS)
        config = model.config  # type: ignore[attr-defined]
        self.runner = PagedRunner(model, device, pad_token_id)
        self.pool = PagedKVCachePool(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=self.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=next(model.parameters()).dtype,
            device=device,
        )
        self.kv_cache_manager = KVCacheManager(self.block_size, num_blocks)

    def determine_admission(self, waiting_queue: WaitingQueue, running_queue: RunningQueue) -> int:
        if running_queue.can_admit() and waiting_queue:
            candidates = waiting_queue.requests[: running_queue.num_free_batch_slots]
            return self.kv_cache_manager.num_admissible([s.spec.prompt_len for s in candidates])
        return 0

    def execute_prefill(self, admitted: list[RequestState]) -> ScheduleStepOutput:
        prefill_out = self.runner.run_prefill_step([s.spec.token_ids for s in admitted])
        cache_lengths = [s.cache_length for s in admitted]
        synchronize(self.device)
        cache_start = time.perf_counter()
        for state in admitted:
            self.kv_cache_manager.allocate_slots(state.spec.request_idx, state.cache_length)
        self.runner.store_prefill(
            self.pool,
            prefill_out.prefill_cache,
            [self.kv_cache_manager.block_table(s.spec.request_idx) for s in admitted],
            cache_lengths,
        )
        synchronize(self.device)
        cache_ms = (time.perf_counter() - cache_start) * 1000
        next_tokens = prefill_out.next_tokens
        model_ms = prefill_out.model_ms
        del prefill_out
        return ScheduleStepOutput(
            next_tokens=next_tokens,
            model_ms=model_ms,
            cache_ms=cache_ms,
            physical_batch_size=len(admitted),
            live_kv_cache_bytes=self.kv_cache_manager.num_used_blocks * self.pool.bytes_per_block,
        )

    def execute_decode(self, running_queue: RunningQueue) -> ScheduleStepOutput:
        active_before = len(running_queue)
        synchronize(self.device)
        prepare_start = time.perf_counter()
        for state in running_queue:
            if not self.kv_cache_manager.allocate_slots(state.spec.request_idx, 1):
                raise RuntimeError("Paged KV Cache 物理块已耗尽（当前未实现抢占），请增大 num_blocks")
        input_ids, position_ids, metadata = self.runner.prepare_decode_inputs(
            [self.kv_cache_manager.block_table(s.spec.request_idx) for s in running_queue],
            last_tokens=[s.generated[-1] for s in running_queue],
            positions=[s.cache_length for s in running_queue],
            block_size=self.block_size,
        )
        synchronize(self.device)
        cache_ms = (time.perf_counter() - prepare_start) * 1000
        decode_out = self.runner.run_decode_step(input_ids, position_ids, metadata, self.pool)
        return ScheduleStepOutput(
            next_tokens=decode_out.next_tokens,
            model_ms=decode_out.model_ms,
            cache_ms=cache_ms,
            physical_batch_size=active_before,
            live_kv_cache_bytes=self.kv_cache_manager.num_used_blocks * self.pool.bytes_per_block,
        )

    def retire(self, finished: Sequence[RequestState], running_queue: RunningQueue) -> None:
        for state in finished:
            self.kv_cache_manager.free(state.spec.request_idx)
        running_queue.retire(finished)


@torch.inference_mode()
def _run_schedule_loop(
    strategy: ExecutionStrategy,
    request_specs: Sequence[RequestSpec],
    max_running_requests: int,
    eos_token_id: int | None,
    device: torch.device,
    stop_on_eos: bool = True,
) -> ScheduleResult:
    """采用三级队列（ArrivalQueue -> WaitingQueue -> RunningQueue）驱动统一的调度主循环。"""
    states, arrival_queue, clock_ms = prepare_schedule(request_specs, max_running_requests)
    waiting_queue = WaitingQueue()
    running_queue = RunningQueue(max_running_requests)
    events: list[ScheduleEvent] = []
    wall_start = time.perf_counter()

    while arrival_queue or waiting_queue or running_queue:
        # 1. 外部到达：把当前时刻已到达的请求从 arrival_queue 移入 waiting_queue
        arrived = arrival_queue.pop_arrived(clock_ms)
        waiting_queue.extend(arrived)

        # 2. 调度入场：Prefill 阶段
        num_admit = strategy.determine_admission(waiting_queue, running_queue)
        if num_admit:
            admitted = running_queue.admit(waiting_queue.pop_batch(num_admit), clock_ms)
            step_output = strategy.execute_prefill(admitted)
            step_ms = step_output.model_ms + step_output.cache_ms
            clock_ms += step_ms
            finished = running_queue.process_step_output(
                step_output.next_tokens, clock_ms, eos_token_id, stop_on_eos, states=admitted
            )
            events.append(
                ScheduleEvent(
                    phase=SchedulePhase.PREFILL,
                    admitted_request_ids=tuple(s.spec.request_idx for s in admitted),
                    finished_request_ids=tuple(s.spec.request_idx for s in finished),
                    physical_batch_size=step_output.physical_batch_size,
                    num_active_requests=len(running_queue),
                    num_generated_tokens=len(admitted),
                    step_ms=step_ms,
                    model_ms=step_output.model_ms,
                    cache_management_ms=step_output.cache_ms,
                    live_kv_cache_bytes=step_output.live_kv_cache_bytes,
                )
            )
            strategy.retire(finished, running_queue)
            continue

        # 3. 运行中请求 Decode
        if running_queue:
            active_before = len(running_queue)
            step_output = strategy.execute_decode(running_queue)
            for state in running_queue:
                state.cache_length += 1
            step_ms = step_output.model_ms + step_output.cache_ms
            clock_ms += step_ms
            finished = running_queue.process_step_output(step_output.next_tokens, clock_ms, eos_token_id, stop_on_eos)
            events.append(
                ScheduleEvent(
                    phase=SchedulePhase.DECODE,
                    admitted_request_ids=(),
                    finished_request_ids=tuple(s.spec.request_idx for s in finished),
                    physical_batch_size=step_output.physical_batch_size,
                    num_active_requests=active_before,
                    num_generated_tokens=active_before,
                    step_ms=step_ms,
                    model_ms=step_output.model_ms,
                    cache_management_ms=step_output.cache_ms,
                    live_kv_cache_bytes=step_output.live_kv_cache_bytes,
                )
            )
            strategy.retire(finished, running_queue)
            continue

        # 4. 既无运行中请求、也无等待中请求：时钟直接跳到下一个到达时间（逻辑时间线，不真实 sleep）
        clock_ms = arrival_queue.next_arrival_ms()

    synchronize(device)
    e2e_wall_ms = (time.perf_counter() - wall_start) * 1000.0

    return summarize_schedule(
        strategy.method,
        states,
        events,
        max_running_requests,
        *peak_memory_mb(device),
        baseline_mb=strategy.baseline_mb,
        e2e_wall_ms=e2e_wall_ms,
        is_cuda=device.type == "cuda",
        is_paged_kv_cache=isinstance(strategy, PagedExecutionStrategy),
    )


@torch.inference_mode()
def run_scheduler(
    batching_method: str | BatchingMethod,
    model: torch.nn.Module,
    request_specs: Sequence[RequestSpec],
    max_running_requests: int,
    eos_token_id: int | None,
    device: torch.device,
    pad_token_id: int = 0,
    stop_on_eos: bool = True,
    paged_cache_config: PagedCacheConfig | None = None,
) -> ScheduleResult:
    """校验批处理策略并执行统一的队列调度循环（CONTINUOUS、FIXED 或 NONE 串行单槽调度）。

    Args:
        batching_method: 批处理策略（BatchingMethod.CONTINUOUS / FIXED / NONE 或对应字符串）。
        model: 因果语言模型。
        request_specs: 待调度的请求规格列表。
        max_running_requests: 调度队列最大并发数（NONE 模式下自动约束为 1）。
        eos_token_id: 结束 Token ID。
        device: 执行目标设备。
        pad_token_id: 填充 Token ID。
        stop_on_eos: 命中 EOS 时是否提前停止生成。
        paged_cache_config: 非 None 时以 Paged KV Cache（物理块池 + 块表）取代 Dense 槽位 Cache，仅支持 CONTINUOUS。

    Returns:
        ScheduleResult: 调度阶段事件、指标统计与各请求结果。
    """
    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")

    if isinstance(batching_method, str):
        normalized = batching_method.strip().lower()
        if normalized == "fixed":
            method = BatchingMethod.FIXED
        elif normalized == "continuous":
            method = BatchingMethod.CONTINUOUS
        elif normalized == "none":
            method = BatchingMethod.NONE
        else:
            method = BatchingMethod.from_str(normalized)
    else:
        method = BatchingMethod.from_str(batching_method)

    if paged_cache_config is not None:
        if method is not BatchingMethod.CONTINUOUS:
            raise ValueError(f"Paged KV Cache 仅支持 {BatchingMethod.CONTINUOUS.value}，实际为 {method.value}")
        strategy: ExecutionStrategy = PagedExecutionStrategy(
            model=model,
            device=device,
            request_specs=request_specs,
            max_running_requests=max_running_requests,
            paged_cache_config=paged_cache_config,
            pad_token_id=pad_token_id,
        )
        effective_max_running_requests = max_running_requests
    else:
        effective_max_running_requests = (
            1 if (method is BatchingMethod.NONE and max_running_requests >= 1) else max_running_requests
        )
        strategy = DenseExecutionStrategy(
            model=model,
            device=device,
            max_running_requests=effective_max_running_requests,
            method=method,
            pad_token_id=pad_token_id,
        )

    return _run_schedule_loop(
        strategy=strategy,
        request_specs=request_specs,
        max_running_requests=effective_max_running_requests,
        eos_token_id=eos_token_id,
        device=device,
        stop_on_eos=stop_on_eos,
    )
