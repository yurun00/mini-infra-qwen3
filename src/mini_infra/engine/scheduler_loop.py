"""核心批处理调度循环（Core Scheduler Loop）。

作为底座引擎核心，串联三类组件驱动“调度 → 执行 → batch slot / block table 更新”的迭代循环：
- 调度后端（入场、batch slot / block table 分配、退场与 EOS/预算判定）：engine/{dense,paged}_schedule_backend.py
- 模型执行（单步 Prefill / Decode 前向与计时）：worker/slot_runner.py 与 worker/paged_runner.py
- KV Cache 物理显存管理：worker/dense_slot_cache.py (DenseSlotCache) 与 worker/paged_kv_cache.py (PagedKVCachePool)

后端接口与单步产出契约见 engine/schedule_backend.py（ScheduleBackend / ScheduleStepOutput）；
Dense / Paged 两个具体实现分别位于 engine/dense_schedule_backend.py 与 engine/paged_schedule_backend.py。

本模块只保留调度主循环（_run_schedule_loop）与统一入口（run_scheduler），对外通过 run_scheduler 驱动三类批处理调度策略：
- CONTINUOUS（连续批处理 Dense 路径）：有空闲 batch slot 即准入，并将 Prefill Cache 拷入持久化 batch slot；
- FIXED（固定批次/波次基准）：仅当 RunningQueue 完全排空时准入下一波请求，零拷贝直接接管波次 Cache；
- PAGED（连续批处理 Paged 路径）：以 physical block pool 与 block table 映射驱动，按需按块分配，消除最长序列预留碎片。
"""

import time
from collections.abc import Sequence

import torch

from mini_infra.core import (
    KVCacheConfig,
    NoCacheConfig,
    PagedCacheConfig,
    Request,
    RunningQueue,
    WaitingQueue,
    prepare_schedule,
)
from mini_infra.core.types import BatchingMethod
from mini_infra.device import peak_memory_mb, synchronize
from mini_infra.engine.dense_schedule_backend import DenseScheduleBackend
from mini_infra.engine.metrics import ScheduleEvent, SchedulePhase, summarize_schedule
from mini_infra.engine.paged_schedule_backend import PagedScheduleBackend
from mini_infra.engine.results import ScheduleResult
from mini_infra.engine.schedule_backend import ScheduleBackend

__all__ = ["run_scheduler"]


@torch.inference_mode()
def _run_schedule_loop(
    backend: ScheduleBackend,
    requests: Sequence[Request],
    max_running_requests: int,
    eos_token_id: int | None,
    stop_on_eos: bool = True,
) -> ScheduleResult:
    """采用三级队列（ArrivalQueue -> WaitingQueue -> RunningQueue）驱动统一的调度主循环。"""
    device = backend.device
    states, arrival_queue, clock_ms = prepare_schedule(requests, max_running_requests)
    waiting_queue = WaitingQueue()
    running_queue = RunningQueue(max_running_requests)
    events: list[ScheduleEvent] = []
    wall_start = time.perf_counter()

    while arrival_queue or waiting_queue or running_queue:
        # 1. 外部到达：把当前时刻已到达的请求从 arrival_queue 移入 waiting_queue
        arrived = arrival_queue.pop_arrived(clock_ms)
        waiting_queue.extend(arrived)

        # 2. 调度入场：Prefill 阶段
        num_admit = backend.determine_admission(waiting_queue, running_queue)
        if num_admit:
            admitted = running_queue.admit(waiting_queue.pop_batch(num_admit), clock_ms)
            step_output = backend.execute_prefill(admitted)
            step_ms = step_output.model_ms + step_output.cache_ms
            clock_ms += step_ms
            finished = running_queue.process_step_output(
                step_output.next_tokens, clock_ms, eos_token_id, stop_on_eos, states=admitted
            )
            events.append(
                ScheduleEvent(
                    phase=SchedulePhase.PREFILL,
                    admitted_request_ids=tuple(s.request.request_idx for s in admitted),
                    finished_request_ids=tuple(s.request.request_idx for s in finished),
                    physical_batch_size=step_output.physical_batch_size,
                    num_active_requests=len(running_queue),
                    num_generated_tokens=len(admitted),
                    step_ms=step_ms,
                    model_ms=step_output.model_ms,
                    cache_management_ms=step_output.cache_ms,
                    live_kv_cache_bytes=step_output.live_kv_cache_bytes,
                    kv_cache_pool_bytes=step_output.kv_cache_pool_bytes,
                    cache_realign_moved_bytes=step_output.cache_realign_moved_bytes,
                    accessed_kv_cache_tokens=step_output.accessed_kv_cache_tokens,
                )
            )
            backend.retire(finished, running_queue)
            continue

        # 3. 运行中请求 Decode
        if running_queue:
            active_before = len(running_queue)
            step_output = backend.execute_decode(running_queue)
            for state in running_queue:
                state.cache_length += 1
            step_ms = step_output.model_ms + step_output.cache_ms
            clock_ms += step_ms
            finished = running_queue.process_step_output(step_output.next_tokens, clock_ms, eos_token_id, stop_on_eos)
            events.append(
                ScheduleEvent(
                    phase=SchedulePhase.DECODE,
                    admitted_request_ids=(),
                    finished_request_ids=tuple(s.request.request_idx for s in finished),
                    physical_batch_size=step_output.physical_batch_size,
                    num_active_requests=active_before,
                    num_generated_tokens=active_before,
                    step_ms=step_ms,
                    model_ms=step_output.model_ms,
                    cache_management_ms=step_output.cache_ms,
                    live_kv_cache_bytes=step_output.live_kv_cache_bytes,
                    kv_cache_pool_bytes=step_output.kv_cache_pool_bytes,
                    cache_realign_moved_bytes=step_output.cache_realign_moved_bytes,
                    accessed_kv_cache_tokens=step_output.accessed_kv_cache_tokens,
                )
            )
            backend.retire(finished, running_queue)
            continue

        # 4. 既无运行中请求、也无等待中请求：时钟直接跳到下一个到达时间（逻辑时间线，不真实 sleep）
        clock_ms = arrival_queue.next_arrival_ms()

    synchronize(device)
    e2e_wall_ms = (time.perf_counter() - wall_start) * 1000.0

    return summarize_schedule(
        backend.batching_method,
        states,
        events,
        max_running_requests,
        *peak_memory_mb(device),
        baseline_mb=backend.baseline_mb,
        e2e_wall_ms=e2e_wall_ms,
        is_cuda=device.type == "cuda",
        kv_cache_config=backend.kv_cache_config,
        paged_cache_profile=backend.collect_paged_cache_profile(),
    )


@torch.inference_mode()
def run_scheduler(
    batching_method: BatchingMethod,
    model: torch.nn.Module,
    requests: Sequence[Request],
    max_running_requests: int,
    eos_token_id: int | None,
    *,
    kv_cache_config: KVCacheConfig,
    pad_token_id: int = 0,
    stop_on_eos: bool = True,
) -> ScheduleResult:
    """校验批处理策略并执行统一的队列调度循环（CONTINUOUS、FIXED 或 NONE 串行单槽调度）。

    Args:
        batching_method: 批处理策略（BatchingMethod.CONTINUOUS / FIXED / NONE）。
        model: 因果语言模型。
        requests: 待调度的请求列表。
        max_running_requests: 调度队列最大并发数（NONE 模式下自动约束为 1）。
        eos_token_id: 结束 Token ID。
        kv_cache_config: 运行时 KV Cache 配置（DenseCacheConfig 或 PagedCacheConfig；NoCacheConfig 不支持）。
        pad_token_id: 填充 Token ID。
        stop_on_eos: 命中 EOS 时是否提前停止生成。

    Returns:
        ScheduleResult: 调度阶段事件、指标统计与各请求结果。
    """
    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")

    if isinstance(kv_cache_config, NoCacheConfig):
        raise ValueError("scheduler 不支持无 KV Cache，请改用 DenseCacheConfig 或 PagedCacheConfig")

    backend: ScheduleBackend
    if isinstance(kv_cache_config, PagedCacheConfig):
        if batching_method not in (BatchingMethod.CONTINUOUS, BatchingMethod.FIXED):
            raise ValueError(
                f"Paged KV Cache 仅支持 {BatchingMethod.CONTINUOUS.value} 或 {BatchingMethod.FIXED.value}，"
                f"实际为 {batching_method.value}"
            )
        backend = PagedScheduleBackend(
            model=model,
            requests=requests,
            max_running_requests=max_running_requests,
            paged_cache_config=kv_cache_config,
            batching_method=batching_method,
            pad_token_id=pad_token_id,
        )
        effective_max_running_requests = max_running_requests
    else:
        effective_max_running_requests = (
            1 if (batching_method is BatchingMethod.NONE and max_running_requests >= 1) else max_running_requests
        )
        backend = DenseScheduleBackend(
            model=model,
            max_running_requests=effective_max_running_requests,
            batching_method=batching_method,
            pad_token_id=pad_token_id,
        )

    return _run_schedule_loop(
        backend=backend,
        requests=requests,
        max_running_requests=effective_max_running_requests,
        eos_token_id=eos_token_id,
        stop_on_eos=stop_on_eos,
    )
