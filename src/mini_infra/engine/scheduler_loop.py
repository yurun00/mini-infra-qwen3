"""核心批处理调度循环（Core Scheduler Loop）。

作为底座引擎核心，串联三类组件驱动“调度 → 执行 → batch slot / block table 更新”的迭代循环：
- 调度后端（入场、batch slot / block table 分配、退场与 EOS/预算判定）：engine/{dense,paged}_schedule_backend.py
- 模型执行（单步 Prefill / Decode 前向与计时）：worker/slot_runner.py 与 worker/paged_runner.py
- KV Cache 物理显存管理：worker/dense_slot_cache.py (DenseSlotCache) 与 worker/paged_block_cache.py (PagedBlockCache)

后端接口与单步产出契约见 engine/schedule_backend.py（ScheduleBackend / ScheduleStepOutput）；
Dense / Paged 两个具体实现分别位于 engine/dense_schedule_backend.py 与 engine/paged_schedule_backend.py。

本模块只保留调度主循环（_run_schedule_loop）与统一入口（run_scheduler），对外通过 run_scheduler 驱动三类批处理调度策略：
- CONTINUOUS（连续批处理 Dense 路径）：有空闲 batch slot 即准入，并将 Prefill Cache 拷入持久化 batch slot；
- FIXED（固定批次/波次基准）：仅当 RunningQueue 完全排空时准入下一波请求，零拷贝直接接管波次 Cache；
- PAGED（连续批处理 Paged 路径）：以 physical block pool 与 block table 映射驱动，按需按块分配，消除最长序列预留碎片。

每步「Prefill 还是 Decode、哪些请求入场」由 core.IterationScheduler 按 EngineConfig.scheduler_config 决策
（prefill-first / token-budget，含 batching、batch slot 与 KV 显存准入），后端只负责执行与资源回收。
"""

import time
from collections.abc import Sequence

import torch

from mini_infra.core import (
    IterationScheduler,
    NoCacheConfig,
    PagedCacheConfig,
    Request,
    SchedulePhase,
    SchedulerConfig,
    prepare_schedule,
)
from mini_infra.core.types import BatchingMethod
from mini_infra.device import peak_memory_mb, synchronize
from mini_infra.engine.config import EngineConfig, validate_engine_config
from mini_infra.engine.dense_schedule_backend import DenseScheduleBackend
from mini_infra.engine.metrics import ScheduleEvent, summarize_schedule
from mini_infra.engine.paged_schedule_backend import PagedScheduleBackend
from mini_infra.engine.results import ScheduleResult
from mini_infra.engine.schedule_backend import ScheduleBackend

__all__ = ["run_scheduler"]


@torch.inference_mode()
def _run_schedule_loop(
    backend: ScheduleBackend,
    requests: Sequence[Request],
    max_running_requests: int,
    scheduler_config: SchedulerConfig,
    eos_token_id: int | None,
    stop_on_eos: bool = True,
    should_capture_first_token_logits: bool = False,
) -> ScheduleResult:
    """采用三级队列（ArrivalQueue -> IterationScheduler[WaitingQueue -> RunningQueue]）驱动统一的调度主循环。"""
    device = backend.device
    states, arrival_queue, clock_ms = prepare_schedule(requests, max_running_requests)
    scheduler = IterationScheduler(
        scheduler_config.with_max_running_requests(max_running_requests),
        backend.batching_method,
        backend.kv_cache_manager,
    )
    events: list[ScheduleEvent] = []
    first_token_logits_by_request_idx: dict[int, torch.Tensor] | None = (
        {} if should_capture_first_token_logits else None
    )
    wall_start = time.perf_counter()
    pending_scheduler_ms = 0.0

    while arrival_queue or scheduler.has_unfinished_requests:
        # 1. 外部到达：把当前时刻已到达的请求从 arrival_queue 移入 scheduler 等待队列
        sched_start = time.perf_counter()
        scheduler.add_requests(arrival_queue.pop_arrived(clock_ms))

        # 2. 迭代调度决策与资源分配：batching / batch slot / 逻辑 KV 块 / 策略约束全部由 IterationScheduler 统一完成
        output = scheduler.schedule(clock_ms)
        if output is None:
            # 既无运行中请求、也无等待中请求：时钟直接跳到下一个到达时间（逻辑时间线，不真实 sleep）
            clock_ms = arrival_queue.next_arrival_ms()
            pending_scheduler_ms += (time.perf_counter() - sched_start) * 1000.0
            continue
        pending_scheduler_ms += (time.perf_counter() - sched_start) * 1000.0

        # 3. 后端执行本步前向（Prefill 或 Decode）
        num_active_requests = len(scheduler.running_queue)
        if output.phase is SchedulePhase.PREFILL:
            step_output = backend.execute_prefill(
                output.prefill_chunks,
                should_capture_first_token_logits=should_capture_first_token_logits,
            )
            admitted_request_ids = tuple(
                c.state.request.request_idx for c in output.prefill_chunks if c.is_first
            )
            if first_token_logits_by_request_idx is not None:
                assert step_output.first_token_logits is not None
                completed_prefill_ids = (
                    c.state.request.request_idx
                    for c in output.prefill_chunks
                    if c.is_last
                )
                for request_idx, req_logits in zip(
                    completed_prefill_ids, step_output.first_token_logits
                ):
                    first_token_logits_by_request_idx[request_idx] = req_logits
        else:
            step_output = backend.execute_decode(output.scheduled_requests)
            admitted_request_ids = ()

        # 4. 推进逻辑时钟并写回本步产出（Token 追加、EOS/预算判定、逻辑块与 batch slot 释放）
        step_ms = step_output.model_ms + step_output.cache_ms
        clock_ms += step_ms
        update_start = time.perf_counter()
        finished = scheduler.update_from_output(
            output, step_output.next_tokens, clock_ms, eos_token_id, stop_on_eos
        )
        scheduler_ms = (
            pending_scheduler_ms + (time.perf_counter() - update_start) * 1000.0
        )
        pending_scheduler_ms = 0.0
        backend.retire(finished, is_idle=not scheduler.running_queue)

        events.append(
            ScheduleEvent(
                phase=output.phase,
                admitted_request_ids=admitted_request_ids,
                finished_request_ids=tuple(s.request.request_idx for s in finished),
                physical_batch_size=step_output.physical_batch_size,
                num_active_requests=num_active_requests,
                num_generated_tokens=len(step_output.next_tokens),
                num_scheduled_tokens=output.num_scheduled_tokens,
                step_ms=step_ms,
                model_ms=step_output.model_ms,
                cache_management_ms=step_output.cache_ms,
                scheduler_ms=scheduler_ms,
                live_kv_cache_bytes=step_output.live_kv_cache_bytes,
                kv_cache_pool_bytes=step_output.kv_cache_pool_bytes,
                cache_realign_moved_bytes=step_output.cache_realign_moved_bytes,
                accessed_kv_cache_tokens=step_output.accessed_kv_cache_tokens,
                is_oversize_prefill=output.is_oversize_prefill,
            )
        )

    synchronize(device)
    e2e_wall_ms = (time.perf_counter() - wall_start) * 1000.0
    first_token_logits = (
        tuple(
            first_token_logits_by_request_idx[state.request.request_idx]
            for state in states
        )
        if first_token_logits_by_request_idx is not None
        else None
    )

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
        first_token_logits=first_token_logits,
    )


@torch.inference_mode()
def run_scheduler(
    config: EngineConfig,
    model: torch.nn.Module,
    requests: Sequence[Request],
    eos_token_id: int | None = None,
    *,
    pad_token_id: int = 0,
    stop_on_eos: bool = True,
    should_capture_first_token_logits: bool = False,
) -> ScheduleResult:
    """校验批处理策略并执行统一的队列调度循环（CONTINUOUS、FIXED 或 NONE 串行单槽调度）。

    Args:
        config: 引擎运行配置（包含 kv_cache_config、batching_method 与 scheduler_config）。
        model: 因果语言模型。
        requests: 待调度的请求列表。
        eos_token_id: 结束 Token ID。
        pad_token_id: 填充 Token ID。
        stop_on_eos: 命中 EOS 时是否提前停止生成。
        should_capture_first_token_logits: 是否按请求采集首 Token Logits（供数值等价性校验使用）。

    Returns:
        ScheduleResult: 调度阶段事件、指标统计与各请求结果。
    """
    if not requests:
        raise ValueError("requests 不能为空")
    raw_mrr = config.scheduler_config.max_running_requests
    max_running_requests = raw_mrr if raw_mrr is not None else len(requests)

    batching_method = config.batching_method
    kv_cache_config = config.kv_cache_config
    if isinstance(kv_cache_config, NoCacheConfig):
        raise ValueError(
            "scheduler 不支持无 KV Cache，请改用 DenseCacheConfig 或 PagedCacheConfig"
        )
    # 按实际生效的并发上限校验配置：max_running_requests=None 时已推导为请求总数
    validate_engine_config(config.with_max_running_requests(max_running_requests))

    backend: ScheduleBackend
    if isinstance(kv_cache_config, PagedCacheConfig):
        if batching_method not in (
            BatchingMethod.CONTINUOUS,
            BatchingMethod.FIXED,
        ):
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
            1
            if (batching_method is BatchingMethod.NONE and max_running_requests >= 1)
            else max_running_requests
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
        scheduler_config=config.scheduler_config,
        eos_token_id=eos_token_id,
        stop_on_eos=stop_on_eos,
        should_capture_first_token_logits=should_capture_first_token_logits,
    )
