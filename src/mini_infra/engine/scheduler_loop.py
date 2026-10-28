"""核心批处理调度循环（Core Scheduler Loop）。

作为底座引擎核心，串联三类组件驱动“调度 → 执行 → 槽位更新”的迭代循环：
- 调度策略（入场、槽位分配、退场与 EOS/预算判定）：core/scheduler.py
- 模型执行（单步 Prefill / 槽位 Decode 前向与计时）：engine/runner.py (ModelRunner)
- KV Cache 槽位管理：memory/slot_cache.py (DenseSlotCache)

对外通过 run_scheduler 统一驱动两类批处理调度策略，共享同一套三级队列（ArrivalQueue -> WaitingQueue -> RunningQueue）
与 ModelRunner 执行协议，仅在准入时机与槽位绑定策略上区分：
- CONTINUOUS（连续批处理）：只要 RunningQueue 有空闲槽位即准入新请求，并将 Prefill Cache 拷入持久化槽位；
- FIXED（固定批次/波次基准）：仅当 RunningQueue 完全排空（当前波次全部完成）才准入下一波请求，零拷贝直接绑定波次 Cache。
在逻辑时间线上精确追踪模型与搬移耗时，输出 ScheduleResult（含 TTFT/TPOT/makespan 等事件流）。
"""

from collections.abc import Sequence

import torch

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
from mini_infra.engine.mode import BatchingMethod
from mini_infra.engine.runner import ModelRunner
from mini_infra.engine.types import (
    ScheduleEvent,
    SchedulePhase,
    ScheduleResult,
    summarize_schedule,
)
from mini_infra.memory.slot_cache import DenseSlotCache

__all__ = ["run_scheduler"]


def _retire(
    finished: Sequence[RequestState],
    running_queue: RunningQueue,
    slots: DenseSlotCache,
) -> None:
    """释放已结束请求的槽位（调度侧归还槽位号，Cache 侧清 mask）；若没有运行中请求则整块回收 Cache。"""
    for state in finished:
        slots.release(state)
    running_queue.retire(finished)
    if not running_queue:
        slots.reset()


@torch.inference_mode()
def _run_schedule(
    method: BatchingMethod,
    model: torch.nn.Module,
    request_specs: Sequence[RequestSpec],
    max_running_requests: int,
    eos_token_id: int | None,
    device: torch.device,
    pad_token_id: int = 0,
    stop_on_eos: bool = True,
) -> ScheduleResult:
    """采用三级队列（ArrivalQueue -> WaitingQueue -> RunningQueue）统一驱动 CONTINUOUS 与 FIXED 调度。"""
    is_continuous = method is BatchingMethod.CONTINUOUS
    states, arrival_queue, clock_ms = prepare_schedule(request_specs, max_running_requests)
    synchronize(device)
    baseline_mb = allocated_memory_mb(device)
    reset_peak_memory(device)
    runner = ModelRunner(model, device, pad_token_id)
    waiting_queue = WaitingQueue()
    running_queue = RunningQueue(max_running_requests)
    slots_cache = DenseSlotCache(max_running_requests, device)
    events: list[ScheduleEvent] = []

    while arrival_queue or waiting_queue or running_queue:
        # 1. 外部到达：把当前时刻已到达的请求从 arrival_queue 移入 waiting_queue
        arrived = arrival_queue.pop_arrived(clock_ms)
        waiting_queue.extend(arrived)

        # 2. 调度入场：Continuous 有空槽即入场；Fixed 必须等当前波次全部完成（running_queue 为空）才入场
        can_admit = running_queue.can_admit() if is_continuous else not running_queue
        if can_admit and waiting_queue:
            batch = waiting_queue.pop_batch(running_queue.num_free_slots)
            admitted = running_queue.admit(batch, clock_ms)
            token_ids, prefill_cache, model_ms = runner.prefill_requests(admitted)
            if is_continuous:
                cache_ms = slots_cache.admit(admitted, prefill_cache)
            else:
                slots_cache.bind_wave(admitted, prefill_cache)
                cache_ms = 0.0
            del prefill_cache
            total_ms = model_ms + cache_ms
            clock_ms += total_ms
            finished = running_queue.step(token_ids, clock_ms, eos_token_id, stop_on_eos, states=admitted)
            events.append(
                ScheduleEvent(
                    phase=SchedulePhase.PREFILL,
                    admitted=tuple(s.spec.request_index for s in admitted),
                    completed=tuple(s.spec.request_index for s in finished),
                    executed_batch_size=len(admitted),
                    num_active_requests=len(running_queue),
                    num_generated_tokens=len(admitted),
                    total_ms=total_ms,
                    model_ms=model_ms,
                    cache_management_ms=cache_ms,
                    live_cache_bytes=slots_cache.total_bytes,
                )
            )
            _retire(finished, running_queue, slots_cache)
            continue

        # 3. 运行中请求 Decode
        if running_queue:
            active_before = len(running_queue)
            decode_result = runner.decode_slots(slots_cache, running_queue.requests)
            for state in running_queue:
                state.cache_length += 1
            cache_ms = decode_result.prepare_ms if is_continuous else 0.0
            total_ms = decode_result.model_ms + cache_ms
            clock_ms += total_ms
            finished = running_queue.step(
                [decode_result.slot_tokens[s.slot_index] for s in running_queue],
                clock_ms,
                eos_token_id,
                stop_on_eos,
            )
            events.append(
                ScheduleEvent(
                    phase=SchedulePhase.DECODE,
                    admitted=(),
                    completed=tuple(s.spec.request_index for s in finished),
                    executed_batch_size=slots_cache.capacity,
                    num_active_requests=active_before,
                    num_generated_tokens=active_before,
                    total_ms=total_ms,
                    model_ms=decode_result.model_ms,
                    cache_management_ms=cache_ms,
                    live_cache_bytes=slots_cache.total_bytes,
                )
            )
            _retire(finished, running_queue, slots_cache)
            continue

        # 既无运行中请求、也无等待中请求：时钟直接跳到下一个到达时间（逻辑时间线，不真实 sleep）。
        clock_ms = arrival_queue.next_arrival_ms()

    return summarize_schedule(
        method,
        states,
        events,
        max_running_requests,
        *peak_memory_mb(device),
        baseline_mb=baseline_mb,
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
) -> ScheduleResult:
    """校验批处理策略并执行统一的队列调度循环（CONTINUOUS 或 FIXED）。

    Args:
        batching_method: 批处理策略（BatchingMethod.CONTINUOUS / FIXED 或字符串 "continuous" / "fixed"）。
        model: 因果语言模型。
        request_specs: 待调度的请求规格列表。
        max_running_requests: 调度队列最大并发数。
        eos_token_id: 结束 Token ID。
        device: 执行目标设备。
        pad_token_id: 填充 Token ID。
        stop_on_eos: 命中 EOS 时是否提前停止生成。

    Returns:
        ScheduleResult: 调度阶段事件、指标统计与各请求结果。
    """
    if isinstance(batching_method, str):
        normalized = batching_method.strip().lower()
        if normalized == "fixed":
            method = BatchingMethod.FIXED
        elif normalized == "continuous":
            method = BatchingMethod.CONTINUOUS
        else:
            method = BatchingMethod.from_str(normalized)
    else:
        method = batching_method

    if method in (BatchingMethod.CONTINUOUS, BatchingMethod.FIXED):
        return _run_schedule(
            method=method,
            model=model,
            request_specs=request_specs,
            max_running_requests=max_running_requests,
            eos_token_id=eos_token_id,
            device=device,
            pad_token_id=pad_token_id,
            stop_on_eos=stop_on_eos,
        )
    raise ValueError(f"不支持的调度批处理策略: {method}，仅支持 CONTINUOUS 或 FIXED")
