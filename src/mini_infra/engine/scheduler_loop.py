"""核心批处理调度循环（Core Scheduler Loop）。

作为底座引擎核心，串联三类组件驱动“调度 → 执行 → 槽位更新”的迭代循环：
- 调度策略（入场、槽位分配、退场）：core/scheduler.py
- 模型执行（单步前向与计时）：engine/runner.py
- KV Cache 槽位数据搬移：memory/slot_cache.py (DenseSlotCache)

对外通过 run_scheduler 统一分发两类内部批处理调度执行引擎，供离线生成入口（entrypoints/offline.py）与在线评测服务共享：
- _run_continuous_batching：连续批处理迭代调度（请求随到随走，空槽即时补入）；
- _run_fixed_batching：固定批次（波次）基准调度（整批完成前不补入新请求）。
在逻辑时间线上精确追踪模型与搬移耗时，输出 ScheduleResult（含 TTFT/TPOT/makespan 等事件流）。
"""

from collections.abc import Sequence

import torch

from mini_infra.core.request import (
    RequestSpec,
    RequestState,
    RequestStatus,
)
from mini_infra.core.scheduler import (
    RunningQueue,
    WaitingQueue,
    prepare_schedule,
)
from mini_infra.core.types import (
    ScheduleEvent,
    SchedulePhase,
    ScheduleResult,
    summarize_schedule,
)
from mini_infra.device import peak_memory_mb, reset_peak_memory
from mini_infra.engine.batch_state import BatchGenerationState
from mini_infra.engine.mode import BatchingMethod
from mini_infra.engine.preprocessing import left_pad_sequences
from mini_infra.engine.runner import ModelRunner
from mini_infra.memory.kv_cache import KVCache
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
def _run_continuous_batching(
    model: torch.nn.Module,
    request_specs: Sequence[RequestSpec],
    max_running_requests: int,
    eos_token_id: int | None,
    device: torch.device,
    pad_token_id: int = 0,
    stop_on_eos: bool = True,
) -> ScheduleResult:
    """FCFS Continuous Batching：采用三级队列（ArrivalQueue -> WaitingQueue -> RunningQueue）驱动调度。"""
    states, arrival_queue, clock_ms = prepare_schedule(request_specs, max_running_requests)
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

        # 2. 调度入场：有空闲槽位且有等待就绪请求时，入场并执行 Prefill
        if running_queue.can_admit() and waiting_queue:
            batch = waiting_queue.pop_batch(running_queue.num_free_slots)
            admitted = running_queue.admit(batch, clock_ms)
            token_ids, prefill_cache, model_ms = runner.prefill_requests(admitted)
            cache_ms = slots_cache.admit(admitted, prefill_cache)
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
                    active_requests=len(running_queue),
                    generated_tokens=len(admitted),
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
            total_ms = decode_result.prepare_ms + decode_result.model_ms
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
                    executed_batch_size=max_running_requests,
                    active_requests=active_before,
                    generated_tokens=active_before,
                    total_ms=total_ms,
                    model_ms=decode_result.model_ms,
                    cache_management_ms=decode_result.prepare_ms,
                    live_cache_bytes=slots_cache.total_bytes,
                )
            )
            _retire(finished, running_queue, slots_cache)
            continue

        # 既无运行中请求、也无等待中请求：时钟直接跳到下一个到达时间（逻辑时间线，不真实 sleep）。
        clock_ms = arrival_queue.next_arrival_ms()

    return summarize_schedule(BatchingMethod.CONTINUOUS, states, events, max_running_requests, *peak_memory_mb(device))


@torch.inference_mode()
def _run_fixed_batching(
    model: torch.nn.Module,
    request_specs: Sequence[RequestSpec],
    max_running_requests: int,
    eos_token_id: int | None,
    device: torch.device,
    pad_token_id: int = 0,
    stop_on_eos: bool = True,
) -> ScheduleResult:
    """固定批次（波次）baseline：每波最多 max_running_requests 个已到达请求，整波结束前不补入新请求。"""
    states, arrival_queue, clock_ms = prepare_schedule(request_specs, max_running_requests)
    reset_peak_memory(device)
    runner = ModelRunner(model, device, pad_token_id)
    waiting_queue = WaitingQueue()
    events: list[ScheduleEvent] = []

    while arrival_queue or waiting_queue:
        arrived = arrival_queue.pop_arrived(clock_ms)
        waiting_queue.extend(arrived)

        if not waiting_queue:
            clock_ms = arrival_queue.next_arrival_ms()
            continue

        wave = waiting_queue.pop_batch(max_running_requests)
        for s in wave:
            s.status = RequestStatus.RUNNING
            s.admitted_ms = clock_ms

        input_ids, history_mask, position_ids, prompt_lengths = left_pad_sequences(
            [s.spec.token_ids for s in wave], pad_token_id, device
        )
        budgets = torch.tensor([s.spec.max_new_tokens for s in wave], dtype=torch.long, device=device)
        state = BatchGenerationState.create(
            input_ids, history_mask, position_ids, prompt_lengths, budgets, device, pad_token_id, use_cache=True
        )
        cache = KVCache()
        for step in range(state.max_steps):
            counts_before = [len(output) for output in state.outputs]
            next_tokens, total_ms = runner.step_batch(state, cache)
            clock_ms += total_ms

            all_finished = state.update_step(next_tokens, step, eos_token_id, stop_on_eos)
            stepped = [i for i, output in enumerate(state.outputs) if len(output) > counts_before[i]]
            finished = [
                wave[i]
                for i in stepped
                if wave[i].append_token(state.outputs[i][-1], clock_ms, eos_token_id, stop_on_eos)
            ]
            for s in finished:
                s.status = RequestStatus.FINISHED
            events.append(
                ScheduleEvent(
                    phase=SchedulePhase.PREFILL if step == 0 else SchedulePhase.DECODE,
                    admitted=tuple(s.spec.request_index for s in wave) if step == 0 else (),
                    completed=tuple(s.spec.request_index for s in finished),
                    executed_batch_size=len(wave),
                    active_requests=len(stepped),
                    generated_tokens=len(stepped),
                    total_ms=total_ms,
                    model_ms=total_ms,
                    cache_management_ms=0.0,
                    live_cache_bytes=cache.total_bytes(),
                )
            )
            if all_finished:
                break
            state.prepare_next_step(next_tokens, use_cache=True)
        del cache

    return summarize_schedule(BatchingMethod.FIXED, states, events, max_running_requests, *peak_memory_mb(device))


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
    """根据批处理策略分发到具体的调度执行函数（CONTINUOUS 或 FIXED）。

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

    if method is BatchingMethod.CONTINUOUS:
        return _run_continuous_batching(
            model=model,
            request_specs=request_specs,
            max_running_requests=max_running_requests,
            eos_token_id=eos_token_id,
            device=device,
            pad_token_id=pad_token_id,
            stop_on_eos=stop_on_eos,
        )
    if method is BatchingMethod.FIXED:
        return _run_fixed_batching(
            model=model,
            request_specs=request_specs,
            max_running_requests=max_running_requests,
            eos_token_id=eos_token_id,
            device=device,
            pad_token_id=pad_token_id,
            stop_on_eos=stop_on_eos,
        )
    raise ValueError(f"不支持的调度批处理策略: {method}，仅支持 CONTINUOUS 或 FIXED")
