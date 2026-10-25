"""文本生成主引擎与调度流程。"""

import time
from collections.abc import Sequence
from typing import Any, NamedTuple

import torch

from mini_infra.generation.mode import BatchingMethod, FeatureFlag, Mode
from mini_infra.generation.preprocessing import (
    _normalize_budgets,
    _normalize_token_sequences,
    left_pad_sequences,
)
from mini_infra.generation.result import (
    GenerationResult,
    GenerationTrace,
    SequenceOutput,
    aggregate_generation_results,
    build_memory_profile,
    compute_step_aggregated_metrics,
)
from mini_infra.generation.state import BatchGenerationState
from mini_infra.profiling import TensorBoardOptions, profile_scope
from mini_infra.qwen3_model import KVCache, cache_size_bytes


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_prompt_sequences(
    input_ids: torch.Tensor | Sequence[Any],
    device: torch.device | None,
) -> tuple[list[torch.Tensor], torch.device, bool]:
    """解析多样化的 Prompt 输入形态，规约为一维序列列表并推断目标设备。

    Returns:
        tuple[list[torch.Tensor], torch.device, bool]:
            - token_sequences (list[torch.Tensor]): 规约后的各请求一维 Token ID 张量列表。
            - device (torch.device): 推断或确认后的计算设备（CUDA 或 CPU）。
            - has_uniform_prompt_len (bool): 输入各 Prompt 长度是否天然等长（True 则后续可跳过左 Padding）。
    """
    has_uniform_prompt_len = isinstance(input_ids, torch.Tensor)
    if has_uniform_prompt_len:
        tensor = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
        token_sequences = list(tensor)
        if device is None:
            device = tensor.device
    else:
        token_sequences = _normalize_token_sequences(input_ids)
        if device is None:
            device = token_sequences[0].device if token_sequences[0].device.type != "cpu" else torch.device("cpu")
    return token_sequences, device, has_uniform_prompt_len


def _init_batch_tensors(
    input_ids: torch.Tensor | Sequence[Any],
    token_sequences: list[torch.Tensor],
    has_uniform_prompt_len: bool,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构建批处理初始 Tensor（天然等长跳过 padding，不等长采用左 Padding）。"""
    if has_uniform_prompt_len:
        input_ids_tensor = (input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)).to(
            device=device, dtype=torch.long
        )
        batch_size, padded_prompt_length = input_ids_tensor.shape
        history_mask = torch.ones((batch_size, padded_prompt_length), dtype=torch.bool, device=device)
        position_ids = (
            torch.arange(padded_prompt_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )
        prompt_lengths = torch.full((batch_size,), padded_prompt_length, dtype=torch.long, device=device)
        return input_ids_tensor, history_mask, position_ids, prompt_lengths

    return left_pad_sequences(token_sequences, pad_token_id, device)


class StepForwardResult(NamedTuple):
    """单步前向计时输出对象：

    - 属性访问：.next_token, .elapsed_ms, .event_pair, .last_logits
    - 兼容元组解包：(next_token, elapsed_ms, event_pair)（last_logits 仅在 return_logits=True 时填充）
    """

    next_token: torch.Tensor
    elapsed_ms: float = 0.0
    event_pair: tuple[Any, Any] | None = None
    last_logits: torch.Tensor | None = None


def _timed_forward(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    device: torch.device | None = None,
    kv_cache: KVCache | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    return_logits: bool = False,
) -> StepForwardResult:
    """单步前向推理并计时。

    attention_mask / position_ids 用于批处理（左 Padding）；return_logits=True 时额外返回最后位置的 logits。
    """
    if device is None:
        device = model_input.device

    event_pair = None
    elapsed_ms = 0.0

    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        t0 = time.perf_counter()

    logits = model(model_input, kv_cache=kv_cache, attention_mask=attention_mask, position_ids=position_ids)
    last_logits = logits[:, -1, :]
    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

    if device.type == "cuda":
        end.record()
        event_pair = (start, end)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000

    result_logits = last_logits if return_logits else None
    del logits
    return StepForwardResult(
        next_token=next_token, elapsed_ms=elapsed_ms, event_pair=event_pair, last_logits=result_logits
    )


def _resolve_step_latencies(
    step_event_pairs: list[tuple[Any, Any]], step_times_ms: list[float], is_cuda: bool
) -> list[float]:
    """CUDA 下使用 Event 硬件测时，非 CUDA 下使用 CPU 墙钟。"""
    if is_cuda and step_event_pairs:
        return [s.elapsed_time(e) for s, e in step_event_pairs]
    return step_times_ms


def _run_decode_loop(
    model: torch.nn.Module,
    state: BatchGenerationState,
    use_cache: bool,
    kv_cache: KVCache | None,
    eos_token_id: int | None,
    stop_on_eos: bool,
    capture_logits: bool,
    device: torch.device,
    is_cuda: bool,
) -> tuple[list[float], list[torch.Tensor] | None, int, float]:
    """执行 Decode 迭代循环，返回 (step_latencies_ms, logits_trace, memory_after_prefill_bytes, e2e_wall_ms)。"""
    step_event_pairs: list[tuple[Any, Any]] = []
    step_times_ms: list[float] = []
    logits_trace = [] if capture_logits else None
    memory_after_prefill_bytes = 0

    wall_start = time.perf_counter()

    for step in range(state.max_steps):
        model_input, attention_mask, position_ids = state.get_model_inputs(use_cache)
        step_out = _timed_forward(
            model=model,
            model_input=model_input,
            device=device,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_logits=capture_logits,
        )
        next_tokens = step_out.next_token.squeeze(1)

        if step_out.event_pair is not None:
            step_event_pairs.append(step_out.event_pair)
        else:
            step_times_ms.append(step_out.elapsed_ms)

        if step == 0 and is_cuda:
            memory_after_prefill_bytes = torch.cuda.memory_allocated(device)

        if capture_logits and step_out.last_logits is not None:
            logits_trace.append(step_out.last_logits.float().cpu())

        all_finished = state.update_step(next_tokens, step, eos_token_id, stop_on_eos)
        if all_finished:
            break

        state.prepare_next_step(next_tokens, use_cache)

    synchronize(device)
    e2e_wall_ms = (time.perf_counter() - wall_start) * 1000
    step_latencies_ms = _resolve_step_latencies(step_event_pairs, step_times_ms, is_cuda)
    return step_latencies_ms, logits_trace, memory_after_prefill_bytes, e2e_wall_ms


@torch.inference_mode()
def generate(
    mode: Mode,
    model: torch.nn.Module,
    input_ids: torch.Tensor | Sequence[Any],
    max_new_tokens: int | Sequence[int],
    eos_token_id: int | None = None,
    device: torch.device | None = None,
    stop_on_eos: bool = True,
    pad_token_id: int = 0,
    capture_logits: bool = False,
    profile: bool = False,
    top_kernels: int = 10,
    tensorboard_options: TensorBoardOptions | None = None,
) -> GenerationResult:
    """统一生成入口；mode 为 Mode 结构体对象。

    不同 mode 对应不同的 feature 开关，但生成的 output token 完全一致。
    支持 profile=True 采集底层算子与模块级耗时明细（保存在 result.op_breakdown 中）。
    支持指定 tensorboard_options（TensorBoardOptions 结构体）将详细 trace 导出为 TensorBoard 格式。
    """
    # 1. 规约输入 Prompt、预算与设备
    token_sequences, device, has_uniform_len = _prepare_prompt_sequences(input_ids, device)
    budgets = _normalize_budgets(max_new_tokens, len(token_sequences), device)

    # 2. 串行 baseline 分发
    if mode.batching_method == BatchingMethod.NONE and len(token_sequences) > 1:
        return _dispatch_serial_baseline(
            mode,
            model,
            token_sequences,
            budgets,
            eos_token_id,
            device,
            stop_on_eos,
            pad_token_id,
            capture_logits=capture_logits,
            profile=profile,
            top_kernels=top_kernels,
            tensorboard_options=tensorboard_options,
        )

    # 3. 构造批 Tensor 与状态机
    input_tensor, history_mask, pos_ids, prompt_lens = _init_batch_tensors(
        input_ids, token_sequences, has_uniform_len, pad_token_id, device
    )
    use_cache = mode.is_enabled(FeatureFlag.KV_CACHE)
    kv_cache = KVCache() if use_cache else None

    state = BatchGenerationState.create(
        input_tensor, history_mask, pos_ids, prompt_lens, budgets, device, pad_token_id, use_cache
    )

    # 4. 显存打点与执行循环
    is_cuda = device.type == "cuda"
    synchronize(device)
    baseline_allocated_bytes = torch.cuda.memory_allocated(device) if is_cuda else 0
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    if profile:
        with profile_scope(model, device=device, tensorboard_options=tensorboard_options) as prof_collector:
            step_latencies, logits_trace, mem_prefill, e2e_wall_ms = _run_decode_loop(
                model, state, use_cache, kv_cache, eos_token_id, stop_on_eos, capture_logits, device, is_cuda
            )
        op_latency_breakdown = prof_collector.get_latency_breakdown(top=top_kernels)
    else:
        step_latencies, logits_trace, mem_prefill, e2e_wall_ms = _run_decode_loop(
            model, state, use_cache, kv_cache, eos_token_id, stop_on_eos, capture_logits, device, is_cuda
        )
        op_latency_breakdown = None

    valid_tokens = sum(len(o) for o in state.outputs)
    e2e_model_ms = sum(step_latencies)
    batch_info = state.compute_batch_metrics(step_latencies, kv_cache)

    sequences = [
        SequenceOutput(
            request_index=idx,
            token_ids=state.outputs[idx],
            prompt_len=int(state.prompt_lengths[idx].item()),
            output_len=len(state.outputs[idx]),
            finish_reason=state.finish_reasons[idx],
        )
        for idx in range(state.batch_size)
    ]
    trace = GenerationTrace(
        step_latencies_ms=step_latencies,
        active_tokens_per_step=state.active_counts_per_step,
        logits_trace=logits_trace,
    )
    metrics = compute_step_aggregated_metrics(
        step_latencies_ms=step_latencies,
        e2e_wall_ms=e2e_wall_ms,
        e2e_model_ms=e2e_model_ms,
        total_tokens=valid_tokens,
        batch_size=state.batch_size,
        is_cuda=is_cuda,
    )
    memory = build_memory_profile(
        baseline_allocated_bytes=baseline_allocated_bytes,
        memory_after_prefill_bytes=mem_prefill,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if is_cuda else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if is_cuda else 0,
        cache_bytes=cache_size_bytes(kv_cache) if kv_cache is not None else 0,
    )

    return GenerationResult(
        sequences=sequences,
        metrics=metrics,
        memory=memory,
        trace=trace,
        batch_size=state.batch_size,
        op_breakdown=op_latency_breakdown,
        batch_info=batch_info,
    )


def _dispatch_serial_baseline(
    mode: Mode,
    model: torch.nn.Module,
    token_sequences: list[torch.Tensor],
    budgets: torch.Tensor,
    eos_token_id: int | None,
    device: torch.device,
    stop_on_eos: bool,
    pad_token_id: int,
    capture_logits: bool = False,
    profile: bool = False,
    top_kernels: int = 10,
    tensorboard_options: TensorBoardOptions | None = None,
) -> GenerationResult:
    """按顺序逐个执行请求（串行 baseline 模式）。"""
    runs = [
        generate(
            mode,
            model,
            seq.to(device=device, dtype=torch.long).unsqueeze(0),
            budget,
            eos_token_id=eos_token_id,
            device=device,
            stop_on_eos=stop_on_eos,
            pad_token_id=pad_token_id,
            capture_logits=capture_logits,
            profile=profile,
            top_kernels=top_kernels,
            tensorboard_options=tensorboard_options,
        )
        for seq, budget in zip(token_sequences, budgets.tolist())
    ]
    return aggregate_generation_results(runs)
