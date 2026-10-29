"""单用例基准测试（run_case）：一次调用只执行正好一个 Case。

三个核心参数遵循 numpy 式广播规则：
    --batch-size      标量 int（可选）。若未指定，自动由变长 prompt_lens / max_new_tokens
                      长度推断；若均为单值则默认为 1。
    --prompt-lens     int 列表。1 项表示全 Batch 等长；多项表示逐请求长度。
    --max-new-tokens  int 列表。1 项表示全 Batch 等长；多项表示逐请求预算。

用法示例：
    # 单用例快速测试（batch=1, prompt=32, max_new_tokens=16）
    mini-infra bench lockstep case --tag dev --prompt-lens 32 --max-new-tokens 16

    # 批处理等长测试（batch=4, prompt=128, max_new_tokens=32）
    mini-infra bench lockstep case --tag dev --batch-size 4 --prompt-lens 128 \
        --max-new-tokens 32 --batching-method fixed-batching

    # 不等长变长测试（batch 自动由 prompt_lens 推断为 4）
    mini-infra bench lockstep case --tag dev --prompt-lens 16 32 64 128 \
        --max-new-tokens 32 --batching-method fixed-batching
"""

from collections.abc import Sequence

import torch

from mini_infra.bench.common import ProfileMode, efficiency
from mini_infra.bench.lockstep.types import (
    DistributionStats,
    LockstepCaseResult,
    WorkloadSpec,
    summarize_samples,
)
from mini_infra.bench.workloads import make_prompt
from mini_infra.core.types import MemoryProfile, StepAggregatedMetrics
from mini_infra.engine import GenerationOptions, Mode, generate
from mini_infra.profiling import TensorBoardOptions, merge_op_latency_breakdowns


def resolve_case_specs(
    prompt_lens: Sequence[int],
    max_new_tokens: Sequence[int],
    batch_size: int | None = None,
) -> tuple[int, int | tuple[int, ...], int | tuple[int, ...]]:
    """根据 numpy 式广播规则，从输入参数解析出 (effective_batch_size, prompt_len, max_new_tokens)。"""
    if not prompt_lens or any(p < 1 for p in prompt_lens):
        raise ValueError("prompt_lens 必须非空且每项大于 0")
    if not max_new_tokens or any(o < 2 for o in max_new_tokens):
        raise ValueError("max_new_tokens 必须非空且每项至少为 2，才能测量 Decode")

    len_p = len(prompt_lens)
    len_o = len(max_new_tokens)

    if batch_size is None:
        if len_p > 1 and len_o > 1:
            if len_p != len_o:
                raise ValueError(f"prompt_lens 长度 ({len_p}) 与 max_new_tokens 长度 ({len_o}) 不匹配")
            target_bs = len_p
        elif len_p > 1:
            target_bs = len_p
        elif len_o > 1:
            target_bs = len_o
        else:
            target_bs = 1
    else:
        if batch_size < 1:
            raise ValueError(f"batch_size 必须为正整数，得到 {batch_size}")
        if len_p != 1 and len_p != batch_size:
            raise ValueError(f"prompt_lens 数量 ({len_p}) 既不为 1，也不等于指定的 batch_size ({batch_size})")
        if len_o != 1 and len_o != batch_size:
            raise ValueError(f"max_new_tokens 数量 ({len_o}) 既不为 1，也不等于指定的 batch_size ({batch_size})")
        target_bs = batch_size

    resolved_prompt_len: int | tuple[int, ...] = prompt_lens[0] if len_p == 1 else tuple(prompt_lens)
    resolved_max_new_tokens: int | tuple[int, ...] = max_new_tokens[0] if len_o == 1 else tuple(max_new_tokens)
    return target_bs, resolved_prompt_len, resolved_max_new_tokens


def run_case(
    model: torch.nn.Module,
    mode: Mode,
    prompt_len: int | Sequence[int],
    max_new_tokens: int | Sequence[int],
    batch_size: int,
    warmup: int,
    repeats: int,
    vocab: int,
    device: torch.device,
    cost: dict | None = None,
    peaks: dict | None = None,
    profile_mode: ProfileMode | None = None,
    tensorboard_options: TensorBoardOptions | None = None,
) -> LockstepCaseResult:
    """运行单一用例基准测试并返回 LockstepCaseResult 强类型结果。"""
    min_tokens = min(max_new_tokens) if isinstance(max_new_tokens, Sequence) else max_new_tokens
    if min_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2，才能测量 Decode")

    is_variable_prompt_lens = isinstance(prompt_len, Sequence) and not isinstance(prompt_len, str)
    effective_batch_size = len(prompt_len) if is_variable_prompt_lens else batch_size
    input_ids = make_prompt(vocab, prompt_len, effective_batch_size, device)

    warmup_opts = GenerationOptions(
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
    )
    for _ in range(warmup):
        generate(
            mode,
            model,
            input_ids,
            max_new_tokens,
            options=warmup_opts,
        )

    is_full_profile = profile_mode == ProfileMode.FULL_PROFILE
    run_opts = GenerationOptions(
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
        profile=is_full_profile,
        tensorboard_options=tensorboard_options if is_full_profile else None,
    )
    runs = [
        generate(
            mode,
            model,
            input_ids,
            max_new_tokens,
            options=run_opts,
        )
        for _ in range(repeats)
    ]

    raw = {
        "ttft_ms": [r.metrics.ttft_ms for r in runs],
        "tpot_ms": [r.metrics.mean_tpot_ms for r in runs],
        "e2e_wall_ms": [r.metrics.e2e_wall_ms for r in runs],
        "e2e_model_ms": [r.metrics.e2e_model_ms for r in runs],
        "throughput_tok_s": [r.metrics.throughput_tok_s for r in runs],
        "throughput_req_s": [r.metrics.throughput_req_s for r in runs],
        "gpu_time_ratio": [r.metrics.gpu_time_ratio for r in runs],
    }
    if runs and runs[0].batch_info is not None:
        raw["effective_prompt_token_throughput_per_second"] = [
            r.batch_info.effective_prompt_tok_s for r in runs if r.batch_info is not None
        ]
        raw["padded_prompt_token_throughput_per_second"] = [
            r.batch_info.padded_prompt_tok_s for r in runs if r.batch_info is not None
        ]

    workload = WorkloadSpec(
        mode=str(mode),
        batch_size=effective_batch_size,
        prompt_len=tuple(prompt_len) if is_variable_prompt_lens else prompt_len,
        max_new_tokens=tuple(max_new_tokens) if isinstance(max_new_tokens, Sequence) else max_new_tokens,
        repeats=repeats,
        warmup=warmup,
    )

    eff_p_stat = (
        summarize_samples(raw["effective_prompt_token_throughput_per_second"])
        if "effective_prompt_token_throughput_per_second" in raw
        else None
    )
    pad_p_stat = (
        summarize_samples(raw["padded_prompt_token_throughput_per_second"])
        if "padded_prompt_token_throughput_per_second" in raw
        else None
    )
    gpu_ratio_stat = (
        summarize_samples(raw["gpu_time_ratio"]) if all(v is not None for v in raw["gpu_time_ratio"]) else None
    )

    metrics: StepAggregatedMetrics[DistributionStats] = StepAggregatedMetrics(
        ttft_ms=summarize_samples(raw["ttft_ms"]),
        mean_tpot_ms=summarize_samples(raw["tpot_ms"]),
        e2e_wall_ms=summarize_samples(raw["e2e_wall_ms"]),
        e2e_model_ms=summarize_samples(raw["e2e_model_ms"]),
        throughput_tok_s=summarize_samples(raw["throughput_tok_s"]),
        throughput_req_s=summarize_samples(raw["throughput_req_s"]),
        gpu_time_ratio=gpu_ratio_stat,
        effective_prompt_tok_s=eff_p_stat,
        padded_prompt_tok_s=pad_p_stat,
    )

    mib = 1024**2
    weights_mb = cost["weight_bytes"] / mib if cost else None
    case_memory = MemoryProfile(
        weights_mb=weights_mb,
        baseline_mb=max((r.memory.baseline_mb for r in runs), default=0.0),
        prefill_mb=max((r.memory.prefill_mb or 0.0 for r in runs), default=0.0),
        peak_allocated_mb=max((r.memory.peak_allocated_mb for r in runs), default=0.0),
        peak_reserved_mb=max((r.memory.peak_reserved_mb for r in runs), default=0.0),
        kv_cache_mb=max((r.memory.kv_cache_mb for r in runs), default=0.0),
    )

    packing = runs[0].batch_info if runs else None

    eff_obj = (
        efficiency(
            cost,
            peaks,
            batch_size,
            prompt_len,
            max_new_tokens,
            metrics.ttft_ms.median if metrics.ttft_ms is not None else 0.0,
            metrics.mean_tpot_ms.median if metrics.mean_tpot_ms is not None else 0.0,
        )
        if cost and peaks
        else None
    )

    op_breakdown = None
    if is_full_profile and runs:
        breakdowns = [r.op_breakdown for r in runs if r.op_breakdown is not None]
        op_breakdown = merge_op_latency_breakdowns(breakdowns)
    elif profile_mode == ProfileMode.EXTRA_PROFILE:
        prof_opts = GenerationOptions(
            eos_token_id=-1,
            device=device,
            stop_on_eos=False,
            profile=True,
            tensorboard_options=tensorboard_options,
        )
        prof_res = generate(
            mode=mode,
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            options=prof_opts,
        )
        op_breakdown = prof_res.op_breakdown

    return LockstepCaseResult(
        workload=workload,
        metrics=metrics,
        memory=case_memory,
        packing=packing,
        efficiency=eff_obj,
        op_breakdown=op_breakdown,
        raw_samples=raw,
    )


__all__ = [
    "resolve_case_specs",
    "run_case",
]
