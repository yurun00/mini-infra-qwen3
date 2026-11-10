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

import gc
from collections.abc import Sequence

import torch

from mini_infra.bench.cases import LockstepCase
from mini_infra.bench.common import DEFAULT_REPEATS, DEFAULT_WARMUP, ProfileMode, efficiency
from mini_infra.bench.lockstep.types import (
    DistributionStats,
    LockstepCaseResult,
    summarize_samples,
)
from mini_infra.engine import GenerationOptions, GenerationResult, generate
from mini_infra.engine.metrics import (
    BatchScheduleInfo,
    MemoryProfile,
    StepAggregatedMetrics,
    summarize_repeated_kv_cache_profiles,
)
from mini_infra.profiling import TensorBoardOptions, merge_op_latency_breakdowns


def broadcast_case_lengths(
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


def run_case_once(
    model: torch.nn.Module,
    device: torch.device,
    case: LockstepCase,
    options: GenerationOptions | None = None,
) -> GenerationResult:
    """执行单一 LockstepCase 的单次生成运行（原子操作）。"""
    prompts = [list(s.token_ids) for s in case.payload.specs]
    budgets = case.payload.output_budgets
    max_new_tokens = budgets[0] if len(set(budgets)) == 1 else budgets
    opts = options or GenerationOptions(
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
    )
    res = generate(
        case.mode,
        model,
        prompts,
        max_new_tokens,
        options=opts,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def run_case(
    model: torch.nn.Module,
    device: torch.device,
    case: LockstepCase,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    cost: dict | None = None,
    peaks: dict | None = None,
    profile_mode: ProfileMode | None = None,
    tensorboard_options: TensorBoardOptions | None = None,
    scale: str | None = None,
    seed: int | None = None,
) -> LockstepCaseResult:
    """运行单一用例基准测试并返回 LockstepCaseResult 强类型结果。"""
    output_budgets = case.payload.output_budgets
    batch_size = case.payload.request_count

    min_tokens = min(output_budgets)
    if min_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2，才能测量 Decode")

    workload = case.payload.profile

    warmup_opts = GenerationOptions(
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
    )
    for _ in range(warmup):
        run_case_once(model, device, case, options=warmup_opts)

    is_full_profile = profile_mode == ProfileMode.FULL_PROFILE
    run_opts = GenerationOptions(
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
        profile=is_full_profile,
        tensorboard_options=tensorboard_options if is_full_profile else None,
    )
    runs = [run_case_once(model, device, case, options=run_opts) for _ in range(repeats)]

    eff_prompt_vals = [r.metrics.effective_prompt_tok_s for r in runs if r.metrics.effective_prompt_tok_s is not None]
    pad_prompt_vals = [r.metrics.padded_prompt_tok_s for r in runs if r.metrics.padded_prompt_tok_s is not None]
    gpu_ratios = [r.metrics.gpu_time_ratio for r in runs]
    gpu_ratio_stat = (
        summarize_samples([v for v in gpu_ratios if v is not None]) if all(v is not None for v in gpu_ratios) else None
    )

    metrics: StepAggregatedMetrics[DistributionStats] = StepAggregatedMetrics(
        ttft_ms=summarize_samples([r.metrics.ttft_ms for r in runs if r.metrics.ttft_ms is not None]),
        mean_tpot_ms=summarize_samples([r.metrics.mean_tpot_ms for r in runs if r.metrics.mean_tpot_ms is not None]),
        e2e_wall_ms=summarize_samples([r.metrics.e2e_wall_ms for r in runs]),
        e2e_model_ms=summarize_samples([r.metrics.e2e_model_ms for r in runs]),
        throughput_tok_s=summarize_samples([r.metrics.throughput_tok_s for r in runs]),
        throughput_req_s=summarize_samples([r.metrics.throughput_req_s for r in runs]),
        gpu_time_ratio=gpu_ratio_stat,
        effective_prompt_tok_s=summarize_samples(eff_prompt_vals) if eff_prompt_vals else None,
        padded_prompt_tok_s=summarize_samples(pad_prompt_vals) if pad_prompt_vals else None,
    )

    mib = 1024**2
    weights_mb = cost["weight_bytes"] / mib if cost else None
    case_memory = MemoryProfile(
        weights_mb=weights_mb,
        baseline_mb=max((r.memory.baseline_mb for r in runs), default=0.0),
        prefill_mb=max((r.memory.prefill_mb or 0.0 for r in runs), default=0.0),
        peak_allocated_mb=max((r.memory.peak_allocated_mb for r in runs), default=0.0),
        peak_reserved_mb=max((r.memory.peak_reserved_mb for r in runs), default=0.0),
        kv_cache=summarize_repeated_kv_cache_profiles([r.memory.kv_cache for r in runs]),
    )

    batch_infos = [r.batch_info for r in runs if r.batch_info is not None]
    packing: BatchScheduleInfo[DistributionStats] | None = None
    if batch_infos:
        packing = BatchScheduleInfo(
            padding_efficiency=summarize_samples([b.padding_efficiency for b in batch_infos]),
            slot_utilization=summarize_samples([b.slot_utilization for b in batch_infos]),
        )

    eff = (
        efficiency(
            cost,
            peaks,
            batch_size,
            workload.prompt_len,
            workload.max_new_tokens,
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
        prof_res = run_case_once(model, device, case, options=prof_opts)
        op_breakdown = prof_res.op_breakdown

    return LockstepCaseResult(
        key=case.key,
        workload=workload,
        metrics=metrics,
        memory=case_memory,
        repeats=repeats,
        warmup=warmup,
        packing=packing,
        efficiency=eff,
        op_breakdown=op_breakdown,
        scale=scale,
        seed=seed,
    )


__all__ = [
    "broadcast_case_lengths",
    "run_case",
    "run_case_once",
]
