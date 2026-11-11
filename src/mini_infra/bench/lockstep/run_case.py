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

from mini_infra.bench.cases import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    BenchmarkCase,
    ProfilingMode,
)
from mini_infra.bench.efficiency import HardwarePeakSpecs, ModelCostProfile
from mini_infra.bench.lockstep.types import (
    LockstepCaseResult,
)
from mini_infra.engine import GenerationOptions, GenerationResult, generate
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
    case: BenchmarkCase,
    options: GenerationOptions | None = None,
    prompts: Sequence[Sequence[int]] | None = None,
) -> GenerationResult:
    """执行单一 BenchmarkCase 的单次生成运行（原子操作）。"""
    if prompts is None:
        requests = case.profile.to_payload(model.config.vocab_size)
        resolved_prompts = [list(r.token_ids) for r in requests]
    else:
        resolved_prompts = prompts
    budgets = case.profile.output_budgets
    max_new_tokens = budgets[0] if len(set(budgets)) == 1 else budgets
    opts = options or GenerationOptions(
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
    )
    res = generate(
        case.mode,
        model,
        resolved_prompts,
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
    case: BenchmarkCase,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    cost: ModelCostProfile | None = None,
    peaks: HardwarePeakSpecs | None = None,
    profiling_mode: ProfilingMode | None = None,
    tensorboard_options: TensorBoardOptions | None = None,
    scale: str | None = None,
    seed: int | None = None,
) -> LockstepCaseResult:
    """运行单一用例基准测试并返回 LockstepCaseResult 强类型结果。"""
    output_budgets = case.profile.output_budgets

    min_tokens = min(output_budgets)
    if min_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2，才能测量 Decode")

    workload = case.profile
    requests = case.profile.to_payload(model.config.vocab_size)
    prompts = [list(r.token_ids) for r in requests]

    warmup_opts = GenerationOptions(
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
    )
    for _ in range(warmup):
        run_case_once(model, device, case, options=warmup_opts, prompts=prompts)

    is_all_repeats_profile = profiling_mode == ProfilingMode.ALL_REPEATS
    run_opts = GenerationOptions(
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
        profile=is_all_repeats_profile,
        tensorboard_options=tensorboard_options if is_all_repeats_profile else None,
    )
    runs = [run_case_once(model, device, case, options=run_opts, prompts=prompts) for _ in range(repeats)]

    op_breakdown = None
    if is_all_repeats_profile and runs:
        breakdowns = [r.op_breakdown for r in runs if r.op_breakdown is not None]
        op_breakdown = merge_op_latency_breakdowns(breakdowns)
    elif profiling_mode == ProfilingMode.DEDICATED_PASS:
        prof_opts = GenerationOptions(
            eos_token_id=-1,
            device=device,
            stop_on_eos=False,
            profile=True,
            tensorboard_options=tensorboard_options,
        )
        prof_res = run_case_once(model, device, case, options=prof_opts)
        op_breakdown = prof_res.op_breakdown

    return LockstepCaseResult.from_runs(
        case_key=case.key,
        workload=workload,
        runs=runs,
        warmup=warmup,
        cost=cost,
        peaks=peaks,
        op_breakdown=op_breakdown,
        scale=scale,
        seed=seed,
    )


__all__ = [
    "broadcast_case_lengths",
    "run_case",
    "run_case_once",
]
