"""单用例基准测试（run_case）：一次调用只执行正好一个 Case。

三个核心参数遵循 numpy 式广播规则：
    --batch-size      标量 int（可选）。若未指定，自动由变长 prompt_lens / max_new_tokens
                      长度推断；若均为单值则默认为 1。
    --prompt-lens     int 列表。1 项表示全 Batch 等长；多项表示逐请求长度。
    --max-new-tokens  int 列表。1 项表示全 Batch 等长；多项表示逐请求预算。

用法示例：
    # 单用例快速测试（batch=1, prompt=32, max_new_tokens=16）
    mini-infra-bench-case --tag dev --prompt-lens 32 --max-new-tokens 16

    # 批处理等长测试（batch=4, prompt=128, max_new_tokens=32）
    mini-infra-bench-case --tag dev --batch-size 4 --prompt-lens 128 \
        --max-new-tokens 32 --batching-method fixed-batching

    # 不等长变长测试（batch 自动由 prompt_lens 推断为 4）
    mini-infra-bench-case --tag dev --prompt-lens 16 32 64 128 --max-new-tokens 32 --batching-method fixed-batching
"""

import argparse
import sys
from collections.abc import Sequence

import torch

from mini_infra.bench.batch.suites import SuiteCase
from mini_infra.bench.batch.types import (
    BatchPackingProfile,
    CaseBenchmarkResult,
    CaseMemoryProfile,
    HardwareEfficiency,
    LatencyDistributions,
    ThroughputDistributions,
    WorkloadSpec,
    summarize_samples,
)
from mini_infra.bench.common import (
    RESULT_HISTORY_DIR,
    ProfileMode,
    add_common_args,
    build_benchmark_record,
    check_correctness,
    efficiency,
    format_lengths,
    git_info,
    gpu_info,
    model_cost,
    print_case_summary,
    print_table,
    require_cuda,
    resolve_peak_specs,
    save_benchmark_record,
)
from mini_infra.bench.workloads import make_prompt
from mini_infra.engine import (
    BatchingMethod,
    FeatureFlag,
    Mode,
    generate,
)
from mini_infra.model import (
    Qwen3Tokenizer,
    load_model,
    resolve_model_directory,
)
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
) -> CaseBenchmarkResult:
    """运行单一用例基准测试并返回 CaseBenchmarkResult 强类型结果。"""
    min_tokens = min(max_new_tokens) if isinstance(max_new_tokens, Sequence) else max_new_tokens
    if min_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2，才能测量 Decode")

    is_variable_prompt_lens = isinstance(prompt_len, Sequence) and not isinstance(prompt_len, str)
    effective_batch_size = len(prompt_len) if is_variable_prompt_lens else batch_size
    input_ids = make_prompt(vocab, prompt_len, effective_batch_size, device)

    for _ in range(warmup):
        generate(
            mode,
            model,
            input_ids,
            max_new_tokens,
            eos_token_id=-1,
            device=device,
            stop_on_eos=False,
        )

    is_full_profile = profile_mode == ProfileMode.FULL_PROFILE
    runs = [
        generate(
            mode,
            model,
            input_ids,
            max_new_tokens,
            eos_token_id=-1,
            device=device,
            stop_on_eos=False,
            profile=is_full_profile,
            tensorboard_options=tensorboard_options if is_full_profile else None,
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

    latency = LatencyDistributions(
        ttft_ms=summarize_samples(raw["ttft_ms"]),
        tpot_ms=summarize_samples(raw["tpot_ms"]),
        e2e_wall_ms=summarize_samples(raw["e2e_wall_ms"]),
        e2e_model_ms=summarize_samples(raw["e2e_model_ms"]),
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

    throughput = ThroughputDistributions(
        throughput_tok_s=summarize_samples(raw["throughput_tok_s"]),
        throughput_req_s=summarize_samples(raw["throughput_req_s"]),
        effective_prompt_tok_s=eff_p_stat,
        padded_prompt_tok_s=pad_p_stat,
    )

    mib = 1024**2
    worst_run = max(runs, key=lambda r: r.memory.peak_allocated_mb) if runs else None
    case_memory = CaseMemoryProfile(
        weights_mb=cost["weight_bytes"] / mib if cost else None,
        baseline_allocated_mb=worst_run.memory.baseline_mb if worst_run else 0.0,
        prefill_allocated_mb=worst_run.memory.prefill_mb if worst_run else 0.0,
        peak_allocated_mb=worst_run.memory.peak_allocated_mb if worst_run else 0.0,
        peak_reserved_mb=worst_run.memory.peak_reserved_mb if worst_run else 0.0,
        kv_cache_mb=worst_run.memory.kv_cache_mb if worst_run else 0.0,
    )

    packing = None
    if runs and runs[0].batch_info is not None:
        b0 = runs[0].batch_info
        packing = BatchPackingProfile(
            padding_efficiency=b0.padding_efficiency,
            slot_utilization=b0.slot_utilization,
            prompt_tokens=b0.prompt_tokens,
            padded_prompt_tokens=b0.padded_prompt_tokens,
            prompt_lengths_per_req=b0.prompt_lengths_per_req,
            completion_steps_per_req=b0.completion_steps_per_req,
            effective_prompt_throughput=eff_p_stat,
            padded_prompt_throughput=pad_p_stat,
        )

    eff_obj = None
    if cost and peaks:
        eff_dict = efficiency(
            cost,
            peaks,
            batch_size,
            prompt_len,
            max_new_tokens,
            latency.ttft_ms.median,
            latency.tpot_ms.median,
        )
        eff_obj = HardwareEfficiency(
            prefill_mfu=eff_dict.get("prefill_mfu"),
            decode_mfu=eff_dict.get("decode_mfu"),
            decode_mbu=eff_dict.get("decode_mbu"),
        )

    op_breakdown = None
    if is_full_profile and runs:
        breakdowns = [r.op_breakdown for r in runs if r.op_breakdown is not None]
        op_breakdown = merge_op_latency_breakdowns(breakdowns)
    elif profile_mode == ProfileMode.EXTRA_PROFILE:
        prof_res = generate(
            mode=mode,
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=-1,
            stop_on_eos=False,
            profile=True,
            tensorboard_options=tensorboard_options,
        )
        op_breakdown = prof_res.op_breakdown

    gpu_ratio_stat = (
        summarize_samples(raw["gpu_time_ratio"]) if all(v is not None for v in raw["gpu_time_ratio"]) else None
    )

    return CaseBenchmarkResult(
        workload=workload,
        latency=latency,
        throughput=throughput,
        memory=case_memory,
        packing=packing,
        efficiency=eff_obj,
        gpu_time_ratio=gpu_ratio_stat,
        op_breakdown=op_breakdown,
        raw_samples=raw,
    )


def parse_args():
    p = argparse.ArgumentParser(description="mini-infra-qwen3 benchmark single case runner")
    add_common_args(p)
    p.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 KV Cache 特性（默认: True）",
    )
    p.add_argument(
        "--batching-method",
        default=BatchingMethod.NONE.value,
        choices=[m.value for m in BatchingMethod],
        help="指定批处理策略（默认: none）",
    )
    p.add_argument(
        "--batch-size", type=int, default=None, help="批处理大小；未指定时自动由 prompt_lens/max_new_tokens 推断"
    )
    p.add_argument(
        "--prompt-lens",
        type=int,
        nargs="+",
        default=[128],
        help="Prompt 长度（1 项表示等长，多项表示逐请求长度，默认: 128）",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[64],
        help="最大新生成 Token 预算（1 项表示等长，多项表示逐请求长度，默认: 64）",
    )
    p.add_argument("--update-golden", action="store_true", help="用当前输出覆盖 golden（仅在 baseline 上使用）")
    return p.parse_args()


def main():
    args = parse_args()
    device = require_cuda()

    target_bs, resolved_prompt_len, resolved_max_new_tokens = resolve_case_specs(
        args.prompt_lens, args.max_new_tokens, args.batch_size
    )

    flags = {FeatureFlag.KV_CACHE} if args.kv_cache else set()
    mode = Mode(flags, batching_method=args.batching_method)
    case = SuiteCase(mode, target_bs, resolved_prompt_len, resolved_max_new_tokens)

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    vocab = model.config.vocab_size

    correctness = check_correctness(model, tokenizer, case.mode, device, args.update_golden)
    if correctness["passed"] is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info()
    peaks = resolve_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = model_cost(model)
    if None in peaks.values():
        print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
    tb_options = TensorBoardOptions(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None

    print(
        f"[bench-case] mode={case.mode} batch={case.batch_size} prompt={format_lengths(case.prompt_len)} "
        f"max_new_tokens={format_lengths(case.max_new_tokens)} ...",
        flush=True,
    )

    result = run_case(
        model,
        case.mode,
        list(case.prompt_len) if isinstance(case.prompt_len, tuple) else case.prompt_len,
        list(case.max_new_tokens) if isinstance(case.max_new_tokens, tuple) else case.max_new_tokens,
        case.batch_size,
        args.warmup,
        args.repeats,
        vocab,
        device,
        cost=cost,
        peaks=peaks,
        profile_mode=args.profile_mode,
        tensorboard_options=tb_options,
    )

    print_case_summary(result)

    git = git_info()
    record = build_benchmark_record(
        args=args,
        model=model,
        mode_label=str(case.mode),
        cases=[result],
        correctness=correctness,
        env=env,
        peaks=peaks,
        cost=cost,
        git=git,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    output = args.output or RESULT_HISTORY_DIR / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, dirty=git["dirty"])
    print()
    # 末尾输出标准单行 CSV 表格：供脚本重定向收集或复制至电子表格对比（与前文人类可读的 summary 互补）
    print_table([result], mode=str(case.mode))


if __name__ == "__main__":
    main()
