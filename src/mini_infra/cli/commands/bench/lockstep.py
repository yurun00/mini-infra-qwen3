"""CLI 子命令：mini-infra bench lockstep case

齐步（Lockstep：所有请求同时开始，无调度器）单用例基准测试执行入口。
套件执行请使用 `mini-infra bench sweep --engine lockstep`。
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from mini_infra.bench.cases import BenchmarkCase, WorkloadProfile
from mini_infra.bench.common import (
    LOCKSTEP_RESULT_HISTORY_DIR,
    HardwarePeakSpecs,
    ModelCostProfile,
    add_common_args,
    check_correctness,
    format_lengths,
    git_info,
    gpu_info,
    require_cuda,
    save_benchmark_record,
)
from mini_infra.bench.context import BenchmarkExecutionContext
from mini_infra.bench.lockstep.format import (
    format_suite_plan,
    format_table,
)
from mini_infra.bench.lockstep.run_case import broadcast_case_lengths, run_case
from mini_infra.bench.lockstep.run_suite import run_suite
from mini_infra.bench.schema import build_benchmark_record
from mini_infra.bench.suite import BenchmarkSuite
from mini_infra.bench.sweep import SuiteRunSummary
from mini_infra.engine import BatchingMethod, CachingMethod, Mode, Qwen3Tokenizer
from mini_infra.model import load_model, resolve_model_directory
from mini_infra.profiling import TensorBoardOptions


def add_lockstep_case_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 lockstep case 参数。"""
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
        choices=[BatchingMethod.NONE.value, BatchingMethod.FIXED.value],
        help="指定齐步批处理策略（默认: none）",
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
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 lockstep 组及 case 子命令。"""
    lockstep_parser = subparsers.add_parser("lockstep", help="齐步（Lockstep：所有请求同时开始，无调度器）推理基准测试")
    lockstep_subs = lockstep_parser.add_subparsers(dest="lockstep_subcommand", required=True)

    case_parser = lockstep_subs.add_parser("case", help="运行单用例齐步（Lockstep）基准测试")
    add_lockstep_case_args(case_parser)
    case_parser.set_defaults(func=handle_case)
    return lockstep_parser


def handle_case(args: argparse.Namespace) -> int:
    device = require_cuda()

    target_bs, resolved_prompt_len, resolved_max_new_tokens = broadcast_case_lengths(
        args.prompt_lens, args.max_new_tokens, args.batch_size
    )

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)

    caching_method = CachingMethod.DENSE if args.kv_cache else CachingMethod.NONE
    mode = Mode(caching_method, batching_method=args.batching_method)

    plens = (resolved_prompt_len,) * target_bs if isinstance(resolved_prompt_len, int) else resolved_prompt_len
    budgets = (
        (resolved_max_new_tokens,) * target_bs if isinstance(resolved_max_new_tokens, int) else resolved_max_new_tokens
    )
    profile = WorkloadProfile(
        workload_id="custom",
        num_total_requests=target_bs,
        prompt_lens=plens,
        output_budgets=budgets,
        arrival_ms=(0.0,) * target_bs,
    )
    case = BenchmarkCase(
        profile=profile,
        mode=mode,
        suite_name="custom",
        engine="lockstep",
    )

    correctness = check_correctness(model, tokenizer, case.mode, device, args.update_golden)
    if correctness["passed"] is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info()
    peaks = HardwarePeakSpecs.lookup(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = ModelCostProfile.from_model(model)
    if None in peaks.values():
        print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
    tb_options = TensorBoardOptions(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None

    print(
        f"[bench-case] mode={case.mode} batch={case.profile.num_total_requests} "
        f"prompt={format_lengths(case.profile.prompt_lens)} "
        f"max_new_tokens={format_lengths(case.profile.output_budgets)} ...",
        flush=True,
    )

    result = run_case(
        model=model,
        device=device,
        case=case,
        warmup=args.warmup,
        repeats=args.repeats,
        cost=cost,
        peaks=peaks,
        profile_mode=args.profile_mode,
        tensorboard_options=tb_options,
    )
    if getattr(args, "scale", None) is not None or getattr(args, "seed", None) is not None:
        result = replace(
            result,
            scale=getattr(args, "scale", None),
            seed=getattr(args, "seed", None),
        )

    print(result.format_summary())

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

    output = args.output or LOCKSTEP_RESULT_HISTORY_DIR / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, dirty=git["dirty"])
    print()
    print(format_table([result]))
    return 0


def execute_lockstep_suite(
    ctx: BenchmarkExecutionContext, args: argparse.Namespace, suite: BenchmarkSuite, output_path: Path
) -> SuiteRunSummary:
    """在已准备好的环境中运行一个 lockstep suite：正确性检查 → 执行 → 落盘 → 打印表格。"""
    modes = suite.modes()
    cases = suite.cases
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    mode_label = ",".join(str(m) for m in modes)

    for m in modes:
        if m not in ctx.correctness_cache:
            assert ctx.tokenizer is not None
            ctx.correctness_cache[m] = check_correctness(ctx.model, ctx.tokenizer, m, ctx.device, update_golden=False)
    correctness = {str(m): ctx.correctness_cache[m] for m in modes}
    if any(c["passed"] is False for c in correctness.values()) and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    tb_options = TensorBoardOptions(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None

    print(
        f"[bench-suite] suite={suite.name} modes={mode_label} cases={len(cases)} warmup={warmup} repeats={repeats}",
        flush=True,
    )

    def on_progress(case_number: int, total_cases: int, case: BenchmarkCase) -> None:
        print(
            f"[{case_number}/{total_cases}] mode={case.mode} batch={case.profile.num_total_requests} "
            f"prompt={format_lengths(case.profile.prompt_lens)} "
            f"max_new_tokens={format_lengths(case.profile.output_budgets)} ...",
            flush=True,
        )

    results = run_suite(
        model=ctx.model,
        device=ctx.device,
        suite=suite,
        warmup=warmup,
        repeats=repeats,
        cost=ctx.cost,
        peaks=ctx.peaks,
        profile_mode=args.profile_mode,
        tensorboard_options=tb_options,
        on_case_progress=on_progress,
        on_case_result=lambda r: print(r.format_summary()),
    )

    record = build_benchmark_record(
        args=args,
        model=ctx.model,
        mode_label=mode_label,
        cases=results,
        correctness=correctness,
        env=ctx.env,
        peaks=ctx.peaks,
        cost=ctx.cost,
        git=ctx.git,
        warmup=warmup,
        repeats=repeats,
        suite_info=suite.to_dict(),
    )
    save_benchmark_record(output_path, record, dirty=ctx.dirty)
    print()
    print(format_table(results))
    return SuiteRunSummary(name=suite.name, engine="lockstep", num_cases=len(results), output=output_path)


def print_lockstep_dry_run(args: argparse.Namespace, suite: BenchmarkSuite) -> None:
    """展开并打印用例列表（不加载模型、不要求 CUDA）。"""
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    print(f"=== [Dry Run] Suite: {suite.name} ===")
    print(format_suite_plan(suite.name, suite.cases, warmup, repeats))
