"""CLI 子命令：mini-infra bench lockstep (case / suite)

齐步（Lockstep：所有请求同时开始，无调度器）基准测试执行入口。
"""

import argparse
import sys

from mini_infra.bench.common import (
    RESULT_HISTORY_DIR,
    SUITES_DIR,
    add_common_args,
    check_correctness,
    format_lengths,
    git_info,
    gpu_info,
    model_cost,
    require_cuda,
    resolve_peak_specs,
    save_benchmark_record,
)
from mini_infra.bench.lockstep.format import (
    format_case_summary,
    format_table,
)
from mini_infra.bench.lockstep.run_case import resolve_case_specs, run_case
from mini_infra.bench.lockstep.run_suite import run_suite
from mini_infra.bench.lockstep.suites import SUITES, SuiteCase, get_suite
from mini_infra.bench.schema import build_benchmark_record
from mini_infra.engine import BatchingMethod, FeatureFlag, Mode
from mini_infra.model import Qwen3Tokenizer, load_model, resolve_model_directory
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


def add_lockstep_suite_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 lockstep suite 参数。"""
    add_common_args(p)
    p.add_argument(
        "--suite",
        choices=sorted(SUITES),
        required=True,
        help="运行 suites.py 中注册的命名 suite（配置来自该文件，结果写入 benchmarks/suites/<name>/）",
    )
    return p


def build_case_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行单用例齐步（Lockstep）基准测试")
    add_lockstep_case_args(p)
    return p


def build_suite_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行预设齐步（Lockstep）套件")
    add_lockstep_suite_args(p)
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 lockstep 组及 case/suite 子命令。"""
    lockstep_parser = subparsers.add_parser("lockstep", help="齐步（Lockstep：所有请求同时开始，无调度器）推理基准测试")
    lockstep_subs = lockstep_parser.add_subparsers(dest="lockstep_subcommand", required=True)

    case_parser = lockstep_subs.add_parser("case", help="运行单用例齐步（Lockstep）基准测试")
    add_lockstep_case_args(case_parser)
    case_parser.set_defaults(func=handle_case)

    suite_parser = lockstep_subs.add_parser("suite", help="运行预设齐步（Lockstep）套件")
    add_lockstep_suite_args(suite_parser)
    suite_parser.set_defaults(func=handle_suite)
    return lockstep_parser


def handle_case(args: argparse.Namespace) -> int:
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

    print(format_case_summary(result))

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
    print(format_table([result]))
    return 0


def handle_suite(args: argparse.Namespace) -> int:
    device = require_cuda()

    suite = get_suite(args.suite)
    modes = suite.modes()
    cases = suite.cases()
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    mode_label = ",".join(str(m) for m in modes)

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    vocab = model.config.vocab_size

    correctness = {str(m): check_correctness(model, tokenizer, m, device, update_golden=False) for m in modes}
    if any(c["passed"] is False for c in correctness.values()) and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info()
    peaks = resolve_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = model_cost(model)
    if None in peaks.values():
        print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
    tb_options = TensorBoardOptions(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None

    print(
        f"[bench-suite] suite={args.suite} modes={mode_label} cases={len(cases)} warmup={warmup} repeats={repeats}",
        flush=True,
    )

    def on_progress(idx: int, total: int, case: SuiteCase) -> None:
        print(
            f"[{idx}/{total}] mode={case.mode} batch={case.batch_size} "
            f"prompt={format_lengths(case.prompt_len)} max_new_tokens={format_lengths(case.max_new_tokens)} ...",
            flush=True,
        )

    results = run_suite(
        model=model,
        device=device,
        suite=suite,
        vocab=vocab,
        warmup=warmup,
        repeats=repeats,
        cost=cost,
        peaks=peaks,
        profile_mode=args.profile_mode,
        tensorboard_options=tb_options,
        on_case_progress=on_progress,
        on_case_result=lambda r: print(format_case_summary(r)),
    )

    git = git_info()
    record = build_benchmark_record(
        args=args,
        model=model,
        mode_label=mode_label,
        cases=results,
        correctness=correctness,
        env=env,
        peaks=peaks,
        cost=cost,
        git=git,
        warmup=warmup,
        repeats=repeats,
        suite_info=suite.to_dict(),
    )

    suite_dir = SUITES_DIR / args.suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or suite_dir / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, dirty=git["dirty"])
    print()
    print(format_table(results))
    return 0
