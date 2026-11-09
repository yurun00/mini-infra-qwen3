"""CLI 子命令：mini-infra bench lockstep case

齐步（Lockstep：所有请求同时开始，无调度器）单用例基准测试执行入口。
套件执行请使用 `mini-infra bench sweep --engine lockstep`。
"""

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from mini_infra.bench.cases import LockstepCase, RequestPayload
from mini_infra.bench.common import (
    LOCKSTEP_RESULT_HISTORY_DIR,
    add_common_args,
    check_correctness,
    format_lengths,
    git_info,
    gpu_info,
    lookup_gpu_peak_specs,
    model_cost,
    require_cuda,
    save_benchmark_record,
)
from mini_infra.bench.lockstep.format import (
    format_case_summary,
    format_suite_plan,
    format_table,
)
from mini_infra.bench.lockstep.run_case import broadcast_case_lengths, run_case
from mini_infra.bench.lockstep.run_suite import run_suite
from mini_infra.bench.lockstep.suites import LockstepSuite
from mini_infra.bench.schema import build_benchmark_record
from mini_infra.bench.sweep import SuiteRunSummary
from mini_infra.bench.workloads import make_sequences
from mini_infra.core.request import make_request_specs
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


def build_case_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行单用例齐步（Lockstep）基准测试")
    add_lockstep_case_args(p)
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
    vocab = model.config.vocab_size

    caching_method = CachingMethod.DENSE if args.kv_cache else CachingMethod.NONE
    mode = Mode(caching_method, batching_method=args.batching_method)

    plens = (resolved_prompt_len,) * target_bs if isinstance(resolved_prompt_len, int) else resolved_prompt_len
    budgets = (
        (resolved_max_new_tokens,) * target_bs if isinstance(resolved_max_new_tokens, int) else resolved_max_new_tokens
    )
    specs = tuple(make_request_specs(make_sequences(plens, vocab), budgets, [0.0] * target_bs))
    payload = RequestPayload(workload_id="custom", specs=specs)
    case = LockstepCase(
        payload=payload,
        mode=mode,
        suite_name="custom",
    )

    correctness = check_correctness(model, tokenizer, case.mode, device, args.update_golden)
    if correctness["passed"] is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info()
    peaks = lookup_gpu_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = model_cost(model)
    if None in peaks.values():
        print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
    tb_options = TensorBoardOptions(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None

    print(
        f"[bench-case] mode={case.mode} batch={case.payload.request_count} "
        f"prompt={format_lengths(case.payload.prompt_lengths)} "
        f"max_new_tokens={format_lengths(case.payload.output_budgets)} ...",
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

    output = args.output or LOCKSTEP_RESULT_HISTORY_DIR / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, dirty=git["dirty"])
    print()
    print(format_table([result]))
    return 0


@dataclass
class LockstepRunEnv:
    """一次 lockstep 运行（单个 suite 或 sweep 内多个 suite）共享的模型与环境信息。"""

    model: torch.nn.Module
    tokenizer: Qwen3Tokenizer
    device: torch.device
    vocab: int
    env: dict[str, Any]
    peaks: dict[str, float | None]
    cost: dict[str, Any]
    git: dict[str, Any]
    correctness_cache: dict[Mode, dict[str, Any]] = field(default_factory=dict)


def prepare_lockstep_env(args: argparse.Namespace) -> LockstepRunEnv:
    """要求 CUDA、加载模型并采集环境信息。"""
    device = require_cuda()
    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    env = gpu_info()
    peaks = lookup_gpu_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    if None in peaks.values():
        print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
    return LockstepRunEnv(
        model=model,
        tokenizer=tokenizer,
        device=device,
        vocab=model.config.vocab_size,
        env=env,
        peaks=peaks,
        cost=model_cost(model),
        git=git_info(),
    )


def execute_lockstep_suite(
    run_env: LockstepRunEnv, args: argparse.Namespace, suite: LockstepSuite, output_path: Path
) -> SuiteRunSummary:
    """在已准备好的环境中运行一个 lockstep suite：正确性检查 → 执行 → 落盘 → 打印表格。"""
    modes = suite.modes()
    cases = suite.cases(run_env.vocab)
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    mode_label = ",".join(str(m) for m in modes)

    for m in modes:
        if m not in run_env.correctness_cache:
            run_env.correctness_cache[m] = check_correctness(
                run_env.model, run_env.tokenizer, m, run_env.device, update_golden=False
            )
    correctness = {str(m): run_env.correctness_cache[m] for m in modes}
    if any(c["passed"] is False for c in correctness.values()) and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    tb_options = TensorBoardOptions(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None

    print(
        f"[bench-suite] suite={suite.name} modes={mode_label} cases={len(cases)} warmup={warmup} repeats={repeats}",
        flush=True,
    )

    def on_progress(case_number: int, total_cases: int, case: LockstepCase) -> None:
        print(
            f"[{case_number}/{total_cases}] mode={case.mode} batch={case.payload.request_count} "
            f"prompt={format_lengths(case.payload.prompt_lengths)} "
            f"max_new_tokens={format_lengths(case.payload.output_budgets)} ...",
            flush=True,
        )

    results = run_suite(
        model=run_env.model,
        device=run_env.device,
        suite=suite,
        vocab=run_env.vocab,
        warmup=warmup,
        repeats=repeats,
        cost=run_env.cost,
        peaks=run_env.peaks,
        profile_mode=args.profile_mode,
        tensorboard_options=tb_options,
        on_case_progress=on_progress,
        on_case_result=lambda r: print(format_case_summary(r)),
    )

    record = build_benchmark_record(
        args=args,
        model=run_env.model,
        mode_label=mode_label,
        cases=results,
        correctness=correctness,
        env=run_env.env,
        peaks=run_env.peaks,
        cost=run_env.cost,
        git=run_env.git,
        warmup=warmup,
        repeats=repeats,
        suite_info=suite.to_dict(),
    )
    save_benchmark_record(output_path, record, dirty=run_env.git["dirty"])
    print()
    print(format_table(results))
    return SuiteRunSummary(name=suite.name, engine="lockstep", num_cases=len(results), output=output_path)


def print_lockstep_dry_run(args: argparse.Namespace, suite: LockstepSuite) -> None:
    """展开并打印用例列表（不加载模型、不要求 CUDA）。"""
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    print(f"=== [Dry Run] Suite: {suite.name} ===")
    print(format_suite_plan(suite.name, suite.cases(vocab_size=100), warmup, repeats))
