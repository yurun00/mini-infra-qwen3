"""CLI 子命令：mini-infra bench scheduler case

队列调度器（Scheduler）单用例评测执行入口。套件执行请使用 `mini-infra bench sweep --engine scheduler`。
"""

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from mini_infra.bench.cases import SchedulerSuiteCase
from mini_infra.bench.common import (
    CASE_DIR,
    add_common_args,
    check_correctness,
    git_info,
    gpu_info,
    model_cost,
    require_cuda,
    resolve_peak_specs,
    save_benchmark_record,
)
from mini_infra.bench.scheduler.format import (
    format_case_summary,
    format_comparison,
    format_table,
)
from mini_infra.bench.scheduler.plan import PlannedRun, build_scheduler_run_plan, format_scheduler_run_plan
from mini_infra.bench.scheduler.run_case import run_case
from mini_infra.bench.scheduler.run_suite import (
    parse_interleave_args,
    run_suite,
)
from mini_infra.bench.scheduler.suites import SchedulerSuite
from mini_infra.bench.scheduler.types import MetricComparison, compute_comparisons
from mini_infra.bench.schema import build_benchmark_record
from mini_infra.bench.sweep import SuiteRunSummary
from mini_infra.bench.workloads import Workload, build_scheduler_request_specs, make_arrival_times
from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.engine import BatchingMethod, CachingMethod, Mode, Qwen3Tokenizer
from mini_infra.model import load_model, resolve_model_directory


def add_scheduler_case_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 scheduler case 参数。"""
    add_common_args(p)
    p.add_argument(
        "--caching-method",
        choices=[CachingMethod.DENSE.value, CachingMethod.PAGED.value],
        default=CachingMethod.DENSE.value,
        help=f"执行的 KV 缓存管理策略（默认: {CachingMethod.DENSE.value}）",
    )
    p.add_argument(
        "--batching-method",
        choices=[BatchingMethod.CONTINUOUS.value, BatchingMethod.FIXED.value],
        default=BatchingMethod.CONTINUOUS.value,
        help=f"执行的批处理调度策略（默认: {BatchingMethod.CONTINUOUS.value}）",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Paged KV Cache 的 physical block 大小（默认: 16，仅 paged-kv-cache 时生效）",
    )
    p.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Paged KV Cache 的 physical block pool 总块数（可选，默认按负载自动推导，仅 paged-kv-cache 时生效）",
    )
    p.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="调度队列最大运行并发数（默认: 4）",
    )
    p.add_argument("--prompt-length", type=int, default=128, help="统一输入 Prompt 长度（默认: 128）")
    p.add_argument("--prompt-lens", type=int, nargs="+", default=None, help="逐请求输入 Prompt 长度列表（可选）")
    p.add_argument("--request-count", type=int, default=8, help="请求总数（默认: 8）")
    p.add_argument("--output-budget", type=int, default=20, help="统一输出 Token 预算（默认: 20）")
    p.add_argument("--output-budgets", type=int, nargs="+", default=None, help="逐请求输出 Token 预算列表（可选）")
    p.add_argument("--arrival-interval-ms", type=float, default=0.0, help="错峰到达时间间隔 ms（默认: 0.0）")
    p.add_argument("--arrival-times-ms", type=float, nargs="+", default=None, help="逐请求到达时间列表 ms（可选）")
    return p


def build_case_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行单用例 Scheduler 基准测试")
    add_scheduler_case_args(p)
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 scheduler 组及 case 子命令。"""
    scheduler_parser = subparsers.add_parser("scheduler", help="队列调度器（Scheduler: Fixed vs Continuous）基准测试")
    scheduler_subs = scheduler_parser.add_subparsers(dest="scheduler_subcommand", required=True)

    case_parser = scheduler_subs.add_parser("case", help="运行单用例 Scheduler 基准测试")
    add_scheduler_case_args(case_parser)
    case_parser.set_defaults(func=handle_case)
    return scheduler_parser


def handle_case(args: argparse.Namespace) -> int:
    device = require_cuda()

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    vocab = model.config.vocab_size

    caching = CachingMethod(args.caching_method)
    method = BatchingMethod(args.batching_method)
    target_mode = Mode(caching, batching_method=method)
    paged_cache_config = (
        PagedCacheConfig(block_size=args.block_size, num_blocks=args.num_blocks)
        if caching is CachingMethod.PAGED
        else None
    )

    correctness = check_correctness(
        model,
        tokenizer,
        target_mode,
        device,
        update_golden=False,
        paged_cache_config=paged_cache_config,
    )
    if correctness["passed"] is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    req_count = len(args.prompt_lens) if args.prompt_lens is not None else args.request_count
    max_running = args.max_running_requests or 4
    prompt_len: int | tuple[int, ...] = tuple(args.prompt_lens) if args.prompt_lens is not None else args.prompt_length
    max_new_tokens: int | tuple[int, ...] = (
        tuple(args.output_budgets) if args.output_budgets is not None else args.output_budget
    )
    arrivals: float | tuple[float, ...] | None = (
        tuple(args.arrival_times_ms)
        if args.arrival_times_ms is not None
        else tuple(make_arrival_times(req_count, args.arrival_interval_ms))
    )
    warmup = args.warmup if args.warmup is not None else 1
    repeats = args.repeats if args.repeats is not None else 3

    workload = Workload(
        mode=target_mode,
        num_total_requests=req_count,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        repeats=repeats,
        warmup=warmup,
        max_running_requests=max_running,
        arrival_times_ms=arrivals,
    )
    specs = tuple(build_scheduler_request_specs(workload, vocab))
    case_name = "custom"

    case = SchedulerSuiteCase(
        suite_name=case_name,
        mode=target_mode,
        max_running_requests=max_running,
        specs=specs,
        workload_id="custom",
        paged_cache_config=paged_cache_config,
    )

    block_info = f" block_size={paged_cache_config.block_size}" if paged_cache_config is not None else ""
    print(
        f"[bench-scheduler-case] case={case_name} mode={target_mode}{block_info} requests={len(specs)} "
        f"max_running={max_running} warmup={warmup} repeats={repeats} ...",
        flush=True,
    )

    def on_case_step(phase: str, step_number: int, total_steps: int) -> None:
        print(f"  [{phase} {step_number}/{total_steps}] mode={target_mode} requests={len(specs)} ...", flush=True)

    result = run_case(
        model=model,
        device=device,
        case=case,
        warmup=warmup,
        repeats=repeats,
        on_step_progress=on_case_step,
    )
    if getattr(args, "scale", None) is not None or getattr(args, "seed", None) is not None:
        result = replace(
            result,
            workload=replace(result.workload, scale=getattr(args, "scale", None), seed=getattr(args, "seed", None)),
        )
    print(format_case_summary(result))

    env = gpu_info(device)
    peaks = resolve_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = model_cost(model)
    git = git_info()
    record = build_benchmark_record(
        args=args,
        model=model,
        mode_label=str(target_mode),
        cases=[result],
        correctness=correctness,
        env=env,
        peaks=peaks,
        cost=cost,
        git=git,
        warmup=warmup,
        repeats=repeats,
        engine="scheduler",
    )
    output = args.output or CASE_DIR / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, dirty=git["dirty"])

    print()
    print(format_table([result], suite_name=case_name))
    return 0


@dataclass
class SchedulerRunEnv:
    """一次 scheduler 运行（单个 suite 或 sweep 内多个 suite）共享的模型与环境信息。"""

    model: torch.nn.Module
    tokenizer: Qwen3Tokenizer
    device: torch.device
    vocab: int
    env: dict[str, Any]
    peaks: dict[str, float | None]
    cost: dict[str, Any]
    git: dict[str, Any]
    correctness_cache: dict[Mode, dict[str, Any]] = field(default_factory=dict)


def prepare_scheduler_env(args: argparse.Namespace) -> SchedulerRunEnv:
    """要求 CUDA、加载模型并采集环境信息。"""
    device = require_cuda()
    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    env = gpu_info(device)
    return SchedulerRunEnv(
        model=model,
        tokenizer=tokenizer,
        device=device,
        vocab=model.config.vocab_size,
        env=env,
        peaks=resolve_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops),
        cost=model_cost(model),
        git=git_info(),
    )


def execute_scheduler_suite(
    run_env: SchedulerRunEnv, args: argparse.Namespace, suite: SchedulerSuite, output_path: Path
) -> SuiteRunSummary:
    """在已准备好的环境中运行一个 scheduler suite：正确性检查 → 执行 → 对比 → 落盘 → 打印表格。"""
    modes = suite.modes()
    mode_label = ",".join(str(m) for m in modes)
    for m in modes:
        if m not in run_env.correctness_cache:
            run_env.correctness_cache[m] = check_correctness(
                run_env.model, run_env.tokenizer, m, run_env.device, update_golden=False
            )
    correctness = {str(m): run_env.correctness_cache[m] for m in modes}
    if any(c["passed"] is False for c in correctness.values()) and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    interleave_features = parse_interleave_args(getattr(args, "interleave", None), suite.interleave)

    print(
        f"[bench-scheduler-suite] suite={suite.name} cases={len(suite.blueprints)} "
        f"warmup={warmup} repeats={repeats} ...",
        flush=True,
    )

    def on_suite_step(step_number: int, total_steps: int, planned_run: PlannedRun) -> None:
        rep = f"repeat {planned_run.repeat_idx + 1}" if planned_run.repeat_idx is not None else "warmup"
        case = planned_run.case
        print(
            f"  [{step_number:02d}/{total_steps:02d}] {planned_run.phase.value:<7} | {rep:<10} | {case.key} ...",
            flush=True,
        )

    results = run_suite(
        model=run_env.model,
        device=run_env.device,
        suite=suite,
        vocab_size=run_env.vocab,
        warmup=warmup,
        repeats=repeats,
        interleave=interleave_features,
        on_step_progress=on_suite_step,
    )

    for res in results:
        print(format_case_summary(res))

    comparisons: list[MetricComparison] = []
    if suite.comparison is not None:
        comparisons = compute_comparisons(results, suite.comparison)
        if comparisons:
            print(f"\n--- Comparisons (suite: {suite.name}) ---")
            for comp in comparisons:
                print(format_comparison(comp))

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
        engine="scheduler",
        comparisons=comparisons,
    )
    save_benchmark_record(output_path, record, dirty=run_env.git["dirty"])

    print()
    print(format_table(results, suite_name=suite.name))
    return SuiteRunSummary(
        name=suite.name,
        engine="scheduler",
        num_cases=len(results),
        output=output_path,
        comparison_lines=tuple(format_comparison(c) for c in comparisons),
    )


def print_scheduler_dry_run(args: argparse.Namespace, suite: SchedulerSuite) -> None:
    """生成并打印执行计划（不加载模型、不要求 CUDA）。"""
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    interleave_features = parse_interleave_args(getattr(args, "interleave", None), suite.interleave)
    cases = suite.cases(vocab_size=100)
    plan = build_scheduler_run_plan(cases, warmup=warmup, repeats=repeats, interleave=interleave_features)
    print(f"=== [Dry Run] Suite: {suite.name} ===")
    print(format_scheduler_run_plan(plan))
