"""CLI 子命令：mini-infra bench scheduler (case / suite)

队列调度器（Scheduler）评测套件与用例执行入口。
"""

import argparse
import sys
from dataclasses import replace

import torch

from mini_infra.bench.common import (
    BENCH_ARTIFACTS_DIR,
    SCHEDULER_RESULT_HISTORY_DIR,
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
    compute_continuous_makespan_speedup,
    format_case_summary,
    format_table,
)
from mini_infra.bench.scheduler.plan import PlannedRun, build_scheduler_run_plan, format_scheduler_run_plan
from mini_infra.bench.scheduler.run_case import run_case
from mini_infra.bench.scheduler.run_suite import (
    parse_interleave_args,
    run_suite,
)
from mini_infra.bench.scheduler.suites import (
    SCHEDULER_SUITES,
    SchedulerSuiteCase,
    get_suite,
)
from mini_infra.bench.schema import build_benchmark_record
from mini_infra.bench.workloads import WorkloadSpec, build_scheduler_request_specs, make_arrival_times
from mini_infra.engine import BatchingMethod, FeatureFlag, Mode
from mini_infra.model import Qwen3Tokenizer, load_model, resolve_model_directory


def add_scheduler_case_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 scheduler case 参数。"""
    add_common_args(p)
    p.add_argument(
        "--batching-method",
        choices=[BatchingMethod.CONTINUOUS.value, BatchingMethod.FIXED.value],
        default=BatchingMethod.CONTINUOUS.value,
        help=f"执行的批处理调度策略（默认: {BatchingMethod.CONTINUOUS.value}）",
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


def add_scheduler_suite_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 scheduler suite 参数。"""
    add_common_args(p)
    p.add_argument(
        "--suite",
        choices=sorted(SCHEDULER_SUITES),
        required=True,
        help="运行 suites.py 中注册的命名 Scheduler 套件（结果写入 benchmarks/suites/scheduler/<name>/）",
    )
    p.add_argument(
        "--interleave",
        nargs="*",
        default=None,
        help="选择交替轮换的特征维度（如 batching_method kv-cache，或 none 关闭交替；默认使用 Suite 的配置）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成并打印执行计划，不加载模型也不要求 CUDA",
    )
    return p


def build_case_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行单用例 Scheduler 基准测试")
    add_scheduler_case_args(p)
    return p


def build_suite_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行预设 Scheduler 套件")
    add_scheduler_suite_args(p)
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 scheduler 组及 case/suite 子命令。"""
    scheduler_parser = subparsers.add_parser("scheduler", help="队列调度器（Scheduler: Fixed vs Continuous）基准测试")
    scheduler_subs = scheduler_parser.add_subparsers(dest="scheduler_subcommand", required=True)

    case_parser = scheduler_subs.add_parser("case", help="运行单用例 Scheduler 基准测试")
    add_scheduler_case_args(case_parser)
    case_parser.set_defaults(func=handle_case)

    suite_parser = scheduler_subs.add_parser("suite", help="运行预设 Scheduler 套件")
    add_scheduler_suite_args(suite_parser)
    suite_parser.set_defaults(func=handle_suite)
    return scheduler_parser


def handle_case(args: argparse.Namespace) -> int:
    device = require_cuda()

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    vocab = model.config.vocab_size

    method = BatchingMethod(args.batching_method)
    target_mode = Mode({FeatureFlag.KV_CACHE}, batching_method=method)

    correctness = check_correctness(model, tokenizer, target_mode, device, update_golden=False)
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

    workload = WorkloadSpec(
        mode=target_mode,
        batch_size=req_count,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        repeats=repeats,
        warmup=warmup,
        max_running_requests=max_running,
        arrival_times_ms=arrivals,
    )
    specs = tuple(build_scheduler_request_specs(workload, vocab, device=device))
    case_name = "custom"

    case_variant_id = "continuous" if method == BatchingMethod.CONTINUOUS else "fixed"
    case = SchedulerSuiteCase(
        id=case_variant_id,
        suite_name=case_name,
        mode=target_mode,
        max_running_requests=max_running,
        specs=specs,
    )

    print(
        f"[bench-scheduler-case] case={case_name} mode={target_mode} requests={len(specs)} "
        f"max_running={max_running} warmup={warmup} repeats={repeats} ...",
        flush=True,
    )

    def on_case_step(phase: str, idx: int, total: int) -> None:
        print(f"  [{phase} {idx}/{total}] mode={target_mode} requests={len(specs)} ...", flush=True)

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
    output = args.output or SCHEDULER_RESULT_HISTORY_DIR / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, dirty=git["dirty"])

    print()
    print(format_table([result], suite_name=case_name))
    return 0


def handle_suite(args: argparse.Namespace) -> int:
    suite = get_suite(args.suite)

    if args.dry_run:
        warmup = args.warmup if args.warmup is not None else suite.warmup
        repeats = args.repeats if args.repeats is not None else suite.repeats
        interleave_features = parse_interleave_args(args.interleave, suite.interleave)
        cases = suite.cases(vocab_size=100, device=torch.device("cpu"))
        plan = build_scheduler_run_plan(cases, warmup=warmup, repeats=repeats, interleave=interleave_features)
        print(f"=== [Dry Run] Suite: {args.suite} ===")
        print(format_scheduler_run_plan(plan))
        return 0

    device = require_cuda()

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    vocab = model.config.vocab_size

    modes = suite.modes()
    mode_label = ",".join(str(m) for m in modes)
    correctness = {str(m): check_correctness(model, tokenizer, m, device, update_golden=False) for m in modes}
    if any(c["passed"] is False for c in correctness.values()) and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info(device)
    peaks = resolve_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = model_cost(model)
    git = git_info()

    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    interleave_features = parse_interleave_args(args.interleave, suite.interleave)

    print(
        f"[bench-scheduler-suite] suite={suite.name} max_running={suite.max_running_requests} "
        f"warmup={warmup} repeats={repeats} ...",
        flush=True,
    )

    def on_suite_step(idx: int, total: int, planned_run: PlannedRun) -> None:
        rep = f"repeat {planned_run.repeat_index + 1}" if planned_run.repeat_index is not None else "warmup"
        case = planned_run.case
        print(
            f"  [{idx:02d}/{total:02d}] {planned_run.phase.value:<7} | {rep:<10} | "
            f"case_id={case.id} mode={case.mode} max_running={case.max_running_requests} ...",
            flush=True,
        )

    results = run_suite(
        model=model,
        device=device,
        suite=suite,
        vocab_size=vocab,
        warmup=warmup,
        repeats=repeats,
        interleave=interleave_features,
        on_step_progress=on_suite_step,
    )

    speedup = compute_continuous_makespan_speedup(results)
    for res in results:
        print(format_case_summary(res, speedup=speedup))

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
        engine="scheduler",
    )

    scheduler_dir = BENCH_ARTIFACTS_DIR / "suites" / "scheduler"
    output_path = args.output or (scheduler_dir / args.suite / f"{git['commit']}_{args.tag}.json")
    save_benchmark_record(output_path, record, dirty=git["dirty"])

    print()
    print(format_table(results, suite_name=suite.name))
    return 0
