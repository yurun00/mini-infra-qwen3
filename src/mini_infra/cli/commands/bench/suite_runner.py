"""CLI 套件运行与 Dry-Run 预览驱动器（供 `mini-infra bench sweep` 统一调用）。"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mini_infra.bench.cases import BenchmarkCase, CaseDimension
from mini_infra.bench.common import (
    CorrectnessReport,
    check_correctness,
    format_lengths,
    save_benchmark_record,
)
from mini_infra.bench.comparison import ComparisonResult, compute_comparisons
from mini_infra.bench.context import BenchmarkExecutionContext
from mini_infra.bench.lockstep.format import (
    format_suite_plan,
)
from mini_infra.bench.lockstep.format import (
    format_table as format_lockstep_table,
)
from mini_infra.bench.lockstep.run_suite import run_suite as run_lockstep_suite
from mini_infra.bench.scheduler.format import format_table as format_scheduler_table
from mini_infra.bench.scheduler.plan import PlannedRun, build_scheduler_run_plan, format_scheduler_run_plan
from mini_infra.bench.scheduler.run_suite import run_suite as run_scheduler_suite
from mini_infra.bench.schema import build_benchmark_record
from mini_infra.bench.suite import BenchmarkSuite
from mini_infra.bench.sweep import SuiteRunSummary
from mini_infra.engine import Mode
from mini_infra.profiling import TensorBoardOptions


def _ensure_suite_correctness(
    ctx: BenchmarkExecutionContext,
    modes: Sequence[Mode],
    allow_mismatch: bool,
) -> dict[str, CorrectnessReport | dict[str, Any]]:
    """按需检查并缓存各运行模式的输出正确性；若存在不一致且未开启 allow_mismatch 则退出进程。"""
    for m in modes:
        if m not in ctx.correctness_cache:
            assert ctx.tokenizer is not None
            ctx.correctness_cache[m] = check_correctness(ctx.model, ctx.tokenizer, m, ctx.device, update_golden=False)
    correctness = {str(m): ctx.correctness_cache[m] for m in modes}
    if any(c["passed"] is False for c in correctness.values()) and not allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")
    return correctness


def execute_lockstep_suite(
    ctx: BenchmarkExecutionContext, args: argparse.Namespace, suite: BenchmarkSuite, output_path: Path
) -> SuiteRunSummary:
    """在已准备好的环境中运行一个 lockstep suite：正确性检查 → 执行 → 落盘 → 打印表格。"""
    modes = suite.modes()
    cases = suite.cases
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    mode_label = ",".join(str(m) for m in modes)
    correctness = _ensure_suite_correctness(ctx, modes, allow_mismatch=args.allow_mismatch)

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

    results = run_lockstep_suite(
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
    print(format_lockstep_table(results))
    return SuiteRunSummary(name=suite.name, engine="lockstep", num_cases=len(results), output=output_path)


def print_lockstep_dry_run(args: argparse.Namespace, suite: BenchmarkSuite) -> None:
    """展开并打印用例列表（不加载模型、不要求 CUDA）。"""
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    print(f"=== [Dry Run] Suite: {suite.name} ===")
    print(format_suite_plan(suite.name, suite.cases, warmup, repeats))


def execute_scheduler_suite(
    ctx: BenchmarkExecutionContext, args: argparse.Namespace, suite: BenchmarkSuite, output_path: Path
) -> SuiteRunSummary:
    """在已准备好的环境中运行一个 scheduler suite：正确性检查 → 执行 → 对比 → 落盘 → 打印表格。"""
    modes = suite.modes()
    mode_label = ",".join(str(m) for m in modes)
    correctness = _ensure_suite_correctness(ctx, modes, allow_mismatch=args.allow_mismatch)

    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    interleave_dims = CaseDimension.from_interleave_args(getattr(args, "interleave", None), suite.interleave)

    print(
        f"[bench-scheduler-suite] suite={suite.name} cases={len(suite.cases)} warmup={warmup} repeats={repeats} ...",
        flush=True,
    )

    def on_suite_step(step_number: int, total_steps: int, planned_run: PlannedRun) -> None:
        rep = f"repeat {planned_run.repeat_idx + 1}" if planned_run.repeat_idx is not None else "warmup"
        case = planned_run.case
        print(
            f"  [{step_number:02d}/{total_steps:02d}] {planned_run.phase.value:<7} | {rep:<10} | {case.key} ...",
            flush=True,
        )

    results = run_scheduler_suite(
        model=ctx.model,
        device=ctx.device,
        suite=suite,
        vocab_size=ctx.vocab,
        warmup=warmup,
        repeats=repeats,
        interleave=interleave_dims,
        on_step_progress=on_suite_step,
    )

    for res in results:
        print(res.format_summary())

    comparisons: list[ComparisonResult] = []
    if suite.comparison is not None:
        comparisons = compute_comparisons(results, suite.comparison)
        if comparisons:
            print(f"\n--- Comparisons (suite: {suite.name}) ---")
            for comp in comparisons:
                print(comp.format_summary())

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
        engine="scheduler",
        comparisons=comparisons,
    )
    save_benchmark_record(output_path, record, dirty=ctx.dirty)

    print()
    print(format_scheduler_table(results, suite_name=suite.name))
    return SuiteRunSummary(
        name=suite.name,
        engine="scheduler",
        num_cases=len(results),
        output=output_path,
        comparison_lines=tuple(c.format_summary() for c in comparisons),
    )


def print_scheduler_dry_run(args: argparse.Namespace, suite: BenchmarkSuite) -> None:
    """生成并打印执行计划（不加载模型、不要求 CUDA）。"""
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    interleave_dims = CaseDimension.from_interleave_args(getattr(args, "interleave", None), suite.interleave)
    cases = suite.cases
    plan = build_scheduler_run_plan(cases, warmup=warmup, repeats=repeats, interleave=interleave_dims)
    print(f"=== [Dry Run] Suite: {suite.name} ===")
    print(format_scheduler_run_plan(plan))


__all__ = [
    "execute_lockstep_suite",
    "execute_scheduler_suite",
    "print_lockstep_dry_run",
    "print_scheduler_dry_run",
]
