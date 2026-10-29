"""CLI 子命令：mini-infra bench release

使用 `mini_infra.bench.workloads` 中预定义的标准真实 Traffic 工作负载
（`LOCKSTEP_WORKLOAD` 与 `SCHEDULER_WORKLOAD`），根据输入的 `--type`（lockstep / scheduler）、
`--tag` 与 `--release-feature` 一键生成版本发布所需的全部基准评测产物：
1. 单标杆历史记录：`benchmarks/result-history/{lockstep,scheduler}/<commit>_<tag>.json`
2. 套件评测记录：`benchmarks/suites/.../<commit>_<tag>.json`（可选）
3. 演进报告与剖析大图：
   `benchmarks/reports/{lockstep,scheduler}/{SUMMARY.md, evolution_and_scaling.png, profiling_and_optimization.png}`
4. 零退化回归参考基线：`benchmarks/data/reference.json`

用法示例：
    # 生成 Scheduler 固定批处理基准线 (baseline: Fixed Batching，仅 Mode 不同，其余 WorkloadSpec 完全一致)
    mini-infra bench release \
        --type scheduler \
        --tag baseline \
        --release-feature "fixed batching" \
        --batching-method fixed-batching \
        --no-suites \
        --skip-regression

    # 发布 v0.4.0 (Scheduler: Continuous Batching)
    mini-infra bench release \
        --type scheduler \
        --tag v0.4.0 \
        --release-feature "continuous batching"

    # 发布 v0.3.0 (Lockstep: Fixed Batching)
    mini-infra bench release \
        --type lockstep \
        --tag v0.3.0 \
        --release-feature "fixed batching" \
        --batching-method fixed-batching
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from mini_infra.bench.common import (
    LOCKSTEP_REPORTS_DIR,
    LOCKSTEP_RESULT_HISTORY_DIR,
    REPO_ROOT,
    SCHEDULER_REPORTS_DIR,
    SCHEDULER_RESULT_HISTORY_DIR,
    SUITES_DIR,
)
from mini_infra.bench.workloads import (
    LOCKSTEP_WORKLOAD,
    SCHEDULER_WORKLOAD,
    WorkloadSpec,
)
from mini_infra.engine import BatchingMethod, FeatureFlag, Mode


@dataclass(frozen=True)
class ReleaseStep:
    """Release 基准流水线中的单个子步骤描述、执行命令与预期产物。"""

    description: str
    cmd: list[str]
    artifacts: tuple[Path, ...]


def build_lockstep_case_cmd(
    workload: WorkloadSpec,
    tag: str,
    release_feature: str,
    profile_mode: str | None = "extra-profile",
) -> list[str]:
    """将 LOCKSTEP_WORKLOAD 转换为 `mini-infra bench lockstep case` CLI 命令参数列表。"""
    prompt_args = (
        [str(workload.prompt_len)] if isinstance(workload.prompt_len, int) else [str(x) for x in workload.prompt_len]
    )
    budget_args = (
        [str(workload.max_new_tokens)]
        if isinstance(workload.max_new_tokens, int)
        else [str(x) for x in workload.max_new_tokens]
    )
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "mini_infra.cli",
        "bench",
        "lockstep",
        "case",
        "--tag",
        tag,
        "--release-feature",
        release_feature,
        "--kv-cache" if workload.mode.is_enabled(FeatureFlag.KV_CACHE) else "--no-kv-cache",
        "--batching-method",
        workload.mode.batching_method.value,
        "--batch-size",
        str(workload.batch_size),
        "--prompt-lens",
        *prompt_args,
        "--max-new-tokens",
        *budget_args,
        "--warmup",
        str(workload.warmup),
        "--repeats",
        str(workload.repeats),
    ]
    if profile_mode and profile_mode != "none":
        cmd.extend(["--profile-mode", profile_mode])
    return cmd


def build_scheduler_case_cmd(
    workload: WorkloadSpec,
    tag: str,
    release_feature: str,
) -> list[str]:
    """将 SCHEDULER_WORKLOAD 转换为 `mini-infra bench scheduler case` CLI 命令参数列表。"""
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "mini_infra.cli",
        "bench",
        "scheduler",
        "case",
        "--tag",
        tag,
        "--release-feature",
        release_feature,
        "--batching-method",
        workload.mode.batching_method.value,
        "--request-count",
        str(workload.batch_size),
        "--max-running-requests",
        str(workload.max_running_requests or 4),
        "--warmup",
        str(workload.warmup),
        "--repeats",
        str(workload.repeats),
    ]
    if isinstance(workload.prompt_len, int):
        cmd.extend(["--prompt-length", str(workload.prompt_len)])
    else:
        cmd.extend(["--prompt-lens", *[str(x) for x in workload.prompt_len]])

    if isinstance(workload.max_new_tokens, int):
        cmd.extend(["--output-budget", str(workload.max_new_tokens)])
    else:
        cmd.extend(["--output-budgets", *[str(x) for x in workload.max_new_tokens]])

    if isinstance(workload.arrival_times_ms, tuple | list):
        cmd.extend(["--arrival-times-ms", *[str(x) for x in workload.arrival_times_ms]])
    elif isinstance(workload.arrival_times_ms, int | float):
        cmd.extend(["--arrival-interval-ms", str(workload.arrival_times_ms)])

    return cmd


def build_suite_cmd(benchmark_type: str, tag: str, suite_name: str) -> list[str]:
    """构建套件评测命令。"""
    return [
        sys.executable,
        "-u",
        "-m",
        "mini_infra.cli",
        "bench",
        benchmark_type,
        "suite",
        "--tag",
        tag,
        "--suite",
        suite_name,
    ]


def build_plot_history_cmd(benchmark_type: str) -> list[str]:
    """构建演进报告与图表生成命令。"""
    return [
        sys.executable,
        "-u",
        "-m",
        "mini_infra.cli",
        "bench",
        "plot",
        "history",
        "--type",
        benchmark_type,
    ]


def build_update_reference_cmd() -> list[str]:
    """构建更新零退化性能基线 reference.json 的命令。"""
    return [
        sys.executable,
        "-u",
        "-m",
        "mini_infra.cli",
        "bench",
        "check-regression",
        "--update-reference",
    ]


def add_release_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 release 基准流水线参数。"""
    p.add_argument(
        "-t",
        "--type",
        dest="benchmark_type",
        choices=["lockstep", "scheduler"],
        required=True,
        help="所属基准演进分支：lockstep（齐步推理）或 scheduler（服务化队列调度）",
    )
    p.add_argument("--tag", required=True, help="发布版本标签，如 baseline、v0.3.0、v0.4.0")
    p.add_argument(
        "--release-feature",
        required=True,
        help="当前版本对应的核心特性描述，如 'fixed batching'、'continuous batching'",
    )
    p.add_argument(
        "--batching-method",
        choices=[m.value for m in BatchingMethod],
        default=None,
        help="覆盖批处理策略（lockstep 默认 fixed-batching；scheduler 默认 continuous-batching）",
    )
    p.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 KV Cache（默认: True）",
    )
    p.add_argument(
        "--profile-mode",
        choices=["extra-profile", "full-profile", "none"],
        default=None,
        help="Profiler 模式（lockstep 默认 extra-profile，scheduler 默认 none）",
    )
    p.add_argument(
        "--suites",
        nargs="*",
        default=None,
        help="附带运行的套件名称列表（scheduler 默认运行 fixed-vs-continuous-variable-output-length）",
    )
    p.add_argument(
        "--no-suites",
        action="store_true",
        help="跳过附带套件运行",
    )
    p.add_argument(
        "--skip-regression",
        action="store_true",
        help="跳过更新 benchmarks/data/reference.json 回归基线",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的命令序列与预期产物路径，不实际运行 GPU 评测",
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="使用预定义 LOCKSTEP_WORKLOAD / SCHEDULER_WORKLOAD 一键生成版本发布所需的基准与报告产物"
    )
    add_release_args(p)
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 release 子命令。"""
    p = subparsers.add_parser(
        "release",
        help="使用预定义真实 Workload 一键生成版本发布基准、演进报告与回归基线",
    )
    add_release_args(p)
    p.set_defaults(func=handle_release)
    return p


def plan_release_steps(args: argparse.Namespace) -> list[ReleaseStep]:
    """根据解析后的参数生成有序的 ReleaseStep 列表（仅替换 Mode，其余 WorkloadSpec 字段保持一致）。"""
    steps: list[ReleaseStep] = []
    flags = {FeatureFlag.KV_CACHE} if args.kv_cache else set()

    if args.benchmark_type == "lockstep":
        bm = BatchingMethod(args.batching_method) if args.batching_method else LOCKSTEP_WORKLOAD.mode.batching_method
        mode = Mode(flags, batching_method=bm)
        workload = replace(LOCKSTEP_WORKLOAD, mode=mode)
        prof = args.profile_mode if args.profile_mode is not None else "extra-profile"

        steps.append(
            ReleaseStep(
                description=f"运行 Lockstep 单标杆基准 ({args.tag}: {args.release_feature})",
                cmd=build_lockstep_case_cmd(workload, args.tag, args.release_feature, profile_mode=prof),
                artifacts=(LOCKSTEP_RESULT_HISTORY_DIR / f"<commit>_{args.tag}.json",),
            )
        )

        suites = [] if args.no_suites else (args.suites or [])
        for s_name in suites:
            steps.append(
                ReleaseStep(
                    description=f"运行 Lockstep 套件 ({s_name})",
                    cmd=build_suite_cmd("lockstep", args.tag, s_name),
                    artifacts=(SUITES_DIR / s_name / f"<commit>_{args.tag}.json",),
                )
            )

        steps.append(
            ReleaseStep(
                description="生成 Lockstep 演进报告与剖析图",
                cmd=build_plot_history_cmd("lockstep"),
                artifacts=(
                    LOCKSTEP_REPORTS_DIR / "SUMMARY.md",
                    LOCKSTEP_REPORTS_DIR / "evolution_and_scaling.png",
                    LOCKSTEP_REPORTS_DIR / "profiling_and_optimization.png",
                ),
            )
        )
    else:
        bm = BatchingMethod(args.batching_method) if args.batching_method else SCHEDULER_WORKLOAD.mode.batching_method
        target_mode = Mode(flags, batching_method=bm)
        target_workload = replace(SCHEDULER_WORKLOAD, mode=target_mode)
        steps.append(
            ReleaseStep(
                description=f"运行 Scheduler 单标杆基准 ({args.tag}: {args.release_feature})",
                cmd=build_scheduler_case_cmd(target_workload, args.tag, args.release_feature),
                artifacts=(SCHEDULER_RESULT_HISTORY_DIR / f"<commit>_{args.tag}.json",),
            )
        )

        default_suites = ["fixed-vs-continuous-variable-output-length"]
        suites = [] if args.no_suites else (args.suites if args.suites is not None else default_suites)
        for s_name in suites:
            steps.append(
                ReleaseStep(
                    description=f"运行 Scheduler 套件 ({s_name})",
                    cmd=build_suite_cmd("scheduler", args.tag, s_name),
                    artifacts=(SUITES_DIR / "scheduler" / s_name / f"<commit>_{args.tag}.json",),
                )
            )

        steps.append(
            ReleaseStep(
                description="生成 Scheduler 演进报告与剖析图",
                cmd=build_plot_history_cmd("scheduler"),
                artifacts=(
                    SCHEDULER_REPORTS_DIR / "SUMMARY.md",
                    SCHEDULER_REPORTS_DIR / "evolution_and_scaling.png",
                    SCHEDULER_REPORTS_DIR / "profiling_and_optimization.png",
                ),
            )
        )

    if not args.skip_regression:
        steps.append(
            ReleaseStep(
                description="更新日常零退化守门基线 (reference.json)",
                cmd=build_update_reference_cmd(),
                artifacts=(REPO_ROOT / "benchmarks" / "data" / "reference.json",),
            )
        )

    return steps


def plan_release_commands(args: argparse.Namespace) -> tuple[list[tuple[str, list[str]]], list[Path]]:
    """兼容接口：返回 (步骤说明与命令列表, 预期产物路径列表)。"""
    steps = plan_release_steps(args)
    cmd_pairs = [(s.description, s.cmd) for s in steps]
    artifacts = [p for s in steps for p in s.artifacts]
    return cmd_pairs, artifacts


def handle_release(args: argparse.Namespace) -> int:
    """执行 Release 基准流水线：在每一步开始时立即打印该步骤信息、命令与目标产物。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    steps = plan_release_steps(args)
    total = len(steps)

    print(
        f"=== Release Benchmark Pipeline (type={args.benchmark_type}, tag={args.tag}, steps={total}) ===",
        flush=True,
    )
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    for idx, step in enumerate(steps, 1):
        print(f"\n[{idx}/{total}] 开始执行: {step.description}", flush=True)
        print("  $", " ".join(step.cmd), flush=True)
        for p in step.artifacts:
            rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
            print(f"  -> 目标产物: {rel}", flush=True)

        if not args.dry_run:
            rc = subprocess.run(step.cmd, cwd=REPO_ROOT, env=child_env).returncode
            if rc != 0:
                print(
                    f"\n[ERROR] [{idx}/{total}] 执行失败 (exit code={rc}): {step.description}",
                    file=sys.stderr,
                    flush=True,
                )
                return rc
            print(f"[{idx}/{total}] 完成: {step.description}", flush=True)

    print(f"\n=== Release Benchmark Pipeline 全部完成 (共 {total} 步) ===", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return handle_release(args)


if __name__ == "__main__":
    raise SystemExit(main())

