"""CLI 子命令：mini-infra bench release

使用 `mini_infra.bench.spec.release` 加载的多规模伪随机真实 Traffic 工作负载
（声明于 `benchmarks/specs/release/{lockstep,scheduler}.yaml`，含 small / medium / large 三档），
根据输入的 `--engine`（lockstep / scheduler）、`--tag` 与 `--release-feature` 一键生成版本发布所需的全部基准评测产物：
1. 多规模历史记录：`benchmarks/results/history/{lockstep,scheduler}/<commit>_<tag>.json`（含 small/medium/large 三档）
2. 演进报告与剖析大图：
   `benchmarks/reports/{lockstep,scheduler}/{SUMMARY.md, evolution_and_scaling.png, profiling_and_optimization.png}`
3. 零退化回归参考基线：`benchmarks/golden/reference.json`

用法示例：
    # 生成 Scheduler 固定批处理基准线 (baseline: Fixed Batching，仅 Mode 不同，其余 Workload 完全一致)
    mini-infra bench release \
        --engine scheduler \
        --tag baseline \
        --release-feature "fixed batching" \
        --batching-method fixed-batching \
        --skip-regression

    # 发布 v0.5.0 (Scheduler: Paged KV Cache)
    mini-infra bench release \
        --engine scheduler \
        --tag v0.5.0 \
        --release-feature "paged kv cache" \
        --caching-method paged-kv-cache

    # 发布 v0.4.0 (Scheduler: Continuous Batching)
    mini-infra bench release \
        --engine scheduler \
        --tag v0.4.0 \
        --release-feature "continuous batching"

    # 发布 v0.3.0 (Lockstep: Fixed Batching)
    mini-infra bench release \
        --engine lockstep \
        --tag v0.3.0 \
        --release-feature "fixed batching" \
        --batching-method fixed-batching
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mini_infra.bench.cases import WorkloadProfile
from mini_infra.bench.env import git_info
from mini_infra.bench.paths import (
    LOCKSTEP_REPORTS_DIR,
    LOCKSTEP_RESULT_HISTORY_DIR,
    REPO_ROOT,
    SCHEDULER_REPORTS_DIR,
    SCHEDULER_RESULT_HISTORY_DIR,
)
from mini_infra.bench.spec import RELEASE_SCALES, ReleaseScale, get_release_spec
from mini_infra.engine import BatchingMethod, CachingMethod, Mode


@dataclass(frozen=True)
class _ReleaseStep:
    """Release 基准流水线中的单个子步骤描述、执行命令与预期产物。"""

    description: str
    cmd: list[str]
    artifacts: tuple[Path, ...]


def _build_lockstep_case_cmd(
    workload: WorkloadProfile,
    tag: str,
    release_feature: str,
    mode: Mode,
    warmup: int = 2,
    repeats: int = 5,
    scale: str | None = None,
    seed: int | None = None,
    profiling_mode: str | None = "dedicated-pass",
) -> list[str]:
    """将 WorkloadProfile 转换为 `mini-infra bench lockstep case` CLI 命令参数列表。"""
    prompt_args = [str(x) for x in workload.prompt_lens]
    budget_args = [str(x) for x in workload.output_budgets]
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
        "--kv-cache" if mode.use_kv_cache else "--no-kv-cache",
        "--batching-method",
        mode.batching_method.value,
        "--batch-size",
        str(workload.num_total_requests),
        "--prompt-lens",
        *prompt_args,
        "--max-new-tokens",
        *budget_args,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    if scale is not None:
        cmd.extend(["--scale", str(scale)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if profiling_mode and profiling_mode != "none":
        cmd.extend(["--profiling-mode", profiling_mode])
    return cmd


def _build_scheduler_case_cmd(
    workload: WorkloadProfile,
    tag: str,
    release_feature: str,
    mode: Mode,
    max_running_requests: int = 4,
    warmup: int = 2,
    repeats: int = 5,
    scale: str | None = None,
    seed: int | None = None,
    block_size: int | None = None,
    output: Path | str | None = None,
) -> list[str]:
    """将 WorkloadProfile 转换为 `mini-infra bench scheduler case` CLI 命令参数列表。"""
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
        "--caching-method",
        mode.caching_method.value,
        "--batching-method",
        mode.batching_method.value,
        "--request-count",
        str(workload.num_total_requests),
        "--max-running-requests",
        str(max_running_requests),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
    ]
    if mode.caching_method == CachingMethod.PAGED and block_size is not None:
        cmd.extend(["--block-size", str(block_size)])
    if scale is not None:
        cmd.extend(["--scale", str(scale)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    cmd.extend(["--prompt-lens", *[str(x) for x in workload.prompt_lens]])
    cmd.extend(["--output-budgets", *[str(x) for x in workload.output_budgets]])
    cmd.extend(["--arrival-times-ms", *[str(x) for x in workload.arrival_ms]])

    if output is not None:
        cmd.extend(["--output", str(output)])

    return cmd


def _build_plot_history_cmd(engine: str) -> list[str]:
    """构建演进报告与图表生成命令。"""
    return [
        sys.executable,
        "-u",
        "-m",
        "mini_infra.cli",
        "bench",
        "plot",
        "history",
        "--engine",
        engine,
    ]


def _build_update_reference_cmd() -> list[str]:
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


def _add_release_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 release 基准流水线参数。"""
    p.add_argument(
        "-e",
        "--engine",
        dest="engine",
        choices=["lockstep", "scheduler"],
        required=True,
        help="所属基准引擎分支：lockstep（齐步推理）或 scheduler（服务化队列调度）",
    )
    p.add_argument("--tag", required=True, help="发布版本标签，如 baseline、v0.3.0、v0.4.0")
    p.add_argument(
        "--release-feature",
        required=True,
        help="当前版本对应的核心特性描述，如 'fixed batching'、'continuous batching'",
    )
    p.add_argument(
        "--scales",
        nargs="+",
        choices=list(RELEASE_SCALES),
        default=list(RELEASE_SCALES),
        help="要运行的负载规模档位（默认: small medium large）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="覆盖伪随机种子（默认使用 benchmarks/specs/release/<engine>.yaml 的 default_seed）",
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
        "--caching-method",
        choices=[m.value for m in CachingMethod],
        default=None,
        help="覆盖 KV Cache 策略（默认根据 --kv-cache 为 dense-kv-cache 或 none）",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Paged KV Cache 的 physical block 大小（默认: 16，仅 paged-kv-cache 时生效）",
    )
    p.add_argument(
        "--profiling-mode",
        choices=["dedicated-pass", "all-repeats", "none"],
        default=None,
        help="Profiler 模式（lockstep 默认仅在 small 规模启用 dedicated-pass，scheduler 默认 none）",
    )
    p.add_argument(
        "--skip-regression",
        action="store_true",
        help="跳过更新 benchmarks/golden/reference.json 回归基线",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要执行的命令序列与预期产物路径，不实际运行 GPU 评测",
    )
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 release 子命令。"""
    p = subparsers.add_parser(
        "release",
        help="使用预定义多规模真实 Workload 一键生成版本发布基准、演进报告与回归基线",
    )
    _add_release_args(p)
    p.set_defaults(func=_handle_release)
    return p


def _plan_release_steps(args: argparse.Namespace) -> list[_ReleaseStep]:
    """根据解析后的参数生成有序的 _ReleaseStep 列表（仅替换 Mode，其余 Workload 字段保持一致）。"""
    steps: list[_ReleaseStep] = []
    if getattr(args, "caching_method", None):
        caching_method = CachingMethod(args.caching_method)
    else:
        caching_method = CachingMethod.DENSE if args.kv_cache else CachingMethod.NONE
    scales: list[ReleaseScale] = list(getattr(args, "scales", None) or RELEASE_SCALES)
    custom_seed: int | None = getattr(args, "seed", None)

    if args.engine == "lockstep":
        loaded = get_release_spec("lockstep")
        default_bm = loaded.default_mode.batching_method
        bm = BatchingMethod(args.batching_method) if args.batching_method else default_bm
        mode = Mode(caching_method, batching_method=bm)
        prof = args.profiling_mode if args.profiling_mode is not None else "dedicated-pass"

        scales_spec = loaded.resolve(custom_seed)
        for sc in scales:
            scale_prof = prof if sc == "small" else None
            res_scale = scales_spec[sc]
            steps.append(
                _ReleaseStep(
                    description=f"运行 Lockstep 基准 [{sc}] ({args.tag}: {args.release_feature})",
                    cmd=_build_lockstep_case_cmd(
                        res_scale.workload.to_profile(),
                        args.tag,
                        args.release_feature,
                        mode=mode,
                        warmup=loaded.spec.execution.warmup,
                        repeats=loaded.spec.execution.repeats,
                        scale=sc,
                        seed=res_scale.seed,
                        profiling_mode=scale_prof,
                    ),
                    artifacts=(LOCKSTEP_RESULT_HISTORY_DIR / f"<commit>_{args.tag}.json",),
                )
            )

        steps.append(
            _ReleaseStep(
                description="生成 Lockstep 演进报告与剖析图",
                cmd=_build_plot_history_cmd("lockstep"),
                artifacts=(
                    LOCKSTEP_REPORTS_DIR / "SUMMARY.md",
                    LOCKSTEP_REPORTS_DIR / "evolution_and_scaling.png",
                    LOCKSTEP_REPORTS_DIR / "profiling_and_optimization.png",
                ),
            )
        )
    else:
        loaded = get_release_spec("scheduler")
        default_bm = loaded.default_mode.batching_method
        bm = BatchingMethod(args.batching_method) if args.batching_method else default_bm
        target_mode = Mode(caching_method, batching_method=bm)
        block_size = getattr(args, "block_size", None)
        git = git_info()
        hist_file = SCHEDULER_RESULT_HISTORY_DIR / f"{git['commit']}_{args.tag}.json"

        scales_spec = loaded.resolve(custom_seed)
        for sc in scales:
            res_scale = scales_spec[sc]
            steps.append(
                _ReleaseStep(
                    description=f"运行 Scheduler 基准 [{sc}] ({args.tag}: {args.release_feature})",
                    cmd=_build_scheduler_case_cmd(
                        res_scale.workload.to_profile(),
                        args.tag,
                        args.release_feature,
                        mode=target_mode,
                        max_running_requests=res_scale.max_running_requests or 4,
                        warmup=loaded.spec.execution.warmup,
                        repeats=loaded.spec.execution.repeats,
                        scale=sc,
                        seed=res_scale.seed,
                        block_size=block_size,
                        output=hist_file,
                    ),
                    artifacts=(SCHEDULER_RESULT_HISTORY_DIR / f"<commit>_{args.tag}.json",),
                )
            )

        steps.append(
            _ReleaseStep(
                description="生成 Scheduler 演进报告与剖析图",
                cmd=_build_plot_history_cmd("scheduler"),
                artifacts=(
                    SCHEDULER_REPORTS_DIR / "SUMMARY.md",
                    SCHEDULER_REPORTS_DIR / "evolution_and_scaling.png",
                    SCHEDULER_REPORTS_DIR / "profiling_and_optimization.png",
                ),
            )
        )

    if not args.skip_regression:
        steps.append(
            _ReleaseStep(
                description="更新日常零退化守门基线 (reference.json)",
                cmd=_build_update_reference_cmd(),
                artifacts=(REPO_ROOT / "benchmarks" / "golden" / "reference.json",),
            )
        )

    return steps


def _handle_release(args: argparse.Namespace) -> int:
    """执行 Release 基准流水线：在每一步开始时立即打印该步骤信息、命令与目标产物。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    steps = _plan_release_steps(args)
    total = len(steps)

    print(
        f"=== Release Benchmark Pipeline (engine={args.engine}, tag={args.tag}, steps={total}) ===",
        flush=True,
    )
    child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    for step_number, step in enumerate(steps, 1):
        print(f"\n[{step_number}/{total}] 开始执行: {step.description}", flush=True)
        print("  $", " ".join(step.cmd), flush=True)
        for p in step.artifacts:
            rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
            print(f"  -> 目标产物: {rel}", flush=True)

        if not args.dry_run:
            rc = subprocess.run(step.cmd, cwd=REPO_ROOT, env=child_env).returncode
            if rc != 0:
                print(
                    f"\n[ERROR] [{step_number}/{total}] 执行失败 (exit code={rc}): {step.description}",
                    file=sys.stderr,
                    flush=True,
                )
                return rc
            print(f"[{step_number}/{total}] 完成: {step.description}", flush=True)

    print(f"\n=== Release Benchmark Pipeline 全部完成 (共 {total} 步) ===", flush=True)
    return 0
