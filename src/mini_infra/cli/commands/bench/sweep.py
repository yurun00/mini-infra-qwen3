"""CLI 子命令：mini-infra bench sweep

一次调用顺序运行多个 suite（内置、任意 YAML 或全部内置），共享同一份模型加载、正确性检查与环境信息，
并在结尾汇总各 suite 的产物与对比结论。类比 `vllm bench sweep`：配置层面的 sweep 由 YAML 的 `sweep:` 块声明，
本命令负责「跨 suite 批量运行」与命令行 `--set` 覆盖。
"""

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mini_infra.bench.common import add_common_args
from mini_infra.bench.context import prepare_execution_context
from mini_infra.bench.spec import SuiteEngine, discover_suite_specs, parse_set_args
from mini_infra.bench.suite import BenchmarkSuite
from mini_infra.bench.suites import load_benchmark_suite
from mini_infra.bench.sweep import SuiteRunSummary, format_sweep_summary, run_sweep
from mini_infra.cli.commands.bench.lockstep import (
    execute_lockstep_suite,
    print_lockstep_dry_run,
)
from mini_infra.cli.commands.bench.scheduler import (
    execute_scheduler_suite,
    print_scheduler_dry_run,
)
from mini_infra.cli.commands.bench.suite_source import add_set_arg, suite_output_path, use_suite_execution_defaults


@dataclass(frozen=True)
class _Target:
    path: Path
    builtin: bool


def add_sweep_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_common_args(parser)
    use_suite_execution_defaults(parser)
    parser.add_argument("-e", "--engine", choices=["lockstep", "scheduler"], required=True, help="suite 所属引擎")
    parser.add_argument("--suite", nargs="+", metavar="NAME", default=[], help="要运行的内置 suite 名称（可多个）")
    parser.add_argument("--spec", nargs="+", metavar="PATH", type=Path, default=[], help="要运行的 suite YAML 路径")
    parser.add_argument("--all", action="store_true", help="运行该引擎的全部内置 suite")
    add_set_arg(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "结果根目录，写入 <dir>/<engine>/<suite>/<commit>_<tag>.json"
            "（默认沿用 suite 落盘约定；单个 suite 时可用 --output 指定精确路径）"
        ),
    )
    parser.add_argument(
        "--interleave",
        nargs="*",
        default=None,
        help="覆盖 scheduler suite 的交替维度（none 关闭；默认使用各 suite 的配置）；lockstep 忽略",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅展开并打印各 suite 的计划，不加载模型也不要求 CUDA")
    return parser


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser("sweep", help="一次调用顺序运行多个 suite 并汇总（共享模型加载与正确性检查）")
    add_sweep_args(p)
    p.set_defaults(func=handle_sweep)
    return p


def resolve_targets(args: argparse.Namespace) -> list[_Target]:
    """把 --suite / --spec / --all 解析为有序、去重后的运行目标；非法组合直接退出。"""
    chosen = [bool(args.suite), bool(args.spec), args.all]
    if sum(chosen) != 1:
        sys.exit("必须且只能指定 --suite、--spec、--all 之一")
    builtin = discover_suite_specs(args.engine)
    if args.all:
        return [_Target(path, True) for path in builtin.values()]
    if args.suite:
        unknown = [name for name in args.suite if name not in builtin]
        if unknown:
            sys.exit(f"未知 {args.engine} suite: {unknown}，可选: {sorted(builtin)}")
        return [_Target(builtin[name], True) for name in args.suite]
    return [_Target(path, False) for path in args.spec]


def load_suites(engine: SuiteEngine, targets: Sequence[_Target], overrides: dict[str, object]) -> list[BenchmarkSuite]:
    """加载全部目标（任何一个非法都在运行前暴露）；suite 名称必须唯一，否则结果文件会互相覆盖。"""
    suites = [load_benchmark_suite(target.path, overrides, expected_engine=engine) for target in targets]
    names = [suite.name for suite in suites]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        sys.exit(f"sweep 中的 suite 名称必须唯一，重复: {duplicated}")
    return suites


def handle_sweep(args: argparse.Namespace) -> int:
    if args.output is not None and args.output_dir is not None:
        sys.exit("不能同时指定 --output 与 --output-dir")
    overrides = parse_set_args(args.overrides)
    targets = resolve_targets(args)
    suites = load_suites(args.engine, targets, overrides)
    if args.output is not None and len(suites) > 1:
        sys.exit("--output 仅支持单个 suite，多个 suite 请使用 --output-dir")

    if args.dry_run:
        for suite in suites:
            if suite.engine == "scheduler":
                print_scheduler_dry_run(args, suite)
            else:
                print_lockstep_dry_run(args, suite)
        return 0

    experimental_by_suite = {id(s): (not t.builtin or bool(overrides)) for s, t in zip(suites, targets)}

    ctx = prepare_execution_context(args)
    commit = ctx.commit

    def run_one(suite: BenchmarkSuite, output: Path) -> SuiteRunSummary:
        if args.engine == "scheduler":
            assert suite.engine == "scheduler"
            return execute_scheduler_suite(ctx, args, suite, output)
        assert suite.engine == "lockstep"
        return execute_lockstep_suite(ctx, args, suite, output)

    def execute(suite: BenchmarkSuite) -> SuiteRunSummary:
        if args.output is not None:
            output = args.output
        elif args.output_dir is not None:
            output = args.output_dir / args.engine / suite.name / f"{commit}_{args.tag}.json"
        else:
            output = suite_output_path(
                args, args.engine, suite.name, commit, experimental=experimental_by_suite[id(suite)]
            )
        return run_one(suite, output)

    def on_start(index: int, total: int, suite: BenchmarkSuite) -> None:
        print(f"\n===== [sweep {index}/{total}] {suite.name} =====", flush=True)

    summaries = run_sweep(suites, execute, on_start=on_start)
    print()
    print(format_sweep_summary(summaries))
    return 0
