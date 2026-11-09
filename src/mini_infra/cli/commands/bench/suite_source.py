"""suite 相关的 CLI 辅助函数：--set 覆盖参数、YAML 默认执行参数与结果输出路径解析。"""

import argparse
from pathlib import Path

from mini_infra.bench.common import REPO_ROOT, SUITES_DIR
from mini_infra.bench.spec import SuiteEngine


def add_set_arg(parser: argparse.ArgumentParser) -> None:
    """添加可重复的 --set KEY=VALUE 覆盖参数（解析见 bench.spec.overrides）。"""
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖 suite 声明中的字段，KEY 为点分路径、VALUE 按 YAML 解析（可重复），如 --set execution.repeats=5；"
        "带覆盖的运行视为实验，结果默认写入 tmp/suites/",
    )


def use_suite_execution_defaults(parser: argparse.ArgumentParser) -> None:
    """suite 的 warmup / repeats 由 YAML 的 execution 声明；CLI 的 --warmup / --repeats 仅在显式传入时覆盖。"""
    for action in parser._actions:
        if action.dest in ("warmup", "repeats"):
            action.default = None
            action.help = f"覆盖 suite 声明的 {action.dest}（默认: 取 YAML execution.{action.dest}）"


def suite_output_path(
    args: argparse.Namespace, engine: SuiteEngine, suite_name: str, commit: str, *, experimental: bool
) -> Path:
    """`--output` 优先；内置套件写入 benchmarks/suites/<engine>/<name>/；

    experimental（`--spec` 或带 `--set` 覆盖的非标准配置）写入被 .gitignore 忽略的 tmp/suites/<engine>/<name>/，
    以免污染受控的 benchmarks/。
    """
    if args.output is not None:
        return args.output
    root = REPO_ROOT / "tmp" / "suites" if experimental else SUITES_DIR
    return root / engine / suite_name / f"{commit}_{args.tag}.json"


__all__ = ["add_set_arg", "suite_output_path", "use_suite_execution_defaults"]
