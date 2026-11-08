"""suite 子命令共享的 CLI 片段：`--suite NAME` 与 `--spec-file PATH` 二选一，及结果输出路径解析。"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from mini_infra.bench.common import REPO_ROOT, SUITES_DIR
from mini_infra.bench.spec import SuiteEngine


def add_suite_source_args(parser: argparse.ArgumentParser, engine: SuiteEngine, suite_names: Sequence[str]) -> None:
    """添加互斥且必选的 --suite / --spec-file 参数。"""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--suite",
        metavar="NAME",
        choices=list(suite_names),
        help=f"运行 benchmarks/specs/{engine}/ 中声明的命名套件（结果写入 benchmarks/suites/{engine}/<name>/）",
    )
    group.add_argument(
        "--spec",
        metavar="PATH",
        type=Path,
        help="运行任意 suite YAML 文件（临时实验用；结果默认写入 tmp/suites/ 以免污染 benchmarks/）",
    )


def suite_output_path(args: argparse.Namespace, engine: SuiteEngine, suite_name: str, commit: str) -> Path:
    """`--output` 优先；内置套件写入 benchmarks/suites/<engine>/<name>/；`--spec` 写入 tmp/suites/<engine>/<name>/。"""
    if args.output is not None:
        return args.output
    root = REPO_ROOT / "tmp" / "suites" if args.spec is not None else SUITES_DIR
    return root / engine / suite_name / f"{commit}_{args.tag}.json"


__all__ = ["add_suite_source_args", "suite_output_path"]
