"""mini-infra 统一根 CLI 入口。

组织结构：
    mini-infra run [...]
    mini-infra bench lockstep case / suite [...]
    mini-infra bench scheduler case / suite [...]
    mini-infra bench plot history / compare [...]
    mini-infra bench check-regression [...]
    mini-infra verify smoke [...]
    mini-infra verify transformers / cache / batch [...]
"""

import argparse
import sys
from collections.abc import Sequence

from mini_infra.cli.commands import bench, run, verify


def build_parser() -> argparse.ArgumentParser:
    """构建 mini-infra 统一命令行根解析器。"""
    parser = argparse.ArgumentParser(
        prog="mini-infra",
        description="mini-infra-qwen3: 手写 Qwen3 推理与性能工程统一 CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run.register_parser(subparsers)
    bench.register_parser(subparsers)
    verify.register_parser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = build_parser()
    if argv is None and len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    ret = args.func(args)
    code = ret if isinstance(ret, int) else 0
    if code != 0:
        sys.exit(code)
    return 0


if __name__ == "__main__":
    main()
