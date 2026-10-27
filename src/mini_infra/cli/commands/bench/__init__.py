"""CLI 基准测试与性能工程子命令模块 (mini-infra bench ...)"""

import argparse

from mini_infra.cli.commands.bench import batch, check_regression, plot, serving


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在根命令下注册 bench 组。"""
    bench_parser = subparsers.add_parser("bench", help="基准测试、Serving 调度与性能分析")
    bench_subs = bench_parser.add_subparsers(dest="bench_command", required=True)

    batch.register_parser(bench_subs)
    serving.register_parser(bench_subs)
    plot.register_parser(bench_subs)
    check_regression.register_parser(bench_subs)
    return bench_parser
