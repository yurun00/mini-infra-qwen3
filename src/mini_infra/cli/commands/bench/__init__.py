"""CLI 基准测试与性能工程子命令模块 (mini-infra bench ...)"""

import argparse

from mini_infra.cli.commands.bench import check_regression, lockstep, plot, release, scheduler


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在根命令下注册 bench 组。"""
    bench_parser = subparsers.add_parser("bench", help="基准测试（Lockstep 齐步与 Scheduler 调度）及性能分析")
    bench_subs = bench_parser.add_subparsers(dest="bench_command", required=True)

    lockstep.register_parser(bench_subs)
    scheduler.register_parser(bench_subs)
    plot.register_parser(bench_subs)
    check_regression.register_parser(bench_subs)
    release.register_parser(bench_subs)
    return bench_parser
