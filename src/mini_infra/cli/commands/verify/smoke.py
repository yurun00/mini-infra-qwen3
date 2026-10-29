"""CLI 子命令：mini-infra verify smoke

基于微型配置模型在 CPU 上进行秒级快速冒烟自检。
"""

import argparse

from mini_infra.verification.smoke import run_smoke_test


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """注册 mini-infra verify smoke 子命令。"""
    parser = subparsers.add_parser("smoke", help="基于微型模型的 CPU 快速冒烟自检")
    parser.set_defaults(func=handle_smoke)
    return parser


def handle_smoke(args: argparse.Namespace) -> int:
    run_smoke_test()
    print("smoke test passed")
    return 0
