"""CLI 验证子命令模块 (mini-infra verify ...)"""

import argparse

from mini_infra.cli.commands.verify import parity, smoke


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在根命令下注册 verify 组。"""
    verify_parser = subparsers.add_parser("verify", help="快速冒烟与数值对齐测试")
    verify_subs = verify_parser.add_subparsers(dest="verify_command", required=True)

    smoke.register_parser(verify_subs)
    parity.register_parser(verify_subs)
    return verify_parser
