"""CLI 子命令：mini-infra verify (transformers / cache / lockstep-batch / scheduler-batch)

数值对齐与请求隔离验证。
"""

import argparse

from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)
from mini_infra.verification.parity.compare_cache import run_compare_cache
from mini_infra.verification.parity.compare_lockstep_batch import execute_compare_lockstep_batch
from mini_infra.verification.parity.compare_scheduler_batch import execute_compare_scheduler_batch
from mini_infra.verification.parity.compare_transformers import run_compare_transformers


def add_transformers_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本")
    parser.add_argument("--prompt", default="请用一句话介绍 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--logits-atol", type=float, default=0.25, help="Logits 最大绝对误差阈值")
    return parser


def add_cache_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本")
    parser.add_argument("--prompt", default="请用一句话介绍 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="正确性验证默认使用 float32，避免低精度路径差异干扰判断",
    )
    parser.add_argument("--logits-atol", type=float, default=1e-4, help="允许的最大 Logits 绝对误差")
    return parser


def add_lockstep_batch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本")
    parser.add_argument("--prompt", default="用一句话解释固定批处理。")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="生成的最大 Token 数")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="正确性验证默认使用 float32，避免低精度路径差异干扰判断",
    )
    parser.add_argument("--logits-atol", type=float, default=1e-4, help="允许的最大 Logits 绝对误差")
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser


def add_scheduler_batch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="正确性验证默认使用 float32，避免低精度路径差异干扰判断",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8, help="生成的最大 Token 数（至少为 2，默认: 8）")
    parser.add_argument("--max-running-requests", type=int, default=2, help="调度器最大并发请求数（默认: 2）")
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """注册 parity 相关验证命令，直接挂载到 verify 下，同时也挂载到 verify parity 下。"""
    # 1. 直接挂载 mini-infra verify <subcommand>
    p_tf = subparsers.add_parser("transformers", help="与 Hugging Face Transformers 对齐")
    add_transformers_args(p_tf)
    p_tf.set_defaults(func=handle_transformers)

    p_cache = subparsers.add_parser("cache", help="KV Cache 数值对齐")
    add_cache_args(p_cache)
    p_cache.set_defaults(func=handle_cache)

    p_batch = subparsers.add_parser(
        "lockstep-batch", aliases=["batch"], help="Lockstep 齐步批处理数值与请求隔离对齐"
    )
    add_lockstep_batch_args(p_batch)
    p_batch.set_defaults(func=handle_lockstep_batch)

    p_cont = subparsers.add_parser(
        "scheduler-batch", aliases=["continuous"], help="Scheduler 批处理动态请求归属与输出一致性验证"
    )
    add_scheduler_batch_args(p_cont)
    p_cont.set_defaults(func=handle_scheduler_batch)

    # 2. 支持分层命令 mini-infra verify parity <subcommand>
    p_parity = subparsers.add_parser("parity", help="模型与特征数值对齐验证")
    parity_subs = p_parity.add_subparsers(dest="parity_subcommand", required=True)

    p_p_tf = parity_subs.add_parser("transformers", help="与 Hugging Face Transformers 对齐")
    add_transformers_args(p_p_tf)
    p_p_tf.set_defaults(func=handle_transformers)

    p_p_cache = parity_subs.add_parser("cache", help="KV Cache 数值对齐")
    add_cache_args(p_p_cache)
    p_p_cache.set_defaults(func=handle_cache)

    p_p_batch = parity_subs.add_parser(
        "lockstep-batch", aliases=["batch"], help="Lockstep 齐步批处理数值与请求隔离对齐"
    )
    add_lockstep_batch_args(p_p_batch)
    p_p_batch.set_defaults(func=handle_lockstep_batch)

    p_p_cont = parity_subs.add_parser(
        "scheduler-batch", aliases=["continuous"], help="Scheduler 批处理动态请求归属与输出一致性验证"
    )
    add_scheduler_batch_args(p_p_cont)
    p_p_cont.set_defaults(func=handle_scheduler_batch)


def handle_transformers(args: argparse.Namespace) -> int:
    return run_compare_transformers(
        model=args.model,
        revision=args.revision,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        logits_atol=args.logits_atol,
    )


def handle_cache(args: argparse.Namespace) -> int:
    return run_compare_cache(
        model=args.model,
        revision=args.revision,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        logits_atol=args.logits_atol,
    )


def handle_lockstep_batch(args: argparse.Namespace) -> int:
    return execute_compare_lockstep_batch(
        model=args.model,
        revision=args.revision,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        logits_atol=args.logits_atol,
        output=args.output,
    )


def handle_scheduler_batch(args: argparse.Namespace) -> int:
    return execute_compare_scheduler_batch(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        max_running_requests=args.max_running_requests,
        output=args.output,
    )


# 保持别名兼容
handle_batch = handle_lockstep_batch
handle_continuous = handle_scheduler_batch
