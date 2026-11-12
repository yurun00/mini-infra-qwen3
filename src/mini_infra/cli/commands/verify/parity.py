"""CLI 子命令：mini-infra verify (transformers / cache / lockstep-batch / scheduler-batch)

数值对齐与请求隔离验证。
"""

import argparse

from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)
from mini_infra.verification.parity.compare_cache import execute_compare_cache
from mini_infra.verification.parity.compare_lockstep_batch import execute_compare_lockstep_batch
from mini_infra.verification.parity.compare_paged_cache_logits import (
    DEFAULT_PROMPTS as DEFAULT_PAGED_PROMPTS,
)
from mini_infra.verification.parity.compare_paged_cache_logits import (
    execute_compare_paged_logits,
)
from mini_infra.verification.parity.compare_paged_cache_scheduler import (
    DEFAULT_PROMPTS as DEFAULT_PAGED_SCHEDULER_PROMPTS,
)
from mini_infra.verification.parity.compare_paged_cache_scheduler import (
    execute_compare_paged_cache_scheduler,
)
from mini_infra.verification.parity.compare_scheduler_batch import execute_compare_scheduler_batch
from mini_infra.verification.parity.compare_transformers import execute_compare_transformers


def _add_transformers_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本")
    parser.add_argument("--prompt", default="请用一句话介绍 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--logits-atol", type=float, default=0.25, help="Logits 最大绝对误差阈值")
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser


def _add_cache_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser


def _add_lockstep_batch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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


def _add_scheduler_batch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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


def _add_paged_cache_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PAGED_PROMPTS,
        help="待测试的一条或多条 prompt",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="正确性验证默认使用 float32，避免低精度路径差异干扰判断",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8, help="生成的最大 Token 数（至少为 2，默认: 8）")
    parser.add_argument("--block-size", type=int, default=16, help="Paged KV Cache 物理块大小（默认: 16）")
    parser.add_argument("--logits-atol", type=float, default=None, help="允许的最大 Logits 绝对误差")
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser


def _add_paged_cache_scheduler_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=DEFAULT_PAGED_SCHEDULER_PROMPTS,
        help="待测试的一条或多条 prompt",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="正确性验证默认使用 float32，避免低精度路径差异干扰判断",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8, help="生成的最大 Token 数（至少为 2，默认: 8）")
    parser.add_argument(
        "--block-sizes",
        type=int,
        nargs="+",
        default=[4, 16],
        help="Paged KV Cache 物理块大小列表（默认: [4, 16]）",
    )
    parser.add_argument("--max-running-requests", type=int, default=2, help="调度器最大并发请求数（默认: 2）")
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """注册 parity 相关验证命令，直接挂载到 verify 下。"""
    p_tf = subparsers.add_parser("transformers", help="与 Hugging Face Transformers 对齐")
    _add_transformers_args(p_tf)
    p_tf.set_defaults(func=_handle_transformers)

    p_cache = subparsers.add_parser("cache", help="KV Cache 数值对齐")
    _add_cache_args(p_cache)
    p_cache.set_defaults(func=_handle_cache)

    p_batch = subparsers.add_parser("lockstep-batch", help="Lockstep 齐步批处理数值与请求隔离对齐")
    _add_lockstep_batch_args(p_batch)
    p_batch.set_defaults(func=_handle_lockstep_batch)

    p_cont = subparsers.add_parser("scheduler-batch", help="Scheduler 批处理动态请求归属与输出一致性验证")
    _add_scheduler_batch_args(p_cont)
    p_cont.set_defaults(func=_handle_scheduler_batch)

    p_paged = subparsers.add_parser("paged-cache", help="Dense 与 Paged KV Cache 逐步 Logits 数值对齐")
    _add_paged_cache_args(p_paged)
    p_paged.set_defaults(func=_handle_paged_cache)

    p_paged_sched = subparsers.add_parser(
        "paged-cache-scheduler",
        help="Scheduler 调度环境下 Dense 与多 Block Size Paged KV Cache 输出一致性验证",
    )
    _add_paged_cache_scheduler_args(p_paged_sched)
    p_paged_sched.set_defaults(func=_handle_paged_cache_scheduler)


def _handle_transformers(args: argparse.Namespace) -> int:
    return execute_compare_transformers(
        model=args.model,
        revision=args.revision,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        logits_atol=args.logits_atol,
        output=args.output,
    )


def _handle_cache(args: argparse.Namespace) -> int:
    return execute_compare_cache(
        model=args.model,
        revision=args.revision,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        logits_atol=args.logits_atol,
        output=args.output,
    )


def _handle_lockstep_batch(args: argparse.Namespace) -> int:
    return execute_compare_lockstep_batch(
        model=args.model,
        revision=args.revision,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        logits_atol=args.logits_atol,
        output=args.output,
    )


def _handle_scheduler_batch(args: argparse.Namespace) -> int:
    return execute_compare_scheduler_batch(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        max_running_requests=args.max_running_requests,
        output=args.output,
    )


def _handle_paged_cache(args: argparse.Namespace) -> int:
    return execute_compare_paged_logits(
        model=args.model,
        revision=args.revision,
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        dtype=args.dtype,
        logits_atol=args.logits_atol,
        output=args.output,
    )


def _handle_paged_cache_scheduler(args: argparse.Namespace) -> int:
    return execute_compare_paged_cache_scheduler(
        model=args.model,
        revision=args.revision,
        prompts=args.prompts,
        max_new_tokens=args.max_new_tokens,
        block_sizes=args.block_sizes,
        max_running_requests=args.max_running_requests,
        dtype=args.dtype,
        output=args.output,
    )
