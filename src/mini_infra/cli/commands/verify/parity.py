"""CLI 子命令：mini-infra verify 数值对齐与请求隔离验证。

包含 transformers / cache / batch-lockstep / batch-scheduler / schedulers-token-budget /
chunked-prefill / prefix-caching / paged-cache-logits / paged-cache-scheduler。
各子命令的参数集合与默认值均以对应 execute_compare_* 函数签名为唯一真源。
"""

import argparse
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from mini_infra.verification.parity.compare_batch_lockstep import (
    execute_compare_lockstep_batch,
)
from mini_infra.verification.parity.compare_batch_scheduler import (
    execute_compare_scheduler_batch,
)
from mini_infra.verification.parity.compare_cache import execute_compare_cache
from mini_infra.verification.parity.compare_chunked_prefill import (
    execute_compare_chunked_prefill,
)
from mini_infra.verification.parity.compare_flash_attention import (
    execute_compare_flash_attention,
)
from mini_infra.verification.parity.compare_paged_cache_logits import (
    execute_compare_paged_logits,
)
from mini_infra.verification.parity.compare_paged_cache_scheduler import (
    execute_compare_paged_cache_scheduler,
)
from mini_infra.verification.parity.compare_prefix_caching import (
    execute_compare_prefix_caching,
)
from mini_infra.verification.parity.compare_schedulers_token_budget import (
    execute_compare_schedulers,
)
from mini_infra.verification.parity.compare_transformers import (
    execute_compare_transformers,
)


@dataclass(frozen=True)
class _ArgSpec:
    """单个标准 CLI 参数的解析规格（默认值从 execute_compare_* 签名提取）。"""

    help: str
    arg_type: type[int] | type[float] | None = None
    choices: tuple[str, ...] | None = None
    nargs: Literal["+"] | None = None

    def register(
        self,
        parser: argparse.ArgumentParser,
        dest: str,
        default: object,
        help_override: str | None = None,
    ) -> None:
        parser.add_argument(
            f"--{dest.replace('_', '-')}",
            type=self.arg_type,
            choices=self.choices,
            nargs=self.nargs,
            default=default,
            help=help_override or self.help,
        )


_ARG_SPECS: Mapping[str, _ArgSpec] = {
    "model": _ArgSpec(help="模型 ID 或本地目录（默认: %(default)s）"),
    "revision": _ArgSpec(help="Hugging Face 模型版本（默认: %(default)s）"),
    "prompt": _ArgSpec(help="单条测试 prompt（默认: %(default)s）"),
    "prompts": _ArgSpec(
        nargs="+",
        help="待测试的一条或多条 prompt（默认使用内置对照集）",
    ),
    "dtype": _ArgSpec(
        choices=("float32", "bfloat16"),
        help="正确性验证默认使用 float32，避免低精度路径差异干扰判断（默认: %(default)s）",
    ),
    "max_new_tokens": _ArgSpec(
        arg_type=int,
        help="生成的最大 Token 数（默认: %(default)s）",
    ),
    "logits_atol": _ArgSpec(
        arg_type=float,
        help="允许的最大 Logits 绝对误差（默认: %(default)s）",
    ),
    "block_size": _ArgSpec(
        arg_type=int,
        help="Paged KV Cache physical block 大小（默认: %(default)s）",
    ),
    "block_sizes": _ArgSpec(
        arg_type=int,
        nargs="+",
        help="Paged KV Cache physical block 大小列表（默认: %(default)s）",
    ),
    "max_running_requests": _ArgSpec(
        arg_type=int,
        help="scheduler 最大并发请求数（默认: %(default)s）",
    ),
    "token_budget": _ArgSpec(
        arg_type=int,
        help="单次 Prefill 的 padded token 预算（默认: %(default)s）",
    ),
    "output": _ArgSpec(help="可选 JSON 输出路径"),
}


@dataclass(frozen=True)
class _ParityCommandSpec:
    """单个 verify parity 子命令的声明。"""

    name: str
    help: str
    runner: Callable[..., int]
    help_overrides: Mapping[str, str] = field(default_factory=dict)


_DYNAMIC_ATOL_HELP = (
    "允许的最大 Logits 绝对误差（默认: float32 为 1e-4，bfloat16 为 1.0）"
)

_COMMANDS: tuple[_ParityCommandSpec, ...] = (
    _ParityCommandSpec(
        name="transformers",
        help="与 Hugging Face Transformers 对齐",
        runner=execute_compare_transformers,
    ),
    _ParityCommandSpec(
        name="cache",
        help="KV Cache 数值对齐",
        runner=execute_compare_cache,
    ),
    _ParityCommandSpec(
        name="batch-lockstep",
        help="Lockstep 齐步批处理数值与请求隔离对齐",
        runner=execute_compare_lockstep_batch,
    ),
    _ParityCommandSpec(
        name="batch-scheduler",
        help="Scheduler 批处理动态请求归属与输出一致性验证",
        runner=execute_compare_scheduler_batch,
    ),
    _ParityCommandSpec(
        name="schedulers-token-budget",
        help="prefill-first 与 token-budget 两种迭代级调度策略的输出与轨迹对齐验证",
        runner=execute_compare_schedulers,
    ),
    _ParityCommandSpec(
        name="chunked-prefill",
        help="token-budget 与 chunked-prefill 的输出一致性、首 Token Logits 误差与轨迹验证",
        runner=execute_compare_chunked_prefill,
        help_overrides={
            "logits_atol": "首 Token Logits 最大绝对误差阈值（默认: %(default)s）",
        },
    ),
    _ParityCommandSpec(
        name="prefix-caching",
        help="prefix caching 关闭 / 开启的输出一致性、首 Token Logits 误差与 Prefill 计算量对照",
        runner=execute_compare_prefix_caching,
        help_overrides={
            "prompts": "自定义共享前缀 prompt（至少 2 条；默认运行内置 4 条多边界自然语言对照）",
            "max_running_requests": (
                "scheduler 最大并发请求数（默认: 1，后续请求入场时前序请求的公共前缀已写入 cached block）"
            ),
            "logits_atol": "首 Token Logits 最大绝对误差阈值（默认: float32 为 1e-4，bfloat16 为 1.0）",
        },
    ),
    _ParityCommandSpec(
        name="paged-cache-logits",
        help="Dense 与 Paged KV Cache 逐步 Logits 数值对齐",
        runner=execute_compare_paged_logits,
        help_overrides={"logits_atol": _DYNAMIC_ATOL_HELP},
    ),
    _ParityCommandSpec(
        name="paged-cache-scheduler",
        help="Scheduler 调度环境下 Dense 与多 Block Size Paged KV Cache 输出一致性验证",
        runner=execute_compare_paged_cache_scheduler,
    ),
    _ParityCommandSpec(
        name="flash-attention",
        help="eager 与 flash attention backend 的 Kernel 级误差与 Lockstep / Paged chunked-prefill 模型级输出对齐",
        runner=execute_compare_flash_attention,
        help_overrides={
            "logits_atol": "模型级 Logits 最大绝对误差阈值，仅 float32 判定（默认: %(default)s）",
        },
    ),
)


def _register_command(
    subparsers: argparse._SubParsersAction,
    spec: _ParityCommandSpec,
) -> None:
    parser = subparsers.add_parser(spec.name, help=spec.help)
    sig = inspect.signature(spec.runner)
    param_names: list[str] = []
    for name, param in sig.parameters.items():
        if name not in _ARG_SPECS:
            raise KeyError(
                f"{spec.runner.__name__} 的参数 '{name}' 未在 _ARG_SPECS 中登记"
            )
        default = None if param.default is inspect.Parameter.empty else param.default
        _ARG_SPECS[name].register(
            parser,
            dest=name,
            default=default,
            help_override=spec.help_overrides.get(name),
        )
        param_names.append(name)

    bound_names = tuple(param_names)

    def _handle(args: argparse.Namespace) -> int:
        return spec.runner(**{name: getattr(args, name) for name in bound_names})

    parser.set_defaults(func=_handle)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    """注册 parity 相关验证命令，直接挂载到 verify 下。"""
    for spec in _COMMANDS:
        _register_command(subparsers, spec)


__all__ = ["register_parser"]
