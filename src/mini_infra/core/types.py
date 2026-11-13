"""control plane 基础数据结构体（BatchingMethod、KVCacheConfig、SchedulerConfig、SequenceOutput、RequestMetrics）。

注意：此模块处于纯 CPU control plane，严禁 import torch 或引入任何 GPU 张量操作。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal

KVCacheKind = Literal["none", "dense-kv-cache", "paged-kv-cache"]
KV_CACHE_KINDS: tuple[KVCacheKind, ...] = ("none", "dense-kv-cache", "paged-kv-cache")


def parse_kv_cache_kind(value: str | None) -> KVCacheKind:
    """将字符串解析为合法的 KVCacheKind。"""
    if value is None or value == "none":
        return "none"
    if value == "dense-kv-cache":
        return "dense-kv-cache"
    if value == "paged-kv-cache":
        return "paged-kv-cache"
    raise ValueError(f"未知 KV Cache 策略: {value}，可选策略: {list(KV_CACHE_KINDS)}")


@dataclass(frozen=True)
class NoCacheConfig:
    """不启用 KV Cache 的运行时配置。"""

    kind: ClassVar[KVCacheKind] = "none"

    def __str__(self) -> str:
        return "no-cache"


@dataclass(frozen=True)
class DenseCacheConfig:
    """Dense KV Cache（预分配连续 batch slot）的运行时配置。"""

    kind: ClassVar[KVCacheKind] = "dense-kv-cache"

    def __str__(self) -> str:
        return self.kind


@dataclass(frozen=True)
class PagedCacheConfig:
    """Paged KV Cache 配置。

    block_size: 每块容纳的 Token 数；num_blocks: physical block pool 大小（含 null block），None 时按负载自动推导。
    """

    kind: ClassVar[KVCacheKind] = "paged-kv-cache"
    block_size: int = 16
    num_blocks: int | None = None

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size 必须大于 0")
        if self.num_blocks is not None and self.num_blocks < 2:
            raise ValueError("num_blocks 至少为 2（block 0 为 null block，至少还需 1 个可分配块）")

    def __str__(self) -> str:
        if self.num_blocks is not None:
            return f"{self.kind}(block_size={self.block_size},num_blocks={self.num_blocks})"
        return f"{self.kind}(block_size={self.block_size})"


KVCacheConfig = NoCacheConfig | DenseCacheConfig | PagedCacheConfig


def kv_cache_config_from_str(
    kind: str | None,
    *,
    block_size: int | None = None,
    num_blocks: int | None = None,
) -> KVCacheConfig:
    """根据字符串标识与可选块参数构造对应的 KVCacheConfig 实例。"""
    parsed = parse_kv_cache_kind(kind)
    if parsed == "none":
        return NoCacheConfig()
    if parsed == "dense-kv-cache":
        return DenseCacheConfig()
    return PagedCacheConfig(
        block_size=block_size if block_size is not None else PagedCacheConfig.block_size,
        num_blocks=num_blocks,
    )


SchedulingPolicyKind = Literal["prefill-first", "token-budget"]
SCHEDULING_POLICY_KINDS: tuple[SchedulingPolicyKind, ...] = ("prefill-first", "token-budget")


def parse_scheduling_policy_kind(value: str | None) -> SchedulingPolicyKind:
    """将字符串解析为合法的 SchedulingPolicyKind（None 视为默认的 prefill-first）。"""
    if value is None or value == "prefill-first":
        return "prefill-first"
    if value == "token-budget":
        return "token-budget"
    raise ValueError(f"未知调度策略: {value}，可选策略: {list(SCHEDULING_POLICY_KINDS)}")


@dataclass(frozen=True)
class PrefillFirstConfig:
    """prefill-first 调度策略（默认）：WaitingQueue 有可入场请求即整批 Prefill，否则 Decode。"""

    kind: ClassVar[SchedulingPolicyKind] = "prefill-first"

    def __str__(self) -> str:
        return self.kind


@dataclass(frozen=True)
class TokenBudgetConfig:
    """token-budget 调度策略：以 padded prompt token 数限制单次 Batched Prefill 规模，并启用 decode protection。

    token_budget: 单次 Prefill 允许的 padded prompt token 上限（软预算：队头请求单独超预算时仍放行）。
    """

    kind: ClassVar[SchedulingPolicyKind] = "token-budget"
    token_budget: int

    def __post_init__(self) -> None:
        if self.token_budget < 1:
            raise ValueError("token_budget 必须大于 0")

    def __str__(self) -> str:
        return f"{self.kind}(token_budget={self.token_budget})"


SchedulerConfig = PrefillFirstConfig | TokenBudgetConfig


def scheduler_config_from_str(kind: str | None, *, token_budget: int | None = None) -> SchedulerConfig:
    """根据策略标识与可选预算构造对应的 SchedulerConfig 实例（prefill-first 忽略 token_budget）。"""
    parsed = parse_scheduling_policy_kind(kind)
    if parsed == "prefill-first":
        return PrefillFirstConfig()
    if token_budget is None:
        raise ValueError("token-budget 调度策略必须指定 token_budget")
    return TokenBudgetConfig(token_budget)


class BatchingMethod(str, Enum):
    """互斥的批处理策略。"""

    NONE = "none"
    FIXED = "fixed-batching"
    CONTINUOUS = "continuous-batching"

    @classmethod
    def from_str(cls, value: "str | BatchingMethod | None") -> "BatchingMethod":
        if value is None:
            return cls.NONE
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"未知批处理策略: {value}，可选策略: {[m.value for m in cls]}") from err

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SequenceOutput:
    """单个请求的生成序列输出（Offline 与 Serving 共享）。"""

    request_idx: int
    token_ids: list[int]
    prompt_len: int
    output_len: int
    finish_reason: str  # 结束原因: "stop"（命中 EOS）或 "max_tokens"（达到 Token 预算上限）
    completion_step: int | None = None


@dataclass(frozen=True)
class RequestMetrics:
    """单个请求在逻辑时间线上的关键时间戳与延迟指标（单位 ms）。"""

    request_idx: int
    arrival_ms: float
    admitted_ms: float
    first_token_ms: float
    completion_ms: float
    num_output_tokens: int
    token_times_ms: tuple[float, ...]

    @property
    def queue_ms(self) -> float:
        return self.admitted_ms - self.arrival_ms

    @property
    def ttft_ms(self) -> float:
        return self.first_token_ms - self.arrival_ms

    @property
    def end_to_end_ms(self) -> float:
        return self.completion_ms - self.arrival_ms

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        return [right - left for left, right in zip(self.token_times_ms, self.token_times_ms[1:])]


def collapse_uniform(values: Sequence[Any]) -> Any:
    """若序列所有元素相等则折叠为标量，否则返回元组（用于评测结果序列化与紧凑展示）。"""
    if not values:
        return ()
    first = values[0]
    if all(v == first for v in values):
        return first
    return tuple(values)


__all__ = [
    "KV_CACHE_KINDS",
    "SCHEDULING_POLICY_KINDS",
    "BatchingMethod",
    "DenseCacheConfig",
    "KVCacheConfig",
    "KVCacheKind",
    "NoCacheConfig",
    "PagedCacheConfig",
    "PrefillFirstConfig",
    "RequestMetrics",
    "SchedulerConfig",
    "SchedulingPolicyKind",
    "SequenceOutput",
    "TokenBudgetConfig",
    "collapse_uniform",
    "kv_cache_config_from_str",
    "parse_kv_cache_kind",
    "parse_scheduling_policy_kind",
    "scheduler_config_from_str",
]
