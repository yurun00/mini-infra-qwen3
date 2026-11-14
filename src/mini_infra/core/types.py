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


# ============================================================================
# SchedulerConfig 与正交调度维度演进路线（防概念冲突契约）
#
# 单步迭代调度（IterationScheduler.schedule）由四个彼此正交的决策维度组合而成，
# 严禁将不同维度的策略拍平塞入同一个 Enum（避免笛卡尔积爆炸与与 vLLM 术语冲突）：
#
#   1. 队列排序策略 `queue_policy` (QueuePolicyKind) — [TODO: 待演进]
#      - 回答问题：WaitingQueue 按什么次序接纳请求？RunningQueue 满载时按什么逆序选 Victim？
#      - 业界对齐：对应 vLLM 的 `--scheduling-policy` (`SchedulerConfig.policy`)。
#      - 规划取值："fcfs"（默认，按 arrival_ms 稳定排序）| "priority"（按业务优先级）| "srtf"（最短剩余时间优先）。
#
#   2. 单步组批与阶段准入策略 `batching_policy` (BatchingPolicyKind) — [当前已实现]
#      - 回答问题：本轮 Step 如何划定 Prefill / Decode 阶段与单步 Token 预算？
#      - 注意区分：与宏观批处理机制 `BatchingMethod`（none / fixed-batching / continuous-batching）正交；
#        `BatchingMethod` 决定跨步生命周期（是否有 wave barrier），`BatchingPolicyKind` 决定单步组批规则。
#      - 当前取值："prefill-first"（有可入场请求即整批 Prefill）| "token-budget"（受单步 padded prompt token
#        软预算约束并启用 decode protection）。
#      - TODO 演进："chunked-prefill"（vLLM V1 统一 Token-Budget 范式：解除 P/D 互斥，先扣减在途 Decode 预算，
#        剩余预算将长 Prompt 切块混批执行）。
#
#   3. 抢占/驱逐模式 `preemption_mode` (PreemptionMode) — [TODO: 待演进]
#      - 回答问题：Decode 步追加新 Token 导致 free_block_queue 耗尽时，被选中的 Victim 请求如何处置？
#      - 业界对齐：对应 vLLM 的 `--preemption-mode`。
#      - 规划取值："recompute"（释放物理 KV blocks 并退回 WaitingQueue 重算 Prefill）
#        | "swap"（通过 PCIe 将物理 KV blocks 换出至 CPU Host Swap Pool，恢复时换回）。
#
#   4. 准入容量水印 `watermark` (float) — [TODO: 待演进]
#      - 回答问题：Prefill 准入时预留多少比例空闲 physical blocks 作安全垫，防止接纳后立刻在 Decode 步触发抢占抖动？
# ============================================================================

BatchingPolicyKind = Literal["prefill-first", "token-budget"]
BATCHING_POLICY_KINDS: tuple[BatchingPolicyKind, ...] = ("prefill-first", "token-budget")


def parse_batching_policy_kind(value: str | None) -> BatchingPolicyKind:
    """将字符串解析为合法的 BatchingPolicyKind（None 视为默认的 prefill-first）。"""
    if value is None or value == "prefill-first":
        return "prefill-first"
    if value == "token-budget":
        return "token-budget"
    raise ValueError(f"未知 batching policy: {value}，可选策略: {list(BATCHING_POLICY_KINDS)}")


@dataclass(frozen=True)
class PrefillFirstConfig:
    """prefill-first 组批策略（默认）：WaitingQueue 有可入场请求即整批 Prefill，否则 Decode。"""

    kind: ClassVar[BatchingPolicyKind] = "prefill-first"

    def __str__(self) -> str:
        return self.kind


@dataclass(frozen=True)
class TokenBudgetConfig:
    """token-budget 组批策略：以 padded prompt token 数限制单次 Batched Prefill 规模，并启用 decode protection。

    token_budget: 单次 Prefill 允许的 padded prompt token 上限（软预算：队头请求单独超预算时仍放行）。
    """

    kind: ClassVar[BatchingPolicyKind] = "token-budget"
    token_budget: int

    def __post_init__(self) -> None:
        if self.token_budget < 1:
            raise ValueError("token_budget 必须大于 0")

    def __str__(self) -> str:
        return f"{self.kind}(token_budget={self.token_budget})"


# 当前 SchedulerConfig 仅承载维度 2（batching_policy）的判别式联合；
# TODO: 引入 queue_policy（priority / srtf）或 preemption_mode（recompute / swap）时，
# 演进为聚合 `queue_policy`、`batching_policy`、`preemption_mode` 与 `watermark` 的复合配置。
SchedulerConfig = PrefillFirstConfig | TokenBudgetConfig


def scheduler_config_from_str(kind: str | None, *, token_budget: int | None = None) -> SchedulerConfig:
    """根据 batching policy 标识与可选预算构造对应的 SchedulerConfig 实例（prefill-first 忽略 token_budget）。"""
    parsed = parse_batching_policy_kind(kind)
    if parsed == "prefill-first":
        return PrefillFirstConfig()
    if token_budget is None:
        raise ValueError("token-budget 组批策略必须指定 token_budget")
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
    "BATCHING_POLICY_KINDS",
    "KV_CACHE_KINDS",
    "BatchingMethod",
    "BatchingPolicyKind",
    "DenseCacheConfig",
    "KVCacheConfig",
    "KVCacheKind",
    "NoCacheConfig",
    "PagedCacheConfig",
    "PrefillFirstConfig",
    "RequestMetrics",
    "SchedulerConfig",
    "SequenceOutput",
    "TokenBudgetConfig",
    "collapse_uniform",
    "kv_cache_config_from_str",
    "parse_batching_policy_kind",
    "parse_kv_cache_kind",
    "scheduler_config_from_str",
]
