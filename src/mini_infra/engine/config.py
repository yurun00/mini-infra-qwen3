"""引擎集中式运行配置（EngineConfig）与特性标志（FeatureFlag）定义。"""

from dataclasses import dataclass, replace
from enum import Enum

from mini_infra.core.types import (
    BatchingMethod,
    BatchingPolicy,
    BudgetedBatchingPolicy,
    ChunkedPrefillConfig,
    KVCacheConfig,
    NoCacheConfig,
    PagedCacheConfig,
    PrefillFirstConfig,
    SchedulerConfig,
)
from mini_infra.model.attention import AttentionBackendKind


class FeatureFlag(str, Enum):
    """支持的所有布尔型特性标志（Feature Flag）；每个成员对应一个可独立开启/关闭（enable/disable）的二值特性。

    当前尚无布尔型特性，仅保留 PLACEHOLDER 占位成员，供后续新增特性标志时沿用 EngineConfig.flags 通路。
    """

    PLACEHOLDER = "placeholder"

    @classmethod
    def from_str(cls, value: "str | FeatureFlag") -> "FeatureFlag":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(
                f"未知特性标志: {value}，可选特性: {[m.value for m in cls]}"
            ) from err

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class EngineConfig:
    """引擎集中式运行配置（对标 vLLM VllmConfig）：聚合 KV Cache 配置、批处理策略、特性标志与调度器配置。

    - kv_cache_config: 互斥的 KV Cache 配置（NoCacheConfig / DenseCacheConfig / PagedCacheConfig）。
    - batching_method: 互斥的宏观批处理策略（BatchingMethod.NONE / FIXED / CONTINUOUS）。
    - flags: 正交的布尔型功能开关集合。
    - scheduler_config: 迭代级调度器配置（承载 max_running_requests 与 batching_policy；
      TODO 后续可扩展聚合 queue_policy 与 preemption_mode，对标 vLLM SchedulerConfig）。
    - attention_backend: attention 核心计算后端（eager / flash，对标 vLLM --attention-backend）；
      与 KV Cache、批处理、调度维度正交，由 worker 层 Runner 解析为 attn_backend 实例。
    """

    kv_cache_config: KVCacheConfig = NoCacheConfig()
    batching_method: BatchingMethod = BatchingMethod.NONE
    flags: frozenset[FeatureFlag] = frozenset()
    scheduler_config: SchedulerConfig = SchedulerConfig()
    attention_backend: AttentionBackendKind = "eager"

    def with_flag(self, flag: FeatureFlag, enabled: bool = True) -> "EngineConfig":
        """返回指定 FeatureFlag 状态修改后的新 EngineConfig 实例。"""
        new_flags = (self.flags | {flag}) if enabled else (self.flags - {flag})
        return replace(self, flags=new_flags)

    def with_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> "EngineConfig":
        """返回指定 kv_cache_config 修改后的新 EngineConfig 实例。"""
        return replace(self, kv_cache_config=kv_cache_config)

    def with_batching_method(self, batching_method: BatchingMethod) -> "EngineConfig":
        """返回指定 batching_method 策略修改后的新 EngineConfig 实例。"""
        return replace(self, batching_method=batching_method)

    def with_scheduler_config(self, scheduler_config: SchedulerConfig) -> "EngineConfig":
        """返回指定 scheduler_config 修改后的新 EngineConfig 实例。"""
        return replace(self, scheduler_config=scheduler_config)

    def with_max_running_requests(
        self, max_running_requests: int | None
    ) -> "EngineConfig":
        """返回指定 max_running_requests 修改后的新 EngineConfig 实例。"""
        return replace(
            self,
            scheduler_config=self.scheduler_config.with_max_running_requests(
                max_running_requests
            ),
        )

    def with_batching_policy(self, batching_policy: BatchingPolicy) -> "EngineConfig":
        """返回指定 batching_policy 修改后的新 EngineConfig 实例。"""
        return replace(
            self,
            scheduler_config=self.scheduler_config.with_batching_policy(batching_policy),
        )

    def with_attention_backend(
        self, attention_backend: AttentionBackendKind
    ) -> "EngineConfig":
        """返回指定 attention_backend 修改后的新 EngineConfig 实例。"""
        return replace(self, attention_backend=attention_backend)

    @property
    def has_kv_cache(self) -> bool:
        """是否使用任一 KV Cache 实现（Dense 或 Paged）。"""
        return not isinstance(self.kv_cache_config, NoCacheConfig)

    @property
    def is_batch(self) -> bool:
        return self.batching_method != BatchingMethod.NONE

    def __str__(self) -> str:
        parts = [
            *sorted(flag.value for flag in self.flags),
            str(self.kv_cache_config),
        ]
        if self.batching_method != BatchingMethod.NONE:
            parts.append(self.batching_method.value)
        if self.scheduler_config.max_running_requests is not None:
            parts.append(f"mrr={self.scheduler_config.max_running_requests}")
        # 默认 prefill-first 不输出，保证历史 Record 的 engine_config 标识不变
        if not isinstance(self.scheduler_config.batching_policy, PrefillFirstConfig):
            parts.append(str(self.scheduler_config.batching_policy))
        # 默认 eager 不输出，保证历史 Record 的 engine_config 标识不变
        if self.attention_backend != "eager":
            parts.append(f"{self.attention_backend}-attention")
        return "+".join(parts)


def _validate_scheduler_config(
    scheduler_config: SchedulerConfig,
    batching_method: BatchingMethod,
) -> None:
    """校验 SchedulerConfig 与批处理策略的组合；max_running_requests 为 None（未推导）时跳过预算下限校验。

    token-budget / chunked-prefill 仅对 continuous-batching 有意义（fixed 的 wave barrier 与 none 串行都不存在
    跨步入场决策）；且 token_budget 不得小于并发上限，否则一次完整 Decode 即超预算（对标 vLLM
    max_num_batched_tokens >= max_num_seqs），chunked-prefill 下还会使部分在途请求无法推进 1 个 Token。
    """
    policy = scheduler_config.batching_policy
    if not isinstance(policy, BudgetedBatchingPolicy):
        return
    if batching_method is not BatchingMethod.CONTINUOUS:
        raise ValueError(
            f"{policy.kind} 组批策略仅支持 continuous-batching，实际为 {batching_method.value}"
        )
    mrr = scheduler_config.max_running_requests
    if mrr is not None and policy.token_budget < mrr:
        raise ValueError(
            f"token_budget={policy.token_budget} 不能小于 max_running_requests={mrr}，否则一次完整 Decode 即超预算"
        )


def validate_engine_config(config: EngineConfig) -> None:
    """校验 EngineConfig 的 (kv_cache_config, batching_method, scheduler_config) 是否落在合法组合内，非法即抛错。

    只编码与具体驱动无关的硬约束；run_scheduler / run_lockstep 再各自校验自身边界。
    """
    if isinstance(
        config.kv_cache_config, PagedCacheConfig
    ) and config.batching_method in (BatchingMethod.NONE):
        raise ValueError("paged-kv-cache 仅支持 continuous-batching 或 fixed-batching")
    if config.batching_method is BatchingMethod.CONTINUOUS and isinstance(
        config.kv_cache_config, NoCacheConfig
    ):
        raise ValueError(
            "continuous-batching 依赖 KV Cache，请将 kv_cache_config 设为 DenseCacheConfig 或 PagedCacheConfig"
        )
    if isinstance(
        config.scheduler_config.batching_policy, ChunkedPrefillConfig
    ) and not isinstance(config.kv_cache_config, PagedCacheConfig):
        raise ValueError(
            "chunked-prefill 组批策略仅支持 paged-kv-cache（chunk 的历史 K/V 取自 physical block pool）"
        )
    _validate_scheduler_config(config.scheduler_config, config.batching_method)


__all__ = [
    "BatchingMethod",
    "EngineConfig",
    "FeatureFlag",
    "validate_engine_config",
]
