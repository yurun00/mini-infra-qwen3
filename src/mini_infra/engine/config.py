"""引擎集中式运行配置（EngineConfig）与特性标志（FeatureFlag）定义。"""

from dataclasses import dataclass, replace
from enum import Enum

from mini_infra.core.types import (
    BatchingMethod,
    KVCacheConfig,
    NoCacheConfig,
    PagedCacheConfig,
    PrefillFirstConfig,
    SchedulerConfig,
    TokenBudgetConfig,
)


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
            raise ValueError(f"未知特性标志: {value}，可选特性: {[m.value for m in cls]}") from err

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class EngineConfig:
    """引擎集中式运行配置（对标 vLLM VllmConfig）：聚合 KV Cache 配置、批处理策略、并发上限、特性标志与调度策略。

    - kv_cache_config: 互斥的 KV Cache 配置（NoCacheConfig / DenseCacheConfig / PagedCacheConfig）。
    - batching_method: 互斥的批处理策略（BatchingMethod.NONE / FIXED / CONTINUOUS）。
    - max_running_requests: 调度队列最大并发请求数（None 表示不限，按工作负载请求总数推导）。
    - flags: 正交的布尔型功能开关集合。
    - scheduler_config: 互斥的迭代级调度策略配置（PrefillFirstConfig / TokenBudgetConfig，对标 vLLM SchedulerConfig）。
    """

    kv_cache_config: KVCacheConfig = NoCacheConfig()
    batching_method: BatchingMethod = BatchingMethod.NONE
    max_running_requests: int | None = None
    flags: frozenset[FeatureFlag] = frozenset()
    scheduler_config: SchedulerConfig = PrefillFirstConfig()

    def __post_init__(self) -> None:
        if self.max_running_requests is not None and self.max_running_requests < 1:
            raise ValueError(f"max_running_requests 必须大于 0，实际为 {self.max_running_requests}")

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

    def with_max_running_requests(self, max_running_requests: int | None) -> "EngineConfig":
        """返回指定 max_running_requests 修改后的新 EngineConfig 实例。"""
        return replace(self, max_running_requests=max_running_requests)

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
        if self.max_running_requests is not None:
            parts.append(f"mrr={self.max_running_requests}")
        # 默认 prefill-first 不输出，保证历史 Record 的 engine_config 标识不变
        if not isinstance(self.scheduler_config, PrefillFirstConfig):
            parts.append(str(self.scheduler_config))
        return "+".join(parts)


def _validate_scheduler_config(
    scheduler_config: SchedulerConfig,
    batching_method: BatchingMethod,
    max_running_requests: int | None,
) -> None:
    """校验调度策略与批处理策略、并发上限的组合；max_running_requests 为 None（尚未推导）时跳过预算下限校验。

    token-budget 仅对 continuous-batching 有意义（fixed 的 wave barrier 与 none 串行都不存在跨步入场决策）；
    且 token_budget 不得小于并发上限，否则一次完整 Decode 即超预算（对标 vLLM max_num_batched_tokens >= max_num_seqs）。
    """
    if not isinstance(scheduler_config, TokenBudgetConfig):
        return
    if batching_method is not BatchingMethod.CONTINUOUS:
        raise ValueError(f"token-budget 调度策略仅支持 continuous-batching，实际为 {batching_method.value}")
    if max_running_requests is not None and scheduler_config.token_budget < max_running_requests:
        raise ValueError(
            f"token_budget={scheduler_config.token_budget} 不能小于 max_running_requests={max_running_requests}，"
            "否则一次完整 Decode 即超预算"
        )


def validate_engine_config(config: EngineConfig) -> None:
    """校验 EngineConfig 的 (kv_cache_config, batching_method, scheduler_config) 是否落在合法组合内，非法即抛错。

    只编码与具体驱动无关的硬约束；run_scheduler / run_lockstep 再各自校验自身边界。
    """
    if isinstance(config.kv_cache_config, PagedCacheConfig) and config.batching_method in (BatchingMethod.NONE):
        raise ValueError("paged-kv-cache 仅支持 continuous-batching 或 fixed-batching")
    if config.batching_method is BatchingMethod.CONTINUOUS and isinstance(config.kv_cache_config, NoCacheConfig):
        raise ValueError(
            "continuous-batching 依赖 KV Cache，请将 kv_cache_config 设为 DenseCacheConfig 或 PagedCacheConfig"
        )
    _validate_scheduler_config(config.scheduler_config, config.batching_method, config.max_running_requests)


__all__ = [
    "BatchingMethod",
    "EngineConfig",
    "FeatureFlag",
    "validate_engine_config",
]
