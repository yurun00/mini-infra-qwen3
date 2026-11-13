"""引擎集中式运行配置（EngineConfig）与特性标志（FeatureFlag）定义。"""

from dataclasses import dataclass
from enum import Enum

from mini_infra.core.types import (
    BatchingMethod,
    KVCacheConfig,
    NoCacheConfig,
    PagedCacheConfig,
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
    """引擎集中式运行配置（对标 vLLM VllmConfig）：聚合 KV Cache 配置、批处理策略、并发上限与特性标志。

    - kv_cache_config: 互斥的 KV Cache 配置（NoCacheConfig / DenseCacheConfig / PagedCacheConfig）。
    - batching_method: 互斥的批处理策略（BatchingMethod.NONE / FIXED / CONTINUOUS）。
    - max_running_requests: 调度队列最大并发请求数（None 表示不限，按工作负载请求总数推导）。
    - flags: 正交的布尔型功能开关集合。
    """

    kv_cache_config: KVCacheConfig = NoCacheConfig()
    batching_method: BatchingMethod = BatchingMethod.NONE
    max_running_requests: int | None = None
    flags: frozenset[FeatureFlag] = frozenset()

    def __post_init__(self) -> None:
        if self.max_running_requests is not None and self.max_running_requests < 1:
            raise ValueError(f"max_running_requests 必须大于 0，实际为 {self.max_running_requests}")

    def with_flag(self, flag: FeatureFlag, enabled: bool = True) -> "EngineConfig":
        """返回指定 FeatureFlag 状态修改后的新 EngineConfig 实例。"""
        new_flags = (self.flags | {flag}) if enabled else (self.flags - {flag})
        return EngineConfig(self.kv_cache_config, self.batching_method, self.max_running_requests, new_flags)

    def with_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> "EngineConfig":
        """返回指定 kv_cache_config 修改后的新 EngineConfig 实例。"""
        return EngineConfig(kv_cache_config, self.batching_method, self.max_running_requests, self.flags)

    def with_batching_method(self, batching_method: BatchingMethod) -> "EngineConfig":
        """返回指定 batching_method 策略修改后的新 EngineConfig 实例。"""
        return EngineConfig(self.kv_cache_config, batching_method, self.max_running_requests, self.flags)

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
        return "+".join(parts)


def validate_engine_config(config: EngineConfig) -> None:
    """校验 EngineConfig 的 (kv_cache_config, batching_method) 是否落在合法组合内，非法即抛错。

    只编码与具体驱动无关的硬约束；run_scheduler / run_lockstep 再各自校验自身边界。
    """
    if isinstance(config.kv_cache_config, PagedCacheConfig) and config.batching_method in (
        BatchingMethod.NONE
    ):
        raise ValueError("paged-kv-cache 仅支持 continuous-batching 或 fixed-batching")
    if config.batching_method is BatchingMethod.CONTINUOUS and isinstance(config.kv_cache_config, NoCacheConfig):
        raise ValueError(
            "continuous-batching 依赖 KV Cache，请将 kv_cache_config 设为 DenseCacheConfig 或 PagedCacheConfig"
        )


__all__ = [
    "BatchingMethod",
    "EngineConfig",
    "FeatureFlag",
    "validate_engine_config",
]
