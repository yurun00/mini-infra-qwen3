"""运行模式与特性标志定义。"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from mini_infra.core.types import BatchingMethod, CachingMethod


class FeatureFlag(str, Enum):
    """支持的所有布尔型特性标志（Feature Flag）；每个成员对应一个可独立开启/关闭（enable/disable）的二值特性。

    当前尚无布尔型特性，仅保留 PLACEHOLDER 占位成员，供后续新增特性标志时沿用 Mode.flags 通路。
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


FeatureDimension = FeatureFlag | type[CachingMethod] | type[BatchingMethod]


@dataclass(frozen=True)
class Mode:
    """运行模式结构体：内部存储 KV Cache 策略（CachingMethod）、批处理策略（BatchingMethod）与 FeatureFlag 集合。

    - caching_method: 互斥的 KV Cache 策略（CachingMethod.NONE / DENSE / PAGED）。
    - batching_method: 互斥的批处理策略（BatchingMethod.NONE / FIXED / CONTINUOUS）。
    - flags: 正交的布尔型功能开关集合。
    """

    FEATURE_DIMENSIONS: ClassVar[tuple[FeatureDimension, ...]] = (
        *FeatureFlag,
        CachingMethod,
        BatchingMethod,
    )

    caching_method: CachingMethod = CachingMethod.NONE
    batching_method: BatchingMethod = BatchingMethod.NONE
    flags: frozenset[FeatureFlag] = frozenset()

    def __init__(
        self,
        caching_method: CachingMethod | str = CachingMethod.NONE,
        batching_method: BatchingMethod | str = BatchingMethod.NONE,
        flags: "Iterable[FeatureFlag | str]" = (),
    ):
        object.__setattr__(self, "caching_method", CachingMethod.from_str(caching_method))
        object.__setattr__(self, "batching_method", BatchingMethod.from_str(batching_method))
        object.__setattr__(self, "flags", frozenset(FeatureFlag.from_str(f) for f in flags))

    def is_enabled(self, flag: FeatureFlag) -> bool:
        """检查指定的 FeatureFlag 是否启用。"""
        return flag in self.flags

    def with_flag(self, flag: FeatureFlag, enabled: bool = True) -> "Mode":
        """返回指定 FeatureFlag 状态修改后的新 Mode 实例。"""
        new_flags = (self.flags | {flag}) if enabled else (self.flags - {flag})
        return Mode(self.caching_method, self.batching_method, new_flags)

    def with_caching_method(self, caching_method: CachingMethod | str) -> "Mode":
        """返回指定 caching_method 策略修改后的新 Mode 实例。"""
        return Mode(caching_method, self.batching_method, self.flags)

    def with_batching_method(self, batching_method: BatchingMethod | str) -> "Mode":
        """返回指定 batching_method 策略修改后的新 Mode 实例。"""
        return Mode(self.caching_method, batching_method, self.flags)

    @property
    def use_kv_cache(self) -> bool:
        """是否使用任一 KV Cache 实现（Dense 或 Paged）。"""
        return self.caching_method != CachingMethod.NONE

    @property
    def is_batch(self) -> bool:
        return self.batching_method != BatchingMethod.NONE

    def feature_value(self, feature: FeatureDimension) -> Any:
        """提取指定特征维度的当前取值。专供测试使用（Test-only）。"""
        if feature is CachingMethod:
            return self.caching_method
        if feature is BatchingMethod:
            return self.batching_method
        if isinstance(feature, FeatureFlag):
            return self.is_enabled(feature)
        raise TypeError(f"未知的特征维度类型: {feature!r}，必须为 FeatureFlag 成员、CachingMethod 或 BatchingMethod")

    def __str__(self) -> str:
        parts = sorted(flag.value for flag in self.flags)
        if self.caching_method != CachingMethod.NONE:
            parts.append(self.caching_method.value)
        if self.batching_method != BatchingMethod.NONE:
            parts.append(self.batching_method.value)
        if not parts:
            return "no-cache"
        return "+".join(parts)


def validate_mode(mode: Mode) -> None:
    """校验 Mode 的 (caching_method, batching_method) 是否落在合法组合内，非法即抛错。

    只编码与具体驱动无关的硬约束；run_scheduler / run_lockstep 再各自校验自身边界。
    """
    if mode.caching_method is CachingMethod.PAGED and mode.batching_method not in (
        BatchingMethod.CONTINUOUS,
        BatchingMethod.FIXED,
    ):
        raise ValueError("paged-kv-cache 仅支持 continuous-batching 或 fixed-batching")
    if mode.batching_method is BatchingMethod.CONTINUOUS and mode.caching_method is CachingMethod.NONE:
        raise ValueError("continuous-batching 依赖 KV Cache，请将 caching_method 设为 DENSE 或 PAGED")


__all__ = [
    "BatchingMethod",
    "FeatureDimension",
    "FeatureFlag",
    "CachingMethod",
    "Mode",
    "validate_mode",
]
