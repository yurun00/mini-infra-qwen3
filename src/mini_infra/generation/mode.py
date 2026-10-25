"""运行模式与特性标志定义。"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FeatureFlag(str, Enum):
    """支持的所有布尔型特性标志（Feature Flag）；每个成员对应一个可独立开启/关闭（enable/disable）的二值特性。"""

    KV_CACHE = "kv-cache"

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
class Mode:
    """运行模式结构体：内部存储已启用的 FeatureFlag 集合及批处理策略（BatchingMethod）。

    - flags: 正交的功能开关集合（如 FeatureFlag.KV_CACHE）。
    - batching_method: 互斥的批处理策略（BatchingMethod.NONE / FIXED / CONTINUOUS）。
    """

    flags: frozenset[FeatureFlag] = frozenset()
    batching_method: BatchingMethod = BatchingMethod.NONE

    def __init__(
        self,
        flags: "Iterable[FeatureFlag | str] | None" = None,
        batching_method: BatchingMethod | str = BatchingMethod.NONE,
    ):
        if flags is None:
            flag_set = frozenset()
        elif isinstance(flags, frozenset):
            flag_set = flags
        else:
            flag_set = frozenset(FeatureFlag.from_str(f) for f in flags)

        method = BatchingMethod.from_str(batching_method)

        object.__setattr__(self, "flags", flag_set)
        object.__setattr__(self, "batching_method", method)

    def is_enabled(self, flag: FeatureFlag | str) -> bool:
        """检查指定的 FeatureFlag 是否启用。"""
        if isinstance(flag, str):
            flag = FeatureFlag.from_str(flag)
        return flag in self.flags

    def with_flag(self, flag: FeatureFlag, enabled: bool = True) -> "Mode":
        """返回指定 FeatureFlag 状态修改后的新 Mode 实例。"""
        new_flags = (self.flags | {flag}) if enabled else (self.flags - {flag})
        return Mode(flags=new_flags, batching_method=self.batching_method)

    def with_batching_method(self, batching_method: BatchingMethod | str) -> "Mode":
        """返回指定 batching_method 策略修改后的新 Mode 实例。"""
        return Mode(flags=self.flags, batching_method=batching_method)

    @property
    def kv_cache(self) -> bool:
        return self.is_enabled(FeatureFlag.KV_CACHE)

    @property
    def fixed_batching(self) -> bool:
        return self.batching_method == BatchingMethod.FIXED

    @property
    def continuous_batching(self) -> bool:
        return self.batching_method == BatchingMethod.CONTINUOUS

    @property
    def is_batch(self) -> bool:
        return self.batching_method != BatchingMethod.NONE

    def to_dict(self) -> dict[str, Any]:
        """将 mode 中的所有 feature flag 状态及 batching 策略转换为字典。"""
        result = {flag.value: self.is_enabled(flag) for flag in FeatureFlag}
        result["batching_method"] = self.batching_method.value
        return result

    def __contains__(self, item: FeatureFlag | BatchingMethod | str) -> bool:
        if isinstance(item, BatchingMethod):
            return self.batching_method == item
        if isinstance(item, str):
            if item in [b.value for b in BatchingMethod]:
                return self.batching_method == BatchingMethod(item)
            return FeatureFlag.from_str(item) in self.flags
        return item in self.flags

    def __str__(self) -> str:
        parts = sorted(flag.value for flag in self.flags)
        if self.batching_method != BatchingMethod.NONE:
            parts.append(self.batching_method.value)
        if not parts:
            return "no-cache"
        return "+".join(parts)
