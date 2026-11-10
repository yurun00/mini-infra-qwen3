"""控制面基础数据结构体（CachingMethod、BatchingMethod、SequenceOutput、RequestMetrics、collapse_uniform）。

注意：此模块处于纯 CPU 控制面，严禁 import torch 或引入任何 GPU 张量操作。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CachingMethod(str, Enum):
    """互斥的 KV Cache 策略。"""

    NONE = "none"
    DENSE = "dense-kv-cache"
    PAGED = "paged-kv-cache"

    @classmethod
    def from_str(cls, value: "str | CachingMethod | None") -> "CachingMethod":
        if value is None:
            return cls.NONE
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"未知 KV Cache 策略: {value}，可选策略: {[m.value for m in cls]}") from err

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
    """若序列所有元素相等则折叠为标量，否则返回元组（与 WorkloadProfile 表达对齐）。"""
    if not values:
        return ()
    first = values[0]
    if all(v == first for v in values):
        return first
    return tuple(values)


__all__ = [
    "BatchingMethod",
    "CachingMethod",
    "RequestMetrics",
    "SequenceOutput",
    "collapse_uniform",
]
