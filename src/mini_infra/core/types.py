"""跨层共享基础领域结构体（BatchingMethod、SequenceOutput、RequestMetrics、StepAggregatedMetrics、MemoryProfile、BatchScheduleInfo、GenerationTrace）。"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

import torch

from mini_infra.utils import percentile

_MetricT = TypeVar("_MetricT")


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

    request_index: int
    token_ids: list[int]
    prompt_len: int
    output_len: int
    finish_reason: str  # 结束原因: "stop"（命中 EOS）或 "max_tokens"（达到 Token 预算上限）
    completion_step: int | None = None


@dataclass(frozen=True)
class GenerationTrace:
    """生成时序逐步追踪轨迹（Offline 与 Engine 共享）。"""

    step_latencies_ms: list[float]
    num_active_tokens_per_step: list[int]
    logits_trace: list[torch.Tensor] | None = None


@dataclass(frozen=True)
class RequestMetrics:
    """单个请求在逻辑时间线上的关键时间戳与延迟指标（单位 ms）。"""

    request_index: int
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


def pad_output_tokens(
    sequences: Sequence[SequenceOutput],
    request_idx: int | Sequence[int] | None = None,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """从 SequenceOutput 列表动态计算并返回对齐输出张量（GenerationResult 与 ScheduleResult 共用）。

    Args:
        sequences: 请求序列输出列表。
        request_idx: 单个请求 index (int)、请求 index 序列 (list/tuple) 或 None。
            - 若传入单个 int：返回该请求的 1D 输出张量 [output_len]；若未找到引发 IndexError。
            - 若传入 Sequence[int]：按指定顺序返回选定请求的 2D 对齐张量 [len(request_idx), max_output_len]。
            - 若为 None（默认）：返回整个批次的 2D 对齐张量 [B, max_output_len]。
        pad_token_id: 填充 token ID，默认为 0。仅在 2D 对齐时生效。
    """
    if isinstance(request_idx, int):
        for seq in sequences:
            if seq.request_index == request_idx:
                return torch.tensor(seq.token_ids, dtype=torch.long)
        raise IndexError(f"Request index {request_idx} not found in sequences")

    if request_idx is not None:
        seq_map = {seq.request_index: seq for seq in sequences}
        selected_seqs: Sequence[SequenceOutput] = []
        for r_idx in request_idx:
            if r_idx not in seq_map:
                raise IndexError(f"Request index {r_idx} not found in sequences")
            selected_seqs.append(seq_map[r_idx])
    else:
        selected_seqs = sequences

    if not selected_seqs:
        return torch.empty((0, 0), dtype=torch.long)

    max_len = max((seq.output_len for seq in selected_seqs), default=0)
    tensor = torch.full((len(selected_seqs), max_len), pad_token_id, dtype=torch.long)
    for idx, seq in enumerate(selected_seqs):
        if seq.token_ids:
            tensor[idx, : seq.output_len] = torch.tensor(seq.token_ids, dtype=torch.long)
    return tensor


@dataclass(frozen=True)
class MemoryProfile:
    """显存监控强类型结构体（统一单位 MB，Offline、Serving 与 Benchmark 共享）。"""

    baseline_mb: float = 0.0
    prefill_mb: float | None = 0.0
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    kv_cache_mb: float = 0.0
    weights_mb: float | None = None

    @property
    def dynamic_peak_mb(self) -> float:
        return max(0.0, self.peak_allocated_mb - self.baseline_mb)

    @property
    def fragmentation_mb(self) -> float:
        return max(0.0, self.peak_reserved_mb - self.peak_allocated_mb)

    @property
    def peak_live_cache_mb(self) -> float:
        """Serving 口径下的活跃 KV Cache 峰值别名（与 kv_cache_mb 等价）。"""
        return self.kv_cache_mb

    def to_dict(self) -> dict[str, Any]:
        """导出为标准评测记录字典结构。"""
        return {
            "weights_mb": self.weights_mb,
            "baseline_allocated_mb": self.baseline_mb,
            "prefill_allocated_mb": self.prefill_mb,
            "peak_allocated_mb": self.peak_allocated_mb,
            "dynamic_peak_mb": self.dynamic_peak_mb,
            "peak_reserved_mb": self.peak_reserved_mb,
            "fragmentation_mb": self.fragmentation_mb,
            "kv_cache_mb": self.kv_cache_mb,
        }


def build_memory_profile(
    baseline_allocated_bytes: int,
    memory_after_prefill_bytes: int,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    cache_bytes: int,
    weights_mb: float | None = None,
) -> MemoryProfile:
    """将监控字节数转换为以 MB 为单位的 MemoryProfile。"""
    mib = 1024**2
    return MemoryProfile(
        baseline_mb=baseline_allocated_bytes / mib,
        prefill_mb=memory_after_prefill_bytes / mib,
        peak_allocated_mb=peak_allocated_bytes / mib,
        peak_reserved_mb=peak_reserved_bytes / mib,
        kv_cache_mb=cache_bytes / mib,
        weights_mb=weights_mb,
    )


@dataclass(frozen=True)
class BatchScheduleInfo(Generic[_MetricT]):
    """批处理打包（Padding 效率）与槽位利用率明细（支持单次运行标量 float 或多轮评测统计分布 DistributionStats）。"""

    padding_efficiency: _MetricT
    slot_utilization: _MetricT
    running_capacity_utilization: _MetricT | None = None

    @property
    def execution_slot_utilization(self) -> _MetricT:
        """Serving 口径下的执行槽位利用率别名（与 slot_utilization 等价）。"""
        return self.slot_utilization

    def to_dict(self) -> dict[str, Any]:
        """导出为批处理打包特征字典结构。"""
        data: dict[str, Any] = {
            "padding_efficiency": (
                self.padding_efficiency.to_dict()
                if hasattr(self.padding_efficiency, "to_dict")
                else self.padding_efficiency
            ),
            "slot_utilization": (
                self.slot_utilization.to_dict() if hasattr(self.slot_utilization, "to_dict") else self.slot_utilization
            ),
        }
        if self.running_capacity_utilization is not None:
            data["running_capacity_utilization"] = (
                self.running_capacity_utilization.to_dict()
                if hasattr(self.running_capacity_utilization, "to_dict")
                else self.running_capacity_utilization
            )
        return data


@dataclass(frozen=True)
class StepAggregatedMetrics(Generic[_MetricT]):
    """生成性能度量指标（支持单次运行标量 float 或多轮评测统计分布 DistributionStats）。"""

    ttft_ms: _MetricT | None = None
    mean_tpot_ms: _MetricT | None = None
    p50_tpot_ms: _MetricT | None = None
    p95_tpot_ms: _MetricT | None = None
    e2e_wall_ms: _MetricT = 0.0  # type: ignore[assignment]
    e2e_model_ms: _MetricT = 0.0  # type: ignore[assignment]
    throughput_tok_s: _MetricT = 0.0  # type: ignore[assignment]
    throughput_req_s: _MetricT = 0.0  # type: ignore[assignment]
    gpu_time_ratio: _MetricT | None = None
    effective_prompt_tok_s: _MetricT | None = None
    padded_prompt_tok_s: _MetricT | None = None


def compute_step_aggregated_metrics(
    step_latencies_ms: list[float],
    e2e_wall_ms: float,
    e2e_model_ms: float | None = None,
    *,
    total_tokens: int,
    batch_size: int,
    is_cuda: bool = False,
    num_prompt_tokens: int | None = None,
    num_padded_prompt_tokens: int | None = None,
) -> StepAggregatedMetrics:
    """计算时延分位数、吞吐率与 GPU 利用率纯标量指标（由各 step 统计聚合而成）。"""
    decode_latencies = step_latencies_ms[1:] if len(step_latencies_ms) > 1 else []
    ttft_ms = step_latencies_ms[0] if step_latencies_ms else 0.0
    mean_tpot_ms = (sum(decode_latencies) / len(decode_latencies)) if decode_latencies else 0.0
    p50_tpot_ms = percentile(decode_latencies, 50)
    p95_tpot_ms = percentile(decode_latencies, 95)
    if e2e_model_ms is None:
        e2e_model_ms = sum(step_latencies_ms)
    throughput_tok_s = total_tokens / (e2e_model_ms / 1000) if e2e_model_ms > 0 else 0.0
    throughput_req_s = batch_size * 1000 / e2e_model_ms if e2e_model_ms > 0 else 0.0
    gpu_time_ratio = (min(1.0, e2e_model_ms / e2e_wall_ms) if e2e_wall_ms > 0 else 0.0) if is_cuda else None
    effective_prompt_tok_s = (
        (num_prompt_tokens * 1000.0 / ttft_ms if ttft_ms > 0 else 0.0) if num_prompt_tokens is not None else None
    )
    padded_prompt_tok_s = (
        (num_padded_prompt_tokens * 1000.0 / ttft_ms if ttft_ms > 0 else 0.0)
        if num_padded_prompt_tokens is not None
        else None
    )

    return StepAggregatedMetrics(
        ttft_ms=ttft_ms,
        mean_tpot_ms=mean_tpot_ms,
        p50_tpot_ms=p50_tpot_ms,
        p95_tpot_ms=p95_tpot_ms,
        e2e_wall_ms=e2e_wall_ms,
        e2e_model_ms=e2e_model_ms,
        throughput_tok_s=throughput_tok_s,
        throughput_req_s=throughput_req_s,
        gpu_time_ratio=gpu_time_ratio,
        effective_prompt_tok_s=effective_prompt_tok_s,
        padded_prompt_tok_s=padded_prompt_tok_s,
    )


def collapse_uniform(values: Sequence[Any]) -> Any:
    """若序列所有元素相等则折叠为标量，否则返回元组（与 WorkloadSpec 表达对齐）。"""
    if not values:
        return ()
    first = values[0]
    if all(v == first for v in values):
        return first
    return tuple(values)


__all__ = [
    "BatchScheduleInfo",
    "BatchingMethod",
    "GenerationTrace",
    "MemoryProfile",
    "RequestMetrics",
    "SequenceOutput",
    "StepAggregatedMetrics",
    "build_memory_profile",
    "collapse_uniform",
    "compute_step_aggregated_metrics",
    "pad_output_tokens",
]
