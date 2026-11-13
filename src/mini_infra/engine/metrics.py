"""生成性能度量、显存画像与批调度明细结构体及聚合函数。"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import torch

from mini_infra.core.request import RequestState
from mini_infra.core.scheduler import SchedulePhase
from mini_infra.core.types import (
    BatchingMethod,
    DenseCacheConfig,
    KVCacheConfig,
)
from mini_infra.utils import percentile

_MetricT = TypeVar("_MetricT")


@dataclass(frozen=True)
class GenerationTrace:
    """生成时序逐步追踪轨迹（Offline 与 Engine 共享）。"""

    step_latencies_ms: list[float]
    num_active_tokens_per_step: list[int]
    logits_trace: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        if self.logits_trace is not None:
            for idx, t in enumerate(self.logits_trace):
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"logits_trace 元素必须为 torch.Tensor，但在索引 {idx} 处检测到 {type(t).__name__}")
                if t.device.type != "cpu":
                    raise ValueError(
                        f"logits_trace 元素必须驻留在 CPU 设备上，但在索引 {idx} 处检测到 device='{t.device}'"
                    )


@dataclass(frozen=True)
class PagedCacheProfile:
    """Paged KV Cache 专属 physical block pool 画像。"""

    block_size: int = 0
    bytes_per_block: int = 0
    num_blocks: int = 0
    peak_used_blocks: int = 0
    block_cold_allocation_count: int = 0
    block_reuse_count: int = 0
    block_release_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "bytes_per_block": self.bytes_per_block,
            "num_blocks": self.num_blocks,
            "peak_used_blocks": self.peak_used_blocks,
            "block_cold_allocation_count": self.block_cold_allocation_count,
            "block_reuse_count": self.block_reuse_count,
            "block_release_count": self.block_release_count,
        }


@dataclass(frozen=True)
class KVCacheProfile:
    """KV Cache 画像：活跃/池容量、扩容搬移、带宽访问，及 Paged 专属 physical block pool 明细。"""

    peak_live_kv_cache_mb: float = 0.0
    peak_kv_cache_pool_mb: float = 0.0
    cache_realign_moved_bytes: int = 0
    accessed_kv_cache_tokens: int = 0
    paged_cache: PagedCacheProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "peak_live_kv_cache_mb": self.peak_live_kv_cache_mb,
            "peak_kv_cache_pool_mb": self.peak_kv_cache_pool_mb,
            "cache_realign_moved_bytes": self.cache_realign_moved_bytes,
            "accessed_kv_cache_tokens": self.accessed_kv_cache_tokens,
        }
        if self.paged_cache is not None:
            data["paged_cache"] = self.paged_cache.to_dict()
        return data


@dataclass(frozen=True)
class MemoryProfile:
    """显存监控强类型结构体（统一单位 MB，Offline、Serving 与 Benchmark 共享）。"""

    baseline_mb: float = 0.0
    prefill_mb: float | None = 0.0
    peak_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    weights_mb: float | None = None
    kv_cache: KVCacheProfile = field(default_factory=KVCacheProfile)

    @property
    def dynamic_peak_mb(self) -> float:
        return max(0.0, self.peak_allocated_mb - self.baseline_mb)

    @property
    def fragmentation_mb(self) -> float:
        return max(0.0, self.peak_reserved_mb - self.peak_allocated_mb)

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
            "kv_cache": self.kv_cache.to_dict(),
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
        kv_cache=KVCacheProfile(peak_live_kv_cache_mb=cache_bytes / mib),
        weights_mb=weights_mb,
    )


def summarize_repeated_paged_cache_profiles(paged: Sequence[PagedCacheProfile]) -> PagedCacheProfile | None:
    """汇总同一 Case 多轮重复运行的 Paged physical block pool 画像：固定配置保持一致，峰值取 max，计数取 sum。"""
    if not paged:
        return None
    first = paged[0]
    assert all(
        (p.block_size, p.bytes_per_block, p.num_blocks) == (first.block_size, first.bytes_per_block, first.num_blocks)
        for p in paged
    ), (
        "All paged cache profiles must share identical physical pool configuration "
        "(block_size, bytes_per_block, num_blocks)"
    )
    return PagedCacheProfile(
        block_size=first.block_size,
        bytes_per_block=first.bytes_per_block,
        num_blocks=first.num_blocks,
        peak_used_blocks=max((p.peak_used_blocks for p in paged), default=0),
        block_cold_allocation_count=sum(p.block_cold_allocation_count for p in paged),
        block_reuse_count=sum(p.block_reuse_count for p in paged),
        block_release_count=sum(p.block_release_count for p in paged),
    )


def summarize_repeated_kv_cache_profiles(profiles: Sequence[KVCacheProfile]) -> KVCacheProfile:
    """汇总同一 Case 多轮重复运行的 KV Cache 画像：容量峰值取 max，累计搬移/访问取 sum，
    Paged 明细用 summarize_repeated_paged_cache_profiles。
    """
    return KVCacheProfile(
        peak_live_kv_cache_mb=max((p.peak_live_kv_cache_mb for p in profiles), default=0.0),
        peak_kv_cache_pool_mb=max((p.peak_kv_cache_pool_mb for p in profiles), default=0.0),
        cache_realign_moved_bytes=sum(p.cache_realign_moved_bytes for p in profiles),
        accessed_kv_cache_tokens=sum(p.accessed_kv_cache_tokens for p in profiles),
        paged_cache=summarize_repeated_paged_cache_profiles(
            [p.paged_cache for p in profiles if p.paged_cache is not None]
        ),
    )


@dataclass(frozen=True)
class BatchScheduleInfo(Generic[_MetricT]):
    """批处理打包（Padding 效率）与 slot_utilization 明细（支持标量 float 或统计分布 DistributionStats）。"""

    padding_efficiency: _MetricT
    slot_utilization: _MetricT
    running_capacity_utilization: _MetricT | None = None

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


@dataclass(frozen=True)
class ScheduleEvent:
    """一次模型调用（Prefill 或 Decode）及其搬移开销的调度事件记录。"""

    phase: SchedulePhase
    admitted_request_ids: tuple[int, ...]  # 本事件入场的 request_idx
    finished_request_ids: tuple[int, ...]  # 本事件结束的 request_idx
    physical_batch_size: int  # 模型实际执行的行数（含空闲/已结束 batch slot）
    num_active_requests: int  # 事件期间仍在占用 batch slot 的请求数
    num_generated_tokens: int  # 本事件产出的有效 Token 数
    num_scheduled_tokens: int  # 本事件计费 Token 数（透传 SchedulerOutput：PREFILL 为 padded prompt token 数）
    step_ms: float
    model_ms: float
    cache_management_ms: float
    live_kv_cache_bytes: int
    kv_cache_pool_bytes: int = 0
    cache_realign_moved_bytes: int = 0
    accessed_kv_cache_tokens: int = 0


@dataclass(frozen=True)
class ScheduleMetrics(Generic[_MetricT]):
    """一次或多次调度运行的调度器专属聚合指标（支持单次标量 float 或多轮评测统计分布 DistributionStats）。"""

    request_count: int
    max_running_requests: int
    makespan_ms: _MetricT
    busy_ms: _MetricT
    cache_management_ms: _MetricT
    cache_management_fraction: _MetricT
    queue_ms_p50: _MetricT
    queue_ms_p95: _MetricT
    ttft_ms_p50: _MetricT
    ttft_ms_p95: _MetricT
    end_to_end_ms_p50: _MetricT
    end_to_end_ms_p95: _MetricT
    itl_ms_p50: _MetricT
    itl_ms_p95: _MetricT
    itl_ms_max: _MetricT


def summarize_schedule(
    batching_method: BatchingMethod,
    states: Sequence[RequestState],
    events: Sequence[ScheduleEvent],
    max_running_requests: int,
    peak_allocated_mb: float,
    peak_reserved_mb: float,
    baseline_mb: float = 0.0,
    e2e_wall_ms: float | None = None,
    is_cuda: bool = False,
    kv_cache_config: KVCacheConfig | None = None,
    paged_cache_profile: PagedCacheProfile | None = None,
) -> Any:
    """把运行结束后的请求状态与事件流汇总为 ScheduleResult（固定批次与 Continuous 共用）。"""
    from mini_infra.engine.results import ScheduleResult

    resolved_kv_config = kv_cache_config if kv_cache_config is not None else DenseCacheConfig()

    requests = [state.to_request_metrics() for state in states]
    completion_step_by_request_idx = {
        request_idx: step_idx + 1 for step_idx, e in enumerate(events) for request_idx in e.finished_request_ids
    }
    sequences = [
        state.to_sequence_output(
            completion_step=completion_step_by_request_idx.get(state.request.request_idx, len(events))
        )
        for state in states
    ]

    makespan_ms = max(r.completion_ms for r in requests) - min(r.arrival_ms for r in requests)
    busy_ms = sum(e.step_ms for e in events)
    model_ms = sum(e.model_ms for e in events)
    cache_ms = sum(e.cache_management_ms for e in events)
    prefill_ms = sum(e.step_ms for e in events if e.phase is SchedulePhase.PREFILL)
    e2e_wall_ms = e2e_wall_ms if e2e_wall_ms is not None else busy_ms
    num_output_tokens = sum(r.num_output_tokens for r in requests)
    total_executed_slots = sum(e.physical_batch_size for e in events)
    capacity_area = sum(e.num_active_requests * e.step_ms for e in events)
    itls = [itl for r in requests for itl in r.inter_token_latencies_ms]
    mib = 1024**2

    slot_util = num_output_tokens / total_executed_slots if total_executed_slots else 0.0
    running_capacity_util = capacity_area / (max_running_requests * busy_ms) if busy_ms else 0.0
    peak_live_kv_cache_mb = max((e.live_kv_cache_bytes for e in events), default=0) / mib
    peak_kv_cache_pool_mb = max((e.kv_cache_pool_bytes for e in events), default=0) / mib
    cache_realign_moved_bytes = sum(e.cache_realign_moved_bytes for e in events)
    accessed_kv_cache_tokens = sum(e.accessed_kv_cache_tokens for e in events)

    num_prompt_tokens = sum(state.request.prompt_len for state in states)
    num_padded_prompt_tokens = sum(e.num_scheduled_tokens for e in events if e.phase is SchedulePhase.PREFILL)
    if num_padded_prompt_tokens == 0:
        num_padded_prompt_tokens = num_prompt_tokens
    padding_efficiency = num_prompt_tokens / num_padded_prompt_tokens if num_padded_prompt_tokens else 0.0

    gpu_time_ratio = (min(1.0, model_ms / e2e_wall_ms) if e2e_wall_ms > 0 else 0.0) if is_cuda else None
    step_agg_metrics: StepAggregatedMetrics[float] = StepAggregatedMetrics(
        e2e_wall_ms=e2e_wall_ms,
        e2e_model_ms=model_ms,
        throughput_tok_s=num_output_tokens * 1000 / busy_ms if busy_ms else 0.0,
        throughput_req_s=len(requests) * 1000 / busy_ms if busy_ms else 0.0,
        gpu_time_ratio=gpu_time_ratio,
        effective_prompt_tok_s=num_prompt_tokens * 1000 / prefill_ms if prefill_ms else 0.0,
        padded_prompt_tok_s=num_padded_prompt_tokens * 1000 / prefill_ms if prefill_ms else 0.0,
        ttft_ms=None,
        mean_tpot_ms=None,
        p50_tpot_ms=None,
        p95_tpot_ms=None,
    )
    memory = MemoryProfile(
        baseline_mb=baseline_mb,
        prefill_mb=None,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
        kv_cache=KVCacheProfile(
            peak_live_kv_cache_mb=peak_live_kv_cache_mb,
            peak_kv_cache_pool_mb=peak_kv_cache_pool_mb,
            cache_realign_moved_bytes=cache_realign_moved_bytes,
            accessed_kv_cache_tokens=accessed_kv_cache_tokens,
            paged_cache=paged_cache_profile,
        ),
    )
    batch_info: BatchScheduleInfo[float] = BatchScheduleInfo(
        padding_efficiency=padding_efficiency,
        slot_utilization=slot_util,
        running_capacity_utilization=running_capacity_util,
    )
    schedule_metrics: ScheduleMetrics[float] = ScheduleMetrics(
        request_count=len(requests),
        max_running_requests=max_running_requests,
        makespan_ms=makespan_ms,
        busy_ms=busy_ms,
        cache_management_ms=cache_ms,
        cache_management_fraction=cache_ms / busy_ms if busy_ms else 0.0,
        queue_ms_p50=percentile([r.queue_ms for r in requests], 50),
        queue_ms_p95=percentile([r.queue_ms for r in requests], 95),
        ttft_ms_p50=percentile([r.ttft_ms for r in requests], 50),
        ttft_ms_p95=percentile([r.ttft_ms for r in requests], 95),
        end_to_end_ms_p50=percentile([r.end_to_end_ms for r in requests], 50),
        end_to_end_ms_p95=percentile([r.end_to_end_ms for r in requests], 95),
        itl_ms_p50=percentile(itls, 50),
        itl_ms_p95=percentile(itls, 95),
        itl_ms_max=max(itls, default=0.0),
    )
    trace = GenerationTrace(
        step_latencies_ms=[event.step_ms for event in events],
        num_active_tokens_per_step=[event.num_generated_tokens for event in events],
        logits_trace=None,
    )
    return ScheduleResult(
        sequences=sequences,
        metrics=step_agg_metrics,
        memory=memory,
        trace=trace,
        batch_size=len(requests),
        batch_info=batch_info,
        schedule_metrics=schedule_metrics,
        events=list(events),
        request_metrics=requests,
        batching_method=batching_method,
        kv_cache_config=resolved_kv_config,
    )


__all__ = [
    "BatchScheduleInfo",
    "GenerationTrace",
    "KVCacheProfile",
    "MemoryProfile",
    "PagedCacheProfile",
    "ScheduleEvent",
    "ScheduleMetrics",
    "StepAggregatedMetrics",
    "build_memory_profile",
    "compute_step_aggregated_metrics",
    "summarize_repeated_kv_cache_profiles",
    "summarize_repeated_paged_cache_profiles",
    "summarize_schedule",
]
