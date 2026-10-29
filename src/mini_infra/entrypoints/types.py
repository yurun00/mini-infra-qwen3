"""离线生成入口（Offline Entrypoint）专用数据结构与结果聚合。"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from mini_infra.core.types import (
    BatchScheduleInfo,
    GenerationTrace,
    MemoryProfile,
    SequenceOutput,
    StepAggregatedMetrics,
    build_memory_profile,
    compute_step_aggregated_metrics,
    pad_output_tokens,
)
from mini_infra.engine.types import GenerationResult
from mini_infra.profiling import OpLatencyBreakdown, TensorBoardOptions, merge_op_latency_breakdowns
from mini_infra.utils import percentile


@dataclass(frozen=True)
class GenerationOptions:
    """文本生成可选控制项容器。"""

    eos_token_id: int | None = None
    device: torch.device | None = None
    stop_on_eos: bool = True
    pad_token_id: int = 0
    capture_logits: bool = False
    profile: bool = False
    top_kernels: int = 10
    tensorboard_options: TensorBoardOptions | None = None
    max_running_requests: int | None = None


AggFunc = Literal["sum", "mean", "max", "min"] | Callable[[Sequence[float]], float]


def _resolve_agg_fn(agg: AggFunc) -> Callable[[Sequence[float]], float]:
    """将聚合函数名称或可调用对象解析为标准 Callable。"""
    if callable(agg):
        return agg
    agg_str = agg.lower()
    if agg_str == "sum":
        return sum
    elif agg_str == "mean":
        return lambda vals: sum(vals) / len(vals) if vals else 0.0
    elif agg_str == "max":
        return lambda vals: max(vals) if vals else 0.0
    elif agg_str == "min":
        return lambda vals: min(vals) if vals else 0.0
    else:
        raise ValueError(f"Unsupported aggregate function: {agg!r}. Expected 'sum', 'mean', 'max', 'min', or callable.")


@dataclass(frozen=True)
class AggregationOptions:
    """GenerationResult 聚合策略配置。

    支持双层配置：
    - 域级默认聚合：metrics, memory。
    - 细粒度字段覆盖：可针对特定字段覆盖所属域的默认聚合函数。
    """

    # 顶层域默认聚合策略
    metrics: AggFunc = "mean"
    memory: AggFunc = "max"

    # 细粒度 Metrics 字段覆盖（可选）
    e2e_wall_ms: AggFunc | None = None
    e2e_model_ms: AggFunc | None = None
    ttft_ms: AggFunc | None = None
    mean_tpot_ms: AggFunc | None = None
    p50_tpot_ms: AggFunc | None = None
    p95_tpot_ms: AggFunc | None = None
    throughput_tok_s: AggFunc | None = None
    throughput_req_s: AggFunc | None = None

    # 细粒度 Memory 字段覆盖（可选）
    baseline_mb: AggFunc | None = None
    prefill_mb: AggFunc | None = None
    peak_allocated_mb: AggFunc | None = None
    peak_reserved_mb: AggFunc | None = None
    kv_cache_mb: AggFunc | None = None

    def get_agg_fn(self, domain: str, field_name: str | None = None) -> Callable[[Sequence[float]], float]:
        """按优先级解析聚合函数：特定字段覆盖 > 域级默认。"""
        if field_name and hasattr(self, field_name):
            override = getattr(self, field_name)
            if override is not None:
                return _resolve_agg_fn(override)
        domain_default = getattr(self, domain, "mean")
        return _resolve_agg_fn(domain_default)


def aggregate_generation_results(
    runs: list[GenerationResult],
    options: AggregationOptions | None = None,
) -> GenerationResult:
    """把各运行结果聚合为一个 GenerationResult。

    Args:
        runs: 待聚合的生成结果列表。
        options: 聚合策略配置，默认为 AggregationOptions()（指标默认取均值 mean，显存默认取峰值 max）。
    """
    if options is None:
        options = AggregationOptions()

    sequences: list[SequenceOutput] = []
    req_counter = 0
    for run in runs:
        for seq in run.sequences:
            sequences.append(
                SequenceOutput(
                    request_index=req_counter,
                    token_ids=seq.token_ids,
                    prompt_len=seq.prompt_len,
                    output_len=seq.output_len,
                    finish_reason=seq.finish_reason,
                    completion_step=seq.completion_step,
                )
            )
            req_counter += 1

    total_requests = len(sequences)
    step_latencies = [x for run in runs for x in run.trace.step_latencies_ms]
    num_active_tokens = [x for run in runs for x in run.trace.num_active_tokens_per_step]

    # 1. 指标聚合
    e2e_wall_ms = options.get_agg_fn("metrics", "e2e_wall_ms")([r.metrics.e2e_wall_ms for r in runs])
    e2e_model_ms = options.get_agg_fn("metrics", "e2e_model_ms")([r.metrics.e2e_model_ms for r in runs])
    ttft_vals = [r.metrics.ttft_ms for r in runs if r.metrics.ttft_ms is not None]
    mean_tpot_vals = [r.metrics.mean_tpot_ms for r in runs if r.metrics.mean_tpot_ms is not None]
    p50_tpot_vals = [r.metrics.p50_tpot_ms for r in runs if r.metrics.p50_tpot_ms is not None]
    p95_tpot_vals = [r.metrics.p95_tpot_ms for r in runs if r.metrics.p95_tpot_ms is not None]
    ttft_ms = options.get_agg_fn("metrics", "ttft_ms")(ttft_vals) if ttft_vals else None
    mean_tpot_ms = options.get_agg_fn("metrics", "mean_tpot_ms")(mean_tpot_vals) if mean_tpot_vals else None
    p50_tpot_ms = options.get_agg_fn("metrics", "p50_tpot_ms")(p50_tpot_vals) if p50_tpot_vals else None
    p95_tpot_ms = options.get_agg_fn("metrics", "p95_tpot_ms")(p95_tpot_vals) if p95_tpot_vals else None
    throughput_tok_s = options.get_agg_fn("metrics", "throughput_tok_s")([r.metrics.throughput_tok_s for r in runs])
    throughput_req_s = options.get_agg_fn("metrics", "throughput_req_s")([r.metrics.throughput_req_s for r in runs])

    has_cuda = any(r.metrics.gpu_time_ratio is not None for r in runs)
    total_model_ms = sum(r.metrics.e2e_model_ms for r in runs)
    total_wall_ms = sum(r.metrics.e2e_wall_ms for r in runs)
    gpu_time_ratio = min(1.0, total_model_ms / total_wall_ms) if (has_cuda and total_wall_ms > 0) else None

    metrics = StepAggregatedMetrics(
        ttft_ms=ttft_ms,
        mean_tpot_ms=mean_tpot_ms,
        p50_tpot_ms=p50_tpot_ms,
        p95_tpot_ms=p95_tpot_ms,
        e2e_wall_ms=e2e_wall_ms,
        e2e_model_ms=e2e_model_ms,
        throughput_tok_s=throughput_tok_s,
        throughput_req_s=throughput_req_s,
        gpu_time_ratio=gpu_time_ratio,
    )

    # 2. 显存聚合
    baseline_mb_fn = options.get_agg_fn("memory", "baseline_mb")
    prefill_mb_fn = options.get_agg_fn("memory", "prefill_mb")
    peak_allocated_mb_fn = options.get_agg_fn("memory", "peak_allocated_mb")
    peak_reserved_mb_fn = options.get_agg_fn("memory", "peak_reserved_mb")
    kv_cache_mb_fn = options.get_agg_fn("memory", "kv_cache_mb")
    prefill_vals = [r.memory.prefill_mb for r in runs if r.memory.prefill_mb is not None]

    memory = MemoryProfile(
        baseline_mb=baseline_mb_fn([r.memory.baseline_mb for r in runs]),
        prefill_mb=prefill_mb_fn(prefill_vals) if prefill_vals else None,
        peak_allocated_mb=peak_allocated_mb_fn([r.memory.peak_allocated_mb for r in runs]),
        peak_reserved_mb=peak_reserved_mb_fn([r.memory.peak_reserved_mb for r in runs]),
        kv_cache_mb=kv_cache_mb_fn([r.memory.kv_cache_mb for r in runs]),
    )

    op_breakdowns = [r.op_breakdown for r in runs if r.op_breakdown is not None]
    op_breakdown = merge_op_latency_breakdowns(op_breakdowns)

    trace = GenerationTrace(
        step_latencies_ms=step_latencies,
        num_active_tokens_per_step=num_active_tokens,
        logits_trace=None,
    )

    return GenerationResult(
        sequences=sequences,
        metrics=metrics,
        memory=memory,
        trace=trace,
        batch_size=total_requests,
        op_breakdown=op_breakdown,
        batch_info=None,
    )


__all__ = [
    "AggregationOptions",
    "BatchScheduleInfo",
    "GenerationOptions",
    "GenerationResult",
    "GenerationTrace",
    "MemoryProfile",
    "OpLatencyBreakdown",
    "SequenceOutput",
    "StepAggregatedMetrics",
    "aggregate_generation_results",
    "build_memory_profile",
    "compute_step_aggregated_metrics",
    "pad_output_tokens",
    "percentile",
]
