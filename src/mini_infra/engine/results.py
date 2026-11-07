"""生成结果容器、输出对齐与多轮结果聚合。"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mini_infra.core.types import (
    BatchingMethod,
    CachingMethod,
    RequestMetrics,
    SequenceOutput,
    collapse_uniform,
)
from mini_infra.engine.metrics import (
    BatchScheduleInfo,
    GenerationTrace,
    KVCacheProfile,
    MemoryProfile,
    ScheduleEvent,
    ScheduleMetrics,
    StepAggregatedMetrics,
    merge_paged_cache_profiles,
)
from mini_infra.profiling import (
    OpLatencyBreakdown,
    merge_op_latency_breakdowns,
)

if TYPE_CHECKING:
    from mini_infra.bench.workloads import WorkloadSpec


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
            if seq.request_idx == request_idx:
                return torch.tensor(seq.token_ids, dtype=torch.long)
        raise IndexError(f"Request index {request_idx} not found in sequences")

    if request_idx is not None:
        seq_map = {seq.request_idx: seq for seq in sequences}
        selected_seqs: Sequence[SequenceOutput] = []
        for target_idx in request_idx:
            if target_idx not in seq_map:
                raise IndexError(f"Request index {target_idx} not found in sequences")
            selected_seqs.append(seq_map[target_idx])
    else:
        selected_seqs = sequences

    if not selected_seqs:
        return torch.empty((0, 0), dtype=torch.long)

    max_output_len = max((seq.output_len for seq in selected_seqs), default=0)
    tensor = torch.full((len(selected_seqs), max_output_len), pad_token_id, dtype=torch.long)
    for row_idx, seq in enumerate(selected_seqs):
        if seq.token_ids:
            tensor[row_idx, : seq.output_len] = torch.tensor(seq.token_ids, dtype=torch.long)
    return tensor


@dataclass(frozen=True)
class GenerationResult:
    """所有文本生成模式的公共结果基类契约（彻底属性化，不可变强类型容器）。"""

    sequences: list[SequenceOutput]
    metrics: StepAggregatedMetrics[float]
    memory: MemoryProfile
    trace: GenerationTrace
    batch_size: int
    op_breakdown: OpLatencyBreakdown | None = None
    batch_info: BatchScheduleInfo[float] | None = None

    def padded_output_tokens(
        self,
        request_idx: int | Sequence[int] | None = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """从 sequences 动态计算并返回对齐输出张量。"""
        return pad_output_tokens(self.sequences, request_idx=request_idx, pad_token_id=pad_token_id)


@dataclass(frozen=True)
class ScheduleResult(GenerationResult):
    """批调度运行的顶层不可变结果容器（继承公共 GenerationResult 并扩展非空调度专属字段）。"""

    schedule_metrics: ScheduleMetrics[float] = field(kw_only=True)
    events: list[ScheduleEvent] = field(kw_only=True)
    request_metrics: list[RequestMetrics] = field(kw_only=True)
    batching_method: BatchingMethod = BatchingMethod.CONTINUOUS
    caching_method: CachingMethod = CachingMethod.DENSE

    @property
    def step_agg_metrics(self) -> StepAggregatedMetrics[float]:
        """调度器视角下的步级聚合指标别名（与基类 metrics 指向同一对象，用以同 schedule_metrics 区分）。"""
        return self.metrics

    @property
    def workload(self) -> "WorkloadSpec":
        """从单次调度结果提取负载规格 WorkloadSpec（相等比较时忽略 repeats 与 warmup）。"""
        from mini_infra.bench.workloads import WorkloadSpec
        from mini_infra.engine.mode import Mode

        prompt_lengths = [s.prompt_len for s in self.sequences]
        output_budgets = [r.num_output_tokens for r in self.request_metrics]
        arrivals = [r.arrival_ms for r in self.request_metrics]
        return WorkloadSpec(
            mode=Mode(self.caching_method, self.batching_method),
            batch_size=self.schedule_metrics.request_count,
            prompt_len=collapse_uniform(prompt_lengths),
            max_new_tokens=collapse_uniform(output_budgets),
            max_running_requests=self.schedule_metrics.max_running_requests,
            arrival_times_ms=collapse_uniform(arrivals),
        )


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

    metrics: AggFunc = "mean"
    memory: AggFunc = "max"

    e2e_wall_ms: AggFunc | None = None
    e2e_model_ms: AggFunc | None = None
    ttft_ms: AggFunc | None = None
    mean_tpot_ms: AggFunc | None = None
    p50_tpot_ms: AggFunc | None = None
    p95_tpot_ms: AggFunc | None = None
    throughput_tok_s: AggFunc | None = None
    throughput_req_s: AggFunc | None = None

    baseline_mb: AggFunc | None = None
    prefill_mb: AggFunc | None = None
    peak_allocated_mb: AggFunc | None = None
    peak_reserved_mb: AggFunc | None = None
    peak_live_kv_cache_mb: AggFunc | None = None

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
    """把各运行结果聚合为一个 GenerationResult。"""
    if options is None:
        options = AggregationOptions()

    sequences: list[SequenceOutput] = []
    request_counter = 0
    for run in runs:
        for seq in run.sequences:
            sequences.append(
                SequenceOutput(
                    request_idx=request_counter,
                    token_ids=seq.token_ids,
                    prompt_len=seq.prompt_len,
                    output_len=seq.output_len,
                    finish_reason=seq.finish_reason,
                    completion_step=seq.completion_step,
                )
            )
            request_counter += 1

    total_requests = len(sequences)
    step_latencies = [x for run in runs for x in run.trace.step_latencies_ms]
    num_active_tokens = [x for run in runs for x in run.trace.num_active_tokens_per_step]

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

    baseline_mb_fn = options.get_agg_fn("memory", "baseline_mb")
    prefill_mb_fn = options.get_agg_fn("memory", "prefill_mb")
    peak_allocated_mb_fn = options.get_agg_fn("memory", "peak_allocated_mb")
    peak_reserved_mb_fn = options.get_agg_fn("memory", "peak_reserved_mb")
    peak_live_kv_cache_mb_fn = options.get_agg_fn("memory", "peak_live_kv_cache_mb")
    prefill_vals = [r.memory.prefill_mb for r in runs if r.memory.prefill_mb is not None]

    memory = MemoryProfile(
        baseline_mb=baseline_mb_fn([r.memory.baseline_mb for r in runs]),
        prefill_mb=prefill_mb_fn(prefill_vals) if prefill_vals else None,
        peak_allocated_mb=peak_allocated_mb_fn([r.memory.peak_allocated_mb for r in runs]),
        peak_reserved_mb=peak_reserved_mb_fn([r.memory.peak_reserved_mb for r in runs]),
        kv_cache=KVCacheProfile(
            peak_live_kv_cache_mb=peak_live_kv_cache_mb_fn([r.memory.kv_cache.peak_live_kv_cache_mb for r in runs]),
            peak_kv_cache_pool_mb=max((r.memory.kv_cache.peak_kv_cache_pool_mb for r in runs), default=0.0),
            cache_realign_moved_bytes=sum(r.memory.kv_cache.cache_realign_moved_bytes for r in runs),
            accessed_kv_cache_tokens=sum(r.memory.kv_cache.accessed_kv_cache_tokens for r in runs),
            paged_cache=merge_paged_cache_profiles(
                [r.memory.kv_cache.paged_cache for r in runs if r.memory.kv_cache.paged_cache is not None]
            ),
        ),
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
    "AggFunc",
    "AggregationOptions",
    "GenerationResult",
    "ScheduleResult",
    "aggregate_generation_results",
    "pad_output_tokens",
]
