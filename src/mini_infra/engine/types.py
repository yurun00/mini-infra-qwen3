"""生成结果强类型结构体与指标计算。"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from mini_infra.core.types import BatchScheduleInfo, ScheduleResult, SequenceOutput
from mini_infra.profiling import OpLatencyBreakdown, TensorBoardOptions, merge_op_latency_breakdowns
from mini_infra.utils import percentile


@dataclass(frozen=True)
class OfflineGenerationOptions:
    """离线文本生成可选控制项容器。"""

    eos_token_id: int | None = None
    device: torch.device | None = None
    stop_on_eos: bool = True
    pad_token_id: int = 0
    capture_logits: bool = False
    profile: bool = False
    top_kernels: int = 10
    tensorboard_options: TensorBoardOptions | None = None
    max_running_requests: int | None = None


@dataclass(frozen=True)
class MemoryProfile:
    """显存监控强类型结构体（统一单位 MB）。"""

    baseline_mb: float
    prefill_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    kv_cache_mb: float

    @property
    def dynamic_peak_mb(self) -> float:
        return max(0.0, self.peak_allocated_mb - self.baseline_mb)

    @property
    def fragmentation_mb(self) -> float:
        return max(0.0, self.peak_reserved_mb - self.peak_allocated_mb)


@dataclass(frozen=True)
class StepAggregatedMetrics:
    """生成性能度量纯标量指标（由 step-level 统计聚合而成）。"""

    ttft_ms: float
    mean_tpot_ms: float
    p50_tpot_ms: float
    p95_tpot_ms: float
    e2e_wall_ms: float
    e2e_model_ms: float
    throughput_tok_s: float
    throughput_req_s: float
    gpu_time_ratio: float | None = None


@dataclass(frozen=True)
class GenerationTrace:
    """生成时序逐步追踪轨迹。"""

    step_latencies_ms: list[float]
    active_tokens_per_step: list[int]
    logits_trace: list[torch.Tensor] | None = None


@dataclass(frozen=True)
class GenerationResult:
    """单次生成结果顶层不可变强类型容器。

    彻底属性化（Dot-attribute Access），零字典穿透或代理属性访问。
    """

    sequences: list[SequenceOutput]
    metrics: StepAggregatedMetrics
    memory: MemoryProfile
    trace: GenerationTrace
    batch_size: int
    op_breakdown: OpLatencyBreakdown | None = None
    batch_info: BatchScheduleInfo | None = None

    def padded_output_tokens(
        self,
        request_idx: int | Sequence[int] | None = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """从 sequences 动态计算并返回对齐输出张量。

        Args:
            request_idx: 单个请求 index (int)、请求 index 序列 (list/tuple) 或 None。
                - 若传入单个 int：返回该请求的 1D 输出张量 [output_len]；若未找到引发 IndexError。
                - 若传入 Sequence[int]：按指定顺序返回选定请求的 2D 对齐张量 [len(request_idx), max_output_len]。
                - 若为 None（默认）：返回整个批次的 2D 对齐张量 [B, max_output_len]。
            pad_token_id: 填充 token ID，默认为 0。仅在 2D 对齐时生效。
        """
        if isinstance(request_idx, int):
            for seq in self.sequences:
                if seq.request_index == request_idx:
                    return torch.tensor(seq.token_ids, dtype=torch.long)
            raise IndexError(f"Request index {request_idx} not found in sequences")

        if request_idx is not None:
            seq_map = {seq.request_index: seq for seq in self.sequences}
            selected_seqs: list[SequenceOutput] = []
            for r_idx in request_idx:
                if r_idx not in seq_map:
                    raise IndexError(f"Request index {r_idx} not found in sequences")
                selected_seqs.append(seq_map[r_idx])
        else:
            selected_seqs = self.sequences

        if not selected_seqs:
            return torch.empty((0, 0), dtype=torch.long)

        max_len = max((seq.output_len for seq in selected_seqs), default=0)
        tensor = torch.full((len(selected_seqs), max_len), pad_token_id, dtype=torch.long)
        for idx, seq in enumerate(selected_seqs):
            if seq.token_ids:
                tensor[idx, : seq.output_len] = torch.tensor(seq.token_ids, dtype=torch.long)
        return tensor


def build_memory_profile(
    baseline_allocated_bytes: int,
    memory_after_prefill_bytes: int,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    cache_bytes: int,
) -> MemoryProfile:
    """将监控字节数转换为以 MB 为单位的 MemoryProfile。"""
    mib = 1024**2
    return MemoryProfile(
        baseline_mb=baseline_allocated_bytes / mib,
        prefill_mb=memory_after_prefill_bytes / mib,
        peak_allocated_mb=peak_allocated_bytes / mib,
        peak_reserved_mb=peak_reserved_bytes / mib,
        kv_cache_mb=cache_bytes / mib,
    )


def compute_step_aggregated_metrics(
    step_latencies_ms: list[float],
    e2e_wall_ms: float,
    e2e_model_ms: float | None = None,
    *,
    total_tokens: int,
    batch_size: int,
    is_cuda: bool = False,
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
                )
            )
            req_counter += 1

    total_requests = len(sequences)
    step_latencies = [x for run in runs for x in run.trace.step_latencies_ms]
    active_tokens = [x for run in runs for x in run.trace.active_tokens_per_step]

    # 1. 指标聚合
    e2e_wall_ms = options.get_agg_fn("metrics", "e2e_wall_ms")([r.metrics.e2e_wall_ms for r in runs])
    e2e_model_ms = options.get_agg_fn("metrics", "e2e_model_ms")([r.metrics.e2e_model_ms for r in runs])
    ttft_ms = options.get_agg_fn("metrics", "ttft_ms")([r.metrics.ttft_ms for r in runs])
    mean_tpot_ms = options.get_agg_fn("metrics", "mean_tpot_ms")([r.metrics.mean_tpot_ms for r in runs])
    p50_tpot_ms = options.get_agg_fn("metrics", "p50_tpot_ms")([r.metrics.p50_tpot_ms for r in runs])
    p95_tpot_ms = options.get_agg_fn("metrics", "p95_tpot_ms")([r.metrics.p95_tpot_ms for r in runs])
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

    memory = MemoryProfile(
        baseline_mb=baseline_mb_fn([r.memory.baseline_mb for r in runs]),
        prefill_mb=prefill_mb_fn([r.memory.prefill_mb for r in runs]),
        peak_allocated_mb=peak_allocated_mb_fn([r.memory.peak_allocated_mb for r in runs]),
        peak_reserved_mb=peak_reserved_mb_fn([r.memory.peak_reserved_mb for r in runs]),
        kv_cache_mb=kv_cache_mb_fn([r.memory.kv_cache_mb for r in runs]),
    )

    op_breakdowns = [r.op_breakdown for r in runs if r.op_breakdown is not None]
    op_breakdown = merge_op_latency_breakdowns(op_breakdowns)

    trace = GenerationTrace(
        step_latencies_ms=step_latencies,
        active_tokens_per_step=active_tokens,
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


def to_generation_result(result: ScheduleResult, baseline_allocated_mb: float, is_cuda: bool) -> GenerationResult:
    """把 ScheduleResult 映射为统一的 GenerationResult（所有请求同时到达的场景），以复用现有 parity/正确性门禁。"""
    step_latencies = [event.total_ms for event in result.events]
    metrics = compute_step_aggregated_metrics(
        step_latencies_ms=step_latencies,
        e2e_wall_ms=result.metrics.makespan_ms,
        e2e_model_ms=result.metrics.model_ms,
        total_tokens=sum(seq.output_len for seq in result.sequences),
        batch_size=result.metrics.request_count,
        is_cuda=is_cuda,
    )
    memory = MemoryProfile(
        baseline_mb=baseline_allocated_mb,
        prefill_mb=0.0,
        peak_allocated_mb=result.metrics.peak_allocated_mb,
        peak_reserved_mb=result.metrics.peak_reserved_mb,
        kv_cache_mb=result.metrics.peak_live_cache_mb,
    )
    trace = GenerationTrace(
        step_latencies_ms=step_latencies,
        active_tokens_per_step=[event.generated_tokens for event in result.events],
    )
    return GenerationResult(
        sequences=result.sequences,
        metrics=metrics,
        memory=memory,
        trace=trace,
        batch_size=result.metrics.request_count,
    )
