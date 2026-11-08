"""Continuous Batching 与固定批次 Scheduler 评测结果领域类型。"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from mini_infra.bench.common import (
    DistributionStats,
    EfficiencyMetrics,
    WorkloadSpec,
    summarize_samples,
)
from mini_infra.bench.scheduler.cases import (
    CaseDimensionValue,
    CaseKey,
    ComparisonSpec,
    MetricKind,
)
from mini_infra.engine.metrics import (
    BatchScheduleInfo,
    MemoryProfile,
    ScheduleEvent,
    ScheduleMetrics,
    StepAggregatedMetrics,
    summarize_repeated_kv_cache_profiles,
)
from mini_infra.engine.mode import Mode
from mini_infra.engine.results import ScheduleResult


@dataclass(frozen=True)
class MetricComparison:
    """通用基准指标相对对比结果项。"""

    baseline_key: CaseKey
    candidate_key: CaseKey
    metric_path: str
    metric_name: str
    kind: MetricKind
    baseline_value: float
    candidate_value: float
    ratio: float
    relative_delta: float

    @property
    def speedup(self) -> float:
        """baseline / candidate（针对越小越好的指标计算加速比，如 makespan）。"""
        return self.baseline_value / self.candidate_value if self.candidate_value > 0 else 1.0

    @property
    def gain(self) -> float:
        """(candidate - baseline) / baseline（针对越大越好的指标计算相对增益，如吞吐量）。"""
        return (self.candidate_value - self.baseline_value) / self.baseline_value if self.baseline_value > 0 else 0.0

    @property
    def reduction(self) -> float:
        """(baseline - candidate) / baseline（针对越小越好的指标计算相对削减比，如显存/Token）。"""
        return (self.baseline_value - self.candidate_value) / self.baseline_value if self.baseline_value > 0 else 0.0

    def formatted_value(self) -> str:
        if self.kind == MetricKind.SPEEDUP:
            return f"speedup {self.speedup:.2f}x"
        if self.kind == MetricKind.GAIN:
            sign = "+" if self.gain >= 0 else ""
            return f"{self.metric_name} {sign}{self.gain * 100:.1f}%"
        if self.kind == MetricKind.REDUCTION:
            sign = "-" if self.reduction >= 0 else "+"
            return f"{self.metric_name} {sign}{abs(self.reduction) * 100:.1f}%"
        return f"{self.ratio:.4f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_key": self.baseline_key.to_dict(),
            "candidate_key": self.candidate_key.to_dict(),
            "metric_path": self.metric_path,
            "metric_name": self.metric_name,
            "kind": self.kind.value,
            "baseline_value": round(self.baseline_value, 4),
            "candidate_value": round(self.candidate_value, 4),
            "ratio": round(self.ratio, 4),
            "relative_delta": round(self.relative_delta, 4),
            "speedup": round(self.speedup, 4),
            "gain": round(self.gain, 4),
            "reduction": round(self.reduction, 4),
            "formatted": self.formatted_value(),
        }


def resolve_metric_value(obj: Any, path: str) -> float:
    """沿点分路径从对象中解析出目标标量浮点值。"""
    curr = obj
    for segment in path.split("."):
        if not hasattr(curr, segment):
            raise AttributeError(f"{type(curr).__name__} 不包含属性 '{segment}'（完整路径: '{path}'）")
        curr = getattr(curr, segment)
    if hasattr(curr, "mean"):
        val = curr.mean
    elif isinstance(curr, (int, float)):
        val = curr
    else:
        raise TypeError(f"指标 '{path}' 解析得到不可转换为数值的对象: {curr!r} ({type(curr)})")
    return float(val)


def compute_comparisons(
    case_results: Sequence["SchedulerCaseResult"],
    spec: ComparisonSpec,
) -> list[MetricComparison]:
    """根据 ComparisonSpec 将 case_results 分组，在每组中按 baseline 规则选出基线并计算指标对比。"""
    groups: dict[tuple[CaseDimensionValue, ...], list[SchedulerCaseResult]] = {}
    for r in case_results:
        g_key = r.key.project(spec.group_by)
        groups.setdefault(g_key, []).append(r)

    comparisons: list[MetricComparison] = []
    for g_key, group in groups.items():
        matching_baselines = [
            r for r in group if all(r.key.value(dim) == expected for dim, expected in spec.baseline.items())
        ]
        if len(matching_baselines) == 0:
            raise ValueError(
                f"分组 {g_key} 内未找到符合 baseline 约束 {spec.baseline} 的基线用例。"
                f"组内用例 keys: {[r.key.display_name for r in group]}"
            )
        if len(matching_baselines) > 1:
            raise ValueError(
                f"分组 {g_key} 内找到多个符合 baseline 约束 {spec.baseline} 的候选用例: "
                f"{[r.key.display_name for r in matching_baselines]}"
            )
        baseline_res = matching_baselines[0]

        for candidate_res in group:
            if candidate_res.key == baseline_res.key:
                continue
            for metric in spec.metrics:
                b_val = resolve_metric_value(baseline_res, metric.path)
                c_val = resolve_metric_value(candidate_res, metric.path)
                ratio = c_val / b_val if b_val != 0 else 1.0
                rel_delta = (c_val - b_val) / b_val if b_val != 0 else 0.0

                comparisons.append(
                    MetricComparison(
                        baseline_key=baseline_res.key,
                        candidate_key=candidate_res.key,
                        metric_path=metric.path,
                        metric_name=metric.display_name,
                        kind=metric.kind,
                        baseline_value=b_val,
                        candidate_value=c_val,
                        ratio=ratio,
                        relative_delta=rel_delta,
                    )
                )
    return comparisons


@dataclass(frozen=True)
class SchedulerCaseResult:
    """单个 Scheduler 用例在多轮 repeat 测量下的汇总结果。

    以 CaseKey 作为全局唯一身份键，不再保留非正交的 case_id。
    """

    key: CaseKey
    workload: WorkloadSpec
    metrics: StepAggregatedMetrics[DistributionStats]
    schedule_metrics: ScheduleMetrics[DistributionStats]
    memory: MemoryProfile
    packing: BatchScheduleInfo[DistributionStats] | None = None
    efficiency: EfficiencyMetrics | None = None
    repeat_samples: list[dict[str, Any]] = field(default_factory=list)
    sample_events: list[ScheduleEvent] = field(default_factory=list)

    @property
    def mode(self) -> Mode:
        return self.key.mode

    @property
    def workload_id(self) -> str:
        return self.key.workload_id

    @property
    def step_agg_metrics(self) -> StepAggregatedMetrics[DistributionStats]:
        """与 ScheduleResult.step_agg_metrics 保持一致的别名。"""
        return self.metrics

    @property
    def batch_info(self) -> BatchScheduleInfo[DistributionStats] | None:
        """与 ScheduleResult.batch_info 保持一致的别名。"""
        return self.packing

    def to_record_dict(self) -> dict[str, Any]:
        """导出为与 BaseCaseRecord / SchedulerCaseRecord 完全对齐的标准字典结构。"""
        sm = self.schedule_metrics
        rec: dict[str, Any] = {
            "workload_id": self.key.workload_id,
            "mode": str(self.key.mode),
            "num_total_requests": self.workload.num_total_requests,
            "prompt_len": (
                list(self.workload.prompt_len)
                if isinstance(self.workload.prompt_len, tuple)
                else self.workload.prompt_len
            ),
            "max_new_tokens": (
                list(self.workload.max_new_tokens)
                if isinstance(self.workload.max_new_tokens, tuple)
                else self.workload.max_new_tokens
            ),
            "repeats": self.workload.repeats,
            "warmup": self.workload.warmup,
            "max_running_requests": self.key.max_running_requests,
            "arrival_times_ms": (
                list(self.workload.arrival_times_ms)
                if isinstance(self.workload.arrival_times_ms, tuple)
                else (self.workload.arrival_times_ms if self.workload.arrival_times_ms is not None else 0.0)
            ),
            "e2e_wall_ms": self.metrics.e2e_wall_ms.to_dict(),
            "e2e_model_ms": self.metrics.e2e_model_ms.to_dict(),
            "throughput_tok_s": self.metrics.throughput_tok_s.to_dict(),
            "throughput_req_s": self.metrics.throughput_req_s.to_dict(),
            "gpu_time_ratio": (
                self.metrics.gpu_time_ratio.to_dict() if self.metrics.gpu_time_ratio is not None else None
            ),
            "effective_prompt_tok_s": (
                self.metrics.effective_prompt_tok_s.to_dict()
                if self.metrics.effective_prompt_tok_s is not None
                else None
            ),
            "padded_prompt_tok_s": (
                self.metrics.padded_prompt_tok_s.to_dict() if self.metrics.padded_prompt_tok_s is not None else None
            ),
            "memory": self.memory.to_dict(),
            "schedule": {
                "request_count": sm.request_count,
                "max_running_requests": sm.max_running_requests,
                "makespan_ms": sm.makespan_ms.to_dict(),
                "busy_ms": sm.busy_ms.to_dict(),
                "cache_management_ms": sm.cache_management_ms.to_dict(),
                "cache_management_fraction": sm.cache_management_fraction.to_dict(),
                "queue_ms_p50": sm.queue_ms_p50.to_dict(),
                "queue_ms_p95": sm.queue_ms_p95.to_dict(),
                "ttft_ms_p50": sm.ttft_ms_p50.to_dict(),
                "ttft_ms_p95": sm.ttft_ms_p95.to_dict(),
                "end_to_end_ms_p50": sm.end_to_end_ms_p50.to_dict(),
                "end_to_end_ms_p95": sm.end_to_end_ms_p95.to_dict(),
                "itl_ms_p50": sm.itl_ms_p50.to_dict(),
                "itl_ms_p95": sm.itl_ms_p95.to_dict(),
                "itl_ms_max": sm.itl_ms_max.to_dict(),
            },
            "repeat_samples": self.repeat_samples,
            "sample_event_trace": [{**asdict(e), "phase": str(e.phase)} for e in self.sample_events],
        }
        if self.key.paged_cache_config is not None:
            rec["paged_cache_config"] = {
                "block_size": self.key.paged_cache_config.block_size,
                "num_blocks": self.key.paged_cache_config.num_blocks,
            }
        if self.workload.scale is not None:
            rec["scale"] = self.workload.scale
        if self.workload.seed is not None:
            rec["seed"] = self.workload.seed
        if self.packing is not None:
            rec["packing"] = self.packing.to_dict()
        if self.efficiency is not None:
            rec["efficiency"] = self.efficiency.to_dict()
        return rec


def summarize_same_case_runs(
    case_key: CaseKey,
    runs: Sequence[ScheduleResult],
) -> SchedulerCaseResult:
    """把具有相同 CaseKey 与 WorkloadSpec 的多轮运行聚合为一个 SchedulerCaseResult。"""
    if not runs:
        raise ValueError("runs 不能为空")

    first_run = runs[0]
    if any(r.workload != first_run.workload for r in runs):
        raise ValueError("summarize_same_case_runs 中的所有 run 必须具有相同的 WorkloadSpec")
    workload = replace(first_run.workload, repeats=len(runs))

    repeat_samples = [
        {
            "makespan_ms": r.schedule_metrics.makespan_ms,
            "throughput_req_s": r.step_agg_metrics.throughput_req_s,
            "throughput_tok_s": r.step_agg_metrics.throughput_tok_s,
            "execution_slot_utilization": r.batch_info.execution_slot_utilization if r.batch_info is not None else 0.0,
            "ttft_ms_p95": r.schedule_metrics.ttft_ms_p95,
            "itl_ms_p95": r.schedule_metrics.itl_ms_p95,
            "cache_management_fraction": r.schedule_metrics.cache_management_fraction,
            "peak_allocated_mb": r.memory.peak_allocated_mb,
        }
        for r in runs
    ]

    worst_mem = max(runs, key=lambda r: r.memory.peak_allocated_mb).memory
    case_memory = replace(
        worst_mem,
        peak_allocated_mb=max(r.memory.peak_allocated_mb for r in runs),
        peak_reserved_mb=max(r.memory.peak_reserved_mb for r in runs),
        kv_cache=summarize_repeated_kv_cache_profiles([r.memory.kv_cache for r in runs]),
    )

    gpu_ratios = [r.step_agg_metrics.gpu_time_ratio for r in runs]
    gpu_ratio_stat = (
        summarize_samples([v for v in gpu_ratios if v is not None]) if all(v is not None for v in gpu_ratios) else None
    )
    eff_prompt_vals = [
        r.step_agg_metrics.effective_prompt_tok_s for r in runs if r.step_agg_metrics.effective_prompt_tok_s is not None
    ]
    pad_prompt_vals = [
        r.step_agg_metrics.padded_prompt_tok_s for r in runs if r.step_agg_metrics.padded_prompt_tok_s is not None
    ]

    step_agg_dist: StepAggregatedMetrics[DistributionStats] = StepAggregatedMetrics(
        e2e_wall_ms=summarize_samples([r.step_agg_metrics.e2e_wall_ms for r in runs]),
        e2e_model_ms=summarize_samples([r.step_agg_metrics.e2e_model_ms for r in runs]),
        throughput_tok_s=summarize_samples([r.step_agg_metrics.throughput_tok_s for r in runs]),
        throughput_req_s=summarize_samples([r.step_agg_metrics.throughput_req_s for r in runs]),
        gpu_time_ratio=gpu_ratio_stat,
        effective_prompt_tok_s=summarize_samples(eff_prompt_vals) if eff_prompt_vals else None,
        padded_prompt_tok_s=summarize_samples(pad_prompt_vals) if pad_prompt_vals else None,
        ttft_ms=None,
        mean_tpot_ms=None,
        p50_tpot_ms=None,
        p95_tpot_ms=None,
    )

    sched_dist: ScheduleMetrics[DistributionStats] = ScheduleMetrics(
        request_count=first_run.schedule_metrics.request_count,
        max_running_requests=first_run.schedule_metrics.max_running_requests,
        makespan_ms=summarize_samples([r.schedule_metrics.makespan_ms for r in runs]),
        busy_ms=summarize_samples([r.schedule_metrics.busy_ms for r in runs]),
        cache_management_ms=summarize_samples([r.schedule_metrics.cache_management_ms for r in runs]),
        cache_management_fraction=summarize_samples([r.schedule_metrics.cache_management_fraction for r in runs]),
        queue_ms_p50=summarize_samples([r.schedule_metrics.queue_ms_p50 for r in runs]),
        queue_ms_p95=summarize_samples([r.schedule_metrics.queue_ms_p95 for r in runs]),
        ttft_ms_p50=summarize_samples([r.schedule_metrics.ttft_ms_p50 for r in runs]),
        ttft_ms_p95=summarize_samples([r.schedule_metrics.ttft_ms_p95 for r in runs]),
        end_to_end_ms_p50=summarize_samples([r.schedule_metrics.end_to_end_ms_p50 for r in runs]),
        end_to_end_ms_p95=summarize_samples([r.schedule_metrics.end_to_end_ms_p95 for r in runs]),
        itl_ms_p50=summarize_samples([r.schedule_metrics.itl_ms_p50 for r in runs]),
        itl_ms_p95=summarize_samples([r.schedule_metrics.itl_ms_p95 for r in runs]),
        itl_ms_max=summarize_samples([r.schedule_metrics.itl_ms_max for r in runs]),
    )

    batch_infos = [r.batch_info for r in runs if r.batch_info is not None]
    packing: BatchScheduleInfo[DistributionStats] | None = None
    if batch_infos:
        cap_vals = [b.running_capacity_utilization for b in batch_infos if b.running_capacity_utilization is not None]
        packing = BatchScheduleInfo(
            padding_efficiency=summarize_samples([b.padding_efficiency for b in batch_infos]),
            slot_utilization=summarize_samples([b.slot_utilization for b in batch_infos]),
            running_capacity_utilization=summarize_samples(cap_vals) if cap_vals else None,
        )

    return SchedulerCaseResult(
        key=case_key,
        workload=workload,
        metrics=step_agg_dist,
        schedule_metrics=sched_dist,
        memory=case_memory,
        packing=packing,
        repeat_samples=repeat_samples,
        sample_events=list(first_run.events),
    )


__all__ = [
    "DistributionStats",
    "MetricComparison",
    "SchedulerCaseResult",
    "WorkloadSpec",
    "compute_comparisons",
    "resolve_metric_value",
    "summarize_same_case_runs",
    "summarize_samples",
]
