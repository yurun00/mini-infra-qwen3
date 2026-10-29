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
from mini_infra.core.types import BatchScheduleInfo, MemoryProfile, StepAggregatedMetrics
from mini_infra.engine.mode import Mode
from mini_infra.engine.types import ScheduleEvent, ScheduleMetrics, ScheduleResult


@dataclass(frozen=True)
class SchedulerCaseResult:
    """单个 Scheduler 用例在多轮 repeat 测量下的汇总结果。

    直接组合复用 StepAggregatedMetrics[DistributionStats]、ScheduleMetrics[DistributionStats]、
    MemoryProfile、BatchScheduleInfo[DistributionStats] 与 ScheduleEvent，与 ScheduleResult 保持顶层结构镜像对齐。
    与 LockstepCaseResult 共享顶层结构（workload, metrics, memory, packing, efficiency），
    并扩展调度器专属统计（schedule_metrics）。
    """

    case_id: str
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
        return self.workload.mode

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
            "case_id": self.case_id,
            "mode": str(self.workload.mode),
            "batch_size": self.workload.batch_size,
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
            "max_running_requests": self.workload.max_running_requests or sm.max_running_requests,
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
        if self.packing is not None:
            rec["packing"] = self.packing.to_dict()
        if self.efficiency is not None:
            rec["efficiency"] = self.efficiency.to_dict()
        return rec


def summarize_same_workload_runs(
    case_id: str,
    runs: Sequence[ScheduleResult],
) -> SchedulerCaseResult:
    """把具有相同 WorkloadSpec 的多轮运行聚合为一个包含 DistributionStats 的 SchedulerCaseResult。"""
    if not runs:
        raise ValueError("runs 不能为空")

    first_run = runs[0]
    if any(r.workload != first_run.workload for r in runs):
        raise ValueError("summarize_same_workload_runs 中的所有 run 必须具有相同的 WorkloadSpec")
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
        }
        for r in runs
    ]

    worst_mem = max(runs, key=lambda r: r.memory.peak_allocated_mb).memory
    case_memory = replace(
        worst_mem,
        peak_allocated_mb=max(r.memory.peak_allocated_mb for r in runs),
        peak_reserved_mb=max(r.memory.peak_reserved_mb for r in runs),
        kv_cache_mb=max(r.memory.kv_cache_mb for r in runs),
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
        # 对以下指标来说step层级的聚合没有意义。
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
        case_id=case_id,
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
    "SchedulerCaseResult",
    "WorkloadSpec",
    "summarize_same_workload_runs",
    "summarize_samples",
]
