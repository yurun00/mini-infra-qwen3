"""Continuous Batching 与固定批次 Scheduler 评测结果领域类型。"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from mini_infra.bench.lockstep.types import DistributionStats, summarize_samples
from mini_infra.core.types import BatchScheduleInfo, MemoryProfile, StepAggregatedMetrics
from mini_infra.engine.types import ScheduleEvent, ScheduleMetrics, ScheduleResult


@dataclass(frozen=True)
class SchedulerCaseResult:
    """单个 Scheduler 用例在多轮 repeat 测量下的汇总结果。

    直接组合复用 StepAggregatedMetrics[DistributionStats]、ScheduleMetrics[DistributionStats]、
    MemoryProfile、BatchScheduleInfo 与 ScheduleEvent，与 ScheduleResult 保持顶层结构镜像对齐。
    """

    case_id: str
    mode: str
    step_agg_metrics: StepAggregatedMetrics[DistributionStats]
    schedule_metrics: ScheduleMetrics[DistributionStats]
    execution_slot_utilization: DistributionStats
    running_capacity_utilization: DistributionStats
    memory: MemoryProfile
    batch_info: BatchScheduleInfo | None = None
    repeat_samples: list[dict[str, Any]] = field(default_factory=list)
    sample_events: list[ScheduleEvent] = field(default_factory=list)

    def to_record_dict(self) -> dict[str, Any]:
        """导出为标准评测记录的字典结构。"""
        sm = self.schedule_metrics
        rec: dict[str, Any] = {
            "case_id": self.case_id,
            "mode": self.mode,
            "makespan_ms": sm.makespan_ms.to_dict(),
            "busy_ms": sm.busy_ms.to_dict(),
            "model_ms": self.step_agg_metrics.e2e_model_ms.to_dict(),
            "cache_management_ms": sm.cache_management_ms.to_dict(),
            "cache_management_fraction": sm.cache_management_fraction.to_dict(),
            "request_throughput_per_second": self.step_agg_metrics.throughput_req_s.to_dict(),
            "throughput_tok_s": self.step_agg_metrics.throughput_tok_s.to_dict(),
            "execution_slot_utilization": self.execution_slot_utilization.to_dict(),
            "running_capacity_utilization": self.running_capacity_utilization.to_dict(),
            "queue_ms_p50": sm.queue_ms_p50.to_dict(),
            "queue_ms_p95": sm.queue_ms_p95.to_dict(),
            "ttft_ms_p50": sm.ttft_ms_p50.to_dict(),
            "ttft_ms_p95": sm.ttft_ms_p95.to_dict(),
            "end_to_end_ms_p50": sm.end_to_end_ms_p50.to_dict(),
            "end_to_end_ms_p95": sm.end_to_end_ms_p95.to_dict(),
            "itl_ms_p50": sm.itl_ms_p50.to_dict(),
            "itl_ms_p95": sm.itl_ms_p95.to_dict(),
            "itl_ms_max": sm.itl_ms_max.to_dict(),
            "peak_allocated_mb_max": self.memory.peak_allocated_mb,
            "peak_live_cache_mb_max": self.memory.kv_cache_mb,
            "repeat_samples": self.repeat_samples,
            "sample_event_trace": [asdict(e) for e in self.sample_events],
        }
        return rec


def summarize_scheduler_runs(
    case_id: str,
    mode: str,
    runs: Sequence[ScheduleResult],
) -> SchedulerCaseResult:
    """把同一个用例的多轮运行聚合为一个包含 DistributionStats 的 SchedulerCaseResult。"""
    if not runs:
        raise ValueError("runs 不能为空")

    repeat_samples = [
        {
            "makespan_ms": r.schedule_metrics.makespan_ms,
            "request_throughput_per_second": r.step_agg_metrics.throughput_req_s,
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

    step_agg_dist: StepAggregatedMetrics[DistributionStats] = StepAggregatedMetrics(
        e2e_wall_ms=summarize_samples([r.step_agg_metrics.e2e_wall_ms for r in runs]),
        e2e_model_ms=summarize_samples([r.step_agg_metrics.e2e_model_ms for r in runs]),
        throughput_tok_s=summarize_samples([r.step_agg_metrics.throughput_tok_s for r in runs]),
        throughput_req_s=summarize_samples([r.step_agg_metrics.throughput_req_s for r in runs]),
        ttft_ms=None,
        mean_tpot_ms=None,
        p50_tpot_ms=None,
        p95_tpot_ms=None,
    )

    sched_dist: ScheduleMetrics[DistributionStats] = ScheduleMetrics(
        request_count=runs[0].schedule_metrics.request_count,
        max_running_requests=runs[0].schedule_metrics.max_running_requests,
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

    return SchedulerCaseResult(
        case_id=case_id,
        mode=mode,
        step_agg_metrics=step_agg_dist,
        schedule_metrics=sched_dist,
        execution_slot_utilization=summarize_samples(
            [r.batch_info.execution_slot_utilization if r.batch_info is not None else 0.0 for r in runs]
        ),
        running_capacity_utilization=summarize_samples(
            [(r.batch_info.running_capacity_utilization or 0.0) if r.batch_info is not None else 0.0 for r in runs]
        ),
        memory=case_memory,
        batch_info=runs[0].batch_info,
        repeat_samples=repeat_samples,
        sample_events=list(runs[0].events),
    )


@dataclass(frozen=True)
class SchedulerSuiteResult:
    """整套 Scheduler 对照实验套件的运行结果（包含该套件所有 Case 的评测结果及跨 Case 加速比）。"""

    suite_name: str
    cases: dict[str, SchedulerCaseResult]
    continuous_makespan_speedup: float

    def to_record_dict(self) -> dict[str, Any]:
        """导出为标准评测记录的字典结构。"""
        return {
            "suite": self.suite_name,
            "cases": {cid: res.to_record_dict() for cid, res in self.cases.items()},
            "continuous_makespan_speedup": self.continuous_makespan_speedup,
        }


__all__ = [
    "SchedulerCaseResult",
    "SchedulerSuiteResult",
    "summarize_scheduler_runs",
]
