"""Continuous Batching 与固定批次 Scheduler 评测结果领域类型。"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mini_infra.bench.lockstep.types import DistributionStats, summarize_samples
from mini_infra.engine.types import ScheduleResult


@dataclass(frozen=True)
class SchedulerCaseResult:
    """单个 Scheduler 用例在多轮 repeat 测量下的汇总结果。"""

    case_id: str
    mode: str
    makespan_ms: DistributionStats
    busy_ms: DistributionStats
    model_ms: DistributionStats
    cache_management_ms: DistributionStats
    cache_management_fraction: DistributionStats
    throughput_req_s: DistributionStats
    throughput_tok_s: DistributionStats
    execution_slot_utilization: DistributionStats
    running_capacity_utilization: DistributionStats
    queue_ms_p50: DistributionStats
    queue_ms_p95: DistributionStats
    ttft_ms_p50: DistributionStats
    ttft_ms_p95: DistributionStats
    end_to_end_ms_p50: DistributionStats
    end_to_end_ms_p95: DistributionStats
    itl_ms_p50: DistributionStats
    itl_ms_p95: DistributionStats
    itl_ms_max: DistributionStats
    peak_allocated_mb_max: float
    peak_live_cache_mb_max: float
    repeat_samples: list[dict[str, Any]] = field(default_factory=list)
    sample_event_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_record_dict(self) -> dict[str, Any]:
        """导出为标准评测记录的字典结构。"""
        return {
            "case_id": self.case_id,
            "mode": self.mode,
            "makespan_ms": self.makespan_ms.to_dict(),
            "busy_ms": self.busy_ms.to_dict(),
            "model_ms": self.model_ms.to_dict(),
            "cache_management_ms": self.cache_management_ms.to_dict(),
            "cache_management_fraction": self.cache_management_fraction.to_dict(),
            "request_throughput_per_second": self.throughput_req_s.to_dict(),
            "output_token_throughput_per_second": self.throughput_tok_s.to_dict(),
            "execution_slot_utilization": self.execution_slot_utilization.to_dict(),
            "running_capacity_utilization": self.running_capacity_utilization.to_dict(),
            "queue_ms_p50": self.queue_ms_p50.to_dict(),
            "queue_ms_p95": self.queue_ms_p95.to_dict(),
            "ttft_ms_p50": self.ttft_ms_p50.to_dict(),
            "ttft_ms_p95": self.ttft_ms_p95.to_dict(),
            "end_to_end_ms_p50": self.end_to_end_ms_p50.to_dict(),
            "end_to_end_ms_p95": self.end_to_end_ms_p95.to_dict(),
            "itl_ms_p50": self.itl_ms_p50.to_dict(),
            "itl_ms_p95": self.itl_ms_p95.to_dict(),
            "itl_ms_max": self.itl_ms_max.to_dict(),
            "peak_allocated_mb_max": self.peak_allocated_mb_max,
            "peak_live_cache_mb_max": self.peak_live_cache_mb_max,
            "repeat_samples": self.repeat_samples,
            "sample_event_trace": self.sample_event_trace,
        }


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
            "makespan_ms": r.step_agg_metrics.e2e_wall_ms,
            "request_throughput_per_second": r.step_agg_metrics.throughput_req_s,
            "output_token_throughput_per_second": r.step_agg_metrics.throughput_tok_s,
            "execution_slot_utilization": r.batch_info.execution_slot_utilization,
            "ttft_ms_p95": r.schedule_metrics.ttft_ms_p95,
            "itl_ms_p95": r.schedule_metrics.itl_ms_p95,
            "cache_management_fraction": r.schedule_metrics.cache_management_fraction,
        }
        for r in runs
    ]

    sample_event_trace = [asdict(e) for e in runs[0].events]

    return SchedulerCaseResult(
        case_id=case_id,
        mode=mode,
        makespan_ms=summarize_samples([r.step_agg_metrics.e2e_wall_ms for r in runs]),
        busy_ms=summarize_samples([r.schedule_metrics.busy_ms for r in runs]),
        model_ms=summarize_samples([r.step_agg_metrics.e2e_model_ms for r in runs]),
        cache_management_ms=summarize_samples([r.schedule_metrics.cache_management_ms for r in runs]),
        cache_management_fraction=summarize_samples([r.schedule_metrics.cache_management_fraction for r in runs]),
        throughput_req_s=summarize_samples([r.step_agg_metrics.throughput_req_s for r in runs]),
        throughput_tok_s=summarize_samples([r.step_agg_metrics.throughput_tok_s for r in runs]),
        execution_slot_utilization=summarize_samples([r.batch_info.execution_slot_utilization for r in runs]),
        running_capacity_utilization=summarize_samples(
            [r.batch_info.running_capacity_utilization or 0.0 for r in runs]
        ),
        queue_ms_p50=summarize_samples([r.schedule_metrics.queue_ms_p50 for r in runs]),
        queue_ms_p95=summarize_samples([r.schedule_metrics.queue_ms_p95 for r in runs]),
        ttft_ms_p50=summarize_samples([r.schedule_metrics.ttft_ms_p50 for r in runs]),
        ttft_ms_p95=summarize_samples([r.schedule_metrics.ttft_ms_p95 for r in runs]),
        end_to_end_ms_p50=summarize_samples([r.schedule_metrics.end_to_end_ms_p50 for r in runs]),
        end_to_end_ms_p95=summarize_samples([r.schedule_metrics.end_to_end_ms_p95 for r in runs]),
        itl_ms_p50=summarize_samples([r.schedule_metrics.itl_ms_p50 for r in runs]),
        itl_ms_p95=summarize_samples([r.schedule_metrics.itl_ms_p95 for r in runs]),
        itl_ms_max=summarize_samples([r.schedule_metrics.itl_ms_max for r in runs]),
        peak_allocated_mb_max=max(r.memory.peak_allocated_mb for r in runs),
        peak_live_cache_mb_max=max(r.memory.kv_cache_mb for r in runs),
        repeat_samples=repeat_samples,
        sample_event_trace=sample_event_trace,
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
