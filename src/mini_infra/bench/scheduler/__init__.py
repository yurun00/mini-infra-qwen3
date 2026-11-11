"""队列调度器（Scheduler: Fixed vs Continuous Batching）基准测试模块。"""

from mini_infra.bench.cases import (
    CaseDimension,
    CaseDimensionValue,
    CaseKey,
    Comparison,
    ComparisonMetric,
    MetricKind,
    WorkloadProfile,
)
from mini_infra.bench.scheduler.format import (
    format_table,
)
from mini_infra.bench.scheduler.plan import (
    PlannedRun,
    RunPhase,
    build_scheduler_run_plan,
    format_scheduler_run_plan,
)
from mini_infra.bench.scheduler.run_case import (
    run_case,
    run_case_once,
)
from mini_infra.bench.scheduler.run_suite import (
    run_suite,
)
from mini_infra.bench.scheduler.types import (
    ComparisonResult,
    DistributionStats,
    SchedulerCaseResult,
    compute_comparisons,
    extract_metric_by_path,
    extract_workload_profile,
)

__all__ = [
    "CaseDimension",
    "CaseDimensionValue",
    "CaseKey",
    "Comparison",
    "ComparisonMetric",
    "ComparisonResult",
    "DistributionStats",
    "MetricKind",
    "PlannedRun",
    "RunPhase",
    "SchedulerCaseResult",
    "WorkloadProfile",
    "build_scheduler_run_plan",
    "compute_comparisons",
    "extract_metric_by_path",
    "extract_workload_profile",
    "format_scheduler_run_plan",
    "format_table",
    "run_case",
    "run_case_once",
    "run_suite",
]
