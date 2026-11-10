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
    build_benchmark_record,
    format_case_summary,
    format_comparison,
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
    parse_interleave_args,
    parse_interleave_dimension,
    run_suite,
)
from mini_infra.bench.scheduler.suites import (
    get_all_scheduler_suites,
    get_scheduler_suite,
    list_scheduler_suite_names,
    load_scheduler_suite,
)
from mini_infra.bench.scheduler.types import (
    ComparisonResult,
    DistributionStats,
    SchedulerCaseResult,
    compute_comparisons,
    extract_metric_by_path,
    extract_workload_profile,
    summarize_same_case_runs,
    summarize_samples,
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
    "build_benchmark_record",
    "build_scheduler_run_plan",
    "compute_comparisons",
    "extract_metric_by_path",
    "extract_workload_profile",
    "format_case_summary",
    "format_comparison",
    "format_scheduler_run_plan",
    "format_table",
    "get_all_scheduler_suites",
    "get_scheduler_suite",
    "list_scheduler_suite_names",
    "load_scheduler_suite",
    "parse_interleave_args",
    "parse_interleave_dimension",
    "run_case",
    "run_case_once",
    "run_suite",
    "summarize_same_case_runs",
    "summarize_samples",
]
