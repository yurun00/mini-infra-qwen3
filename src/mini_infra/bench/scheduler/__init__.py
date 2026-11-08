"""队列调度器（Scheduler: Fixed vs Continuous Batching）基准测试模块。"""

from mini_infra.bench.cases import (
    CaseDimension,
    CaseDimensionValue,
    CaseKey,
    ComparisonMetric,
    ComparisonSpec,
    MetricKind,
    SchedulerSuiteCase,
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
    SchedulerSuite,
    get_all_scheduler_suites,
    get_scheduler_suite,
    list_scheduler_suite_names,
    load_scheduler_suite,
)
from mini_infra.bench.scheduler.types import (
    DistributionStats,
    MetricComparison,
    SchedulerCaseResult,
    WorkloadSpec,
    compute_comparisons,
    resolve_metric_value,
    summarize_same_case_runs,
    summarize_samples,
)

__all__ = [
    "CaseDimension",
    "CaseDimensionValue",
    "CaseKey",
    "ComparisonMetric",
    "ComparisonSpec",
    "DistributionStats",
    "MetricComparison",
    "MetricKind",
    "PlannedRun",
    "RunPhase",
    "SchedulerCaseResult",
    "SchedulerSuite",
    "SchedulerSuiteCase",
    "WorkloadSpec",
    "build_benchmark_record",
    "build_scheduler_run_plan",
    "compute_comparisons",
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
    "resolve_metric_value",
    "run_case",
    "run_case_once",
    "run_suite",
    "summarize_same_case_runs",
    "summarize_samples",
]
