"""队列调度器（Scheduler: Fixed vs Continuous Batching）基准测试模块。"""

from mini_infra.bench.scheduler.format import (
    build_benchmark_record,
    compute_continuous_makespan_speedup,
    format_case_summary,
    format_table,
)
from mini_infra.bench.scheduler.plan import (
    PlannedRun,
    RunPhase,
    build_scheduler_run_plan,
    format_scheduler_run_plan,
    parse_interleave_dimension,
)
from mini_infra.bench.scheduler.run_case import (
    run_case,
    run_case_once,
)
from mini_infra.bench.scheduler.run_suite import (
    parse_interleave_args,
    run_suite,
)
from mini_infra.bench.scheduler.suites import (
    SCHEDULER_SUITES,
    SUITES,
    FixedVsContinuousPrefillInterferenceSuite,
    FixedVsContinuousStaggeredArrivalsSuite,
    FixedVsContinuousUniformOutputLengthSuite,
    FixedVsContinuousVariableOutputLengthSuite,
    SchedulerSuite,
    SchedulerSuiteCase,
    get_all_scheduler_suites,
    get_scheduler_suite,
    get_suite,
)
from mini_infra.bench.scheduler.types import (
    DistributionStats,
    SchedulerCaseResult,
    WorkloadSpec,
    summarize_same_workload_runs,
    summarize_samples,
)

__all__ = [
    "DistributionStats",
    "FixedVsContinuousPrefillInterferenceSuite",
    "FixedVsContinuousStaggeredArrivalsSuite",
    "FixedVsContinuousUniformOutputLengthSuite",
    "FixedVsContinuousVariableOutputLengthSuite",
    "PlannedRun",
    "RunPhase",
    "SCHEDULER_SUITES",
    "SUITES",
    "SchedulerCaseResult",
    "SchedulerSuite",
    "SchedulerSuiteCase",
    "WorkloadSpec",
    "build_benchmark_record",
    "build_scheduler_run_plan",
    "compute_continuous_makespan_speedup",
    "format_case_summary",
    "format_scheduler_run_plan",
    "format_table",
    "get_all_scheduler_suites",
    "get_scheduler_suite",
    "get_suite",
    "parse_interleave_args",
    "parse_interleave_dimension",
    "run_case",
    "run_case_once",
    "run_suite",
    "summarize_same_workload_runs",
    "summarize_samples",
]
