"""在线与动态连续批处理（Continuous Batching）Serving 基准测试模块。"""

from mini_infra.bench.serving.plan import (
    PlannedRun,
    RunPhase,
    build_serving_run_plan,
    format_serving_run_plan,
    parse_interleave_dimension,
)
from mini_infra.bench.serving.run_case import (
    main as run_case_main,
)
from mini_infra.bench.serving.run_case import (
    run_case,
    run_case_once,
)
from mini_infra.bench.serving.run_suite import (
    build_serving_record,
    print_serving_table,
    run_suite,
)
from mini_infra.bench.serving.run_suite import (
    main as run_suite_main,
)
from mini_infra.bench.serving.suites import (
    SERVING_SUITES,
    FixedVsContinuousPrefillInterferenceSuite,
    FixedVsContinuousStaggeredArrivalsSuite,
    FixedVsContinuousUniformOutputLengthSuite,
    FixedVsContinuousVariableOutputLengthSuite,
    ServingSuite,
    ServingSuiteCase,
    get_all_serving_suites,
    get_serving_suite,
)
from mini_infra.bench.serving.types import (
    ServingCaseResult,
    ServingSuiteResult,
    summarize_serving_runs,
)

__all__ = [
    "FixedVsContinuousPrefillInterferenceSuite",
    "FixedVsContinuousStaggeredArrivalsSuite",
    "FixedVsContinuousUniformOutputLengthSuite",
    "FixedVsContinuousVariableOutputLengthSuite",
    "PlannedRun",
    "RunPhase",
    "SERVING_SUITES",
    "ServingCaseResult",
    "ServingSuite",
    "ServingSuiteCase",
    "ServingSuiteResult",
    "build_serving_record",
    "build_serving_run_plan",
    "format_serving_run_plan",
    "get_all_serving_suites",
    "get_serving_suite",
    "parse_interleave_dimension",
    "print_serving_table",
    "run_case",
    "run_case_main",
    "run_case_once",
    "run_suite",
    "run_suite_main",
    "summarize_serving_runs",
]
