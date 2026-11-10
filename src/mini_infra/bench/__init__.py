"""推理性能评测套件（bench 模块）。

包含模型静态开销、硬件理论峰值上限、动态指标计算与统计聚合标准结构。
统一承载固定批处理、动态批处理与无 Cache 场景下的吞吐与延迟基准对比。
"""

from mini_infra.bench.cases import BenchmarkCase
from mini_infra.bench.common import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    REPO_ROOT,
    CorrectnessReport,
    DistributionStats,
    EfficiencyMetrics,
    HardwarePeakSpecs,
    ModelCostProfile,
    add_common_args,
    check_correctness,
    format_lengths,
    git_info,
    gpu_info,
    lookup_gpu_peak_specs,
    model_cost,
    require_cuda,
    save_benchmark_record,
    summarize_samples,
)
from mini_infra.bench.context import (
    BenchmarkExecutionContext,
    prepare_execution_context,
)
from mini_infra.bench.lockstep import (
    LockstepCaseResult,
    format_case_summary,
    format_table,
)
from mini_infra.bench.scheduler import (
    CaseDimension,
    CaseKey,
    Comparison,
    ComparisonMetric,
    ComparisonResult,
    MetricKind,
    SchedulerCaseResult,
    WorkloadProfile,
    compute_comparisons,
    format_comparison,
    summarize_same_case_runs,
)
from mini_infra.bench.schema import (
    AnyBenchmarkRecord,
    CaseKeyRecord,
    ComparisonResultRecord,
    LockstepBenchmarkRecord,
    PagedCacheConfigRecord,
    SchedulerBenchmarkRecord,
    build_benchmark_record,
    load_benchmark_record,
    validate_benchmark_record,
)
from mini_infra.bench.spec.common.lengths import make_arrival_times
from mini_infra.bench.spec.release import (
    RELEASE_SCALES,
    ReleaseScale,
    release_default_mode,
    release_workloads,
)
from mini_infra.bench.suite import BenchmarkSuite
from mini_infra.bench.suites import (
    get_all_benchmark_suites,
    get_benchmark_suite,
    list_benchmark_suite_names,
    load_benchmark_suite,
    suite_from_loaded,
)
from mini_infra.bench.synthetic import (
    make_sequence,
    make_sequences,
)

__all__ = [
    "AnyBenchmarkRecord",
    "BenchmarkCase",
    "BenchmarkExecutionContext",
    "BenchmarkSuite",
    "CaseDimension",
    "CaseKey",
    "CaseKeyRecord",
    "Comparison",
    "ComparisonMetric",
    "ComparisonResult",
    "ComparisonResultRecord",
    "CorrectnessReport",
    "DEFAULT_REPEATS",
    "DEFAULT_WARMUP",
    "DistributionStats",
    "EfficiencyMetrics",
    "HardwarePeakSpecs",
    "LockstepBenchmarkRecord",
    "LockstepCaseResult",
    "MetricKind",
    "ModelCostProfile",
    "PagedCacheConfigRecord",
    "RELEASE_SCALES",
    "REPO_ROOT",
    "ReleaseScale",
    "SchedulerBenchmarkRecord",
    "SchedulerCaseResult",
    "WorkloadProfile",
    "add_common_args",
    "build_benchmark_record",
    "check_correctness",
    "compute_comparisons",
    "format_case_summary",
    "format_comparison",
    "format_lengths",
    "format_table",
    "get_all_benchmark_suites",
    "get_benchmark_suite",
    "git_info",
    "gpu_info",
    "list_benchmark_suite_names",
    "load_benchmark_record",
    "load_benchmark_suite",
    "lookup_gpu_peak_specs",
    "make_arrival_times",
    "make_sequence",
    "make_sequences",
    "model_cost",
    "prepare_execution_context",
    "release_default_mode",
    "release_workloads",
    "require_cuda",
    "save_benchmark_record",
    "suite_from_loaded",
    "summarize_same_case_runs",
    "summarize_samples",
    "validate_benchmark_record",
]
