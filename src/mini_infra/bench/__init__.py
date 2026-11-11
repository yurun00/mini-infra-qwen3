"""推理性能评测套件（bench 模块）。

包含模型静态开销、硬件理论峰值上限、动态指标计算与统计聚合标准结构。
统一承载固定批处理、动态批处理与无 Cache 场景下的吞吐与延迟基准对比。
"""

from mini_infra.bench.cases import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP,
    BenchmarkCase,
    CaseDimension,
    CaseKey,
    ProfilingMode,
    WorkloadProfile,
)
from mini_infra.bench.comparison import (
    Comparison,
    ComparisonMetric,
    ComparisonResult,
    MetricKind,
    compute_comparisons,
)
from mini_infra.bench.context import (
    BenchmarkExecutionContext,
)
from mini_infra.bench.correctness import (
    CorrectnessReport,
    check_correctness,
)
from mini_infra.bench.efficiency import (
    EfficiencyMetrics,
    HardwarePeakSpecs,
    ModelCostProfile,
)
from mini_infra.bench.env import (
    git_info,
    gpu_info,
    require_cuda,
)
from mini_infra.bench.format import format_lengths
from mini_infra.bench.lockstep import (
    LockstepCaseResult,
    format_table,
)
from mini_infra.bench.paths import REPO_ROOT
from mini_infra.bench.result import (
    SCHEMA_VERSION,
    AnyBenchmarkRecord,
    CaseKeyRecord,
    ComparisonResultRecord,
    LockstepBenchmarkRecord,
    PagedCacheConfigRecord,
    SchedulerBenchmarkRecord,
    build_benchmark_record,
    load_benchmark_record,
    save_benchmark_record,
    validate_benchmark_record,
)
from mini_infra.bench.scheduler import (
    SchedulerCaseResult,
)
from mini_infra.bench.spec.common.lengths import make_arrival_times
from mini_infra.bench.spec.release import (
    RELEASE_SCALES,
    ReleaseScale,
    release_workloads,
)
from mini_infra.bench.stats import DistributionStats
from mini_infra.bench.suite import (
    BenchmarkSuite,
    get_all_benchmark_suites,
    get_benchmark_suite,
    list_benchmark_suite_names,
    load_benchmark_suite,
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
    "ProfilingMode",
    "RELEASE_SCALES",
    "REPO_ROOT",
    "ReleaseScale",
    "SCHEMA_VERSION",
    "SchedulerBenchmarkRecord",
    "SchedulerCaseResult",
    "WorkloadProfile",
    "build_benchmark_record",
    "check_correctness",
    "compute_comparisons",
    "format_lengths",
    "format_table",
    "get_all_benchmark_suites",
    "get_benchmark_suite",
    "git_info",
    "gpu_info",
    "list_benchmark_suite_names",
    "load_benchmark_record",
    "load_benchmark_suite",
    "make_arrival_times",
    "make_sequence",
    "make_sequences",
    "release_workloads",
    "require_cuda",
    "save_benchmark_record",
    "validate_benchmark_record",
]
