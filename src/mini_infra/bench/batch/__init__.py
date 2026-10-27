"""离线与静态 Batch 基准测试模块（单用例与套件）。"""

from mini_infra.bench.batch.run_case import main as run_case_main
from mini_infra.bench.batch.run_case import run_case
from mini_infra.bench.batch.run_suite import main as run_suite_main
from mini_infra.bench.batch.suites import (
    SUITES,
    SerialVsFixedBatchThroughputSuite,
    Suite,
    SuiteCase,
    UniformVsVariableBudgetSlotsSuite,
    UniformVsVariablePromptPaddingSuite,
    get_suite,
)
from mini_infra.bench.batch.types import (
    BatchPackingProfile,
    CaseBenchmarkResult,
    CaseMemoryProfile,
    DistributionStats,
    HardwareEfficiency,
    LatencyDistributions,
    ThroughputDistributions,
    WorkloadSpec,
    summarize_samples,
)

__all__ = [
    "SUITES",
    "BatchPackingProfile",
    "CaseBenchmarkResult",
    "CaseMemoryProfile",
    "DistributionStats",
    "HardwareEfficiency",
    "LatencyDistributions",
    "SerialVsFixedBatchThroughputSuite",
    "Suite",
    "SuiteCase",
    "ThroughputDistributions",
    "UniformVsVariableBudgetSlotsSuite",
    "UniformVsVariablePromptPaddingSuite",
    "WorkloadSpec",
    "get_suite",
    "run_case",
    "run_case_main",
    "run_suite_main",
    "summarize_samples",
]
