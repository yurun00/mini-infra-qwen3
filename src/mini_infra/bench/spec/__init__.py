"""Benchmark 声明式配置（YAML）：schema、加载、展开与生命周期。

本模块划分为三大子域：
    - `common/`：跨 spec 共享的基础模型（分布、Workload、Execution）、纯 CPU 展开与底层 YAML 解析
    - `suite/`：Suite 领域的声明（`SuiteSpec`）、语义守卫、笛卡尔积展开与套件发现加载
    - `release/`：Release 领域的声明（`ReleaseSpec`）、三档确定性负载展开与加载
"""

from mini_infra.bench.spec.common import (
    CheckSpec,
    PagedSpec,
    ResolvedWorkload,
    SuiteEngine,
    SystemSpec,
    apply_overrides,
    deep_merge,
    load_raw_spec,
    parse_set_args,
    run_check,
)
from mini_infra.bench.spec.release import (
    RELEASE_SCALES,
    LoadedReleaseSpec,
    ReleaseScale,
    ReleaseSpec,
    ResolvedReleaseScale,
    expand_release_workloads,
    get_release_spec,
    load_release_spec,
    release_default_mode,
    release_json_schema,
    release_workloads,
    resolve_release_spec,
)
from mini_infra.bench.spec.suite import (
    CaseBlueprint,
    ComparisonConfig,
    LoadedSuiteSpec,
    MetricSpec,
    SuiteSpec,
    discover_suite_specs,
    expand_suite_spec,
    get_suite_spec,
    load_suite_spec,
    suite_json_schema,
)

__all__ = [
    "RELEASE_SCALES",
    "CaseBlueprint",
    "CheckSpec",
    "ComparisonConfig",
    "LoadedReleaseSpec",
    "LoadedSuiteSpec",
    "MetricSpec",
    "PagedSpec",
    "ReleaseScale",
    "ReleaseSpec",
    "ResolvedReleaseScale",
    "ResolvedWorkload",
    "SuiteEngine",
    "SuiteSpec",
    "SystemSpec",
    "apply_overrides",
    "deep_merge",
    "discover_suite_specs",
    "expand_release_workloads",
    "expand_suite_spec",
    "get_release_spec",
    "get_suite_spec",
    "load_raw_spec",
    "load_release_spec",
    "load_suite_spec",
    "parse_set_args",
    "release_default_mode",
    "release_json_schema",
    "release_workloads",
    "resolve_release_spec",
    "run_check",
    "suite_json_schema",
]
