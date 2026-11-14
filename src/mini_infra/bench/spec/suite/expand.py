"""SuiteSpec 展开：sweep 笛卡尔积 / 显式场景 → 与引擎无关的 CaseSpec 列表（纯 CPU，不依赖 vocab 与 torch）。

展开顺序（与历史手写 suite 的 case 顺序严格一致）：
    workload sweep 点（外层） → max_running_requests → batching_method → caching_spec → scheduler_spec
"""

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from mini_infra.bench.spec.common.checks import run_check
from mini_infra.bench.spec.common.schema import CachingSpec, SchedulerEngineSpec, SchedulerSpec
from mini_infra.bench.spec.common.workload import ResolvedWorkloadSpec, resolve_workload_body
from mini_infra.bench.spec.suite.schema import SuiteSpec
from mini_infra.engine import BatchingMethod


@dataclass(frozen=True)
class CaseSpec:
    """一个展开后的用例规格：具体负载 + 具体系统配置（尚未绑定 vocab / 具体 token）。"""

    workload: ResolvedWorkloadSpec
    batching_method: BatchingMethod
    caching_spec: CachingSpec
    max_running_requests: int | None
    scheduler_spec: SchedulerSpec


def _expand_workload_points(spec: SuiteSpec) -> list[dict[str, Any]]:
    sweep = spec.sweep.workload
    if not sweep:
        return [{}]
    if isinstance(sweep, list):
        return [dict(scenario) for scenario in sweep]
    keys = list(sweep)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(sweep[k] for k in keys))]


def _resolve_workload(spec: SuiteSpec, point: dict[str, Any]) -> ResolvedWorkloadSpec:
    """在一个 sweep 点上解析 suite 的 workload 模板。"""
    variables = {**spec.workload.vars, **point}
    context = f"suite {spec.name} (sweep 点 {point or '{}'})"
    return resolve_workload_body(spec.workload.id, spec.workload.body, variables, context)


def expand_workload(spec: SuiteSpec) -> list[ResolvedWorkloadSpec]:
    """把 SuiteSpec 的 workload 模板按 sweep.workload 展开为 ResolvedWorkloadSpec 列表。"""
    return [_resolve_workload(spec, point) for point in _expand_workload_points(spec)]


_SystemCase = tuple[int | None, BatchingMethod, CachingSpec, SchedulerSpec]


def _generate_system_cases(spec: SuiteSpec) -> Iterator[_SystemCase]:
    """展开系统配置维度的组合：(max_running_requests, batching_method, caching_spec, scheduler_spec)。"""
    base_sys = spec.system
    sweep_sys = spec.sweep.system

    if isinstance(base_sys.engine_mode, SchedulerEngineSpec):
        base_mrr = base_sys.engine_mode.max_running_requests
    else:
        base_mrr = None

    mrr_axis: list[int | None] = (
        list(sweep_sys.max_running_requests) if sweep_sys.max_running_requests is not None else [base_mrr]
    )
    batching_axis: list[BatchingMethod] = (
        list(sweep_sys.batching_method) if sweep_sys.batching_method is not None else [base_sys.batching_method]
    )
    caching_axis: list[CachingSpec] = (
        list(sweep_sys.caching_spec) if sweep_sys.caching_spec is not None else [base_sys.caching_spec]
    )
    scheduler_axis: list[SchedulerSpec] = (
        list(sweep_sys.scheduler_spec) if sweep_sys.scheduler_spec is not None else [base_sys.scheduler_spec]
    )

    yield from itertools.product(mrr_axis, batching_axis, caching_axis, scheduler_axis)


def _validate_checks(
    spec: SuiteSpec,
    workloads: list[ResolvedWorkloadSpec],
    sys_cases: list[_SystemCase],
) -> None:
    if not spec.checks:
        return
    mrr_axis = list(dict.fromkeys(mrr for mrr, _, _, _ in sys_cases))
    for workload, mrr in itertools.product(workloads, mrr_axis):
        for check in spec.checks:
            run_check(
                check,
                suite_name=spec.name,
                max_running_requests=mrr,
                output_budgets=workload.output_budgets,
                arrival_ms=workload.arrival_ms,
            )


def expand_suite_spec(spec: SuiteSpec) -> list[CaseSpec]:
    """把 SuiteSpec 展开为有序的 CaseSpec 列表。"""
    workloads = expand_workload(spec)
    sys_cases = list(_generate_system_cases(spec))
    _validate_checks(spec, workloads, sys_cases)

    return [
        CaseSpec(
            workload=workload,
            batching_method=batch,
            caching_spec=caching_spec,
            max_running_requests=mrr,
            scheduler_spec=scheduler_spec,
        )
        for workload in workloads
        for mrr, batch, caching_spec, scheduler_spec in sys_cases
    ]


__all__ = [
    "CaseSpec",
    "expand_suite_spec",
    "expand_workload",
]
