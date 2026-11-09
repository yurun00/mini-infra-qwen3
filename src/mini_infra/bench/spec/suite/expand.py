"""SuiteSpec 展开：sweep 笛卡尔积 / 显式场景 → 与引擎无关的 CaseSpec 列表（纯 CPU，不依赖 vocab 与 torch）。

展开顺序（与历史手写 suite 的 case 顺序严格一致）：
    workload sweep 点（外层） → max_running_requests → batching_method → caching_method → block_size（仅 paged）
"""

import itertools
from dataclasses import dataclass
from typing import Any

from mini_infra.bench.spec.common.workload import ResolvedWorkloadSpec, resolve_workload_body
from mini_infra.bench.spec.suite.checks import run_check
from mini_infra.bench.spec.suite.schema import SuiteSpec
from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.engine import CachingMethod, Mode


@dataclass(frozen=True)
class CaseSpec:
    """一个展开后的用例规格：具体负载 + 具体系统配置（尚未绑定 vocab / 具体 token）。"""

    workload: ResolvedWorkloadSpec
    mode: Mode
    max_running_requests: int | None
    paged_cache_config: PagedCacheConfig | None


def _workload_points(spec: SuiteSpec) -> list[dict[str, Any]]:
    sweep = spec.sweep.workload
    if not sweep:
        return [{}]
    if isinstance(sweep, list):
        return [dict(scenario) for scenario in sweep]
    keys = list(sweep)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(sweep[k] for k in keys))]


def resolve_workload(spec: SuiteSpec, point: dict[str, Any]) -> ResolvedWorkloadSpec:
    """在一个 sweep 点上解析 suite 的 workload 模板。"""
    variables = {**spec.workload.vars, **point}
    context = f"suite {spec.name} (sweep 点 {point or '{}'})"
    return resolve_workload_body(spec.workload.id, spec.workload.body, variables, context)


def expand_suite_spec(spec: SuiteSpec) -> list[CaseSpec]:
    """把 SuiteSpec 展开为有序的 CaseSpec 列表。"""
    system = spec.system
    mrr_axis: list[int | None] = list(spec.sweep.system.get("max_running_requests", [system.max_running_requests]))
    block_axis = spec.sweep.system.get("block_size", [system.paged.block_size])

    case_specs: list[CaseSpec] = []
    for point in _workload_points(spec):
        workload = resolve_workload(spec, point)
        for mrr in mrr_axis:
            for check in spec.checks:
                run_check(
                    check,
                    suite_name=spec.name,
                    max_running_requests=mrr,
                    output_budgets=workload.output_budgets,
                    arrival_ms=workload.arrival_ms,
                )
            for batching_method in system.batching_method:
                for caching_method in system.caching_method:
                    paged_configs: list[PagedCacheConfig | None] = (
                        [PagedCacheConfig(block_size=bs, num_blocks=system.paged.num_blocks) for bs in block_axis]
                        if caching_method is CachingMethod.PAGED
                        else [None]
                    )
                    for paged_cache_config in paged_configs:
                        case_specs.append(
                            CaseSpec(
                                workload=workload,
                                mode=Mode(caching_method, batching_method=batching_method),
                                max_running_requests=mrr,
                                paged_cache_config=paged_cache_config,
                            )
                        )
    return case_specs


__all__ = [
    "CaseSpec",
    "expand_suite_spec",
    "resolve_workload",
]
