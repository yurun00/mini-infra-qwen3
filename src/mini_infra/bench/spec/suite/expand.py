"""SuiteSpec 展开：sweep 笛卡尔积 / 显式场景 → 与引擎无关的 CaseSpec 列表（纯 CPU，不依赖 vocab 与 torch）。

展开顺序（与历史手写 suite 的 case 顺序严格一致）：
    workload sweep 点（外层） → max_running_requests → batching_method → caching_method → block_size（仅 paged）
"""

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from mini_infra.bench.cases import CaseKey
from mini_infra.bench.common import format_lengths
from mini_infra.bench.spec.common.schema import PagedCacheSpec, SchedulerEngineSpec, SuiteEngine
from mini_infra.bench.spec.common.workload import ResolvedWorkloadSpec, resolve_workload_body
from mini_infra.bench.spec.suite.checks import run_check
from mini_infra.bench.spec.suite.schema import SuiteSpec
from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.engine import BatchingMethod, CachingMethod, Mode


@dataclass(frozen=True)
class CaseSpec:
    """一个展开后的用例规格：具体负载 + 具体系统配置（尚未绑定 vocab / 具体 token）。"""

    workload: ResolvedWorkloadSpec
    mode: Mode
    max_running_requests: int | None
    paged_cache_config: PagedCacheConfig | None

    def format_summary(self, engine: SuiteEngine) -> str:
        """格式化展开后的单用例规格摘要（供 `bench suites show` 预览）。"""
        if engine == "scheduler":
            assert self.max_running_requests is not None
            return str(
                CaseKey.from_mode(
                    self.workload.workload_id,
                    self.mode,
                    self.max_running_requests,
                    self.paged_cache_config,
                )
            )
        return (
            f"[{self.workload.workload_id}] | {self.mode} | batch={self.workload.num_total_requests} "
            f"prompt={format_lengths(self.workload.prompt_lens)} "
            f"max_new_tokens={format_lengths(self.workload.output_budgets)}"
        )


def _expand_workload_points(spec: SuiteSpec) -> list[dict[str, Any]]:
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


def expand_workload(spec: SuiteSpec) -> list[ResolvedWorkloadSpec]:
    """把 SuiteSpec 的 workload 模板按 sweep.workload 展开为 ResolvedWorkloadSpec 列表。"""
    return [resolve_workload(spec, point) for point in _expand_workload_points(spec)]


def _generate_system_cases(
    spec: SuiteSpec,
) -> Iterator[tuple[int | None, Mode, PagedCacheConfig | None]]:
    """展开系统配置维度的组合：(max_running_requests, mode, paged_cache_config)。"""
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
    caching_axis: list[CachingMethod] = (
        list(sweep_sys.caching_method) if sweep_sys.caching_method is not None else [base_sys.caching_method.method]
    )

    if isinstance(base_sys.caching_method, PagedCacheSpec):
        default_block_size = base_sys.caching_method.block_size
        default_num_blocks = base_sys.caching_method.num_blocks
    else:
        default_block_size = 16
        default_num_blocks = None

    block_axis: list[int] = list(sweep_sys.block_size) if sweep_sys.block_size is not None else [default_block_size]

    for mrr, batch, cache in itertools.product(mrr_axis, batching_axis, caching_axis):
        mode = Mode(cache, batching_method=batch)
        if cache is CachingMethod.PAGED:
            for bs in block_axis:
                yield mrr, mode, PagedCacheConfig(block_size=bs, num_blocks=default_num_blocks)
        else:
            yield mrr, mode, None


def _validate_checks(
    spec: SuiteSpec,
    workloads: list[ResolvedWorkloadSpec],
    sys_cases: list[tuple[int | None, Mode, PagedCacheConfig | None]],
) -> None:
    if not spec.checks:
        return
    mrr_axis = list(dict.fromkeys(mrr for mrr, _, _ in sys_cases))
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
            mode=mode,
            max_running_requests=mrr,
            paged_cache_config=paged_config,
        )
        for workload in workloads
        for mrr, mode, paged_config in sys_cases
    ]


__all__ = [
    "CaseSpec",
    "expand_suite_spec",
    "expand_workload",
    "resolve_workload",
]
