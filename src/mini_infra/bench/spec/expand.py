"""SuiteSpec 展开：sweep 笛卡尔积 / 显式场景 → 与引擎无关的 CaseBlueprint 列表（纯 CPU，不依赖 vocab 与 torch）。

展开顺序（与历史手写 suite 的 case 顺序严格一致）：
    workload sweep 点（外层） → max_running_requests → batching_method → caching_method → block_size（仅 paged）
"""

import copy
import itertools
import re
from dataclasses import dataclass
from typing import Any

from mini_infra.bench.spec.checks import run_check
from mini_infra.bench.spec.lengths import expand_arrival, expand_length
from mini_infra.bench.spec.schema import ConstantLength, SuiteSpec, WorkloadBody, normalize_workload_body
from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.engine import CachingMethod, Mode

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


@dataclass(frozen=True)
class ResolvedWorkload:
    """插值并展开后的具体请求负载。

    prompt_len / max_new_tokens 保留 lockstep 的广播语义：单组且为 constant 时是标量 int，其余为逐请求 tuple。
    """

    workload_id: str
    prompt_len: int | tuple[int, ...]
    max_new_tokens: int | tuple[int, ...]
    prompt_lens: tuple[int, ...]
    output_budgets: tuple[int, ...]
    arrival_ms: tuple[float, ...]

    @property
    def num_requests(self) -> int:
        return len(self.prompt_lens)


@dataclass(frozen=True)
class CaseBlueprint:
    """一个展开后的用例蓝图：具体负载 + 具体系统配置（尚未绑定 vocab / 具体 token）。"""

    workload: ResolvedWorkload
    mode: Mode
    max_running_requests: int | None
    paged_cache_config: PagedCacheConfig | None


def interpolate(node: Any, variables: dict[str, Any]) -> Any:
    """递归展开 ${var}：整值引用保留原类型（可替换为整个结构），内嵌引用按字符串拼接。"""

    def lookup(name: str) -> Any:
        if name not in variables:
            raise ValueError(f"未定义的变量 ${{{name}}}，可用变量: {sorted(variables)}")
        return variables[name]

    if isinstance(node, str):
        whole = _PLACEHOLDER.fullmatch(node)
        if whole is not None:
            return copy.deepcopy(lookup(whole.group(1)))
        return _PLACEHOLDER.sub(lambda m: str(lookup(m.group(1))), node)
    if isinstance(node, list):
        return [interpolate(item, variables) for item in node]
    if isinstance(node, dict):
        return {key: interpolate(value, variables) for key, value in node.items()}
    return node


def _workload_points(spec: SuiteSpec) -> list[dict[str, Any]]:
    sweep = spec.sweep.workload
    if not sweep:
        return [{}]
    if isinstance(sweep, list):
        return [dict(scenario) for scenario in sweep]
    keys = list(sweep)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(sweep[k] for k in keys))]


def resolve_workload(spec: SuiteSpec, point: dict[str, Any]) -> ResolvedWorkload:
    """在一个 sweep 点上插值并校验 workload，展开为逐请求数值。"""
    variables = {**spec.workload.vars, **point}
    try:
        workload_id = interpolate(spec.workload.id, variables)
        if not isinstance(workload_id, str) or not workload_id:
            raise ValueError(f"workload.id 插值后必须是非空字符串，实际为 {workload_id!r}")
        body = WorkloadBody.model_validate(normalize_workload_body(interpolate(spec.workload.body, variables)))
    except ValueError as err:
        raise ValueError(f"suite {spec.name}: workload 在 sweep 点 {point or '{}'} 处非法: {err}") from err

    prompt_lens: list[int] = []
    budgets: list[int] = []
    arrivals: list[float] = []
    for group in body.groups:
        prompt_lens.extend(expand_length(group.prompt_len, group.count))
        budgets.extend(expand_length(group.output_len, group.count))
        arrivals.extend(expand_arrival(group.arrival_ms, group.count))

    first = body.groups[0]
    single_group = len(body.groups) == 1
    prompt_len: int | tuple[int, ...] = (
        first.prompt_len.value if single_group and isinstance(first.prompt_len, ConstantLength) else tuple(prompt_lens)
    )
    max_new_tokens: int | tuple[int, ...] = (
        first.output_len.value if single_group and isinstance(first.output_len, ConstantLength) else tuple(budgets)
    )
    return ResolvedWorkload(
        workload_id=workload_id,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        prompt_lens=tuple(prompt_lens),
        output_budgets=tuple(budgets),
        arrival_ms=tuple(arrivals),
    )


def expand_suite_spec(spec: SuiteSpec) -> list[CaseBlueprint]:
    """把 SuiteSpec 展开为有序的 CaseBlueprint 列表。"""
    system = spec.system
    mrr_axis: list[int | None] = list(spec.sweep.system.get("max_running_requests", [system.max_running_requests]))
    block_axis = spec.sweep.system.get("block_size", [system.paged.block_size])

    blueprints: list[CaseBlueprint] = []
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
                        blueprints.append(
                            CaseBlueprint(
                                workload=workload,
                                mode=Mode(caching_method, batching_method=batching_method),
                                max_running_requests=mrr,
                                paged_cache_config=paged_cache_config,
                            )
                        )
    return blueprints


__all__ = [
    "CaseBlueprint",
    "ResolvedWorkload",
    "expand_suite_spec",
    "interpolate",
    "resolve_workload",
]
