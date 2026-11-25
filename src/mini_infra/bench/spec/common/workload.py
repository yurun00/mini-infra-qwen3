"""通用 Workload 解析与变量插值（纯 CPU）。"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mini_infra.bench.cases import WorkloadGroup, _check_workload_groups
from mini_infra.bench.spec.common.lengths import expand_arrival, expand_length
from mini_infra.bench.spec.common.schema import (
    WorkloadSpec,
    normalize_workload_body,
)

if TYPE_CHECKING:
    from mini_infra.bench.cases import WorkloadProfile

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


@dataclass(frozen=True)
class ResolvedWorkloadSpec:
    """插值并展开后的具体请求负载规格（纯 CPU，不依赖 vocab 与 torch）。

    持有确定性的逐请求流量序列（prompt_lens / output_budgets / arrival_ms）与全体共享的公共前缀长度。
    """

    workload_id: str
    num_total_requests: int
    prompt_lens: tuple[int, ...]
    output_budgets: tuple[int, ...]
    arrival_ms: tuple[float, ...]
    shared_prefix_len: int = 0
    groups: tuple[WorkloadGroup, ...] = ()

    def __post_init__(self) -> None:
        if self.num_total_requests <= 0:
            raise ValueError(
                f"num_total_requests 必须大于 0，实际为 {self.num_total_requests}"
            )
        if len(self.prompt_lens) != self.num_total_requests:
            raise ValueError(
                f"prompt_lens 长度 ({len(self.prompt_lens)}) 与 num_total_requests ({self.num_total_requests}) 不一致"
            )
        if len(self.output_budgets) != self.num_total_requests:
            raise ValueError(
                f"output_budgets 长度 ({len(self.output_budgets)}) 与 "
                f"num_total_requests ({self.num_total_requests}) 不一致"
            )
        if len(self.arrival_ms) != self.num_total_requests:
            raise ValueError(
                f"arrival_ms 长度 ({len(self.arrival_ms)}) 与 num_total_requests ({self.num_total_requests}) 不一致"
            )
        if self.shared_prefix_len > min(self.prompt_lens):
            raise ValueError(
                f"shared_prefix_len ({self.shared_prefix_len}) 不能大于最短 prompt_len ({min(self.prompt_lens)})"
            )
        _check_workload_groups(self.groups, self.num_total_requests)

    def to_profile(self) -> WorkloadProfile:
        """转换为运行期轻量数值画像 WorkloadProfile。"""
        from mini_infra.bench.cases import WorkloadProfile

        return WorkloadProfile(
            workload_id=self.workload_id,
            num_total_requests=self.num_total_requests,
            prompt_lens=self.prompt_lens,
            output_budgets=self.output_budgets,
            arrival_ms=self.arrival_ms,
            shared_prefix_len=self.shared_prefix_len,
            groups=self.groups,
        )


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


def resolve_workload_body(
    workload_id: str,
    body: dict[str, Any],
    variables: dict[str, Any],
    context: str,
) -> ResolvedWorkloadSpec:
    """插值、校验并展开一份 workload（id 与 body 均可含 ${var}），返回逐请求数值。

    context 仅用于错误信息（如 `suite demo` 或 `release scheduler/small`）。
    """
    try:
        resolved_id = interpolate(workload_id, variables)
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError(
                f"workload.id 插值后必须是非空字符串，实际为 {resolved_id!r}"
            )
        parsed = WorkloadSpec.model_validate(
            normalize_workload_body(interpolate(body, variables))
        )
    except ValueError as err:
        raise ValueError(f"{context}: workload 非法: {err}") from err

    rng = (
        random.Random(parsed.seed * 1000 + parsed.seed_salt)
        if parsed.seed is not None
        else None
    )
    prompt_lens: list[int] = []
    budgets: list[int] = []
    arrivals: list[float] = []
    groups: list[WorkloadGroup] = []

    for group in parsed.groups:
        start = len(prompt_lens)
        prompt_lens.extend(expand_length(group.prompt_len, group.count, rng))
        budgets.extend(expand_length(group.output_len, group.count, rng))
        arrivals.extend(expand_arrival(group.arrival_ms, group.count, rng))
        if group.name is not None:
            groups.append(WorkloadGroup(group.name, start, len(prompt_lens)))

    return ResolvedWorkloadSpec(
        workload_id=resolved_id,
        num_total_requests=len(prompt_lens),
        prompt_lens=tuple(prompt_lens),
        output_budgets=tuple(budgets),
        arrival_ms=tuple(arrivals),
        shared_prefix_len=parsed.shared_prefix_len,
        groups=tuple(groups),
    )


__all__ = [
    "ResolvedWorkloadSpec",
    "interpolate",
    "resolve_workload_body",
]
