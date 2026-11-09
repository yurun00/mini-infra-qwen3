"""通用 Workload 解析与变量插值（纯 CPU）。"""

import copy
import random
import re
from dataclasses import dataclass
from typing import Any

from mini_infra.bench.spec.common.lengths import expand_arrival, expand_length
from mini_infra.bench.spec.common.schema import ConstantLength, WorkloadBody, normalize_workload_body

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
    workload_id: str, body: dict[str, Any], variables: dict[str, Any], context: str
) -> ResolvedWorkload:
    """插值、校验并展开一份 workload（id 与 body 均可含 ${var}），返回逐请求数值。

    context 仅用于错误信息（如 `suite demo` 或 `release scheduler/small`）。
    """
    try:
        resolved_id = interpolate(workload_id, variables)
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError(f"workload.id 插值后必须是非空字符串，实际为 {resolved_id!r}")
        parsed = WorkloadBody.model_validate(normalize_workload_body(interpolate(body, variables)))
    except ValueError as err:
        raise ValueError(f"{context}: workload 非法: {err}") from err

    rng = random.Random(parsed.seed * 1000 + parsed.seed_salt) if parsed.seed is not None else None
    prompt_lens: list[int] = []
    budgets: list[int] = []
    arrivals: list[float] = []

    for group in parsed.groups:
        prompt_lens.extend(expand_length(group.prompt_len, group.count, rng))
        budgets.extend(expand_length(group.output_len, group.count, rng))
        arrivals.extend(expand_arrival(group.arrival_ms, group.count, rng))

    first = parsed.groups[0]
    single_group = len(parsed.groups) == 1
    prompt_len: int | tuple[int, ...] = (
        first.prompt_len.value if single_group and isinstance(first.prompt_len, ConstantLength) else tuple(prompt_lens)
    )
    max_new_tokens: int | tuple[int, ...] = (
        first.output_len.value if single_group and isinstance(first.output_len, ConstantLength) else tuple(budgets)
    )
    return ResolvedWorkload(
        workload_id=resolved_id,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        prompt_lens=tuple(prompt_lens),
        output_budgets=tuple(budgets),
        arrival_ms=tuple(arrivals),
    )


__all__ = [
    "ResolvedWorkload",
    "interpolate",
    "resolve_workload_body",
]
