"""具名语义守卫：校验 workload 与 system 联动的约束（在 expand 阶段对每个展开 case 执行）。

这里只放无法由 schema 单独表达的跨块约束；新增检查需同步扩展 schema.CheckSpec。
"""

from collections.abc import Sequence

from mini_infra.bench.spec.common.schema import (
    CheckSpec,
    MinMaxRunningRequestsCheck,
    MinOutputLenCheck,
)


def run_check(
    check: CheckSpec,
    *,
    suite_name: str | None = None,
    context: str = "",
    max_running_requests: int | None,
    output_budgets: Sequence[int],
    arrival_ms: Sequence[float],
) -> None:
    """对单个展开 case 执行一条检查，违反则抛 ValueError。"""
    label = f"suite {suite_name}" if suite_name is not None else context
    prefix = f"{label}: " if label else ""

    if check == "free_batch_slot_at_t0":
        assert max_running_requests is not None
        at_t0 = sum(1 for a in arrival_ms if a == 0.0)
        if at_t0 >= max_running_requests:
            raise ValueError(
                f"{prefix}t=0 到达的请求数 ({at_t0}) 必须小于 max_running_requests "
                f"({max_running_requests})，以保留空闲 batch slot"
            )
    elif isinstance(check, MinMaxRunningRequestsCheck):
        assert max_running_requests is not None
        if max_running_requests < check.min_max_running_requests:
            raise ValueError(
                f"{prefix}要求 max_running_requests >= {check.min_max_running_requests}，实际为 {max_running_requests}"
            )
    elif isinstance(check, MinOutputLenCheck):
        if min(output_budgets) < check.min_output_len:
            raise ValueError(f"{prefix}要求每个请求的输出预算 >= {check.min_output_len}")
    else:
        raise TypeError(f"未知检查: {check!r}")


__all__ = ["run_check"]
