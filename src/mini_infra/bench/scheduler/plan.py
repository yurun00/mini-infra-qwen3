"""Scheduler 基准评测执行计划生成器（纯 CPU、无 PyTorch 依赖）。

负责根据 Suite 的用例列表与 interleave 特征配置，
生成严格确定性的执行计划序列（PlannedRun 列表）。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from mini_infra.bench.cases import (
    CaseDimension,
    CaseDimensionValue,
    CaseKey,
    SchedulerCase,
)
from mini_infra.core.request import RequestSpec


class RunPhase(str, Enum):
    """运行阶段枚举：WARMUP（预热，不计入统计）或 MEASURE（正式测量）。"""

    WARMUP = "warmup"
    MEASURE = "measure"


@dataclass(frozen=True)
class PlannedRun:
    """单一计划执行步：包含目标用例、执行阶段、测量轮次及所属特征分组 ID。"""

    case: SchedulerCase
    phase: RunPhase
    repeat_idx: int | None
    group_id: int


def build_scheduler_run_plan(
    cases: Sequence[SchedulerCase],
    warmup: int,
    repeats: int,
    interleave: Sequence[CaseDimension] = (CaseDimension.BATCHING_METHOD,),
) -> list[PlannedRun]:
    """根据用例列表、预热轮数、测量轮数以及交替特征选项构建执行计划序列。

    调度语义：
    1. 选中的特征（interleave）：同一分组内按 repeat 轮次轮换执行顺序，以消除热漂移。
    2. 未选中的特征：作为外层分组键，每个分组连续跑完所有 repeat 后才切换到下一分组。

    Args:
        cases: 展开后的 SchedulerCase 用例列表。
        warmup: 每个用例在测量前执行的预热轮数。
        repeats: 正式测量轮数。
        interleave: 选中的交替特征序列（如 (CaseDimension.BATCHING_METHOD,) 或 ()）。

    Returns:
        list[PlannedRun]: 有序的计划执行步列表。
    """
    if not cases or repeats < 1:
        return []

    interleaved_dims: list[CaseDimension] = []
    for feat in interleave:
        if isinstance(feat, CaseDimension):
            interleaved_dims.append(feat)
        else:
            try:
                interleaved_dims.append(CaseDimension(feat))
            except ValueError as err:
                raise TypeError(f"未知的 interleave 特征维度: {feat!r}，必须为 CaseDimension 类型") from err

    # 1. 守门检查：确保不存在重复的用例全局唯一身份键（CaseKey）
    seen_keys: set[CaseKey] = set()
    for case in cases:
        if case.key in seen_keys:
            raise ValueError(f"存在重复的用例身份键（CaseKey）: {case.key}")
        seen_keys.add(case.key)

    # 2. 守门检查：同一 workload_id 下的所有用例请求规格（specs）必须一致
    specs_by_workload: dict[str, tuple[RequestSpec, ...]] = {}
    for case in cases:
        if case.workload_id in specs_by_workload:
            if case.specs != specs_by_workload[case.workload_id]:
                raise ValueError(f"同一 workload_id '{case.workload_id}' 下的用例请求规格（specs）不一致")
        else:
            specs_by_workload[case.workload_id] = case.specs

    # 3. 选中的特征与未选中的特征（作为外层分组键）
    interleaved_set = set(interleaved_dims)
    non_interleaved_dims = [d for d in CaseDimension if d not in interleaved_set]

    # 4. 按未选中的特征将用例归类到不同的分组，保持用例原本的插入顺序
    groups: dict[tuple[CaseDimensionValue, ...], list[SchedulerCase]] = {}
    for case in cases:
        key = case.key.project(non_interleaved_dims)
        groups.setdefault(key, []).append(case)

    # 5. 分组依次连续跑完所有 repeat
    plan: list[PlannedRun] = []
    for group_id, (_group_key, group_cases) in enumerate(groups.items()):
        # 5.1 组内用例预热
        if warmup > 0:
            for case in group_cases:
                for _ in range(warmup):
                    plan.append(
                        PlannedRun(
                            case=case,
                            phase=RunPhase.WARMUP,
                            repeat_idx=None,
                            group_id=group_id,
                        )
                    )

        # 5.2 组内用例正式测量：连续跑完 repeats 轮，每轮轮换顺序
        n_cases = len(group_cases)
        for r in range(repeats):
            if n_cases <= 1:
                ordered_cases = list(group_cases)
            elif n_cases == 2:
                ordered_cases = list(group_cases) if r % 2 == 0 else list(reversed(group_cases))
            else:
                shift = r % n_cases
                ordered_cases = list(group_cases[shift:]) + list(group_cases[:shift])

            for case in ordered_cases:
                plan.append(
                    PlannedRun(
                        case=case,
                        phase=RunPhase.MEASURE,
                        repeat_idx=r,
                        group_id=group_id,
                    )
                )

    return plan


def format_scheduler_run_plan(plan: Sequence[PlannedRun]) -> str:
    """格式化打印执行序列，用于日志记录与 --dry-run 预览。"""
    if not plan:
        return "执行计划为空。"
    lines = [f"Scheduler 基准评测执行计划（共 {len(plan)} 步）："]
    for i, planned_run in enumerate(plan, 1):
        rep = f"repeat {planned_run.repeat_idx + 1}" if planned_run.repeat_idx is not None else "warmup"
        case = planned_run.case
        details = f"suite={case.suite_name} key={case.key}"
        lines.append(f"  [{i:02d}] 组 {planned_run.group_id} | {planned_run.phase.value:<7} | {rep:<10} | {details}")
    return "\n".join(lines)


__all__ = [
    "PlannedRun",
    "RunPhase",
    "build_scheduler_run_plan",
    "format_scheduler_run_plan",
]
