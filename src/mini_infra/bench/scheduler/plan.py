"""Scheduler 基准评测执行计划生成器（纯 CPU、无 PyTorch 依赖）。

负责根据 Suite 的用例列表与 interleave 特征配置，
生成严格确定性的执行计划序列（PlannedRun 列表）。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mini_infra.bench.scheduler.suites import SchedulerSuiteCase
from mini_infra.engine import BatchingMethod, FeatureDimension, FeatureFlag, Mode

SUPPORTED_FEATURES = set(Mode.FEATURE_DIMENSIONS)
ALL_FEATURE_DIMENSIONS = Mode.FEATURE_DIMENSIONS


class RunPhase(str, Enum):
    """运行阶段枚举：WARMUP（预热，不计入统计）或 MEASURE（正式测量）。"""

    WARMUP = "warmup"
    MEASURE = "measure"


@dataclass(frozen=True)
class PlannedRun:
    """单一计划执行步：包含目标用例、执行阶段、测量轮次及所属特征分组 ID。"""

    case: SchedulerSuiteCase
    phase: RunPhase
    repeat_index: int | None
    group_id: int


def parse_interleave_dimension(name: str) -> FeatureDimension:
    """将 CLI 字符串参数解析为强类型 FeatureDimension。"""
    cleaned = name.strip()
    if cleaned == "batching_method":
        return BatchingMethod
    if cleaned == FeatureFlag.KV_CACHE.value:
        return FeatureFlag.KV_CACHE
    raise ValueError(f"未知的 interleave 特征: {name}，可选: ['batching_method', '{FeatureFlag.KV_CACHE.value}']")


def extract_case_feature(case: SchedulerSuiteCase, feature: FeatureDimension) -> Any:
    """从用例的 Mode 中提取指定特征维度的取值。"""
    return case.mode.feature_value(feature)


def build_scheduler_run_plan(
    cases: Sequence[SchedulerSuiteCase],
    warmup: int,
    repeats: int,
    interleave: Sequence[FeatureDimension] = (BatchingMethod,),
) -> list[PlannedRun]:
    """根据用例列表、预热轮数、测量轮数以及交替特征选项构建执行计划序列。

    调度语义：
    1. 选中的 feature（interleave）：同一分组内按 repeat 轮次轮换执行顺序，以消除热漂移。
    2. 未选中的 feature：作为外层分组键，每个分组连续跑完所有 repeat 后才切换到下一分组。

    Args:
        cases: 展开后的 SchedulerSuiteCase 用例列表。
        warmup: 每个用例在测量前执行的预热轮数。
        repeats: 正式测量轮数。
        interleave: 选中的交替特征序列（如 (BatchingMethod,) 或 ()）。

    Returns:
        list[PlannedRun]: 有序的计划执行步列表。
    """
    if not cases or repeats < 1:
        return []

    for feat in interleave:
        if feat not in SUPPORTED_FEATURES:
            raise TypeError(f"未知的 interleave 特征维度: {feat!r}，必须为 FeatureDimension 类型")

    # 1. 选中的特征集合
    interleaved_features = set(interleave)

    # 2. 未选中的特征作为分组键
    non_interleaved_features = [f for f in ALL_FEATURE_DIMENSIONS if f not in interleaved_features]

    # 3. 按未选中的特征将用例归类到不同的分组，保持用例原本的插入顺序
    groups: dict[tuple[Any, ...], list[SchedulerSuiteCase]] = {}
    for case in cases:
        key = tuple(extract_case_feature(case, f) for f in non_interleaved_features)
        groups.setdefault(key, []).append(case)

    # 4. 分组依次连续跑完所有 repeat
    plan: list[PlannedRun] = []
    for group_id, (_group_key, group_cases) in enumerate(groups.items()):
        # 4.1 组内用例预热
        if warmup > 0:
            for case in group_cases:
                for _ in range(warmup):
                    plan.append(
                        PlannedRun(
                            case=case,
                            phase=RunPhase.WARMUP,
                            repeat_index=None,
                            group_id=group_id,
                        )
                    )

        # 4.2 组内用例正式测量：连续跑完 repeats 轮，每轮轮换顺序
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
                        repeat_index=r,
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
        rep = f"repeat {planned_run.repeat_index + 1}" if planned_run.repeat_index is not None else "warmup"
        case = planned_run.case
        lines.append(
            f"  [{i:02d}] 组 {planned_run.group_id} | {planned_run.phase.value:<7} | {rep:<10} | "
            f"suite={case.suite_name} case_id={case.id} mode={case.mode} max_running={case.max_running_requests}"
        )
    return "\n".join(lines)


__all__ = [
    "ALL_FEATURE_DIMENSIONS",
    "PlannedRun",
    "RunPhase",
    "SUPPORTED_FEATURES",
    "build_scheduler_run_plan",
    "extract_case_feature",
    "format_scheduler_run_plan",
    "parse_interleave_dimension",
]
