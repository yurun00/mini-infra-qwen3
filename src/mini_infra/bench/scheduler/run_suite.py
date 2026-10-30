"""套件基准测试（run_suite）：执行整套 Scheduler 评测套件的纯执行调度内核。

根据 SchedulerSuite 与配置生成执行计划（支持按特征 interleave），
逐项调用 run_case 模块的原子 step 执行，返回用例结果列表 list[SchedulerCaseResult]。
"""

from collections.abc import Callable, Sequence
from dataclasses import replace

import torch

from mini_infra.bench.scheduler.plan import (
    PlannedRun,
    RunPhase,
    build_scheduler_run_plan,
    parse_interleave_dimension,
)
from mini_infra.bench.scheduler.run_case import run_case_once
from mini_infra.bench.scheduler.suites import SchedulerSuite
from mini_infra.bench.scheduler.types import (
    SchedulerCaseResult,
    summarize_same_workload_runs,
)
from mini_infra.engine import FeatureDimension, ScheduleResult


def parse_interleave_args(
    raw: Sequence[str] | None, default: tuple[FeatureDimension, ...]
) -> tuple[FeatureDimension, ...]:
    """将 CLI 字符串参数解析为强类型 FeatureDimension 元组。"""
    if raw is None:
        return default
    if not raw or "none" in raw:
        return ()
    return tuple(parse_interleave_dimension(f) for f in raw)


def run_suite(
    model: torch.nn.Module,
    device: torch.device,
    suite: SchedulerSuite,
    vocab_size: int,
    warmup: int | None = None,
    repeats: int | None = None,
    interleave: Sequence[FeatureDimension] | None = None,
    on_step_progress: Callable[[int, int, PlannedRun], None] | None = None,
) -> list[SchedulerCaseResult]:
    """运行一个 Scheduler 评测套件：根据计划驱动执行，返回有序的 list[SchedulerCaseResult]。"""
    warmup_val = warmup if warmup is not None else suite.warmup
    repeats_val = repeats if repeats is not None else suite.repeats
    interleave_features = tuple(interleave) if interleave is not None else suite.interleave

    cases = suite.cases(vocab_size, device=device)

    # 生成执行计划
    plan = build_scheduler_run_plan(cases, warmup=warmup_val, repeats=repeats_val, interleave=interleave_features)

    # 驱动执行计划，按用例 ID 归集测量结果
    runs_by_case_id: dict[str, list[ScheduleResult]] = {case.id: [] for case in cases}
    total_steps = len(plan)
    for idx, planned_run in enumerate(plan, 1):
        if on_step_progress is not None:
            on_step_progress(idx, total_steps, planned_run)
        res = run_case_once(model, device, planned_run.case)
        if planned_run.phase == RunPhase.MEASURE:
            runs_by_case_id[planned_run.case.id].append(res)

    results: list[SchedulerCaseResult] = []
    for case in cases:
        res = summarize_same_workload_runs(case.id, runs_by_case_id[case.id])
        results.append(replace(res, workload=replace(res.workload, warmup=warmup_val)))
    return results


__all__ = [
    "parse_interleave_args",
    "run_suite",
]
