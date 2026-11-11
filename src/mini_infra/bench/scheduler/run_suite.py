"""套件基准测试（run_suite）：执行整套 Scheduler 评测套件的纯执行调度内核。

根据 BenchmarkSuite 与配置生成执行计划（支持按特征 interleave），
逐项调用 run_case 模块的原子 step 执行，返回用例结果列表 list[SchedulerCaseResult]。
"""

from collections.abc import Callable, Sequence

import torch

from mini_infra.bench.cases import CaseDimension, CaseKey
from mini_infra.bench.scheduler.plan import (
    PlannedRun,
    RunPhase,
    build_scheduler_run_plan,
)
from mini_infra.bench.scheduler.run_case import run_case_once
from mini_infra.bench.scheduler.types import (
    SchedulerCaseResult,
)
from mini_infra.bench.suite import BenchmarkSuite
from mini_infra.engine import ScheduleResult


def run_suite(
    model: torch.nn.Module,
    device: torch.device,
    suite: BenchmarkSuite,
    vocab_size: int,
    warmup: int | None = None,
    repeats: int | None = None,
    interleave: Sequence[CaseDimension] | None = None,
    on_step_progress: Callable[[int, int, PlannedRun], None] | None = None,
) -> list[SchedulerCaseResult]:
    """运行一个 Scheduler 评测套件：根据计划驱动执行，返回有序的 list[SchedulerCaseResult]。"""
    warmup_val = warmup if warmup is not None else suite.warmup
    repeats_val = repeats if repeats is not None else suite.repeats
    interleave_dims = tuple(interleave) if interleave is not None else suite.interleave

    cases = suite.cases

    # 局部材质化同一 workload 的 requests，保证 suite 内同 workload 严格使用同一份输入
    requests_by_workload = {case.workload_id: case.profile.to_payload(vocab_size) for case in cases}

    # 生成执行计划
    plan = build_scheduler_run_plan(cases, warmup=warmup_val, repeats=repeats_val, interleave=interleave_dims)

    # 驱动执行计划，按用例身份键（CaseKey）归集测量结果
    runs_by_case_key: dict[CaseKey, list[ScheduleResult]] = {case.key: [] for case in cases}
    total_steps = len(plan)
    for step_number, planned_run in enumerate(plan, 1):
        if on_step_progress is not None:
            on_step_progress(step_number, total_steps, planned_run)
        res = run_case_once(
            model,
            device,
            planned_run.case,
            requests=requests_by_workload[planned_run.case.workload_id],
        )
        if planned_run.phase == RunPhase.MEASURE:
            runs_by_case_key[planned_run.case.key].append(res)

    results: list[SchedulerCaseResult] = []
    for case in cases:
        res = SchedulerCaseResult.from_runs(
            case.key,
            runs_by_case_key[case.key],
            warmup=warmup_val,
        )
        results.append(res)
    return results


__all__ = [
    "run_suite",
]
