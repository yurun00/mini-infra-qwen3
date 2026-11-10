"""套件基准测试（run_suite）：执行整套 Scheduler 评测套件的纯执行调度内核。

根据 SchedulerSuite 与配置生成执行计划（支持按特征 interleave），
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
from mini_infra.bench.scheduler.suites import SchedulerSuite
from mini_infra.bench.scheduler.types import (
    SchedulerCaseResult,
    summarize_same_case_runs,
)
from mini_infra.engine import ScheduleResult


def parse_interleave_dimension(name: str) -> CaseDimension:
    """将 CLI 字符串参数解析为强类型 CaseDimension。"""
    cleaned = name.strip()
    try:
        return CaseDimension(cleaned)
    except ValueError as err:
        raise ValueError(f"未知的 interleave 特征维度: {name}，可选: {[d.value for d in CaseDimension]}") from err


def parse_interleave_args(raw: Sequence[str] | None, default: tuple[CaseDimension, ...]) -> tuple[CaseDimension, ...]:
    """将 CLI 字符串参数解析为强类型 CaseDimension 元组。"""
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
    interleave: Sequence[CaseDimension] | None = None,
    on_step_progress: Callable[[int, int, PlannedRun], None] | None = None,
) -> list[SchedulerCaseResult]:
    """运行一个 Scheduler 评测套件：根据计划驱动执行，返回有序的 list[SchedulerCaseResult]。"""
    warmup_val = warmup if warmup is not None else suite.warmup
    repeats_val = repeats if repeats is not None else suite.repeats
    interleave_dims = tuple(interleave) if interleave is not None else suite.interleave

    cases = suite.cases(vocab_size)

    # 局部材质化同一 workload 的 request_specs，保证 suite 内同 workload 严格使用同一份输入
    specs_by_workload = {case.workload_id: case.profile.to_payload(vocab_size) for case in cases}

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
            request_specs=specs_by_workload[planned_run.case.workload_id],
        )
        if planned_run.phase == RunPhase.MEASURE:
            runs_by_case_key[planned_run.case.key].append(res)

    results: list[SchedulerCaseResult] = []
    for case in cases:
        res = summarize_same_case_runs(
            case.key,
            runs_by_case_key[case.key],
            warmup=warmup_val,
        )
        results.append(res)
    return results


__all__ = [
    "parse_interleave_args",
    "parse_interleave_dimension",
    "run_suite",
]
