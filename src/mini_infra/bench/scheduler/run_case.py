"""单用例 Scheduler 基准测试执行内核（run_case）：一次调用只执行正好一个 Case。

提供 SchedulerSuiteCase 的原子单次执行与多次评测汇总纯函数接口，供 CLI 或上层评测套件调用。
"""

from collections.abc import Callable
from dataclasses import replace

import torch

from mini_infra.bench.scheduler.suites import SchedulerSuiteCase
from mini_infra.bench.scheduler.types import (
    SchedulerCaseResult,
    summarize_same_workload_runs,
)
from mini_infra.engine import ScheduleResult, run_scheduler


def run_case_once(
    model: torch.nn.Module,
    device: torch.device,
    case: SchedulerSuiteCase,
) -> ScheduleResult:
    """执行单一 SchedulerSuiteCase 的单次调度运行（原子操作）。"""
    return run_scheduler(
        batching_method=case.mode.batching_method,
        model=model,
        request_specs=case.specs,
        max_running_requests=case.max_running_requests,
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
    )


def run_case(
    model: torch.nn.Module,
    device: torch.device,
    case: SchedulerSuiteCase,
    warmup: int = 1,
    repeats: int = 3,
    on_step_progress: Callable[[str, int, int], None] | None = None,
) -> SchedulerCaseResult:
    """连续执行单个 Case（warmup + repeats），汇总并返回 SchedulerCaseResult。"""
    for i in range(1, warmup + 1):
        if on_step_progress is not None:
            on_step_progress("warmup", i, warmup)
        run_case_once(model, device, case)

    runs: list[ScheduleResult] = []
    for i in range(1, repeats + 1):
        if on_step_progress is not None:
            on_step_progress("measure", i, repeats)
        runs.append(run_case_once(model, device, case))
    res = summarize_same_workload_runs(case.id, runs)
    return replace(res, workload=replace(res.workload, warmup=warmup))


__all__ = [
    "run_case",
    "run_case_once",
]
