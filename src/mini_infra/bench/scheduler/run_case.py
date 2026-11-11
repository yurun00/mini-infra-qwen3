"""单用例 Scheduler 基准测试执行内核（run_case）：一次调用只执行正好一个 Case。

提供 BenchmarkCase 的原子单次执行与多次评测汇总纯函数接口，供 CLI 或上层评测套件调用。
"""

import gc
from collections.abc import Callable, Sequence

import torch

from mini_infra.bench.cases import BenchmarkCase
from mini_infra.bench.scheduler.types import (
    SchedulerCaseResult,
)
from mini_infra.core.request import Request
from mini_infra.engine import ScheduleResult, run_scheduler


def run_case_once(
    model: torch.nn.Module,
    device: torch.device,
    case: BenchmarkCase,
    requests: Sequence[Request] | None = None,
) -> ScheduleResult:
    """执行单一 BenchmarkCase 的单次调度运行（原子操作）。"""
    reqs = requests or case.profile.to_payload(model.config.vocab_size)
    res = run_scheduler(
        batching_method=case.mode.batching_method,
        caching_method=case.mode.caching_method,
        model=model,
        requests=reqs,
        max_running_requests=case.max_running_requests,
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
        paged_cache_config=case.paged_cache_config,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def run_case(
    model: torch.nn.Module,
    device: torch.device,
    case: BenchmarkCase,
    warmup: int = 1,
    repeats: int = 3,
    on_step_progress: Callable[[str, int, int], None] | None = None,
    requests: Sequence[Request] | None = None,
) -> SchedulerCaseResult:
    """连续执行单个 Case（warmup + repeats），汇总并返回 SchedulerCaseResult。"""
    reqs = requests or case.profile.to_payload(model.config.vocab_size)
    for i in range(1, warmup + 1):
        if on_step_progress is not None:
            on_step_progress("warmup", i, warmup)
        run_case_once(model, device, case, requests=reqs)

    runs: list[ScheduleResult] = []
    for i in range(1, repeats + 1):
        if on_step_progress is not None:
            on_step_progress("measure", i, repeats)
        runs.append(run_case_once(model, device, case, requests=reqs))
    return SchedulerCaseResult.from_runs(case.key, runs, warmup=warmup)


__all__ = [
    "run_case",
    "run_case_once",
]
