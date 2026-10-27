"""单用例 Serving 基准测试执行内核（run_case）：一次调用只执行正好一个 Case。

提供 ServingSuiteCase 的原子单次执行与多次评测汇总纯函数接口，供 CLI 或上层评测套件调用。
"""

import torch

from mini_infra.bench.serving.suites import ServingSuiteCase
from mini_infra.bench.serving.types import (
    ServingCaseResult,
    summarize_serving_runs,
)
from mini_infra.core.types import ScheduleResult
from mini_infra.engine import run_scheduler


def run_case_once(
    model: torch.nn.Module,
    device: torch.device,
    case: ServingSuiteCase,
) -> ScheduleResult:
    """执行单一 ServingSuiteCase 的单次调度运行（原子操作）。"""
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
    case: ServingSuiteCase,
    warmup: int = 1,
    repeats: int = 3,
) -> ServingCaseResult:
    """连续执行单个 Case（warmup + repeats），汇总并返回 ServingCaseResult。"""
    for _ in range(warmup):
        run_case_once(model, device, case)

    runs = [run_case_once(model, device, case) for _ in range(repeats)]
    return summarize_serving_runs(case.id, str(case.mode), runs)


def print_single_case_table(case_name: str, res: ServingCaseResult) -> None:
    """控制台打印单一 Serving 用例的指标结果。"""
    print(
        "  [%s | %s]: makespan %.2f ms | %.1f tok/s | %.2f req/s | "
        "slot_util %.1f%% | ttft_p95 %.2f ms | itl_p95 %.2f ms | cache %.1f%%"
        % (
            case_name,
            res.case_id,
            res.makespan_ms.mean,
            res.output_token_throughput_tok_s.mean,
            res.request_throughput_req_s.mean,
            res.execution_slot_utilization.mean * 100,
            res.ttft_ms_p95.mean,
            res.itl_ms_p95.mean,
            res.cache_management_fraction.mean * 100,
        )
    )


__all__ = [
    "print_single_case_table",
    "run_case",
    "run_case_once",
]
