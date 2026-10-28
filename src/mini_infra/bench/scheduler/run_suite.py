"""套件基准测试（run_suite）：执行整套 Scheduler 评测套件的纯执行调度内核。

根据 SchedulerSuite 与配置生成执行计划（支持按特征 interleave），
逐项调用 run_case 模块的原子 step 执行，最后汇总结算加速比与吞吐。
"""

import time
from collections.abc import Sequence
from typing import Any

import torch

from mini_infra.bench.scheduler.plan import (
    RunPhase,
    build_scheduler_run_plan,
    parse_interleave_dimension,
)
from mini_infra.bench.scheduler.run_case import run_case_once
from mini_infra.bench.scheduler.suites import SchedulerSuite
from mini_infra.bench.scheduler.types import (
    SchedulerCaseResult,
    SchedulerSuiteResult,
    summarize_scheduler_runs,
)
from mini_infra.engine import FeatureDimension, ScheduleResult

SCHEMA_VERSION = "1.0"


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
) -> SchedulerSuiteResult:
    """运行一个 Scheduler 评测套件：根据计划驱动执行，支持选定特征在 repeat 间轮换交替。"""
    warmup_val = warmup if warmup is not None else suite.warmup
    repeats_val = repeats if repeats is not None else suite.repeats
    interleave_features = tuple(interleave) if interleave is not None else suite.interleave

    cases = suite.cases(vocab_size, device=device)
    mode_by_case_id = {case.id: str(case.mode) for case in cases}

    # 生成执行计划
    plan = build_scheduler_run_plan(cases, warmup=warmup_val, repeats=repeats_val, interleave=interleave_features)

    # 驱动执行计划，按用例 ID 归集测量结果
    collected: dict[str, list[ScheduleResult]] = {case.id: [] for case in cases}
    for planned_run in plan:
        res = run_case_once(model, device, planned_run.case)
        if planned_run.phase == RunPhase.MEASURE:
            collected[planned_run.case.id].append(res)

    case_results: dict[str, SchedulerCaseResult] = {
        cid: summarize_scheduler_runs(cid, mode_by_case_id[cid], runs) for cid, runs in collected.items()
    }

    # 计算加速比：若存在 fixed 与 continuous 对比，计算 makespan 加速比
    speedup = 1.0
    if "fixed" in case_results and "continuous" in case_results:
        f_mean = case_results["fixed"].makespan_ms.mean
        c_mean = case_results["continuous"].makespan_ms.mean
        speedup = f_mean / c_mean if c_mean > 0 else 1.0

    return SchedulerSuiteResult(
        suite_name=suite.name,
        cases=case_results,
        continuous_makespan_speedup=speedup,
    )


def print_scheduler_table(suite_results: Sequence[SchedulerSuiteResult]) -> None:
    """打印控制台标准 CSV 汇总表格。"""
    header = [
        "suite",
        "case_id",
        "mode",
        "makespan_ms_mean",
        "request_throughput_per_second_mean",
        "output_token_throughput_per_second_mean",
        "execution_slot_utilization_mean",
        "ttft_ms_p95_mean",
        "itl_ms_p95_mean",
        "cache_management_fraction_mean",
        "speedup_vs_fixed",
    ]
    print(",".join(header))
    for suite_res in suite_results:
        for cid, res in suite_res.cases.items():
            speedup_str = f"{suite_res.continuous_makespan_speedup:.2f}x" if "continuous" in cid else "1.00x"
            row = [
                suite_res.suite_name,
                cid,
                res.mode,
                f"{res.makespan_ms.mean:.3f}",
                f"{res.throughput_req_s.mean:.3f}",
                f"{res.throughput_tok_s.mean:.3f}",
                f"{res.execution_slot_utilization.mean * 100:.2f}%",
                f"{res.ttft_ms_p95.mean:.3f}",
                f"{res.itl_ms_p95.mean:.3f}",
                f"{res.cache_management_fraction.mean * 100:.2f}%",
                speedup_str,
            ]
            print(",".join(row))


def build_scheduler_record(
    args: Any,
    model: torch.nn.Module,
    suite: SchedulerSuite,
    suite_result: SchedulerSuiteResult,
    correctness: dict[str, Any],
    env: dict[str, Any],
    peaks: dict[str, float | None],
    cost: dict[str, Any] | None,
    git: dict[str, Any],
) -> dict[str, Any]:
    """构建 Scheduler 评测结果标准 JSON 记录。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "tag": args.tag,
        "benchmark_type": "scheduler",
        "suite": suite_result.suite_name,
        "suite_config": suite.to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git,
        "env": env,
        "peak_specs": peaks,
        "model_cost": cost,
        "config": {
            "model": args.model,
            "revision": args.revision,
            "dtype": str(next(model.parameters()).dtype),
            "decoding": "greedy, forced output budget, EOS disabled",
            "tokenizer_included": False,
        },
        "correctness": correctness,
        "continuous_makespan_speedup": suite_result.continuous_makespan_speedup,
        "cases": {cid: c.to_record_dict() for cid, c in suite_result.cases.items()},
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_scheduler_record",
    "parse_interleave_args",
    "print_scheduler_table",
    "run_suite",
]
