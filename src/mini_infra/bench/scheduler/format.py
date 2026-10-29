"""队列调度器（Scheduler）基准测试终端展示与表格格式化（format.py）。"""

from collections.abc import Sequence

from mini_infra.bench.scheduler.types import SchedulerCaseResult
from mini_infra.bench.schema import build_benchmark_record


def compute_continuous_makespan_speedup(case_results: Sequence[SchedulerCaseResult]) -> float:
    """若结果列表中同时存在 fixed 与 continuous 用例，动态计算 makespan 加速比。"""
    by_id = {c.case_id: c for c in case_results}
    if "fixed" in by_id and "continuous" in by_id:
        f_mean = by_id["fixed"].schedule_metrics.makespan_ms.mean
        c_mean = by_id["continuous"].schedule_metrics.makespan_ms.mean
        return f_mean / c_mean if c_mean > 0 else 1.0
    return 1.0


def _format_case_row(res: SchedulerCaseResult, suite_name: str, speedup: float) -> list[str]:
    """渲染单个 Scheduler 用例的一行 CSV 字段。"""
    speedup_str = f"{speedup:.2f}x" if "continuous" in res.case_id else "1.00x"
    slot_util = res.packing.slot_utilization.mean if res.packing is not None else 0.0
    return [
        suite_name,
        res.case_id,
        str(res.mode),
        f"{res.schedule_metrics.makespan_ms.mean:.3f}",
        f"{res.metrics.throughput_req_s.mean:.3f}",
        f"{res.metrics.throughput_tok_s.mean:.3f}",
        f"{slot_util * 100:.2f}%",
        f"{res.schedule_metrics.ttft_ms_p95.mean:.3f}",
        f"{res.schedule_metrics.itl_ms_p95.mean:.3f}",
        f"{res.schedule_metrics.cache_management_fraction.mean * 100:.2f}%",
        speedup_str,
    ]


def format_table(case_results: Sequence[SchedulerCaseResult], suite_name: str = "custom") -> str:
    """将 Scheduler 用例结果列表格式化为标准 CSV 文本。"""
    header = [
        "suite",
        "case_id",
        "mode",
        "makespan_ms_mean",
        "throughput_req_s_mean",
        "throughput_tok_s_mean",
        "execution_slot_utilization_mean",
        "ttft_ms_p95_mean",
        "itl_ms_p95_mean",
        "cache_management_fraction_mean",
        "speedup_vs_fixed",
    ]
    speedup = compute_continuous_makespan_speedup(case_results)
    lines = [",".join(header)]
    for res in case_results:
        lines.append(",".join(_format_case_row(res, suite_name, speedup)))
    return "\n".join(lines)


def format_case_summary(result: SchedulerCaseResult, speedup: float | None = None) -> str:
    """格式化单用例的延迟/吞吐/槽位利用率摘要行（与 lockstep.format.format_case_summary 对齐）。"""
    cache_frac = result.schedule_metrics.cache_management_fraction.mean
    cache_info = f" | cache {cache_frac * 100:.1f}%" if cache_frac > 0 else ""
    speedup_info = (
        f" | speedup {speedup:.2f}x"
        if speedup is not None and speedup != 1.0 and "continuous" in result.case_id
        else ""
    )
    slot_util = result.packing.slot_utilization.mean if result.packing is not None else 0.0
    return (
        f"  {result.case_id:<12}: makespan {result.schedule_metrics.makespan_ms.mean:.2f} ms | "
        f"{result.metrics.throughput_tok_s.mean:.1f} tok/s | "
        f"{result.metrics.throughput_req_s.mean:.2f} req/s | "
        f"slot_util {slot_util * 100:.1f}% | "
        f"ttft_p95 {result.schedule_metrics.ttft_ms_p95.mean:.2f} ms | "
        f"itl_p95 {result.schedule_metrics.itl_ms_p95.mean:.2f} ms"
        f"{cache_info}{speedup_info}"
    )


__all__ = [
    "_format_case_row",
    "build_benchmark_record",
    "compute_continuous_makespan_speedup",
    "format_case_summary",
    "format_table",
]
