"""队列调度器（Scheduler）基准测试终端展示与表格格式化（format.py）。"""

from collections.abc import Sequence

from mini_infra.bench.scheduler.types import MetricComparison, SchedulerCaseResult
from mini_infra.bench.schema import build_benchmark_record


def _format_case_row(res: SchedulerCaseResult, suite_name: str) -> list[str]:
    """渲染单个 Scheduler 用例的一行 CSV 字段。"""
    slot_util = res.packing.slot_utilization.mean if res.packing is not None else 0.0
    block_size_str = str(res.key.paged_cache_config.block_size) if res.key.paged_cache_config is not None else ""
    return [
        suite_name,
        res.key.workload_id,
        str(res.key.mode),
        str(res.key.max_running_requests),
        block_size_str,
        f"{res.schedule_metrics.makespan_ms.mean:.3f}",
        f"{res.metrics.throughput_req_s.mean:.3f}",
        f"{res.metrics.throughput_tok_s.mean:.3f}",
        f"{slot_util * 100:.2f}%",
        f"{res.schedule_metrics.ttft_ms_p95.mean:.3f}",
        f"{res.schedule_metrics.itl_ms_p95.mean:.3f}",
        f"{res.schedule_metrics.cache_management_fraction.mean * 100:.2f}%",
        f"{res.memory.kv_cache.peak_kv_cache_pool_mb:.2f}",
        str(res.memory.kv_cache.accessed_kv_cache_tokens),
    ]


def format_table(case_results: Sequence[SchedulerCaseResult], suite_name: str = "custom") -> str:
    """将 Scheduler 用例结果列表格式化为标准 CSV 文本。"""
    header = [
        "suite",
        "workload_id",
        "mode",
        "max_running_requests",
        "block_size",
        "makespan_ms_mean",
        "throughput_req_s_mean",
        "throughput_tok_s_mean",
        "execution_slot_utilization_mean",
        "ttft_ms_p95_mean",
        "itl_ms_p95_mean",
        "cache_management_fraction_mean",
        "peak_kv_cache_pool_mb",
        "accessed_kv_cache_tokens",
    ]
    lines = [",".join(header)]
    for res in case_results:
        lines.append(",".join(_format_case_row(res, suite_name)))
    return "\n".join(lines)


def format_case_summary(result: SchedulerCaseResult) -> str:
    """格式化单用例的延迟/吞吐/槽位利用率摘要行（与 lockstep.format.format_case_summary 对齐）。"""
    cache_frac = result.schedule_metrics.cache_management_fraction.mean
    cache_info = f" | cache {cache_frac * 100:.1f}%" if cache_frac > 0 else ""
    slot_util = result.packing.slot_utilization.mean if result.packing is not None else 0.0
    pool_info = (
        f" | pool {result.memory.kv_cache.peak_kv_cache_pool_mb:.1f} MB"
        if result.memory.kv_cache.peak_kv_cache_pool_mb > 0
        else ""
    )
    return (
        f"  {result.key.display_name:<32}: makespan {result.schedule_metrics.makespan_ms.mean:.2f} ms | "
        f"{result.metrics.throughput_tok_s.mean:.1f} tok/s | "
        f"{result.metrics.throughput_req_s.mean:.2f} req/s | "
        f"slot_util {slot_util * 100:.1f}% | "
        f"ttft_p95 {result.schedule_metrics.ttft_ms_p95.mean:.2f} ms | "
        f"itl_p95 {result.schedule_metrics.itl_ms_p95.mean:.2f} ms"
        f"{cache_info}{pool_info}"
    )


def format_comparison(comp: MetricComparison) -> str:
    """格式化单条基准对比摘要行。"""
    prefix = f"  [{comp.candidate_key.workload_id}]"
    cand_info = f"{comp.candidate_key.mode}"
    if comp.candidate_key.paged_cache_config is not None:
        cand_info += f" (block={comp.candidate_key.paged_cache_config.block_size})"
    return f"{prefix} {comp.metric_name} ({cand_info} vs baseline): {comp.formatted_value()}"


__all__ = [
    "_format_case_row",
    "build_benchmark_record",
    "format_case_summary",
    "format_comparison",
    "format_table",
]
