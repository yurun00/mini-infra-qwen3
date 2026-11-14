"""队列调度器（Scheduler）基准测试终端展示与表格格式化（format.py）。"""

from collections.abc import Sequence

from mini_infra.bench.scheduler.types import SchedulerCaseResult


def _abbreviate_suite_name(name: str) -> str:
    """提取 suite name 各词前 3 个字符缩写，保留 '-' 分隔符（如 dense-vs-paged-throughput -> den-vs-pag-thr）。"""
    if not name:
        return ""
    parts = [p for p in name.replace("_", "-").split("-") if p]
    return "-".join(p if p.isdigit() else p[:3] for p in parts)


def _format_case_row(res: SchedulerCaseResult, suite_name: str) -> list[str]:
    """渲染单个 Scheduler 用例的一行表格字段。"""
    slot_util = res.packing.slot_utilization.mean if res.packing is not None else 0.0
    block_size_str = str(res.key.block_size) if res.key.block_size is not None else ""
    token_budget_str = str(res.key.token_budget) if res.key.token_budget is not None else ""
    return [
        _abbreviate_suite_name(suite_name),
        res.key.abbreviated_workload_id,
        res.key.abbreviated_engine_config,
        str(res.key.max_running_requests),
        block_size_str,
        token_budget_str,
        f"{res.schedule_metrics.makespan_ms.mean:.3f}",
        f"{res.metrics.throughput_req_s.mean:.3f}",
        f"{res.metrics.throughput_tok_s.mean:.3f}",
        f"{slot_util * 100:.2f}%",
        f"{res.schedule_metrics.ttft_ms_p95.mean:.3f}",
        f"{res.schedule_metrics.itl_ms_p95.mean:.3f}",
        f"{res.schedule_metrics.itl_ms_max.mean:.3f}",
        f"{res.schedule_metrics.max_consecutive_prefill_steps.median:.0f}",
        f"{res.schedule_metrics.cache_management_fraction.mean * 100:.2f}%",
        f"{res.memory.kv_cache.peak_kv_cache_pool_mb:.2f}",
        str(res.memory.kv_cache.accessed_kv_cache_tokens),
    ]


def format_table(case_results: Sequence[SchedulerCaseResult], suite_name: str = "custom") -> str:
    """将 Scheduler 用例结果列表格式化为列对齐的终端展示表格。"""
    header = [
        "suite",
        "workload_id",
        "engine_config",
        "max_running_requests",
        "block_size",
        "token_budget",
        "makespan_ms_mean",
        "throughput_req_s_mean",
        "throughput_tok_s_mean",
        "slot_utilization_mean",
        "ttft_ms_p95_mean",
        "itl_ms_p95_mean",
        "itl_ms_max_mean",
        "max_consecutive_prefill_steps",
        "cache_management_fraction_mean",
        "peak_kv_cache_pool_mb",
        "accessed_kv_cache_tokens",
    ]
    rows = [_format_case_row(res, suite_name) for res in case_results]
    all_rows = [header, *rows]
    col_widths = [max(len(row[i]) for row in all_rows) for i in range(len(header))]

    def align(cell: str, i: int) -> str:
        return cell.ljust(col_widths[i]) if i < 3 else cell.rjust(col_widths[i])

    header_line = " | ".join(align(h, i) for i, h in enumerate(header))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(header)))
    data_lines = [" | ".join(align(val, i) for i, val in enumerate(r)) for r in rows]
    return "\n".join([header_line, sep_line, *data_lines])


__all__ = [
    "format_table",
]
