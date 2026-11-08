"""队列调度器（Scheduler）基准测试终端展示与表格格式化（format.py）。"""

from collections.abc import Sequence

from mini_infra.bench.scheduler.cases import CaseKey
from mini_infra.bench.scheduler.types import MetricComparison, SchedulerCaseResult
from mini_infra.bench.schema import build_benchmark_record


def _abbreviate_workload_id(wid: str) -> str:
    """提取 workload_id 各词前 3 个字符缩写，保留 '_' 分隔符（如 long_short_stress -> lon_sho_str）。"""
    if not wid:
        return ""
    parts = [p for p in wid.replace("-", "_").split("_") if p]
    return "_".join(p if p.isdigit() else p[:3] for p in parts)


def _abbreviate_suite_name(name: str) -> str:
    """提取 suite name 各词前 3 个字符缩写，保留 '-' 分隔符（如 dense-vs-paged-throughput -> den-vs-pag-thr）。"""
    if not name:
        return ""
    parts = [p for p in name.replace("_", "-").split("-") if p]
    return "-".join(p if p.isdigit() else p[:3] for p in parts)


def _abbreviate_mode(mode_str: str) -> str:
    """提取 mode 各维度各词首字母缩写，保留 '+' 分隔符（如 dense-kv-cache+continuous-batching -> dkc+cb）。"""
    if not mode_str:
        return ""
    abbr_parts: list[str] = []
    for part in mode_str.split("+"):
        words = [w for w in part.replace("_", "-").split("-") if w]
        abbr_parts.append("".join(w[0] for w in words))
    return "+".join(abbr_parts)


def _format_case_row(res: SchedulerCaseResult, suite_name: str) -> list[str]:
    """渲染单个 Scheduler 用例的一行表格字段。"""
    slot_util = res.packing.slot_utilization.mean if res.packing is not None else 0.0
    block_size_str = str(res.key.paged_cache_config.block_size) if res.key.paged_cache_config is not None else ""
    return [
        _abbreviate_suite_name(suite_name),
        _abbreviate_workload_id(res.key.workload_id),
        _abbreviate_mode(str(res.key.mode)),
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
    """将 Scheduler 用例结果列表格式化为列对齐的终端展示表格。"""
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
    rows = [_format_case_row(res, suite_name) for res in case_results]
    all_rows = [header, *rows]
    col_widths = [max(len(row[i]) for row in all_rows) for i in range(len(header))]

    def align(cell: str, i: int) -> str:
        return cell.ljust(col_widths[i]) if i < 3 else cell.rjust(col_widths[i])

    header_line = " | ".join(align(h, i) for i, h in enumerate(header))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(header)))
    data_lines = [" | ".join(align(val, i) for i, val in enumerate(r)) for r in rows]
    return "\n".join([header_line, sep_line, *data_lines])


def _abbreviate_case_key(key: CaseKey) -> str:
    """提取用例键的缩写显示名（如 [vol] | dkc+cb | mrr=16 或 [lss] | pkc+cb | mrr=16 | blk-sz=16）。"""
    parts = [
        f"[{_abbreviate_workload_id(key.workload_id)}]",
        _abbreviate_mode(str(key.mode)),
        f"mrr={key.max_running_requests}",
    ]
    if key.paged_cache_config is not None:
        parts.append(f"blk-sz={key.paged_cache_config.block_size}")
    return " | ".join(parts)


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
        f"  {_abbreviate_case_key(result.key):<32}: makespan {result.schedule_metrics.makespan_ms.mean:.2f} ms | "
        f"{result.metrics.throughput_tok_s.mean:.1f} tok/s | "
        f"{result.metrics.throughput_req_s.mean:.2f} req/s | "
        f"slot_util {slot_util * 100:.1f}% | "
        f"ttft_p95 {result.schedule_metrics.ttft_ms_p95.mean:.2f} ms | "
        f"itl_p95 {result.schedule_metrics.itl_ms_p95.mean:.2f} ms"
        f"{cache_info}{pool_info}"
    )


def format_comparison(comp: MetricComparison) -> str:
    """格式化单条基准对比摘要行。"""
    prefix = f"  [{_abbreviate_workload_id(comp.candidate_key.workload_id)}]"
    cand_info = _abbreviate_mode(str(comp.candidate_key.mode))
    if comp.candidate_key.paged_cache_config is not None:
        cand_info += f" (block={comp.candidate_key.paged_cache_config.block_size})"
    return f"{prefix} {comp.metric_name} ({cand_info} vs baseline): {comp.formatted_value()}"


__all__ = [
    "_abbreviate_case_key",
    "_abbreviate_mode",
    "_abbreviate_suite_name",
    "_abbreviate_workload_id",
    "_format_case_row",
    "build_benchmark_record",
    "format_case_summary",
    "format_comparison",
    "format_table",
]
