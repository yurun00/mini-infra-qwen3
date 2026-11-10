"""齐步（Lockstep）基准测试终端展示与表格格式化（format.py）。"""

from collections.abc import Sequence

from mini_infra.bench.cases import LockstepCase
from mini_infra.bench.common import format_lengths
from mini_infra.bench.lockstep.types import LockstepCaseResult
from mini_infra.bench.schema import build_benchmark_record


def _format_case_row(w: LockstepCaseResult, has_packing: bool) -> list[str]:
    """渲染单个用例的一行 CSV 字段（列顺序与 format_table 表头一致）。"""
    ttft = w.metrics.ttft_ms
    tpot = w.metrics.mean_tpot_ms
    row = [
        str(w.key.mode),
        format_lengths(w.workload.prompt_lens),
        format_lengths(w.workload.output_budgets),
        f"{(ttft.mean if ttft else 0.0):.3f}",
        f"{(tpot.mean if tpot else 0.0):.3f}",
        f"{(tpot.median if tpot else 0.0):.3f}",
        f"{(tpot.p95 if tpot else 0.0):.3f}",
        f"{w.metrics.e2e_wall_ms.mean:.3f}",
        f"{w.metrics.e2e_model_ms.mean:.3f}",
        f"{(w.memory.prefill_mb or 0.0):.2f}",
        f"{w.memory.peak_allocated_mb:.2f}",
        f"{w.memory.kv_cache.peak_live_kv_cache_mb:.2f}",
    ]
    pack = w.packing
    if has_packing:
        if pack:
            eff_p = f"{w.metrics.effective_prompt_tok_s.mean:.1f}" if w.metrics.effective_prompt_tok_s else ""
            row.extend(
                [
                    f"{pack.padding_efficiency.mean * 100:.2f}%",
                    f"{pack.slot_utilization.mean * 100:.2f}%",
                    eff_p,
                ]
            )
        else:
            row.extend(["", "", ""])
    return row


def format_table(case_results: Sequence[LockstepCaseResult]) -> str:
    """将用例结果列表格式化为标准 CSV 文本。"""
    header = [
        "mode",
        "prompt_tokens",
        "generated_tokens",
        "prefill_ms_mean",
        "decode_ms_mean",
        "decode_ms_p50",
        "decode_ms_p95",
        "end_to_end_wall_ms_mean",
        "end_to_end_model_ms_mean",
        "memory_after_prefill_mib_max",
        "peak_memory_mib_max",
        "peak_live_kv_cache_mb",
    ]

    has_packing = any(w.packing is not None for w in case_results)
    if has_packing:
        header.extend(["padding_efficiency", "slot_utilization", "effective_prompt_tok_s"])

    lines = [",".join(header)]
    for w in case_results:
        lines.append(",".join(_format_case_row(w, has_packing)))
    return "\n".join(lines)


def format_case_summary(result: LockstepCaseResult) -> str:
    """格式化单用例的延迟/吞吐/显存摘要与 MBU/MFU 行。"""
    ttft = result.metrics.ttft_ms
    tpot = result.metrics.mean_tpot_ms
    lines = [
        "  TTFT %.2f ms | TPOT %.2f±%.2f ms | %.1f tok/s | %.2f req/s | %.0f MiB"
        % (
            ttft.median if ttft else 0.0,
            tpot.median if tpot else 0.0,
            tpot.std if tpot else 0.0,
            result.metrics.throughput_tok_s.median,
            result.metrics.throughput_req_s.median,
            result.memory.peak_allocated_mb,
        )
    ]
    if result.efficiency and result.efficiency.decode_mbu is not None:
        gpu_busy_str = (
            "%.0f%%" % (100 * result.metrics.gpu_time_ratio.median)
            if result.metrics.gpu_time_ratio is not None
            else "N/A"
        )
        lines.append(
            "  MBU %.1f%% | decode MFU %.2f%% | prefill MFU %.1f%% | GPU busy %s"
            % (
                100 * result.efficiency.decode_mbu,
                100 * (result.efficiency.decode_mfu or 0),
                100 * (result.efficiency.prefill_mfu or 0),
                gpu_busy_str,
            )
        )
    return "\n".join(lines)


def format_suite_plan(suite_name: str, cases: "Sequence[LockstepCase]", warmup: int, repeats: int) -> str:
    """格式化 Lockstep 套件的用例列表，用于 --dry-run 预览（用例顺序即执行顺序）。"""
    lines = [
        f"Lockstep 基准评测执行计划（suite={suite_name}，共 {len(cases)} 个用例，"
        f"每个 warmup={warmup} repeats={repeats}）："
    ]
    for i, case in enumerate(cases, 1):
        lines.append(
            f"  [{i:02d}] mode={case.mode} batch={case.profile.num_total_requests} "
            f"prompt={format_lengths(case.profile.prompt_lens)} "
            f"max_new_tokens={format_lengths(case.profile.output_budgets)}"
        )
    return "\n".join(lines)


__all__ = [
    "_format_case_row",
    "build_benchmark_record",
    "format_case_summary",
    "format_suite_plan",
    "format_table",
]
