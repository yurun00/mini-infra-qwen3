"""离线与静态 Batch 基准测试终端展示、表格格式化与结果记录构建。"""

import argparse
import time
from collections.abc import Sequence
from typing import Any

import torch

from mini_infra.bench.batch.types import CaseBenchmarkResult
from mini_infra.bench.common import (
    SCHEMA_VERSION,
    CorrectnessReport,
    HardwarePeakSpecs,
    ModelCostProfile,
    format_lengths,
)


def _format_case_row(w: CaseBenchmarkResult, mode: str | None, has_packing: bool) -> list[str]:
    """渲染单个用例的一行 CSV 字段（列顺序与 print_table 表头一致）。"""
    row = [
        str(w.workload.mode or mode or ""),
        format_lengths(w.workload.prompt_len),
        format_lengths(w.workload.max_new_tokens),
        f"{w.latency.ttft_ms.mean:.3f}",
        f"{w.latency.tpot_ms.mean:.3f}",
        f"{w.latency.tpot_ms.median:.3f}",
        f"{w.latency.tpot_ms.p95:.3f}",
        f"{w.latency.e2e_wall_ms.mean:.3f}",
        f"{w.latency.e2e_model_ms.mean:.3f}",
        f"{w.memory.prefill_allocated_mb:.2f}",
        f"{w.memory.peak_allocated_mb:.2f}",
        f"{w.memory.kv_cache_mb:.2f}",
    ]
    pack = w.packing
    if has_packing:
        if pack:
            eff_p = f"{pack.effective_prompt_throughput.mean:.1f}" if pack.effective_prompt_throughput else ""
            row.extend([f"{pack.padding_efficiency * 100:.2f}%", f"{pack.slot_utilization * 100:.2f}%", eff_p])
        else:
            row.extend(["", "", ""])
    return row


def print_table(case_results: Sequence[CaseBenchmarkResult], mode: str | None = None) -> None:
    """控制台标准 CSV 格式输出。"""
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
        "kv_cache_mb",
    ]

    has_packing = any(w.packing is not None for w in case_results)
    if has_packing:
        header.extend(["padding_efficiency", "slot_utilization", "effective_prompt_tok_s"])

    print(",".join(header))
    for w in case_results:
        print(",".join(_format_case_row(w, mode, has_packing)))


def print_case_summary(result: CaseBenchmarkResult) -> None:
    """打印单用例的延迟/吞吐/显存摘要与 MBU/MFU 行（run_case 与 run_suite 共用）。"""
    print(
        "  TTFT %.2f ms | TPOT %.2f±%.2f ms | %.1f tok/s | %.2f req/s | %.0f MiB"
        % (
            result.latency.ttft_ms.median,
            result.latency.tpot_ms.median,
            result.latency.tpot_ms.std,
            result.throughput.throughput_tok_s.median,
            result.throughput.throughput_req_s.median,
            result.memory.peak_allocated_mb,
        )
    )
    if result.efficiency and result.efficiency.decode_mbu is not None:
        gpu_busy_str = "%.0f%%" % (100 * result.gpu_time_ratio.median) if result.gpu_time_ratio is not None else "N/A"
        print(
            "  MBU %.1f%% | decode MFU %.2f%% | prefill MFU %.1f%% | GPU busy %s"
            % (
                100 * result.efficiency.decode_mbu,
                100 * (result.efficiency.decode_mfu or 0),
                100 * (result.efficiency.prefill_mfu or 0),
                gpu_busy_str,
            )
        )


def build_benchmark_record(
    args: argparse.Namespace,
    model: torch.nn.Module,
    mode_label: str,
    cases: Sequence[CaseBenchmarkResult],
    correctness: CorrectnessReport | dict[str, Any],
    env: dict[str, Any],
    peaks: HardwarePeakSpecs | dict[str, float | None],
    cost: ModelCostProfile | dict[str, Any] | None,
    git: dict[str, Any],
    warmup: int,
    repeats: int,
    suite_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建标准离线批处理评测结果 JSON 字典。"""
    record_cases = [c.to_record_dict() for c in cases]
    peaks_dict = peaks.to_dict() if hasattr(peaks, "to_dict") else peaks
    cost_dict = cost.to_dict() if cost is not None and hasattr(cost, "to_dict") else cost
    if isinstance(correctness, dict):
        correctness_dict = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in correctness.items()}
    elif hasattr(correctness, "to_dict"):
        correctness_dict = correctness.to_dict()
    else:
        correctness_dict = correctness

    try:
        dtype_str = str(next(model.parameters()).dtype)
    except StopIteration:
        dtype_str = "unknown"

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tag": args.tag,
        "release_feature": getattr(args, "release_feature", None),
        "mode": mode_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git,
        "env": env,
        "peak_specs": peaks_dict,
        "model_cost": cost_dict,
        "config": {
            "warmup": warmup,
            "repeats": repeats,
            "model": args.model,
            "revision": args.revision,
            "dtype": dtype_str,
            "tokenizer_included": False,
            "profile_mode": str(args.profile_mode) if getattr(args, "profile_mode", None) else None,
        },
        "correctness": correctness_dict,
        "cases": record_cases,
    }
    if suite_info is not None:
        record["suite"] = suite_info
    return record


__all__ = [
    "_format_case_row",
    "build_benchmark_record",
    "print_case_summary",
    "print_table",
]
