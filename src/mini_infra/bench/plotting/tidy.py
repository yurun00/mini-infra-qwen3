"""Benchmark 结果 Pandas 宽表与 Tidy Data 转换及跨边界声明式绘图配置（tidy.py）。"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from mini_infra.bench.common import format_lengths
from mini_infra.bench.schema import (
    LockstepBenchmarkRecord,
    SchedulerBenchmarkRecord,
    validate_benchmark_record,
)


@dataclass(frozen=True)
class MetricPlotSpec:
    """单个绘图子图的声明式指标规格。"""

    metric_key: str  # 宽表中的统一指标前缀，如 "ttft_ms", "tpot_ms", "itl_ms_p95"
    title: str  # 子图标题
    ylabel: str  # Y 轴标签
    chart_kind: Literal["line", "bar"] = "line"  # 默认使用折线图
    scale: float = 1.0  # 数值缩放系数（如利用率 * 100 转为百分比）


# 1. 跨边界共享指标（Lockstep 与 Scheduler 均可提取并对齐）
SHARED_PLOT_SPECS: tuple[MetricPlotSpec, ...] = (
    MetricPlotSpec("ttft_ms", "TTFT / Prefill Latency (ms)", "TTFT (ms)"),
    MetricPlotSpec("tpot_ms", "TPOT / ITL P50 Latency (ms/token)", "Latency (ms/token)"),
    MetricPlotSpec("throughput_tok_s", "Output Throughput (tokens/s)", "Throughput (tokens/s)"),
    MetricPlotSpec("throughput_req_s", "Request Throughput (req/s)", "Throughput (req/s)"),
    MetricPlotSpec("e2e_wall_ms", "End-to-End Wall Latency (ms)", "E2E Latency (ms)"),
    MetricPlotSpec("peak_allocated_mb", "Peak Allocated Memory (MiB)", "Peak Memory (MiB)"),
    MetricPlotSpec(
        "slot_utilization",
        "Execution Slot Utilization (%)",
        "Slot Utilization (%)",
        scale=100.0,
    ),
)

# 2. Lockstep 专属指标（仅当待比较记录全部为 lockstep 时显示）
LOCKSTEP_ONLY_PLOT_SPECS: tuple[MetricPlotSpec, ...] = ()

# 3. Scheduler 专属指标（仅当待比较记录全部为 scheduler 时显示）
SCHEDULER_ONLY_PLOT_SPECS: tuple[MetricPlotSpec, ...] = (
    MetricPlotSpec("makespan_ms", "Schedule Makespan (ms)", "Makespan (ms)"),
    MetricPlotSpec("queue_ms_p95", "Queue Latency P95 (ms)", "Queue P95 (ms)"),
    MetricPlotSpec("ttft_ms_p95", "TTFT P95 Latency (ms)", "TTFT P95 (ms)"),
    MetricPlotSpec("itl_ms_p95", "ITL P95 Latency (ms/token)", "ITL P95 (ms/token)"),
    MetricPlotSpec(
        "running_capacity_utilization",
        "Running Capacity Utilization (%)",
        "Capacity Utilization (%)",
        scale=100.0,
    ),
)


def resolve_record_label(
    rec: LockstepBenchmarkRecord | SchedulerBenchmarkRecord | dict[str, Any], custom: str | None = None
) -> str:
    """提取对齐显示的模式/版本标签。"""
    if custom:
        return custom
    tag = rec.get("tag") or ""
    feat = rec.get("release_feature")
    mode = rec.get("mode") or ""
    dirty = "*" if (rec.get("git") or {}).get("dirty") else ""
    if feat:
        return f"{tag} ({feat}){dirty}"
    if mode and tag and mode != tag:
        return f"{mode} ({tag}){dirty}"
    return f"{mode or tag or 'unknown'}{dirty}"


def resolve_plot_specs(
    records: Sequence[LockstepBenchmarkRecord | SchedulerBenchmarkRecord | dict[str, Any]],
) -> list[MetricPlotSpec]:
    """根据待比较记录的 benchmark_type 集合，动态解析应绘制的指标面板：

    - 全为 lockstep：SHARED_PLOT_SPECS + LOCKSTEP_ONLY_PLOT_SPECS（含 tpot_ms）
    - 全为 scheduler：SHARED_PLOT_SPECS + SCHEDULER_ONLY_PLOT_SPECS（含 makespan/queue/itl 等）
    - 跨越 lockstep <-> scheduler 边界：仅返回 SHARED_PLOT_SPECS（自动隐藏不可跨边界的专属子图）
    """
    validated = [validate_benchmark_record(r, require_single_case=True) for r in records]
    btypes = {r.benchmark_type for r in validated}
    if btypes == {"lockstep"}:
        return [*SHARED_PLOT_SPECS, *LOCKSTEP_ONLY_PLOT_SPECS]
    if btypes == {"scheduler"}:
        return [*SHARED_PLOT_SPECS, *SCHEDULER_ONLY_PLOT_SPECS]
    return list(SHARED_PLOT_SPECS)


def records_to_wide_df(
    records: Sequence[LockstepBenchmarkRecord | SchedulerBenchmarkRecord | dict[str, Any]],
    labels: Sequence[str | None] | None = None,
) -> pd.DataFrame:
    """将多个单 Case BenchmarkRecord 通过 pd.json_normalize 转换为对齐的 Pandas 宽表。

    强制校验每个输入记录只包含 1 个 case (require_single_case=True)。
    自动将 Lockstep 的 ttft_ms 与 Scheduler 的 schedule.ttft_ms_p50 对齐至统一的 ttft_ms.* 列。
    """
    validated = [validate_benchmark_record(r, require_single_case=True) for r in records]
    if not validated:
        return pd.DataFrame()

    resolved_labels = [
        resolve_record_label(rec, labels[i] if labels and i < len(labels) else None) for i, rec in enumerate(validated)
    ]

    case_dicts: list[dict[str, Any]] = []
    for idx, (rec, lbl) in enumerate(zip(validated, resolved_labels)):
        c_dict = rec.cases[0].to_dict()
        c_dict["order_idx"] = idx
        c_dict["label"] = lbl
        c_dict["tag"] = rec.tag
        c_dict["benchmark_type"] = rec.benchmark_type
        c_dict["commit"] = rec.git.get("commit", "n/a")
        c_dict["gpu"] = rec.env.get("gpu", "Unknown GPU")
        c_dict["prompt_len_str"] = format_lengths(c_dict["prompt_len"])
        c_dict["max_new_tokens_str"] = format_lengths(c_dict["max_new_tokens"])
        c_dict["workload_desc"] = (
            f"bs={c_dict['batch_size']}, prompt={c_dict['prompt_len_str']}, out={c_dict['max_new_tokens_str']}"
        )
        corr = rec.correctness
        if isinstance(corr, dict) and "passed" in corr:
            c_dict["correctness_passed"] = corr.get("passed")
            c_dict["correctness_match_ratio"] = corr.get("match_ratio")
        elif isinstance(corr, dict):
            sub_passes = [v.get("passed") for v in corr.values() if isinstance(v, dict)]
            sub_ratios = [
                v.get("match_ratio") for v in corr.values() if isinstance(v, dict) and v.get("match_ratio") is not None
            ]
            c_dict["correctness_passed"] = all(sub_passes) if sub_passes else None
            c_dict["correctness_match_ratio"] = min(sub_ratios) if sub_ratios else None
        else:
            c_dict["correctness_passed"] = None
            c_dict["correctness_match_ratio"] = None
        case_dicts.append(c_dict)

    wide_df = pd.json_normalize(case_dicts)
    extra_cols: dict[str, Any] = {}

    # 1. 跨边界对齐：
    #    - Lockstep 的 ttft_ms <-> Scheduler 的 schedule.ttft_ms_p50
    #    - Lockstep 的 tpot_ms <-> Scheduler 的 schedule.itl_ms_p50
    for target_prefix, sched_prefix in (
        ("ttft_ms", "schedule.ttft_ms_p50"),
        ("tpot_ms", "schedule.itl_ms_p50"),
    ):
        for stat in ("median", "mean", "std", "min", "p90", "p95", "p99", "max"):
            target_col = f"{target_prefix}.{stat}"
            sched_col = f"{sched_prefix}.{stat}"
            if target_col not in wide_df.columns and sched_col in wide_df.columns:
                extra_cols[target_col] = wide_df[sched_col]
            elif target_col in wide_df.columns and sched_col in wide_df.columns:
                wide_df[target_col] = wide_df[target_col].fillna(wide_df[sched_col])

    # 2. 将 Scheduler 专属 schedule.<metric>.* 提升至顶层 <metric>.* 供 MetricPlotSpec 统一索引
    sched_exclusive_keys = (
        "makespan_ms",
        "busy_ms",
        "cache_management_ms",
        "cache_management_fraction",
        "queue_ms_p50",
        "queue_ms_p95",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "end_to_end_ms_p50",
        "end_to_end_ms_p95",
        "itl_ms_p50",
        "itl_ms_p95",
        "itl_ms_max",
    )
    for key in sched_exclusive_keys:
        for stat in ("median", "mean", "std", "min", "p90", "p95", "p99", "max"):
            src = f"schedule.{key}.{stat}"
            dst = f"{key}.{stat}"
            if src in wide_df.columns:
                extra_cols[dst] = wide_df[src]

    # 3. 将 packing.<metric>.* 提升至顶层 <metric>.* 供 MetricPlotSpec 统一索引
    for key in ("padding_efficiency", "slot_utilization", "running_capacity_utilization"):
        for stat in ("median", "mean", "std", "min", "p90", "p95", "p99", "max"):
            src = f"packing.{key}.{stat}"
            dst = f"{key}.{stat}"
            if src in wide_df.columns:
                extra_cols[dst] = wide_df[src]

    # 4. 统一标量/分布混合指标：peak_allocated_mb 与无 packing 单请求记录的 slot_utilization 默认回退
    if "memory.peak_allocated_mb" in wide_df.columns:
        extra_cols["peak_allocated_mb.median"] = wide_df["memory.peak_allocated_mb"]
        extra_cols["peak_allocated_mb.std"] = 0.0

    if "slot_utilization.median" in extra_cols:
        extra_cols["slot_utilization.median"] = extra_cols["slot_utilization.median"].fillna(1.0)
    else:
        extra_cols["slot_utilization.median"] = pd.Series(1.0, index=wide_df.index, dtype=float)
    if "slot_utilization.std" in extra_cols:
        extra_cols["slot_utilization.std"] = extra_cols["slot_utilization.std"].fillna(0.0)
    else:
        extra_cols["slot_utilization.std"] = pd.Series(0.0, index=wide_df.index, dtype=float)

    if extra_cols:
        wide_df = pd.concat([wide_df, pd.DataFrame(extra_cols, index=wide_df.index)], axis=1)

    return wide_df


def wide_to_tidy_df(
    wide_df: pd.DataFrame,
    specs: Sequence[MetricPlotSpec],
) -> pd.DataFrame:
    """利用 DataFrame.melt 将宽表按指定的 MetricPlotSpec 列表转换为 Seaborn 消费的 Tidy 长表。"""
    if wide_df.empty or not specs:
        return pd.DataFrame()

    spec_by_key = {s.metric_key: s for s in specs}
    valid_keys = [s.metric_key for s in specs if f"{s.metric_key}.median" in wide_df.columns]
    if not valid_keys:
        return pd.DataFrame()

    id_vars = ["order_idx", "label", "tag", "benchmark_type", "workload_desc"]
    median_cols = {f"{k}.median": k for k in valid_keys}
    std_cols = {f"{k}.std": k for k in valid_keys if f"{k}.std" in wide_df.columns}

    melted_median = wide_df[id_vars + list(median_cols.keys())].melt(
        id_vars=id_vars,
        value_vars=list(median_cols.keys()),
        var_name="metric_col",
        value_name="median",
    )
    melted_median["metric_key"] = melted_median["metric_col"].map(median_cols)
    melted_median = melted_median.drop(columns=["metric_col"])

    if std_cols:
        melted_std = wide_df[["order_idx", *std_cols.keys()]].melt(
            id_vars=["order_idx"],
            value_vars=list(std_cols.keys()),
            var_name="metric_col",
            value_name="std",
        )
        melted_std["metric_key"] = melted_std["metric_col"].map(std_cols)
        melted_std = melted_std.drop(columns=["metric_col"])
        tidy = melted_median.merge(melted_std, on=["order_idx", "metric_key"], how="left")
        tidy["std"] = tidy["std"].fillna(0.0)
    else:
        tidy = melted_median
        tidy["std"] = 0.0

    tidy["scale"] = tidy["metric_key"].map(lambda k: spec_by_key[k].scale)
    tidy["median"] = tidy["median"] * tidy["scale"]
    tidy["std"] = tidy["std"] * tidy["scale"]
    tidy["title"] = tidy["metric_key"].map(lambda k: spec_by_key[k].title)
    tidy["ylabel"] = tidy["metric_key"].map(lambda k: spec_by_key[k].ylabel)
    tidy["chart_kind"] = tidy["metric_key"].map(lambda k: spec_by_key[k].chart_kind)

    return tidy.dropna(subset=["median"]).reset_index(drop=True)


__all__ = [
    "LOCKSTEP_ONLY_PLOT_SPECS",
    "MetricPlotSpec",
    "SCHEDULER_ONLY_PLOT_SPECS",
    "SHARED_PLOT_SPECS",
    "records_to_wide_df",
    "resolve_plot_specs",
    "resolve_record_label",
    "wide_to_tidy_df",
]
