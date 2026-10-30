"""对比两个独立单用例 benchmark JSON 结果，并生成对比图表与数据报表。

通过 Pydantic 统一信封校验单 Case 约束（require_single_case=True），
利用 Pandas 宽表/长表转换与 Seaborn 声明式绘图，自动感知是否跨越 Lockstep <-> Scheduler 边界：
- 纯 Lockstep 对比：绘制共享指标 + Lockstep 专属指标（如 tpot_ms）
- 纯 Scheduler 对比：绘制共享指标 + Scheduler 专属指标（如 makespan_ms, queue_ms_p95, itl_ms_p95 等）
- 跨边界对比（Lockstep vs Scheduler）：仅绘制跨边界共享指标（ttft_ms 自动对齐至 schedule.ttft_ms_p50）
"""

import math
from pathlib import Path
from typing import Any

from mini_infra.bench.plotting.tidy import (
    records_to_wide_df,
    resolve_plot_specs,
    resolve_record_label,
    wide_to_tidy_df,
)
from mini_infra.bench.schema import (
    LockstepBenchmarkRecord,
    SchedulerBenchmarkRecord,
    load_benchmark_record,
)

# 耗时类指标（越小越好，变化比计算为 base / target 加速比）
_LATENCY_KEYS = {
    "ttft_ms",
    "tpot_ms",
    "e2e_wall_ms",
    "makespan_ms",
    "queue_ms_p95",
    "ttft_ms_p95",
    "itl_ms_p95",
}
# 吞吐/利用率类指标（越大越好，变化比计算为 target / base 提升比）
_THROUGHPUT_KEYS = {
    "throughput_tok_s",
    "throughput_req_s",
    "slot_utilization",
    "running_capacity_utilization",
}


def load_record(path: Path) -> LockstepBenchmarkRecord | SchedulerBenchmarkRecord:
    """读取并校验单 Case benchmark JSON 结果（若包含多个 case 则报错）。"""
    return load_benchmark_record(path, require_single_case=True)


def resolve_label(rec: Any, custom: str | None = None) -> str:
    """提取对齐显示的模式/版本标签。"""
    return resolve_record_label(rec, custom)


def _format_delta(metric_key: str, val_a: float, val_b: float) -> str:
    """根据指标物理性质格式化 A -> B 的相对变化（加速比或显存增量）。"""
    if metric_key == "peak_allocated_mb":
        return f"{val_b - val_a:+.1f} MiB"
    if metric_key in _LATENCY_KEYS:
        ratio = val_a / val_b if val_b > 0 else 1.0
        return f"**{ratio:.2f}x**"
    if metric_key in _THROUGHPUT_KEYS:
        ratio = val_b / val_a if val_a > 0 else 1.0
        return f"**{ratio:.2f}x**"
    return f"{val_b - val_a:+.2f}"


def generate_comparison_table(
    rec1: Any,
    rec2: Any,
    label1: str,
    label2: str,
) -> str:
    """基于 Pandas 宽表与边界感知规格生成 Markdown 对比表格。"""
    wide_df = records_to_wide_df([rec1, rec2], labels=[label1, label2])
    if len(wide_df) > 2 and "medium" in set(wide_df["scale"]):
        wide_df = wide_df[wide_df["scale"] == "medium"].reset_index(drop=True)
    specs = resolve_plot_specs([rec1, rec2])
    tidy_df = wide_to_tidy_df(wide_df, specs)

    row_a = wide_df.iloc[0]
    row_b = wide_df.iloc[1]
    gpu_info = row_a["gpu"] if row_a["gpu"] == row_b["gpu"] else f"{row_a['gpu']} vs {row_b['gpu']}"
    engines = {row_a["engine"], row_b["engine"]}
    boundary_note = (
        "跨边界对比 (`lockstep` vs `scheduler`，已仅保留跨边界共享指标)"
        if len(engines) > 1
        else f"同域对比 (`{next(iter(engines))}`)"
    )

    lines = [
        f"### 引擎性能对比：`{label1}` vs `{label2}`",
        f"- **对比类型**：{boundary_note}",
        f"- **测试设备**：{gpu_info}",
        f"- **基准 (A)**：`{label1}` (commit: `{row_a['commit']}`, {row_a['workload_desc']})",
        f"- **对比 (B)**：`{label2}` (commit: `{row_b['commit']}`, {row_b['workload_desc']})",
        "",
        f"| 评测指标 | 基准 A (`{label1}`) | 对比 B (`{label2}`) | 相对变化 (B vs A) |",
        "| :--- | :-: | :-: | :-: |",
    ]

    for spec in specs:
        sub = tidy_df[tidy_df["metric_key"] == spec.metric_key].sort_values("order_idx")
        if len(sub) < 2:
            continue
        val_a = float(sub.iloc[0]["median"])
        val_b = float(sub.iloc[1]["median"])
        delta_str = _format_delta(spec.metric_key, val_a, val_b)
        lines.append(f"| {spec.title} | {val_a:.2f} | {val_b:.2f} | {delta_str} |")

    return "\n".join(lines)


def plot_mode_comparison(
    rec1: Any,
    rec2: Any,
    label1: str,
    label2: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    """使用 Seaborn 与边界感知 MetricPlotSpec 绘制两份单 Case 结果的多维对比图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    wide_df = records_to_wide_df([rec1, rec2], labels=[label1, label2])
    if len(wide_df) > 2 and "medium" in set(wide_df["scale"]):
        wide_df = wide_df[wide_df["scale"] == "medium"].reset_index(drop=True)
    specs = resolve_plot_specs([rec1, rec2])
    tidy_df = wide_to_tidy_df(wide_df, specs)

    ncols = 3
    nrows = math.ceil(len(specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    palette = ["#e74c3c", "#2ecc71"]

    for idx, spec in enumerate(specs):
        ax = flat_axes[idx]
        sub = tidy_df[tidy_df["metric_key"] == spec.metric_key].sort_values("order_idx")
        if sub.empty:
            ax.set_axis_off()
            continue

        if spec.chart_kind == "bar":
            sns.barplot(
                data=sub,
                x="label",
                y="median",
                hue="label",
                palette=palette[: len(sub)],
                legend=False,
                ax=ax,
                alpha=0.85,
                width=0.5,
            )
            for bar in ax.patches:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8.5,
                    )
        else:
            sns.lineplot(
                data=sub,
                x="label",
                y="median",
                marker="o",
                markersize=8,
                linewidth=2.2,
                color="#2980b9",
                ax=ax,
            )
            if sub["std"].gt(0).any():
                ax.errorbar(
                    range(len(sub)),
                    sub["median"],
                    yerr=sub["std"],
                    fmt="none",
                    ecolor="#2980b9",
                    capsize=4,
                    alpha=0.7,
                )
            for x_i, (_, row) in enumerate(sub.iterrows()):
                ax.annotate(
                    f"{row['median']:.2f}",
                    xy=(x_i, row["median"]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )

        ax.set_title(f"{idx + 1}. {spec.title}", fontsize=10.5)
        ax.set_xlabel("")
        ax.set_ylabel(spec.ylabel)
        ax.grid(True, linestyle="--", alpha=0.45)

    for extra_ax in flat_axes[len(specs) :]:
        extra_ax.set_axis_off()

    gpu_name = wide_df.iloc[0]["gpu"]
    default_title = f"Benchmark Comparison: {label1} vs {label2} [{gpu_name}]"
    fig.suptitle(title or default_title, fontsize=13, y=0.995)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已保存至: {out_path}")


__all__ = [
    "generate_comparison_table",
    "load_record",
    "plot_mode_comparison",
    "resolve_label",
]
