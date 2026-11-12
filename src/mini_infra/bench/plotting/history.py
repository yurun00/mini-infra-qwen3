"""读取 benchmarks/results/history/*.json（严格校验每个文件只包含 1 个标杆 Case），
渲染两大类分析图表（纯 Matplotlib / Seaborn 可视化，Markdown 报表与瓶颈诊断见 `plotting.report`）：

第一大类：跨版本演化趋势 (Evolution Across Releases)
    自动通过 resolve_plot_specs(records) 感知历史记录是否跨越 lockstep <-> scheduler 边界：
    - 未跨边界（纯 Lockstep 或纯 Scheduler）：绘制共享指标 + 当前域专属指标
      （如 Scheduler 的 itl_ms_p95 / queue_ms_p95 / makespan_ms）
    - 跨越边界（含 Lockstep 与 Scheduler）：绘制跨边界共享指标
      （ttft_ms 对齐至 schedule.ttft_ms_p50，tpot_ms 对齐至 schedule.itl_ms_p50）

第二大类：当前模型深度剖析与优化指引 (Profiling & Optimization Guidance)
    基于 target 版本的单 Case 数据展示 MBU/MFU、模块归因、算子类别与显存足迹。
"""

import math
from pathlib import Path
from typing import Any

from mini_infra.bench.paths import (
    LOCKSTEP_RESULT_HISTORY_DIR,
)
from mini_infra.bench.plotting.tidy import (
    extract_primary_case_dict,
    records_to_wide_df,
    resolve_plot_specs,
    resolve_record_label,
    wide_to_tidy_df,
)
from mini_infra.bench.result import (
    LockstepBenchmarkRecord,
    SchedulerBenchmarkRecord,
    load_benchmark_record,
    validate_benchmark_record,
)


def load_results(history_dir: Path | None = None) -> list[LockstepBenchmarkRecord | SchedulerBenchmarkRecord]:
    """加载指定历史目录中的 *.json 并强制校验每个文件只包含 1 个 Case (require_single_case=True)。"""
    target_dir = history_dir or LOCKSTEP_RESULT_HISTORY_DIR
    json_files = list(target_dir.glob("*.json")) if target_dir.exists() else []
    records = [load_benchmark_record(p, require_single_case=True) for p in json_files]
    records.sort(key=lambda r: r.timestamp)
    return records


_SCALE_PALETTE = {"small": "#27ae60", "medium": "#2980b9", "large": "#e74c3c"}
_SCALE_MARKERS = {"small": "o", "medium": "s", "large": "^"}


def plot_evolution_and_scaling(records: list[Any], base: Any, out_path: Path) -> None:
    """使用 Seaborn 与边界感知 MetricPlotSpec 绘制跨版本多规模（small/medium/large）演化趋势图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not records:
        return

    wide_df = records_to_wide_df(records)
    base_df = records_to_wide_df([base])
    specs = resolve_plot_specs(records)
    tidy_df = wide_to_tidy_df(wide_df, specs)

    release_labels = list(dict.fromkeys(wide_df.sort_values("release_idx")["label"]))
    present_scales = [s for s in ("small", "medium", "large") if s in set(wide_df["scale"].unique())]
    if not present_scales:
        present_scales = list(dict.fromkeys(wide_df["scale"]))
    has_multi_scale = len(present_scales) > 1

    total_panels = len(specs) + 2  # +2: Throughput Speedup vs Baseline & Correctness Gate
    ncols = 3
    nrows = math.ceil(total_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.4 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for spec_idx, spec in enumerate(specs):
        ax = flat_axes[spec_idx]
        sub = tidy_df[tidy_df["metric_key"] == spec.metric_key].sort_values("order_idx")
        if sub.empty:
            ax.set_axis_off()
            continue

        if spec.chart_kind == "bar":
            sns.barplot(
                data=sub,
                x="label",
                y="median",
                hue="scale" if has_multi_scale else None,
                order=release_labels,
                hue_order=present_scales if has_multi_scale else None,
                palette=_SCALE_PALETTE if has_multi_scale else None,
                color=None if has_multi_scale else ("#8e44ad" if spec.metric_key == "peak_allocated_mb" else "#16a085"),
                ax=ax,
                alpha=0.85,
            )
            for bar in ax.patches:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                    )
        else:
            for scale_idx, sc in enumerate(present_scales):
                sub_sc = sub[sub["scale"] == sc].sort_values("release_idx")
                if sub_sc.empty:
                    continue
                xs = [release_labels.index(lbl) for lbl in sub_sc["label"]]
                ys = sub_sc["median"].tolist()
                stds = sub_sc["std"].tolist()
                color = _SCALE_PALETTE.get(sc, "#2980b9") if has_multi_scale else "#2980b9"
                marker = _SCALE_MARKERS.get(sc, "o")
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    markersize=6.5,
                    linewidth=2.0,
                    color=color,
                    label=sc if has_multi_scale else None,
                )
                if any(s > 0 for s in stds):
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=stds,
                        fmt="none",
                        ecolor=color,
                        capsize=3,
                        alpha=0.55,
                    )
                y_offset = 5 + (scale_idx - 1) * 2 if has_multi_scale else 5
                for x_i, y_val in zip(xs, ys):
                    ax.annotate(
                        f"{y_val:.1f}",
                        xy=(x_i, y_val),
                        xytext=(0, y_offset),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color=color if has_multi_scale else "black",
                    )
            ax.set_xticks(range(len(release_labels)))
            ax.set_xticklabels(release_labels, rotation=15, ha="right")
            ax.margins(x=0.15, y=0.15)
            if has_multi_scale:
                ax.legend(title="Scale", fontsize=7.5, title_fontsize=8)

        ax.set_title(f"{spec_idx + 1}. {spec.title}", fontsize=10.5)
        ax.set_xlabel("")
        ax.set_ylabel(spec.ylabel)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, linestyle="--", alpha=0.45)

    # 附加面板 A：相对同 Scale Baseline 吞吐加速比
    ax_spd = flat_axes[len(specs)]
    default_b_thr = float(base_df.iloc[0]["throughput_tok_s.median"])
    base_thr_by_scale = {str(r["scale"]): float(r["throughput_tok_s.median"]) for _, r in base_df.iterrows()}
    wide_df["speedup_vs_base"] = [
        float(r["throughput_tok_s.median"]) / base_thr_by_scale.get(str(r["scale"]), default_b_thr)
        if base_thr_by_scale.get(str(r["scale"]), default_b_thr) > 0
        else 1.0
        for _, r in wide_df.iterrows()
    ]
    sns.barplot(
        data=wide_df,
        x="label",
        y="speedup_vs_base",
        hue="scale" if has_multi_scale else None,
        order=release_labels,
        hue_order=present_scales if has_multi_scale else None,
        palette=_SCALE_PALETTE if has_multi_scale else None,
        color=None if has_multi_scale else "#27ae60",
        ax=ax_spd,
        alpha=0.85,
    )
    ax_spd.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    for bar in ax_spd.patches:
        h = bar.get_height()
        if h > 0:
            ax_spd.annotate(
                f"{h:.2f}x",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax_spd.set_title(f"{len(specs) + 1}. Throughput Speedup vs Baseline (x)", fontsize=10.5)
    ax_spd.set_xlabel("")
    ax_spd.set_ylabel("Speedup Ratio (x)")
    ax_spd.margins(y=0.14)
    ax_spd.tick_params(axis="x", rotation=15)
    if has_multi_scale:
        ax_spd.legend(title="Scale", fontsize=7.5, title_fontsize=8)
    ax_spd.grid(True, linestyle="--", alpha=0.45)

    # 附加面板 B：按版本聚合的正确性门禁
    ax_corr = flat_axes[len(specs) + 1]
    rel_df = wide_df.drop_duplicates(subset=["release_idx"]).sort_values("release_idx")
    ratios = [
        0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        for v in rel_df["correctness_match_ratio"]
    ]
    colors = ["gray" if p is None else ("#2ecc71" if p else "#e74c3c") for p in rel_df["correctness_passed"]]
    bars = ax_corr.bar(range(len(rel_df)), ratios, color=colors, alpha=0.85, width=0.55)
    for i, (bar, p) in enumerate(zip(bars, rel_df["correctness_passed"])):
        text = "n/a" if p is None else f"{ratios[i]:.2f}"
        ax_corr.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax_corr.set_ylim(0, 1.15)
    ax_corr.set_xticks(range(len(rel_df)))
    ax_corr.set_xticklabels(rel_df["label"], rotation=15, ha="right")
    ax_corr.set_ylabel("Token Match Ratio")
    ax_corr.set_title(f"{len(specs) + 2}. Correctness Gate", fontsize=10.5)
    ax_corr.grid(True, linestyle="--", alpha=0.45)

    for extra_ax in flat_axes[total_panels:]:
        extra_ax.set_axis_off()

    fig.suptitle(
        "mini-infra-qwen3 Category 1: Release Evolution & Multi-Scale Analysis "
        f"(Baseline: {resolve_record_label(base)})",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"第一类图表（版本演进与多规模趋势）已保存：{out_path}")


def plot_profiling_and_optimization(records: list[Any], target_rec: Any, out_path: Path) -> None:
    """绘制第二大类图表：当前模型深度剖析与优化指引（2行2列面板，支持 Lockstep 与 Scheduler 双域）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validated_target = validate_benchmark_record(target_rec, require_single_case=True)
    w = extract_primary_case_dict(validated_target)
    is_scheduler = isinstance(validated_target, SchedulerBenchmarkRecord)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_ceil, ax_mem), (ax_mod, ax_cat) = axes

    # 2.1 硬件物理天花板与调度容量利用率
    gpu_busy = (w.get("gpu_time_ratio") or {}).get("median")
    if is_scheduler:
        pack = w.get("packing") or {}
        slot_u = (pack.get("slot_utilization") or {}).get("median", 0.0)
        cap_u = (pack.get("running_capacity_utilization") or {}).get("median", 0.0)
        pad_e = (pack.get("padding_efficiency") or {}).get("median", 0.0)
        cache_frac = ((w.get("schedule") or {}).get("cache_management_fraction") or {}).get("median", 0.0)
        ceil_items = [
            ("Slot Util", 100.0 * float(slot_u), "#2980b9"),
            ("Capacity Util", 100.0 * float(cap_u), "#8e44ad"),
            ("Padding Eff", 100.0 * float(pad_e), "#e67e22"),
            ("GPU Busy", 100.0 * float(gpu_busy if gpu_busy is not None else 0.0), "#27ae60"),
            ("Cache Mgmt", 100.0 * float(cache_frac), "#e74c3c"),
        ]
        title_21 = "2.1 Scheduler & GPU Utilization (Slot / Capacity / Padding / Busy / Cache Mgmt)"
    else:
        eff = w.get("efficiency") or {}
        ceil_items = [
            ("Prefill MFU", 100.0 * (eff.get("prefill_mfu") or 0.0), "#e67e22"),
            ("Decode MFU", 100.0 * (eff.get("decode_mfu") or 0.0), "#e74c3c"),
            ("Decode MBU", 100.0 * (eff.get("decode_mbu") or 0.0), "#2980b9"),
            ("GPU Busy", 100.0 * (gpu_busy if gpu_busy is not None else 0.0), "#27ae60"),
        ]
        title_21 = "2.1 Hardware Ceiling Utilization (MFU / MBU / Busy)"

    names = [k for k, _, _ in ceil_items]
    vals = [v for _, v, _ in ceil_items]
    cols = [c for _, _, c in ceil_items]
    bars = ax_ceil.bar(names, vals, color=cols, alpha=0.85, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax_ceil.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax_ceil.set_ylim(0, 108)
    ax_ceil.set_ylabel("Utilization (%)")
    ax_ceil.set_title(title_21)
    ax_ceil.grid(True, linestyle="--", alpha=0.45)

    # 2.2 显存足迹剖析（跨版本或当前版本堆叠图：同 tag/feature 不同 scale 紧靠成组，不同 tag/feature 分开）
    wide_df = records_to_wide_df(records) if records else records_to_wide_df([validated_target])
    has_multi_mem_scale = len(set(wide_df["scale"].unique())) > 1
    scale_short = {"small": "S", "medium": "M", "large": "L"}

    xs: list[float] = []
    group_ticks: list[float] = []
    group_labels: list[str] = []
    bar_scale_tags: list[str] = []
    intra_step = 0.24 if has_multi_mem_scale else 0.5
    bar_width = 0.22 if has_multi_mem_scale else 0.5

    for group_idx, (_, g_df) in enumerate(wide_df.groupby("release_idx", sort=False)):
        n_in_group = len(g_df)
        group_ticks.append(float(group_idx))
        group_labels.append(str(g_df.iloc[0]["label"]))
        for i, (_, r) in enumerate(g_df.iterrows()):
            offset = (i - (n_in_group - 1) / 2.0) * intra_step
            xs.append(float(group_idx) + offset)
            sc_name = str(r.get("scale") or "medium")
            bar_scale_tags.append(scale_short.get(sc_name, sc_name[:1].upper()))

    bottom = [0.0] * len(wide_df)
    wide_df["memory.weights_mb"] = wide_df["memory.weights_mb"].fillna(wide_df["memory.baseline_allocated_mb"])
    components = [
        ("memory.weights_mb", "Resident Weights & Base", "tab:gray"),
        ("memory.dynamic_peak_mb", "Dynamic Peak (Act/KV)", "tab:blue"),
        ("memory.fragmentation_mb", "Allocator Fragmentation", "tab:orange"),
    ]
    for col, name, color in components:
        vals = wide_df[col].fillna(0.0).tolist() if col in wide_df.columns else [0.0] * len(wide_df)
        ax_mem.bar(xs, vals, bottom=bottom, label=name, color=color, alpha=0.85, width=bar_width)
        bottom = [b + v for b, v in zip(bottom, vals)]
    if has_multi_mem_scale:
        for x_pos, top_y, s_tag in zip(xs, bottom, bar_scale_tags):
            if top_y > 0:
                ax_mem.annotate(
                    s_tag,
                    xy=(x_pos, top_y),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                )
        ax_mem.margins(y=0.12)
    ax_mem.set_xticks(group_ticks)
    ax_mem.set_xticklabels(group_labels, rotation=15, ha="right", fontsize=8.5)
    ax_mem.set_ylabel("Memory (MiB)")
    ax_mem.set_title(
        "2.2 Memory Footprint Breakdown (S / M / L per Release)"
        if has_multi_mem_scale
        else "2.2 Memory Footprint Breakdown"
    )
    ax_mem.legend(fontsize=8)
    ax_mem.grid(True, linestyle="--", alpha=0.45)

    # 2.3 & 2.4：Scheduler 迭代轨迹与分位数剖析 vs Lockstep 算子剖析
    if is_scheduler:
        events = w.get("sample_event_trace") or []
        if events:
            step_ids = list(range(1, len(events) + 1))
            step_ms = [float(e.get("step_ms", 0.0)) for e in events]
            active_reqs = [int(e.get("num_active_requests", 0)) for e in events]
            step_colors = ["#e67e22" if e.get("phase") == "prefill" else "#2980b9" for e in events]
            ax_mod.bar(step_ids, step_ms, color=step_colors, alpha=0.8, width=0.8, label="Step Latency (ms)")
            ax_mod.set_xlabel("Scheduler Iteration Step (Orange=Prefill, Blue=Decode)")
            ax_mod.set_ylabel("Step Latency (ms)", color="#2980b9")
            ax_mod.set_title(f"2.3 Iteration Step Latency & Active Slots ({len(events)} steps)")
            ax_mod.grid(True, linestyle="--", alpha=0.35)

            ax_mod_r = ax_mod.twinx()
            ax_mod_r.plot(step_ids, active_reqs, color="#27ae60", linewidth=1.8, label="Active Slots")
            ax_mod_r.set_ylabel("Active Running Requests", color="#27ae60")
            max_cap = int((w.get("schedule") or {}).get("max_running_requests", max(active_reqs, default=4)))
            ax_mod_r.set_ylim(0, max(max_cap + 1, 5))
        else:
            ax_mod.text(0.5, 0.5, "No sample event trace", ha="center", va="center", transform=ax_mod.transAxes)
            ax_mod.set_axis_off()

        sched = w.get("schedule") or {}
        lat_categories = ["Queue Delay", "TTFT", "ITL (per tok)", "End-to-End"]
        p50_vals = [
            float((sched.get("queue_ms_p50") or {}).get("median", 0.0)),
            float((sched.get("ttft_ms_p50") or {}).get("median", 0.0)),
            float((sched.get("itl_ms_p50") or {}).get("median", 0.0)),
            float((sched.get("end_to_end_ms_p50") or {}).get("median", 0.0)),
        ]
        p95_vals = [
            float((sched.get("queue_ms_p95") or {}).get("median", 0.0)),
            float((sched.get("ttft_ms_p95") or {}).get("median", 0.0)),
            float((sched.get("itl_ms_p95") or {}).get("median", 0.0)),
            float((sched.get("end_to_end_ms_p95") or {}).get("median", 0.0)),
        ]
        x_ticks = list(range(len(lat_categories)))
        bar_w = 0.36
        bars_50 = ax_cat.bar([x - bar_w / 2 for x in x_ticks], p50_vals, width=bar_w, label="P50", color="#3498db")
        bars_95 = ax_cat.bar([x + bar_w / 2 for x in x_ticks], p95_vals, width=bar_w, label="P95", color="#e74c3c")
        for bar in (*bars_50, *bars_95):
            h = bar.get_height()
            ax_cat.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax_cat.set_xticks(x_ticks)
        ax_cat.set_xticklabels(lat_categories)
        ax_cat.set_ylabel("Latency (ms)")
        ax_cat.set_title("2.4 Request Latency Percentiles (P50 vs P95)")
        ax_cat.legend(fontsize=8.5)
        ax_cat.grid(True, linestyle="--", alpha=0.45)
    else:
        ob = w.get("op_latency_breakdown")
        if isinstance(ob, dict) and ob.get("time_share_by_module"):
            mod_items = list(ob.get("time_share_by_module", {}).items())[::-1]
            bars_m = ax_mod.barh(
                [k for k, _ in mod_items], [100 * v for _, v in mod_items], color="tab:cyan", alpha=0.85
            )
            for bar in bars_m:
                w_val = bar.get_width()
                ax_mod.annotate(
                    f"{w_val:.1f}%",
                    (w_val, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
            ax_mod.set_xlabel("Share of GPU Time (%)")
            ax_mod.set_title(f"2.3 Module Attribution (prompt={w['prompt_len']})")
            ax_mod.grid(alpha=0.3)

            cat_items = list(ob.get("time_share_by_kernel_category", {}).items())[::-1]
            bars_c = ax_cat.barh(
                [k for k, _ in cat_items], [100 * v for _, v in cat_items], color="tab:olive", alpha=0.85
            )
            for bar in bars_c:
                w_val = bar.get_width()
                ax_cat.annotate(
                    f"{w_val:.1f}%",
                    (w_val, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
            ax_cat.set_xlabel("Share of GPU Time (%)")
            ax_cat.set_title(f"2.4 Kernel Category (prompt={w['prompt_len']})")
            ax_cat.grid(alpha=0.3)
        else:
            for ax in (ax_mod, ax_cat):
                ax.text(
                    0.5,
                    0.5,
                    "No op latency breakdown profile\n(rerun with --profiling-mode dedicated-pass)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()

    fig.suptitle(
        "mini-infra-qwen3 Category 2: Current Model Profiling & Optimization Guidance "
        f"({resolve_record_label(validated_target)})",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"第二类图表（当前模型剖析与优化指引）已保存：{out_path}")


__all__ = [
    "load_results",
    "plot_evolution_and_scaling",
    "plot_profiling_and_optimization",
]
