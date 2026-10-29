"""读取 benchmarks/result-history/*.json（严格校验每个文件只包含 1 个标杆 Case），生成 Markdown 汇总表与两大类分析图表：

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

from mini_infra.bench.common import (
    LOCKSTEP_RESULT_HISTORY_DIR,
    RESULT_HISTORY_DIR,
)
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
    validate_benchmark_record,
)


def load_results(history_dir: Path | None = None) -> list[LockstepBenchmarkRecord | SchedulerBenchmarkRecord]:
    """加载指定历史目录中的 *.json 并强制校验每个文件只包含 1 个 Case (require_single_case=True)。"""
    target_dir = history_dir or LOCKSTEP_RESULT_HISTORY_DIR
    json_files = list(target_dir.glob("*.json")) if target_dir.exists() else []
    if not json_files and history_dir is None and RESULT_HISTORY_DIR.exists():
        json_files = list(RESULT_HISTORY_DIR.glob("*.json"))
    records = [load_benchmark_record(p, require_single_case=True) for p in json_files]
    records.sort(key=lambda r: r.timestamp)
    return records


def label(record: Any) -> str:
    """提取版本演进展示标签。"""
    return resolve_record_label(record)


def markdown_table(records: list[Any], base: Any) -> str:
    """基于 Pandas 宽表与边界感知规则生成跨版本演进 Markdown 汇总表。"""
    wide_df = records_to_wide_df(records)
    base_df = records_to_wide_df([base])
    b_row = base_df.iloc[0]

    btypes = set(wide_df["benchmark_type"].unique())
    is_pure_lockstep = btypes == {"lockstep"}
    is_pure_scheduler = btypes == {"scheduler"}

    lines = [
        f"基准版本：`{b_row['label']}` (`{b_row['commit']}`)，GPU：{b_row['gpu']}，"
        "数值为 median，括号内为相对基准的变化比（>1 表示更快/更高）。",
        "",
    ]

    if is_pure_lockstep:
        lines.extend(
            [
                "| 版本 | commit | batch | prompt | max_new_tokens | TTFT (ms) "
                "| TPOT (ms) | 吞吐 (tok/s) | 显存 (MiB) | 槽位利用率 | 正确性 |",
                "| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
            ]
        )
    elif is_pure_scheduler:
        lines.extend(
            [
                "| 版本 | commit | batch | prompt | max_new_tokens | Makespan (ms) "
                "| TTFT P50 (ms) | ITL P95 (ms) | 吞吐 (tok/s) | 显存 (MiB) | 槽位利用率 | 正确性 |",
                "| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
            ]
        )
    else:
        lines.extend(
            [
                "| 版本 | 类型 | commit | batch | prompt | max_new_tokens | TTFT (ms) "
                "| TPOT / ITL P50 (ms) | E2E (ms) | 吞吐 (tok/s) | 显存 (MiB) | 槽位利用率 | 正确性 |",
                "| :--- | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
            ]
        )

    b_ttft = float(b_row.get("ttft_ms.median", 0.0))
    b_thr = float(b_row.get("throughput_tok_s.median", 0.0))
    b_e2e = float(b_row.get("e2e_wall_ms.median", 0.0))
    b_tpot = float(b_row.get("tpot_ms.median", 0.0)) if "tpot_ms.median" in b_row else 0.0
    b_mksp = float(b_row.get("makespan_ms.median", 0.0)) if "makespan_ms.median" in b_row else 0.0

    for _, row in wide_df.iterrows():
        flag = {True: "✅", False: "❌", None: "—"}.get(row["correctness_passed"], "—")
        ttft = float(row.get("ttft_ms.median", 0.0))
        thr = float(row.get("throughput_tok_s.median", 0.0))
        mem = float(row.get("peak_allocated_mb.median", 0.0))
        slot = float(row.get("slot_utilization.median", 1.0)) * 100.0

        s_ttft = f" ({b_ttft / ttft:.2f}x)" if b_ttft > 0 and ttft > 0 else ""
        s_thr = f" ({thr / b_thr:.2f}x)" if b_thr > 0 else ""

        if is_pure_lockstep:
            tpot = float(row.get("tpot_ms.median", 0.0))
            s_tpot = f" ({b_tpot / tpot:.2f}x)" if b_tpot > 0 and tpot > 0 else ""
            lines.append(
                f"| {row['label']} | {row['commit']} | {row['batch_size']} | {row['prompt_len_str']} | "
                f"{row['max_new_tokens_str']} | {ttft:.2f}{s_ttft} | {tpot:.2f}{s_tpot} | "
                f"{thr:.1f}{s_thr} | {mem:.0f} | {slot:.1f}% | {flag} |"
            )
        elif is_pure_scheduler:
            mksp = float(row.get("makespan_ms.median", 0.0))
            itl_p95 = float(row.get("itl_ms_p95.median", 0.0))
            s_mksp = f" ({b_mksp / mksp:.2f}x)" if b_mksp > 0 and mksp > 0 else ""
            lines.append(
                f"| {row['label']} | {row['commit']} | {row['batch_size']} | {row['prompt_len_str']} | "
                f"{row['max_new_tokens_str']} | {mksp:.2f}{s_mksp} | {ttft:.2f}{s_ttft} | {itl_p95:.2f} | "
                f"{thr:.1f}{s_thr} | {mem:.0f} | {slot:.1f}% | {flag} |"
            )
        else:
            tpot = float(row.get("tpot_ms.median", 0.0))
            s_tpot = f" ({b_tpot / tpot:.2f}x)" if b_tpot > 0 and tpot > 0 else ""
            e2e = float(row.get("e2e_wall_ms.median", 0.0))
            s_e2e = f" ({b_e2e / e2e:.2f}x)" if b_e2e > 0 and e2e > 0 else ""
            lines.append(
                f"| {row['label']} | `{row['benchmark_type']}` | {row['commit']} | {row['batch_size']} | "
                f"{row['prompt_len_str']} | {row['max_new_tokens_str']} | {ttft:.2f}{s_ttft} | "
                f"{tpot:.2f}{s_tpot} | {e2e:.2f}{s_e2e} | {thr:.1f}{s_thr} | {mem:.0f} | {slot:.1f}% | {flag} |"
            )

    return "\n".join(lines)


def plot_evolution_and_scaling(records: list[Any], base: Any, out_path: Path) -> None:
    """使用 Seaborn 与边界感知 MetricPlotSpec 绘制跨版本演化趋势图。"""
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

    total_panels = len(specs) + 2  # +2: Throughput Speedup vs Baseline & Correctness Gate
    ncols = 3
    nrows = math.ceil(total_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.3 * nrows), squeeze=False)
    flat_axes = axes.flatten()

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
                color="#8e44ad" if spec.metric_key == "peak_allocated_mb" else "#16a085",
                ax=ax,
                alpha=0.85,
                width=0.55,
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
                        fontsize=8,
                    )
        else:
            sns.lineplot(
                data=sub,
                x="label",
                y="median",
                marker="o",
                markersize=7,
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
                    capsize=3,
                    alpha=0.65,
                )
            for x_i, (_, row) in enumerate(sub.iterrows()):
                ax.annotate(
                    f"{row['median']:.1f}",
                    xy=(x_i, row["median"]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(f"{idx + 1}. {spec.title}", fontsize=10.5)
        ax.set_xlabel("")
        ax.set_ylabel(spec.ylabel)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, linestyle="--", alpha=0.45)

    # 附加面板 A：相对 Baseline 吞吐加速比
    ax_spd = flat_axes[len(specs)]
    b_thr = float(base_df.iloc[0]["throughput_tok_s.median"])
    wide_df["speedup_vs_base"] = wide_df["throughput_tok_s.median"].apply(lambda v: v / b_thr if b_thr > 0 else 1.0)
    sns.barplot(data=wide_df, x="label", y="speedup_vs_base", color="#27ae60", ax=ax_spd, alpha=0.85, width=0.55)
    ax_spd.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    for bar in ax_spd.patches:
        h = bar.get_height()
        ax_spd.annotate(
            f"{h:.2f}x",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_spd.set_title(f"{len(specs) + 1}. Throughput Speedup vs Baseline (x)", fontsize=10.5)
    ax_spd.set_xlabel("")
    ax_spd.set_ylabel("Speedup Ratio (x)")
    ax_spd.tick_params(axis="x", rotation=15)
    ax_spd.grid(True, linestyle="--", alpha=0.45)

    # 附加面板 B：正确性门禁
    ax_corr = flat_axes[len(specs) + 1]
    ratios = [
        0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        for v in wide_df["correctness_match_ratio"]
    ]
    colors = ["gray" if p is None else ("#2ecc71" if p else "#e74c3c") for p in wide_df["correctness_passed"]]
    bars = ax_corr.bar(range(len(wide_df)), ratios, color=colors, alpha=0.85, width=0.55)
    for i, (bar, p) in enumerate(zip(bars, wide_df["correctness_passed"])):
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
    ax_corr.set_xticks(range(len(wide_df)))
    ax_corr.set_xticklabels(wide_df["label"], rotation=15, ha="right")
    ax_corr.set_ylabel("Token Match Ratio")
    ax_corr.set_title(f"{len(specs) + 2}. Correctness Gate", fontsize=10.5)
    ax_corr.grid(True, linestyle="--", alpha=0.45)

    for extra_ax in flat_axes[total_panels:]:
        extra_ax.set_axis_off()

    fig.suptitle(
        f"mini-infra-qwen3 Category 1: Release Evolution Analysis (Baseline: {label(base)})",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"第一类图表（版本演进趋势）已保存：{out_path}")


def plot_profiling_and_optimization(records: list[Any], target_rec: Any, out_path: Path) -> None:
    """绘制第二大类图表：当前模型深度剖析与优化指引（2行2列面板，支持 Lockstep 与 Scheduler 双域）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validated_target = validate_benchmark_record(target_rec, require_single_case=True)
    w = validated_target.cases[0].to_dict()
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
        ceil_items = [
            ("Slot Util", 100.0 * float(slot_u), "#2980b9"),
            ("Capacity Util", 100.0 * float(cap_u), "#8e44ad"),
            ("Padding Eff", 100.0 * float(pad_e), "#e67e22"),
            ("GPU Busy", 100.0 * float(gpu_busy if gpu_busy is not None else 0.0), "#27ae60"),
        ]
        title_21 = "2.1 Scheduler & GPU Utilization (Slot / Capacity / Padding / Busy)"
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

    # 2.2 显存足迹剖析（跨版本或当前版本堆叠图）
    wide_df = records_to_wide_df(records) if records else records_to_wide_df([validated_target])
    xs = list(range(len(wide_df)))
    bottom = [0.0] * len(wide_df)
    components = [
        ("memory.weight_mb", "Resident Weights & Base", "tab:gray"),
        ("memory.dynamic_peak_mb", "Dynamic Peak (Act/KV)", "tab:blue"),
        ("memory.fragmentation_mb", "Allocator Fragmentation", "tab:orange"),
    ]
    for col, name, color in components:
        vals = wide_df[col].fillna(0.0).tolist() if col in wide_df.columns else [0.0] * len(wide_df)
        ax_mem.bar(xs, vals, bottom=bottom, label=name, color=color, alpha=0.85, width=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax_mem.set_xticks(xs)
    ax_mem.set_xticklabels(wide_df["label"], rotation=15, ha="right")
    ax_mem.set_ylabel("Memory (MiB)")
    ax_mem.set_title("2.2 Memory Footprint Breakdown")
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
        x_idx = list(range(len(lat_categories)))
        bar_w = 0.36
        bars_50 = ax_cat.bar([x - bar_w / 2 for x in x_idx], p50_vals, width=bar_w, label="P50", color="#3498db")
        bars_95 = ax_cat.bar([x + bar_w / 2 for x in x_idx], p95_vals, width=bar_w, label="P95", color="#e74c3c")
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
        ax_cat.set_xticks(x_idx)
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
                    "No op latency breakdown profile\n(rerun with --profile-mode extra-profile)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()

    fig.suptitle(
        f"mini-infra-qwen3 Category 2: Current Model Profiling & Optimization Guidance ({label(validated_target)})",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"第二类图表（当前模型剖析与优化指引）已保存：{out_path}")


def generate_optimization_guidance(target_rec: Any) -> str:
    """根据当前版本的单 Case profiling 数据，自动提炼诊断结论与下一步优化建议（支持 Lockstep 与 Scheduler）。"""
    validated_target = validate_benchmark_record(target_rec, require_single_case=True)
    w = validated_target.cases[0].to_dict()

    lines = [
        "## 二、当前版本深度剖析与优化指引 (Profiling & Next Steps)",
        "",
        "![Profiling](./profiling_and_optimization.png)",
        "",
    ]

    if isinstance(validated_target, SchedulerBenchmarkRecord):
        sched = w.get("schedule") or {}
        pack = w.get("packing") or {}
        mem = w.get("memory") or {}
        slot_u = float((pack.get("slot_utilization") or {}).get("median", 0.0))
        cap_u = float((pack.get("running_capacity_utilization") or {}).get("median", 0.0))
        pad_e = float((pack.get("padding_efficiency") or {}).get("median", 0.0))
        q_p50 = float((sched.get("queue_ms_p50") or {}).get("median", 0.0))
        q_p95 = float((sched.get("queue_ms_p95") or {}).get("median", 0.0))
        ttft_p50 = float((sched.get("ttft_ms_p50") or {}).get("median", 0.0))
        ttft_p95 = float((sched.get("ttft_ms_p95") or {}).get("median", 0.0))
        itl_p50 = float((sched.get("itl_ms_p50") or {}).get("median", 0.0))
        itl_p95 = float((sched.get("itl_ms_p95") or {}).get("median", 0.0))
        itl_max = float((sched.get("itl_ms_max") or {}).get("median", 0.0))
        cache_frac = float((sched.get("cache_management_fraction") or {}).get("median", 0.0))
        kv_mb = float(mem.get("kv_cache_mb") or 0.0)
        dyn_mb = float(mem.get("dynamic_peak_mb") or 0.0)
        frag_mb = float(mem.get("fragmentation_mb") or 0.0)

        lines.append(
            f"基于当前调度器版本 `{label(validated_target)}`（`requests={w['batch_size']}, "
            f"max_running={sched.get('max_running_requests', 4)}`）下的基准测量与轨迹剖析，诊断如下：\n"
        )
        lines.append("### 1. 槽位与并发容量利用率诊断（调度效率）")
        lines.append(
            f"- 🔹 **执行槽位有效率（Slot Utilization）为 {slot_u * 100:.1f}%**，"
            f"**并发容量利用率（Running Capacity Utilization）为 {cap_u * 100:.1f}%**，"
            f"**Prompt Padding 有效率为 {pad_e * 100:.1f}%**。"
        )
        if pad_e < 0.75:
            lines.append(
                "- 🟡 **变长 Prompt 左填充浪费明显**：建议后续演进引入 "
                "**FlashAttention-Varlen / Ragged Prefill** 消除无效 Padding 计算。"
            )
        lines.append("")

        lines.append("### 2. 请求排队与尾延迟诊断（P50 vs P95）")
        lines.append(
            f"- 🔹 **排队延迟**：P50 = `{q_p50:.1f} ms`，P95 = `{q_p95:.1f} ms`；"
            f"**TTFT**：P50 = `{ttft_p50:.1f} ms`，P95 = `{ttft_p95:.1f} ms`。"
        )
        lines.append(
            f"- 🔹 **Decode 步间延迟（ITL）**：P50 = `{itl_p50:.2f} ms`，"
            f"P95 = `{itl_p95:.2f} ms`，Max = `{itl_max:.2f} ms`。"
        )
        if itl_max > itl_p50 * 1.4:
            lines.append(
                "- 🟡 **观测到新请求 Prefill 插队导致的 Decode 尾延迟毛刺（ITL Max > 1.4x P50）**："
                "建议下一步演进引入 **Token Budget 调度器与 Chunked Prefill**，"
                "将长 Prompt 拆分为小块与 Decode 交织执行。"
            )
        lines.append("")

        lines.append("### 3. 显存足迹与 KV Cache 管理诊断")
        lines.append(
            f"- 🔹 **峰值活跃 KV Cache 为 {kv_mb:.1f} MiB**，动态显存峰值增量为 **{dyn_mb:.1f} MiB**，"
            f"Allocator 预留碎片为 **{frag_mb:.1f} MiB**，Cache 搬移开销占比为 **{cache_frac * 100:.2f}%**。"
        )
        lines.append(
            "- 👉 **下一步核心演进方向**：引入 **Paged KV Cache（分页键值缓存）** 与 **Prefix Cache**，"
            "按固定大小物理 Block 按需分配与复用，彻底消除按最长序列预分配的显存碎片与稠密槽位搬移。"
        )
        lines.append("")
        return "\n".join(lines)

    ob = w.get("op_latency_breakdown")
    if not (isinstance(ob, dict) and ob.get("time_share_by_module")):
        lines.append(
            f"> 提示：当前版本 `{label(validated_target)}` 中未包含详细的算子剖析采样数据。"
            "使用 `mini-infra bench lockstep case --profile-mode extra-profile` 重跑可获取完整算子诊断。"
        )
        return "\n".join(lines)

    by_mod = ob.get("time_share_by_module", {})
    by_cat = ob.get("time_share_by_kernel_category", {})
    eff = w.get("efficiency") or {}
    mem = w.get("memory") or {}
    busy = (w.get("gpu_time_ratio") or {}).get("median", 1.0)

    prompt_desc = f"prompt={w['prompt_len']}"
    lines.append(
        f"基于当前版本 `{label(validated_target)}`（`{prompt_desc}`）下的基准测量与 profiling 结果，诊断如下：\n"
    )

    attn_share = by_mod.get("attention_block", 0.0)
    norm_share = by_mod.get("norm", 0.0)
    mlp_share = by_mod.get("mlp_block", 0.0)

    lines.append("### 1. 架构模块瓶颈诊断（改哪里？）")
    if attn_share >= 0.25:
        lines.append(
            f"- 🔴 **Attention 模块耗时占比最高（{attn_share * 100:.1f}%）**：\n"
            "  - 👉 **优先优化行动**：引入 KV Cache / FlashAttention 算子融合。"
        )
    if norm_share > 0.08:
        lines.append(f"- 🟡 **RMSNorm 调用开销显著（耗时占比 {norm_share * 100:.1f}%）**：建议实施 Fused RMSNorm。")
    if mlp_share > 0.15:
        lines.append(
            f"- 🟡 **MLP Block 占比偏高（耗时占比 {mlp_share * 100:.1f}%）**：建议针对 Gate/Up 实施 SwiGLU 融合。"
        )
    if not (attn_share >= 0.25 or norm_share > 0.08 or mlp_share > 0.15):
        top_mod = max(by_mod.items(), key=lambda kv: kv[1])[0] if by_mod else "unknown"
        lines.append(f"- 🔹 当前耗时最高的架构模块为 `{top_mod}`（占比 {by_mod.get(top_mod, 0) * 100:.1f}%）。")
    lines.append("")

    gemm_share = by_cat.get("gemm", 0.0)
    elem_mem_share = by_cat.get("elementwise", 0.0) + by_cat.get("memory", 0.0)
    lines.append("### 2. 算子类型与访存诊断（怎么改？）")
    lines.append(
        f"- 🔹 **GEMM 矩阵乘占 {gemm_share * 100:.1f}%**，**Elementwise + Memory 搬运占 {elem_mem_share * 100:.1f}%**。"
    )
    lines.append("")

    mbu = eff.get("decode_mbu")
    lines.append("### 3. 硬件物理天花板差距")
    if mbu is not None:
        lines.append(f"- 🔹 **Decode MBU（显存带宽利用率）为 {mbu * 100:.1f}%**。")
    if busy < 0.90:
        lines.append(f"- 🔹 **GPU 忙碌比例为 {busy * 100:.1f}%**：建议引入 CUDA Graph 消除 CPU 发射开销。")
    else:
        lines.append(f"- 🔹 **GPU 忙碌比例达到 {busy * 100:.1f}%**：瓶颈主要在 GPU 端计算与访存。")
    lines.append("")

    frag = mem.get("fragmentation_mb", 0.0)
    dyn = mem.get("dynamic_peak_mb", 0.0)
    lines.append("### 4. 显存足迹与碎片诊断")
    lines.append(f"- 🔹 纯动态增量为 **{dyn:.1f} MiB**，Allocator 碎片与预留为 **{frag:.1f} MiB**。")
    lines.append("")

    top_kernels = ob.get("top_kernels", [])
    if top_kernels:
        lines.append("### 5. 当前 Top 8 热点 Kernel 明细")
        lines.append("| 序号 | Kernel 名称 | 类别 | CUDA 耗时 (ms) | 耗时占比 | 调用次数 |")
        lines.append("| :-: | :--- | :-: | :-: | :-: | :-: |")
        for idx, k in enumerate(top_kernels[:8], 1):
            k_name = k["name"][:40]
            lines.append(
                f"| {idx} | `{k_name}` | {k['category']} | {k['cuda_ms']:.2f} | "
                f"{k['share'] * 100:.1f}% | {k['calls']} |"
            )
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "generate_optimization_guidance",
    "label",
    "load_results",
    "markdown_table",
    "plot_evolution_and_scaling",
    "plot_profiling_and_optimization",
]
