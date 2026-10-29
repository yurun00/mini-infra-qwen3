"""读取 benchmarks/result-history/*.json（严格校验每个文件只包含 1 个标杆 Case），生成 Markdown 汇总表与两大类分析图表：

第一大类：跨版本演化趋势 (Evolution Across Releases)
    自动通过 resolve_plot_specs(records) 感知历史记录是否跨越 lockstep <-> scheduler 边界：
    - 未跨边界（纯 Lockstep 或纯 Scheduler）：绘制共享指标 + 当前域专属指标
      （如 Lockstep 的 tpot_ms，或 Scheduler 的 itl_ms_p95 / queue_ms_p95 / makespan_ms）
    - 跨越边界（含 Lockstep 与 Scheduler）：仅绘制跨边界共享指标（ttft_ms 自动对齐至 schedule.ttft_ms_p50）

第二大类：当前模型深度剖析与优化指引 (Profiling & Optimization Guidance)
    基于 target 版本的单 Case 数据展示 MBU/MFU、模块归因、算子类别与显存足迹。
"""

import math
from pathlib import Path
from typing import Any

from mini_infra.bench.common import RESULT_HISTORY_DIR
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
    """加载 result-history/*.json 并强制校验每个文件只包含 1 个 Case (require_single_case=True)。"""
    target_dir = history_dir or RESULT_HISTORY_DIR
    records = [load_benchmark_record(p, require_single_case=True) for p in target_dir.glob("*.json")]
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
                "| E2E (ms) | 吞吐 (tok/s) | 显存 (MiB) | 槽位利用率 | 正确性 |",
                "| :--- | :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
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
            e2e = float(row.get("e2e_wall_ms.median", 0.0))
            s_e2e = f" ({b_e2e / e2e:.2f}x)" if b_e2e > 0 and e2e > 0 else ""
            lines.append(
                f"| {row['label']} | `{row['benchmark_type']}` | {row['commit']} | {row['batch_size']} | "
                f"{row['prompt_len_str']} | {row['max_new_tokens_str']} | {ttft:.2f}{s_ttft} | "
                f"{e2e:.2f}{s_e2e} | {thr:.1f}{s_thr} | {mem:.0f} | {slot:.1f}% | {flag} |"
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
    """绘制第二大类图表：当前模型深度剖析与优化指引（2行2列面板）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validated_target = validate_benchmark_record(target_rec, require_single_case=True)
    w = validated_target.cases[0].to_dict()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_ceil, ax_mem), (ax_mod, ax_cat) = axes

    # 2.1 硬件物理天花板差距（MBU / MFU / GPU Busy）
    eff = w.get("efficiency") or {}
    gpu_busy = (w.get("gpu_time_ratio") or {}).get("median")
    ceil_items = [
        ("Prefill MFU", 100.0 * (eff.get("prefill_mfu") or 0.0), "#e67e22"),
        ("Decode MFU", 100.0 * (eff.get("decode_mfu") or 0.0), "#e74c3c"),
        ("Decode MBU", 100.0 * (eff.get("decode_mbu") or 0.0), "#2980b9"),
        ("GPU Busy", 100.0 * (gpu_busy if gpu_busy is not None else 0.0), "#27ae60"),
    ]
    if eff or gpu_busy is not None:
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
        ax_ceil.set_title("2.1 Hardware Ceiling Utilization (MFU / MBU / Busy)")
        ax_ceil.grid(True, linestyle="--", alpha=0.45)
    else:
        ax_ceil.text(0.5, 0.5, "No efficiency metrics", ha="center", va="center", transform=ax_ceil.transAxes)
        ax_ceil.set_axis_off()

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

    # 2.3 & 2.4 模块与算子类型透视
    ob = w.get("op_latency_breakdown")
    if isinstance(ob, dict) and ob.get("time_share_by_module"):
        mod_items = list(ob.get("time_share_by_module", {}).items())[::-1]
        bars_m = ax_mod.barh([k for k, _ in mod_items], [100 * v for _, v in mod_items], color="tab:cyan", alpha=0.85)
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
        bars_c = ax_cat.barh([k for k, _ in cat_items], [100 * v for _, v in cat_items], color="tab:olive", alpha=0.85)
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
    """根据当前版本的单 Case profiling 数据，自动提炼诊断结论与下一步优化建议。"""
    validated_target = validate_benchmark_record(target_rec, require_single_case=True)
    w = validated_target.cases[0].to_dict()

    lines = [
        "## 二、当前版本深度剖析与优化指引 (Profiling & Next Steps)",
        "",
        "![Profiling](./profiling_and_optimization.png)",
        "",
    ]

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
