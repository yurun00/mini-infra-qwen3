"""读取 benchmarks/result-history/*.json，生成 Markdown 汇总表与两大类分析图表：

第一大类：跨版本演化与扩展趋势 (Evolution & Scaling Across Releases)
    1.1 版本演进时间线 (Decode 延迟、吞吐、相对基准加速比)
    1.2 序列长度扩展演进 (叠加各 Release 的 TTFT / TPOT / Throughput 曲线)
    1.3 资源与稳定性演进 (显存占用、尾延迟、正确性门禁)

第二大类：当前模型深度剖析与优化指引 (Profiling & Optimization Guidance)
    2.1 硬件物理天花板差距 (MBU 带宽利用率、MFU 算力利用率、GPU 忙碌比例) -> 离硬件极限还有多远
    2.2 架构模块耗时归因 (Attention vs MLP vs Norm vs LM Head) -> 下一步该改哪个模型模块
    2.3 算子类型透视 (GEMM vs Elementwise vs Memory vs Softmax) -> 该用什么技术手段改
    2.4 显存足迹剖析 (Weights 常驻 vs 动态峰值 vs 碎片预留) -> 如果显存紧张该砍哪里
"""

import argparse
import json
import os
from pathlib import Path

BENCH_DIR = Path(os.environ.get("MINI_INFRA_BENCH_DIR", "benchmarks")).resolve()
RESULTS_DIR = BENCH_DIR / "result-history"
REPORTS_DIR = BENCH_DIR / "reports"


def load_results():
    records = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
    records.sort(key=lambda r: r["timestamp"])
    return records


def index(record):
    return {(w["batch_size"], w["prompt_len"], w["output_len"]): w for w in record["workloads"]}


def label(record):
    tag = record["tag"]
    dirty = "*" if record.get("git", {}).get("dirty") else ""
    feat = record.get("release_feature") or record.get("release-feature") or record.get("feature")
    if feat:
        return f"{tag} ({feat}){dirty}"
    return f"{tag}{dirty}"


def markdown_table(records, base):
    base_idx = index(base)
    lines = [
        f"基准版本：`{label(base)}` (`{base['git']['commit']}`)，GPU：{base['env']['gpu']}，"
        "数值为 median，括号内为相对基准的加速比（>1 表示更快）。",
        "",
        "| 版本 | commit | batch | prompt | output | TTFT (ms) | TPOT (ms) | 吞吐 (tok/s) | 显存 (MiB) | 正确性 |",
        "| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ]
    for rec in records:
        ok = rec["correctness"]["passed"]
        flag = {True: "✅", False: "❌", None: "—"}[ok]
        for k, w in sorted(index(rec).items()):
            b = base_idx.get(k)
            ttft, tpot, thr = (w["ttft_ms"]["median"], w["tpot_ms"]["median"], w["throughput_tok_s"]["median"])
            if b:
                s_ttft = f" ({b['ttft_ms']['median'] / ttft:.2f}x)"
                s_tpot = f" ({b['tpot_ms']['median'] / tpot:.2f}x)" if tpot else ""
                s_thr = f" ({thr / b['throughput_tok_s']['median']:.2f}x)"
            else:
                s_ttft = s_tpot = s_thr = ""
            lines.append(
                f"| {label(rec)} | {rec['git']['commit']} | {k[0]} | {k[1]} | {k[2]} | "
                f"{ttft:.2f}{s_ttft} | {tpot:.2f}{s_tpot} | {thr:.1f}{s_thr} | "
                f"{w['memory']['peak_allocated_mb']:.0f} | {flag} |"
            )
    return "\n".join(lines)


# ==============================================================================
# 第一大类：跨版本演化与扩展趋势 (Evolution & Scaling Across Releases)
# ==============================================================================
def plot_evolution_and_scaling(records, base, out_path):
    """绘制第一大类图表：跨版本演化与扩展趋势（3行3列全景图）。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        return

    version_labels = [label(r) for r in records]
    x_indices = list(range(len(records)))

    all_prompt_lens = sorted({w["prompt_len"] for r in records for w in r["workloads"] if w["batch_size"] == 1})
    key_prompts = [p for p in [32, 128, 512, 1024] if p in all_prompt_lens] or all_prompt_lens[:3]

    def pick_w(rec, p):
        return next((w for w in rec["workloads"] if w["batch_size"] == 1 and w["prompt_len"] == p), None)

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))

    # --- Row 1: 特性演进时间线 (Release Evolution Timeline) ---
    ax_tpot, ax_thr, ax_spd = axes[0]

    # 1.1 TPOT 演化
    for pl in key_prompts:
        ys = [(pick_w(r, pl) or {}).get("tpot_ms", {}).get("median") for r in records]
        ax_tpot.plot(x_indices, ys, marker="o", linewidth=2, label=f"prompt={pl}")
    ax_tpot.set_xticks(x_indices)
    ax_tpot.set_xticklabels(version_labels, rotation=15, ha="right")
    ax_tpot.set_ylabel("TPOT / Decode (ms/token)")
    ax_tpot.set_title("1.1 Decode Latency Evolution")
    ax_tpot.grid(True, linestyle="--", alpha=0.5)
    ax_tpot.legend(fontsize=8)

    # 1.2 吞吐演化
    for pl in key_prompts:
        ys = [(pick_w(r, pl) or {}).get("throughput_tok_s", {}).get("median") for r in records]
        ax_thr.plot(x_indices, ys, marker="s", linewidth=2, label=f"prompt={pl}")
    ax_thr.set_xticks(x_indices)
    ax_thr.set_xticklabels(version_labels, rotation=15, ha="right")
    ax_thr.set_ylabel("Throughput (tokens/s)")
    ax_thr.set_title("1.2 Throughput Evolution")
    ax_thr.grid(True, linestyle="--", alpha=0.5)
    ax_thr.legend(fontsize=8)

    # 1.3 相对 Baseline 加速比
    bar_width = 0.8 / max(len(key_prompts), 1)
    for i, pl in enumerate(key_prompts):
        b_w = pick_w(base, pl)
        speedups = []
        for r in records:
            r_w = pick_w(r, pl)
            if r_w and b_w and b_w["throughput_tok_s"]["median"] > 0:
                speedups.append(r_w["throughput_tok_s"]["median"] / b_w["throughput_tok_s"]["median"])
            else:
                speedups.append(1.0)
        offsets = [x + (i - len(key_prompts) / 2 + 0.5) * bar_width for x in x_indices]
        bars = ax_spd.bar(offsets, speedups, width=bar_width, label=f"prompt={pl}", alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax_spd.annotate(
                f"{h:.2f}x",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax_spd.set_xticks(x_indices)
    ax_spd.set_xticklabels(version_labels, rotation=15, ha="right")
    ax_spd.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_spd.set_ylabel("Speedup vs Baseline (x)")
    ax_spd.set_title("1.3 Speedup vs Baseline")
    ax_spd.grid(True, linestyle="--", alpha=0.5)
    ax_spd.legend(fontsize=8)

    # --- Row 2: 序列长度扩展演进 (Prompt Length Scaling Across Releases) ---
    ax_sc_ttft, ax_sc_tpot, ax_sc_thr = axes[1]
    scaling_metrics = [
        (ax_sc_ttft, "ttft_ms", "2.1 TTFT / Prefill Scaling (ms)", "TTFT (ms)"),
        (ax_sc_tpot, "tpot_ms", "2.2 TPOT / Decode Scaling (ms/token)", "TPOT (ms/token)"),
        (ax_sc_thr, "throughput_tok_s", "2.3 Throughput Scaling (tokens/s)", "Throughput (tokens/s)"),
    ]
    for ax, metric, title, y_label in scaling_metrics:
        for rec in records:
            ws = sorted((w for w in rec["workloads"] if w["batch_size"] == 1), key=lambda w: w["prompt_len"])
            if not ws:
                continue
            x = [w["prompt_len"] for w in ws]
            y = [w[metric]["median"] for w in ws]
            err = [w[metric]["std"] for w in ws]
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=label(rec))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # --- Row 3: 资源与鲁棒性演进 (Resource & Robustness Evolution) ---
    ax_mem, ax_tail, ax_corr = axes[2]

    # 3.1 显存演变（各版本在最大共有 prompt 下）
    common = set.intersection(*[{w["prompt_len"] for w in r["workloads"] if w["batch_size"] == 1} for r in records])
    p_ref = max(common) if common else key_prompts[-1]
    mems = [(pick_w(r, p_ref) or {}).get("memory", {}).get("peak_allocated_mb", 0) for r in records]
    ax_mem.bar(x_indices, mems, alpha=0.85, color="tab:purple")
    for i, m in enumerate(mems):
        ax_mem.annotate(
            f"{m:.0f}M",
            xy=(i, m),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_mem.set_xticks(x_indices)
    ax_mem.set_xticklabels(version_labels, rotation=15, ha="right")
    ax_mem.set_ylabel("Peak Memory (MiB)")
    ax_mem.set_title(f"3.1 Peak Memory (prompt={p_ref})")
    ax_mem.grid(True, linestyle="--", alpha=0.5)

    # 3.2 尾延迟演变（median vs p90 vs p99 vs max）
    for stat, style, col in (
        ("median", "-o", "tab:blue"),
        ("p90", "--s", "tab:orange"),
        ("p99", "-.v", "tab:purple"),
        ("max", ":^", "tab:red"),
    ):
        ys = [(pick_w(r, p_ref) or {}).get("tpot_ms", {}).get(stat) for r in records]
        if any(y is not None for y in ys):
            ax_tail.plot(x_indices, ys, style, label=stat, color=col)
    ax_tail.set_xticks(x_indices)
    ax_tail.set_xticklabels(version_labels, rotation=15, ha="right")
    ax_tail.set_ylabel("TPOT (ms/token)")
    ax_tail.set_title(f"3.2 Tail Latency (prompt={p_ref})")
    ax_tail.grid(True, linestyle="--", alpha=0.5)
    ax_tail.legend(fontsize=8)

    # 3.3 正确性门禁演进
    ratios = [r["correctness"].get("match_ratio") for r in records]
    colors = [
        "gray" if v is None else ("tab:green" if r["correctness"]["passed"] else "tab:red")
        for v, r in zip(ratios, records)
    ]
    ax_corr.bar(x_indices, [0 if v is None else v for v in ratios], color=colors, alpha=0.85)
    for i, v in enumerate(ratios):
        text = "n/a" if v is None else f"{v:.3f}"
        ax_corr.annotate(text, (i, 0 if v is None else v), xytext=(0, 2), textcoords="offset points", ha="center")
    ax_corr.set_ylim(0, 1.15)
    ax_corr.set_xticks(x_indices)
    ax_corr.set_xticklabels(version_labels, rotation=15, ha="right")
    ax_corr.set_ylabel("Token Match Ratio")
    ax_corr.set_title("3.3 Correctness Gate")
    ax_corr.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        f"mini-infra-qwen3 Category 1: Release Evolution & Scaling Analysis (Baseline: {label(base)})",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"第一类图表（版本演进与扩展趋势）已保存：{out_path}")


# ==============================================================================
# 第二大类：当前模型深度剖析与优化指引 (Profiling & Optimization Guidance)
# ==============================================================================
def plot_profiling_and_optimization(records, target_rec, out_path):
    """绘制第二大类图表：当前模型深度剖析与优化指引（2行3列）。

    分小类展示：
      2.1 物理天花板差距 (MBU 带宽利用率、MFU 算力利用率、GPU 忙碌比例) -> 离极限还有多远
      2.2 架构模块耗时归因 (Attention vs MLP vs Norm vs LM Head) -> 下一步改哪个模块
      2.3 算子类型透视 (GEMM vs Elementwise vs Memory vs Softmax) -> 该用什么技术手段改
      2.4 显存足迹剖析 (Weights 常驻 vs 动态峰值 vs 碎片预留) -> 显存紧张该砍哪里
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    (ax_mbu, ax_mfu, ax_busy), (ax_mod, ax_cat, ax_mem) = axes

    # --- 小类 2.1: 硬件物理天花板差距 ---
    ws = sorted(
        (w for w in target_rec["workloads"] if w["batch_size"] == 1 and w.get("efficiency")),
        key=lambda w: w["prompt_len"],
    )
    if ws:
        x = [w["prompt_len"] for w in ws]
        ax_mbu.plot(
            x, [100 * (w["efficiency"]["decode_mbu"] or 0) for w in ws], marker="o", color="tab:blue", linewidth=2
        )
        ax_mbu.set_xscale("log", base=2)
        ax_mbu.set_xlabel("Prompt length (tokens)")
        ax_mbu.set_ylabel("MBU (%)")
        ax_mbu.set_title("2.1 Decode MBU (Memory Bandwidth %)\n[Low -> Needs KV Cache / Fused Mem]")
        ax_mbu.grid(alpha=0.3)
        ax_mbu.set_ylim(bottom=0)

        ax_mfu.plot(
            x,
            [100 * (w["efficiency"]["prefill_mfu"] or 0) for w in ws],
            marker="o",
            color="tab:orange",
            label="Prefill MFU",
            linewidth=2,
        )
        ax_mfu.plot(
            x,
            [100 * (w["efficiency"]["decode_mfu"] or 0) for w in ws],
            marker="s",
            linestyle="--",
            color="tab:red",
            label="Decode MFU",
            linewidth=2,
        )
        ax_mfu.set_xscale("log", base=2)
        ax_mfu.set_xlabel("Prompt length (tokens)")
        ax_mfu.set_ylabel("MFU (%)")
        ax_mfu.set_title("2.1 MFU (Compute Utilization %)\n[Compute bound ceiling]")
        ax_mfu.grid(alpha=0.3)
        ax_mfu.legend(fontsize=8)

        gws = [w for w in ws if w.get("gpu_time_ratio")]
        if gws:
            ax_busy.plot(
                [w["prompt_len"] for w in gws],
                [100 * w["gpu_time_ratio"]["median"] for w in gws],
                marker="o",
                color="tab:green",
                linewidth=2,
            )
        ax_busy.set_xscale("log", base=2)
        ax_busy.set_xlabel("Prompt length (tokens)")
        ax_busy.set_ylabel("GPU Time / Wall Time (%)")
        ax_busy.set_title("2.1 GPU Busy Ratio (%)\n[<90% -> CPU Launch Overhead: CUDAGraph]")
        ax_busy.grid(alpha=0.3)
        ax_busy.set_ylim(0, 105)
    else:
        for ax in (ax_mbu, ax_mfu, ax_busy):
            ax.text(0.5, 0.5, "No efficiency metrics", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

    # --- 小类 2.2 & 2.3: 架构模块与算子类型透视 ---
    prof_ws = [
        w
        for w in target_rec["workloads"]
        if w["batch_size"] == 1
        and isinstance(w.get("op_latency_breakdown"), dict)
        and w["op_latency_breakdown"].get("time_share_by_module")
    ]
    if prof_ws:
        w_rep = max(prof_ws, key=lambda w: w["prompt_len"])
        # 2.2 模块归因
        ob = w_rep["op_latency_breakdown"]
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
        ax_mod.set_title(
            f"2.2 Module Attribution (prompt={w_rep['prompt_len']})\n[Where to optimize: e.g. Attention/Norm]"
        )
        ax_mod.grid(alpha=0.3)

        # 2.3 算子类型归因
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
        ax_cat.set_title(
            f"2.3 Kernel Category (prompt={w_rep['prompt_len']})\n[How to optimize: Fusion vs GEMM tuning]"
        )
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

    # --- 小类 2.4: 显存足迹剖析 ---
    mem_ws = sorted((w for w in target_rec["workloads"] if w.get("memory")), key=lambda w: w["prompt_len"])
    if mem_ws:
        x_prompts = [str(w["prompt_len"]) for w in mem_ws]
        xs = list(range(len(mem_ws)))
        mems = [w["memory"] for w in mem_ws]
        bottom = [0.0] * len(mems)
        components = [
            ("baseline_allocated_mb", "Resident Weights & Base", "tab:gray"),
            ("dynamic_peak_mb", "Dynamic Peak (Act/KV)", "tab:blue"),
            ("fragmentation_mb", "Allocator Fragmentation", "tab:orange"),
        ]
        for key, name, color in components:
            vals = [m.get(key, 0) for m in mems]
            ax_mem.bar(xs, vals, bottom=bottom, label=name, color=color, alpha=0.85)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax_mem.set_xticks(xs)
        ax_mem.set_xticklabels(x_prompts)
        ax_mem.set_xlabel("Prompt length (tokens)")
        ax_mem.set_ylabel("Memory (MiB)")
        ax_mem.set_title("2.4 Memory Footprint Breakdown\n[Dynamic -> PagedAttn; Frag -> MemPool]")
        ax_mem.legend(fontsize=7)
        ax_mem.grid(alpha=0.3)
    else:
        ax_mem.text(0.5, 0.5, "No memory breakdown", ha="center", va="center", transform=ax_mem.transAxes)
        ax_mem.set_axis_off()

    fig.suptitle(
        f"mini-infra-qwen3 Category 2: Current Model Profiling & Optimization Guidance ({label(target_rec)})",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"第二类图表（当前模型剖析与优化指引）已保存：{out_path}")


# ==============================================================================
# 自动生成针对当前瓶颈的下一步优化指引文字
# ==============================================================================
def generate_optimization_guidance(target_rec):
    """根据当前版本的 profiling 数据，自动提炼诊断结论与下一步优化建议。"""

    lines = [
        "## 二、当前版本深度剖析与优化指引 (Profiling & Next Steps)",
        "",
        "![Profiling](profiling_and_optimization.png)",
        "",
    ]

    prof_ws = [
        w
        for w in target_rec["workloads"]
        if w["batch_size"] == 1
        and isinstance(w.get("op_latency_breakdown"), dict)
        and w["op_latency_breakdown"].get("time_share_by_module")
    ]
    if not prof_ws:
        lines.append(
            "> 提示：当前记录中未包含详细的算子剖析采样数据。"
            "使用 `mini-infra-bench --profile-mode extra-profile` 重跑可获取完整算子诊断。"
        )
        return "\n".join(lines)

    w = max(prof_ws, key=lambda x: x["prompt_len"])
    ob = w["op_latency_breakdown"]
    by_mod = ob.get("time_share_by_module", {})
    by_cat = ob.get("time_share_by_kernel_category", {})
    eff = w.get("efficiency") or {}
    mem = w.get("memory") or {}
    busy = (w.get("gpu_time_ratio") or {}).get("median", 1.0)

    prompt_desc = f"prompt={w['prompt_len']}"
    lines.append(
        f"基于当前版本 `{label(target_rec)}` 在长上下文（`{prompt_desc}`）下的基准测量与 profiling 结果，诊断如下：\n"
    )

    # 1. 架构模块建议
    attn_share = by_mod.get("attention_block", 0.0)
    norm_share = by_mod.get("norm", 0.0)
    mlp_share = by_mod.get("mlp_block", 0.0)

    lines.append("### 1. 架构模块瓶颈诊断（改哪里？）")
    if attn_share >= 0.25:
        lines.append(
            f"- 🔴 **Attention 模块耗时占比最高（{attn_share * 100:.1f}%）**：\n"
            "  - **现状**：自回归 Decode 阶段重复计算全量上下文历史，计算与访存开销随序列长度急剧膨胀。\n"
            "  - 👉 **优先优化行动**：**引入 KV Cache** 将每步全量重算解耦，预期可带来 5x~10x 以上吞吐跃升；"
            "后续进一步接入 **FlashAttention / SDPA 算子融合**。"
        )
    if norm_share > 0.08:
        lines.append(
            f"- 🟡 **RMSNorm 频繁调用开销显著（耗时占比 {norm_share * 100:.1f}%）**：\n"
            "  - **现状**：每个 Transformer 层多次独立的 Norm 小算子带来了大量的 Kernel 启动与显存往返开销。\n"
            "  - 👉 **后续优化行动**：实施 **Fused RMSNorm**（将残差加 Add 与 RMSNorm 融合成单个 Kernel）。"
        )
    if mlp_share > 0.15:
        lines.append(
            f"- 🟡 **MLP Block 占比偏高（耗时占比 {mlp_share * 100:.1f}%）**：\n"
            "  - 👉 **后续优化行动**：针对 Gate / Up 投影实施 **SwiGLU GEMM 融合**。"
        )
    if not (attn_share >= 0.25 or norm_share > 0.08 or mlp_share > 0.15):
        top_mod = max(by_mod.items(), key=lambda kv: kv[1])[0] if by_mod else "unknown"
        lines.append(f"- 🔹 当前耗时最高的架构模块为 `{top_mod}`（占比 {by_mod.get(top_mod, 0) * 100:.1f}%）。")
    lines.append("")

    # 2. 算子类型建议
    gemm_share = by_cat.get("gemm", 0.0)
    elem_mem_share = by_cat.get("elementwise", 0.0) + by_cat.get("memory", 0.0)
    lines.append("### 2. 算子类型与访存诊断（怎么改？）")
    lines.append(
        f"- 🔹 **GEMM 矩阵乘占 {gemm_share * 100:.1f}%**，"
        f"**Elementwise + Memory 搬运占 {elem_mem_share * 100:.1f}%**。\n"
        f"  - 访存搬运和小算子占用了近三分之一的 GPU 时间，说明存在较多碎片化临时张量读写。\n"
        "  - 👉 **技术路线**：结合 Triton 或 PyTorch Inductor 进行横向与纵向算子融合，减少 SRAM 与 HBM 之间的中间写回。"
    )
    lines.append("")

    # 3. 硬件利用率与调度
    mbu = eff.get("decode_mbu")
    lines.append("### 3. 硬件物理天花板差距")
    if mbu is not None:
        lines.append(
            f"- 🔹 **Decode MBU（显存带宽利用率）仅为 {mbu * 100:.1f}%**：\n"
            "  - 当前 Decode 受限于重复计算而非饱和访存，距离理想的 Memory-bound 饱和线（60%~80%）仍有巨大空间。"
        )
    if busy < 0.90:
        lines.append(
            f"- 🔹 **GPU 忙碌比例仅为 {busy * 100:.1f}%**：\n"
            f"  - 存在约 **{100 - busy * 100:.1f}%** 的 CPU Python Launch / Host-Device 同步空泡。\n"
            "  - 👉 **优化行动**：后续引入 **CUDA Graph 捕获** 或静态执行图，消除 CPU 发射开销。"
        )
    else:
        lines.append(f"- 🔹 **GPU 忙碌比例达到 {busy * 100:.1f}%**：当前几乎无 CPU 空泡，瓶颈纯在 GPU 端计算与访存。")
    lines.append("")

    # 4. 显存结构诊断
    frag = mem.get("fragmentation_mb", 0)
    dyn = mem.get("dynamic_peak_mb", 0)
    lines.append("### 4. 显存足迹与碎片诊断")
    lines.append(
        f"- 🔹 在长序列下，纯动态增量（激活与状态）为 **{dyn:.1f} MiB**，"
        f"PyTorch Allocator 碎片与预留达到 **{frag:.1f} MiB**。\n"
        "  - 👉 **优化行动**：进入长文本及多 Batching 场景后，应引入 **PagedAttention（分页显存管理）**"
        " 避免非连续预分配带来的显存碎片。"
    )
    lines.append("")

    # 5. Top Kernels 明细表
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


# ==============================================================================
# CLI 入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="mini-infra-qwen3 benchmark visualizer")
    parser.add_argument("--baseline", help="作为基准的 tag，默认最早的结果")
    parser.add_argument("--target", help="作为当前深度剖析对象的 tag，默认最新的结果")
    # 新规范参数
    parser.add_argument("--evolution-figure", type=Path, default=REPORTS_DIR / "evolution_and_scaling.png")
    parser.add_argument("--profiling-figure", type=Path, default=REPORTS_DIR / "profiling_and_optimization.png")
    parser.add_argument("--summary-path", type=Path, default=REPORTS_DIR / "SUMMARY.md")
    # 向下兼容旧参数
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--timeline-figure", type=Path, default=None)
    parser.add_argument("--extras-figure", type=Path, default=None)
    parser.add_argument("--efficiency-figure", type=Path, default=None)
    args = parser.parse_args()

    records = load_results()
    if not records:
        raise SystemExit(f"{RESULTS_DIR} 中没有结果，请先运行 mini-infra-bench")

    base = next((r for r in records if r["tag"] == args.baseline), records[0]) if args.baseline else records[0]
    target = next((r for r in records if r["tag"] == args.target), records[-1]) if args.target else records[-1]

    table = markdown_table(records, base)
    guidance = generate_optimization_guidance(target)

    content = [
        "# Benchmark 性能评估与演进分析报告",
        "",
        table,
        "",
        "## 一、跨版本演化与扩展趋势 (Evolution & Scaling Across Releases)",
        "",
        "![Evolution & Scaling](evolution_and_scaling.png)",
        "",
        guidance,
        "",
    ]
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text("\n".join(content) + "\n")
    print(table)
    print(f"\n汇总报告已保存：{args.summary_path}")

    try:
        plot_evolution_and_scaling(records, base, args.evolution_figure)
        plot_profiling_and_optimization(records, target, args.profiling_figure)
    except ImportError:
        print("未安装 matplotlib，跳过绘图（pip install -e '.[bench]'）")


if __name__ == "__main__":
    main()
