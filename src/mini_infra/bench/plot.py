"""读取 benchmarks/results/*.json，生成 Markdown 汇总表、序列长度扩展曲线与特性演进时间线图。

用法：
    mini-infra-bench-plot                  # 以最早的结果作为 baseline
    mini-infra-bench-plot --baseline baseline
"""

import argparse
import json
import os
from pathlib import Path

BENCH_DIR = Path(os.environ.get("MINI_INFRA_BENCH_DIR", "benchmarks")).resolve()
RESULTS_DIR = BENCH_DIR / "results"
REPORTS_DIR = BENCH_DIR / "reports"


def load_results():
    records = [json.loads(p.read_text()) for p in RESULTS_DIR.glob("*.json")]
    records.sort(key=lambda r: r["timestamp"])
    return records


def index(record):
    return {(w["batch_size"], w["prompt_len"], w["output_len"]): w for w in record["workloads"]}


def label(record):
    return record["tag"] + ("*" if record["git"]["dirty"] else "")


def markdown_table(records, base, key="prompt_len"):
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
            ttft, tpot, thr = (w["ttft_ms"]["median"], w["tpot_ms"]["median"],
                               w["throughput_tok_s"]["median"])
            if b:
                s_ttft = f" ({b['ttft_ms']['median'] / ttft:.2f}x)"
                s_tpot = f" ({b['tpot_ms']['median'] / tpot:.2f}x)" if tpot else ""
                s_thr = f" ({thr / b['throughput_tok_s']['median']:.2f}x)"
            else:
                s_ttft = s_tpot = s_thr = ""
            lines.append(
                f"| {label(rec)} | {rec['git']['commit']} | {k[0]} | {k[1]} | {k[2]} | "
                f"{ttft:.2f}{s_ttft} | {tpot:.2f}{s_tpot} | {thr:.1f}{s_thr} | "
                f"{w['peak_memory_mb']:.0f} | {flag} |")
    return "\n".join(lines)


def plot_scaling(records, out_path):
    """序列长度扩展曲线：横轴为 Prompt 长度，展示不同版本在各上下文长度下的表现。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("ttft_ms", "TTFT / Prefill (ms)"), ("tpot_ms", "TPOT / Decode (ms/token)"),
               ("throughput_tok_s", "Throughput (tokens/s)")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (metric, title) in zip(axes, metrics):
        for rec in records:
            ws = sorted((w for w in rec["workloads"] if w["batch_size"] == 1),
                        key=lambda w: w["prompt_len"])
            if not ws:
                continue
            x = [w["prompt_len"] for w in ws]
            y = [w[metric]["median"] for w in ws]
            err = [w[metric]["std"] for w in ws]
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=label(rec))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(title="version (*=dirty)")
    fig.suptitle("mini-infra-qwen3 Prompt Length Scaling (batch=1, error bar = std)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"扩展曲线图已保存：{out_path}")


def plot_timeline(records, base, out_path):
    """特性演进时间线图：横轴为特性/版本里程碑，展示 Decode 延迟下降、吞吐提升及相对基准的加速比演进。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        return

    version_labels = [label(r) for r in records]
    x_indices = list(range(len(records)))
    base_idx = index(base)

    all_prompt_lens = sorted({w["prompt_len"] for r in records for w in r["workloads"] if w["batch_size"] == 1})
    key_prompts = [p for p in [32, 128, 512, 1024] if p in all_prompt_lens]
    if not key_prompts:
        key_prompts = all_prompt_lens[:3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Decode 延迟演化（TPOT）
    ax1 = axes[0]
    for pl in key_prompts:
        y = []
        for r in records:
            r_idx = index(r)
            match = next((w for k, w in r_idx.items() if k[0] == 1 and k[1] == pl), None)
            y.append(match["tpot_ms"]["median"] if match else None)
        ax1.plot(x_indices, y, marker="o", linewidth=2, label=f"prompt={pl}")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(version_labels, rotation=15, ha="right")
    ax1.set_ylabel("TPOT / Decode (ms/token)")
    ax1.set_title("Decode Latency Evolution")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # 2. 吞吐演化（Throughput）
    ax2 = axes[1]
    for pl in key_prompts:
        y = []
        for r in records:
            r_idx = index(r)
            match = next((w for k, w in r_idx.items() if k[0] == 1 and k[1] == pl), None)
            y.append(match["throughput_tok_s"]["median"] if match else None)
        ax2.plot(x_indices, y, marker="s", linewidth=2, label=f"prompt={pl}")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(version_labels, rotation=15, ha="right")
    ax2.set_ylabel("Throughput (tokens/s)")
    ax2.set_title("Throughput Evolution")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # 3. 相对 Baseline 的加速比演化（Speedup）
    ax3 = axes[2]
    bar_width = 0.8 / max(len(key_prompts), 1)
    for i, pl in enumerate(key_prompts):
        base_match = next((w for k, w in base_idx.items() if k[0] == 1 and k[1] == pl), None)
        speedups = []
        for r in records:
            r_idx = index(r)
            match = next((w for k, w in r_idx.items() if k[0] == 1 and k[1] == pl), None)
            if match and base_match and base_match["throughput_tok_s"]["median"] > 0:
                speedups.append(match["throughput_tok_s"]["median"] / base_match["throughput_tok_s"]["median"])
            else:
                speedups.append(1.0)
        offsets = [x + (i - len(key_prompts) / 2 + 0.5) * bar_width for x in x_indices]
        bars = ax3.bar(offsets, speedups, width=bar_width, label=f"prompt={pl}", alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f"{height:.2f}x",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)

    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(version_labels, rotation=15, ha="right")
    ax3.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax3.set_ylabel("Speedup vs Baseline (x)")
    ax3.set_title("Speedup Evolution (Throughput)")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    fig.suptitle(f"mini-infra-qwen3 Feature Evolution Timeline (Baseline: {label(base)})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"特性演进时间线图已保存：{out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", help="作为基准的 tag，默认最早的结果")
    parser.add_argument("--figure", type=Path, default=REPORTS_DIR / "trend.png")
    parser.add_argument("--timeline-figure", type=Path, default=REPORTS_DIR / "timeline.png")
    parser.add_argument("--table", type=Path, default=REPORTS_DIR / "SUMMARY.md")
    args = parser.parse_args()

    records = load_results()
    if not records:
        raise SystemExit(f"{RESULTS_DIR} 中没有结果，请先运行 run_benchmark.py")
    base = next((r for r in records if r["tag"] == args.baseline), records[0]) \
        if args.baseline else records[0]

    table = markdown_table(records, base)
    content = [
        "# Benchmark 汇总",
        "",
        table,
        "",
        "## 可视化趋势分析",
        "",
        "### 1. 特性演进时间线 (Feature Evolution Timeline)",
        "![Feature Timeline](timeline.png)",
        "",
        "### 2. 序列长度扩展曲线 (Prompt Length Scaling)",
        "![Prompt Scaling](trend.png)",
        "",
    ]
    args.table.write_text("\n".join(content) + "\n")
    print(table)
    print(f"\n汇总表已保存：{args.table}")
    try:
        plot_scaling(records, args.figure)
        plot_timeline(records, base, args.timeline_figure)
    except ImportError:
        print("未安装 matplotlib，跳过绘图（pip install -e '.[bench]'）")


if __name__ == "__main__":
    main()
