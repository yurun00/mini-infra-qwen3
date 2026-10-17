"""对比两个独立 benchmark JSON 结果（例如比较不同的 mode），并生成对比图表与数据报表。

适用场景：
    日常调优时使用 mini-infra-bench --output 输出的独立 JSON 文件，
    通过本脚本直接对比两个结果（例如 no-cache vs kv-cache），画出性能对比图，
    而无需沉淀到 benchmarks/result-history/ 污染正式发版时间线。

用法示例：
    # 命令行直接运行
    mini-infra-bench-plot-compare tmp/test_nocache.json tmp/test_kvcache.json -o tmp/compare.png

    # 或者作为模块调用
    python -m mini_infra.bench.plot_compare tmp/test_nocache.json tmp/test_kvcache.json
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_record(path: Path) -> dict[str, Any]:
    """读取单份 benchmark JSON 结果。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_label(rec: dict[str, Any], custom: str | None = None) -> str:
    """提取对齐显示的模式/版本标签。"""
    if custom:
        return custom
    mode = rec.get("mode")
    tag = rec.get("tag")
    if mode and tag and mode != tag:
        return f"{mode} ({tag})"
    return mode or tag or "unknown"


def index_workloads(rec: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    """按 (batch_size, prompt_len, output_len) 建立索引。"""
    res = {}
    for w in rec.get("workloads", []):
        key = (w["batch_size"], w["prompt_len"], w["output_len"])
        res[key] = w
    return res


def generate_comparison_table(
    rec1: dict[str, Any],
    rec2: dict[str, Any],
    label1: str,
    label2: str,
) -> str:
    """生成 Markdown 对比表格与文字总结。"""
    idx1 = index_workloads(rec1)
    idx2 = index_workloads(rec2)
    common_keys = sorted(set(idx1.keys()) & set(idx2.keys()))

    gpu1 = rec1.get("env", {}).get("gpu", "Unknown GPU")
    gpu2 = rec2.get("env", {}).get("gpu", "Unknown GPU")
    gpu_info = gpu1 if gpu1 == gpu2 else f"{gpu1} vs {gpu2}"

    lines = [
        f"### 引擎性能对比：`{label1}` vs `{label2}`",
        f"- **测试设备**：{gpu_info}",
        f"- **基准 (A)**：`{label1}` (commit: `{rec1.get('git', {}).get('commit', 'n/a')}`)",
        f"- **对比 (B)**：`{label2}` (commit: `{rec2.get('git', {}).get('commit', 'n/a')}`)",
        "",
        "| Batch | Prompt | Output | TTFT A (ms) | TTFT B (ms) | TTFT 加速比 | "
        "TPOT A (ms) | TPOT B (ms) | TPOT 加速比 | "
        "吞吐 A (tok/s) | 吞吐 B (tok/s) | 吞吐加速比 | 显存 A (MiB) | 显存 B (MiB) | 显存增量 (MiB) |",
        "| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ]

    tpot_speedups = []
    thr_speedups = []

    for k in common_keys:
        bs, pl, ol = k
        w1 = idx1[k]
        w2 = idx2[k]

        ttft1 = w1["ttft_ms"]["median"]
        ttft2 = w2["ttft_ms"]["median"]
        s_ttft = ttft1 / ttft2 if ttft2 > 0 else 1.0

        tpot1 = w1["tpot_ms"]["median"]
        tpot2 = w2["tpot_ms"]["median"]
        s_tpot = tpot1 / tpot2 if tpot2 > 0 else 1.0
        tpot_speedups.append(s_tpot)

        thr1 = w1["throughput_tok_s"]["median"]
        thr2 = w2["throughput_tok_s"]["median"]
        s_thr = thr2 / thr1 if thr1 > 0 else 1.0
        thr_speedups.append(s_thr)

        mem1 = w1.get("memory", {}).get("peak_allocated_mb", 0.0)
        mem2 = w2.get("memory", {}).get("peak_allocated_mb", 0.0)
        d_mem = mem2 - mem1

        lines.append(
            f"| {bs} | {pl} | {ol} | "
            f"{ttft1:.2f} | {ttft2:.2f} | **{s_ttft:.2f}x** | "
            f"{tpot1:.2f} | {tpot2:.2f} | **{s_tpot:.2f}x** | "
            f"{thr1:.1f} | {thr2:.1f} | **{s_thr:.2f}x** | "
            f"{mem1:.0f} | {mem2:.0f} | {d_mem:+.0f} |"
        )

    if tpot_speedups:
        max_tpot_s = max(tpot_speedups)
        avg_thr_s = sum(thr_speedups) / len(thr_speedups)
        lines.append("")
        lines.append(
            f"> 💡 **核心结论**：相比 `{label1}`，`{label2}` 的 Decode 最高加速比达 **{max_tpot_s:.2f}x**，"
            f"平均吞吐加速达 **{avg_thr_s:.2f}x**。"
        )

    return "\n".join(lines)


def plot_mode_comparison(
    rec1: dict[str, Any],
    rec2: dict[str, Any],
    label1: str,
    label2: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    """绘制两份结果的详细多维度性能对比图（2行3列）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx1 = index_workloads(rec1)
    idx2 = index_workloads(rec2)
    common_keys = sorted(set(idx1.keys()) & set(idx2.keys()), key=lambda k: (k[0], k[2], k[1]))

    if not common_keys:
        raise ValueError("两份结果中没有找到匹配的 Workload (batch_size, prompt_len, output_len)，无法绘图。")

    # 优先筛选 batch_size == 1 且 output_len 最常见的子集画趋势曲线
    primary_bs = 1 if any(k[0] == 1 for k in common_keys) else common_keys[0][0]
    output_lens = [k[2] for k in common_keys if k[0] == primary_bs]
    primary_ol = max(set(output_lens), key=output_lens.count) if output_lens else common_keys[0][2]

    plot_keys = [k for k in common_keys if k[0] == primary_bs and k[2] == primary_ol]
    if not plot_keys:
        plot_keys = common_keys

    prompt_lens = [k[1] for k in plot_keys]
    x_indices = list(range(len(prompt_lens)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    (ax_tpot, ax_thr, ax_spd), (ax_ttft, ax_mem, ax_e2e) = axes

    c1 = "#e74c3c"  # Red for Base / No-Cache
    c2 = "#2ecc71"  # Green for Optimized / KV-Cache
    c_spd = "#3498db"  # Blue for Speedup
    c_mem1 = "#9b59b6"  # Purple
    c_mem2 = "#1abc9c"  # Teal

    # 1. TPOT / Decode Latency (ms/token)
    tpot1 = [idx1[k]["tpot_ms"]["median"] for k in plot_keys]
    tpot2 = [idx2[k]["tpot_ms"]["median"] for k in plot_keys]
    err1 = [idx1[k]["tpot_ms"].get("std", 0.0) for k in plot_keys]
    err2 = [idx2[k]["tpot_ms"].get("std", 0.0) for k in plot_keys]

    ax_tpot.errorbar(x_indices, tpot1, yerr=err1, marker="o", linewidth=2, color=c1, capsize=3, label=label1)
    ax_tpot.errorbar(x_indices, tpot2, yerr=err2, marker="s", linewidth=2, color=c2, capsize=3, label=label2)
    ax_tpot.set_xticks(x_indices)
    ax_tpot.set_xticklabels(prompt_lens)
    ax_tpot.set_xlabel("Prompt Length (tokens)")
    ax_tpot.set_ylabel("TPOT / Decode (ms/token)")
    ax_tpot.set_title("1. Decode Latency (Lower is Better)")
    ax_tpot.grid(True, linestyle="--", alpha=0.5)
    ax_tpot.legend(fontsize=9)

    # 2. Throughput (tokens/s)
    thr1 = [idx1[k]["throughput_tok_s"]["median"] for k in plot_keys]
    thr2 = [idx2[k]["throughput_tok_s"]["median"] for k in plot_keys]
    ax_thr.plot(x_indices, thr1, marker="o", linewidth=2, color=c1, label=label1)
    ax_thr.plot(x_indices, thr2, marker="s", linewidth=2, color=c2, label=label2)
    ax_thr.set_xticks(x_indices)
    ax_thr.set_xticklabels(prompt_lens)
    ax_thr.set_xlabel("Prompt Length (tokens)")
    ax_thr.set_ylabel("Throughput (tokens/s)")
    ax_thr.set_title("2. Generation Throughput (Higher is Better)")
    ax_thr.grid(True, linestyle="--", alpha=0.5)
    ax_thr.legend(fontsize=9)

    # 3. Speedup (x)
    speedups = [(t1 / t2 if t2 > 0 else 1.0) for t1, t2 in zip(tpot1, tpot2)]
    bars = ax_spd.bar(x_indices, speedups, width=0.5, color=c_spd, alpha=0.85, label=f"{label2} vs {label1}")
    ax_spd.axhline(1.0, color="gray", linestyle="--", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        ax_spd.annotate(
            f"{h:.2f}x",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax_spd.set_xticks(x_indices)
    ax_spd.set_xticklabels(prompt_lens)
    ax_spd.set_xlabel("Prompt Length (tokens)")
    ax_spd.set_ylabel("Speedup Ratio (x)")
    ax_spd.set_title("3. Decode Speedup Ratio")
    ax_spd.grid(True, linestyle="--", alpha=0.5)
    ax_spd.legend(fontsize=9)

    # 4. TTFT / Prefill Latency (ms)
    ttft1 = [idx1[k]["ttft_ms"]["median"] for k in plot_keys]
    ttft2 = [idx2[k]["ttft_ms"]["median"] for k in plot_keys]
    ax_ttft.plot(x_indices, ttft1, marker="o", linewidth=2, color=c1, label=label1)
    ax_ttft.plot(x_indices, ttft2, marker="s", linewidth=2, color=c2, label=label2)
    ax_ttft.set_xticks(x_indices)
    ax_ttft.set_xticklabels(prompt_lens)
    ax_ttft.set_xlabel("Prompt Length (tokens)")
    ax_ttft.set_ylabel("TTFT / Prefill (ms)")
    ax_ttft.set_title("4. Prefill Latency (Lower is Better)")
    ax_ttft.grid(True, linestyle="--", alpha=0.5)
    ax_ttft.legend(fontsize=9)

    # 5. Peak GPU Memory (MiB) - Grouped Bar Chart
    bar_w = 0.35
    x_left = [x - bar_w / 2 for x in x_indices]
    x_right = [x + bar_w / 2 for x in x_indices]
    mem1 = [idx1[k].get("memory", {}).get("peak_allocated_mb", 0.0) for k in plot_keys]
    mem2 = [idx2[k].get("memory", {}).get("peak_allocated_mb", 0.0) for k in plot_keys]

    b1 = ax_mem.bar(x_left, mem1, width=bar_w, color=c_mem1, alpha=0.85, label=label1)
    b2 = ax_mem.bar(x_right, mem2, width=bar_w, color=c_mem2, alpha=0.85, label=label2)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax_mem.annotate(
                f"{h:.0f}M",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax_mem.set_xticks(x_indices)
    ax_mem.set_xticklabels(prompt_lens)
    ax_mem.set_xlabel("Prompt Length (tokens)")
    ax_mem.set_ylabel("Peak Allocated (MiB)")
    ax_mem.set_title("5. Peak GPU Memory Comparison")
    ax_mem.grid(True, linestyle="--", alpha=0.5)
    ax_mem.legend(fontsize=9)

    # 6. End-to-End Latency (ms)
    e2e1 = [idx1[k].get("e2e_ms", {}).get("median", 0.0) for k in plot_keys]
    e2e2 = [idx2[k].get("e2e_ms", {}).get("median", 0.0) for k in plot_keys]
    ax_e2e.plot(x_indices, e2e1, marker="o", linewidth=2, color=c1, label=label1)
    ax_e2e.plot(x_indices, e2e2, marker="s", linewidth=2, color=c2, label=label2)
    ax_e2e.set_xticks(x_indices)
    ax_e2e.set_xticklabels(prompt_lens)
    ax_e2e.set_xlabel("Prompt Length (tokens)")
    ax_e2e.set_ylabel("E2E Latency (ms)")
    ax_e2e.set_title("6. End-to-End Latency (Lower is Better)")
    ax_e2e.grid(True, linestyle="--", alpha=0.5)
    ax_e2e.legend(fontsize=9)

    gpu_name = rec1.get("env", {}).get("gpu") or rec2.get("env", {}).get("gpu") or "GPU"
    default_title = (
        f"Benchmark Comparison: {label1} vs {label2}\n[{gpu_name} | batch={primary_bs}, output_len={primary_ol}]"
    )
    fig.suptitle(title or default_title, fontsize=13, y=0.995)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"对比图已保存至: {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="比较两个独立 benchmark 运行结果 JSON 文件，打印详细指标并绘制性能对比图。"
    )
    parser.add_argument("file_a", type=Path, help="基准结果 JSON 文件路径 (如 no-cache)")
    parser.add_argument("file_b", type=Path, help="对比结果 JSON 文件路径 (如 kv-cache)")
    parser.add_argument(
        "-o",
        "--output-figure",
        type=Path,
        default=Path("tmp/compare.png"),
        help="对比图保存路径 (默认: tmp/compare.png)",
    )
    parser.add_argument(
        "-m",
        "--output-markdown",
        type=Path,
        default=None,
        help="Markdown 汇总表格保存路径 (可选)",
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default=None,
        help="文件 A 的图例标签 (默认根据 JSON 中的 mode 与 tag 自动推断)",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default=None,
        help="文件 B 的图例标签 (默认根据 JSON 中的 mode 与 tag 自动推断)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="图表顶部主标题 (可选)",
    )

    args = parser.parse_args(argv)

    rec1 = load_record(args.file_a)
    rec2 = load_record(args.file_b)

    label1 = resolve_label(rec1, args.label_a)
    label2 = resolve_label(rec2, args.label_b)

    table_md = generate_comparison_table(rec1, rec2, label1, label2)
    print()
    print(table_md)
    print()

    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(table_md + "\n", encoding="utf-8")
        print(f"Markdown 对比表格已保存至: {args.output_markdown}")

    try:
        plot_mode_comparison(rec1, rec2, label1, label2, args.output_figure, args.title)
    except ImportError:
        print("未安装 matplotlib，跳过生成对比图（请运行 pip install -e '.[bench]'）")


if __name__ == "__main__":
    main()
