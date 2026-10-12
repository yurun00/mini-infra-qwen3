"""读取 benchmarks/results/*.json，生成 Markdown 汇总表和趋势图。

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


def plot(records, out_path):
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
    fig.suptitle("mini-infra-qwen3 performance trend (batch=1, error bar = std)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"趋势图已保存：{out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", help="作为基准的 tag，默认最早的结果")
    parser.add_argument("--figure", type=Path, default=REPORTS_DIR / "trend.png")
    parser.add_argument("--table", type=Path, default=REPORTS_DIR / "SUMMARY.md")
    args = parser.parse_args()

    records = load_results()
    if not records:
        raise SystemExit(f"{RESULTS_DIR} 中没有结果，请先运行 run_benchmark.py")
    base = next((r for r in records if r["tag"] == args.baseline), records[0]) \
        if args.baseline else records[0]

    table = markdown_table(records, base)
    args.table.write_text("# Benchmark 汇总\n\n" + table + "\n")
    print(table)
    print(f"\n汇总表已保存：{args.table}")
    try:
        plot(records, args.figure)
    except ImportError:
        print("未安装 matplotlib，跳过绘图（pip install -r requirements-bench.txt）")


if __name__ == "__main__":
    main()
