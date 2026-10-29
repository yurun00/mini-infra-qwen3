"""CLI 子命令：mini-infra bench plot (history / compare)

性能可视化与对比图表绘制入口。
支持按 lockstep / scheduler 子目录分别读取 benchmarks/result-history/{lockstep,scheduler}
并生成对应的 benchmarks/reports/{lockstep,scheduler}/ 下的报告与演进剖析图表。
"""

import argparse
from pathlib import Path
from typing import Any

from mini_infra.bench.common import (
    LOCKSTEP_REPORTS_DIR,
    LOCKSTEP_RESULT_HISTORY_DIR,
    REPORTS_DIR,
    RESULT_HISTORY_DIR,
    SCHEDULER_REPORTS_DIR,
    SCHEDULER_RESULT_HISTORY_DIR,
)
from mini_infra.bench.plotting.compare import (
    generate_comparison_table,
    load_record,
    plot_mode_comparison,
    resolve_label,
)
from mini_infra.bench.plotting.history import (
    generate_optimization_guidance,
    load_results,
    markdown_table,
    plot_evolution_and_scaling,
    plot_profiling_and_optimization,
)


def add_history_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument(
        "--type",
        dest="benchmark_type",
        choices=["all", "lockstep", "scheduler"],
        default="all",
        help="要生成的历史演进报告类型（默认: all，同时生成 lockstep 与 scheduler 报告）",
    )
    p.add_argument("--baseline", help="作为基准的 tag，默认最早的结果")
    p.add_argument("--target", help="作为当前深度剖析对象的 tag，默认最新的结果")
    p.add_argument("--evolution-figure", type=Path, default=None, help="自定义演进趋势图输出路径")
    p.add_argument("--profiling-figure", type=Path, default=None, help="自定义剖析与优化图输出路径")
    p.add_argument("--summary-path", type=Path, default=None, help="自定义汇总报告 SUMMARY.md 输出路径")
    return p


def add_compare_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("file_a", type=Path, help="待比较的第一个 benchmark JSON 文件路径 (通常为 baseline)")
    p.add_argument("file_b", type=Path, help="待比较的第二个 benchmark JSON 文件路径 (通常为实验组/优化版)")
    p.add_argument(
        "-o",
        "--output-figure",
        type=Path,
        default=Path("benchmark_comparison.png"),
        help="对比图表输出路径 (PNG/PDF，默认: ./benchmark_comparison.png)",
    )
    p.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="可选：将生成的 Markdown 详细对比表格保存至独立文件",
    )
    p.add_argument(
        "--label-a",
        type=str,
        default=None,
        help="文件 A 的图例标签 (默认根据 JSON 中的 mode 与 tag 自动推断)",
    )
    p.add_argument(
        "--label-b",
        type=str,
        default=None,
        help="文件 B 的图例标签 (默认根据 JSON 中的 mode 与 tag 自动推断)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="图表顶部主标题 (可选)",
    )
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 plot 组及 history/compare 子命令。"""
    plot_parser = subparsers.add_parser("plot", help="性能图表绘制与对比分析")
    plot_subs = plot_parser.add_subparsers(dest="plot_subcommand", required=True)

    history_parser = plot_subs.add_parser("history", help="读取 result-history/ 生成跨版本演化与深度剖析图表")
    add_history_args(history_parser)
    history_parser.set_defaults(func=handle_history)

    compare_parser = plot_subs.add_parser("compare", help="对比两个独立 benchmark JSON 结果并绘制多维对比图")
    add_compare_args(compare_parser)
    compare_parser.set_defaults(func=handle_compare)
    return plot_parser


def _load_domain_results(hist_dir: Path) -> list[Any]:
    try:
        return load_results(hist_dir)
    except TypeError:
        return load_results()


def _render_single_history_report(
    records: list[Any],
    domain_title: str,
    summary_path: Path,
    evolution_figure: Path,
    profiling_figure: Path,
    baseline_tag: str | None = None,
    target_tag: str | None = None,
) -> None:
    base = next((r for r in records if r["tag"] == baseline_tag), records[0]) if baseline_tag else records[0]
    target = next((r for r in records if r["tag"] == target_tag), records[-1]) if target_tag else records[-1]

    table = markdown_table(records, base)
    guidance = generate_optimization_guidance(target)

    content = [
        f"# {domain_title}",
        "",
        table,
        "",
        "## 一、跨版本演化与扩展趋势 (Evolution & Scaling Across Releases)",
        "",
        f"![Evolution & Scaling](./{evolution_figure.name})",
        "",
        guidance,
        "",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"=== {domain_title} ===")
    print(table)
    print(f"\n汇总报告已保存：{summary_path}")

    try:
        plot_evolution_and_scaling(records, base, evolution_figure)
        plot_profiling_and_optimization(records, target, profiling_figure)
    except ImportError:
        print("未安装 matplotlib，跳过绘图（pip install -e '.[bench]'）")


def handle_history(args: argparse.Namespace) -> int:
    btype = getattr(args, "benchmark_type", "all")
    targets: list[tuple[str, str, Path, Path]] = []
    if btype in ("all", "lockstep"):
        targets.append(
            (
                "lockstep",
                "Lockstep 齐步推理基准演进与性能剖析报告",
                LOCKSTEP_RESULT_HISTORY_DIR,
                LOCKSTEP_REPORTS_DIR,
            )
        )
    if btype in ("all", "scheduler"):
        targets.append(
            (
                "scheduler",
                "Scheduler 队列调度基准演进与性能剖析报告",
                SCHEDULER_RESULT_HISTORY_DIR,
                SCHEDULER_REPORTS_DIR,
            )
        )

    rendered_any = False
    for domain_key, domain_title, hist_dir, rep_dir in targets:
        records = _load_domain_results(hist_dir)
        if not records:
            if btype != "all":
                print(f"{hist_dir} 中没有结果，请先运行 mini-infra bench {domain_key} case")
                return 1
            continue

        rendered_any = True
        summary_path = args.summary_path or (rep_dir / "SUMMARY.md")
        evolution_figure = args.evolution_figure or (rep_dir / "evolution_and_scaling.png")
        profiling_figure = args.profiling_figure or (rep_dir / "profiling_and_optimization.png")
        _render_single_history_report(
            records=records,
            domain_title=domain_title,
            summary_path=summary_path,
            evolution_figure=evolution_figure,
            profiling_figure=profiling_figure,
            baseline_tag=args.baseline,
            target_tag=args.target,
        )

    if not rendered_any:
        print(f"{RESULT_HISTORY_DIR} 中没有结果，请先运行 mini-infra bench lockstep case 或 scheduler case")
        return 1

    return 0


def handle_compare(args: argparse.Namespace) -> int:
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
    return 0


__all__ = [
    "LOCKSTEP_REPORTS_DIR",
    "LOCKSTEP_RESULT_HISTORY_DIR",
    "REPORTS_DIR",
    "RESULT_HISTORY_DIR",
    "SCHEDULER_REPORTS_DIR",
    "SCHEDULER_RESULT_HISTORY_DIR",
    "add_compare_args",
    "add_history_args",
    "handle_compare",
    "handle_history",
    "register_parser",
]
