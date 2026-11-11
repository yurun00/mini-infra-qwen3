"""Suite 批量运行（sweep）的汇总层：一次调用顺序执行多个 suite，并汇总每个 suite 的产物与对比（纯 CPU）。

执行本身由调用方注入的 `execute(suite)` 完成（它共享同一份已加载模型、正确性检查与环境信息），
这里只负责顺序驱动、计时与汇总输出，因而无需 GPU 即可单测。
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from mini_infra.bench.suite import BenchmarkSuite

SuiteT = TypeVar("SuiteT", bound=BenchmarkSuite)


@dataclass(frozen=True)
class SuiteRunSummary:
    """单个 suite 的执行摘要。"""

    name: str
    engine: str
    num_cases: int
    output: Path | None
    comparison_lines: tuple[str, ...] = ()
    wall_s: float = 0.0


def run_sweep(
    targets: Sequence[SuiteT],
    execute: Callable[[SuiteT], SuiteRunSummary],
    on_start: Callable[[int, int, SuiteT], None] | None = None,
) -> list[SuiteRunSummary]:
    """按给定顺序执行全部 suite；任一 suite 抛错即中止（不吞异常），已完成的结果已各自落盘。"""
    summaries: list[SuiteRunSummary] = []
    for index, suite in enumerate(targets, 1):
        if on_start is not None:
            on_start(index, len(targets), suite)
        start = time.perf_counter()
        summary = execute(suite)
        summaries.append(
            SuiteRunSummary(
                name=summary.name,
                engine=summary.engine,
                num_cases=summary.num_cases,
                output=summary.output,
                comparison_lines=summary.comparison_lines,
                wall_s=time.perf_counter() - start,
            )
        )
    return summaries


def format_sweep_summary(summaries: Sequence[SuiteRunSummary]) -> str:
    """汇总表 + 各 suite 的对比结论（scheduler suite 带 comparison 时）。"""
    if not summaries:
        return "sweep 未执行任何 suite。"
    name_width = max(len(s.name) for s in summaries)
    lines = [f"=== Sweep 汇总（共 {len(summaries)} 个 suite）===", ""]
    lines.append(f"{'suite':<{name_width}} | {'engine':<9} | {'cases':>5} | {'wall_s':>8} | output")
    for s in summaries:
        out = str(s.output) if s.output is not None else "-"
        lines.append(f"{s.name:<{name_width}} | {s.engine:<9} | {s.num_cases:>5} | {s.wall_s:>8.1f} | {out}")
    for s in summaries:
        if s.comparison_lines:
            lines.append("")
            lines.append(f"--- Comparisons (suite: {s.name}) ---")
            lines.extend(s.comparison_lines)
    return "\n".join(lines)


__all__ = ["SuiteRunSummary", "format_sweep_summary", "run_sweep"]
