"""Benchmark 可视化与图表分析模块。"""

from mini_infra.bench.plotting.compare import (
    generate_comparison_table,
    load_record,
    plot_mode_comparison,
    resolve_label,
)
from mini_infra.bench.plotting.history import (
    generate_optimization_guidance,
    label,
    load_results,
    markdown_table,
    plot_evolution_and_scaling,
    plot_profiling_and_optimization,
)
from mini_infra.bench.plotting.tidy import (
    LOCKSTEP_ONLY_PLOT_SPECS,
    SCHEDULER_ONLY_PLOT_SPECS,
    SHARED_PLOT_SPECS,
    MetricPlotSpec,
    records_to_wide_df,
    resolve_plot_specs,
    resolve_record_label,
    wide_to_tidy_df,
)

__all__ = [
    "LOCKSTEP_ONLY_PLOT_SPECS",
    "MetricPlotSpec",
    "SCHEDULER_ONLY_PLOT_SPECS",
    "SHARED_PLOT_SPECS",
    "generate_comparison_table",
    "generate_optimization_guidance",
    "label",
    "load_record",
    "load_results",
    "markdown_table",
    "plot_evolution_and_scaling",
    "plot_mode_comparison",
    "plot_profiling_and_optimization",
    "records_to_wide_df",
    "resolve_label",
    "resolve_plot_specs",
    "resolve_record_label",
    "wide_to_tidy_df",
]
