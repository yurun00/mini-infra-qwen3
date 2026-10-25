"""性能剖析域：Module Scope 打点、算子分类与 Trace 导出。"""

from mini_infra.profiling.analyzer import (
    KERNEL_CATEGORIES,
    kernel_category,
    merge_op_latency_breakdowns,
    parse_op_latency_breakdown,
)
from mini_infra.profiling.tracer import (
    MODULE_SCOPES,
    ProfileCollector,
    install_scope_hooks,
    profile_scope,
)
from mini_infra.profiling.types import (
    KernelInfo,
    OpLatencyBreakdown,
    TensorBoardOptions,
)

__all__ = [
    "KERNEL_CATEGORIES",
    "KernelInfo",
    "MODULE_SCOPES",
    "OpLatencyBreakdown",
    "ProfileCollector",
    "TensorBoardOptions",
    "install_scope_hooks",
    "kernel_category",
    "merge_op_latency_breakdowns",
    "parse_op_latency_breakdown",
    "profile_scope",
]
