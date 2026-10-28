"""用户级推理入口域（Entrypoints）：提供离线文本生成与统一高层 API。"""

from mini_infra.entrypoints.offline import generate
from mini_infra.entrypoints.types import (
    AggregationOptions,
    GenerationOptions,
    GenerationResult,
    GenerationTrace,
    StepAggregatedMetrics,
    aggregate_generation_results,
    compute_step_aggregated_metrics,
)

__all__ = [
    "AggregationOptions",
    "GenerationOptions",
    "GenerationResult",
    "GenerationTrace",
    "StepAggregatedMetrics",
    "aggregate_generation_results",
    "compute_step_aggregated_metrics",
    "generate",
]
