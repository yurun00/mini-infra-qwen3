"""用户级推理入口域（Entrypoints）：提供离线文本生成与统一高层 API。"""

from mini_infra.entrypoints.offline import generate
from mini_infra.entrypoints.types import (
    AggregationOptions,
    GenerationTrace,
    OfflineGenerationOptions,
    OfflineGenerationResult,
    StepAggregatedMetrics,
    aggregate_offline_generation_results,
    compute_step_aggregated_metrics,
    to_offline_generation_result,
)

__all__ = [
    "AggregationOptions",
    "GenerationTrace",
    "OfflineGenerationOptions",
    "OfflineGenerationResult",
    "StepAggregatedMetrics",
    "aggregate_offline_generation_results",
    "compute_step_aggregated_metrics",
    "generate",
    "to_offline_generation_result",
]
