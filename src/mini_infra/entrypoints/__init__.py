"""用户级推理入口域（Entrypoints）：提供离线文本生成与统一高层 API。"""

from mini_infra.engine.metrics import (
    GenerationTrace,
    StepAggregatedMetrics,
    compute_step_aggregated_metrics,
)
from mini_infra.engine.results import (
    AggregationOptions,
    GenerationResult,
    aggregate_generation_results,
)
from mini_infra.entrypoints.offline import generate
from mini_infra.entrypoints.types import GenerationOptions, PromptInput

__all__ = [
    "AggregationOptions",
    "GenerationOptions",
    "GenerationResult",
    "GenerationTrace",
    "PromptInput",
    "StepAggregatedMetrics",
    "aggregate_generation_results",
    "compute_step_aggregated_metrics",
    "generate",
]
