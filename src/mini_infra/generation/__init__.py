"""生成模块：包含生成模式、批处理与 KV Cache 调度器及计时结果定义。"""

from mini_infra.generation.engine import (
    StepForwardResult,
    generate,
    synchronize,
)
from mini_infra.generation.mode import (
    BatchingMethod,
    FeatureFlag,
    Mode,
)
from mini_infra.generation.preprocessing import (
    left_pad_sequences,
)
from mini_infra.generation.state import BatchGenerationState
from mini_infra.generation.types import (
    AggregationOptions,
    BatchScheduleInfo,
    GenerationResult,
    GenerationTrace,
    MemoryProfile,
    SequenceOutput,
    StepAggregatedMetrics,
    aggregate_generation_results,
    build_memory_profile,
    compute_step_aggregated_metrics,
    percentile,
)

__all__ = [
    "AggregationOptions",
    "BatchGenerationState",
    "BatchScheduleInfo",
    "BatchingMethod",
    "FeatureFlag",
    "GenerationResult",
    "GenerationTrace",
    "MemoryProfile",
    "Mode",
    "SequenceOutput",
    "StepAggregatedMetrics",
    "StepForwardResult",
    "aggregate_generation_results",
    "build_memory_profile",
    "compute_step_aggregated_metrics",
    "generate",
    "left_pad_sequences",
    "percentile",
    "synchronize",
]
