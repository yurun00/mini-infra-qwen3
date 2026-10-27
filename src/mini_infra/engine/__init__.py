"""推理与 Serving 执行引擎域：包含模型执行器、批状态机、批处理调度循环、运行模式与结果类型。"""

from mini_infra.device import synchronize
from mini_infra.engine.batch_state import BatchGenerationState
from mini_infra.engine.mode import (
    BatchingMethod,
    FeatureDimension,
    FeatureFlag,
    Mode,
)
from mini_infra.engine.preprocessing import (
    left_pad_sequences,
    normalize_budgets,
    normalize_token_sequences,
)
from mini_infra.engine.runner import (
    ModelRunner,
    SlotDecodeResult,
    StepForwardResult,
)
from mini_infra.engine.scheduler_loop import run_scheduler
from mini_infra.engine.types import (
    AggregationOptions,
    BatchScheduleInfo,
    GenerationResult,
    GenerationTrace,
    MemoryProfile,
    OfflineGenerationOptions,
    SequenceOutput,
    StepAggregatedMetrics,
    aggregate_generation_results,
    build_memory_profile,
    compute_step_aggregated_metrics,
    percentile,
    to_generation_result,
)
from mini_infra.entrypoints.offline import generate

__all__ = [
    "AggregationOptions",
    "BatchGenerationState",
    "BatchScheduleInfo",
    "BatchingMethod",
    "FeatureDimension",
    "FeatureFlag",
    "GenerationResult",
    "GenerationTrace",
    "MemoryProfile",
    "Mode",
    "ModelRunner",
    "OfflineGenerationOptions",
    "SequenceOutput",
    "SlotDecodeResult",
    "StepAggregatedMetrics",
    "StepForwardResult",
    "aggregate_generation_results",
    "build_memory_profile",
    "compute_step_aggregated_metrics",
    "generate",
    "left_pad_sequences",
    "normalize_budgets",
    "normalize_token_sequences",
    "percentile",
    "run_scheduler",
    "synchronize",
    "to_generation_result",
]
