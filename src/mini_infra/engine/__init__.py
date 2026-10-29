"""推理与 Serving 执行引擎域：包含模型执行器、批状态机、批处理调度循环、运行模式与结果类型。"""

from mini_infra.device import synchronize
from mini_infra.engine.lockstep_loop import (
    LockstepState,
    run_lockstep,
)
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
    prepare_prefill_inputs,
    prepare_prompt_sequences,
)
from mini_infra.engine.runner import (
    ModelRunner,
    SlotDecodeResult,
    StepForwardResult,
)
from mini_infra.engine.scheduler_loop import run_scheduler
from mini_infra.engine.types import (
    BatchScheduleInfo,
    GenerationResult,
    LockstepResult,
    MemoryProfile,
    RequestMetrics,
    ScheduleEvent,
    ScheduleMetrics,
    SchedulePhase,
    ScheduleResult,
    SequenceOutput,
    build_memory_profile,
    pad_output_tokens,
    percentile,
    summarize_schedule,
)
from mini_infra.entrypoints.offline import generate
from mini_infra.entrypoints.types import (
    AggregationOptions,
    GenerationOptions,
    GenerationTrace,
    StepAggregatedMetrics,
    aggregate_generation_results,
    compute_step_aggregated_metrics,
)

__all__ = [
    "AggregationOptions",
    "BatchScheduleInfo",
    "BatchingMethod",
    "FeatureDimension",
    "FeatureFlag",
    "GenerationOptions",
    "GenerationResult",
    "GenerationTrace",
    "LockstepResult",
    "LockstepState",
    "MemoryProfile",
    "Mode",
    "ModelRunner",
    "RequestMetrics",
    "ScheduleEvent",
    "ScheduleMetrics",
    "SchedulePhase",
    "ScheduleResult",
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
    "pad_output_tokens",
    "percentile",
    "prepare_prefill_inputs",
    "prepare_prompt_sequences",
    "run_lockstep",
    "run_scheduler",
    "summarize_schedule",
    "synchronize",
]
