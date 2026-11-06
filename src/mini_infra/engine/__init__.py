"""推理与 Serving 执行引擎域：包含模型执行器、批状态机、批处理调度循环、运行模式与结果类型。"""

from mini_infra.core.types import (
    RequestMetrics,
    SequenceOutput,
)
from mini_infra.device import synchronize
from mini_infra.engine.lockstep_loop import run_lockstep
from mini_infra.engine.metrics import (
    BatchScheduleInfo,
    GenerationTrace,
    KVCacheProfile,
    MemoryProfile,
    PagedCacheProfile,
    ScheduleEvent,
    ScheduleMetrics,
    SchedulePhase,
    StepAggregatedMetrics,
    build_memory_profile,
    compute_step_aggregated_metrics,
    summarize_schedule,
)
from mini_infra.engine.mode import (
    BatchingMethod,
    CachingMethod,
    FeatureDimension,
    FeatureFlag,
    Mode,
)
from mini_infra.engine.preprocessing import (
    normalize_budgets,
    normalize_token_sequences,
)
from mini_infra.engine.results import (
    AggregationOptions,
    GenerationResult,
    ScheduleResult,
    aggregate_generation_results,
    pad_output_tokens,
)
from mini_infra.engine.scheduler_loop import run_scheduler
from mini_infra.engine.tokenizer import (
    PRE_TOKEN_PATTERN,
    Qwen3Tokenizer,
)
from mini_infra.entrypoints.offline import generate
from mini_infra.entrypoints.types import GenerationOptions
from mini_infra.utils import percentile

__all__ = [
    "AggregationOptions",
    "BatchScheduleInfo",
    "BatchingMethod",
    "FeatureDimension",
    "FeatureFlag",
    "GenerationOptions",
    "GenerationResult",
    "GenerationTrace",
    "CachingMethod",
    "KVCacheProfile",
    "MemoryProfile",
    "Mode",
    "PagedCacheProfile",
    "PRE_TOKEN_PATTERN",
    "Qwen3Tokenizer",
    "RequestMetrics",
    "ScheduleEvent",
    "ScheduleMetrics",
    "SchedulePhase",
    "ScheduleResult",
    "SequenceOutput",
    "StepAggregatedMetrics",
    "aggregate_generation_results",
    "build_memory_profile",
    "compute_step_aggregated_metrics",
    "generate",
    "normalize_budgets",
    "normalize_token_sequences",
    "pad_output_tokens",
    "percentile",
    "run_lockstep",
    "run_scheduler",
    "summarize_schedule",
    "synchronize",
]
