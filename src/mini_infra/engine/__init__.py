"""推理与 Serving 执行引擎域：包含模型执行器、批状态机、批处理调度循环、运行模式与结果类型。"""

from mini_infra.core import (
    DenseCacheConfig,
    KVCacheConfig,
    NoCacheConfig,
    PagedCacheConfig,
)
from mini_infra.engine.config import (
    BatchingMethod,
    EngineConfig,
    FeatureFlag,
    validate_engine_config,
)
from mini_infra.engine.lockstep_loop import LockstepState, run_lockstep
from mini_infra.engine.metrics import (
    BatchScheduleInfo,
    GenerationTrace,
    KVCacheProfile,
    MemoryProfile,
    PagedCacheProfile,
    ScheduleEvent,
    ScheduleMetrics,
    StepAggregatedMetrics,
    build_memory_profile,
    compute_step_aggregated_metrics,
    summarize_schedule,
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

__all__ = [
    "AggregationOptions",
    "BatchScheduleInfo",
    "BatchingMethod",
    "DenseCacheConfig",
    "EngineConfig",
    "FeatureFlag",
    "GenerationResult",
    "GenerationTrace",
    "KVCacheConfig",
    "KVCacheProfile",
    "LockstepState",
    "MemoryProfile",
    "NoCacheConfig",
    "PRE_TOKEN_PATTERN",
    "PagedCacheConfig",
    "PagedCacheProfile",
    "Qwen3Tokenizer",
    "ScheduleEvent",
    "ScheduleMetrics",
    "ScheduleResult",
    "StepAggregatedMetrics",
    "aggregate_generation_results",
    "build_memory_profile",
    "compute_step_aggregated_metrics",
    "normalize_budgets",
    "normalize_token_sequences",
    "pad_output_tokens",
    "run_lockstep",
    "run_scheduler",
    "summarize_schedule",
    "validate_engine_config",
]
