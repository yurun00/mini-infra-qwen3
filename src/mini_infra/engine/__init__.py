"""推理与 Serving 执行引擎域：包含模型执行器、批状态机、批处理调度循环、运行模式与结果类型。"""

from mini_infra.core import (
    BatchingPolicy,
    ChunkedPrefillConfig,
    DenseCacheConfig,
    KVCacheConfig,
    NoCacheConfig,
    PagedCacheConfig,
    PrefillFirstConfig,
    SchedulerConfig,
    TokenBudgetConfig,
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
    ExecutionMetrics,
    GenerationTrace,
    KVCacheProfile,
    LockstepMetrics,
    MemoryProfile,
    PagedCacheProfile,
    RequestAggregatedMetrics,
    ScheduleEvent,
    ScheduleMetrics,
    build_memory_profile,
    compute_execution_metrics,
    compute_lockstep_metrics,
    compute_request_aggregated_metrics,
    summarize_schedule,
)
from mini_infra.engine.preprocessing import (
    normalize_budgets,
    normalize_token_sequences,
)
from mini_infra.engine.results import (
    GenerationResult,
    LockstepResult,
    ScheduleResult,
    SerialMergeOptions,
    merge_serial_generation_results,
    pad_output_tokens,
)
from mini_infra.engine.scheduler_loop import run_scheduler
from mini_infra.engine.tokenizer import (
    PRE_TOKEN_PATTERN,
    Qwen3Tokenizer,
)

__all__ = [
    "BatchScheduleInfo",
    "BatchingMethod",
    "BatchingPolicy",
    "ChunkedPrefillConfig",
    "DenseCacheConfig",
    "EngineConfig",
    "ExecutionMetrics",
    "FeatureFlag",
    "GenerationResult",
    "GenerationTrace",
    "KVCacheConfig",
    "KVCacheProfile",
    "LockstepMetrics",
    "LockstepResult",
    "LockstepState",
    "MemoryProfile",
    "NoCacheConfig",
    "PRE_TOKEN_PATTERN",
    "PagedCacheConfig",
    "PagedCacheProfile",
    "PrefillFirstConfig",
    "Qwen3Tokenizer",
    "RequestAggregatedMetrics",
    "ScheduleEvent",
    "ScheduleMetrics",
    "ScheduleResult",
    "SchedulerConfig",
    "SerialMergeOptions",
    "TokenBudgetConfig",
    "build_memory_profile",
    "compute_execution_metrics",
    "compute_lockstep_metrics",
    "compute_request_aggregated_metrics",
    "merge_serial_generation_results",
    "normalize_budgets",
    "normalize_token_sequences",
    "pad_output_tokens",
    "run_lockstep",
    "run_scheduler",
    "summarize_schedule",
    "validate_engine_config",
]
