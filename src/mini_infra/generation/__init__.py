"""生成模块：包含生成模式、批处理与 KV Cache 调度器及计时结果定义。"""

from mini_infra.generation.engine import (
    StepForwardResult,
    _resolve_step_latencies,
    _timed_forward,
    generate,
    synchronize,
)
from mini_infra.generation.mode import (
    BatchingMethod,
    FeatureFlag,
    Mode,
)
from mini_infra.generation.preprocessing import (
    _normalize_budgets,
    _normalize_token_sequences,
    left_pad_sequences,
)
from mini_infra.generation.prompt import (
    DEFAULT_GOLDEN_MAX_OUTPUT_TOKENS,
    DEFAULT_GOLDEN_PROMPT,
    get_homogeneous_prompt_lengths,
    get_mixed_prompt_lengths,
    get_slots_prompt_lengths,
    make_chat_prompt,
    make_homogeneous_prompts,
    make_mixed_prompts,
    make_prompt,
    make_sequence,
    make_sequences,
    make_slots_prompts,
)
from mini_infra.generation.result import GenerationResult, percentile
from mini_infra.generation.state import BatchGenerationState

__all__ = [
    "BatchGenerationState",
    "BatchingMethod",
    "DEFAULT_GOLDEN_MAX_OUTPUT_TOKENS",
    "DEFAULT_GOLDEN_PROMPT",
    "FeatureFlag",
    "GenerationResult",
    "Mode",
    "StepForwardResult",
    "_normalize_budgets",
    "_normalize_token_sequences",
    "_resolve_step_latencies",
    "_timed_forward",
    "generate",
    "get_homogeneous_prompt_lengths",
    "get_mixed_prompt_lengths",
    "get_slots_prompt_lengths",
    "left_pad_sequences",
    "make_chat_prompt",
    "make_homogeneous_prompts",
    "make_mixed_prompts",
    "make_prompt",
    "make_sequence",
    "make_sequences",
    "make_slots_prompts",
    "percentile",
    "synchronize",
]
