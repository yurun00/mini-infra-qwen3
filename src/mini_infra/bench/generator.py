"""generator.py 别名模块，重导出 generation.py 的所有接口。"""

from mini_infra.bench.generation import (
    GenerationResult,
    StepForwardResult,
    _build_generation_result,
    _timed_forward,
    generate,
    generate_kv_cache,
    generate_no_cache,
    percentile,
    synchronize,
)

__all__ = [
    "GenerationResult",
    "StepForwardResult",
    "generate",
    "generate_no_cache",
    "generate_kv_cache",
    "percentile",
    "synchronize",
    "_timed_forward",
    "_build_generation_result",
]
