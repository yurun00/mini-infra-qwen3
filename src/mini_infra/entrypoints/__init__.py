"""用户级推理入口域（Entrypoints）：提供离线文本生成与统一高层 API。"""

from mini_infra.entrypoints.offline import generate, resolve_paged_cache
from mini_infra.entrypoints.types import GenerationOptions, PromptInput

__all__ = [
    "GenerationOptions",
    "PromptInput",
    "generate",
    "resolve_paged_cache",
]
