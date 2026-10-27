"""用户级推理入口域（Entrypoints）：提供离线文本生成与统一高层 API。"""

from mini_infra.entrypoints.offline import OfflineGenerationOptions, generate

__all__ = [
    "OfflineGenerationOptions",
    "generate",
]
