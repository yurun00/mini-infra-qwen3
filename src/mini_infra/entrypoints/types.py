"""用户级推理入口专用数据结构（GenerationOptions 与 PromptInput）。"""

from dataclasses import dataclass
from typing import Any

import torch

from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.inputs import PromptInput
from mini_infra.profiling import TensorBoardOptions


@dataclass(frozen=True)
class GenerationOptions:
    """文本生成可选控制项容器。"""

    eos_token_id: int | None = None
    device: torch.device | str | None = None
    stop_on_eos: bool = True
    pad_token_id: int = 0
    capture_logits: bool = False
    profile: bool = False
    top_kernels: int = 10
    tensorboard_options: TensorBoardOptions | None = None
    max_running_requests: int | None = None
    # 仅 CachingMethod.PAGED 时生效
    paged_cache_config: PagedCacheConfig | None = None
    tokenizer: Any | None = None


__all__ = [
    "GenerationOptions",
    "PromptInput",
]
