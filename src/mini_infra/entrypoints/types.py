"""用户级推理入口专用数据结构（GenerationOptions）。"""

from dataclasses import dataclass

import torch

from mini_infra.profiling import TensorBoardOptions


@dataclass(frozen=True)
class GenerationOptions:
    """文本生成可选控制项容器。"""

    eos_token_id: int | None = None
    device: torch.device | None = None
    stop_on_eos: bool = True
    pad_token_id: int = 0
    capture_logits: bool = False
    profile: bool = False
    top_kernels: int = 10
    tensorboard_options: TensorBoardOptions | None = None
    max_running_requests: int | None = None


__all__ = [
    "GenerationOptions",
]
