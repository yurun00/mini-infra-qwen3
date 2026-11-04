"""输入序列规约、预算校验与左 Padding 预处理工具。"""

from collections.abc import Sequence
from typing import Any

import torch

__all__ = [
    "normalize_budgets",
    "normalize_token_sequences",
    "prepare_prompt_sequences",
]


def normalize_budgets(max_new_tokens: int | Sequence[int], batch_size: int, device: torch.device) -> torch.Tensor:
    """把标量或逐请求的输出预算统一为 [B] 的 long Tensor，并校验每项大于 0。"""
    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * batch_size
    else:
        budgets = list(max_new_tokens)
    if len(budgets) != batch_size or any(value < 1 for value in budgets):
        raise ValueError("每个请求必须提供一个大于 0 的输出预算")
    return torch.tensor(budgets, dtype=torch.long, device=device)


def normalize_token_sequences(input_ids: Sequence[Any]) -> list[torch.Tensor]:
    """统一将非 Tensor 的输入归一化为一维 Token 序列 Tensor 列表。"""
    if len(input_ids) > 0 and isinstance(input_ids[0], int):
        return [torch.as_tensor(input_ids, dtype=torch.long)]
    return [torch.as_tensor(s, dtype=torch.long) for s in input_ids]


def prepare_prompt_sequences(
    input_ids: torch.Tensor | Sequence[Any],
    device: torch.device | None,
) -> tuple[list[torch.Tensor], torch.device, bool]:
    """解析多样化的 Prompt 输入形态，规约为一维序列列表并推断目标设备。

    Returns:
        tuple[list[torch.Tensor], torch.device, bool]:
            - token_sequences (list[torch.Tensor]): 规约后的各请求一维 Token ID 张量列表。
            - device (torch.device): 推断或确认后的计算设备（CUDA 或 CPU）。
            - has_uniform_prompt_len (bool): 输入各 Prompt 长度是否天然等长（True 则后续可跳过左 Padding）。
    """
    has_uniform_prompt_len = isinstance(input_ids, torch.Tensor)
    if has_uniform_prompt_len:
        tensor = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
        token_sequences = list(tensor)
        if device is None:
            device = tensor.device
    else:
        token_sequences = normalize_token_sequences(input_ids)
        if device is None:
            device = token_sequences[0].device
    return token_sequences, device, has_uniform_prompt_len
