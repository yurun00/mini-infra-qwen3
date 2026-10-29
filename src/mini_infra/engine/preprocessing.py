"""输入序列规约、预算校验与左 Padding 预处理工具。"""

from collections.abc import Sequence
from typing import Any

import torch


def left_pad_sequences(
    token_sequences: Sequence[Any], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把一组非空的一维 Token 序列左 Padding 为矩形 Tensor。

    Returns:
        input_ids [B, L], attention_mask [B, L] (bool), position_ids [B, L], lengths [B]
    """
    if not token_sequences:
        raise ValueError("token_sequences 不能为空")
    lengths = []
    normalized = []
    for sequence in token_sequences:
        tensor = torch.as_tensor(sequence, dtype=torch.long, device=device)
        if tensor.ndim != 1 or tensor.numel() < 1:
            raise ValueError("每个 Prompt 必须是一维非空 Token 序列")
        normalized.append(tensor)
        lengths.append(tensor.numel())

    max_length = max(lengths)
    batch_size = len(normalized)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    for row, sequence in enumerate(normalized):
        length = sequence.numel()
        input_ids[row, max_length - length :] = sequence
        attention_mask[row, max_length - length :] = True

    # 真实 Token 的逻辑位置从 0 开始；Padding 位置记为 0（会被 mask 掉）。
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.clamp_(min=0)
    return input_ids, attention_mask, position_ids, torch.tensor(lengths, dtype=torch.long, device=device)


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


def prepare_prefill_inputs(
    input_ids: torch.Tensor | Sequence[Any],
    token_sequences: list[torch.Tensor],
    has_uniform_prompt_len: bool,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构建 Prefill 阶段直接喂给模型的初始输入张量（天然等长跳过 padding，不等长采用左 Padding）。"""
    if has_uniform_prompt_len:
        input_ids_tensor = (input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)).to(
            device=device, dtype=torch.long
        )
        batch_size, padded_prompt_length = input_ids_tensor.shape
        history_mask = torch.ones((batch_size, padded_prompt_length), dtype=torch.bool, device=device)
        position_ids = (
            torch.arange(padded_prompt_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )
        prompt_lengths = torch.full((batch_size,), padded_prompt_length, dtype=torch.long, device=device)
        return input_ids_tensor, history_mask, position_ids, prompt_lengths

    return left_pad_sequences(token_sequences, pad_token_id, device)
