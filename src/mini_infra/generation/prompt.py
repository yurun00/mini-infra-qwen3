"""Prompt 生成与构造工具。

负责基准测试与日常推理所需的 Prompt 构造函数，包含单序列、批序列（等长/不等长）、
多场景基准测试 Prompt（Throughput、Padding、Slots）以及 Golden 正确性门禁 Prompt。
随机数生成统一使用 CPU Generator 与固定 seed 策略，确保与 run.py 及跨环境实验结果严格对齐且可复现。
"""

from collections.abc import Sequence
from typing import Any

import torch

DEFAULT_GOLDEN_PROMPT: str = "请用一句话介绍 KV Cache。"
DEFAULT_GOLDEN_MAX_OUTPUT_TOKENS: int = 32


def make_sequence(
    length: int,
    vocab_size: int,
    salt: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """生成单条指定长度与词表大小的合成 prompt sequence。

    随机数生成方式以 run.py 为准（基于 CPU Generator 与固定 seed 生成 randint）。

    Args:
        length: 序列长度。
        vocab_size: 词表大小。
        salt: 随机盐值偏移量，默认 0。
        device: 放置张量的目标设备（可选）。

    Returns:
        一维 long 类型 Tensor，形状为 [length]。
    """
    generator = torch.Generator(device="cpu").manual_seed(1234 + length + salt)
    low = 1000 if vocab_size > 1000 else 1
    high = min(vocab_size, 100000)
    ids = torch.randint(low, high, (length,), generator=generator)
    return ids.to(device) if device is not None else ids


def make_sequences(
    lengths: Sequence[int],
    vocab_size: int,
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """批量生成指定长度列表的合成 prompt sequences（支持等长或不等长）。

    每条 sequence 按照固定 salt 规则 (23 * (index + 1)) 传入，
    确保 batch 内各序列随机内容不同且确定可复现。

    Args:
        lengths: 各序列长度列表，如 [128, 128] 或 [16, 32, 64, 128]。
        vocab_size: 词表大小。
        device: 放置张量的目标设备（可选）。

    Returns:
        包含各序列一维 Tensor 的列表。
    """
    return [
        make_sequence(length, vocab_size, salt=23 * (index + 1), device=device) for index, length in enumerate(lengths)
    ]


def make_prompt(
    vocab_size: int,
    length: int | Sequence[int],
    batch_size: int = 1,
    device: torch.device | str | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    """合成 prompt 生成函数（兼容 run.py 原有等长矩形接口，并拓展支持不等长 prompt）。

    Args:
        vocab_size: 词表大小。
        length: 单个序列长度 int，或各序列长度列表 Sequence[int]。
        batch_size: 当 length 为 int 时的批大小，默认 1。
        device: 目标设备。

    Returns:
        若 length 为 int:
            返回形状为 [batch_size, length] 的 2D 矩形 Tensor（与 run.py 原实现完全一致）。
        若 length 为 Sequence[int]:
            返回包含各序列的 1D Tensor 列表（不等长 prompt 规约形式，可直接供 generate 引擎左 Padding 处理）。
    """
    if isinstance(length, int):
        generator = torch.Generator(device="cpu").manual_seed(1234 + length)
        low = 1000 if vocab_size > 1000 else 1
        high = min(vocab_size, 100000)
        ids = torch.randint(low, high, (batch_size, length), generator=generator)
        return ids.to(device) if device is not None else ids

    return make_sequences(length, vocab_size, device=device)


def get_homogeneous_prompt_lengths(prompt_length: int, batch_size: int = 4) -> list[int]:
    """获取等长批处理的 Prompt 长度列表（用于 Throughput 与 Padding 对照组）。"""
    return [prompt_length] * batch_size


def get_mixed_prompt_lengths(max_prompt_length: int) -> list[int]:
    """获取一组不等长（阶梯稀疏）Prompt 长度列表。

    规则为：[max(1, L//8), max(1, L//4), max(1, L//2), L]。
    """
    return [
        max(1, max_prompt_length // 8),
        max(1, max_prompt_length // 4),
        max(1, max_prompt_length // 2),
        max_prompt_length,
    ]


def get_slots_prompt_lengths(prompt_length: int, num_budgets: int = 4) -> list[int]:
    """获取槽位利用率实验（Slots 场景）对应的 Prompt 长度列表。"""
    return [prompt_length] * num_budgets


def get_mixed_output_budgets(max_new_tokens: int) -> list[int]:
    """获取一组不等长（阶梯稀疏）生成 Token 预算列表（用于 Slots 变长预算对照实验）。

    规则为：[max(2, B//8), max(2, B//4), max(2, B//2), B]。
    """
    return [
        max(2, max_new_tokens // 8),
        max(2, max_new_tokens // 4),
        max(2, max_new_tokens // 2),
        max_new_tokens,
    ]


def make_homogeneous_prompts(
    prompt_length: int,
    batch_size: int,
    vocab_size: int,
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """生成 Throughput / Padding homogeneous 实验所需等长 Prompt 序列列表。"""
    return make_sequences(get_homogeneous_prompt_lengths(prompt_length, batch_size), vocab_size, device=device)


def make_mixed_prompts(
    prompt_length: int,
    vocab_size: int,
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """生成不等长 Prompt 序列列表（padding_mixed）。"""
    return make_sequences(get_mixed_prompt_lengths(prompt_length), vocab_size, device=device)


def make_slots_prompts(
    prompt_length: int,
    vocab_size: int,
    budgets: Sequence[int] = (8, 16, 32, 64),
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    """生成 Slots 实验所需 Prompt 序列列表。"""
    return make_sequences(get_slots_prompt_lengths(prompt_length, len(budgets)), vocab_size, device=device)


def make_chat_prompt(
    tokenizer: Any,
    prompt: str = DEFAULT_GOLDEN_PROMPT,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """使用 Tokenizer 编码 Chat Prompt 并返回二维 Tensor [1, L]。"""
    encoded = tokenizer.encode_chat_prompt(prompt)
    tensor = torch.tensor([encoded], dtype=torch.long)
    return tensor.to(device) if device is not None else tensor


__all__ = [
    "DEFAULT_GOLDEN_MAX_OUTPUT_TOKENS",
    "DEFAULT_GOLDEN_PROMPT",
    "get_homogeneous_prompt_lengths",
    "get_mixed_output_budgets",
    "get_mixed_prompt_lengths",
    "get_slots_prompt_lengths",
    "make_chat_prompt",
    "make_homogeneous_prompts",
    "make_mixed_prompts",
    "make_prompt",
    "make_sequence",
    "make_sequences",
    "make_slots_prompts",
]
