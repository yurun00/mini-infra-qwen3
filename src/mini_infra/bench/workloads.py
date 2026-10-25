"""基准测试工作负载配置与序列生成辅助工具。"""

from collections.abc import Sequence

import torch


def make_sequence(
    length: int,
    vocab_size: int,
    salt: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """生成单条指定长度与词表大小的合成 prompt sequence。

    基于 CPU Generator 与固定 seed 生成确定性 randint 序列。

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
    """合成 prompt 生成函数（支持等长矩形张量或不等长变长序列列表）。

    Args:
        vocab_size: 词表大小。
        length: 单个序列长度 int，或各序列长度列表 Sequence[int]。
        batch_size: 当 length 为 int 时的批大小，默认 1。
        device: 目标设备。

    Returns:
        若 length 为 int:
            返回形状为 [batch_size, length] 的 2D 矩形 Tensor。
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


__all__ = [
    "get_mixed_output_budgets",
    "get_mixed_prompt_lengths",
    "make_prompt",
    "make_sequence",
    "make_sequences",
]
