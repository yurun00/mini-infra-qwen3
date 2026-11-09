"""基准测试合成序列生成与到达时间序列辅助工具。"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def make_sequence(
    length: int,
    vocab_size: int,
    salt: int = 0,
) -> list[int]:
    """生成单条指定长度与词表大小的合成 prompt sequence。

    基于 CPU Generator 与固定 seed 生成确定性 randint 序列。

    Args:
        length: 序列长度。
        vocab_size: 词表大小。
        salt: 随机盐值偏移量，默认 0。

    Returns:
        包含 Python int 的序列列表，长度为 length。
    """
    generator = torch.Generator(device="cpu").manual_seed(1234 + length + salt)
    low = 1000 if vocab_size > 1000 else 1
    high = min(vocab_size, 100000)
    ids = torch.randint(low, high, (length,), generator=generator)
    return ids.tolist()


def make_sequences(
    lengths: Sequence[int],
    vocab_size: int,
) -> list[list[int]]:
    """批量生成指定长度列表的合成 prompt sequences（支持等长或不等长）。

    每条 sequence 按照固定 salt 规则 (23 * (index + 1)) 传入，
    确保 batch 内各序列随机内容不同且确定可复现。

    Args:
        lengths: 各序列长度列表，如 [128, 128] 或 [16, 32, 64, 128]。
        vocab_size: 词表大小。

    Returns:
        包含各序列 Python int 列表的列表。
    """
    return [make_sequence(length, vocab_size, salt=23 * (index + 1)) for index, length in enumerate(lengths)]


def make_prompt(
    vocab_size: int,
    length: int | Sequence[int],
    batch_size: int = 1,
) -> list[list[int]]:
    """合成 prompt 生成函数（支持等长或不等长变长序列列表）。

    Args:
        vocab_size: 词表大小。
        length: 单个序列长度 int，或各序列长度列表 Sequence[int]。
        batch_size: 当 length 为 int 时的批大小，默认 1。

    Returns:
        嵌套 Python int 列表，形状为 [batch_size, length] 或对应不等长列表。
    """
    if isinstance(length, int):
        generator = torch.Generator(device="cpu").manual_seed(1234 + length)
        low = 1000 if vocab_size > 1000 else 1
        high = min(vocab_size, 100000)
        ids = torch.randint(low, high, (batch_size, length), generator=generator)
        return ids.tolist()

    return make_sequences(length, vocab_size)


def make_arrival_times(count: int, interval_ms: float) -> list[float]:
    """生成等间隔到达时间列表（单位 ms）。"""
    if count < 1:
        raise ValueError("count 必须大于 0")
    if interval_ms < 0:
        raise ValueError("interval_ms 不能小于 0")
    return [i * float(interval_ms) for i in range(count)]


__all__ = [
    "make_arrival_times",
    "make_prompt",
    "make_sequence",
    "make_sequences",
]
