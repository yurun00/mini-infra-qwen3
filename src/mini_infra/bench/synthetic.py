"""基准测试合成序列生成辅助工具（基于 CPU PyTorch 随机生成器）。"""

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
    shared_prefix_len: int = 0,
) -> list[list[int]]:
    """批量生成指定长度列表的合成 prompt sequences（支持等长或不等长）。

    每条 sequence 由全体共享的 shared_prefix_len 个公共前缀 Token 与按固定 salt 规则
    (23 * (index + 1)) 生成的私有后缀拼接而成，确保 batch 内各序列后缀不同且确定可复现；
    shared_prefix_len = 0 时退化为各序列完全独立。

    Args:
        lengths: 各序列总长度列表（含公共前缀），如 [128, 128] 或 [16, 32, 64, 128]。
        vocab_size: 词表大小。
        shared_prefix_len: 全体序列共享的公共前缀长度，须不大于最短序列长度，默认 0。

    Returns:
        包含各序列 Python int 列表的列表。
    """
    prefix = make_sequence(shared_prefix_len, vocab_size, salt=-1)
    return [
        prefix
        + make_sequence(length - shared_prefix_len, vocab_size, salt=23 * (index + 1))
        for index, length in enumerate(lengths)
    ]


__all__ = [
    "make_sequence",
    "make_sequences",
]
