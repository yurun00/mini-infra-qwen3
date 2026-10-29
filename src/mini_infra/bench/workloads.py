"""基准测试工作负载配置（WorkloadSpec、标准 Lockstep/Scheduler 负载实例）与序列生成辅助工具。"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from mini_infra.core.request import RequestSpec, make_request_specs
from mini_infra.engine.mode import BatchingMethod, FeatureFlag, Mode


@dataclass(frozen=True)
class WorkloadSpec:
    """评测用例的负载规格与运行参数（Lockstep 与 Scheduler 共享，相等比较忽略 repeats 与 warmup）。"""

    mode: Mode
    batch_size: int
    prompt_len: int | tuple[int, ...]
    max_new_tokens: int | tuple[int, ...]
    repeats: int = field(default=1, compare=False)
    warmup: int = field(default=0, compare=False)
    max_running_requests: int | None = None
    arrival_times_ms: float | tuple[float, ...] | None = None


# 标准齐步（Lockstep）基准工作负载：4 条阶梯变长 Prompt 与阶梯变长输出预算请求
LOCKSTEP_WORKLOAD = WorkloadSpec(
    mode=Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.FIXED),
    batch_size=4,
    prompt_len=(64, 128, 256, 512),
    max_new_tokens=(16, 32, 64, 128),
    repeats=10,
    warmup=3,
)

# 标准调度器（Scheduler）基准工作负载：16 条请求（4 个波次 × 并发槽位 4），变长 Prompt、交错长短输出预算与流式错峰到达
SCHEDULER_WORKLOAD = WorkloadSpec(
    mode=Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.CONTINUOUS),
    batch_size=16,
    prompt_len=(64, 128, 256, 512) * 4,
    max_new_tokens=(16, 64, 32, 128) * 4,
    repeats=10,
    warmup=3,
    max_running_requests=4,
    arrival_times_ms=(0.0, 0.0, 0.0, 0.0) + tuple(80.0 * i for i in range(1, 13)),
)


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


def make_variable_prompt_lengths(max_prompt_length: int) -> list[int]:
    """生成一组变长不等长（阶梯稀疏）Prompt 长度列表。

    规则为：[max(1, L//8), max(1, L//4), max(1, L//2), L]。
    """
    return [
        max(1, max_prompt_length // 8),
        max(1, max_prompt_length // 4),
        max(1, max_prompt_length // 2),
        max_prompt_length,
    ]


def make_variable_output_budgets(max_new_tokens: int) -> list[int]:
    """生成一组变长不等长（阶梯稀疏）Token 预算列表（用于 Slots 变长预算对照实验）。

    规则为：[max(2, B//8), max(2, B//4), max(2, B//2), B]。
    """
    return [
        max(2, max_new_tokens // 8),
        max(2, max_new_tokens // 4),
        max(2, max_new_tokens // 2),
        max_new_tokens,
    ]


def make_arrival_times(count: int, interval_ms: float) -> list[float]:
    """生成等间隔到达时间列表（单位 ms）。"""
    if count < 1:
        raise ValueError("count 必须大于 0")
    if interval_ms < 0:
        raise ValueError("interval_ms 不能小于 0")
    return [i * float(interval_ms) for i in range(count)]


def make_alternating_budgets(count: int, short_budget: int, long_budget: int) -> list[int]:
    """生成交替（短/长）输出预算列表。"""
    if count < 1:
        raise ValueError("count 必须大于 0")
    if short_budget < 1 or long_budget < 1:
        raise ValueError("输出预算必须大于 0")
    return [short_budget if i % 2 == 0 else long_budget for i in range(count)]


def build_scheduler_request_specs(
    workload: WorkloadSpec,
    vocab_size: int,
    device: torch.device | str | None = None,
) -> list[RequestSpec]:
    """根据 WorkloadSpec 构建 Scheduler 评测所需的 RequestSpec 列表。"""
    n = workload.batch_size
    lengths = [workload.prompt_len] * n if isinstance(workload.prompt_len, int) else list(workload.prompt_len)
    budgets = (
        [workload.max_new_tokens] * n if isinstance(workload.max_new_tokens, int) else list(workload.max_new_tokens)
    )
    if workload.arrival_times_ms is None:
        arrivals = [0.0] * n
    elif isinstance(workload.arrival_times_ms, int | float):
        arrivals = [float(workload.arrival_times_ms)] * n
    else:
        arrivals = [float(a) for a in workload.arrival_times_ms]

    if len(lengths) != n or len(budgets) != n or len(arrivals) != n:
        raise ValueError(
            f"WorkloadSpec 长度不匹配: batch_size={n}, len(prompt_len)={len(lengths)}, "
            f"len(max_new_tokens)={len(budgets)}, len(arrival_times_ms)={len(arrivals)}"
        )
    sequences = [t.tolist() for t in make_sequences(lengths, vocab_size, device=device)]
    return make_request_specs(sequences, budgets, arrivals)


__all__ = [
    "LOCKSTEP_WORKLOAD",
    "SCHEDULER_WORKLOAD",
    "WorkloadSpec",
    "build_scheduler_request_specs",
    "make_alternating_budgets",
    "make_arrival_times",
    "make_prompt",
    "make_sequence",
    "make_sequences",
    "make_variable_output_budgets",
    "make_variable_prompt_lengths",
]
