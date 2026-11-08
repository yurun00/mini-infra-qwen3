"""基准测试工作负载配置（WorkloadSpec、多规模伪随机真实 Traffic 负载实例）与序列生成辅助工具。"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import torch

from mini_infra.core.request import RequestSpec, make_request_specs
from mini_infra.engine.mode import BatchingMethod, CachingMethod, Mode

WorkloadScale = Literal["small", "medium", "large"]
WORKLOAD_SCALES: tuple[WorkloadScale, ...] = ("small", "medium", "large")
DEFAULT_WORKLOAD_SEED: int = 42

_SCALE_SEED_OFFSET: dict[str, int] = {
    "small": 101,
    "medium": 202,
    "large": 303,
}


@dataclass(frozen=True)
class WorkloadSpec:
    """评测用例的负载规格与运行参数（Lockstep 与 Scheduler 共享，相等比较忽略 repeats、warmup 与 seed）。"""

    mode: Mode
    num_total_requests: int
    prompt_len: int | tuple[int, ...]
    max_new_tokens: int | tuple[int, ...]
    repeats: int = field(default=1, compare=False)
    warmup: int = field(default=0, compare=False)
    seed: int | None = field(default=None, compare=False)
    scale: str | None = None
    max_running_requests: int | None = None
    arrival_times_ms: float | tuple[float, ...] | None = None


def _sample_log_uniform_lengths(
    rng: random.Random,
    count: int,
    min_len: int,
    max_len: int,
) -> tuple[int, ...]:
    """采用分层对数均匀分布（Stratified Log-Uniform）+ 随机打乱采样请求长度序列。

    模拟真实线上流量中“短/中请求居多、长尾请求并存”的偏态分布特征，
    同时保证在小样本（如 batch_size=2/4）下依然覆盖完整跨度，且结果由 seed 严格确定。
    """
    log_min = math.log(max(1, min_len))
    log_max = math.log(max(min_len + 1, max_len))
    values: list[int] = []
    for i in range(count):
        u_low = i / count
        u_high = (i + 1) / count
        u = rng.uniform(u_low, u_high)
        val = int(round(math.exp(log_min + u * (log_max - log_min))))
        values.append(max(min_len, min(max_len, val)))
    rng.shuffle(values)
    return tuple(values)


def _sample_poisson_arrivals(
    rng: random.Random,
    count: int,
    initial_burst: int,
    mean_interval_ms: float,
) -> tuple[float, ...]:
    """生成带初始并发突发的泊松过程（指数分布到达间隔）请求到达时间序列（单位 ms）。"""
    arrivals: list[float] = []
    t = 0.0
    burst = max(1, min(count, initial_burst))
    for i in range(count):
        if i < burst:
            arrivals.append(0.0)
        else:
            dt = rng.expovariate(1.0 / mean_interval_ms)
            dt = max(5.0, min(mean_interval_ms * 3.0, dt))
            t += dt
            arrivals.append(round(t, 1))
    return tuple(arrivals)


_LOCKSTEP_SCALE_CONFIGS: dict[WorkloadScale, dict[str, int]] = {
    "small": {"num_total_requests": 4, "min_prompt": 16, "max_prompt": 64, "min_out": 8, "max_out": 32},
    "medium": {"num_total_requests": 8, "min_prompt": 64, "max_prompt": 256, "min_out": 32, "max_out": 96},
    "large": {"num_total_requests": 16, "min_prompt": 256, "max_prompt": 1024, "min_out": 64, "max_out": 192},
}

_SCHEDULER_SCALE_CONFIGS: dict[WorkloadScale, dict[str, int | float]] = {
    "small": {
        "num_total_requests": 16,
        "max_running_requests": 4,
        "min_prompt": 16,
        "max_prompt": 64,
        "min_out": 8,
        "max_out": 32,
        "initial_burst": 4,
        "mean_interval_ms": 55.0,
    },
    "medium": {
        "num_total_requests": 32,
        "max_running_requests": 8,
        "min_prompt": 64,
        "max_prompt": 256,
        "min_out": 32,
        "max_out": 96,
        "initial_burst": 8,
        "mean_interval_ms": 110.0,
    },
    "large": {
        "num_total_requests": 64,
        "max_running_requests": 16,
        "min_prompt": 256,
        "max_prompt": 1024,
        "min_out": 64,
        "max_out": 192,
        "initial_burst": 16,
        "mean_interval_ms": 180.0,
    },
}


def make_lockstep_workload(
    scale: WorkloadScale = "medium",
    *,
    seed: int = DEFAULT_WORKLOAD_SEED,
    mode: Mode | None = None,
    warmup: int = 2,
    repeats: int = 5,
) -> WorkloadSpec:
    """根据指定 scale (small/medium/large) 与 seed 生成确定性的伪随机 Lockstep 基准负载。"""
    if scale not in _LOCKSTEP_SCALE_CONFIGS:
        raise ValueError(f"未知 scale: {scale!r}，可选: {WORKLOAD_SCALES}")
    cfg = _LOCKSTEP_SCALE_CONFIGS[scale]
    rng = random.Random(seed * 1000 + _SCALE_SEED_OFFSET[scale])
    reqs = cfg["num_total_requests"]
    prompt_lens = _sample_log_uniform_lengths(rng, reqs, cfg["min_prompt"], cfg["max_prompt"])
    out_budgets = _sample_log_uniform_lengths(rng, reqs, cfg["min_out"], cfg["max_out"])
    target_mode = mode or Mode(CachingMethod.DENSE, batching_method=BatchingMethod.FIXED)
    return WorkloadSpec(
        mode=target_mode,
        num_total_requests=reqs,
        prompt_len=prompt_lens,
        max_new_tokens=out_budgets,
        repeats=repeats,
        warmup=warmup,
        seed=seed,
        scale=scale,
    )


def make_scheduler_workload(
    scale: WorkloadScale = "medium",
    *,
    seed: int = DEFAULT_WORKLOAD_SEED,
    mode: Mode | None = None,
    warmup: int = 2,
    repeats: int = 5,
) -> WorkloadSpec:
    """根据指定 scale (small/medium/large) 与 seed 生成确定性的伪随机 Scheduler 基准负载。"""
    if scale not in _SCHEDULER_SCALE_CONFIGS:
        raise ValueError(f"未知 scale: {scale!r}，可选: {WORKLOAD_SCALES}")
    cfg = _SCHEDULER_SCALE_CONFIGS[scale]
    rng = random.Random(seed * 1000 + 500 + _SCALE_SEED_OFFSET[scale])
    reqs = int(cfg["num_total_requests"])
    max_running = int(cfg["max_running_requests"])
    prompt_lens = _sample_log_uniform_lengths(rng, reqs, int(cfg["min_prompt"]), int(cfg["max_prompt"]))
    out_budgets = _sample_log_uniform_lengths(rng, reqs, int(cfg["min_out"]), int(cfg["max_out"]))
    arrivals = _sample_poisson_arrivals(rng, reqs, int(cfg["initial_burst"]), float(cfg["mean_interval_ms"]))
    target_mode = mode or Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
    return WorkloadSpec(
        mode=target_mode,
        num_total_requests=reqs,
        prompt_len=prompt_lens,
        max_new_tokens=out_budgets,
        repeats=repeats,
        warmup=warmup,
        seed=seed,
        scale=scale,
        max_running_requests=max_running,
        arrival_times_ms=arrivals,
    )


# 标准齐步（Lockstep）多规模基准工作负载（small / medium / large）
BENCH_LOCKSTEP_WORKLOAD: dict[WorkloadScale, WorkloadSpec] = {
    s: make_lockstep_workload(s, seed=DEFAULT_WORKLOAD_SEED) for s in WORKLOAD_SCALES
}

# 标准调度器（Scheduler）多规模基准工作负载（small / medium / large）
BENCH_SCHEDULER_WORKLOAD: dict[WorkloadScale, WorkloadSpec] = {
    s: make_scheduler_workload(s, seed=DEFAULT_WORKLOAD_SEED) for s in WORKLOAD_SCALES
}


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


def build_scheduler_request_specs(
    workload: WorkloadSpec,
    vocab_size: int,
) -> list[RequestSpec]:
    """根据 WorkloadSpec 构建 Scheduler 评测所需的 RequestSpec 列表。"""
    n = workload.num_total_requests
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
            f"WorkloadSpec 长度不匹配: num_total_requests={n}, len(prompt_len)={len(lengths)}, "
            f"len(max_new_tokens)={len(budgets)}, len(arrival_times_ms)={len(arrivals)}"
        )
    sequences = make_sequences(lengths, vocab_size)
    return make_request_specs(sequences, budgets, arrivals)


__all__ = [
    "BENCH_LOCKSTEP_WORKLOAD",
    "BENCH_SCHEDULER_WORKLOAD",
    "DEFAULT_WORKLOAD_SEED",
    "WORKLOAD_SCALES",
    "WorkloadScale",
    "WorkloadSpec",
    "build_scheduler_request_specs",
    "make_arrival_times",
    "make_lockstep_workload",
    "make_prompt",
    "make_scheduler_workload",
    "make_sequence",
    "make_sequences",
]
