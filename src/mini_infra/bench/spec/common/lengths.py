"""长度与到达时间分布的展开：把声明式规格转成逐请求的具体数值（纯 CPU）。"""

import math
import random

from mini_infra.bench.spec.common.schema import (
    AlternatingLength,
    ArrivalSpec,
    ConstantArrival,
    ConstantLength,
    ExplicitArrival,
    ExplicitLength,
    IntervalArrival,
    LadderLength,
    LengthSpec,
    LogUniformLength,
    PoissonArrival,
)


def sample_log_uniform_lengths(rng: random.Random, count: int, min_len: int, max_len: int) -> tuple[int, ...]:
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


def sample_poisson_arrivals(
    rng: random.Random, count: int, initial_burst: int, mean_interval_ms: float
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


def expand_length(spec: LengthSpec, count: int, rng: random.Random | None = None) -> tuple[int, ...]:
    """把长度分布展开为 count 个请求的逐请求长度；随机分布需要传入 rng（由 workload.seed 派生）。"""
    if isinstance(spec, ConstantLength):
        return (spec.value,) * count
    if isinstance(spec, ExplicitLength):
        return tuple(spec.values)
    if isinstance(spec, AlternatingLength):
        first, second = spec.values
        return tuple(first if i % 2 == 0 else second for i in range(count))
    if isinstance(spec, LadderLength):
        return (
            max(spec.floor, spec.max // 8),
            max(spec.floor, spec.max // 4),
            max(spec.floor, spec.max // 2),
            spec.max,
        )
    if isinstance(spec, LogUniformLength):
        assert rng is not None
        return sample_log_uniform_lengths(rng, count, spec.min, spec.max)
    raise TypeError(f"未知长度规格: {spec!r}")


def expand_arrival(spec: ArrivalSpec, count: int, rng: random.Random | None = None) -> tuple[float, ...]:
    """把到达过程展开为 count 个请求的逐请求到达时间（ms）；poisson 需要传入 rng。"""
    if isinstance(spec, ConstantArrival):
        return (float(spec.value),) * count
    if isinstance(spec, ExplicitArrival):
        return tuple(float(v) for v in spec.values)
    if isinstance(spec, IntervalArrival):
        return tuple(i * float(spec.interval_ms) for i in range(count))
    if isinstance(spec, PoissonArrival):
        assert rng is not None
        return sample_poisson_arrivals(rng, count, spec.initial_burst, spec.mean_interval_ms)
    raise TypeError(f"未知到达规格: {spec!r}")


__all__ = ["expand_arrival", "expand_length", "sample_log_uniform_lengths", "sample_poisson_arrivals"]
