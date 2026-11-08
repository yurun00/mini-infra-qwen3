"""长度与到达时间分布的展开：把声明式规格转成逐请求的具体数值（纯 CPU）。"""

from mini_infra.bench.spec.schema import (
    AlternatingLength,
    ArrivalSpec,
    ConstantArrival,
    ConstantLength,
    ExplicitArrival,
    ExplicitLength,
    IntervalArrival,
    LadderLength,
    LengthSpec,
)


def expand_length(spec: LengthSpec, count: int) -> tuple[int, ...]:
    """把长度分布展开为 count 个请求的逐请求长度。"""
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
    raise TypeError(f"未知长度规格: {spec!r}")


def expand_arrival(spec: ArrivalSpec, count: int) -> tuple[float, ...]:
    """把到达过程展开为 count 个请求的逐请求到达时间（ms）。"""
    if isinstance(spec, ConstantArrival):
        return (float(spec.value),) * count
    if isinstance(spec, ExplicitArrival):
        return tuple(float(v) for v in spec.values)
    if isinstance(spec, IntervalArrival):
        return tuple(i * float(spec.interval_ms) for i in range(count))
    raise TypeError(f"未知到达规格: {spec!r}")


__all__ = ["expand_arrival", "expand_length"]
