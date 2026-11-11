"""测量样本统计分布（DistributionStats，纯 CPU，零设备依赖）。"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from mini_infra.utils import percentile


@dataclass(frozen=True)
class DistributionStats:
    """一组测量样本的统计分布（均值、中位数、标准差及关键分位数）。"""

    median: float
    mean: float
    std: float
    min: float
    p90: float
    p95: float
    p99: float
    max: float

    def to_dict(self) -> dict[str, float]:
        """导出为与下游工具完全兼容的统计字典。"""
        return {
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.max,
        }

    @classmethod
    def from_samples(cls, values: Sequence[float]) -> DistributionStats:
        """计算样本序列的分位数与均值方差指标，返回 DistributionStats 结构体。"""
        if not values:
            return cls(
                median=0.0,
                mean=0.0,
                std=0.0,
                min=0.0,
                p90=0.0,
                p95=0.0,
                p99=0.0,
                max=0.0,
            )
        return cls(
            median=statistics.median(values),
            mean=statistics.fmean(values),
            std=statistics.pstdev(values) if len(values) > 1 else 0.0,
            min=min(values),
            p90=percentile(values, 90),
            p95=percentile(values, 95),
            p99=percentile(values, 99),
            max=max(values),
        )


__all__ = [
    "DistributionStats",
]
