"""通用工具模块：包含分位数计算等基础数学与统计辅助函数。"""

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], percent: int | float) -> float:
    """使用 nearest-rank 定义计算小样本延迟分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * percent / 100))
    return ordered[min(rank, len(ordered)) - 1]
