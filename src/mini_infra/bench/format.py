"""基准评测通用文本格式化工具（纯 CPU，零第三方与设备依赖）。"""

from collections.abc import Sequence


def format_lengths(value: int | Sequence[int]) -> str:
    """将标量或逐请求长度序列格式化为紧凑可读展示串。"""
    if isinstance(value, Sequence) and not isinstance(value, str):
        items = list(value)
        n = len(items)
        if n > 4:
            for period in range(1, n // 2 + 1):
                if n % period == 0 and items == items[:period] * (n // period):
                    pattern = "/".join(map(str, items[:period]))
                    return f"({pattern})x{n // period}" if period > 1 else f"{pattern}x{n}"
            return f"{min(items)}..{max(items)}(μ={round(sum(items) / n)})"
        return "/".join(map(str, items))
    return str(value)


__all__ = [
    "format_lengths",
]
