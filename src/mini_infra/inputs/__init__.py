"""输入协议共享类型（纯 Python，无 Tensor、无 tokenize），供 entrypoints 与 engine 共同依赖。"""

from collections.abc import Sequence

PromptInput = str | Sequence[int]

__all__ = ["PromptInput"]
