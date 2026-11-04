"""输入序列规约与输出预算校验工具（纯 Python CPU 实现）。"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from mini_infra.inputs import PromptInput

if TYPE_CHECKING:
    from mini_infra.engine.tokenizer import Qwen3Tokenizer

__all__ = [
    "normalize_budgets",
    "normalize_token_sequences",
]


def normalize_budgets(
    max_new_tokens: int | Sequence[int],
    batch_size: int,
) -> list[int]:
    """把标量或逐请求的输出预算统一为长度为 batch_size 的整数列表，并校验每项大于 0。"""
    budgets = [max_new_tokens] * batch_size if isinstance(max_new_tokens, int) else [int(v) for v in max_new_tokens]
    if len(budgets) != batch_size or any(value < 1 for value in budgets):
        raise ValueError("每个请求必须提供一个大于 0 的输出预算")
    return budgets


def _encode_single_prompt(
    prompt: PromptInput,
    tokenizer: "Qwen3Tokenizer | None" = None,
) -> tuple[int, ...]:
    """将单条 Prompt（文本或一维整数序列）规约为不可变整数元组。"""
    if isinstance(prompt, str):
        if tokenizer is None:
            raise ValueError("输入包含文本 Prompt 时必须提供 tokenizer")
        return tuple(tokenizer.encode(prompt))
    return tuple(int(x) for x in prompt)


def normalize_token_sequences(
    prompts: PromptInput | Sequence[PromptInput],
    tokenizer: "Qwen3Tokenizer | None" = None,
) -> tuple[list[tuple[int, ...]], bool]:
    """统一将单条或多条 Prompt 规约为一维整数元组序列列表，并判定是否天然等长。"""
    if isinstance(prompts, str):
        return [_encode_single_prompt(prompts, tokenizer)], True

    prompt_list = list(prompts)
    if isinstance(prompt_list[0], int):
        return [_encode_single_prompt(prompt_list, tokenizer)], True

    token_sequences = [_encode_single_prompt(item, tokenizer) for item in prompt_list]
    has_uniform_len = len({len(s) for s in token_sequences}) <= 1
    return token_sequences, has_uniform_len
