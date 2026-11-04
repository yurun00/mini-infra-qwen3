"""单元测试：engine/preprocessing.py 纯 Python 序列规约与预算校验。"""

import sys
from unittest.mock import MagicMock

import pytest

from mini_infra.engine.preprocessing import (
    normalize_budgets,
    normalize_token_sequences,
)


def test_preprocessing_has_no_torch_import():
    """架构不变量检验：engine/preprocessing 纯 CPU 逻辑严禁依赖 torch。"""
    module = sys.modules["mini_infra.engine.preprocessing"]
    assert "torch" not in module.__dict__


def test_normalize_budgets_scalar():
    budgets = normalize_budgets(10, batch_size=3)
    assert budgets == [10, 10, 10]
    assert isinstance(budgets, list)
    assert all(isinstance(b, int) for b in budgets)


def test_normalize_budgets_sequence():
    budgets = normalize_budgets([5, 10, 15], batch_size=3)
    assert budgets == [5, 10, 15]


def test_normalize_budgets_invalid():
    with pytest.raises(ValueError, match="大于 0"):
        normalize_budgets([5, 0, 10], batch_size=3)

    with pytest.raises(ValueError, match="大于 0"):
        normalize_budgets([5, 10], batch_size=3)


def test_normalize_token_sequences_single_int_sequence():
    seqs, is_uniform = normalize_token_sequences([101, 2049, 102])
    assert seqs == [(101, 2049, 102)]
    assert is_uniform is True


def test_normalize_token_sequences_multi_uniform():
    seqs, is_uniform = normalize_token_sequences([[1, 2], [3, 4]])
    assert seqs == [(1, 2), (3, 4)]
    assert is_uniform is True


def test_normalize_token_sequences_multi_non_uniform():
    seqs, is_uniform = normalize_token_sequences([[1, 2], [3, 4, 5]])
    assert seqs == [(1, 2), (3, 4, 5)]
    assert is_uniform is False


def test_normalize_token_sequences_text_with_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text: [100] * len(text)

    # 单条文本
    seqs, is_uniform = normalize_token_sequences("hello", tokenizer=mock_tokenizer)
    assert seqs == [(100, 100, 100, 100, 100)]
    assert is_uniform is True

    # 多条文本
    seqs, is_uniform = normalize_token_sequences(["hi", "hello"], tokenizer=mock_tokenizer)
    assert seqs == [(100, 100), (100, 100, 100, 100, 100)]
    assert is_uniform is False


def test_normalize_token_sequences_text_without_tokenizer():
    with pytest.raises(ValueError, match="必须提供 tokenizer"):
        normalize_token_sequences("hello")

    with pytest.raises(ValueError, match="必须提供 tokenizer"):
        normalize_token_sequences(["hello", "world"])
