"""CPU 单元测试：prefix caching 关闭 / 开启的输出、首 Token Logits 误差与 Prefill 计算量对照。"""

import json
from unittest.mock import patch

import pytest

from mini_infra.verification.parity.compare_prefix_caching import (
    DEFAULT_PROMPTS,
    _expected_sequential_hits,
    execute_compare_prefix_caching,
    run_compare_prefix_caching,
)


def test_run_compare_prefix_caching_on_cpu(diverse_model, tmp_path):
    """单组 4 序列同时验证：冷启动、块对齐前缀、非对齐前缀截断与同块不同前文隔离。"""
    shared_two_blocks = [3, 5, 7, 11, 13, 17, 19, 23]  # block_size=4 下恰为 2 个满块
    sequences = [
        shared_two_blocks + [31, 32],
        shared_two_blocks + [41, 42],  # 与序列 0 共享精确 8 Token 对齐前缀 -> 命中 8
        shared_two_blocks
        + [41, 51, 52],  # 与序列 1 共享 9 Token 非对齐前缀 -> 截断命中 8
        [61, 62, 63, 64] + shared_two_blocks[4:] + [71],  # 首块不同、次块相同 -> 命中 0
    ]
    expected_hits = _expected_sequential_hits(sequences, block_size=4)
    assert expected_hits == [0, 8, 8, 0]
    out_file = tmp_path / "compare_prefix_caching_result.json"

    result = run_compare_prefix_caching(
        model=diverse_model,
        sequences=sequences,
        budgets=[4, 4, 4, 2],
        eos_token_id=-1,
        token_budget=8,
        block_size=4,
        max_running_requests=1,
        output=str(out_file),
    )

    assert result["tokens_match"] is True
    assert result["max_first_token_logit_error"] <= 1e-4
    assert set(result["per_request_first_token_logit_error"]) == {"0", "1", "2", "3"}
    assert (
        result["enabled_metrics"]["per_request_prefix_cache_hit_tokens"] == expected_hits
    )
    assert result["disabled_metrics"]["per_request_prefix_cache_hit_tokens"] == [
        0,
        0,
        0,
        0,
    ]
    assert (
        result["enabled_metrics"]["num_computed_prompt_tokens"]
        == sum(map(len, sequences)) - 16
    )
    assert result["disabled_metrics"]["num_computed_prompt_tokens"] == sum(
        map(len, sequences)
    )
    assert result["enabled_metrics"]["prefix_cache_hit_tokens"] == 16
    assert result["enabled_metrics"]["prefill_ms_p50"] > 0
    assert result["disabled_metrics"]["prefill_ms_p50"] > 0

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["result"]["tokens_match"] is True


def test_execute_compare_prefix_caching_validations():
    with pytest.raises(ValueError, match="max_new_tokens 必须大于 0"):
        execute_compare_prefix_caching(max_new_tokens=0)
    with pytest.raises(ValueError, match="token_budget 不能小于 max_running_requests"):
        execute_compare_prefix_caching(token_budget=1, max_running_requests=2)
    with pytest.raises(ValueError, match="block_size 必须大于 0"):
        execute_compare_prefix_caching(block_size=0)
    with pytest.raises(ValueError, match="logits_atol 不能为负数"):
        execute_compare_prefix_caching(logits_atol=-0.1)
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="需要可用的 NVIDIA GPU"):
            execute_compare_prefix_caching()


def test_default_prompts_cover_shared_prefix_and_distinct_header():
    assert len(DEFAULT_PROMPTS) == 4
    # 前 3 条共享长前缀，且第 1/2 条比第 0/1 条多共享一段后缀以构造对齐与非对齐两种边界
    common_01 = DEFAULT_PROMPTS[0]
    while not DEFAULT_PROMPTS[1].startswith(common_01):
        common_01 = common_01[:-1]
    common_12 = DEFAULT_PROMPTS[1]
    while not DEFAULT_PROMPTS[2].startswith(common_12):
        common_12 = common_12[:-1]
    assert len(common_12) > len(common_01) > 100
    # 第 3 条首行不同但复用相同的中间长背景，覆盖「不同前文 + 相同中间块」防误命中
    assert DEFAULT_PROMPTS[3][:4] != DEFAULT_PROMPTS[0][:4]
    assert DEFAULT_PROMPTS[0].splitlines()[1] == DEFAULT_PROMPTS[3].splitlines()[1]
