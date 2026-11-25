"""CPU 单元测试：prefix caching 关闭 / 开启的输出、首 Token Logits 误差与 Prefill 计算量对照。"""

import json
from unittest.mock import patch

import pytest

from mini_infra.verification.parity.compare_prefix_caching import (
    DEFAULT_PROMPTS,
    execute_compare_prefix_caching,
    run_compare_prefix_caching,
)


def test_run_compare_prefix_caching_on_cpu(diverse_model, tmp_path):
    shared_prefix = [(j * 7) % 90 + 1 for j in range(10)]
    sequences = [
        shared_prefix + [11, 12],
        shared_prefix + [21, 22, 23],
        shared_prefix + [31],
    ]
    out_file = tmp_path / "compare_prefix_caching_result.json"

    result = run_compare_prefix_caching(
        model=diverse_model,
        sequences=sequences,
        budgets=[5, 4, 6],
        eos_token_id=-1,
        token_budget=8,
        block_size=4,
        max_running_requests=1,
        output=str(out_file),
    )

    assert result["tokens_match"] is True
    assert result["max_first_token_logit_error"] <= 1e-4
    # 后续请求命中 10 Token 公共前缀中的 2 个满块（8 Token）
    assert result["enabled_metrics"]["per_request_prefix_cache_hit_tokens"] == [
        0,
        8,
        8,
    ]
    assert result["disabled_metrics"]["per_request_prefix_cache_hit_tokens"] == [0, 0, 0]
    assert result["per_request_logit_prefix_cache_hit_tokens"] == {
        "0": 0,
        "1": 8,
        "2": 8,
    }
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
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="需要可用的 NVIDIA GPU"):
            execute_compare_prefix_caching()


def test_default_prompts_share_a_long_prefix():
    assert len(DEFAULT_PROMPTS) >= 2
    common = DEFAULT_PROMPTS[0]
    for prompt in DEFAULT_PROMPTS[1:]:
        while not prompt.startswith(common):
            common = common[:-1]
    assert len(common) > 100
