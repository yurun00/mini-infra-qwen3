"""CPU 单元测试：token-budget 与 chunked-prefill 输出、首 Token Logits 误差与调度轨迹比对。"""

import json
from unittest.mock import patch

import pytest

from mini_infra.verification.parity.compare_chunked_prefill import (
    DEFAULT_ARRIVALS,
    DEFAULT_BUDGETS,
    DEFAULT_PROMPTS,
    execute_compare_chunked_prefill,
    run_compare_chunked_prefill,
)


def test_run_compare_chunked_prefill_on_cpu(diverse_model, tmp_path):
    """在 CPU 上用 diverse_model 验证 token-budget 与 chunked-prefill 输出、首 Token Logits 误差与硬预算轨迹。"""
    sequences = [
        [1, 2, 3, 4],
        [
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
        ],  # 长度 14 > token_budget=8，触发 oversize / 分块
        [19, 20, 21, 22, 23],
        [24, 25, 26],
    ]
    budgets = [6, 6, 6, 2]
    arrivals = [0.0, 0.1, 0.1, 0.1]
    out_file = tmp_path / "compare_chunked_prefill_result.json"

    result = run_compare_chunked_prefill(
        model=diverse_model,
        sequences=sequences,
        budgets=budgets,
        arrivals=arrivals,
        eos_token_id=-1,
        token_budget=8,
        block_size=4,
        max_running_requests=4,
        output=str(out_file),
    )

    assert result["tokens_match"] is True
    assert result["full_token_ids"] == result["chunked_token_ids"]
    assert len(result["full_token_ids"]) == 4
    assert result["prompt_token_lengths"] == [len(seq) for seq in sequences]

    # float32 下首 Token Logits 最大绝对误差在 1e-4 内，且覆盖所有请求
    assert set(result["per_request_first_token_logit_error"]) == {
        "0",
        "1",
        "2",
        "3",
    }
    assert result["max_first_token_logit_error"] <= 1e-4

    # token-budget（软预算）触发 oversize prefill；chunked-prefill（硬预算）无 oversize 且单步不超预算
    assert result["full_metrics"]["num_oversize_prefill_steps"] == 1
    assert result["chunked_metrics"]["num_oversize_prefill_steps"] == 0
    assert result["full_metrics"]["max_prefill_step_tokens"] == 14
    assert result["chunked_metrics"]["max_prefill_step_tokens"] <= 8
    assert (
        result["chunked_metrics"]["num_prefill_chunks"]
        > result["full_metrics"]["num_prefill_chunks"]
    )

    # 校验紧凑轨迹字段
    assert any(step["is_oversize_prefill"] for step in result["full_trace"])
    assert not any(step["is_oversize_prefill"] for step in result["chunked_trace"])
    assert all(
        step["num_scheduled_tokens"] <= 8
        for step in result["chunked_trace"]
        if step["phase"] == "prefill"
    )

    # 验证 JSON 导出文件结构
    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "environment" in data
    assert "result" in data
    assert data["result"]["tokens_match"] is True
    assert data["result"]["max_first_token_logit_error"] <= 1e-4


def test_execute_compare_chunked_prefill_validations():
    """验证非法参数在执行前被拦截。"""
    with pytest.raises(ValueError, match="max_new_tokens 至少为 2"):
        execute_compare_chunked_prefill(max_new_tokens=1)

    with pytest.raises(ValueError, match="max_running_requests 必须大于 0"):
        execute_compare_chunked_prefill(max_running_requests=0)

    with pytest.raises(ValueError, match="token_budget 不能小于 max_running_requests"):
        execute_compare_chunked_prefill(token_budget=3, max_running_requests=4)

    with pytest.raises(ValueError, match="block_size 必须大于 0"):
        execute_compare_chunked_prefill(block_size=0)

    with pytest.raises(ValueError, match="logits_atol 不能为负数"):
        execute_compare_chunked_prefill(logits_atol=-1e-4)

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="需要可用的 NVIDIA GPU"):
            execute_compare_chunked_prefill()


def test_default_constants():
    """验证默认负载包含 4 条错峰到达、长短悬殊的请求。"""
    assert len(DEFAULT_PROMPTS) == 4
    assert len(DEFAULT_BUDGETS) == 4
    assert len(DEFAULT_ARRIVALS) == 4
    assert DEFAULT_ARRIVALS == [0.0, 1.0, 1.0, 1.0]
    assert len(DEFAULT_PROMPTS[1]) > len(DEFAULT_PROMPTS[0]) * 4
