"""CPU 单元测试：prefill-first 与 token-budget 两种迭代级调度策略输出与轨迹比对 (compare_schedulers)。"""

import json
from unittest.mock import patch

import pytest

from mini_infra.verification.parity.compare_schedulers_token_budget import (
    DEFAULT_ARRIVALS,
    DEFAULT_BUDGETS,
    DEFAULT_PROMPTS,
    execute_compare_schedulers,
    run_compare_schedulers,
)


def test_run_compare_schedulers_on_cpu(diverse_model, tmp_path):
    """在 CPU 上用 diverse_model 验证 prefill-first 与 token-budget 输出、decode protection 与 oversize 轨迹。"""
    sequences = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14],  # 长度 7 > token_budget=6，触发 oversize prefill
        [15, 16],
        [17, 18, 19],
        [20, 21, 22],
    ]
    budgets = [5, 5, 2, 4, 3, 4]
    arrivals = [0.0, 0.0, 0.1, 0.1, 0.1, 0.1]
    out_file = tmp_path / "compare_schedulers_result.json"

    result = run_compare_schedulers(
        model=diverse_model,
        sequences=sequences,
        budgets=budgets,
        arrivals=arrivals,
        eos_token_id=-1,
        token_budget=6,
        block_size=4,
        max_running_requests=4,
        output=str(out_file),
    )

    assert result["tokens_match"] is True
    assert result["prefill_first_token_ids"] == result["token_budget_token_ids"]
    assert len(result["prefill_first_token_ids"]) == 6

    # prefill-first 会连续 Prefill；token-budget 受 decode protection 约束，最大连续 Prefill 步数为 1
    assert result["prefill_first_metrics"]["max_consecutive_prefill_steps"] > 1
    assert result["token_budget_metrics"]["max_consecutive_prefill_steps"] == 1

    # 请求 2 的 Prompt 长度为 7 > token_budget=6，应在 token_budget_trace 中标记为 is_oversize_prefill
    oversize_steps = [step for step in result["token_budget_trace"] if step["is_oversize_prefill"]]
    assert len(oversize_steps) == 1
    assert oversize_steps[0]["admitted_request_ids"] == [2]
    assert oversize_steps[0]["num_scheduled_tokens"] == 7

    # 验证 JSON 导出文件结构
    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "environment" in data
    assert "result" in data
    assert data["result"]["tokens_match"] is True


def test_execute_compare_schedulers_validations():
    """验证非法参数在执行前被拦截。"""
    with pytest.raises(ValueError, match="max_new_tokens 至少为 2"):
        execute_compare_schedulers(max_new_tokens=1)

    with pytest.raises(ValueError, match="max_running_requests 必须大于 0"):
        execute_compare_schedulers(max_running_requests=0)

    with pytest.raises(ValueError, match="token_budget 不能小于 max_running_requests"):
        execute_compare_schedulers(token_budget=3, max_running_requests=4)

    with pytest.raises(ValueError, match="block_size 必须大于 0"):
        execute_compare_schedulers(block_size=0)

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="需要可用的 NVIDIA GPU"):
            execute_compare_schedulers()


def test_default_constants():
    """验证默认负载包含 6 条错峰到达的请求。"""
    assert len(DEFAULT_PROMPTS) == 6
    assert len(DEFAULT_BUDGETS) == 6
    assert len(DEFAULT_ARRIVALS) == 6
    assert DEFAULT_ARRIVALS == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
