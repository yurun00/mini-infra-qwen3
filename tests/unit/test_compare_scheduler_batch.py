"""CPU 单元测试：Scheduler 批处理动态请求归属与输出一致性比对 (compare_scheduler_batch)。"""

import json
from unittest.mock import patch

import pytest
import torch

from mini_infra.verification.parity.compare_scheduler_batch import (
    DEFAULT_PROMPTS,
    execute_compare_scheduler_batch,
    parse_args,
    run_compare_scheduler_batch,
)


def test_parse_args():
    """验证参数解析默认值及自定义选项。"""
    defaults = parse_args([])
    assert defaults.max_new_tokens == 8
    assert defaults.max_running_requests == 2
    assert defaults.dtype == "float32"
    assert defaults.output is None

    custom = parse_args(
        ["--max-new-tokens", "12", "--max-running-requests", "3", "--dtype", "bfloat16", "--output", "/tmp/out.json"]
    )
    assert custom.max_new_tokens == 12
    assert custom.max_running_requests == 3
    assert custom.dtype == "bfloat16"
    assert custom.output == "/tmp/out.json"


def test_run_compare_scheduler_batch_on_cpu(diverse_model, tmp_path):
    """在 CPU 上使用 diverse_model 验证 serialized、fixed 与 continuous 输出完全一致及 admission trace。"""
    device = torch.device("cpu")
    sequences = [
        [1, 2, 3, 4],
        [5, 6],
        [7, 8, 9],
        [10, 11, 12, 13, 14],
    ]
    budgets = [2, 4, 3, 4]
    out_file = tmp_path / "compare_sb_result.json"

    result = run_compare_scheduler_batch(
        model=diverse_model,
        sequences=sequences,
        budgets=budgets,
        eos_token_id=-1,
        device=device,
        max_running_requests=2,
        output=out_file,
    )

    # 1. 验证三种运行模式的 Token 一致性
    assert result["fixed_matches_serialized"] is True
    assert result["continuous_matches_serialized"] is True
    assert result["fixed_matches_continuous"] is True
    assert len(result["serialized_token_ids"]) == 4
    assert len(result["fixed_token_ids"]) == 4
    assert len(result["continuous_token_ids"]) == 4

    # 2. 验证 admission trace 记录了动态入场
    assert len(result["continuous_admission_trace"]) >= 2
    # 初始入场前两个请求 [0, 1]
    assert result["continuous_admission_trace"][0] == [0, 1]

    # 3. 验证 JSON 导出文件结构
    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "environment" in data
    assert "result" in data
    assert data["result"]["continuous_matches_serialized"] is True


def test_execute_validations():
    """验证非法参数在执行前被拦截。"""
    with pytest.raises(ValueError, match="max_new_tokens 至少为 2"):
        execute_compare_scheduler_batch(max_new_tokens=1)

    with pytest.raises(ValueError, match="max_running_requests 必须大于 0"):
        execute_compare_scheduler_batch(max_running_requests=0)

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="需要可用的 NVIDIA GPU"):
            execute_compare_scheduler_batch()


def test_default_prompts():
    """验证默认 Prompts 包含 4 条自然语言提示词。"""
    assert len(DEFAULT_PROMPTS) == 4
    assert "Continuous Batching" in DEFAULT_PROMPTS[0]
