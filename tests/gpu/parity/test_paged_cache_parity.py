"""GPU 测试：验证 Dense 与 Paged KV Cache 的逐步 Logits 以及 Scheduler 调度输出完全对齐。

包含：
1. 单 Prompt 逐步 Logits 绝对误差与 Greedy Token 一致性测试（CLI + 直接调用）；
2. 调度器连续批处理环境下 Dense 与多 Block Size Paged KV Cache 输出一致性与入场轨迹测试（CLI + 直接调用）。

需要 GPU 和已下载的权重；不满足时自动跳过。
运行：pytest tests/gpu/parity/test_paged_cache_parity.py
"""

from __future__ import annotations

import pytest
import torch

from mini_infra.cli import main
from mini_infra.engine import Qwen3Tokenizer
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_model,
    resolve_model_directory,
)
from mini_infra.verification.parity.compare_paged_cache_logits import compare_paged_prompt
from mini_infra.verification.parity.compare_paged_cache_scheduler import (
    DEFAULT_PROMPTS,
    run_compare_paged_cache_scheduler,
)

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_paged_cache_logits_cli() -> None:
    """通过 CLI 入口验证 mini-infra verify paged-cache 的端到端执行。"""
    exit_code = main(
        [
            "verify",
            "paged-cache",
            "--prompts",
            "用一句话解释 Paged KV Cache。",
            "--max-new-tokens",
            "4",
            "--block-size",
            "16",
            "--dtype",
            "float32",
        ]
    )
    assert exit_code == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_paged_cache_logits_direct() -> None:
    """直接调用核心对比函数，断言 float32 下逐 step 误差与 Token 完全一致。"""
    device = torch.device("cuda")
    model_dir = resolve_model_directory(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device=device, dtype=torch.float32)
    model.eval()

    prompt = "Explain block tables in one short sentence."
    max_new_tokens = 6
    block_size = 4  # 选用较小 block_size 迫使产生跨块追加与索引

    res = compare_paged_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        device=device,
    )

    assert res["all_tokens_match"], f"Tokens divergence: {res['steps']}"
    assert res["max_abs_error"] <= 1e-4, f"Logits max error {res['max_abs_error']} > 1e-4"
    assert len(res["steps"]) == max_new_tokens - 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_paged_cache_scheduler_cli() -> None:
    """通过 CLI 入口验证 mini-infra verify paged-cache-scheduler 的端到端执行。"""
    exit_code = main(
        [
            "verify",
            "paged-cache-scheduler",
            "--max-new-tokens",
            "4",
            "--block-sizes",
            "4",
            "16",
            "--dtype",
            "float32",
        ]
    )
    assert exit_code == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_paged_cache_scheduler_direct() -> None:
    """直接调用调度器比对逻辑，断言 Dense 与 Paged 在多块大小下的逐请求输出 100% 一致。"""
    device = torch.device("cuda")
    model_dir = resolve_model_directory(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device=device, dtype=torch.float32)
    model.eval()

    res = run_compare_paged_cache_scheduler(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=DEFAULT_PROMPTS,
        max_new_tokens=6,
        block_sizes=[4, 16],
        max_running_requests=2,
    )

    assert res["all_match_dense"], "Paged KV Cache 输出与 Dense 不一致"
    for bs in (4, 16):
        assert res["paged"][str(bs)]["matches_dense"]
        assert len(res["paged"][str(bs)]["admission_trace"]) > 0
