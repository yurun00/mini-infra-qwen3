"""GPU 测试：验证 Dense 与 Paged KV Cache 的逐步 Logits 与 Token 生成完全对齐。

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
from mini_infra.verification.parity.compare_paged_cache import compare_paged_prompt

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_paged_cache_cli() -> None:
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
