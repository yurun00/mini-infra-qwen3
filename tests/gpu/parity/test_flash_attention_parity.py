"""GPU 测试：Triton FlashAttention backend 与 eager baseline 的 Kernel 级与真实模型级数值对齐。

包含：
1. Kernel 级（无需权重）：float32 / bfloat16 下 Full Prefill、非整 Tile、Chunked Prefill、左 Padding Batch、Decode
   与 MHA / 小 head_dim 形状，对照 float32 eager Reference；
2. 模型级（需权重）：float32 下 Lockstep / Paged chunked-prefill 两条路径 eager 与 flash
   的 Greedy Token 100% 一致且 Logits 误差 <= 1e-4（直接调用 + CLI）。

运行：pytest tests/gpu/parity/test_flash_attention_parity.py
"""

from __future__ import annotations

import pytest
import torch

from mini_infra.cli import main
from mini_infra.engine import Qwen3Tokenizer
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    CausalPaddingMask,
    Qwen3Config,
    get_attn_backend,
    load_model,
    resolve_model_directory,
)
from mini_infra.verification.parity.compare_flash_attention import (
    DEFAULT_PROMPTS,
    run_compare_flash_attention_model,
    run_kernel_cases,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU"),
]


def _attention_config(num_heads: int, num_kv_heads: int, head_dim: int) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=16,
        hidden_size=num_heads * head_dim,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )


@pytest.mark.parametrize(
    ("dtype", "atol"), [(torch.float32, 1e-5), (torch.bfloat16, 3e-2)]
)
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "head_dim"), [(16, 8, 128), (4, 4, 64)]
)
def test_flash_kernel_matches_eager_reference(
    dtype: torch.dtype, atol: float, num_heads: int, num_kv_heads: int, head_dim: int
) -> None:
    rows = run_kernel_cases(
        _attention_config(num_heads, num_kv_heads, head_dim), torch.device("cuda"), dtype
    )
    for row in rows:
        assert row["is_finite"], row["name"]
        assert row["max_abs_error"] <= atol, row


def test_flash_kernel_zeroes_padding_query_rows() -> None:
    """padding Query 行输出置 0，有效行不受影响；整块左 padding 不产生 NaN。"""
    device = torch.device("cuda")
    flash = get_attn_backend("flash")
    query = torch.randn((2, 4, 80, 64), device=device)
    key = torch.randn((2, 2, 80, 64), device=device)
    value = torch.randn((2, 2, 80, 64), device=device)
    key_valid = torch.ones((2, 80), dtype=torch.bool, device=device)
    key_valid[1, :70] = False  # 跨越多个 BLOCK_N 的左 padding
    mask = CausalPaddingMask(key_valid, key_valid, 0)
    with torch.inference_mode():
        output = flash.forward(
            query, key, value, flash.prepare_inputs(mask, query.dtype), 0.125
        )
    output = output.view(2, 80, 4, 64)
    assert torch.isfinite(output).all()
    assert torch.all(output[1, :70] == 0)
    assert torch.any(output[1, 70:] != 0)


def test_flash_kernel_rejects_invalid_inputs() -> None:
    from mini_infra.model.attention.flash_attention import flash_attention_forward

    device = torch.device("cuda")
    query = torch.randn((1, 4, 8, 48), device=device)
    key = torch.randn((1, 2, 8, 48), device=device)
    valid = torch.ones((1, 8), dtype=torch.bool, device=device)
    with pytest.raises(ValueError, match="head_dim"):
        flash_attention_forward(query, key, key, valid, valid, 0)
    with pytest.raises(RuntimeError, match="CUDA"):
        cpu = torch.randn((1, 4, 8, 64))
        flash_attention_forward(cpu, cpu, cpu, valid.cpu(), valid.cpu(), 0)


def test_compare_flash_attention_model_direct() -> None:
    """float32 真实权重：两条引擎路径 eager 与 flash 的 Greedy Token 完全一致且 Logits 误差 <= 1e-4。"""
    model_dir = resolve_model_directory(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device=torch.device("cuda"), dtype=torch.float32)
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in DEFAULT_PROMPTS]

    rows = run_compare_flash_attention_model(
        model,
        sequences,
        max_new_tokens=6,
        eos_token_id=tokenizer.eos_token_id,
        token_budget=16,
        block_size=4,
        max_running_requests=2,
    )
    assert len(rows) == 2
    for row in rows:
        assert row["tokens_match"], row["engine_config"]
        assert row["max_logit_error"] <= 1e-4, row


def test_compare_flash_attention_cli() -> None:
    exit_code = main(
        ["verify", "flash-attention", "--max-new-tokens", "4", "--dtype", "float32"]
    )
    assert exit_code == 0
