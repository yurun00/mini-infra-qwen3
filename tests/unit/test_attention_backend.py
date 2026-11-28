"""attention backend 抽象（eager / flash 选择、掩码整理、EngineConfig 与 CLI 联动）的 CPU 单元测试。

flash backend 的 Triton Kernel 数值正确性需要 CUDA，见 tests/gpu/parity/test_flash_attention_parity.py。
"""

import argparse

import pytest
import torch

from mini_infra.cli.commands.run import (
    add_run_args,
    build_engine_config,
    resolve_compare_feature,
)
from mini_infra.core import DenseCacheConfig
from mini_infra.engine import BatchingMethod, EngineConfig
from mini_infra.model import (
    ATTENTION_BACKEND_KINDS,
    CausalPaddingMask,
    get_attn_backend,
    parse_attention_backend_kind,
)
from mini_infra.model.attention import EAGER_ATTENTION_BACKEND, EagerAttentionInputs
from mini_infra.verification.smoke import build_tiny_model
from mini_infra.worker import PagedRunner


def test_parse_attention_backend_kind():
    assert ATTENTION_BACKEND_KINDS == ("eager", "flash")
    for kind in ATTENTION_BACKEND_KINDS:
        assert parse_attention_backend_kind(kind) == kind
    with pytest.raises(ValueError, match="未知 attention backend"):
        parse_attention_backend_kind("sdpa")


def test_get_attn_backend_eager_is_stateless_singleton():
    backend = get_attn_backend("eager")
    assert backend is EAGER_ATTENTION_BACKEND
    assert backend.kind == "eager"


def test_eager_prepare_inputs_causal_with_left_padding_and_offset():
    """Chunked Prefill 形态：历史 past_len=3、本步 query_len=2，第 1 行左 padding 2 个位置。"""
    key_valid = torch.tensor(
        [[True, True, True, True, True], [False, False, True, True, True]]
    )
    mask = CausalPaddingMask(key_valid, key_valid[:, 3:], causal_offset=3)
    attn_inputs = EAGER_ATTENTION_BACKEND.prepare_inputs(mask, torch.float32)
    assert isinstance(attn_inputs, EagerAttentionInputs)
    attn_bias = attn_inputs.attn_bias

    assert attn_bias.shape == (2, 1, 2, 5)
    is_allowed = attn_bias == 0.0
    # Query 行 q 可见 k <= 3 + q 且 key_valid
    expected = torch.tensor(
        [
            [[True, True, True, True, False], [True, True, True, True, True]],
            [[False, False, True, True, False], [False, False, True, True, True]],
        ]
    )
    assert torch.equal(is_allowed[:, 0], expected)
    assert torch.all(attn_bias[~is_allowed] == torch.finfo(torch.float32).min)


def test_eager_prepare_inputs_matches_paged_decode_bias():
    """Paged Decode 形态（causal_offset = max_history_len - 1）与历史 context_lens 加性偏置逐元素一致。"""
    context_lens = torch.tensor([3, 8, 5])
    max_history_len = 8
    positions = torch.arange(max_history_len)
    key_valid = positions.unsqueeze(0) < context_lens.unsqueeze(1)
    mask = CausalPaddingMask(
        key_valid, torch.ones((3, 1), dtype=torch.bool), max_history_len - 1
    )
    attn_inputs = EAGER_ATTENTION_BACKEND.prepare_inputs(mask, torch.float32)
    attn_bias = attn_inputs.attn_bias

    legacy_bias = torch.zeros((3, 1, 1, max_history_len))
    legacy_bias.masked_fill_(
        ~key_valid.view(3, 1, 1, max_history_len), torch.finfo(torch.float32).min
    )
    assert torch.equal(attn_bias, legacy_bias)


def test_model_forward_default_backend_is_eager():
    model = build_tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        default_logits, _ = model(input_ids)
        eager_logits, _ = model(input_ids, attn_backend=get_attn_backend("eager"))
    assert torch.equal(default_logits, eager_logits)


def test_runner_resolves_attention_backend():
    model = build_tiny_model()
    assert PagedRunner(model).attn_backend is EAGER_ATTENTION_BACKEND
    assert PagedRunner(model, 0, "eager").attn_backend is EAGER_ATTENTION_BACKEND


def test_engine_config_attention_backend_identity():
    config = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS)
    assert config.attention_backend == "eager"
    # 默认 eager 不进入 engine_config 标识，保持历史 Record 不变
    assert str(config) == "dense-kv-cache+continuous-batching"

    flash_config = config.with_attention_backend("flash")
    assert flash_config.attention_backend == "flash"
    assert flash_config != config
    assert str(flash_config) == "dense-kv-cache+continuous-batching+flash-attention"


def test_cli_attention_backend_and_compare_feature():
    parser = add_run_args(argparse.ArgumentParser())
    assert build_engine_config(parser.parse_args([])).attention_backend == "eager"
    args = parser.parse_args(["--attention-backend", "flash"])
    assert build_engine_config(args).attention_backend == "flash"
    with pytest.raises(SystemExit):
        parser.parse_args(["--attention-backend", "sdpa"])

    args = parser.parse_args(["--compare-feature", "attention-backend", "eager", "flash"])
    (eager_config, flash_config), description = resolve_compare_feature(args)
    assert eager_config.attention_backend == "eager"
    assert flash_config == eager_config.with_attention_backend("flash")
    assert description == "attention-backend: eager vs flash"
