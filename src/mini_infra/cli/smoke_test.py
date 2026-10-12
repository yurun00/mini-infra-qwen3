"""快速自检：用很小的随机模型检查核心代码，不需要下载真实权重，CPU 数秒内完成。

入口：mini-infra-smoke。同一份逻辑也被 tests/unit/test_model_smoke.py 复用。
"""

import torch

from mini_infra.qwen3_model import Qwen3Config, Qwen3ForCausalLM


def build_tiny_model():
    torch.manual_seed(0)
    # 缩小所有维度，使自检可以在 CPU 上几秒内完成。
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )
    config.validate()
    return Qwen3ForCausalLM(config).eval()


def check_output_shape_and_tied_weights(model):
    with torch.inference_mode():
        logits = model(torch.tensor([[1, 2, 3]]))

    assert logits.shape == (1, 3, 32)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def check_causal_mask(model):
    # 两个输入只有最后一个 Token 不同。若 causal mask 正确，前两个位置
    # 的输出不应该受最后一个 Token 影响。
    with torch.inference_mode():
        first = model(torch.tensor([[1, 2, 3]]))
        second = model(torch.tensor([[1, 2, 4]]))

    assert torch.allclose(first[:, :2, :], second[:, :2, :], atol=1e-6)


def run_smoke_test():
    model = build_tiny_model()
    check_output_shape_and_tied_weights(model)
    check_causal_mask(model)


def main():
    run_smoke_test()
    print("smoke test passed")


if __name__ == "__main__":
    main()
