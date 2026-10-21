"""快速自检：用很小的随机模型检查核心代码，不需要下载真实权重，CPU 数秒内完成。

入口：mini-infra-smoke。同一份逻辑也被 tests/unit/test_model_smoke.py 复用。
"""

import torch

from mini_infra.generation import BatchingMethod, FeatureFlag, Mode, generate, left_pad_sequences
from mini_infra.qwen3_model import KVCache, Qwen3Config, Qwen3ForCausalLM, cache_size_bytes


def build_tiny_model():
    torch.manual_seed(0)
    # 缩小所有维度，使自检可以在 CPU 上几秒内完成。
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
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

    assert logits.shape == (1, 3, model.config.vocab_size)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def check_causal_mask(model):
    # 两个输入只有最后一个 Token 不同。若 causal mask 正确，前两个位置
    # 的输出不应该受最后一个 Token 影响。
    with torch.inference_mode():
        first = model(torch.tensor([[1, 2, 3]]))
        second = model(torch.tensor([[1, 2, 4]]))

    assert torch.allclose(first[:, :2, :], second[:, :2, :], atol=1e-6)


def check_kv_cache_correctness(model):
    """验证显式 KVCache 容器与全量重算路径输出完全一致。"""
    prompt = torch.tensor([[1, 2, 3]])
    config = model.config

    with torch.inference_mode():
        baseline_logits = model(prompt)
        cache = KVCache()
        prefill_logits = model(prompt, kv_cache=cache)

    assert torch.allclose(baseline_logits, prefill_logits, atol=1e-6)
    assert len(cache) == config.num_hidden_layers
    assert cache.get_seq_length() == 3
    expected_shape = (1, config.num_key_value_heads, 3, config.head_dim)
    for key, value in cache:
        assert tuple(key.shape) == expected_shape
        assert tuple(value.shape) == expected_shape

    expected_bytes = (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * 3
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    assert cache_size_bytes(cache) == expected_bytes
    assert cache.total_bytes() == expected_bytes

    # 单步 Decode 验证：传入 cache 自动追加
    next_input = torch.tensor([[4]])
    full_context = torch.cat((prompt, next_input), dim=1)
    with torch.inference_mode():
        full_logits = model(full_context)
        cached_logits = model(next_input, kv_cache=cache)

    assert torch.allclose(full_logits[:, -1, :], cached_logits[:, -1, :], atol=1e-5)
    assert cache.get_seq_length() == 4


def check_fixed_batching(model):
    """验证左 Padding、请求隔离、完成槽位指标与非法 Mask 拒绝。"""
    device = torch.device("cpu")
    config = model.config
    request_a = [3, 5, 7, 9]
    request_b = [11, 13]
    request_c = [17, 19]

    padded, mask, positions, lengths = left_pad_sequences([request_a, request_b], 0, device)
    assert padded.tolist() == [[3, 5, 7, 9], [0, 0, 11, 13]]
    assert mask.tolist() == [[True] * 4, [False, False, True, True]]
    assert positions.tolist() == [[0, 1, 2, 3], [0, 0, 0, 1]]
    assert lengths.tolist() == [4, 2]
    print("左 Padding、Position IDs、请求隔离、排列不变性检查通过")

    fixed_batching_mode = Mode({FeatureFlag.KV_CACHE}, batching=BatchingMethod.FIXED)
    kwargs = {"eos_token_id": -1, "device": device, "stop_on_eos": False, "capture_logits": True}
    single = generate(fixed_batching_mode, model, [request_a], 4, **kwargs)
    paired = generate(fixed_batching_mode, model, [request_b, request_a], [4, 4], **kwargs)
    replaced = generate(fixed_batching_mode, model, [request_c, request_a], [4, 4], **kwargs)

    assert (
        single["unpadded_output_token_ids"][0]
        == paired["unpadded_output_token_ids"][1]
        == replaced["unpadded_output_token_ids"][1]
    )
    for step in range(4):
        assert torch.allclose(single["logits_trace"][step][0], paired["logits_trace"][step][1], atol=1e-5)
        assert torch.allclose(paired["logits_trace"][step][1], replaced["logits_trace"][step][1], atol=1e-5)

    slots = generate(
        fixed_batching_mode, model, [request_a, request_b], [2, 4], eos_token_id=-1, device=device, stop_on_eos=False
    )
    assert slots.active_counts_per_step == [2, 2, 1, 1]
    assert slots.completion_steps_per_req == [2, 4]
    assert abs(slots.slot_utilization - 0.75) < 1e-12
    assert slots.cache_length == 7
    expected_bytes = 2 * config.num_hidden_layers * 2 * config.num_key_value_heads * config.head_dim * 7 * 4
    assert abs(slots.memory["kv_cache_mb"] - expected_bytes / (1024**2)) < 1e-12
    print("Batched KV Cache、完成槽位和利用率检查通过")

    # 串行 baseline（单请求 generate 路径）与 Batch 输出的 Token 应完全一致。
    serial = generate(
        Mode({FeatureFlag.KV_CACHE}),
        model,
        [request_a, request_b],
        [2, 4],
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
    )
    assert serial.unpadded_output_token_ids == slots.unpadded_output_token_ids
    print("Batch 与串行输出一致性检查通过")

    # 非法 Cache batch 维和错误长度的 Attention Mask 必须被拒绝。
    cache = KVCache()
    prefill_ids = torch.tensor([request_a])
    with torch.inference_mode():
        model(prefill_ids, kv_cache=cache)
        try:
            model(prefill_ids[:, -1:], kv_cache=cache, attention_mask=torch.ones((1, 1), dtype=torch.bool))
        except ValueError:
            pass
        else:
            raise AssertionError("错误长度的 Attention Mask 未被拒绝")


def run_smoke_test():
    model = build_tiny_model()
    check_output_shape_and_tied_weights(model)
    check_causal_mask(model)
    check_kv_cache_correctness(model)
    check_fixed_batching(model)


def main():
    run_smoke_test()
    print("smoke test passed")


if __name__ == "__main__":
    main()
