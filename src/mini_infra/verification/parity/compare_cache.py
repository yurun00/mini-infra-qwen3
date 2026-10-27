"""逐步比较同一手写模型的 no-cache 与 KV Cache 输出。"""

import torch

from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    KVCache,
    Qwen3Tokenizer,
    load_model,
    resolve_model_directory,
)


@torch.inference_mode()
def compare_step_by_step(model, input_ids, max_new_tokens, eos_token_id):
    """逐步比较同一手写模型在无 Cache 与有 Cache 两种执行路径下的 Logits 与生成结果。"""
    generated_ids = input_ids
    cached_input = input_ids
    cache = KVCache()
    generated_token_ids = []
    largest_error = 0.0
    mean_error_sum = 0.0

    for step in range(max_new_tokens):
        # baseline 路径：全量历史重新计算，不使用 Cache
        baseline_logits = model(generated_ids, kv_cache=None)
        # cached 路径：使用显式 KVCache 容器（Prefill 首次建立，Decode 增量追加）
        cached_logits = model(cached_input, kv_cache=cache)

        baseline_last = baseline_logits[:, -1, :]
        cached_last = cached_logits[:, -1, :]

        # 先转为 float32 再计算绝对误差，避免低精度相减损失信息
        difference = (baseline_last.float() - cached_last.float()).abs()
        max_error = difference.max().item()
        mean_error = difference.mean().item()
        largest_error = max(largest_error, max_error)
        mean_error_sum += mean_error

        baseline_token = torch.argmax(baseline_last, dim=-1, keepdim=True)
        cached_token = torch.argmax(cached_last, dim=-1, keepdim=True)
        baseline_id = baseline_token.item()
        cached_id = cached_token.item()
        cache_length = cache.get_seq_length()

        print(
            "步骤 %02d: no-cache=%d, kv-cache=%d, cache_length=%d, "
            "max_error=%.6f, mean_error=%.6f"
            % (
                step + 1,
                baseline_id,
                cached_id,
                cache_length,
                max_error,
                mean_error,
            )
        )

        if baseline_id != cached_id:
            top3_base_vals, top3_base_ids = torch.topk(baseline_last[0], k=3)
            top3_cached_vals, top3_cached_ids = torch.topk(cached_last[0], k=3)

            print("\n[Token 不一致] 步骤 %02d: no-cache=%d, kv-cache=%d" % (step + 1, baseline_id, cached_id))
            print("no-cache 前 3 大 Logits:")
            for rank, (tid, val) in enumerate(zip(top3_base_ids.tolist(), top3_base_vals.tolist()), 1):
                print("  %d. token=%d, logit=%.6f" % (rank, tid, val))
            print("kv-cache 前 3 大 Logits:")
            for rank, (tid, val) in enumerate(zip(top3_cached_ids.tolist(), top3_cached_vals.tolist()), 1):
                print("  %d. token=%d, logit=%.6f" % (rank, tid, val))
            return {
                "tokens_match": False,
                "first_mismatch_step": step + 1,
                "largest_error": largest_error,
                "average_mean_error": mean_error_sum / (step + 1),
                "token_ids": generated_token_ids,
            }

        generated_token_ids.append(baseline_id)
        generated_ids = torch.cat((generated_ids, baseline_token), dim=1)
        cached_input = baseline_token

        if baseline_id == eos_token_id:
            break

    return {
        "tokens_match": True,
        "first_mismatch_step": None,
        "largest_error": largest_error,
        "average_mean_error": mean_error_sum / len(generated_token_ids) if generated_token_ids else 0.0,
        "token_ids": generated_token_ids,
    }


def run_compare_cache(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompt: str = "请用一句话介绍 KV Cache。",
    max_new_tokens: int = 32,
    dtype: str = "float32",
    logits_atol: float = 1e-4,
) -> int:
    """运行 KV Cache 对齐检查，返回退出码（0 成功，1 Token 不一致，2 误差超标）。"""
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")
    if logits_atol < 0:
        raise ValueError("--logits-atol 不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(model, revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    token_ids = tokenizer.encode_chat_prompt(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    comparison_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]
    model_instance = load_model(model_directory, device, dtype=comparison_dtype)

    print("Prompt Token 数: %d" % len(token_ids))
    print("对比 dtype: %s, 最大允许 Logits 绝对误差: %.6f" % (dtype, logits_atol))
    result = compare_step_by_step(
        model_instance,
        input_ids,
        max_new_tokens,
        tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(result["token_ids"], skip_special_tokens=True)
    print("\n逐 Token 一致:", result["tokens_match"])
    print("所有步骤最大 Logits 绝对误差: %.6f" % result["largest_error"])
    print("各步骤平均 Logits 误差的均值: %.6f" % result["average_mean_error"])
    print("生成 Token IDs:", result["token_ids"])
    print("生成文本:", generated_text)

    if not result["tokens_match"]:
        print("首个不一致步骤:", result["first_mismatch_step"])
        return 1
    if result["largest_error"] > logits_atol:
        print("Token 一致，但 Logits 误差超过设定阈值。")
        return 2
    return 0


__all__ = [
    "compare_step_by_step",
    "run_compare_cache",
]
