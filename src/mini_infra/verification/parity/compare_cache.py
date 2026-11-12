"""逐步比较同一手写模型的 no-cache 与 KV Cache 输出。"""

import torch

from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DenseAttentionMetadata,
)
from mini_infra.verification.parity._common import (
    finalize_stepwise_compare,
    prepare_prompt_compare,
    run_stepwise_greedy_compare,
)


@torch.inference_mode()
def _compare_cache_step_by_step(model, input_ids, max_new_tokens, eos_token_id):
    """逐步比较同一手写模型在无 Cache 与有 Cache 两种执行路径下的 Logits 与生成结果。"""
    generated_ids = input_ids
    cached_input = input_ids
    past_key_values = None

    def step_fn():
        nonlocal generated_ids, cached_input, past_key_values
        # baseline 路径：全量历史重新计算，attn_metadata=None 即不使用 Cache
        baseline_logits, _ = model(generated_ids)
        # cached 路径：首步 attn_metadata=None 即 Prefill，其后用上一步返回的历史增量追加
        attn_metadata = None if past_key_values is None else DenseAttentionMetadata(tuple(past_key_values))
        cached_logits, past_key_values = model(cached_input, attn_metadata=attn_metadata)

        baseline_last = baseline_logits[:, -1, :]
        next_token = torch.argmax(baseline_last, dim=-1, keepdim=True)
        generated_ids = torch.cat((generated_ids, next_token), dim=1)
        cached_input = next_token
        return (
            baseline_last,
            cached_logits[:, -1, :],
            next_token.item(),
            "cache_length=%d" % past_key_values[0][0].shape[2],
        )

    return run_stepwise_greedy_compare(
        step_fn,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        reference_name="no-cache",
        candidate_name="kv-cache",
    )


def execute_compare_cache(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompt: str = "请用一句话介绍 KV Cache。",
    max_new_tokens: int = 32,
    dtype: str = "float32",
    logits_atol: float = 1e-4,
    output: str | None = None,
) -> int:
    """运行 KV Cache 对齐检查，返回退出码（0 成功，1 Token 不一致，2 误差超标）。"""
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")
    if logits_atol < 0:
        raise ValueError("--logits-atol 不能小于 0")

    device, tokenizer, model_instance, environment, token_ids = prepare_prompt_compare(
        model, revision, dtype, prompt, max_new_tokens, logits_atol
    )
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    result = _compare_cache_step_by_step(
        model_instance,
        input_ids,
        max_new_tokens,
        tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(result["token_ids"], skip_special_tokens=True)
    return finalize_stepwise_compare(
        result, generated_text, logits_atol=logits_atol, output=output, environment=environment
    )


__all__ = [
    "execute_compare_cache",
]
