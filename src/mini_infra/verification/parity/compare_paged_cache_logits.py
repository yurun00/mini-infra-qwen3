"""逐 Token 比较 Dense 与 Paged KV Cache 的 Logits 与生成一致性。"""

import math
from collections.abc import Sequence
from typing import Any

import torch

from mini_infra.core.block_manager import KVCacheManager
from mini_infra.engine import Qwen3Tokenizer
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    DenseAttentionMetadata,
)
from mini_infra.verification.parity._common import (
    build_environment,
    compare_logits,
    dump_json,
    prepare_cuda_model,
)
from mini_infra.worker import KVCache, PagedKVCachePool, PagedRunner

DEFAULT_PROMPTS = [
    "用一句话解释 Paged KV Cache。",
    "Explain block tables in one short sentence.",
]


@torch.inference_mode()
def compare_paged_prompt(
    model: torch.nn.Module,
    tokenizer: Qwen3Tokenizer,
    prompt: str,
    max_new_tokens: int,
    block_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """逐 Token 比较单个 prompt 在 Dense 与 Paged KV Cache 下的 Logits。"""
    token_ids = tokenizer.encode_chat_prompt(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # 1. Prefill 阶段（得到第 1 个输出 token 以及初始 dense_cache）
    logits, dense_cache = model(input_ids)
    next_token = int(torch.argmax(logits[0, -1]).item())
    generated = [next_token]

    # 2. 初始化 Paged 物理池与管理器
    max_length = len(token_ids) + max_new_tokens - 1
    blocks_needed = math.ceil(max_length / block_size)
    num_blocks = blocks_needed + 1  # 包含 null block 0

    pool = PagedKVCachePool(
        num_layers=model.config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    manager = KVCacheManager(block_size=block_size, num_blocks=num_blocks)
    if not manager.allocate_slots(0, num_new_tokens=len(token_ids)):
        raise RuntimeError("分配 Prefill 物理块失败")

    runner = PagedRunner(model)
    prefill_cache = KVCache([(k.clone(), v.clone()) for k, v in dense_cache])
    runner.store_prefill(
        pool,
        prefill_cache=prefill_cache,
        block_tables=[manager.block_table(0)],
        cache_lengths=[len(token_ids)],
    )

    steps: list[dict[str, Any]] = []
    # 3. 逐步 Decode 比较
    for step in range(max_new_tokens - 1):
        position = len(token_ids) + step
        # Paged 分配本步新 slot 并构造输入
        if not manager.allocate_slots(0, num_new_tokens=1):
            raise RuntimeError(f"步骤 {step + 1} 分配 Decode 物理块失败")

        inp, pos, metadata = runner.prepare_decode_inputs(
            block_tables=[manager.block_table(0)],
            last_tokens=[next_token],
            positions=[position],
            block_size=block_size,
        )
        attn_metadata = metadata._replace(kv_pools=tuple(pool.kv_pools))
        paged_logits, _ = model(inp, position_ids=pos, attn_metadata=attn_metadata)

        # Dense Decode
        dense_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        dense_pos = torch.tensor([[position]], dtype=torch.long, device=device)
        dense_mask = torch.ones((1, position + 1), dtype=torch.bool, device=device)
        dense_meta = DenseAttentionMetadata(tuple(dense_cache))
        dense_logits, dense_cache = model(
            dense_input,
            position_ids=dense_pos,
            attention_mask=dense_mask,
            attn_metadata=dense_meta,
        )

        dense_last = dense_logits[0, -1]
        paged_last = paged_logits[0, -1]

        comparison = compare_logits(
            dense_last,
            paged_last,
            step=step + 1,
            reference_name="dense",
            candidate_name="paged",
            extra=f"position={position}",
        )

        steps.append(
            {
                "step": step + 1,
                "position": position,
                "max_abs_error": comparison.max_abs_error,
                "mean_abs_error": comparison.mean_abs_error,
                "dense_token_id": comparison.reference_token,
                "paged_token_id": comparison.candidate_token,
                "token_match": comparison.matches,
            }
        )

        next_token = comparison.reference_token
        generated.append(next_token)

    return {
        "prompt": prompt,
        "prompt_tokens": len(token_ids),
        "generated_token_ids": generated,
        "all_tokens_match": all(item["token_match"] for item in steps),
        "max_abs_error": max((item["max_abs_error"] for item in steps), default=0.0),
        "steps": steps,
    }


def execute_compare_paged_logits(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompts: Sequence[str] | None = None,
    max_new_tokens: int = 8,
    block_size: int = 16,
    dtype: str = "float32",
    logits_atol: float | None = None,
    output: str | None = None,
) -> int:
    """运行 Paged KV Cache Logits 对齐检查，返回退出码（0 成功，1 失败）。"""
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2")
    if block_size < 1:
        raise ValueError("block_size 必须为正")

    if logits_atol is None:
        logits_atol = 1e-4 if dtype == "float32" else 1.0

    device, _, tokenizer, model_instance = prepare_cuda_model(model, revision, dtype)
    prompts_to_run = list(prompts) if prompts else DEFAULT_PROMPTS

    print(f"=== 开始 Paged KV Cache Logits 对照 (dtype={dtype}, atol={logits_atol}, block_size={block_size}) ===")
    results = [
        compare_paged_prompt(
            model_instance,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            device=device,
        )
        for prompt in prompts_to_run
    ]

    all_tokens_match = all(r["all_tokens_match"] for r in results)
    overall_max_error = max(r["max_abs_error"] for r in results) if results else 0.0
    passed = all_tokens_match and overall_max_error <= logits_atol

    print("\n逐 Token Top-1 一致:", all_tokens_match)
    print(f"最大 Logits 绝对误差: {overall_max_error:.6f}")
    print(f"阈值检查: passed={passed}, atol={logits_atol}")

    if output:
        environment = build_environment(
            device,
            model=model,
            revision=revision,
            dtype=dtype,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
            logits_atol=logits_atol,
        )
        payload = {
            "environment": environment,
            "passed": passed,
            "results": results,
        }
        dump_json(output, payload)

    return 0 if passed else 1


__all__ = [
    "compare_paged_prompt",
    "execute_compare_paged_logits",
]
