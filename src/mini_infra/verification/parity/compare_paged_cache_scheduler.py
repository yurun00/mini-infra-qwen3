"""使用真实 Qwen3 权重在 Scheduler 调度环境下验证 Dense 与 Paged KV Cache 的输出一致性。"""

from collections.abc import Sequence
from typing import Any

import torch

from mini_infra.core import DenseCacheConfig, PagedCacheConfig, make_requests
from mini_infra.engine import (
    BatchingMethod,
    Qwen3Tokenizer,
    run_scheduler,
)
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)
from mini_infra.verification.parity._common import (
    build_environment,
    dump_json,
    prepare_cuda_model,
)

DEFAULT_PROMPTS = [
    "用一句话解释 Paged KV Cache。",
    "只回答数字：17 乘以 23 等于多少？",
    "Explain KV cache in one short sentence.",
    "用一句话解释物理块和逻辑块的区别。",
]
DEFAULT_BUDGETS = [2, 8, 3, 8]
DEFAULT_ARRIVALS = [0.0, 0.0, 1.0, 2.0]


@torch.inference_mode()
def run_compare_paged_cache_scheduler(
    model: torch.nn.Module,
    tokenizer: Qwen3Tokenizer,
    device: torch.device,
    prompts: Sequence[str] | None = None,
    max_new_tokens: int = 8,
    block_sizes: Sequence[int] = (4, 16),
    max_running_requests: int = 2,
    output: str | None = None,
    environment_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在 Scheduler 调度环境下对比 Dense 与多个 Block Size 的 Paged KV Cache 输出一致性。"""
    prompts_to_run = list(prompts) if prompts else DEFAULT_PROMPTS
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts_to_run]

    # 根据请求数量调整 budgets 和 arrivals
    if len(prompts_to_run) == len(DEFAULT_PROMPTS):
        budgets = [DEFAULT_BUDGETS[0], max_new_tokens, DEFAULT_BUDGETS[2], max_new_tokens]
        arrivals = list(DEFAULT_ARRIVALS)
    else:
        budgets = [max_new_tokens] * len(prompts_to_run)
        arrivals = [float(i) for i in range(len(prompts_to_run))]

    requests = make_requests(sequences, budgets, arrival_times_ms=arrivals)

    # 1. Dense KV Cache 连续批处理基准
    dense = run_scheduler(
        BatchingMethod.CONTINUOUS,
        model,
        requests,
        max_running_requests=max_running_requests,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        kv_cache_config=DenseCacheConfig(),
        stop_on_eos=False,
    )
    dense_token_ids = [s.token_ids for s in dense.sequences]

    # 2. 对每个 Block Size 运行 Paged KV Cache 连续批处理
    paged_results: dict[str, Any] = {}
    for block_size in block_sizes:
        paged_run = run_scheduler(
            BatchingMethod.CONTINUOUS,
            model,
            requests,
            max_running_requests=max_running_requests,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
            kv_cache_config=PagedCacheConfig(block_size=block_size),
            stop_on_eos=False,
        )
        paged_token_ids = [s.token_ids for s in paged_run.sequences]
        matches_dense = paged_token_ids == dense_token_ids
        admission_trace = [list(event.admitted_request_ids) for event in paged_run.events if event.admitted_request_ids]
        paged_profile = paged_run.memory.kv_cache.paged_cache
        block_reuse_count = paged_profile.block_reuse_count if paged_profile is not None else 0

        paged_results[str(block_size)] = {
            "token_ids": paged_token_ids,
            "matches_dense": matches_dense,
            "admission_trace": admission_trace,
            "block_reuse_count": block_reuse_count,
        }

    all_match_dense = all(item["matches_dense"] for item in paged_results.values())
    result: dict[str, Any] = {
        "dense_token_ids": dense_token_ids,
        "paged": paged_results,
        "all_match_dense": all_match_dense,
    }

    print("全部 Block Size 与 dense Token 一致:", all_match_dense)
    for block_size_str, item in paged_results.items():
        print(
            "block=%s, match=%s, reuse=%d, admissions=%s"
            % (
                block_size_str,
                item["matches_dense"],
                item["block_reuse_count"],
                item["admission_trace"],
            )
        )

    if output:
        payload = {
            "environment": environment_metadata or {},
            "result": result,
        }
        dump_json(output, payload)

    return result


def execute_compare_paged_cache_scheduler(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompts: Sequence[str] | None = None,
    max_new_tokens: int = 8,
    block_sizes: Sequence[int] = (4, 16),
    max_running_requests: int = 2,
    dtype: str = "float32",
    output: str | None = None,
) -> int:
    """运行 Paged KV Cache Scheduler 输出一致性检查，返回退出码（0 成功，1 失败）。"""
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2")
    if any(size < 1 for size in block_sizes):
        raise ValueError("block_sizes 元素必须大于 0")
    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")

    device, _, tokenizer, model_instance = prepare_cuda_model(model, revision, dtype)
    environment = build_environment(
        device,
        model=model,
        revision=revision,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        block_sizes=list(block_sizes),
        max_running_requests=max_running_requests,
        decoding="greedy, EOS disabled",
        thinking=False,
    )

    result = run_compare_paged_cache_scheduler(
        model=model_instance,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        block_sizes=block_sizes,
        max_running_requests=max_running_requests,
        output=output,
        environment_metadata=environment,
    )

    if dtype == "float32" and not result["all_match_dense"]:
        print("float32 Paged KV Cache 输出一致性检查失败")
        return 1

    return 0 if result["all_match_dense"] else 1


__all__ = [
    "DEFAULT_ARRIVALS",
    "DEFAULT_BUDGETS",
    "DEFAULT_PROMPTS",
    "execute_compare_paged_cache_scheduler",
    "run_compare_paged_cache_scheduler",
]
