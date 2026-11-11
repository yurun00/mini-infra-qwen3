"""用真实 Qwen3 权重检查 Scheduler 批处理（Fixed 与 Continuous）的请求归属和输出一致性。

入口：mini-infra verify scheduler-batch。
"""

from collections.abc import Sequence
from typing import Any

import torch

from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.core.request import make_requests
from mini_infra.engine import BatchingMethod, CachingMethod, run_scheduler
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
    "用一句话解释 Continuous Batching。",
    "用一句话解释 Prefill。",
    "用一句话解释 Decode。",
    "用一句话解释 KV Cache。",
]


@torch.inference_mode()
def run_compare_scheduler_batch(
    model: torch.nn.Module,
    sequences: Sequence[Sequence[int]],
    budgets: Sequence[int],
    eos_token_id: int | None,
    device: torch.device,
    max_running_requests: int = 2,
    output: str | None = None,
    environment_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行单请求隔离、固定批处理、连续批处理与 Paged KV Cache 连续批处理对比，返回比对统计结果字典。"""
    requests = make_requests(sequences, budgets)

    # 1. 串行调度基准（serialized baseline，并发槽位为 1）
    serialized = run_scheduler(
        BatchingMethod.NONE,
        model,
        requests,
        max_running_requests=1,
        eos_token_id=eos_token_id,
        device=device,
        caching_method=CachingMethod.DENSE,
        stop_on_eos=False,
    )

    # 2. 固定批处理 (Fixed Batching)
    fixed = run_scheduler(
        BatchingMethod.FIXED,
        model,
        requests,
        max_running_requests=max_running_requests,
        eos_token_id=eos_token_id,
        device=device,
        caching_method=CachingMethod.DENSE,
        stop_on_eos=False,
    )

    # 3. 连续批处理 (Continuous Batching)
    continuous = run_scheduler(
        BatchingMethod.CONTINUOUS,
        model,
        requests,
        max_running_requests=max_running_requests,
        eos_token_id=eos_token_id,
        device=device,
        caching_method=CachingMethod.DENSE,
        stop_on_eos=False,
    )

    # 4. 连续批处理 + Paged KV Cache（块大小取 4，使短序列也会跨块、块被回收复用）
    paged = run_scheduler(
        BatchingMethod.CONTINUOUS,
        model,
        requests,
        max_running_requests=max_running_requests,
        eos_token_id=eos_token_id,
        device=device,
        caching_method=CachingMethod.PAGED,
        stop_on_eos=False,
        paged_cache_config=PagedCacheConfig(block_size=4),
    )

    serialized_token_ids = [s.token_ids for s in serialized.sequences]
    fixed_token_ids = [s.token_ids for s in fixed.sequences]
    continuous_token_ids = [s.token_ids for s in continuous.sequences]
    paged_token_ids = [s.token_ids for s in paged.sequences]
    admission_trace = [list(event.admitted_request_ids) for event in continuous.events if event.admitted_request_ids]

    result: dict[str, Any] = {
        "serialized_token_ids": serialized_token_ids,
        "fixed_token_ids": fixed_token_ids,
        "continuous_token_ids": continuous_token_ids,
        "paged_token_ids": paged_token_ids,
        "fixed_matches_serialized": fixed_token_ids == serialized_token_ids,
        "continuous_matches_serialized": continuous_token_ids == serialized_token_ids,
        "fixed_matches_continuous": fixed_token_ids == continuous_token_ids,
        "paged_matches_serialized": paged_token_ids == serialized_token_ids,
        "paged_matches_continuous": paged_token_ids == continuous_token_ids,
        "continuous_admission_trace": admission_trace,
    }

    print("fixed 与串行单请求一致:", result["fixed_matches_serialized"])
    print("continuous 与串行单请求一致:", result["continuous_matches_serialized"])
    print("fixed 与 continuous 一致:", result["fixed_matches_continuous"])
    print("paged 与串行单请求一致:", result["paged_matches_serialized"])
    print("paged 与 continuous 一致:", result["paged_matches_continuous"])
    print("Continuous admission trace:", result["continuous_admission_trace"])

    if output:
        dump_json(output, {"environment": environment_metadata or {}, "result": result})

    return result


def execute_compare_scheduler_batch(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompts: Sequence[str] | None = None,
    max_new_tokens: int = 8,
    max_running_requests: int = 2,
    dtype: str = "float32",
    output: str | None = None,
) -> int:
    """运行 Scheduler 批处理动态请求归属与输出一致性验证，返回退出码（0 成功，1 失败）。"""
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2")
    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")

    device, _, tokenizer, model_instance = prepare_cuda_model(model, revision, dtype)
    prompts_to_use = list(prompts) if prompts is not None else DEFAULT_PROMPTS
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts_to_use]
    if len(prompts_to_use) == 4:
        budgets = [2, max_new_tokens, 3, max_new_tokens]
    else:
        budgets = [2 if i % 2 == 0 else max_new_tokens for i in range(len(prompts_to_use))]

    environment = build_environment(
        device,
        model=model,
        revision=revision,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        max_running_requests=max_running_requests,
        decoding="greedy, EOS disabled",
        thinking=False,
    )

    result = run_compare_scheduler_batch(
        model=model_instance,
        sequences=sequences,
        budgets=budgets,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        max_running_requests=max_running_requests,
        output=output,
        environment_metadata=environment,
    )

    if dtype == "float32" and not (result["continuous_matches_serialized"] and result["paged_matches_serialized"]):
        print("float32 动态请求归属或输出一致性检查失败")
        return 1
    return 0


__all__ = [
    "DEFAULT_PROMPTS",
    "execute_compare_scheduler_batch",
    "run_compare_scheduler_batch",
]
