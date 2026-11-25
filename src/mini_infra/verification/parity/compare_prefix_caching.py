"""使用真实 Qwen3 权重比较 prefix caching 关闭 / 开启时的输出、首 Token Logits 与 Prefill 计算量。

入口：mini-infra verify prefix-caching（对应参考实现 chapter_08 的 run_inference.py / compare_prefix_cache.py）。
两条路径共用完全相同的调度配置（chunked-prefill、FCFS、硬 token budget 与 decode protection），唯一变量是
PagedCacheConfig.is_prefix_caching_enabled。
"""

from collections.abc import Sequence
from typing import Any

import torch

from mini_infra.core import (
    ChunkedPrefillConfig,
    PagedCacheConfig,
    SchedulerConfig,
    make_requests,
)
from mini_infra.engine import (
    BatchingMethod,
    EngineConfig,
    ScheduleResult,
    run_scheduler,
)
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)
from mini_infra.verification.parity._common import (
    build_environment,
    dump_json,
    first_token_logit_errors,
    prepare_cuda_model,
)

# 单组 4 条自然语言 Prompt 在 Qwen3Tokenizer 下同时覆盖四种边界（默认 block_size=16）：
# - Prompt 0：冷启动建立缓存 -> 命中 0；
# - Prompt 1：与 Prompt 0 共享精确 288 Token（18 个整块对齐前缀） -> 命中 288；
# - Prompt 2：与 Prompt 1 共享 290 Token（非块对齐前缀） -> 截断至整块命中 288；
# - Prompt 3：首块（16 Token）不同、后续 17 个满块（[16:288]）与前序请求逐位相同 -> 链式 block hash 隔离命中 0，
#   并在紧凑池容量下触发空闲 cached block 的惰性 LRU 淘汰。
_HEADER_A = "甲组背景（高并发多租户推理服务）：\n"
_HEADER_B = "乙组背景（长文档多轮问答推理系统）：\n"
_SHARED_BODY = (
    "Prefix Cache 会复用多个请求相同前缀已经计算出的 KV Cache，从而减少后续请求的 Prefill 计算。"
    * 9
    + "请阅读上述公共背景后简要作答：\n"
)
DEFAULT_PROMPTS = [
    _HEADER_A + _SHARED_BODY + "概括它的主要收益。",
    _HEADER_A + _SHARED_BODY + "说明它的显存代价。",
    _HEADER_A + _SHARED_BODY + "说明它的核心权衡。",
    _HEADER_B + _SHARED_BODY + "分析它的适用场景。",
]


def _common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    for token_idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return token_idx
    return min(len(left), len(right))


def _expected_sequential_hits(
    sequences: Sequence[Sequence[int]], block_size: int
) -> list[int]:
    """顺序单并发（前序公共前缀仍在缓存中）下各请求的理论 prefix cache 命中 Token 数。"""
    expected: list[int] = []
    for request_idx, seq in enumerate(sequences):
        max_reusable = max(0, len(seq) - 1)
        best_lcp = max(
            (_common_prefix_len(seq, prev) for prev in sequences[:request_idx]),
            default=0,
        )
        expected.append((min(best_lcp, max_reusable) // block_size) * block_size)
    return expected


def _summarize_run(run: ScheduleResult) -> dict[str, Any]:
    sm = run.schedule_metrics
    profile = run.memory.kv_cache.paged_cache
    assert profile is not None
    num_prompt_tokens = sum(s.prompt_len for s in run.sequences)
    return {
        "per_request_prefix_cache_hit_tokens": [
            s.prefix_cache_hit_tokens for s in run.sequences
        ],
        "num_computed_prompt_tokens": num_prompt_tokens
        - sum(s.prefix_cache_hit_tokens for s in run.sequences),
        "num_prompt_tokens": num_prompt_tokens,
        "prefix_cache_hit_tokens": profile.prefix_cache_hit_tokens,
        "prefix_cache_prompt_tokens": profile.prefix_cache_prompt_tokens,
        "prefix_cache_block_eviction_count": profile.prefix_cache_block_eviction_count,
        "peak_used_blocks": profile.peak_used_blocks,
        "num_blocks": profile.num_blocks,
        "makespan_ms": sm.makespan_ms,
        "busy_ms": sm.busy_ms,
        "prefill_ms_p50": sm.prefill_ms_p50,
        "prefill_ms_p95": sm.prefill_ms_p95,
        "ttft_ms_p50": sm.ttft_ms_p50,
        "ttft_ms_p95": sm.ttft_ms_p95,
        "num_prefill_steps": sm.num_prefill_steps,
        "num_decode_steps": sm.num_decode_steps,
        "num_prefill_chunks": sm.num_prefill_chunks,
    }


@torch.inference_mode()
def run_compare_prefix_caching(
    model: torch.nn.Module,
    sequences: Sequence[Sequence[int]],
    budgets: Sequence[int],
    arrivals: Sequence[float] | None = None,
    eos_token_id: int | None = None,
    token_budget: int = 64,
    block_size: int = 16,
    max_running_requests: int = 1,
    output: str | None = None,
    environment_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """比较 prefix caching 关闭 / 开启的逐请求输出、首 Token Logits 误差、命中 Token 数与 Prefill 计算量。"""
    requests = make_requests(sequences, budgets, arrival_times_ms=arrivals)
    disabled_config = EngineConfig(
        PagedCacheConfig(block_size=block_size),
        batching_method=BatchingMethod.CONTINUOUS,
        scheduler_config=SchedulerConfig(
            max_running_requests=max_running_requests,
            batching_policy=ChunkedPrefillConfig(token_budget),
        ),
    )
    enabled_config = disabled_config.with_kv_cache_config(
        PagedCacheConfig(block_size=block_size, is_prefix_caching_enabled=True)
    )
    disabled = run_scheduler(
        disabled_config,
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
        should_capture_first_token_logits=True,
    )
    enabled = run_scheduler(
        enabled_config,
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
        should_capture_first_token_logits=True,
    )

    disabled_token_ids = [s.token_ids for s in disabled.sequences]
    enabled_token_ids = [s.token_ids for s in enabled.sequences]
    is_match = disabled_token_ids == enabled_token_ids
    logit_errors = first_token_logit_errors(disabled, enabled)
    disabled_metrics = _summarize_run(disabled)
    enabled_metrics = _summarize_run(enabled)

    result: dict[str, Any] = {
        "tokens_match": is_match,
        "max_first_token_logit_error": max(logit_errors.values()),
        "per_request_first_token_logit_error": logit_errors,
        "prompt_token_lengths": [len(sequence) for sequence in sequences],
        "disabled_token_ids": disabled_token_ids,
        "enabled_token_ids": enabled_token_ids,
        "disabled_metrics": disabled_metrics,
        "enabled_metrics": enabled_metrics,
    }

    print("prefix caching 关闭 / 开启逐请求 Token 一致:", is_match)
    print("首 Token Logits 最大误差: %.8f" % result["max_first_token_logit_error"])
    print(
        "逐请求 prefix cache 命中 Token 数:",
        enabled_metrics["per_request_prefix_cache_hit_tokens"],
    )
    print(
        "实际 Prefill 计算的 Prompt Token 数: %d -> %d（共 %d）"
        % (
            disabled_metrics["num_computed_prompt_tokens"],
            enabled_metrics["num_computed_prompt_tokens"],
            enabled_metrics["num_prompt_tokens"],
        )
    )
    print(
        "Prefill 延迟 p50: %.2f ms -> %.2f ms"
        % (disabled_metrics["prefill_ms_p50"], enabled_metrics["prefill_ms_p50"])
    )
    print(
        "TTFT p50: %.2f ms -> %.2f ms"
        % (disabled_metrics["ttft_ms_p50"], enabled_metrics["ttft_ms_p50"])
    )

    if output:
        dump_json(
            output,
            {"environment": environment_metadata or {}, "result": result},
        )

    return result


def execute_compare_prefix_caching(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompts: Sequence[str] | None = None,
    max_new_tokens: int = 8,
    token_budget: int = 64,
    block_size: int = 16,
    max_running_requests: int = 1,
    dtype: str = "float32",
    logits_atol: float | None = None,
    output: str | None = None,
) -> int:
    """运行 prefix caching 关闭 / 开启对齐验证，返回退出码（0 成功，1 失败）。"""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须大于 0")
    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    if token_budget < max_running_requests:
        raise ValueError("token_budget 不能小于 max_running_requests")
    if block_size < 1:
        raise ValueError("block_size 必须大于 0")
    resolved_atol = (
        (1e-4 if dtype == "float32" else 1.0) if logits_atol is None else logits_atol
    )
    if resolved_atol < 0:
        raise ValueError("logits_atol 不能为负数")

    device, _, tokenizer, model_instance = prepare_cuda_model(model, revision, dtype)
    prompts_to_use = list(prompts) if prompts is not None else DEFAULT_PROMPTS
    if len(prompts_to_use) < 2:
        raise ValueError("prefix caching 对照至少需要 2 条共享前缀的 prompt")
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts_to_use]
    environment = build_environment(
        device,
        model=model,
        revision=revision,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        token_budget=token_budget,
        block_size=block_size,
        max_running_requests=max_running_requests,
        logits_atol=resolved_atol,
        decoding="greedy, EOS disabled",
        thinking=False,
    )

    result = run_compare_prefix_caching(
        model=model_instance,
        sequences=sequences,
        budgets=[max_new_tokens] * len(sequences),
        eos_token_id=tokenizer.eos_token_id,
        token_budget=token_budget,
        block_size=block_size,
        max_running_requests=max_running_requests,
    )

    actual_hits = result["enabled_metrics"]["per_request_prefix_cache_hit_tokens"]
    is_expected_hits_ok = True
    if prompts is None and max_running_requests == 1:
        expected_hits = _expected_sequential_hits(sequences, block_size)
        result["expected_prefix_cache_hit_tokens"] = expected_hits
        is_expected_hits_ok = actual_hits == expected_hits

    if output:
        dump_json(
            output,
            {"environment": environment, "result": result},
        )

    if sum(actual_hits) == 0:
        print(
            "prefix caching 未产生任何命中：请确认各 prompt 共享不少于 block_size 个 Token 的前缀"
        )
        return 1
    is_tokens_ok = result["tokens_match"] if dtype == "float32" else True
    if (
        not is_tokens_ok
        or not is_expected_hits_ok
        or result["max_first_token_logit_error"] > resolved_atol
    ):
        print(f"{dtype} Prefix Caching 正确性检查失败")
        return 1
    return 0


__all__ = [
    "DEFAULT_PROMPTS",
    "execute_compare_prefix_caching",
    "run_compare_prefix_caching",
]
