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
    KVCacheManager,
    PagedCacheConfig,
    Request,
    RequestState,
    SchedulerConfig,
    make_requests,
    num_blocks_for_tokens,
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
    DenseAttentionMetadata,
)
from mini_infra.verification.parity._common import (
    build_environment,
    dump_json,
    prepare_cuda_model,
)
from mini_infra.worker import KVCache, PagedBlockCache, PagedRunner

_SHARED_CONTEXT = (
    "请先阅读以下共同背景：Prefix Cache 会复用多个请求相同前缀已经计算出的 KV Cache，从而减少后续请求的 Prefill 计算。"
    * 8
)
DEFAULT_PROMPTS = [
    _SHARED_CONTEXT + "\n现在请用一句话概括它的主要收益。",
    _SHARED_CONTEXT + "\n现在请用一句话说明它的显存代价。",
    _SHARED_CONTEXT + "\n现在请用一句话说明它适合哪些业务场景。",
]


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
        "ttft_ms_p50": sm.ttft_ms_p50,
        "ttft_ms_p95": sm.ttft_ms_p95,
        "num_prefill_steps": sm.num_prefill_steps,
        "num_decode_steps": sm.num_decode_steps,
        "num_prefill_chunks": sm.num_prefill_chunks,
    }


def _prefill_logits(
    model: torch.nn.Module,
    runner: PagedRunner,
    block_cache: PagedBlockCache,
    token_ids: Sequence[int],
    start: int,
    block_table: Sequence[int],
) -> torch.Tensor:
    """单请求 Prompt 区间 [start, len(token_ids)) 的 Prefill：历史 K/V 经 block table 读回，本段 K/V 写回。

    返回末位 logits（[vocab]，float32，CPU）。
    """
    device = next(model.parameters()).device
    block_size = block_cache.block_size
    chunk_len = len(token_ids) - start
    chunk_ids = torch.tensor([token_ids[start:]], dtype=torch.long, device=device)
    position_ids = torch.arange(
        start, len(token_ids), dtype=torch.long, device=device
    ).unsqueeze(0)
    attention_mask = torch.ones((1, chunk_len), dtype=torch.bool, device=device)
    attn_metadata = None
    if start > 0:
        num_history_blocks = num_blocks_for_tokens(start, block_size)
        history_block_ids = torch.tensor(
            [block_table[:num_history_blocks]], dtype=torch.long, device=device
        )
        history_mask = (
            torch.arange(num_history_blocks * block_size, device=device).unsqueeze(0)
            < start
        )
        attention_mask = torch.cat((history_mask, attention_mask), dim=1)
        attn_metadata = DenseAttentionMetadata(
            past_key_values=block_cache.gather_blocks(history_block_ids)
        )
    logits, present = model(
        chunk_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        attn_metadata=attn_metadata,
    )
    runner.store_prefill(
        block_cache, KVCache(present), [block_table], [chunk_len], [start]
    )
    return logits[0, -1, :].float().cpu()


def _compare_first_token_logits(
    model: torch.nn.Module,
    sequences: Sequence[Sequence[int]],
    block_size: int,
) -> tuple[dict[str, float], dict[str, int]]:
    """逐请求对比「命中前序请求 cached block 后只算剩余 Prompt」与「整段 Prompt 无缓存 Prefill」的首 Token Logits。

    直接驱动 KVCacheManager（allocate_computed_blocks / cache_blocks）与 PagedBlockCache，复用生产路径的
    prefix cache 查找与 block table 寻址；返回 (逐请求最大绝对误差, 逐请求命中 Token 数)。
    """
    param = next(model.parameters())
    config = model.config  # type: ignore[attr-defined]
    num_blocks = 1 + sum(
        num_blocks_for_tokens(len(sequence), block_size) for sequence in sequences
    )
    block_cache = PagedBlockCache(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=param.dtype,
        device=param.device,
    )
    manager = KVCacheManager(block_size, num_blocks, is_prefix_caching_enabled=True)
    runner = PagedRunner(model)
    errors: dict[str, float] = {}
    prefix_cache_hit_tokens: dict[str, int] = {}

    for request_idx, sequence in enumerate(sequences):
        request = Request(request_idx, tuple(sequence), 1)
        full_logits, _ = model(
            torch.tensor([sequence], dtype=torch.long, device=param.device)
        )
        start = manager.allocate_computed_blocks(request)
        if not manager.allocate_slots(request_idx, request.prompt_len - start):
            raise RuntimeError(f"请求 {request_idx} 分配物理块失败")
        cached_logits = _prefill_logits(
            model,
            runner,
            block_cache,
            sequence,
            start,
            manager.block_table(request_idx),
        )
        manager.cache_blocks(RequestState(request, cache_length=request.prompt_len))
        errors[str(request_idx)] = float(
            torch.max(
                torch.abs(full_logits[0, -1, :].float().cpu() - cached_logits)
            ).item()
        )
        prefix_cache_hit_tokens[str(request_idx)] = start

    return errors, prefix_cache_hit_tokens


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
    )
    enabled = run_scheduler(
        enabled_config,
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
    )

    disabled_token_ids = [s.token_ids for s in disabled.sequences]
    enabled_token_ids = [s.token_ids for s in enabled.sequences]
    is_match = disabled_token_ids == enabled_token_ids
    logit_errors, logit_prefix_cache_hit_tokens = _compare_first_token_logits(
        model, sequences, block_size
    )
    disabled_metrics = _summarize_run(disabled)
    enabled_metrics = _summarize_run(enabled)

    result: dict[str, Any] = {
        "tokens_match": is_match,
        "max_first_token_logit_error": max(logit_errors.values()),
        "per_request_first_token_logit_error": logit_errors,
        "per_request_logit_prefix_cache_hit_tokens": logit_prefix_cache_hit_tokens,
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
    logits_atol: float = 1e-4,
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
    if logits_atol < 0:
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
        logits_atol=logits_atol,
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
        output=output,
        environment_metadata=environment,
    )

    has_hits = sum(result["enabled_metrics"]["per_request_prefix_cache_hit_tokens"]) > 0
    if not has_hits:
        print(
            "prefix caching 未产生任何命中：请确认各 prompt 共享不少于 block_size 个 Token 的前缀"
        )
        return 1
    if dtype == "float32" and (
        not result["tokens_match"] or result["max_first_token_logit_error"] > logits_atol
    ):
        print("float32 Prefix Caching 正确性检查失败")
        return 1
    return 0


__all__ = [
    "DEFAULT_PROMPTS",
    "execute_compare_prefix_caching",
    "run_compare_prefix_caching",
]
