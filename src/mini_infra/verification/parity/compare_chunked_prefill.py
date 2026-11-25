"""使用真实 Qwen3 权重比较 token-budget（整段 Prefill）与 chunked-prefill 的输出、首 Token Logits 与调度轨迹。

入口：mini-infra verify chunked-prefill。
"""

from collections.abc import Sequence
from typing import Any

import torch

from mini_infra.core import (
    ChunkedPrefillConfig,
    PagedCacheConfig,
    SchedulerConfig,
    TokenBudgetConfig,
    make_requests,
)
from mini_infra.engine import (
    BatchingMethod,
    EngineConfig,
    ScheduleEvent,
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

DEFAULT_PROMPTS = [
    "用一句话解释 Chunked Prefill。",
    "请简要说明长提示词为什么会干扰正在进行的逐 Token 解码。" * 8,
    "Explain prefill and decode in one short sentence.",
    "只回答数字：19 乘以 17 等于多少？",
]
DEFAULT_BUDGETS = [8, 8, 8, 2]
DEFAULT_ARRIVALS = [0.0, 1.0, 1.0, 1.0]


def _compact_trace(events: Sequence[ScheduleEvent]) -> list[dict[str, Any]]:
    return [
        {
            "step_idx": step_idx,
            "phase": event.phase.value,
            "admitted_request_ids": list(event.admitted_request_ids),
            "finished_request_ids": list(event.finished_request_ids),
            "physical_batch_size": event.physical_batch_size,
            "num_scheduled_tokens": event.num_scheduled_tokens,
            "is_oversize_prefill": event.is_oversize_prefill,
        }
        for step_idx, event in enumerate(events)
    ]


def _summarize_policy_metrics(run: ScheduleResult) -> dict[str, float | int]:
    sm = run.schedule_metrics
    bi = run.batch_info
    return {
        "makespan_ms": sm.makespan_ms,
        "busy_ms": sm.busy_ms,
        "scheduler_ms": sm.scheduler_ms,
        "queue_ms_p50": sm.queue_ms_p50,
        "queue_ms_p95": sm.queue_ms_p95,
        "prefill_ms_p50": sm.prefill_ms_p50,
        "prefill_ms_p95": sm.prefill_ms_p95,
        "ttft_ms_p50": sm.ttft_ms_p50,
        "ttft_ms_p95": sm.ttft_ms_p95,
        "itl_ms_p50": sm.itl_ms_p50,
        "itl_ms_p95": sm.itl_ms_p95,
        "itl_ms_max": sm.itl_ms_max,
        "padding_efficiency": bi.padding_efficiency if bi is not None else 0.0,
        "slot_utilization": bi.slot_utilization if bi is not None else 0.0,
        "num_prefill_steps": sm.num_prefill_steps,
        "num_decode_steps": sm.num_decode_steps,
        "max_consecutive_prefill_steps": sm.max_consecutive_prefill_steps,
        "num_oversize_prefill_steps": sm.num_oversize_prefill_steps,
        "num_prefill_chunks": sm.num_prefill_chunks,
        "max_prefill_step_tokens": sm.max_prefill_step_tokens,
        "max_prefill_step_ms": sm.max_prefill_step_ms,
    }


@torch.inference_mode()
def run_compare_chunked_prefill(
    model: torch.nn.Module,
    sequences: Sequence[Sequence[int]],
    budgets: Sequence[int],
    arrivals: Sequence[float] | None = None,
    eos_token_id: int | None = None,
    token_budget: int = 32,
    block_size: int = 16,
    max_running_requests: int = 4,
    output: str | None = None,
    environment_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """比较 token-budget（整段 Prefill）与 chunked-prefill 的输出一致性、首 Token Logits 误差与调度轨迹。"""
    requests = make_requests(
        sequences,
        budgets,
        arrival_times_ms=arrivals
        if arrivals is not None
        else [0.0 if i == 0 else 1.0 for i in range(len(sequences))],
    )
    base_config = EngineConfig(
        PagedCacheConfig(block_size=block_size),
        batching_method=BatchingMethod.CONTINUOUS,
        scheduler_config=SchedulerConfig(
            max_running_requests=max_running_requests,
            batching_policy=TokenBudgetConfig(token_budget),
        ),
    )

    full = run_scheduler(
        base_config,
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
        should_capture_first_token_logits=True,
    )
    chunked = run_scheduler(
        base_config.with_batching_policy(ChunkedPrefillConfig(token_budget)),
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
        should_capture_first_token_logits=True,
    )

    full_token_ids = [s.token_ids for s in full.sequences]
    chunked_token_ids = [s.token_ids for s in chunked.sequences]
    is_match = full_token_ids == chunked_token_ids
    logit_errors = first_token_logit_errors(full, chunked)
    max_logit_error = max(logit_errors.values())
    full_metrics = _summarize_policy_metrics(full)
    chunked_metrics = _summarize_policy_metrics(chunked)

    result: dict[str, Any] = {
        "tokens_match": is_match,
        "max_first_token_logit_error": max_logit_error,
        "per_request_first_token_logit_error": logit_errors,
        "prompt_token_lengths": [len(sequence) for sequence in sequences],
        "full_token_ids": full_token_ids,
        "chunked_token_ids": chunked_token_ids,
        "full_trace": _compact_trace(full.events),
        "chunked_trace": _compact_trace(chunked.events),
        "full_metrics": full_metrics,
        "chunked_metrics": chunked_metrics,
    }

    print("token-budget 与 chunked-prefill 逐请求 Token 一致:", is_match)
    print("首 Token Logits 最大误差: %.8f" % max_logit_error)
    print(
        "oversize prefill 步数: token-budget=%d, chunked-prefill=%d"
        % (
            full_metrics["num_oversize_prefill_steps"],
            chunked_metrics["num_oversize_prefill_steps"],
        )
    )
    print(
        "单步最大 padded token: token-budget=%d, chunked-prefill=%d（chunk 总数: %d -> %d）"
        % (
            full_metrics["max_prefill_step_tokens"],
            chunked_metrics["max_prefill_step_tokens"],
            full_metrics["num_prefill_chunks"],
            chunked_metrics["num_prefill_chunks"],
        )
    )

    if output:
        dump_json(
            output,
            {"environment": environment_metadata or {}, "result": result},
        )

    return result


def execute_compare_chunked_prefill(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompts: Sequence[str] | None = None,
    max_new_tokens: int = 8,
    token_budget: int = 32,
    block_size: int = 16,
    max_running_requests: int = 4,
    dtype: str = "float32",
    logits_atol: float = 1e-4,
    output: str | None = None,
) -> int:
    """运行 token-budget 与 chunked-prefill 对齐验证，返回退出码（0 成功，1 失败）。"""
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2")
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
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts_to_use]
    if len(prompts_to_use) == len(DEFAULT_PROMPTS):
        budgets = [
            max_new_tokens,
            max_new_tokens,
            max_new_tokens,
            min(DEFAULT_BUDGETS[3], max_new_tokens),
        ]
        arrivals = list(DEFAULT_ARRIVALS)
    else:
        budgets = [
            2 if i == len(prompts_to_use) - 1 else max_new_tokens
            for i in range(len(prompts_to_use))
        ]
        arrivals = [0.0 if i == 0 else 1.0 for i in range(len(prompts_to_use))]

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

    result = run_compare_chunked_prefill(
        model=model_instance,
        sequences=sequences,
        budgets=budgets,
        arrivals=arrivals,
        eos_token_id=tokenizer.eos_token_id,
        token_budget=token_budget,
        block_size=block_size,
        max_running_requests=max_running_requests,
        output=output,
        environment_metadata=environment,
    )

    if dtype == "float32" and (
        not result["tokens_match"] or result["max_first_token_logit_error"] > logits_atol
    ):
        print("float32 Chunked Prefill 正确性检查失败")
        return 1
    return 0


__all__ = [
    "DEFAULT_ARRIVALS",
    "DEFAULT_BUDGETS",
    "DEFAULT_PROMPTS",
    "execute_compare_chunked_prefill",
    "run_compare_chunked_prefill",
]
