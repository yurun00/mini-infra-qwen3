"""使用真实 Qwen3 权重比较 prefill-first 与 token-budget 两种迭代级调度策略的输出与轨迹。

入口：mini-infra verify schedulers-token-budget。
"""

from collections.abc import Sequence
from typing import Any

import torch

from mini_infra.core import (
    PagedCacheConfig,
    PrefillFirstConfig,
    SchedulePhase,
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
    prepare_cuda_model,
)

DEFAULT_PROMPTS = [
    "用一句话解释推理请求调度器。",
    "Explain token budgets in one short sentence.",
    "只回答数字：19 乘以 17 等于多少？",
    "用一句话说明 Prefill 和 Decode 的区别。",
    "用一句话解释 FCFS。",
    "Explain iteration-level scheduling briefly.",
]
DEFAULT_BUDGETS = [8, 8, 2, 8, 3, 8]
DEFAULT_ARRIVALS = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def _max_consecutive_prefill_steps(events: Sequence[ScheduleEvent]) -> int:
    longest = streak = 0
    for event in events:
        streak = streak + 1 if event.phase is SchedulePhase.PREFILL else 0
        longest = max(longest, streak)
    return longest


def _compact_trace(
    events: Sequence[ScheduleEvent],
    token_budget: int | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "step_idx": step_idx,
            "phase": event.phase.value,
            "admitted_request_ids": list(event.admitted_request_ids),
            "finished_request_ids": list(event.finished_request_ids),
            "num_scheduled_tokens": event.num_scheduled_tokens,
            "is_oversize_prefill": (
                event.phase is SchedulePhase.PREFILL
                and token_budget is not None
                and event.num_scheduled_tokens > token_budget
            ),
        }
        for step_idx, event in enumerate(events)
    ]


def _summarize_policy_metrics(run: ScheduleResult) -> dict[str, float | int]:
    sm = run.schedule_metrics
    bi = run.batch_info
    return {
        "makespan_ms": sm.makespan_ms,
        "busy_ms": sm.busy_ms,
        "queue_ms_p50": sm.queue_ms_p50,
        "queue_ms_p95": sm.queue_ms_p95,
        "ttft_ms_p50": sm.ttft_ms_p50,
        "ttft_ms_p95": sm.ttft_ms_p95,
        "itl_ms_p50": sm.itl_ms_p50,
        "itl_ms_p95": sm.itl_ms_p95,
        "itl_ms_max": sm.itl_ms_max,
        "padding_efficiency": bi.padding_efficiency if bi is not None else 0.0,
        "slot_utilization": bi.slot_utilization if bi is not None else 0.0,
        "max_consecutive_prefill_steps": _max_consecutive_prefill_steps(run.events),
    }


@torch.inference_mode()
def run_compare_schedulers(
    model: torch.nn.Module,
    sequences: Sequence[Sequence[int]],
    budgets: Sequence[int],
    arrivals: Sequence[float] | None = None,
    eos_token_id: int | None = None,
    token_budget: int = 64,
    block_size: int = 16,
    max_running_requests: int = 4,
    output: str | None = None,
    environment_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """比较 prefill-first 与 token-budget 两种迭代级调度策略的输出一致性、调度轨迹与聚合指标。"""
    requests = make_requests(
        sequences,
        budgets,
        arrival_times_ms=arrivals if arrivals is not None else [0.0 if i < 2 else 1.0 for i in range(len(sequences))],
    )
    base_config = EngineConfig(
        PagedCacheConfig(block_size=block_size),
        batching_method=BatchingMethod.CONTINUOUS,
        max_running_requests=max_running_requests,
    )

    prefill_first = run_scheduler(
        base_config.with_scheduler_config(PrefillFirstConfig()),
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
    )
    budgeted = run_scheduler(
        base_config.with_scheduler_config(TokenBudgetConfig(token_budget)),
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
    )

    prefill_first_token_ids = [s.token_ids for s in prefill_first.sequences]
    token_budget_token_ids = [s.token_ids for s in budgeted.sequences]
    is_match = prefill_first_token_ids == token_budget_token_ids
    prefill_first_metrics = _summarize_policy_metrics(prefill_first)
    token_budget_metrics = _summarize_policy_metrics(budgeted)

    result: dict[str, Any] = {
        "tokens_match": is_match,
        "prefill_first_token_ids": prefill_first_token_ids,
        "token_budget_token_ids": token_budget_token_ids,
        "prefill_first_trace": _compact_trace(prefill_first.events),
        "token_budget_trace": _compact_trace(budgeted.events, token_budget=token_budget),
        "prefill_first_metrics": prefill_first_metrics,
        "token_budget_metrics": token_budget_metrics,
    }

    print("prefill-first 与 token-budget 逐请求 Token 一致:", is_match)
    print(
        "最大连续 Prefill 步数: prefill-first=%d, token-budget=%d"
        % (
            prefill_first_metrics["max_consecutive_prefill_steps"],
            token_budget_metrics["max_consecutive_prefill_steps"],
        )
    )

    if output:
        dump_json(output, {"environment": environment_metadata or {}, "result": result})

    return result


def execute_compare_schedulers(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompts: Sequence[str] | None = None,
    max_new_tokens: int = 8,
    token_budget: int = 64,
    block_size: int = 16,
    max_running_requests: int = 4,
    dtype: str = "float32",
    output: str | None = None,
) -> int:
    """运行 prefill-first 与 token-budget 调度策略输出与轨迹一致性检查，返回退出码（0 成功，1 失败）。"""
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2")
    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    if token_budget < max_running_requests:
        raise ValueError("token_budget 不能小于 max_running_requests")
    if block_size < 1:
        raise ValueError("block_size 必须大于 0")

    device, _, tokenizer, model_instance = prepare_cuda_model(model, revision, dtype)
    prompts_to_use = list(prompts) if prompts is not None else DEFAULT_PROMPTS
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts_to_use]
    if len(prompts_to_use) == len(DEFAULT_PROMPTS):
        budgets = [
            max_new_tokens,
            max_new_tokens,
            DEFAULT_BUDGETS[2],
            max_new_tokens,
            min(DEFAULT_BUDGETS[4], max_new_tokens),
            max_new_tokens,
        ]
        arrivals = list(DEFAULT_ARRIVALS)
    else:
        budgets = [2 if i % 2 == 0 else max_new_tokens for i in range(len(prompts_to_use))]
        arrivals = [0.0 if i < 2 else 1.0 for i in range(len(prompts_to_use))]

    environment = build_environment(
        device,
        model=model,
        revision=revision,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        token_budget=token_budget,
        block_size=block_size,
        max_running_requests=max_running_requests,
        decoding="greedy, EOS disabled",
        thinking=False,
    )

    result = run_compare_schedulers(
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

    if dtype == "float32" and not result["tokens_match"]:
        print("float32 调度输出一致性检查失败")
        return 1
    return 0


__all__ = [
    "DEFAULT_ARRIVALS",
    "DEFAULT_BUDGETS",
    "DEFAULT_PROMPTS",
    "execute_compare_schedulers",
    "run_compare_schedulers",
]
