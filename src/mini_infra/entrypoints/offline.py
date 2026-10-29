"""文本生成用户级入口（Generation Entrypoint）。

对标 vLLM entrypoints/llm.py，仅负责：
- 解析与校验用户级配置（GenerationOptions）与多样化 Prompt 输入；
- 按 Mode 分发到底层引擎执行（run_lockstep / run_scheduler）；
- 包装 Profiler 上下文并组装返回 GenerationResult。
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from mini_infra.core.request import make_request_specs
from mini_infra.device import synchronize
from mini_infra.engine.lockstep_loop import run_lockstep
from mini_infra.engine.mode import BatchingMethod, FeatureFlag, Mode
from mini_infra.engine.preprocessing import (
    normalize_budgets,
    prepare_prompt_sequences,
)
from mini_infra.engine.scheduler_loop import run_scheduler
from mini_infra.engine.types import GenerationResult, LockstepResult, ScheduleResult
from mini_infra.entrypoints.types import (
    GenerationOptions,
    aggregate_generation_results,
)
from mini_infra.profiling import profile_scope


@torch.inference_mode()
def generate(
    mode: Mode,
    model: torch.nn.Module,
    input_ids: torch.Tensor | Sequence[Any],
    max_new_tokens: int | Sequence[int],
    options: GenerationOptions | None = None,
) -> GenerationResult:
    """统一生成入口；mode 为 Mode 结构体对象。

    所有可选控制项（device、eos_token_id、stop_on_eos、profile、max_running_requests 等）
    统一通过 options（GenerationOptions 结构体）传入。
    不同 mode 对应不同的 feature 开关，但生成的 output token 完全一致。
    """
    opts = options or GenerationOptions()

    # 1. 规约输入 Prompt、预算与设备
    token_sequences, target_device, has_uniform_len = prepare_prompt_sequences(input_ids, opts.device)
    budgets = normalize_budgets(max_new_tokens, len(token_sequences), target_device)

    # 2. Continuous Batching 分发（不再静默回退到固定批次路径）
    if mode.batching_method == BatchingMethod.CONTINUOUS:
        return _dispatch_continuous(
            mode,
            model,
            token_sequences,
            budgets,
            target_device,
            opts,
        )

    # 3. 串行 baseline 分发
    if mode.batching_method == BatchingMethod.NONE and len(token_sequences) > 1:
        return _dispatch_serial_baseline(
            mode,
            model,
            token_sequences,
            budgets,
            target_device,
            opts,
        )

    # 4. 齐步（Lockstep）执行（所有请求同时开始，无队列调度）
    return _dispatch_lockstep(
        mode,
        model,
        input_ids,
        token_sequences,
        has_uniform_len,
        budgets,
        target_device,
        opts,
    )


def _dispatch_lockstep(
    mode: Mode,
    model: torch.nn.Module,
    input_ids: torch.Tensor | Sequence[Any],
    token_sequences: list[torch.Tensor],
    has_uniform_len: bool,
    budgets: torch.Tensor,
    target_device: torch.device,
    opts: GenerationOptions,
) -> LockstepResult:
    """以齐步（Lockstep）循环执行（支持有/无 KV Cache、Profile、Logits 捕获等）。"""
    is_cuda = target_device.type == "cuda"
    use_cache = mode.is_enabled(FeatureFlag.KV_CACHE)

    synchronize(target_device)
    baseline_allocated_bytes = torch.cuda.memory_allocated(target_device) if is_cuda else 0
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(target_device)

    loop_kwargs = dict(
        model=model,
        input_ids=input_ids,
        token_sequences=token_sequences,
        has_uniform_len=has_uniform_len,
        budgets=budgets,
        use_cache=use_cache,
        device=target_device,
        pad_token_id=opts.pad_token_id,
        eos_token_id=opts.eos_token_id,
        stop_on_eos=opts.stop_on_eos,
        capture_logits=opts.capture_logits,
        baseline_allocated_bytes=baseline_allocated_bytes,
    )

    if opts.profile:
        with profile_scope(model, device=target_device, tensorboard_options=opts.tensorboard_options) as prof_collector:
            gen_result = run_lockstep(**loop_kwargs)
        op_latency_breakdown = prof_collector.get_latency_breakdown(top=opts.top_kernels)
        return replace(gen_result, op_breakdown=op_latency_breakdown)
    return run_lockstep(**loop_kwargs)


def _dispatch_serial_baseline(
    mode: Mode,
    model: torch.nn.Module,
    token_sequences: list[torch.Tensor],
    budgets: torch.Tensor,
    device: torch.device,
    opts: GenerationOptions,
) -> GenerationResult:
    """按顺序逐个执行请求（串行 baseline 模式）。"""
    single_req_options = GenerationOptions(
        eos_token_id=opts.eos_token_id,
        device=device,
        stop_on_eos=opts.stop_on_eos,
        pad_token_id=opts.pad_token_id,
        capture_logits=opts.capture_logits,
        profile=opts.profile,
        top_kernels=opts.top_kernels,
        tensorboard_options=opts.tensorboard_options,
    )
    runs = [
        generate(
            mode,
            model,
            seq.to(device=device, dtype=torch.long).unsqueeze(0),
            budget,
            options=single_req_options,
        )
        for seq, budget in zip(token_sequences, budgets.tolist())
    ]
    return aggregate_generation_results(runs)


def _dispatch_continuous(
    mode: Mode,
    model: torch.nn.Module,
    token_sequences: list[torch.Tensor],
    budgets: torch.Tensor,
    device: torch.device,
    opts: GenerationOptions,
) -> ScheduleResult:
    """以 Continuous Batching 执行（支持自定义并发上限），直接返回 ScheduleResult。"""
    if not mode.is_enabled(FeatureFlag.KV_CACHE):
        raise ValueError("continuous-batching 依赖 KV Cache，请启用 FeatureFlag.KV_CACHE")
    if opts.capture_logits or opts.profile:
        raise ValueError("continuous-batching 暂不支持 capture_logits/profile")
    if opts.max_running_requests is not None and opts.max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    specs = make_request_specs(token_sequences, budgets.tolist())
    concurrency = opts.max_running_requests if opts.max_running_requests is not None else len(specs)
    return run_scheduler(
        BatchingMethod.CONTINUOUS,
        model,
        specs,
        concurrency,
        opts.eos_token_id,
        device,
        pad_token_id=opts.pad_token_id,
        stop_on_eos=opts.stop_on_eos,
    )


__all__ = [
    "GenerationOptions",
    "GenerationResult",
    "generate",
]
