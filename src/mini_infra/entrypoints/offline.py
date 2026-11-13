"""文本生成用户级入口（Generation Entrypoint）。

对标 vLLM entrypoints/llm.py，仅负责：
- 解析与校验用户级配置（EngineConfig / GenerationOptions）与多样化 Prompt 输入；
- 按 EngineConfig 分发到底层引擎执行（run_lockstep / run_scheduler）；
- 包装 Profiler 上下文并组装返回 GenerationResult。
"""

from collections.abc import Sequence

import torch

from mini_infra.core import PagedCacheConfig, make_requests
from mini_infra.engine.config import BatchingMethod, EngineConfig, validate_engine_config
from mini_infra.engine.lockstep_loop import run_lockstep
from mini_infra.engine.preprocessing import (
    normalize_budgets,
    normalize_token_sequences,
)
from mini_infra.engine.results import (
    GenerationResult,
    ScheduleResult,
    aggregate_generation_results,
)
from mini_infra.engine.scheduler_loop import run_scheduler
from mini_infra.entrypoints.types import GenerationOptions, PromptInput


@torch.inference_mode()
def generate(
    config: EngineConfig,
    model: torch.nn.Module,
    prompts: PromptInput | Sequence[PromptInput],
    max_new_tokens: int | Sequence[int],
    options: GenerationOptions | None = None,
) -> GenerationResult:
    """统一生成入口；config 为 EngineConfig 集中式配置对象。

    所有可选执行控制项（eos_token_id、stop_on_eos、profile 等）
    统一通过 options（GenerationOptions 结构体）传入。
    """
    opts = options or GenerationOptions()
    validate_engine_config(config)

    # 1. 规约输入 Prompt 与输出预算
    token_sequences, has_uniform_prompt_len = normalize_token_sequences(prompts, tokenizer=opts.tokenizer)
    budgets = normalize_budgets(max_new_tokens, len(token_sequences))

    # 2. 调度路径分发（Continuous Batching 或启用了 Paged KV Cache 的 Fixed Batching）
    if config.batching_method == BatchingMethod.CONTINUOUS or isinstance(config.kv_cache_config, PagedCacheConfig):
        return _dispatch_continuous(
            config,
            model,
            token_sequences,
            budgets,
            opts,
        )

    # 3. 串行 baseline 分发
    if config.batching_method == BatchingMethod.NONE and len(token_sequences) > 1:
        return _dispatch_serial_baseline(
            config,
            model,
            token_sequences,
            budgets,
            opts,
        )

    # 4. 齐步（Lockstep）执行（所有请求同时开始，无队列调度）
    return _dispatch_lockstep(
        config,
        model,
        token_sequences,
        has_uniform_prompt_len,
        budgets,
        opts,
    )


def _dispatch_lockstep(
    config: EngineConfig,
    model: torch.nn.Module,
    token_sequences: Sequence[Sequence[int]],
    has_uniform_prompt_len: bool,
    budgets: Sequence[int],
    opts: GenerationOptions,
) -> GenerationResult:
    """以齐步（Lockstep）循环执行（支持有/无 KV Cache、Profile、Logits 捕获等）。"""
    return run_lockstep(
        config=config,
        model=model,
        token_sequences=token_sequences,
        has_uniform_prompt_len=has_uniform_prompt_len,
        budgets=budgets,
        pad_token_id=opts.pad_token_id,
        eos_token_id=opts.eos_token_id,
        stop_on_eos=opts.stop_on_eos,
        capture_logits=opts.capture_logits,
        profile=opts.profile,
        top_kernels=opts.top_kernels,
        tensorboard_options=opts.tensorboard_options,
    )


def _dispatch_serial_baseline(
    config: EngineConfig,
    model: torch.nn.Module,
    token_sequences: Sequence[Sequence[int]],
    budgets: Sequence[int],
    opts: GenerationOptions,
) -> GenerationResult:
    """按顺序逐个执行请求（串行 baseline 模式）。"""
    runs = [
        generate(
            config,
            model,
            seq,
            budget,
            options=opts,
        )
        for seq, budget in zip(token_sequences, budgets)
    ]
    return aggregate_generation_results(runs)


def _dispatch_continuous(
    config: EngineConfig,
    model: torch.nn.Module,
    token_sequences: Sequence[Sequence[int]],
    budgets: Sequence[int],
    opts: GenerationOptions,
) -> ScheduleResult:
    """以 Continuous Batching 执行（支持自定义并发上限），直接返回 ScheduleResult。"""
    if opts.capture_logits or opts.profile:
        raise ValueError("continuous-batching 暂不支持 capture_logits/profile")
    requests = make_requests(token_sequences, budgets)
    return run_scheduler(
        config,
        model,
        requests,
        opts.eos_token_id,
        pad_token_id=opts.pad_token_id,
        stop_on_eos=opts.stop_on_eos,
    )


__all__ = [
    "GenerationOptions",
    "GenerationResult",
    "PromptInput",
    "generate",
]
