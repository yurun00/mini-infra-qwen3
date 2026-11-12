"""文本生成用户级入口（Generation Entrypoint）。

对标 vLLM entrypoints/llm.py，仅负责：
- 解析与校验用户级配置（GenerationOptions）与多样化 Prompt 输入；
- 按 Mode 分发到底层引擎执行（run_lockstep / run_scheduler）；
- 包装 Profiler 上下文并组装返回 GenerationResult。
"""

from collections.abc import Sequence

import torch

from mini_infra.core import KVCacheConfig, PagedCacheConfig, make_requests
from mini_infra.engine.lockstep_loop import run_lockstep
from mini_infra.engine.mode import BatchingMethod, Mode, validate_mode
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
    mode: Mode,
    model: torch.nn.Module,
    prompts: PromptInput | Sequence[PromptInput],
    max_new_tokens: int | Sequence[int],
    options: GenerationOptions | None = None,
) -> GenerationResult:
    """统一生成入口；mode 为 Mode 策略标签对象。

    所有可选控制项（device、eos_token_id、stop_on_eos、profile、max_running_requests、kv_cache_config 等）
    统一通过 options（GenerationOptions 结构体）传入。
    不同 mode 对应不同的 feature 开关，但生成的 output token 完全一致。
    """
    opts = options or GenerationOptions()
    validate_mode(mode)
    if opts.kv_cache_config is not None and opts.kv_cache_config.method is not mode.caching_method:
        raise ValueError(
            f"options.kv_cache_config.method ({opts.kv_cache_config.method}) 与 "
            f"mode.caching_method ({mode.caching_method}) 不一致"
        )
    kv_cache_config = opts.kv_cache_config if opts.kv_cache_config is not None else mode.caching_method.default_config

    # 1. 规约设备
    device = torch.device(opts.device) if opts.device is not None else next(model.parameters()).device

    # 2. 规约输入 Prompt 与输出预算
    token_sequences, has_uniform_prompt_len = normalize_token_sequences(prompts, tokenizer=opts.tokenizer)
    budgets = normalize_budgets(max_new_tokens, len(token_sequences))

    # 3. 调度路径分发（Continuous Batching 或启用了 Paged KV Cache 的 Fixed Batching）
    if mode.batching_method == BatchingMethod.CONTINUOUS or isinstance(kv_cache_config, PagedCacheConfig):
        return _dispatch_continuous(
            mode,
            model,
            token_sequences,
            budgets,
            device,
            opts,
            kv_cache_config,
        )

    # 4. 串行 baseline 分发
    if mode.batching_method == BatchingMethod.NONE and len(token_sequences) > 1:
        return _dispatch_serial_baseline(
            mode,
            model,
            token_sequences,
            budgets,
            device,
            opts,
        )

    # 5. 齐步（Lockstep）执行（所有请求同时开始，无队列调度）
    return _dispatch_lockstep(
        model,
        token_sequences,
        has_uniform_prompt_len,
        budgets,
        device,
        opts,
        kv_cache_config,
    )


def _dispatch_lockstep(
    model: torch.nn.Module,
    token_sequences: Sequence[Sequence[int]],
    has_uniform_prompt_len: bool,
    budgets: Sequence[int],
    device: torch.device,
    opts: GenerationOptions,
    kv_cache_config: KVCacheConfig,
) -> GenerationResult:
    """以齐步（Lockstep）循环执行（支持有/无 KV Cache、Profile、Logits 捕获等）。"""
    return run_lockstep(
        model=model,
        token_sequences=token_sequences,
        has_uniform_prompt_len=has_uniform_prompt_len,
        budgets=budgets,
        kv_cache_config=kv_cache_config,
        device=device,
        pad_token_id=opts.pad_token_id,
        eos_token_id=opts.eos_token_id,
        stop_on_eos=opts.stop_on_eos,
        capture_logits=opts.capture_logits,
        profile=opts.profile,
        top_kernels=opts.top_kernels,
        tensorboard_options=opts.tensorboard_options,
    )


def _dispatch_serial_baseline(
    mode: Mode,
    model: torch.nn.Module,
    token_sequences: Sequence[Sequence[int]],
    budgets: Sequence[int],
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
        kv_cache_config=opts.kv_cache_config,
    )
    runs = [
        generate(
            mode,
            model,
            seq,
            budget,
            options=single_req_options,
        )
        for seq, budget in zip(token_sequences, budgets)
    ]
    return aggregate_generation_results(runs)


def _dispatch_continuous(
    mode: Mode,
    model: torch.nn.Module,
    token_sequences: Sequence[Sequence[int]],
    budgets: Sequence[int],
    device: torch.device,
    opts: GenerationOptions,
    kv_cache_config: KVCacheConfig,
) -> ScheduleResult:
    """以 Continuous Batching 执行（支持自定义并发上限），直接返回 ScheduleResult。"""
    if opts.capture_logits or opts.profile:
        raise ValueError("continuous-batching 暂不支持 capture_logits/profile")
    if opts.max_running_requests is not None and opts.max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    requests = make_requests(token_sequences, budgets)
    max_running_requests = opts.max_running_requests if opts.max_running_requests is not None else len(requests)
    return run_scheduler(
        mode.batching_method,
        model,
        requests,
        max_running_requests,
        opts.eos_token_id,
        device,
        kv_cache_config=kv_cache_config,
        pad_token_id=opts.pad_token_id,
        stop_on_eos=opts.stop_on_eos,
    )


__all__ = [
    "GenerationOptions",
    "GenerationResult",
    "PromptInput",
    "generate",
]
