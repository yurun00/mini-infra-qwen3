"""eager 与 flash attention backend 的 Kernel 级与真实模型级数值对齐验证（对应 chapter_09 compare_attention / compare_model）。

入口：mini-infra verify flash-attention。

1. Kernel 级：以真实模型的 head 配置构造随机 Q/K/V（K/V 取 transpose 视图以覆盖 Paged gather 的非连续布局），
   覆盖 Full Prefill、非整 Tile 长度、Q < K 的 Chunked Prefill、左 Padding Batch 与 Decode，
   Reference 为 float32 升维后的 eager backend（显式物化 Score，无 SDPA）；只比较有效 Query 行。
2. 模型级：同一模型、同一负载分别以 eager / flash backend 运行 Lockstep（dense-kv-cache + fixed-batching，
   逐步 Logits）与 Paged chunked-prefill（首 Token Logits），比较 Greedy Token 与 Logits 误差。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mini_infra.core import (
    ChunkedPrefillConfig,
    DenseCacheConfig,
    PagedCacheConfig,
    SchedulerConfig,
    make_requests,
)
from mini_infra.engine import (
    BatchingMethod,
    EngineConfig,
    GenerationResult,
    LockstepResult,
    ScheduleResult,
    run_scheduler,
)
from mini_infra.entrypoints import GenerationOptions, generate
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    CausalPaddingMask,
    Qwen3Config,
    get_attn_backend,
)
from mini_infra.verification.parity._common import (
    build_environment,
    dump_json,
    first_token_logit_errors,
    prepare_cuda_model,
)

DEFAULT_PROMPTS = [
    "用一句话解释 FlashAttention。",
    "请简要说明为什么 Online Softmax 不需要保存完整的 Attention 矩阵。" * 6,
    "Explain prefill and decode in one short sentence.",
]

# Kernel 级输出最大绝对误差阈值（相对 float32 eager Reference；16 位阈值覆盖输出本身的舍入误差）
_KERNEL_ATOL_BY_DTYPE = {
    torch.float32: 1e-5,
    torch.bfloat16: 3e-2,
}


@dataclass(frozen=True)
class _KernelCase:
    """单个 Kernel 对齐用例：left_pad_lens[b] 为第 b 行 Key 序列左侧 padding 长度。"""

    name: str
    query_len: int
    kv_len: int
    left_pad_lens: tuple[int, ...]


_KERNEL_CASES: tuple[_KernelCase, ...] = (
    _KernelCase("full-prefill", 128, 128, (0,)),
    _KernelCase("non-tile-full-prefill", 257, 257, (0, 0)),
    _KernelCase("chunked-prefill", 64, 320, (0, 0)),
    _KernelCase("left-padded-chunk-batch", 64, 320, (0, 100, 290)),
    _KernelCase("left-padded-prefill-batch", 200, 200, (0, 70, 150)),
    _KernelCase("decode", 1, 1025, (0, 3, 500, 1000)),
)


@torch.inference_mode()
def run_kernel_cases(
    config: Qwen3Config, device: torch.device, dtype: torch.dtype
) -> list[dict[str, Any]]:
    """逐用例比较 flash backend 与 float32 eager Reference 的 Attention 输出误差。"""
    eager = get_attn_backend("eager")
    flash = get_attn_backend("flash")
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    scaling = head_dim**-0.5
    generator = torch.Generator(device=device).manual_seed(0)
    rows = []
    for case in _KERNEL_CASES:
        batch_size = len(case.left_pad_lens)
        # query: [batch_size, num_heads, query_len, head_dim], contiguous
        query = torch.randn(
            (batch_size, num_heads, case.query_len, head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        # key / value: [batch_size, kv_len, num_kv_heads, head_dim] -> transpose(1, 2) 非连续视图（同 Paged gather 布局）
        key, value = (
            torch.randn(
                (batch_size, case.kv_len, num_kv_heads, head_dim),
                dtype=dtype,
                device=device,
                generator=generator,
            ).transpose(1, 2)
            for _ in range(2)
        )
        key_positions = torch.arange(case.kv_len, device=device)
        pad_lens = torch.tensor(case.left_pad_lens, device=device)
        # key_valid: [batch_size, kv_len], bool, contiguous
        key_valid = key_positions.unsqueeze(0) >= pad_lens.unsqueeze(1)
        causal_offset = case.kv_len - case.query_len
        mask = CausalPaddingMask(key_valid, key_valid[:, causal_offset:], causal_offset)

        reference = eager.forward(
            query.float(),
            key.float(),
            value.float(),
            eager.prepare_inputs(mask, torch.float32),
            scaling,
        )
        candidate = flash.forward(
            query, key, value, flash.prepare_inputs(mask, dtype), scaling
        )
        # 只比较有效 Query 行：[batch_size, query_len, num_heads * head_dim] 中 query_valid 为 True 的行
        is_query_valid = mask.query_valid
        difference = (candidate.float() - reference).abs()[is_query_valid]
        row = {
            "name": case.name,
            "batch_size": batch_size,
            "query_len": case.query_len,
            "kv_len": case.kv_len,
            "left_pad_lens": list(case.left_pad_lens),
            "is_finite": bool(torch.isfinite(candidate).all().item()),
            "max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item(),
        }
        rows.append(row)
        print(
            "Kernel %-26s | Q=%4d K=%5d B=%d | 最大误差=%.3e | 平均误差=%.3e"
            % (
                case.name,
                case.query_len,
                case.kv_len,
                batch_size,
                row["max_abs_error"],
                row["mean_abs_error"],
            )
        )
    return rows


def _lockstep_logits_error(reference: LockstepResult, candidate: LockstepResult) -> float:
    """逐步 Logits（[batch_size, vocab_size] / 步）的最大绝对误差（仅比较两侧共同步数）。"""
    assert reference.trace.logits_trace is not None
    assert candidate.trace.logits_trace is not None
    return max(
        (ref - cand).abs().max().item()
        for ref, cand in zip(reference.trace.logits_trace, candidate.trace.logits_trace)
    )


def _schedule_logits_error(reference: ScheduleResult, candidate: ScheduleResult) -> float:
    return max(first_token_logit_errors(reference, candidate).values())


@torch.inference_mode()
def run_compare_flash_attention_model(
    model: torch.nn.Module,
    sequences: Sequence[Sequence[int]],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    token_budget: int = 32,
    block_size: int = 16,
    max_running_requests: int = 4,
) -> list[dict[str, Any]]:
    """同一负载分别以 eager / flash backend 运行两种引擎配置，返回逐配置的 Token 一致性与 Logits 误差。"""
    requests = make_requests(sequences, max_new_tokens)
    scheduler_config = SchedulerConfig(max_running_requests=max_running_requests)

    def _run_lockstep(config: EngineConfig) -> GenerationResult:
        return generate(
            config,
            model,
            list(sequences),
            max_new_tokens,
            options=GenerationOptions(
                eos_token_id=eos_token_id, stop_on_eos=False, capture_logits=True
            ),
        )

    def _run_schedule(config: EngineConfig) -> GenerationResult:
        return run_scheduler(
            config,
            model,
            requests,
            eos_token_id=eos_token_id,
            stop_on_eos=False,
            should_capture_first_token_logits=True,
        )

    targets: tuple[
        tuple[EngineConfig, Callable[[EngineConfig], GenerationResult]], ...
    ] = (
        (
            EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED),
            _run_lockstep,
        ),
        (
            EngineConfig(
                PagedCacheConfig(block_size=block_size),
                batching_method=BatchingMethod.CONTINUOUS,
                scheduler_config=scheduler_config.with_batching_policy(
                    ChunkedPrefillConfig(token_budget)
                ),
            ),
            _run_schedule,
        ),
    )

    rows = []
    for eager_config, run_fn in targets:
        eager_result = run_fn(eager_config)
        flash_result = run_fn(eager_config.with_attention_backend("flash"))
        if isinstance(eager_result, LockstepResult):
            assert isinstance(flash_result, LockstepResult)
            logits_scope = "all-steps"
            max_logit_error = _lockstep_logits_error(eager_result, flash_result)
        else:
            assert isinstance(eager_result, ScheduleResult)
            assert isinstance(flash_result, ScheduleResult)
            logits_scope = "first-token"
            max_logit_error = _schedule_logits_error(eager_result, flash_result)
        eager_token_ids = [s.token_ids for s in eager_result.sequences]
        flash_token_ids = [s.token_ids for s in flash_result.sequences]
        row = {
            "engine_config": str(eager_config),
            "tokens_match": eager_token_ids == flash_token_ids,
            "logits_scope": logits_scope,
            "max_logit_error": max_logit_error,
            "eager_token_ids": eager_token_ids,
            "flash_token_ids": flash_token_ids,
            "eager_e2e_model_ms": eager_result.metrics.e2e_model_ms,
            "flash_e2e_model_ms": flash_result.metrics.e2e_model_ms,
        }
        rows.append(row)
        print(
            "模型 [%s] | Token 一致=%s | %s Logits 最大误差=%.3e"
            % (row["engine_config"], row["tokens_match"], logits_scope, max_logit_error)
        )
    return rows


def execute_compare_flash_attention(
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
    """运行 eager 与 flash attention backend 对齐验证，返回退出码（0 成功，1 失败）。

    Kernel 级误差在任意 dtype 下均按 dtype 阈值判定；模型级仅 float32 按 Token 一致与 logits_atol 判定，
    bfloat16 只报告误差（28 层归约顺序差异会累积，Greedy 临界点可能翻转）。
    """
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
    model_dtype = next(model_instance.parameters()).dtype
    prompts_to_use = list(prompts) if prompts is not None else DEFAULT_PROMPTS
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts_to_use]
    environment = build_environment(
        device,
        model=model,
        revision=revision,
        dtype=dtype,
        prompt_lens=[len(s) for s in sequences],
        max_new_tokens=max_new_tokens,
        token_budget=token_budget,
        block_size=block_size,
        max_running_requests=max_running_requests,
        logits_atol=logits_atol,
        decoding="greedy, EOS disabled",
        thinking=False,
    )

    kernel_rows = run_kernel_cases(model_instance.config, device, model_dtype)
    kernel_atol = _KERNEL_ATOL_BY_DTYPE[model_dtype]
    is_kernel_ok = all(
        row["is_finite"] and row["max_abs_error"] <= kernel_atol for row in kernel_rows
    )
    model_rows = run_compare_flash_attention_model(
        model_instance,
        sequences,
        max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        token_budget=token_budget,
        block_size=block_size,
        max_running_requests=max_running_requests,
    )
    is_model_ok = model_dtype != torch.float32 or all(
        row["tokens_match"] and row["max_logit_error"] <= logits_atol
        for row in model_rows
    )

    print("Kernel 级对齐（阈值 %.0e）: %s" % (kernel_atol, is_kernel_ok))
    print("模型级对齐: %s" % is_model_ok)
    if output:
        dump_json(
            output,
            {
                "environment": environment,
                "result": {"kernel": kernel_rows, "model": model_rows},
            },
        )
    if not (is_kernel_ok and is_model_ok):
        print("FlashAttention 正确性检查失败")
        return 1
    return 0


__all__ = [
    "DEFAULT_PROMPTS",
    "execute_compare_flash_attention",
    "run_compare_flash_attention_model",
    "run_kernel_cases",
]
