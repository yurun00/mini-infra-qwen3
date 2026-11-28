"""eager 与 flash attention backend 的 Kernel 级与真实模型级数值对齐验证（对应 chapter_09 compare_attention / compare_model）。

入口：mini-infra verify flash-attention。

1. Kernel 级：以真实模型的 head 配置构造随机 Q/K/V（K/V 取 transpose 视图以覆盖 Paged gather 的非连续布局），
   覆盖 Full Prefill、非整 Tile 长度、Q < K 的 Chunked Prefill、左 Padding Chunk Batch（含有效历史前缀 +
   中间 Padding 空洞）、左 Padding Prefill Batch、Dense Decode 与 Paged Decode（context_lens 右侧屏蔽），
   Reference 为 float32 升维后的 eager backend（显式物化 Score，无 SDPA）；只比较有效 Query 行。
2. 模型级：同一模型、同一负载分别以 eager / flash backend 运行 Lockstep（dense-kv-cache + fixed-batching，
   逐步 Logits）与 Paged chunked-prefill（首 Token Logits），校验输出 Logits 有限性、Greedy Token 与 Logits 误差。
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

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


class _Unpadded(NamedTuple):
    """无 Padding 掩码规格：key_valid / query_valid 全 True。"""

    batch_size: int

    def build_mask(
        self, query_len: int, kv_len: int, device: torch.device
    ) -> CausalPaddingMask:
        causal_offset = kv_len - query_len
        key_valid = torch.ones((self.batch_size, kv_len), dtype=torch.bool, device=device)
        return CausalPaddingMask(key_valid, key_valid[:, causal_offset:], causal_offset)


class _LeftPad(NamedTuple):
    """左侧连续 Padding 规格（Full Prefill / Dense Decode）：key_valid = positions >= pad_lens。"""

    pad_lens: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.pad_lens)

    def build_mask(
        self, query_len: int, kv_len: int, device: torch.device
    ) -> CausalPaddingMask:
        causal_offset = kv_len - query_len
        key_positions = torch.arange(kv_len, device=device).unsqueeze(0)
        pad_lens = torch.tensor(self.pad_lens, dtype=torch.long, device=device)
        key_valid = key_positions >= pad_lens.unsqueeze(1)
        return CausalPaddingMask(key_valid, key_valid[:, causal_offset:], causal_offset)


class _ChunkSpans(NamedTuple):
    """Chunked Prefill 两段式掩码规格：每行 (history_valid_len, query_left_pad)。

    历史区 [0, causal_offset) 前 history_valid_len 位有效、尾位（未写满块 / null block）屏蔽；
    当前区 [causal_offset, kv_len) 左侧 query_left_pad 位屏蔽（构造中间 Padding 空洞）。
    """

    spans: tuple[tuple[int, int], ...]

    @property
    def batch_size(self) -> int:
        return len(self.spans)

    def build_mask(
        self, query_len: int, kv_len: int, device: torch.device
    ) -> CausalPaddingMask:
        causal_offset = kv_len - query_len
        history_positions = torch.arange(causal_offset, device=device).unsqueeze(0)
        query_positions = torch.arange(query_len, device=device).unsqueeze(0)
        spans = torch.tensor(self.spans, dtype=torch.long, device=device)
        history_valid = history_positions < spans[:, 0:1]
        query_valid = query_positions >= spans[:, 1:2]
        key_valid = torch.cat((history_valid, query_valid), dim=1)
        return CausalPaddingMask(key_valid, query_valid, causal_offset)


class _PagedContextLens(NamedTuple):
    """Paged Decode 右侧屏蔽规格：key_valid = positions < context_lens，query_valid 全 True。"""

    context_lens: tuple[int, ...]

    @property
    def batch_size(self) -> int:
        return len(self.context_lens)

    def build_mask(
        self, query_len: int, kv_len: int, device: torch.device
    ) -> CausalPaddingMask:
        causal_offset = kv_len - query_len
        key_positions = torch.arange(kv_len, device=device).unsqueeze(0)
        lengths = torch.tensor(self.context_lens, dtype=torch.long, device=device)
        key_valid = key_positions < lengths.unsqueeze(1)
        query_valid = torch.ones(
            (self.batch_size, query_len), dtype=torch.bool, device=device
        )
        return CausalPaddingMask(key_valid, query_valid, causal_offset)


_MaskSpec = _Unpadded | _LeftPad | _ChunkSpans | _PagedContextLens


@dataclass(frozen=True)
class _KernelCase:
    """单个 Kernel 对齐用例：由封闭联合类型 mask_spec 唯一指定掩码几何。"""

    name: str
    query_len: int
    kv_len: int
    mask_spec: _MaskSpec

    @property
    def batch_size(self) -> int:
        return self.mask_spec.batch_size

    def build_mask(self, device: torch.device) -> CausalPaddingMask:
        """在指定设备上构造当前用例的 CausalPaddingMask。"""
        return self.mask_spec.build_mask(self.query_len, self.kv_len, device)


_KERNEL_CASES: tuple[_KernelCase, ...] = (
    _KernelCase("full-prefill", 128, 128, _Unpadded(1)),
    _KernelCase("non-tile-full-prefill", 257, 257, _Unpadded(2)),
    _KernelCase("chunked-prefill", 64, 320, _Unpadded(2)),
    # 同批覆盖：完整历史+完整 chunk、完整历史+当前 chunk 左 padding（对齐 compare_attention.py）、
    # 部分历史尾块空洞+chunk 左 padding（PagedRunner 多请求混批）、零历史新入场请求+chunk 左 padding。
    _KernelCase(
        "left-padded-chunk-batch",
        64,
        320,
        _ChunkSpans(((256, 0), (256, 11), (180, 20), (0, 34))),
    ),
    _KernelCase("left-padded-prefill-batch", 200, 200, _LeftPad((0, 70, 150))),
    _KernelCase("decode", 1, 1025, _LeftPad((0, 3, 500, 1000))),
    _KernelCase("paged-decode", 1, 1024, _PagedContextLens((1024, 1021, 525, 25))),
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
        batch_size = case.batch_size
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
        mask = case.build_mask(device)

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
            "causal_offset": mask.causal_offset,
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


def _are_tensors_finite(tensors: Sequence[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _lockstep_logits_errors(
    reference: LockstepResult, candidate: LockstepResult
) -> tuple[bool, float, float]:
    """返回 Lockstep 逐步 Logits 的 (全部有限, 首 Token 最大误差, 全部共同步最大误差)。"""
    assert reference.trace.logits_trace is not None
    assert candidate.trace.logits_trace is not None
    is_finite = _are_tensors_finite(
        (*reference.trace.logits_trace, *candidate.trace.logits_trace)
    )
    first_token_error = (
        (reference.trace.logits_trace[0] - candidate.trace.logits_trace[0])
        .abs()
        .max()
        .item()
    )
    all_steps_error = max(
        (ref - cand).abs().max().item()
        for ref, cand in zip(reference.trace.logits_trace, candidate.trace.logits_trace)
    )
    return is_finite, first_token_error, all_steps_error


def _schedule_logits_errors(
    reference: ScheduleResult, candidate: ScheduleResult
) -> tuple[bool, float]:
    """返回 ScheduleResult 首 Token Logits 的 (全部有限, 最大误差)。"""
    assert reference.first_token_logits is not None
    assert candidate.first_token_logits is not None
    is_finite = _are_tensors_finite(
        (*reference.first_token_logits, *candidate.first_token_logits)
    )
    first_token_error = max(first_token_logit_errors(reference, candidate).values())
    return is_finite, first_token_error


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
            is_logits_finite, max_first_token_logit_error, max_logit_error = (
                _lockstep_logits_errors(eager_result, flash_result)
            )
        else:
            assert isinstance(eager_result, ScheduleResult)
            assert isinstance(flash_result, ScheduleResult)
            logits_scope = "first-token"
            is_logits_finite, max_first_token_logit_error = _schedule_logits_errors(
                eager_result, flash_result
            )
            max_logit_error = max_first_token_logit_error
        eager_token_ids = [s.token_ids for s in eager_result.sequences]
        flash_token_ids = [s.token_ids for s in flash_result.sequences]
        row = {
            "engine_config": str(eager_config),
            "tokens_match": eager_token_ids == flash_token_ids,
            "is_logits_finite": is_logits_finite,
            "logits_scope": logits_scope,
            "max_first_token_logit_error": max_first_token_logit_error,
            "max_logit_error": max_logit_error,
            "eager_token_ids": eager_token_ids,
            "flash_token_ids": flash_token_ids,
            "eager_e2e_model_ms": eager_result.metrics.e2e_model_ms,
            "flash_e2e_model_ms": flash_result.metrics.e2e_model_ms,
        }
        rows.append(row)
        print(
            "模型 [%s] | 有限=%s | Token 一致=%s | 首 Token 误差=%.3e | %s 最大误差=%.3e"
            % (
                row["engine_config"],
                is_logits_finite,
                row["tokens_match"],
                max_first_token_logit_error,
                logits_scope,
                max_logit_error,
            )
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
    logits_atol: float | None = None,
    output: str | None = None,
) -> int:
    """运行 eager 与 flash attention backend 对齐验证，返回退出码（0 成功，1 失败）。

    Kernel 级在任意 dtype 下均校验输出有限且最大误差 <= dtype 阈值；
    模型级在 float32 下校验 Logits 有限、Greedy Token 100% 一致且逐步最大误差 <= logits_atol（默认 1e-4），
    在 bfloat16 下校验捕获 Logits 全部有限且同输入的首 Token Logits 最大误差 <= logits_atol（默认 1.0）。
    """
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
        logits_atol=resolved_atol,
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
    if model_dtype == torch.float32:
        is_model_ok = all(
            row["is_logits_finite"]
            and row["tokens_match"]
            and row["max_logit_error"] <= resolved_atol
            for row in model_rows
        )
    else:
        is_model_ok = all(
            row["is_logits_finite"]
            and row["max_first_token_logit_error"] <= resolved_atol
            for row in model_rows
        )

    print("Kernel 级对齐（阈值 %.0e）: %s" % (kernel_atol, is_kernel_ok))
    print("模型级对齐（阈值 %.6g）: %s" % (resolved_atol, is_model_ok))
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
