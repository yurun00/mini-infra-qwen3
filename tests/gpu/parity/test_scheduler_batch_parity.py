"""GPU 测试：验证真实 Qwen3 模型在 Continuous Batching 与 Serial Generate 下的逐 Token 完全对齐。

需要 GPU 和已下载的权重；不满足时自动跳过。
运行：pytest tests/gpu/parity/test_continuous_parity.py
"""

from __future__ import annotations

import pytest
import torch

from mini_infra.core import DenseCacheConfig, PagedCacheConfig, SchedulerConfig, TokenBudgetConfig, make_requests
from mini_infra.engine import (
    BatchingMethod,
    EngineConfig,
    Qwen3Tokenizer,
    run_scheduler,
)
from mini_infra.entrypoints import GenerationOptions, generate
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_model,
    resolve_model_directory,
)

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_continuous_batching_gpu_parity() -> None:
    model_dir = resolve_model_directory(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device="cuda", dtype=torch.float32)
    model.eval()

    prompts = [
        "你好，请介绍一下 Python 语言的优点。",
        "什么是连续批处理（Continuous Batching）？",
        "用简短的语言写一首关于夏天的五言绝句。",
    ]
    budgets = [12, 8, 16]
    prompt_tokens = [tokenizer.encode(p) for p in prompts]
    requests = make_requests(prompt_tokens, budgets, arrival_times_ms=[0.0, 2.0, 4.0])

    eos_token_id = tokenizer.eos_token_id or -1

    # 1. 运行连续批处理（容量 2，测试 batch slot 复用）
    cont_result = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2),
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
    )

    # 2. 运行固定批处理作为对比
    fixed_result = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED, max_running_requests=2),
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
    )

    paged_result = run_scheduler(
        EngineConfig(
            PagedCacheConfig(block_size=16),
            batching_method=BatchingMethod.CONTINUOUS,
            max_running_requests=2,
        ),
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
    )

    # token-budget 调度：预算 2（= 并发上限）使每次 Prefill 都是 oversize prefill；预算 32 可入场多请求
    token_budget_results = {
        (kv_cache_config.kind, token_budget): run_scheduler(
            EngineConfig(
                kv_cache_config,
                batching_method=BatchingMethod.CONTINUOUS,
                max_running_requests=2,
                scheduler_config=SchedulerConfig(TokenBudgetConfig(token_budget)),
            ),
            model,
            requests,
            eos_token_id=eos_token_id,
            stop_on_eos=False,
        )
        for kv_cache_config in (DenseCacheConfig(), PagedCacheConfig(block_size=16))
        for token_budget in (2, 32)
    }

    # 3. 逐个请求运行单请求 generate 作为 Golden Truth
    single_config = EngineConfig(DenseCacheConfig())
    single_opts = GenerationOptions(eos_token_id=eos_token_id, stop_on_eos=False)
    golden_tokens: list[list[int]] = []
    for p_tokens, budget in zip(prompt_tokens, budgets):
        gen_res = generate(
            single_config,
            model,
            [p_tokens],
            budget,
            options=single_opts,
        )
        golden_tokens.append(gen_res.sequences[0].token_ids)

    # 4. 严格逐 Token 比对
    for request_idx, expected in enumerate(golden_tokens):
        cont_tokens = cont_result.sequences[request_idx].token_ids
        fixed_tokens = fixed_result.sequences[request_idx].token_ids
        assert cont_tokens == expected, (
            f"Continuous batching output divergence for req {request_idx}: got {cont_tokens}, expected {expected}"
        )
        assert paged_result.sequences[request_idx].token_ids == expected, (
            f"Paged continuous batching output divergence for req {request_idx}"
        )
        assert fixed_tokens == expected, (
            f"Fixed batching output divergence for req {request_idx}: got {fixed_tokens}, expected {expected}"
        )
        for key, budgeted in token_budget_results.items():
            assert budgeted.sequences[request_idx].token_ids == expected, (
                f"token-budget {key} output divergence for req {request_idx}"
            )

    # 5. 指标健康检查
    assert cont_result.metrics.e2e_wall_ms > 0
    assert cont_result.memory.peak_allocated_mb > 0
    assert len(cont_result.request_metrics) == len(requests)
    for rm in cont_result.request_metrics:
        assert rm.num_output_tokens > 0
        assert rm.ttft_ms > 0
        assert rm.end_to_end_ms >= rm.ttft_ms


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_scheduler_batch_cli():
    from mini_infra.cli import main

    assert main(["verify", "batch-scheduler", "--max-new-tokens", "4"]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_schedulers_cli():
    from mini_infra.cli import main

    assert (
        main(
            [
                "verify",
                "schedulers-token-budget",
                "--max-new-tokens",
                "4",
                "--token-budget",
                "64",
                "--block-size",
                "16",
            ]
        )
        == 0
    )
