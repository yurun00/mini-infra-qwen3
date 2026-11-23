"""GPU 测试：验证真实 Qwen3 模型在 Continuous Batching 与 Serial Generate 下的逐 Token 完全对齐。

需要 GPU 和已下载的权重；不满足时自动跳过。
运行：pytest tests/gpu/parity/test_continuous_parity.py
"""

from __future__ import annotations

import pytest
import torch

from mini_infra.core import (
    ChunkedPrefillConfig,
    DenseCacheConfig,
    PagedCacheConfig,
    SchedulerConfig,
    TokenBudgetConfig,
    make_requests,
)
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
        EngineConfig(
            DenseCacheConfig(),
            batching_method=BatchingMethod.CONTINUOUS,
            scheduler_config=SchedulerConfig(max_running_requests=2),
        ),
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
    )

    # 2. 运行固定批处理作为对比
    fixed_result = run_scheduler(
        EngineConfig(
            DenseCacheConfig(),
            batching_method=BatchingMethod.FIXED,
            scheduler_config=SchedulerConfig(max_running_requests=2),
        ),
        model,
        requests,
        eos_token_id=eos_token_id,
        stop_on_eos=False,
    )

    paged_result = run_scheduler(
        EngineConfig(
            PagedCacheConfig(block_size=16),
            batching_method=BatchingMethod.CONTINUOUS,
            scheduler_config=SchedulerConfig(max_running_requests=2),
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
                scheduler_config=SchedulerConfig(
                    max_running_requests=2,
                    batching_policy=TokenBudgetConfig(token_budget),
                ),
            ),
            model,
            requests,
            eos_token_id=eos_token_id,
            stop_on_eos=False,
        )
        for kv_cache_config in (DenseCacheConfig(), PagedCacheConfig(block_size=16))
        for token_budget in (2, 32)
    }

    # chunked-prefill 调度（仅 Paged）：预算 2（= 并发上限，每请求每步 1 个 Token）、8（Prompt 被切成多个 chunk）、
    # 256（整段 Prompt 一步写满，退化为 prefill-first 的 Prefill 形态）
    chunked_results = {
        token_budget: run_scheduler(
            EngineConfig(
                PagedCacheConfig(block_size=16),
                batching_method=BatchingMethod.CONTINUOUS,
                scheduler_config=SchedulerConfig(
                    max_running_requests=2,
                    batching_policy=ChunkedPrefillConfig(token_budget),
                ),
            ),
            model,
            requests,
            eos_token_id=eos_token_id,
            stop_on_eos=False,
        )
        for token_budget in (2, 8, 256)
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

        for token_budget, chunked in chunked_results.items():
            assert chunked.sequences[request_idx].token_ids == expected, (
                f"chunked-prefill token_budget={token_budget} output divergence for req {request_idx}"
            )
    for token_budget, chunked in chunked_results.items():
        assert chunked.schedule_metrics.max_prefill_step_tokens <= token_budget
        assert chunked.schedule_metrics.num_oversize_prefill_steps == 0
    # 预算 2 把每个 Prompt 逐 Token 切分：chunk 数远多于请求数
    assert chunked_results[2].schedule_metrics.num_prefill_chunks > len(requests)
    assert chunked_results[256].schedule_metrics.num_prefill_chunks == len(requests)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_chunked_prefill_cli():
    from mini_infra.cli import main

    assert (
        main(
            [
                "verify",
                "chunked-prefill",
                "--max-new-tokens",
                "4",
                "--token-budget",
                "32",
                "--block-size",
                "16",
            ]
        )
        == 0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_prefix_caching_cli():
    from mini_infra.cli import main

    assert main(["verify", "prefix-caching", "--max-new-tokens", "4", "--block-size", "16"]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_prefix_caching_gpu_parity_across_policies() -> None:
    """真实权重 float32：共享前缀负载在三种 batching policy、并发 1/2 下，prefix caching 与单请求串行逐 Token 一致。"""
    model_dir = resolve_model_directory(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device="cuda", dtype=torch.float32)
    model.eval()

    shared = "以下是共同的系统背景：你是一名推理引擎工程师，需要用简洁准确的语言回答关于 KV Cache 的问题。" * 3
    questions = ["什么是 Prefill？", "什么是 Decode？", "Paged KV Cache 解决了什么问题？", "前缀缓存何时失效？"]
    prompt_tokens = [tokenizer.encode_chat_prompt(shared + question) for question in questions]
    budgets = [8, 6, 10, 4]
    requests = make_requests(prompt_tokens, budgets, arrival_times_ms=[0.0, 0.0, 2.0, 4.0])
    eos_token_id = tokenizer.eos_token_id or -1

    single_opts = GenerationOptions(eos_token_id=eos_token_id, stop_on_eos=False)
    golden_tokens = [
        generate(EngineConfig(DenseCacheConfig()), model, [p_tokens], budget, options=single_opts)
        .sequences[0]
        .token_ids
        for p_tokens, budget in zip(prompt_tokens, budgets)
    ]

    for policy in (None, TokenBudgetConfig(128), ChunkedPrefillConfig(32)):
        for max_running_requests in (1, 2):
            scheduler_config = SchedulerConfig(max_running_requests=max_running_requests)
            if policy is not None:
                scheduler_config = scheduler_config.with_batching_policy(policy)
            result = run_scheduler(
                EngineConfig(
                    PagedCacheConfig(block_size=16, is_prefix_caching_enabled=True),
                    batching_method=BatchingMethod.CONTINUOUS,
                    scheduler_config=scheduler_config,
                ),
                model,
                requests,
                eos_token_id=eos_token_id,
                stop_on_eos=False,
            )
            label = f"policy={scheduler_config.batching_policy}, mrr={max_running_requests}"
            assert [s.token_ids for s in result.sequences] == golden_tokens, f"prefix caching divergence: {label}"
            assert sum(s.prefix_cache_hit_tokens for s in result.sequences) > 0, f"no prefix cache hit: {label}"
