"""GPU 测试：验证真实 Qwen3 模型在 Continuous Batching 与 Serial Generate 下的逐 Token 完全对齐。

需要 GPU 和已下载的权重；不满足时自动跳过。
运行：pytest tests/gpu/parity/test_continuous_parity.py
"""

from __future__ import annotations

import pytest
import torch

from mini_infra.core.request import make_request_specs
from mini_infra.engine import BatchingMethod, FeatureFlag, Mode, generate, run_scheduler
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Tokenizer,
    load_model,
    resolve_model_directory,
)

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_continuous_batching_gpu_parity() -> None:
    cuda = torch.device("cuda")
    model_dir = resolve_model_directory(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device="cuda", dtype=torch.bfloat16)
    model.eval()

    prompts = [
        "你好，请介绍一下 Python 语言的优点。",
        "什么是连续批处理（Continuous Batching）？",
        "用简短的语言写一首关于夏天的五言绝句。",
    ]
    budgets = [12, 8, 16]
    prompt_tokens = [tokenizer.encode(p) for p in prompts]
    specs = make_request_specs(prompt_tokens, budgets, arrival_times_ms=[0.0, 2.0, 4.0])

    eos_token_id = tokenizer.eos_token_id or -1

    # 1. 运行连续批处理（容量 2，测试槽位复用）
    cont_result = run_scheduler(
        BatchingMethod.CONTINUOUS,
        model,
        specs,
        max_running_requests=2,
        eos_token_id=eos_token_id,
        device=cuda,
        stop_on_eos=False,
    )

    # 2. 运行固定批处理作为对比
    fixed_result = run_scheduler(
        BatchingMethod.FIXED,
        model,
        specs,
        max_running_requests=2,
        eos_token_id=eos_token_id,
        device=cuda,
        stop_on_eos=False,
    )

    # 3. 逐个请求运行单请求 generate 作为 Golden Truth
    single_mode = Mode({FeatureFlag.KV_CACHE})
    golden_tokens: list[list[int]] = []
    for p_tokens, budget in zip(prompt_tokens, budgets):
        gen_res = generate(
            single_mode,
            model,
            [p_tokens],
            budget,
            eos_token_id=eos_token_id,
            device=cuda,
            stop_on_eos=False,
        )
        golden_tokens.append(gen_res.sequences[0].token_ids)

    # 4. 严格逐 Token 比对
    for idx, expected in enumerate(golden_tokens):
        cont_tokens = cont_result.sequences[idx].token_ids
        fixed_tokens = fixed_result.sequences[idx].token_ids
        assert cont_tokens == expected, (
            f"Continuous batching output divergence for req {idx}: got {cont_tokens}, expected {expected}"
        )
        assert fixed_tokens == expected, (
            f"Fixed batching output divergence for req {idx}: got {fixed_tokens}, expected {expected}"
        )

    # 5. 指标健康检查
    assert cont_result.metrics.makespan_ms > 0
    assert cont_result.metrics.peak_allocated_mb > 0
    assert len(cont_result.requests) == len(specs)
    for rm in cont_result.requests:
        assert rm.output_tokens > 0
        assert rm.ttft_ms > 0
        assert rm.end_to_end_ms >= rm.ttft_ms
