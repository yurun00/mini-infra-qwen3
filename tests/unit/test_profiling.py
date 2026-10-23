"""单元测试：算子性能剖析 (profiling) 与在 generate() 中可选启用剖析。"""

import pytest
import torch

from mini_infra.bench.run import ProfileMode
from mini_infra.generation import BatchingMethod, Mode, generate
from mini_infra.profiling import (
    KERNEL_CATEGORIES,
    MODULE_SCOPES,
    KernelInfo,
    OpLatencyBreakdown,
    TensorBoardOptions,
    install_scope_hooks,
    kernel_category,
    merge_op_latency_breakdowns,
)
from mini_infra.testing.smoke_test import build_tiny_model


def test_kernel_categories_and_scopes_definitions():
    assert len(KERNEL_CATEGORIES) > 0
    assert "Qwen3Attention" in MODULE_SCOPES
    assert MODULE_SCOPES["Qwen3Attention"] == "attention_block"
    assert issubclass(OpLatencyBreakdown, dict)
    assert issubclass(KernelInfo, dict)


def test_kernel_category():
    assert kernel_category("aten::mm") == "gemm"
    assert kernel_category("cublasLt_matmul") == "gemm"
    assert kernel_category("aten::scaled_dot_product_attention") == "attention"
    assert kernel_category("flash_attn_fwd") == "attention"
    assert kernel_category("aten::softmax") == "softmax"
    assert kernel_category("rms_norm_kernel") == "norm"
    assert kernel_category("reduce_kernel_sum") == "reduce"
    assert kernel_category("aten::embedding") == "memory"
    assert kernel_category("aten::copy_") == "memory"
    assert kernel_category("aten::silu") == "elementwise"
    assert kernel_category("unknown_custom_op") == "other"


def test_install_scope_hooks():
    model = build_tiny_model()
    handles = install_scope_hooks(model)
    try:
        assert len(handles) > 0
    finally:
        for h in handles:
            h.remove()


def test_generate_without_profile():
    model = build_tiny_model()
    mode = Mode()
    ids = torch.tensor([[1, 2, 3]])

    res = generate(mode, model, ids, max_new_tokens=2, profile=False)
    assert res.op_latency_breakdown is None
    assert "op_latency_breakdown" not in res.extra
    assert res.output_len == 2
    assert res.padded_output_tokens.shape == (1, 2)


def test_generate_with_profile():
    model = build_tiny_model()
    mode = Mode()
    ids = torch.tensor([[1, 2, 3]])

    res_no_prof = generate(mode, model, ids, max_new_tokens=3, profile=False)
    res_prof = generate(mode, model, ids, max_new_tokens=3, profile=True, top_kernels=5)

    # 1. 验证生成 Token 结果一致
    assert torch.equal(res_no_prof.padded_output_tokens, res_prof.padded_output_tokens)

    # 2. 验证 profiling 结构
    assert res_prof.op_latency_breakdown is not None
    assert isinstance(res_prof.op_latency_breakdown, dict)
    assert "total_ms" in res_prof.op_latency_breakdown
    assert isinstance(res_prof.op_latency_breakdown["total_ms"], int | float)
    assert res_prof.op_latency_breakdown["total_ms"] > 0
    assert "top_kernels" in res_prof.op_latency_breakdown
    assert "time_share_by_kernel_category" in res_prof.op_latency_breakdown
    assert "time_share_by_module" in res_prof.op_latency_breakdown

    assert len(res_prof.op_latency_breakdown["top_kernels"]) <= 5
    for item in res_prof.op_latency_breakdown["top_kernels"]:
        assert "name" in item
        assert "category" in item
        assert "cuda_ms" in item
        assert "share" in item
        assert "calls" in item

    # 3. 验证通过属性、字典项和 get() 均可访问
    assert res_prof["op_latency_breakdown"] == res_prof.op_latency_breakdown
    assert res_prof.get("op_latency_breakdown") == res_prof.op_latency_breakdown


def test_generate_profile_breakdown():
    model = build_tiny_model()
    mode = Mode()
    ids = torch.tensor([[1, 2, 3]])

    res = generate(mode, model, ids, max_new_tokens=2, profile=True, top_kernels=3)
    breakdown = res.op_latency_breakdown
    assert isinstance(breakdown, dict)
    assert "total_ms" in breakdown
    assert isinstance(breakdown["total_ms"], int | float)
    assert "top_kernels" in breakdown
    assert "time_share_by_kernel_category" in breakdown
    assert "time_share_by_module" in breakdown
    assert len(breakdown["top_kernels"]) <= 3


def test_generate_with_batching_and_profile():
    model = build_tiny_model()
    mode = Mode(batching_method=BatchingMethod.FIXED)
    prompts = torch.tensor([[1, 2, 3], [4, 5, 0]])

    res = generate(mode, model, prompts, max_new_tokens=2, profile=True)
    assert res.op_latency_breakdown is not None
    assert res.batch_size == 2
    assert res.padded_output_tokens.shape == (2, 2)


def test_tensorboard_options_structure():
    opt = TensorBoardOptions()
    assert str(opt.log_dir) == "./log"
    assert opt.record_shapes is True
    assert opt.profile_memory is True
    assert opt.with_stack is True

    custom_opt = TensorBoardOptions(log_dir="/tmp/test_tb", record_shapes=False, profile_memory=False)
    assert str(custom_opt.log_dir) == "/tmp/test_tb"
    assert custom_opt.record_shapes is False
    assert custom_opt.profile_memory is False
    assert custom_opt.with_stack is True


def test_profile_with_tensorboard_option(tmp_path):
    model = build_tiny_model()
    mode = Mode()
    ids = torch.tensor([[1, 2, 3]])

    # 1. 结构化传参：profile=True 且传入 TensorBoardOptions 对象到 generate
    tb_dir = tmp_path / "profiler_tb"
    tb_opt = TensorBoardOptions(log_dir=tb_dir)
    res = generate(mode, model, ids, max_new_tokens=2, profile=True, tensorboard_options=tb_opt)

    assert res.op_latency_breakdown is not None
    assert "total_ms" in res.op_latency_breakdown

    trace_files = list(tb_dir.glob("*.pt.trace.json"))
    assert len(trace_files) >= 1
    assert trace_files[0].stat().st_size > 0

    # 2. profile=False 时即使传入 tensorboard_options 也不进行剖析
    res_no_profile = generate(mode, model, ids, max_new_tokens=2, profile=False, tensorboard_options=tb_opt)
    assert res_no_profile.op_latency_breakdown is None

def test_profile_mode_enum():
    assert ProfileMode.EXTRA_PROFILE == "extra-profile"
    assert ProfileMode.FULL_PROFILE == "full-profile"
    assert str(ProfileMode.EXTRA_PROFILE) == "extra-profile"
    assert str(ProfileMode.FULL_PROFILE) == "full-profile"
    assert ProfileMode("extra-profile") == ProfileMode.EXTRA_PROFILE
    assert ProfileMode("full-profile") == ProfileMode.FULL_PROFILE
    with pytest.raises(ValueError):
        ProfileMode("invalid-mode")


def test_bench_run_profile_modes():
    from unittest.mock import patch

    from mini_infra.bench.run import parse_args, run_workload

    # 1. 验证命令行参数解析
    with patch("sys.argv", ["mini-infra-bench", "--tag", "test", "--profile-mode", "extra-profile"]):
        args = parse_args()
        assert args.profile_mode == ProfileMode.EXTRA_PROFILE

    with patch("sys.argv", ["mini-infra-bench", "--tag", "test", "--profile-mode", "full-profile"]):
        args = parse_args()
        assert args.profile_mode == ProfileMode.FULL_PROFILE

    with patch("sys.argv", ["mini-infra-bench", "--tag", "test"]):
        args = parse_args()
        assert args.profile_mode is None

    # 验证历史参数不被支持（报错退出）
    with patch("sys.argv", ["mini-infra-bench", "--tag", "test", "--extra-profile-run"]):
        with pytest.raises(SystemExit):
            parse_args()

    # 2. 验证 run_workload 执行逻辑
    model = build_tiny_model()
    mode = Mode()
    device = torch.device("cpu")
    vocab = model.config.vocab_size

    # Mode 1: profile_mode=None -> 无 op_latency_breakdown
    res_none = run_workload(
        model, mode, prompt_len=4, max_new_tokens=2, batch_size=1, warmup=0, repeats=1, vocab=vocab, device=device
    )
    assert "op_latency_breakdown" not in res_none

    # Mode 2: profile_mode=EXTRA_PROFILE -> 有 op_latency_breakdown
    res_extra = run_workload(
        model,
        mode,
        prompt_len=4,
        max_new_tokens=2,
        batch_size=1,
        warmup=0,
        repeats=1,
        vocab=vocab,
        device=device,
        profile_mode=ProfileMode.EXTRA_PROFILE,
    )
    assert "op_latency_breakdown" in res_extra
    assert isinstance(res_extra["op_latency_breakdown"], dict)
    assert "total_ms" in res_extra["op_latency_breakdown"]

    # Mode 3: profile_mode=FULL_PROFILE -> 每次 repeat 均 profile，直接从 runs 中提取 op_latency_breakdown
    res_full = run_workload(
        model,
        mode,
        prompt_len=4,
        max_new_tokens=2,
        batch_size=1,
        warmup=0,
        repeats=2,
        vocab=vocab,
        device=device,
        profile_mode=ProfileMode.FULL_PROFILE,
    )
    assert "op_latency_breakdown" in res_full
    assert isinstance(res_full["op_latency_breakdown"], dict)
    assert "total_ms" in res_full["op_latency_breakdown"]


def test_merge_op_latency_breakdowns():
    # 1. 空列表和单项处理
    assert merge_op_latency_breakdowns([]) is None

    b1: OpLatencyBreakdown = {
        "total_ms": 10.0,
        "top_kernels": [
            {"name": "aten::mm", "category": "gemm", "cuda_ms": 6.0, "share": 0.6, "calls": 2},
            {"name": "aten::silu", "category": "elementwise", "cuda_ms": 4.0, "share": 0.4, "calls": 4},
        ],
        "time_share_by_kernel_category": {"gemm": 0.6, "elementwise": 0.4},
        "time_share_by_module": {"mlp_block": 0.7, "attention_block": 0.3},
    }
    assert merge_op_latency_breakdowns([b1]) == b1

    # 2. 多项聚合
    b2: OpLatencyBreakdown = {
        "total_ms": 20.0,
        "top_kernels": [
            {"name": "aten::mm", "category": "gemm", "cuda_ms": 14.0, "share": 0.7, "calls": 4},
            {"name": "aten::softmax", "category": "softmax", "cuda_ms": 6.0, "share": 0.3, "calls": 2},
        ],
        "time_share_by_kernel_category": {"gemm": 0.7, "softmax": 0.3},
        "time_share_by_module": {"mlp_block": 0.5, "attention_block": 0.5},
    }

    merged = merge_op_latency_breakdowns([b1, b2])
    assert merged is not None
    # 累加 total_ms (10.0 + 20.0)
    assert merged["total_ms"] == pytest.approx(30.0)

    # 聚合 top_kernels (总调用次数与总耗时)
    kernels_dict = {k["name"]: k for k in merged["top_kernels"]}
    assert "aten::mm" in kernels_dict
    assert kernels_dict["aten::mm"]["calls"] == 6  # 2 + 4
    assert kernels_dict["aten::mm"]["cuda_ms"] == pytest.approx(20.0)  # 6 + 14
    assert kernels_dict["aten::silu"]["calls"] == 4  # 4 + 0
    assert kernels_dict["aten::silu"]["cuda_ms"] == pytest.approx(4.0)  # 4 + 0
    assert kernels_dict["aten::softmax"]["calls"] == 2  # 0 + 2
    assert kernels_dict["aten::softmax"]["cuda_ms"] == pytest.approx(6.0)  # 0 + 6

    # 加权聚合 time_share_by_kernel_category (b1 权重 10，b2 权重 20，总时间 30)
    # gemm: (0.6*10 + 0.7*20) / 30 = (6 + 14) / 30 = 20 / 30 ≈ 0.6667
    assert merged["time_share_by_kernel_category"]["gemm"] == pytest.approx(20.0 / 30.0)

    # 加权聚合 time_share_by_module
    # mlp_block: (0.7*10 + 0.5*20) / 30 = (7 + 10) / 30 = 17 / 30
    assert merged["time_share_by_module"]["mlp_block"] == pytest.approx(17.0 / 30.0)


def test_merge_op_latency_breakdowns_global_topk():
    """验证：即使某个 kernel 在每个单次 run 的 top_kernels 里都不在（如排在局部第 3，局部 top=2），
    但从全量 all_kernels 聚合后，其累计耗时可以成为全局第 1 名并入选最终 top_kernels。"""
    # run 1: aten::mm (10ms), aten::silu (9ms), aten::add (8ms) -> top_kernels 只有 mm 和 silu
    b1: OpLatencyBreakdown = {
        "total_ms": 27.0,
        "top_kernels": [
            {"name": "aten::mm", "category": "gemm", "cuda_ms": 10.0, "share": 10 / 27, "calls": 1},
            {"name": "aten::silu", "category": "elementwise", "cuda_ms": 9.0, "share": 9 / 27, "calls": 1},
        ],
        "all_kernels": [
            {"name": "aten::mm", "category": "gemm", "cuda_ms": 10.0, "share": 10 / 27, "calls": 1},
            {"name": "aten::silu", "category": "elementwise", "cuda_ms": 9.0, "share": 9 / 27, "calls": 1},
            {"name": "aten::add", "category": "elementwise", "cuda_ms": 8.0, "share": 8 / 27, "calls": 1},
        ],
        "time_share_by_kernel_category": {"gemm": 10 / 27, "elementwise": 17 / 27},
        "time_share_by_module": {"mlp_block": 1.0},
    }

    # run 2: aten::bmm (10ms), aten::mul (9ms), aten::add (8ms) -> top_kernels 只有 bmm 和 mul
    b2: OpLatencyBreakdown = {
        "total_ms": 27.0,
        "top_kernels": [
            {"name": "aten::bmm", "category": "gemm", "cuda_ms": 10.0, "share": 10 / 27, "calls": 1},
            {"name": "aten::mul", "category": "elementwise", "cuda_ms": 9.0, "share": 9 / 27, "calls": 1},
        ],
        "all_kernels": [
            {"name": "aten::bmm", "category": "gemm", "cuda_ms": 10.0, "share": 10 / 27, "calls": 1},
            {"name": "aten::mul", "category": "elementwise", "cuda_ms": 9.0, "share": 9 / 27, "calls": 1},
            {"name": "aten::add", "category": "elementwise", "cuda_ms": 8.0, "share": 8 / 27, "calls": 1},
        ],
        "time_share_by_kernel_category": {"gemm": 10 / 27, "elementwise": 17 / 27},
        "time_share_by_module": {"mlp_block": 1.0},
    }

    # 聚合时请求 top=2
    merged = merge_op_latency_breakdowns([b1, b2], top=2)
    assert merged is not None
    assert merged["total_ms"] == pytest.approx(54.0)

    # aten::add 累计 8 + 8 = 16ms，超过 mm(10), bmm(10), silu(9), mul(9)，成为全局第 1 名！
    top_names = [k["name"] for k in merged["top_kernels"]]
    assert top_names[0] == "aten::add"
    assert merged["top_kernels"][0]["cuda_ms"] == pytest.approx(16.0)
    assert merged["top_kernels"][0]["calls"] == 2
    assert merged["top_kernels"][0]["share"] == pytest.approx(16.0 / 54.0)
    assert len(merged["top_kernels"]) == 2
    assert len(merged["all_kernels"]) == 5
