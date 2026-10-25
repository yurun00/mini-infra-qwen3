"""方案 B 零妥协强类型架构单元测试。

验证：
1. 全面属性化（Dot-attribute Access）与不可变数据类（frozen dataclass）
2. 彻底消除方括号字典访问与 __getattr__ 穿透
3. 六大正交领域分离（Outputs, Metrics, Memory, Trace, Profiling, BatchInfo）
4. 计算逻辑与容器分离纯函数（build_memory_profile, compute_step_aggregated_metrics, aggregate_generation_results）
"""

from dataclasses import FrozenInstanceError, is_dataclass

import pytest
import torch

from mini_infra.generation.result import (
    AggregationOptions,
    BatchScheduleInfo,
    GenerationResult,
    GenerationTrace,
    MemoryProfile,
    SequenceOutput,
    StepAggregatedMetrics,
    aggregate_generation_results,
    build_memory_profile,
    compute_step_aggregated_metrics,
    percentile,
)
from mini_infra.profiling import KernelInfo, OpLatencyBreakdown


def test_frozen_dataclass_contract():
    """验证所有核心数据类均为 frozen dataclass，且支持不可变性保护。"""
    classes = [
        SequenceOutput,
        MemoryProfile,
        StepAggregatedMetrics,
        GenerationTrace,
        BatchScheduleInfo,
        GenerationResult,
        AggregationOptions,
        KernelInfo,
        OpLatencyBreakdown,
    ]
    for cls in classes:
        assert is_dataclass(cls), f"{cls.__name__} 应当是 dataclass"

    seq = SequenceOutput(
        request_index=0,
        token_ids=[1, 2, 3],
        prompt_len=4,
        output_len=3,
        finish_reason="stop",
    )
    with pytest.raises(FrozenInstanceError):
        seq.finish_reason = "max_tokens"  # type: ignore[misc]

    mem = MemoryProfile(
        baseline_mb=100.0,
        prefill_mb=150.0,
        peak_allocated_mb=200.0,
        peak_reserved_mb=250.0,
        kv_cache_mb=30.0,
    )
    with pytest.raises(FrozenInstanceError):
        mem.baseline_mb = 120.0  # type: ignore[misc]


def test_no_subscript_or_dynamic_attr_proxy():
    """验证彻底移除了字典访问与 __getattr__ 代理。"""
    seq = SequenceOutput(
        request_index=0,
        token_ids=[10, 20],
        prompt_len=2,
        output_len=2,
        finish_reason="max_tokens",
    )
    metrics = StepAggregatedMetrics(
        ttft_ms=10.0,
        mean_tpot_ms=5.0,
        p50_tpot_ms=5.0,
        p95_tpot_ms=5.0,
        e2e_wall_ms=20.0,
        e2e_model_ms=20.0,
        throughput_tok_s=100.0,
        throughput_req_s=50.0,
        gpu_time_ratio=1.0,
    )
    mem = MemoryProfile(
        baseline_mb=100.0,
        prefill_mb=120.0,
        peak_allocated_mb=150.0,
        peak_reserved_mb=200.0,
        kv_cache_mb=10.0,
    )
    trace = GenerationTrace(
        step_latencies_ms=[10.0, 5.0, 5.0],
        active_tokens_per_step=[1, 1, 1],
    )
    result = GenerationResult(
        sequences=[seq],
        metrics=metrics,
        memory=mem,
        trace=trace,
        batch_size=1,
    )
    assert torch.equal(result.padded_output_tokens(), torch.tensor([[10, 20]]))

    # 1. 方括号访问引发 TypeError（'GenerationResult' object is not subscriptable）
    with pytest.raises(TypeError):
        _ = result["metrics"]  # type: ignore[index]
    with pytest.raises(TypeError):
        _ = result["ttft_ms"]  # type: ignore[index]
    with pytest.raises(TypeError):
        _ = result["baseline_mb"]  # type: ignore[index]

    # 2. 彻底无 get / __contains__ / __getattr__ 代理
    assert not hasattr(result, "get")
    with pytest.raises(AttributeError):
        _ = result.ttft_ms  # 不再穿透代理，必须 result.metrics.ttft_ms
    with pytest.raises(AttributeError):
        _ = result.baseline_mb  # 必须 result.memory.baseline_mb
    with pytest.raises(TypeError):
        _ = "metrics" in result  # type: ignore[operator]

    # 3. 强类型点属性访问
    assert result.metrics.ttft_ms == 10.0
    assert result.memory.peak_allocated_mb == 150.0
    assert result.trace.step_latencies_ms == [10.0, 5.0, 5.0]
    assert result.sequences[0].token_ids == [10, 20]


def test_memory_profile_properties():
    """验证 MemoryProfile 的只读动态属性计算。"""
    mem = MemoryProfile(
        baseline_mb=100.0,
        prefill_mb=140.0,
        peak_allocated_mb=180.0,
        peak_reserved_mb=220.0,
        kv_cache_mb=25.0,
    )
    assert mem.dynamic_peak_mb == pytest.approx(80.0)  # 180 - 100
    assert mem.fragmentation_mb == pytest.approx(40.0)  # 220 - 180

    # 验证非负防护
    mem_low = MemoryProfile(
        baseline_mb=200.0,
        prefill_mb=150.0,
        peak_allocated_mb=150.0,
        peak_reserved_mb=100.0,
        kv_cache_mb=10.0,
    )
    assert mem_low.dynamic_peak_mb == 0.0
    assert mem_low.fragmentation_mb == 0.0


def test_build_memory_profile():
    """验证 build_memory_profile 字节到 MB 转换。"""
    mib = 1024**2
    mem = build_memory_profile(
        baseline_allocated_bytes=100 * mib,
        memory_after_prefill_bytes=150 * mib,
        peak_allocated_bytes=200 * mib,
        peak_reserved_bytes=300 * mib,
        cache_bytes=50 * mib,
    )
    assert mem.baseline_mb == 100.0
    assert mem.prefill_mb == 150.0
    assert mem.peak_allocated_mb == 200.0
    assert mem.peak_reserved_mb == 300.0
    assert mem.kv_cache_mb == 50.0
    assert mem.dynamic_peak_mb == 100.0
    assert mem.fragmentation_mb == 100.0


def test_compute_step_aggregated_metrics():
    """验证 compute_step_aggregated_metrics 纯函数。"""
    step_latencies = [10.0, 2.0, 4.0, 6.0, 8.0]
    metrics = compute_step_aggregated_metrics(
        step_latencies_ms=step_latencies,
        e2e_wall_ms=35.0,
        e2e_model_ms=30.0,
        total_tokens=8,
        batch_size=2,
        is_cuda=True,
    )
    assert isinstance(metrics, StepAggregatedMetrics)
    assert metrics.ttft_ms == 10.0
    assert metrics.mean_tpot_ms == (2.0 + 4.0 + 6.0 + 8.0) / 4  # 5.0
    assert metrics.p50_tpot_ms == percentile([2.0, 4.0, 6.0, 8.0], 50)
    assert metrics.p95_tpot_ms == percentile([2.0, 4.0, 6.0, 8.0], 95)
    assert metrics.e2e_wall_ms == 35.0
    assert metrics.e2e_model_ms == 30.0
    assert metrics.throughput_tok_s == pytest.approx(8 / (30.0 / 1000))
    assert metrics.throughput_req_s == pytest.approx(2 * 1000 / 30.0)
    assert metrics.gpu_time_ratio == pytest.approx(30.0 / 35.0)


def test_aggregate_generation_results():
    """验证多 Run 结果聚合逻辑。"""
    res1 = GenerationResult(
        sequences=[SequenceOutput(request_index=0, token_ids=[1, 2], prompt_len=3, output_len=2, finish_reason="stop")],
        metrics=StepAggregatedMetrics(
            ttft_ms=10.0,
            mean_tpot_ms=5.0,
            p50_tpot_ms=5.0,
            p95_tpot_ms=5.0,
            e2e_wall_ms=20.0,
            e2e_model_ms=18.0,
            throughput_tok_s=100.0,
            throughput_req_s=50.0,
            gpu_time_ratio=0.9,
        ),
        memory=MemoryProfile(
            baseline_mb=100.0,
            prefill_mb=120.0,
            peak_allocated_mb=150.0,
            peak_reserved_mb=200.0,
            kv_cache_mb=10.0,
        ),
        trace=GenerationTrace(
            step_latencies_ms=[10.0, 4.0, 4.0],
            active_tokens_per_step=[1, 1, 1],
        ),
        batch_size=1,
    )

    res2 = GenerationResult(
        sequences=[
            SequenceOutput(request_index=0, token_ids=[3, 4, 5], prompt_len=4, output_len=3, finish_reason="max_tokens")
        ],
        metrics=StepAggregatedMetrics(
            ttft_ms=12.0,
            mean_tpot_ms=6.0,
            p50_tpot_ms=6.0,
            p95_tpot_ms=6.0,
            e2e_wall_ms=30.0,
            e2e_model_ms=28.0,
            throughput_tok_s=100.0,
            throughput_req_s=33.3,
            gpu_time_ratio=0.93,
        ),
        memory=MemoryProfile(
            baseline_mb=100.0,
            prefill_mb=130.0,
            peak_allocated_mb=180.0,
            peak_reserved_mb=220.0,
            kv_cache_mb=15.0,
        ),
        trace=GenerationTrace(
            step_latencies_ms=[12.0, 6.0, 6.0, 4.0],
            active_tokens_per_step=[1, 1, 1, 1],
        ),
        batch_size=1,
    )

    merged = aggregate_generation_results([res1, res2])

    # 1. 验证 Sequences 与 Padded Tokens
    assert merged.batch_size == 2
    assert len(merged.sequences) == 2
    assert merged.sequences[0].request_index == 0
    assert merged.sequences[0].token_ids == [1, 2]
    assert merged.sequences[1].request_index == 1
    assert merged.sequences[1].token_ids == [3, 4, 5]
    assert merged.padded_output_tokens().shape == (2, 3)
    assert merged.padded_output_tokens().tolist() == [[1, 2, 0], [3, 4, 5]]

    # 2. 验证 Memory
    assert merged.memory.baseline_mb == 100.0
    assert merged.memory.prefill_mb == 130.0
    assert merged.memory.peak_allocated_mb == 180.0
    assert merged.memory.peak_reserved_mb == 220.0
    assert merged.memory.kv_cache_mb == 15.0

    # 3. 验证 Metrics（默认 mean 聚合）
    assert merged.metrics.e2e_wall_ms == 25.0  # (20 + 30) / 2
    assert merged.metrics.e2e_model_ms == 23.0  # (18 + 28) / 2
    assert merged.metrics.ttft_ms == 11.0  # (10 + 12) / 2
    assert merged.metrics.mean_tpot_ms == 5.5  # (5 + 6) / 2
    assert merged.metrics.p50_tpot_ms == 5.5  # (5 + 6) / 2
    assert merged.metrics.p95_tpot_ms == 5.5  # (5 + 6) / 2
    assert merged.metrics.throughput_req_s == pytest.approx((50.0 + 33.3) / 2)
    assert merged.metrics.throughput_tok_s == pytest.approx(100.0)
    assert merged.metrics.gpu_time_ratio == pytest.approx(46.0 / 50.0)

    # 4. 验证 Trace
    assert len(merged.trace.step_latencies_ms) == 7
    assert merged.trace.active_tokens_per_step == [1, 1, 1, 1, 1, 1, 1]


def test_aggregate_generation_results_with_options():
    """验证 aggregate_generation_results 在各种 AggregationOptions 下的聚合正确性。"""
    res1 = GenerationResult(
        sequences=[SequenceOutput(request_index=0, token_ids=[1, 2], prompt_len=2, output_len=2, finish_reason="stop")],
        metrics=StepAggregatedMetrics(
            ttft_ms=10.0,
            mean_tpot_ms=4.0,
            p50_tpot_ms=4.0,
            p95_tpot_ms=4.0,
            e2e_wall_ms=20.0,
            e2e_model_ms=18.0,
            throughput_tok_s=100.0,
            throughput_req_s=50.0,
            gpu_time_ratio=0.9,
        ),
        memory=MemoryProfile(
            baseline_mb=100.0,
            prefill_mb=120.0,
            peak_allocated_mb=150.0,
            peak_reserved_mb=200.0,
            kv_cache_mb=10.0,
        ),
        trace=GenerationTrace(step_latencies_ms=[10.0, 4.0], active_tokens_per_step=[1, 1]),
        batch_size=1,
    )
    res2 = GenerationResult(
        sequences=[SequenceOutput(request_index=0, token_ids=[3, 4], prompt_len=2, output_len=2, finish_reason="stop")],
        metrics=StepAggregatedMetrics(
            ttft_ms=20.0,
            mean_tpot_ms=8.0,
            p50_tpot_ms=8.0,
            p95_tpot_ms=8.0,
            e2e_wall_ms=40.0,
            e2e_model_ms=36.0,
            throughput_tok_s=50.0,
            throughput_req_s=25.0,
            gpu_time_ratio=0.8,
        ),
        memory=MemoryProfile(
            baseline_mb=120.0,
            prefill_mb=140.0,
            peak_allocated_mb=180.0,
            peak_reserved_mb=250.0,
            kv_cache_mb=20.0,
        ),
        trace=GenerationTrace(step_latencies_ms=[20.0, 8.0], active_tokens_per_step=[1, 1]),
        batch_size=1,
    )

    # 1. 域级别配置：metrics="mean", memory="min"
    opt_mean = AggregationOptions(metrics="mean", memory="min")
    merged_mean = aggregate_generation_results([res1, res2], options=opt_mean)
    assert merged_mean.metrics.e2e_wall_ms == 30.0  # (20 + 40) / 2
    assert merged_mean.metrics.e2e_model_ms == 27.0  # (18 + 36) / 2
    assert merged_mean.metrics.ttft_ms == 15.0  # (10 + 20) / 2
    assert merged_mean.metrics.mean_tpot_ms == 6.0  # (4 + 8) / 2
    assert merged_mean.metrics.p50_tpot_ms == 6.0
    assert merged_mean.metrics.throughput_tok_s == 75.0  # (100 + 50) / 2
    assert merged_mean.metrics.gpu_time_ratio == pytest.approx(27.0 / 30.0)  # 派生自 e2e_model_ms / e2e_wall_ms
    assert merged_mean.memory.baseline_mb == 100.0  # min(100, 120)
    assert merged_mean.memory.peak_allocated_mb == 150.0  # min(150, 180)
    assert merged_mean.memory.kv_cache_mb == 10.0  # min(10, 20)

    # 2. 字段级精确覆盖：metrics 默认为 sum，但 ttft_ms="min", peak_allocated_mb="mean"
    opt_field = AggregationOptions(metrics="sum", ttft_ms="min", peak_allocated_mb="mean")
    merged_field = aggregate_generation_results([res1, res2], options=opt_field)
    assert merged_field.metrics.e2e_wall_ms == 60.0  # sum: 20 + 40
    assert merged_field.metrics.e2e_model_ms == 54.0  # sum: 18 + 36
    assert merged_field.metrics.gpu_time_ratio == pytest.approx(54.0 / 60.0)  # 派生自 e2e_model_ms / e2e_wall_ms
    assert merged_field.metrics.ttft_ms == 10.0  # min(10, 20)
    assert merged_field.memory.peak_allocated_mb == 165.0  # mean(150, 180)
    assert merged_field.memory.peak_reserved_mb == 250.0  # memory 默认 max(200, 250)

    # 3. 验证 gpu_time_ratio 完全不受 e2e 时间聚合函数影响
    opt_diff = AggregationOptions(e2e_wall_ms="max", e2e_model_ms="min")
    merged_diff = aggregate_generation_results([res1, res2], options=opt_diff)
    assert merged_diff.metrics.e2e_wall_ms == 40.0  # max(20, 40)
    assert merged_diff.metrics.e2e_model_ms == 18.0  # min(18, 36)
    # 虽然聚合后的 model/wall 是 18/40=0.45，但 gpu_time_ratio 客观反映总耗时之比 (18+36)/(20+40)=0.9
    assert merged_diff.metrics.gpu_time_ratio == pytest.approx(54.0 / 60.0)

    # 4. 自定义 Callable 聚合函数
    opt_custom = AggregationOptions(ttft_ms=lambda vals: max(vals) * 2.0)
    merged_custom = aggregate_generation_results([res1, res2], options=opt_custom)
    assert merged_custom.metrics.ttft_ms == 40.0  # max(10, 20) * 2.0

    # 5. 非法聚合函数字符串引发 ValueError
    with pytest.raises(ValueError, match="Unsupported aggregate function"):
        _ = aggregate_generation_results([res1, res2], options=AggregationOptions(metrics="unknown_fn"))  # type: ignore[arg-type]


def test_padded_output_tokens_method():
    """验证 padded_output_tokens 方法在单请求、多请求、默认及自定义 pad_token_id 场景下的正确性。"""
    # 1. 变长 sequences
    seqs = [
        SequenceOutput(request_index=0, token_ids=[1, 2], prompt_len=2, output_len=2, finish_reason="stop"),
        SequenceOutput(request_index=1, token_ids=[3, 4, 5, 6], prompt_len=4, output_len=4, finish_reason="max_tokens"),
        SequenceOutput(request_index=2, token_ids=[7], prompt_len=1, output_len=1, finish_reason="stop"),
    ]
    res = GenerationResult(
        sequences=seqs,
        metrics=StepAggregatedMetrics(0, 0, 0, 0, 0, 0, 0, 0),
        memory=MemoryProfile(0, 0, 0, 0, 0),
        trace=GenerationTrace([], []),
        batch_size=3,
    )
    # 默认整批 2D：pad_token_id=0
    assert res.padded_output_tokens().shape == (3, 4)
    assert res.padded_output_tokens().tolist() == [
        [1, 2, 0, 0],
        [3, 4, 5, 6],
        [7, 0, 0, 0],
    ]

    # 显式传入自定义 pad_token_id=99
    custom_padded = res.padded_output_tokens(pad_token_id=99)
    assert custom_padded.shape == (3, 4)
    assert custom_padded.tolist() == [
        [1, 2, 99, 99],
        [3, 4, 5, 6],
        [7, 99, 99, 99],
    ]

    # 2. 单个 request_idx 输入：返回 1D Tensor
    req0 = res.padded_output_tokens(0)
    assert req0.shape == (2,)
    assert req0.tolist() == [1, 2]

    req1 = res.padded_output_tokens(request_idx=1)
    assert req1.shape == (4,)
    assert req1.tolist() == [3, 4, 5, 6]

    # 不存在的 request_idx 引发 IndexError
    with pytest.raises(IndexError, match="Request index 999 not found"):
        res.padded_output_tokens(999)

    # 3. 多个 request_idx 列表输入：返回指定的 2D 对齐 Tensor
    subset = res.padded_output_tokens([2, 0], pad_token_id=-1)
    assert subset.shape == (2, 2)
    assert subset.tolist() == [
        [7, -1],
        [1, 2],
    ]

    with pytest.raises(IndexError, match="Request index 100 not found"):
        res.padded_output_tokens([0, 100])

    # 4. 空 sequences
    empty_res = GenerationResult(
        sequences=[],
        metrics=StepAggregatedMetrics(0, 0, 0, 0, 0, 0, 0, 0),
        memory=MemoryProfile(0, 0, 0, 0, 0),
        trace=GenerationTrace([], []),
        batch_size=0,
    )
    assert empty_res.padded_output_tokens().shape == (0, 0)
    assert empty_res.padded_output_tokens(pad_token_id=123).shape == (0, 0)
