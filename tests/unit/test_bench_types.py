"""单元测试：验证 bench/types.py 中 LockstepCaseResult 强类型领域模型与功能。"""

from dataclasses import FrozenInstanceError

import pytest

from mini_infra.bench.common import EfficiencyMetrics
from mini_infra.bench.lockstep.types import (
    DistributionStats,
    LockstepCaseResult,
    WorkloadSpec,
    summarize_samples,
)
from mini_infra.core.types import BatchScheduleInfo, MemoryProfile, StepAggregatedMetrics
from mini_infra.engine import FeatureFlag, Mode
from mini_infra.profiling import KernelInfo, OpLatencyBreakdown


def test_distribution_stats_computation_and_immutability():
    """验证 DistributionStats 的统计计算、导出与不可变性。"""
    # 1. 正常样本计算
    samples = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = summarize_samples(samples)

    assert stats.median == 30.0
    assert stats.mean == 30.0
    assert stats.min == 10.0
    assert stats.max == 50.0
    assert stats.p90 == 50.0
    assert stats.p95 == 50.0
    assert stats.std > 0.0

    # 2. 导出 dict
    d = stats.to_dict()
    assert set(d.keys()) == {"median", "mean", "std", "min", "p90", "p95", "p99", "max"}
    assert d["median"] == 30.0

    # 3. 不可变性验证
    with pytest.raises(FrozenInstanceError):
        stats.median = 99.0  # type: ignore[misc]

    # 4. 空样本边界防护
    empty_stats = summarize_samples([])
    assert empty_stats.median == 0.0
    assert empty_stats.mean == 0.0
    assert empty_stats.max == 0.0


def test_lockstep_case_result_structure_and_dot_access():
    """验证 LockstepCaseResult 强类型点号访问与各领域对象。"""
    workload = WorkloadSpec(
        mode=Mode({FeatureFlag.KV_CACHE}),
        batch_size=2,
        prompt_len=(16, 32),
        max_new_tokens=16,
        repeats=5,
        warmup=2,
    )

    dummy_stat = DistributionStats(10.0, 10.0, 0.5, 9.0, 10.5, 10.8, 11.0, 11.0)
    metrics: StepAggregatedMetrics[DistributionStats] = StepAggregatedMetrics(
        ttft_ms=dummy_stat,
        mean_tpot_ms=dummy_stat,
        e2e_wall_ms=dummy_stat,
        e2e_model_ms=dummy_stat,
        throughput_tok_s=dummy_stat,
        throughput_req_s=dummy_stat,
        gpu_time_ratio=dummy_stat,
    )
    memory = MemoryProfile(
        weights_mb=500.0,
        baseline_mb=600.0,
        prefill_mb=650.0,
        peak_allocated_mb=700.0,
        peak_reserved_mb=800.0,
        kv_cache_mb=12.0,
    )
    pad_stat = summarize_samples([0.85, 0.85])
    slot_stat = summarize_samples([0.90, 0.90])
    packing: BatchScheduleInfo[DistributionStats] = BatchScheduleInfo(
        padding_efficiency=pad_stat,
        slot_utilization=slot_stat,
    )
    efficiency = EfficiencyMetrics(
        prefill_mfu=0.15,
        decode_mfu=0.25,
        decode_mbu=0.45,
    )
    op_breakdown = OpLatencyBreakdown(
        total_ms=100.0,
        top_kernels=[KernelInfo("gemm", "gemm", 50.0, 0.5, 10)],
        time_share_by_kernel_category={"gemm": 0.5},
        time_share_by_module={"layers": 0.9},
    )

    case_result = LockstepCaseResult(
        workload=workload,
        metrics=metrics,
        memory=memory,
        packing=packing,
        efficiency=efficiency,
        op_breakdown=op_breakdown,
    )

    # 1. 验证强类型属性点号访问
    assert case_result.workload.batch_size == 2
    assert case_result.workload.prompt_len == (16, 32)
    assert case_result.metrics.ttft_ms is not None and case_result.metrics.ttft_ms.median == 10.0
    assert case_result.metrics.throughput_tok_s.mean == 10.0
    assert case_result.memory.peak_allocated_mb == 700.0
    assert case_result.memory.dynamic_peak_mb == 100.0
    assert case_result.memory.fragmentation_mb == 100.0
    assert case_result.packing is not None and case_result.packing.padding_efficiency.median == 0.85
    assert case_result.efficiency is not None and case_result.efficiency.decode_mbu == 0.45
    assert case_result.metrics.gpu_time_ratio is not None and case_result.metrics.gpu_time_ratio.median == 10.0
    assert case_result.op_breakdown is not None and case_result.op_breakdown.total_ms == 100.0

    # 2. 验证不可变性
    with pytest.raises(FrozenInstanceError):
        case_result.workload = workload  # type: ignore[misc]

    # 3. 验证严禁使用字典下标访问
    with pytest.raises(TypeError):
        _ = case_result["latency"]  # type: ignore[index]

    # 4. 验证 to_record_dict() 字典导出结构
    rec = case_result.to_record_dict()
    assert rec["mode"] == "kv-cache"
    assert rec["batch_size"] == 2
    assert rec["prompt_len"] == [16, 32]
    assert rec["max_new_tokens"] == 16
    assert rec["ttft_ms"]["median"] == 10.0
    assert rec["tpot_ms"]["median"] == 10.0
    assert rec["memory"]["peak_allocated_mb"] == 700.0
    assert rec["memory"]["dynamic_peak_mb"] == 100.0
    assert rec["memory"]["fragmentation_mb"] == 100.0
    assert rec["packing"]["padding_efficiency"]["median"] == 0.85
    assert rec["packing"]["slot_utilization"]["median"] == 0.90
    assert rec["efficiency"]["decode_mbu"] == 0.45
    assert rec["op_latency_breakdown"]["total_ms"] == 100.0
