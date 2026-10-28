"""基准评测用例级强类型领域模型（CaseBenchmarkResult）。

遵循标准三层正交领域模型：
1. 负载与配置层：WorkloadSpec
2. 统计分布汇总与共享画像层：DistributionStats, StepAggregatedMetrics,
   MemoryProfile, BatchScheduleInfo, EfficiencyMetrics
3. 诊断与明细层：OpLatencyBreakdown, raw_samples
"""

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mini_infra.bench.common import EfficiencyMetrics
from mini_infra.core.types import BatchScheduleInfo, MemoryProfile, StepAggregatedMetrics
from mini_infra.profiling import OpLatencyBreakdown
from mini_infra.utils import percentile


@dataclass(frozen=True)
class DistributionStats:
    """一组测量样本的统计分布（均值、中位数、标准差及关键分位数）。"""

    median: float
    mean: float
    std: float
    min: float
    p90: float
    p95: float
    p99: float
    max: float

    def to_dict(self) -> dict[str, float]:
        """导出为与现有下游工具完全兼容的统计字典。"""
        return {
            "median": self.median,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "max": self.max,
        }


def summarize_samples(values: Sequence[float]) -> DistributionStats:
    """计算样本序列的分位数与均值方差指标，返回 DistributionStats 结构体。"""
    if not values:
        return DistributionStats(
            median=0.0,
            mean=0.0,
            std=0.0,
            min=0.0,
            p90=0.0,
            p95=0.0,
            p99=0.0,
            max=0.0,
        )
    return DistributionStats(
        median=statistics.median(values),
        mean=statistics.fmean(values),
        std=statistics.pstdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        p90=percentile(values, 90),
        p95=percentile(values, 95),
        p99=percentile(values, 99),
        max=max(values),
    )


@dataclass(frozen=True)
class WorkloadSpec:
    """评测用例的负载规格与运行环境参数。"""

    mode: str
    batch_size: int
    prompt_len: int | tuple[int, ...]
    max_new_tokens: int | tuple[int, ...]
    repeats: int
    warmup: int


@dataclass(frozen=True)
class CaseBenchmarkResult:
    """单评测用例（多轮 Repeats 聚合）的标准性能与资源分析结果。

    直接组合复用 core.types.StepAggregatedMetrics[DistributionStats]、MemoryProfile、
    BatchScheduleInfo 与 EfficiencyMetrics，与 GenerationResult 顶层结构对齐。
    """

    workload: WorkloadSpec
    metrics: StepAggregatedMetrics[DistributionStats]
    memory: MemoryProfile
    packing: BatchScheduleInfo | None = None
    efficiency: EfficiencyMetrics | None = None
    op_breakdown: OpLatencyBreakdown | None = None
    raw_samples: dict[str, list[float]] = field(default_factory=dict)

    def to_record_dict(self) -> dict[str, Any]:
        """导出为标准评测记录字典格式（批处理打包与槽位利用率统一序列化至 'packing' 字段）。"""
        rec: dict[str, Any] = {
            "mode": self.workload.mode,
            "prompt_len": (
                list(self.workload.prompt_len)
                if isinstance(self.workload.prompt_len, tuple)
                else self.workload.prompt_len
            ),
            "max_new_tokens": (
                list(self.workload.max_new_tokens)
                if isinstance(self.workload.max_new_tokens, tuple)
                else self.workload.max_new_tokens
            ),
            "batch_size": self.workload.batch_size,
            "repeats": self.workload.repeats,
            "warmup": self.workload.warmup,
            "ttft_ms": self.metrics.ttft_ms.to_dict() if self.metrics.ttft_ms is not None else None,
            "tpot_ms": self.metrics.mean_tpot_ms.to_dict() if self.metrics.mean_tpot_ms is not None else None,
            "e2e_wall_ms": self.metrics.e2e_wall_ms.to_dict(),
            "e2e_model_ms": self.metrics.e2e_model_ms.to_dict(),
            "throughput_tok_s": self.metrics.throughput_tok_s.to_dict(),
            "throughput_req_s": self.metrics.throughput_req_s.to_dict(),
            "gpu_time_ratio": (
                self.metrics.gpu_time_ratio.to_dict() if self.metrics.gpu_time_ratio is not None else None
            ),
            "memory": self.memory.to_dict(),
            "raw": self.raw_samples,
        }
        if self.packing is not None:
            packing_dict = self.packing.to_dict()
            if self.metrics.effective_prompt_tok_s is not None:
                packing_dict["effective_prompt_token_throughput_per_second"] = (
                    self.metrics.effective_prompt_tok_s.to_dict()
                )
            if self.metrics.padded_prompt_tok_s is not None:
                packing_dict["padded_prompt_token_throughput_per_second"] = (
                    self.metrics.padded_prompt_tok_s.to_dict()
                )
            rec["packing"] = packing_dict
        if self.efficiency is not None:
            rec["efficiency"] = self.efficiency.to_dict()
        if self.op_breakdown is not None:
            rec["op_latency_breakdown"] = asdict(self.op_breakdown)
        return rec
