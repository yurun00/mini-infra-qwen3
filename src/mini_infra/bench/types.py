"""基准评测用例级强类型领域模型（CaseBenchmarkResult）。

遵循标准三层正交领域模型：
1. 负载与配置层：WorkloadSpec
2. 统计分布汇总层：DistributionStats, LatencyDistributions, ThroughputDistributions,
   CaseMemoryProfile, BatchScheduleSummary, HardwareEfficiency
3. 诊断与明细层：OpLatencyBreakdown, raw_samples
"""

import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from mini_infra.generation.result import percentile
from mini_infra.profiling import OpLatencyBreakdown


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
class LatencyDistributions:
    """端到端与各阶段时延的统计分布（单位 ms）。"""

    ttft_ms: DistributionStats
    tpot_ms: DistributionStats
    e2e_wall_ms: DistributionStats
    e2e_model_ms: DistributionStats


@dataclass(frozen=True)
class ThroughputDistributions:
    """吞吐速率的统计分布。"""

    throughput_tok_s: DistributionStats
    throughput_req_s: DistributionStats
    effective_prompt_tok_s: DistributionStats | None = None
    padded_prompt_tok_s: DistributionStats | None = None


@dataclass(frozen=True)
class CaseMemoryProfile:
    """多次测量中的显存占用画像（单位 MB）。"""

    weights_mb: float | None
    baseline_allocated_mb: float
    prefill_allocated_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    kv_cache_mb: float

    @property
    def dynamic_peak_mb(self) -> float:
        """动态显存分配峰值增量 (peak_allocated - baseline)。"""
        return max(0.0, self.peak_allocated_mb - self.baseline_allocated_mb)

    @property
    def fragmentation_mb(self) -> float:
        """显存分配器碎片/未利用预留 (peak_reserved - peak_allocated)。"""
        return max(0.0, self.peak_reserved_mb - self.peak_allocated_mb)

    def to_dict(self) -> dict[str, Any]:
        """导出为标准字典。"""
        return {
            "weights_mb": self.weights_mb,
            "baseline_allocated_mb": self.baseline_allocated_mb,
            "prefill_allocated_mb": self.prefill_allocated_mb,
            "peak_allocated_mb": self.peak_allocated_mb,
            "dynamic_peak_mb": self.dynamic_peak_mb,
            "peak_reserved_mb": self.peak_reserved_mb,
            "fragmentation_mb": self.fragmentation_mb,
            "kv_cache_mb": self.kv_cache_mb,
        }


@dataclass(frozen=True)
class HardwareEfficiency:
    """硬件物理饱和度分析（MBU 带宽利用率与 MFU 算力利用率）。"""

    prefill_mfu: float | None = None
    decode_mfu: float | None = None
    decode_mbu: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        """导出为标准字典。"""
        return {
            "prefill_mfu": self.prefill_mfu,
            "decode_mfu": self.decode_mfu,
            "decode_mbu": self.decode_mbu,
        }


@dataclass(frozen=True)
class BatchScheduleSummary:
    """批处理调度特征与槽位利用率汇总（终结 extra 杂物箱）。"""

    padding_efficiency: float
    slot_utilization: float
    prompt_tokens: int
    padded_prompt_tokens: int
    prompt_lengths_per_req: list[int]
    completion_steps_per_req: list[int]
    effective_prompt_throughput: DistributionStats | None = None
    padded_prompt_throughput: DistributionStats | None = None

    def to_dict(self) -> dict[str, Any]:
        """导出为调度特征字典结构。"""
        d: dict[str, Any] = {
            "padding_efficiency": self.padding_efficiency,
            "slot_utilization": self.slot_utilization,
            "prompt_tokens": self.prompt_tokens,
            "padded_prompt_tokens": self.padded_prompt_tokens,
            "prompt_lengths_per_req": self.prompt_lengths_per_req,
            "completion_steps_per_req": self.completion_steps_per_req,
        }
        if self.effective_prompt_throughput is not None:
            d["effective_prompt_token_throughput_per_second"] = self.effective_prompt_throughput.to_dict()
        if self.padded_prompt_throughput is not None:
            d["padded_prompt_token_throughput_per_second"] = self.padded_prompt_throughput.to_dict()
        return d


@dataclass(frozen=True)
class CaseBenchmarkResult:
    """单评测用例（多轮 Repeats 聚合）的标准性能与资源分析结果。"""

    workload: WorkloadSpec
    latency: LatencyDistributions
    throughput: ThroughputDistributions
    memory: CaseMemoryProfile
    scheduling: BatchScheduleSummary | None = None
    efficiency: HardwareEfficiency | None = None
    gpu_time_ratio: DistributionStats | None = None
    op_breakdown: OpLatencyBreakdown | None = None
    raw_samples: dict[str, list[float]] = field(default_factory=dict)

    def to_record_dict(self) -> dict[str, Any]:
        """导出为标准评测记录字典格式。"""
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
            "ttft_ms": self.latency.ttft_ms.to_dict(),
            "tpot_ms": self.latency.tpot_ms.to_dict(),
            "e2e_wall_ms": self.latency.e2e_wall_ms.to_dict(),
            "e2e_model_ms": self.latency.e2e_model_ms.to_dict(),
            "throughput_tok_s": self.throughput.throughput_tok_s.to_dict(),
            "throughput_req_s": self.throughput.throughput_req_s.to_dict(),
            "gpu_time_ratio": self.gpu_time_ratio.to_dict() if self.gpu_time_ratio is not None else None,
            "memory": self.memory.to_dict(),
            "raw": self.raw_samples,
        }
        if self.scheduling is not None:
            rec["extra"] = self.scheduling.to_dict()
        if self.efficiency is not None:
            rec["efficiency"] = self.efficiency.to_dict()
        if self.op_breakdown is not None:
            rec["op_latency_breakdown"] = asdict(self.op_breakdown)
        return rec
