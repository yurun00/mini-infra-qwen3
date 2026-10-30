"""基准评测用例级强类型领域模型（LockstepCaseResult）。

遵循标准三层正交领域模型：
1. 负载与配置层：WorkloadSpec
2. 统计分布汇总与共享画像层：DistributionStats, StepAggregatedMetrics,
   MemoryProfile, BatchScheduleInfo, EfficiencyMetrics
3. 诊断与明细层：OpLatencyBreakdown
"""

from dataclasses import asdict, dataclass
from typing import Any

from mini_infra.bench.common import (
    DistributionStats,
    EfficiencyMetrics,
    WorkloadSpec,
    summarize_samples,
)
from mini_infra.core.types import BatchScheduleInfo, MemoryProfile, StepAggregatedMetrics
from mini_infra.profiling import OpLatencyBreakdown


@dataclass(frozen=True)
class LockstepCaseResult:
    """单评测用例（多轮 Repeats 聚合）的标准性能与资源分析结果。

    直接组合复用 core.types.StepAggregatedMetrics[DistributionStats]、MemoryProfile、
    BatchScheduleInfo 与 EfficiencyMetrics，与 GenerationResult 顶层结构对齐。
    """

    workload: WorkloadSpec
    metrics: StepAggregatedMetrics[DistributionStats]
    memory: MemoryProfile
    packing: BatchScheduleInfo[DistributionStats] | None = None
    efficiency: EfficiencyMetrics | None = None
    op_breakdown: OpLatencyBreakdown | None = None

    def to_record_dict(self) -> dict[str, Any]:
        """导出为标准评测记录字典格式（批处理打包与槽位利用率统一序列化至 'packing' 字段）。"""
        rec: dict[str, Any] = {
            "mode": str(self.workload.mode),
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
            "effective_prompt_tok_s": (
                self.metrics.effective_prompt_tok_s.to_dict()
                if self.metrics.effective_prompt_tok_s is not None
                else None
            ),
            "padded_prompt_tok_s": (
                self.metrics.padded_prompt_tok_s.to_dict() if self.metrics.padded_prompt_tok_s is not None else None
            ),
            "memory": self.memory.to_dict(),
        }
        if self.workload.scale is not None:
            rec["scale"] = self.workload.scale
        if self.workload.seed is not None:
            rec["seed"] = self.workload.seed
        if self.packing is not None:
            rec["packing"] = self.packing.to_dict()
        if self.efficiency is not None:
            rec["efficiency"] = self.efficiency.to_dict()
        if self.op_breakdown is not None:
            rec["op_latency_breakdown"] = asdict(self.op_breakdown)
        return rec


__all__ = [
    "DistributionStats",
    "LockstepCaseResult",
    "WorkloadSpec",
    "summarize_samples",
]
