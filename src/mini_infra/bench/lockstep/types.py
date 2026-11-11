"""基准评测用例级强类型领域模型（LockstepCaseResult）。

遵循标准三层正交领域模型：
1. 负载画像层：WorkloadProfile
2. 统计分布汇总与共享画像层：DistributionStats, StepAggregatedMetrics,
   MemoryProfile, BatchScheduleInfo, EfficiencyMetrics
3. 诊断与明细层：OpLatencyBreakdown
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from mini_infra.bench.cases import CaseKey, WorkloadProfile
from mini_infra.bench.efficiency import EfficiencyMetrics, HardwarePeakSpecs, ModelCostProfile
from mini_infra.bench.stats import DistributionStats
from mini_infra.core.types import collapse_uniform
from mini_infra.engine.metrics import (
    BatchScheduleInfo,
    MemoryProfile,
    StepAggregatedMetrics,
    summarize_repeated_kv_cache_profiles,
)
from mini_infra.engine.results import GenerationResult
from mini_infra.profiling import OpLatencyBreakdown


@dataclass(frozen=True)
class LockstepCaseResult:
    """单评测用例（多轮 Repeats 聚合）的标准性能与资源分析结果。

    直接组合复用 engine.metrics.StepAggregatedMetrics[DistributionStats]、MemoryProfile、
    BatchScheduleInfo 与 EfficiencyMetrics，与 GenerationResult 顶层结构对齐。
    """

    key: CaseKey
    workload: WorkloadProfile
    metrics: StepAggregatedMetrics[DistributionStats]
    memory: MemoryProfile
    repeats: int = 1
    warmup: int = 0
    packing: BatchScheduleInfo[DistributionStats] | None = None
    efficiency: EfficiencyMetrics | None = None
    op_breakdown: OpLatencyBreakdown | None = None
    scale: str | None = None
    seed: int | None = None

    def to_record_dict(self) -> dict[str, Any]:
        """导出为标准评测记录字典格式（批处理打包与槽位利用率统一序列化至 'packing' 字段）。"""
        pl = collapse_uniform(self.workload.prompt_lens)
        mn = collapse_uniform(self.workload.output_budgets)
        rec: dict[str, Any] = {
            "mode": str(self.key.mode),
            "prompt_len": list(pl) if isinstance(pl, tuple) else pl,
            "max_new_tokens": list(mn) if isinstance(mn, tuple) else mn,
            "num_total_requests": self.workload.num_total_requests,
            "repeats": self.repeats,
            "warmup": self.warmup,
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
        if self.scale is not None:
            rec["scale"] = self.scale
        if self.seed is not None:
            rec["seed"] = self.seed
        if self.packing is not None:
            rec["packing"] = self.packing.to_dict()
        if self.efficiency is not None:
            rec["efficiency"] = self.efficiency.to_dict()
        if self.op_breakdown is not None:
            rec["op_latency_breakdown"] = asdict(self.op_breakdown)
        return rec

    def format_summary(self) -> str:
        """格式化单用例的延迟/吞吐/显存摘要与 MBU/MFU 行。"""
        ttft = self.metrics.ttft_ms
        tpot = self.metrics.mean_tpot_ms
        lines = [
            "  TTFT %.2f ms | TPOT %.2f±%.2f ms | %.1f tok/s | %.2f req/s | %.0f MiB"
            % (
                ttft.median if ttft else 0.0,
                tpot.median if tpot else 0.0,
                tpot.std if tpot else 0.0,
                self.metrics.throughput_tok_s.median,
                self.metrics.throughput_req_s.median,
                self.memory.peak_allocated_mb,
            )
        ]
        if self.efficiency and self.efficiency.decode_mbu is not None:
            gpu_busy_str = (
                "%.0f%%" % (100 * self.metrics.gpu_time_ratio.median)
                if self.metrics.gpu_time_ratio is not None
                else "N/A"
            )
            lines.append(
                "  MBU %.1f%% | decode MFU %.2f%% | prefill MFU %.1f%% | GPU busy %s"
                % (
                    100 * self.efficiency.decode_mbu,
                    100 * (self.efficiency.decode_mfu or 0),
                    100 * (self.efficiency.prefill_mfu or 0),
                    gpu_busy_str,
                )
            )
        return "\n".join(lines)

    @classmethod
    def from_runs(
        cls,
        case_key: CaseKey,
        workload: WorkloadProfile,
        runs: Sequence[GenerationResult],
        *,
        warmup: int = 0,
        cost: ModelCostProfile | None = None,
        peaks: HardwarePeakSpecs | None = None,
        op_breakdown: OpLatencyBreakdown | None = None,
        scale: str | None = None,
        seed: int | None = None,
    ) -> "LockstepCaseResult":
        """将具有相同 CaseKey 与负载规格的多轮 GenerationResult 聚合为 LockstepCaseResult。"""
        if not runs:
            raise ValueError("runs 不能为空")

        eff_prompt_vals = [
            r.metrics.effective_prompt_tok_s for r in runs if r.metrics.effective_prompt_tok_s is not None
        ]
        pad_prompt_vals = [r.metrics.padded_prompt_tok_s for r in runs if r.metrics.padded_prompt_tok_s is not None]
        gpu_ratios = [r.metrics.gpu_time_ratio for r in runs]
        gpu_ratio_stat = (
            DistributionStats.from_samples([v for v in gpu_ratios if v is not None])
            if all(v is not None for v in gpu_ratios)
            else None
        )

        metrics: StepAggregatedMetrics[DistributionStats] = StepAggregatedMetrics(
            ttft_ms=DistributionStats.from_samples([r.metrics.ttft_ms for r in runs if r.metrics.ttft_ms is not None]),
            mean_tpot_ms=DistributionStats.from_samples(
                [r.metrics.mean_tpot_ms for r in runs if r.metrics.mean_tpot_ms is not None]
            ),
            e2e_wall_ms=DistributionStats.from_samples([r.metrics.e2e_wall_ms for r in runs]),
            e2e_model_ms=DistributionStats.from_samples([r.metrics.e2e_model_ms for r in runs]),
            throughput_tok_s=DistributionStats.from_samples([r.metrics.throughput_tok_s for r in runs]),
            throughput_req_s=DistributionStats.from_samples([r.metrics.throughput_req_s for r in runs]),
            gpu_time_ratio=gpu_ratio_stat,
            effective_prompt_tok_s=DistributionStats.from_samples(eff_prompt_vals) if eff_prompt_vals else None,
            padded_prompt_tok_s=DistributionStats.from_samples(pad_prompt_vals) if pad_prompt_vals else None,
        )

        mib = 1024**2
        weights_mb = cost.weight_bytes / mib if cost else None
        case_memory = MemoryProfile(
            weights_mb=weights_mb,
            baseline_mb=max((r.memory.baseline_mb for r in runs), default=0.0),
            prefill_mb=max((r.memory.prefill_mb or 0.0 for r in runs), default=0.0),
            peak_allocated_mb=max((r.memory.peak_allocated_mb for r in runs), default=0.0),
            peak_reserved_mb=max((r.memory.peak_reserved_mb for r in runs), default=0.0),
            kv_cache=summarize_repeated_kv_cache_profiles([r.memory.kv_cache for r in runs]),
        )

        batch_infos = [r.batch_info for r in runs if r.batch_info is not None]
        packing: BatchScheduleInfo[DistributionStats] | None = None
        if batch_infos:
            packing = BatchScheduleInfo(
                padding_efficiency=DistributionStats.from_samples([b.padding_efficiency for b in batch_infos]),
                slot_utilization=DistributionStats.from_samples([b.slot_utilization for b in batch_infos]),
            )

        eff = (
            EfficiencyMetrics.compute(
                cost,
                peaks,
                workload.num_total_requests,
                workload.prompt_lens,
                workload.output_budgets,
                metrics.ttft_ms.median if metrics.ttft_ms is not None else 0.0,
                metrics.mean_tpot_ms.median if metrics.mean_tpot_ms is not None else 0.0,
            )
            if cost and peaks
            else None
        )

        return cls(
            key=case_key,
            workload=workload,
            metrics=metrics,
            memory=case_memory,
            repeats=len(runs),
            warmup=warmup,
            packing=packing,
            efficiency=eff,
            op_breakdown=op_breakdown,
            scale=scale,
            seed=seed,
        )


__all__ = [
    "LockstepCaseResult",
]
