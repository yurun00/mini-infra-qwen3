"""评测结果层内部共享指标聚合辅助（供 LockstepCaseResult 与 SchedulerCaseResult 复用）。"""

from collections.abc import Sequence

from mini_infra.bench.stats import DistributionStats
from mini_infra.engine.metrics import ExecutionMetrics


def build_execution_metrics(
    samples: Sequence[ExecutionMetrics[float]],
) -> ExecutionMetrics[DistributionStats]:
    """将多轮测量的 ExecutionMetrics[float] 样本汇总为 DistributionStats 分布。"""
    gpu_ratios = [m.gpu_time_ratio for m in samples]
    gpu_ratio_stat = (
        DistributionStats.from_samples([v for v in gpu_ratios if v is not None])
        if all(v is not None for v in gpu_ratios)
        else None
    )
    return ExecutionMetrics(
        e2e_wall_ms=DistributionStats.from_samples([m.e2e_wall_ms for m in samples]),
        e2e_model_ms=DistributionStats.from_samples([m.e2e_model_ms for m in samples]),
        throughput_tok_s=DistributionStats.from_samples(
            [m.throughput_tok_s for m in samples]
        ),
        throughput_req_s=DistributionStats.from_samples(
            [m.throughput_req_s for m in samples]
        ),
        effective_prompt_tok_s=DistributionStats.from_samples(
            [m.effective_prompt_tok_s for m in samples]
        ),
        padded_prompt_tok_s=DistributionStats.from_samples(
            [m.padded_prompt_tok_s for m in samples]
        ),
        gpu_time_ratio=gpu_ratio_stat,
    )


__all__ = ["build_execution_metrics"]
