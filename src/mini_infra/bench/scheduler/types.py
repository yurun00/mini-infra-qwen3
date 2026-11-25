"""Continuous Batching 与固定批次 Scheduler 评测结果领域类型。"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from mini_infra.bench.cases import (
    CaseKey,
    WorkloadProfile,
)
from mini_infra.bench.efficiency import EfficiencyMetrics
from mini_infra.bench.stats import DistributionStats
from mini_infra.core import collapse_uniform
from mini_infra.engine.metrics import (
    BatchScheduleInfo,
    MemoryProfile,
    RequestAggregatedMetrics,
    ScheduleEvent,
    ScheduleMetrics,
    StepAggregatedMetrics,
    compute_request_aggregated_metrics,
    summarize_repeated_kv_cache_profiles,
)
from mini_infra.engine.results import ScheduleResult


def _aggregate_request_metrics(
    samples: Sequence[RequestAggregatedMetrics[float]],
) -> RequestAggregatedMetrics[DistributionStats]:
    """将多轮运行的 RequestAggregatedMetrics[float] 样本聚合为 DistributionStats（全量与具名组共用）。"""
    return RequestAggregatedMetrics(
        request_count=samples[0].request_count,
        queue_ms_p50=DistributionStats.from_samples([s.queue_ms_p50 for s in samples]),
        queue_ms_p95=DistributionStats.from_samples([s.queue_ms_p95 for s in samples]),
        prefill_ms_p50=DistributionStats.from_samples(
            [s.prefill_ms_p50 for s in samples]
        ),
        prefill_ms_p95=DistributionStats.from_samples(
            [s.prefill_ms_p95 for s in samples]
        ),
        ttft_ms_p50=DistributionStats.from_samples([s.ttft_ms_p50 for s in samples]),
        ttft_ms_p95=DistributionStats.from_samples([s.ttft_ms_p95 for s in samples]),
        end_to_end_ms_p50=DistributionStats.from_samples(
            [s.end_to_end_ms_p50 for s in samples]
        ),
        end_to_end_ms_p95=DistributionStats.from_samples(
            [s.end_to_end_ms_p95 for s in samples]
        ),
        itl_ms_p50=DistributionStats.from_samples([s.itl_ms_p50 for s in samples]),
        itl_ms_p95=DistributionStats.from_samples([s.itl_ms_p95 for s in samples]),
        itl_ms_max=DistributionStats.from_samples([s.itl_ms_max for s in samples]),
        num_computed_prompt_tokens=DistributionStats.from_samples(
            [s.num_computed_prompt_tokens for s in samples]
        ),
    )


@dataclass(frozen=True)
class SchedulerCaseResult:
    """单个 Scheduler 用例在多轮 repeat 测量下的汇总结果。

    以 CaseKey 作为全局唯一身份键，不再保留非正交的 case_id。
    """

    key: CaseKey
    workload: WorkloadProfile
    metrics: StepAggregatedMetrics[DistributionStats]
    schedule_metrics: ScheduleMetrics[DistributionStats]
    memory: MemoryProfile
    repeats: int = 1
    warmup: int = 0
    packing: BatchScheduleInfo[DistributionStats] | None = None
    efficiency: EfficiencyMetrics | None = None
    repeat_samples: list[dict[str, Any]] = field(default_factory=list)
    sample_events: list[ScheduleEvent] = field(default_factory=list)
    group_metrics: dict[str, RequestAggregatedMetrics[DistributionStats]] = field(
        default_factory=dict
    )
    scale: str | None = None
    seed: int | None = None

    def format_summary(self) -> str:
        """格式化单用例的延迟/吞吐/slot_utilization 摘要行（与 lockstep 对齐）。"""
        cache_frac = self.schedule_metrics.cache_management_fraction.mean
        cache_info = f" | cache {cache_frac * 100:.1f}%" if cache_frac > 0 else ""
        slot_util = (
            self.packing.slot_utilization.mean if self.packing is not None else 0.0
        )
        pool_info = (
            f" | pool {self.memory.kv_cache.peak_kv_cache_pool_mb:.1f} MB"
            if self.memory.kv_cache.peak_kv_cache_pool_mb > 0
            else ""
        )
        return (
            f"  {self.key.abbreviated_display_name:<32}: makespan {self.schedule_metrics.makespan_ms.mean:.2f} ms | "
            f"{self.metrics.throughput_tok_s.mean:.1f} tok/s | "
            f"{self.metrics.throughput_req_s.mean:.2f} req/s | "
            f"slot_util {slot_util * 100:.1f}% | "
            f"ttft_p95 {self.schedule_metrics.ttft_ms_p95.mean:.2f} ms | "
            f"itl_p95 {self.schedule_metrics.itl_ms_p95.mean:.2f} ms"
            f"{cache_info}{pool_info}{self._format_token_budget_info()}"
            f"{self._format_prefix_caching_info()}{self._format_group_info()}"
        )

    def _format_prefix_caching_info(self) -> str:
        """开启 prefix caching 的用例追加实际计算 Prompt Token 数与命中率。"""
        paged = self.memory.kv_cache.paged_cache
        if not self.key.prefix_caching or paged is None:
            return ""
        sm = self.schedule_metrics
        return (
            f" | computed_prompt {sm.num_computed_prompt_tokens.mean:.0f}"
            f" | prefix_hit {paged.prefix_cache_hit_rate * 100:.1f}%"
        )

    def _format_group_info(self) -> str:
        """含具名请求组时追加各组 prefill_ms_p50 摘要。"""
        if not self.group_metrics:
            return ""
        return "".join(
            f" | {name}_pf_p50 {gm.prefill_ms_p50.mean:.2f} ms"
            for name, gm in self.group_metrics.items()
        )

    def _format_token_budget_info(self) -> str:
        """带 token_budget 的用例追加预算、最长连续 Prefill 步数、oversize prefill 次数与最长 Prefill 步耗时。"""
        if self.key.token_budget is None:
            return ""
        sm = self.schedule_metrics
        return (
            f" | budget {self.key.token_budget}"
            f" | max_pf_streak {sm.max_consecutive_prefill_steps.median:.0f}"
            f" | oversize {sm.num_oversize_prefill_steps.median:.0f}"
            f" | max_pf_step {sm.max_prefill_step_ms.median:.2f} ms"
        )

    def to_record_dict(self) -> dict[str, Any]:
        """导出为与 BaseCaseRecord / SchedulerCaseRecord 完全对齐的标准字典结构。"""
        pl = collapse_uniform(self.workload.prompt_lens)
        mn = collapse_uniform(self.workload.output_budgets)
        rec: dict[str, Any] = {
            "workload_id": self.key.workload_id,
            "engine_config": str(self.key.engine_config),
            "num_total_requests": self.workload.num_total_requests,
            "prompt_len": list(pl) if isinstance(pl, tuple) else pl,
            "max_new_tokens": list(mn) if isinstance(mn, tuple) else mn,
            "repeats": self.repeats,
            "warmup": self.warmup,
            "max_running_requests": self.key.max_running_requests,
            "arrival_times_ms": list(self.workload.arrival_ms),
            "e2e_wall_ms": self.metrics.e2e_wall_ms.to_dict(),
            "e2e_model_ms": self.metrics.e2e_model_ms.to_dict(),
            "throughput_tok_s": self.metrics.throughput_tok_s.to_dict(),
            "throughput_req_s": self.metrics.throughput_req_s.to_dict(),
            "gpu_time_ratio": (
                self.metrics.gpu_time_ratio.to_dict()
                if self.metrics.gpu_time_ratio is not None
                else None
            ),
            "effective_prompt_tok_s": (
                self.metrics.effective_prompt_tok_s.to_dict()
                if self.metrics.effective_prompt_tok_s is not None
                else None
            ),
            "padded_prompt_tok_s": (
                self.metrics.padded_prompt_tok_s.to_dict()
                if self.metrics.padded_prompt_tok_s is not None
                else None
            ),
            "memory": self.memory.to_dict(),
            "schedule": self.schedule_metrics.to_dict(),
            "repeat_samples": self.repeat_samples,
            "sample_event_trace": [
                {**asdict(e), "phase": str(e.phase)} for e in self.sample_events
            ],
        }
        if self.key.block_size is not None:
            rec["paged_cache_config"] = {
                "block_size": self.key.block_size,
                "num_blocks": self.key.num_blocks,
            }
            if self.key.prefix_caching:
                rec["paged_cache_config"]["prefix_caching"] = True
        if self.group_metrics:
            rec["group_metrics"] = {
                name: gm.to_dict() for name, gm in self.group_metrics.items()
            }
        if self.scale is not None:
            rec["scale"] = self.scale
        if self.seed is not None:
            rec["seed"] = self.seed
        if self.packing is not None:
            rec["packing"] = self.packing.to_dict()
        if self.efficiency is not None:
            rec["efficiency"] = self.efficiency.to_dict()
        return rec

    @classmethod
    def from_runs(
        cls,
        case_key: CaseKey,
        runs: Sequence[ScheduleResult],
        *,
        workload: WorkloadProfile | None = None,
        warmup: int = 0,
        scale: str | None = None,
        seed: int | None = None,
    ) -> "SchedulerCaseResult":
        """把具有相同 CaseKey 与负载规格的多轮运行聚合为一个 SchedulerCaseResult。"""
        if not runs:
            raise ValueError("runs 不能为空")

        first_run = runs[0]
        extracted_first = extract_workload_profile(
            first_run, workload_id=case_key.workload_id
        )
        if any(
            extract_workload_profile(r, workload_id=case_key.workload_id)
            != extracted_first
            for r in runs
        ):
            raise ValueError(
                "SchedulerCaseResult.from_runs 中的所有 run 必须具有相同的负载规格"
            )
        if workload is not None:
            if (
                workload.workload_id,
                workload.num_total_requests,
                workload.prompt_lens,
                workload.output_budgets,
                workload.arrival_ms,
            ) != (
                extracted_first.workload_id,
                extracted_first.num_total_requests,
                extracted_first.prompt_lens,
                extracted_first.output_budgets,
                extracted_first.arrival_ms,
            ):
                raise ValueError(
                    "SchedulerCaseResult.from_runs 传入的 workload 与实测请求规格不一致"
                )
        else:
            workload = extracted_first

        repeat_samples = [
            {
                "makespan_ms": r.schedule_metrics.makespan_ms,
                "throughput_req_s": r.metrics.throughput_req_s,
                "throughput_tok_s": r.metrics.throughput_tok_s,
                "slot_utilization": (
                    r.batch_info.slot_utilization if r.batch_info is not None else 0.0
                ),
                "ttft_ms_p95": r.schedule_metrics.ttft_ms_p95,
                "itl_ms_p95": r.schedule_metrics.itl_ms_p95,
                "cache_management_fraction": r.schedule_metrics.cache_management_fraction,
                "peak_allocated_mb": r.memory.peak_allocated_mb,
            }
            for r in runs
        ]

        worst_mem = max(runs, key=lambda r: r.memory.peak_allocated_mb).memory
        case_memory = replace(
            worst_mem,
            peak_allocated_mb=max(r.memory.peak_allocated_mb for r in runs),
            peak_reserved_mb=max(r.memory.peak_reserved_mb for r in runs),
            kv_cache=summarize_repeated_kv_cache_profiles(
                [r.memory.kv_cache for r in runs]
            ),
        )

        gpu_ratios = [r.metrics.gpu_time_ratio for r in runs]
        gpu_ratio_stat = (
            DistributionStats.from_samples([v for v in gpu_ratios if v is not None])
            if all(v is not None for v in gpu_ratios)
            else None
        )
        eff_prompt_vals = [
            r.metrics.effective_prompt_tok_s
            for r in runs
            if r.metrics.effective_prompt_tok_s is not None
        ]
        pad_prompt_vals = [
            r.metrics.padded_prompt_tok_s
            for r in runs
            if r.metrics.padded_prompt_tok_s is not None
        ]

        step_agg_dist: StepAggregatedMetrics[DistributionStats] = StepAggregatedMetrics(
            e2e_wall_ms=DistributionStats.from_samples(
                [r.metrics.e2e_wall_ms for r in runs]
            ),
            e2e_model_ms=DistributionStats.from_samples(
                [r.metrics.e2e_model_ms for r in runs]
            ),
            throughput_tok_s=DistributionStats.from_samples(
                [r.metrics.throughput_tok_s for r in runs]
            ),
            throughput_req_s=DistributionStats.from_samples(
                [r.metrics.throughput_req_s for r in runs]
            ),
            gpu_time_ratio=gpu_ratio_stat,
            effective_prompt_tok_s=DistributionStats.from_samples(eff_prompt_vals)
            if eff_prompt_vals
            else None,
            padded_prompt_tok_s=DistributionStats.from_samples(pad_prompt_vals)
            if pad_prompt_vals
            else None,
            ttft_ms=None,
            mean_tpot_ms=None,
            p50_tpot_ms=None,
            p95_tpot_ms=None,
        )

        req_agg_dist = _aggregate_request_metrics([r.schedule_metrics for r in runs])
        req_agg_kwargs = {
            f.name: getattr(req_agg_dist, f.name)
            for f in fields(RequestAggregatedMetrics)
        }
        sched_dist: ScheduleMetrics[DistributionStats] = ScheduleMetrics(
            **req_agg_kwargs,
            max_running_requests=first_run.schedule_metrics.max_running_requests,
            makespan_ms=DistributionStats.from_samples(
                [r.schedule_metrics.makespan_ms for r in runs]
            ),
            busy_ms=DistributionStats.from_samples(
                [r.schedule_metrics.busy_ms for r in runs]
            ),
            cache_management_ms=DistributionStats.from_samples(
                [r.schedule_metrics.cache_management_ms for r in runs]
            ),
            cache_management_fraction=DistributionStats.from_samples(
                [r.schedule_metrics.cache_management_fraction for r in runs]
            ),
            scheduler_ms=DistributionStats.from_samples(
                [r.schedule_metrics.scheduler_ms for r in runs]
            ),
            num_prefill_steps=DistributionStats.from_samples(
                [r.schedule_metrics.num_prefill_steps for r in runs]
            ),
            num_decode_steps=DistributionStats.from_samples(
                [r.schedule_metrics.num_decode_steps for r in runs]
            ),
            max_consecutive_prefill_steps=DistributionStats.from_samples(
                [r.schedule_metrics.max_consecutive_prefill_steps for r in runs]
            ),
            num_oversize_prefill_steps=DistributionStats.from_samples(
                [r.schedule_metrics.num_oversize_prefill_steps for r in runs]
            ),
            num_prefill_chunks=DistributionStats.from_samples(
                [r.schedule_metrics.num_prefill_chunks for r in runs]
            ),
            max_prefill_step_tokens=DistributionStats.from_samples(
                [r.schedule_metrics.max_prefill_step_tokens for r in runs]
            ),
            max_prefill_step_ms=DistributionStats.from_samples(
                [r.schedule_metrics.max_prefill_step_ms for r in runs]
            ),
        )

        batch_infos = [r.batch_info for r in runs if r.batch_info is not None]
        packing: BatchScheduleInfo[DistributionStats] | None = None
        if batch_infos:
            cap_vals = [
                b.running_capacity_utilization
                for b in batch_infos
                if b.running_capacity_utilization is not None
            ]
            packing = BatchScheduleInfo(
                padding_efficiency=DistributionStats.from_samples(
                    [b.padding_efficiency for b in batch_infos]
                ),
                slot_utilization=DistributionStats.from_samples(
                    [b.slot_utilization for b in batch_infos]
                ),
                running_capacity_utilization=DistributionStats.from_samples(cap_vals)
                if cap_vals
                else None,
            )

        group_metrics = {
            g.name: _aggregate_request_metrics(
                [
                    compute_request_aggregated_metrics(
                        r.request_metrics[g.start : g.end],
                        r.sequences[g.start : g.end],
                    )
                    for r in runs
                ]
            )
            for g in workload.groups
        }

        return cls(
            key=case_key,
            workload=workload,
            metrics=step_agg_dist,
            schedule_metrics=sched_dist,
            memory=case_memory,
            repeats=len(runs),
            warmup=warmup,
            packing=packing,
            repeat_samples=repeat_samples,
            sample_events=list(runs[0].events),
            group_metrics=group_metrics,
            scale=scale,
            seed=seed,
        )


def extract_workload_profile(
    res: ScheduleResult, workload_id: str | None = None
) -> WorkloadProfile:
    """从单次调度结果提取纯请求负载画像 WorkloadProfile（纯 CPU）。"""
    prompt_lengths = tuple(s.prompt_len for s in res.sequences)
    output_budgets = tuple(r.num_output_tokens for r in res.request_metrics)
    arrivals = tuple(r.arrival_ms for r in res.request_metrics)
    return WorkloadProfile(
        workload_id=workload_id or "unknown",
        num_total_requests=len(prompt_lengths),
        prompt_lens=prompt_lengths,
        output_budgets=output_budgets,
        arrival_ms=arrivals,
    )


__all__ = [
    "SchedulerCaseResult",
    "extract_workload_profile",
]
