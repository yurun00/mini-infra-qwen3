"""引擎层执行结果与调度领域模型（GenerationResult、LockstepResult、ScheduleResult、ScheduleEvent、ScheduleMetrics）。"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

import torch

from mini_infra.core.request import RequestState
from mini_infra.core.types import (
    BatchingMethod,
    BatchScheduleInfo,
    GenerationTrace,
    MemoryProfile,
    RequestMetrics,
    SequenceOutput,
    StepAggregatedMetrics,
    build_memory_profile,
    collapse_uniform,
    compute_step_aggregated_metrics,
    pad_output_tokens,
)
from mini_infra.engine.mode import FeatureFlag, Mode
from mini_infra.profiling import OpLatencyBreakdown
from mini_infra.utils import percentile

_MetricT = TypeVar("_MetricT")


@dataclass(frozen=True)
class WorkloadSpec:
    """评测用例的负载规格与运行参数（Lockstep 与 Scheduler 共享，相等比较忽略 repeats 与 warmup）。"""

    mode: Mode
    batch_size: int
    prompt_len: int | tuple[int, ...]
    max_new_tokens: int | tuple[int, ...]
    repeats: int = field(default=1, compare=False)
    warmup: int = field(default=0, compare=False)
    max_running_requests: int | None = None
    arrival_times_ms: float | tuple[float, ...] | None = None


class SchedulePhase(str, Enum):
    """一次调度事件所处阶段。"""

    PREFILL = "prefill"
    DECODE = "decode"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ScheduleEvent:
    """一次模型调用（Prefill 或 Decode）及其搬移开销的调度事件记录。"""

    phase: SchedulePhase
    admitted_request_ids: tuple[int, ...]  # 本事件入场的 request_index
    finished_request_ids: tuple[int, ...]  # 本事件结束的 request_index
    physical_batch_size: int  # 模型实际执行的行数（含空槽位/已结束槽位）
    num_active_requests: int  # 事件期间仍在占用槽位的请求数
    num_generated_tokens: int  # 本事件产出的有效 Token 数
    step_ms: float
    model_ms: float
    cache_management_ms: float
    live_kv_cache_bytes: int


@dataclass(frozen=True)
class ScheduleMetrics(Generic[_MetricT]):
    """一次或多次调度运行的调度器专属聚合指标（支持单次标量 float 或多轮评测统计分布 DistributionStats）。"""

    request_count: int
    max_running_requests: int
    makespan_ms: _MetricT
    busy_ms: _MetricT
    cache_management_ms: _MetricT
    cache_management_fraction: _MetricT
    queue_ms_p50: _MetricT
    queue_ms_p95: _MetricT
    ttft_ms_p50: _MetricT
    ttft_ms_p95: _MetricT
    end_to_end_ms_p50: _MetricT
    end_to_end_ms_p95: _MetricT
    itl_ms_p50: _MetricT
    itl_ms_p95: _MetricT
    itl_ms_max: _MetricT


@dataclass(frozen=True)
class GenerationResult:
    """所有文本生成模式的公共结果基类契约（彻底属性化，不可变强类型容器）。"""

    sequences: list[SequenceOutput]
    metrics: StepAggregatedMetrics[float]
    memory: MemoryProfile
    trace: GenerationTrace
    batch_size: int
    op_breakdown: OpLatencyBreakdown | None = None
    batch_info: BatchScheduleInfo[float] | None = None

    def padded_output_tokens(
        self,
        request_idx: int | Sequence[int] | None = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """从 sequences 动态计算并返回对齐输出张量。"""
        return pad_output_tokens(self.sequences, request_idx=request_idx, pad_token_id=pad_token_id)


@dataclass(frozen=True)
class LockstepResult(GenerationResult):
    """齐步（Lockstep）执行循环结果：继承 GenerationResult 公共契约。"""


@dataclass(frozen=True)
class ScheduleResult(GenerationResult):
    """批调度运行的顶层不可变结果容器（继承公共 GenerationResult 并扩展非空调度专属字段）。"""

    schedule_metrics: ScheduleMetrics[float] = field(kw_only=True)
    events: list[ScheduleEvent] = field(kw_only=True)
    request_metrics: list[RequestMetrics] = field(kw_only=True)
    batching_method: BatchingMethod = BatchingMethod.CONTINUOUS

    @property
    def step_agg_metrics(self) -> StepAggregatedMetrics[float]:
        """调度器视角下的步级聚合指标别名（与基类 metrics 指向同一对象，用以同 schedule_metrics 区分）。"""
        return self.metrics

    @property
    def workload(self) -> WorkloadSpec:
        """从单次调度结果提取负载规格 WorkloadSpec（相等比较时忽略 repeats 与 warmup）。"""
        prompt_lens = [s.prompt_len for s in self.sequences]
        output_budgets = [r.num_output_tokens for r in self.request_metrics]
        arrivals = [r.arrival_ms for r in self.request_metrics]
        return WorkloadSpec(
            mode=Mode({FeatureFlag.KV_CACHE}, batching_method=self.batching_method),
            batch_size=self.schedule_metrics.request_count,
            prompt_len=collapse_uniform(prompt_lens),
            max_new_tokens=collapse_uniform(output_budgets),
            max_running_requests=self.schedule_metrics.max_running_requests,
            arrival_times_ms=collapse_uniform(arrivals),
        )


def summarize_schedule(
    batching_method: BatchingMethod,
    states: Sequence[RequestState],
    events: Sequence[ScheduleEvent],
    max_running_requests: int,
    peak_allocated_mb: float,
    peak_reserved_mb: float,
    baseline_mb: float = 0.0,
    e2e_wall_ms: float | None = None,
    is_cuda: bool = False,
) -> ScheduleResult:
    """把运行结束后的请求状态与事件流汇总为 ScheduleResult（固定批次与 Continuous 共用）。"""
    requests = [state.to_request_metrics() for state in states]
    completion_step_by_req_idx = {
        req_idx: step_idx + 1 for step_idx, e in enumerate(events) for req_idx in e.finished_request_ids
    }
    sequences = [
        state.to_sequence_output(completion_step=completion_step_by_req_idx.get(state.spec.request_index, len(events)))
        for state in states
    ]

    makespan_ms = max(r.completion_ms for r in requests) - min(r.arrival_ms for r in requests)
    busy_ms = sum(e.step_ms for e in events)
    model_ms = sum(e.model_ms for e in events)
    cache_ms = sum(e.cache_management_ms for e in events)
    prefill_ms = sum(e.step_ms for e in events if e.phase is SchedulePhase.PREFILL)
    e2e_wall_ms = e2e_wall_ms if e2e_wall_ms is not None else busy_ms
    num_output_tokens = sum(r.num_output_tokens for r in requests)
    total_executed_slots = sum(e.physical_batch_size for e in events)
    capacity_area = sum(e.num_active_requests * e.step_ms for e in events)
    itls = [itl for r in requests for itl in r.inter_token_latencies_ms]
    mib = 1024**2

    execution_slot_util = num_output_tokens / total_executed_slots if total_executed_slots else 0.0
    running_capacity_util = capacity_area / (max_running_requests * busy_ms) if busy_ms else 0.0
    peak_live_kv_cache_mb = max((e.live_kv_cache_bytes for e in events), default=0) / mib

    prompt_len_by_idx = {state.spec.request_index: state.spec.prompt_len for state in states}
    num_prompt_tokens = sum(state.spec.prompt_len for state in states)
    num_padded_prompt_tokens = sum(
        len(e.admitted_request_ids) * max(prompt_len_by_idx[i] for i in e.admitted_request_ids)
        for e in events
        if e.phase is SchedulePhase.PREFILL and e.admitted_request_ids
    )
    if num_padded_prompt_tokens == 0:
        num_padded_prompt_tokens = num_prompt_tokens
    padding_efficiency = num_prompt_tokens / num_padded_prompt_tokens if num_padded_prompt_tokens else 0.0

    # 宏观步级指标：
    # - e2e_wall_ms 使用真实墙钟耗时（含 Python 队列调度、Cache 搬移与模型前向，不含虚拟跳钟空档）
    # - throughput 使用引擎非空闲活跃期耗时 busy_ms 为分母，避免稀疏到达空档期扭曲引擎吞吐
    # - gpu_time_ratio 反映纯模型前向耗时占真实墙钟总耗时的比例
    gpu_time_ratio = (min(1.0, model_ms / e2e_wall_ms) if e2e_wall_ms > 0 else 0.0) if is_cuda else None
    step_agg_metrics: StepAggregatedMetrics[float] = StepAggregatedMetrics(
        e2e_wall_ms=e2e_wall_ms,
        e2e_model_ms=model_ms,
        throughput_tok_s=num_output_tokens * 1000 / busy_ms if busy_ms else 0.0,
        throughput_req_s=len(requests) * 1000 / busy_ms if busy_ms else 0.0,
        gpu_time_ratio=gpu_time_ratio,
        effective_prompt_tok_s=num_prompt_tokens * 1000 / prefill_ms if prefill_ms else 0.0,
        padded_prompt_tok_s=num_padded_prompt_tokens * 1000 / prefill_ms if prefill_ms else 0.0,
        ttft_ms=None,
        mean_tpot_ms=None,
        p50_tpot_ms=None,
        p95_tpot_ms=None,
    )
    memory = MemoryProfile(
        baseline_mb=baseline_mb,
        prefill_mb=None,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
        kv_cache_mb=peak_live_kv_cache_mb,
    )
    batch_info: BatchScheduleInfo[float] = BatchScheduleInfo(
        padding_efficiency=padding_efficiency,
        slot_utilization=execution_slot_util,
        running_capacity_utilization=running_capacity_util,
    )
    schedule_metrics: ScheduleMetrics[float] = ScheduleMetrics(
        request_count=len(requests),
        max_running_requests=max_running_requests,
        makespan_ms=makespan_ms,
        busy_ms=busy_ms,
        cache_management_ms=cache_ms,
        cache_management_fraction=cache_ms / busy_ms if busy_ms else 0.0,
        queue_ms_p50=percentile([r.queue_ms for r in requests], 50),
        queue_ms_p95=percentile([r.queue_ms for r in requests], 95),
        ttft_ms_p50=percentile([r.ttft_ms for r in requests], 50),
        ttft_ms_p95=percentile([r.ttft_ms for r in requests], 95),
        end_to_end_ms_p50=percentile([r.end_to_end_ms for r in requests], 50),
        end_to_end_ms_p95=percentile([r.end_to_end_ms for r in requests], 95),
        itl_ms_p50=percentile(itls, 50),
        itl_ms_p95=percentile(itls, 95),
        itl_ms_max=max(itls, default=0.0),
    )
    trace = GenerationTrace(
        step_latencies_ms=[event.step_ms for event in events],
        num_active_tokens_per_step=[event.num_generated_tokens for event in events],
        logits_trace=None,
    )
    return ScheduleResult(
        sequences=sequences,
        metrics=step_agg_metrics,
        memory=memory,
        trace=trace,
        batch_size=len(requests),
        batch_info=batch_info,
        schedule_metrics=schedule_metrics,
        events=list(events),
        request_metrics=requests,
        batching_method=batching_method,
    )


__all__ = [
    "BatchScheduleInfo",
    "BatchingMethod",
    "GenerationResult",
    "GenerationTrace",
    "LockstepResult",
    "MemoryProfile",
    "RequestMetrics",
    "ScheduleEvent",
    "ScheduleMetrics",
    "SchedulePhase",
    "ScheduleResult",
    "SequenceOutput",
    "StepAggregatedMetrics",
    "WorkloadSpec",
    "build_memory_profile",
    "compute_step_aggregated_metrics",
    "pad_output_tokens",
    "percentile",
    "summarize_schedule",
]
