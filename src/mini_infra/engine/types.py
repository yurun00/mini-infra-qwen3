"""引擎层执行结果与调度领域模型（LockstepResult、ScheduleResult、ScheduleEvent、ScheduleMetrics）。"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch

from mini_infra.core.request import RequestState
from mini_infra.core.types import (
    BatchingMethod,
    BatchScheduleInfo,
    MemoryProfile,
    RequestMetrics,
    SequenceOutput,
    StepAggregatedMetrics,
    build_memory_profile,
    compute_step_aggregated_metrics,
    pad_output_tokens,
)
from mini_infra.utils import percentile


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
    admitted: tuple[int, ...]  # 本事件入场的 request_index
    completed: tuple[int, ...]  # 本事件结束的 request_index
    executed_batch_size: int  # 模型实际执行的行数（含空槽位/已结束槽位）
    num_active_requests: int  # 事件期间仍在占用槽位的请求数
    num_generated_tokens: int  # 本事件产出的有效 Token 数
    total_ms: float
    model_ms: float
    cache_management_ms: float
    live_cache_bytes: int


@dataclass(frozen=True)
class ScheduleMetrics:
    """一次调度运行的调度器专属聚合指标（并发规模、Cache 管理开销与请求级排队/延迟分位数）。"""

    request_count: int
    max_running_requests: int
    busy_ms: float
    cache_management_ms: float
    cache_management_fraction: float
    queue_ms_p50: float
    queue_ms_p95: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    end_to_end_ms_p50: float
    end_to_end_ms_p95: float
    itl_ms_p50: float
    itl_ms_p95: float
    itl_ms_max: float


@dataclass(frozen=True)
class LockstepResult:
    """齐步（Lockstep）执行循环结果：所有请求同时入场、齐步 Prefill + Decode 的产出与测时。"""

    sequences: list[SequenceOutput]
    batch_info: BatchScheduleInfo
    step_latencies_ms: list[float]
    num_active_tokens_per_step: list[int]
    logits_trace: list[torch.Tensor] | None
    e2e_wall_ms: float
    e2e_model_ms: float
    total_tokens: int
    memory_after_prefill_bytes: int
    cache_bytes: int


@dataclass(frozen=True)
class ScheduleResult:
    """批调度运行的顶层不可变结果容器（字段顺序与 OfflineGenerationResult 对齐）。"""

    sequences: list[SequenceOutput]
    step_agg_metrics: StepAggregatedMetrics
    schedule_metrics: ScheduleMetrics
    memory: MemoryProfile
    events: list[ScheduleEvent]
    request_metrics: list[RequestMetrics]
    batch_info: BatchScheduleInfo | None = None
    batching_method: BatchingMethod = BatchingMethod.CONTINUOUS

    def padded_output_tokens(
        self,
        request_idx: int | Sequence[int] | None = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """从 sequences 动态计算并返回对齐输出张量（与 OfflineGenerationResult 接口一致）。"""
        return pad_output_tokens(self.sequences, request_idx=request_idx, pad_token_id=pad_token_id)


def summarize_schedule(
    batching_method: BatchingMethod,
    states: Sequence[RequestState],
    events: Sequence[ScheduleEvent],
    max_running_requests: int,
    peak_allocated_mb: float,
    peak_reserved_mb: float,
    baseline_mb: float = 0.0,
) -> ScheduleResult:
    """把运行结束后的请求状态与事件流汇总为 ScheduleResult（固定批次与 Continuous 共用）。"""
    requests = [state.to_request_metrics() for state in states]
    sequences = [state.to_sequence_output() for state in states]

    makespan_ms = max(r.completion_ms for r in requests) - min(r.arrival_ms for r in requests)
    busy_ms = sum(e.total_ms for e in events)
    model_ms = sum(e.model_ms for e in events)
    cache_ms = sum(e.cache_management_ms for e in events)
    num_output_tokens = sum(r.num_output_tokens for r in requests)
    executed_rows = sum(e.executed_batch_size for e in events)
    capacity_area = sum(e.num_active_requests * e.total_ms for e in events)
    itls = [itl for r in requests for itl in r.inter_token_latencies_ms]
    mib = 1024**2

    execution_slot_util = num_output_tokens / executed_rows if executed_rows else 0.0
    running_capacity_util = capacity_area / (max_running_requests * busy_ms) if busy_ms else 0.0
    peak_live_cache_mb = max((e.live_cache_bytes for e in events), default=0) / mib

    prompt_len_by_idx = {state.spec.request_index: state.spec.prompt_len for state in states}
    num_prompt_tokens = sum(state.spec.prompt_len for state in states)
    num_padded_prompt_tokens = sum(
        len(e.admitted) * max(prompt_len_by_idx[i] for i in e.admitted)
        for e in events
        if e.phase is SchedulePhase.PREFILL and e.admitted
    )
    if num_padded_prompt_tokens == 0:
        num_padded_prompt_tokens = num_prompt_tokens
    padding_efficiency = num_prompt_tokens / num_padded_prompt_tokens if num_padded_prompt_tokens else 0.0

    # 宏观步级指标：仅填写在调度器全局良好定义的吞吐与时延，未良好定义的 ttft/tpot 留空 None
    step_agg_metrics = StepAggregatedMetrics(
        e2e_wall_ms=makespan_ms,
        e2e_model_ms=model_ms,
        throughput_tok_s=num_output_tokens * 1000 / makespan_ms if makespan_ms else 0.0,
        throughput_req_s=len(requests) * 1000 / makespan_ms if makespan_ms else 0.0,
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
        kv_cache_mb=peak_live_cache_mb,
    )
    completion_step_by_idx = {
        req_idx: step_idx + 1 for step_idx, e in enumerate(events) for req_idx in e.completed
    }
    batch_info = BatchScheduleInfo(
        padding_efficiency=padding_efficiency,
        slot_utilization=execution_slot_util,
        num_prompt_tokens=num_prompt_tokens,
        num_padded_prompt_tokens=num_padded_prompt_tokens,
        prompt_lengths_per_req=[state.spec.prompt_len for state in states],
        completion_steps_per_req=[
            completion_step_by_idx.get(state.spec.request_index, len(events)) for state in states
        ],
        ttft_ms=None,
        running_capacity_utilization=running_capacity_util,
    )
    schedule_metrics = ScheduleMetrics(
        request_count=len(requests),
        max_running_requests=max_running_requests,
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
    return ScheduleResult(
        sequences=sequences,
        step_agg_metrics=step_agg_metrics,
        schedule_metrics=schedule_metrics,
        memory=memory,
        events=list(events),
        request_metrics=requests,
        batch_info=batch_info,
        batching_method=batching_method,
    )


__all__ = [
    "BatchScheduleInfo",
    "BatchingMethod",
    "LockstepResult",
    "MemoryProfile",
    "RequestMetrics",
    "ScheduleEvent",
    "ScheduleMetrics",
    "SchedulePhase",
    "ScheduleResult",
    "SequenceOutput",
    "StepAggregatedMetrics",
    "build_memory_profile",
    "compute_step_aggregated_metrics",
    "pad_output_tokens",
    "percentile",
    "summarize_schedule",
]
