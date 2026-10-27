"""调度与请求核心领域强类型结构体与指标汇总。"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from mini_infra.core.request import RequestState
from mini_infra.utils import percentile


class BatchingMethod(str, Enum):
    """互斥的批处理策略。"""

    NONE = "none"
    FIXED = "fixed-batching"
    CONTINUOUS = "continuous-batching"

    @classmethod
    def from_str(cls, value: "str | BatchingMethod | None") -> "BatchingMethod":
        if value is None:
            return cls.NONE
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"未知批处理策略: {value}，可选策略: {[m.value for m in cls]}") from err

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SequenceOutput:
    """单个请求的生成序列输出。"""

    request_index: int
    token_ids: list[int]
    prompt_len: int
    output_len: int
    finish_reason: str  # 结束原因: "stop"（命中 EOS）或 "max_tokens"（达到 Token 预算上限）


@dataclass(frozen=True)
class BatchScheduleInfo:
    """批处理调度效率与完成状态明细。"""

    padding_efficiency: float
    slot_utilization: float
    prompt_tokens: int
    padded_prompt_tokens: int
    prompt_lengths_per_req: list[int]
    completion_steps_per_req: list[int]
    ttft_ms: float = 0.0

    @property
    def effective_prompt_tok_s(self) -> float:
        """Prefill 阶段有效 Prompt 吞吐率 (tokens/s)。"""
        return self.prompt_tokens * 1000.0 / self.ttft_ms if self.ttft_ms > 0 else 0.0

    @property
    def padded_prompt_tok_s(self) -> float:
        """Prefill 阶段包含 Padding 的物理计算吞吐率 (tokens/s)。"""
        return self.padded_prompt_tokens * 1000.0 / self.ttft_ms if self.ttft_ms > 0 else 0.0


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
    active_requests: int  # 事件期间仍在占用槽位的请求数
    generated_tokens: int  # 本事件产出的有效 Token 数
    total_ms: float
    model_ms: float
    cache_management_ms: float
    live_cache_bytes: int


@dataclass(frozen=True)
class RequestMetrics:
    """单个请求在逻辑时间线上的关键时间戳（单位 ms）。"""

    request_index: int
    arrival_ms: float
    admitted_ms: float
    first_token_ms: float
    completion_ms: float
    output_tokens: int
    token_times_ms: tuple[float, ...]

    @property
    def queue_ms(self) -> float:
        return self.admitted_ms - self.arrival_ms

    @property
    def ttft_ms(self) -> float:
        return self.first_token_ms - self.arrival_ms

    @property
    def end_to_end_ms(self) -> float:
        return self.completion_ms - self.arrival_ms

    @property
    def inter_token_latencies_ms(self) -> list[float]:
        return [right - left for left, right in zip(self.token_times_ms, self.token_times_ms[1:])]


@dataclass(frozen=True)
class ScheduleMetrics:
    """一次调度运行的聚合指标。"""

    request_count: int
    max_running_requests: int
    makespan_ms: float
    busy_ms: float
    model_ms: float
    cache_management_ms: float
    cache_management_fraction: float
    request_throughput_req_s: float
    output_token_throughput_tok_s: float
    execution_slot_utilization: float  # 有效输出 Token 数 / 模型实际执行的行数
    running_capacity_utilization: float  # 按时间加权的并发槽位占用率
    queue_ms_p50: float
    queue_ms_p95: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    end_to_end_ms_p50: float
    end_to_end_ms_p95: float
    itl_ms_p50: float
    itl_ms_p95: float
    itl_ms_max: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    peak_live_cache_mb: float


@dataclass(frozen=True)
class ScheduleResult:
    """批调度运行的顶层不可变结果容器。"""

    batching_method: BatchingMethod
    sequences: list[SequenceOutput]
    requests: list[RequestMetrics]
    events: list[ScheduleEvent]
    metrics: ScheduleMetrics


def _require(value: float | None, name: str, request_index: int) -> float:
    if value is None:
        raise RuntimeError(f"请求 {request_index} 未完成调度，缺少 {name}")
    return value


def summarize_schedule(
    batching_method: BatchingMethod,
    states: Sequence[RequestState],
    events: Sequence[ScheduleEvent],
    max_running_requests: int,
    peak_allocated_mb: float,
    peak_reserved_mb: float,
) -> ScheduleResult:
    """把运行结束后的请求状态与事件流汇总为 ScheduleResult（固定批次与 Continuous 共用）。"""
    requests = [
        RequestMetrics(
            request_index=state.spec.request_index,
            arrival_ms=state.spec.arrival_ms,
            admitted_ms=_require(state.admitted_ms, "admitted_ms", state.spec.request_index),
            first_token_ms=_require(state.first_token_ms, "first_token_ms", state.spec.request_index),
            completion_ms=_require(state.completion_ms, "completion_ms", state.spec.request_index),
            output_tokens=len(state.generated),
            token_times_ms=tuple(state.token_times_ms),
        )
        for state in states
    ]
    sequences = [
        SequenceOutput(
            request_index=state.spec.request_index,
            token_ids=list(state.generated),
            prompt_len=state.spec.prompt_len,
            output_len=len(state.generated),
            finish_reason=state.finish_reason,
        )
        for state in states
    ]

    makespan_ms = max(r.completion_ms for r in requests) - min(r.arrival_ms for r in requests)
    busy_ms = sum(e.total_ms for e in events)
    model_ms = sum(e.model_ms for e in events)
    cache_ms = sum(e.cache_management_ms for e in events)
    output_tokens = sum(r.output_tokens for r in requests)
    executed_rows = sum(e.executed_batch_size for e in events)
    capacity_area = sum(e.active_requests * e.total_ms for e in events)
    itls = [itl for r in requests for itl in r.inter_token_latencies_ms]
    mib = 1024**2

    metrics = ScheduleMetrics(
        request_count=len(requests),
        max_running_requests=max_running_requests,
        makespan_ms=makespan_ms,
        busy_ms=busy_ms,
        model_ms=model_ms,
        cache_management_ms=cache_ms,
        cache_management_fraction=cache_ms / busy_ms if busy_ms else 0.0,
        request_throughput_req_s=len(requests) * 1000 / makespan_ms if makespan_ms else 0.0,
        output_token_throughput_tok_s=output_tokens * 1000 / makespan_ms if makespan_ms else 0.0,
        execution_slot_utilization=output_tokens / executed_rows if executed_rows else 0.0,
        running_capacity_utilization=capacity_area / (max_running_requests * busy_ms) if busy_ms else 0.0,
        queue_ms_p50=percentile([r.queue_ms for r in requests], 50),
        queue_ms_p95=percentile([r.queue_ms for r in requests], 95),
        ttft_ms_p50=percentile([r.ttft_ms for r in requests], 50),
        ttft_ms_p95=percentile([r.ttft_ms for r in requests], 95),
        end_to_end_ms_p50=percentile([r.end_to_end_ms for r in requests], 50),
        end_to_end_ms_p95=percentile([r.end_to_end_ms for r in requests], 95),
        itl_ms_p50=percentile(itls, 50),
        itl_ms_p95=percentile(itls, 95),
        itl_ms_max=max(itls, default=0.0),
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
        peak_live_cache_mb=max((e.live_cache_bytes for e in events), default=0) / mib,
    )
    return ScheduleResult(
        batching_method=batching_method,
        sequences=sequences,
        requests=requests,
        events=list(events),
        metrics=metrics,
    )
