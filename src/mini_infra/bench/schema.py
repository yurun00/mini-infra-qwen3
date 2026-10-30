"""Benchmark 评测结果统一 Pydantic v2 Schema 与多态信封（Lockstep & Scheduler 共享）。

核心架构设计：**三层结构 + 判别式联合（Discriminated Union）**

```
pydantic.BaseModel
 └── _DictAccessModel                      # 基础能力基类：强类型属性访问 + 只读字典下标兼容 (rec["k"])
      ├── 第一层：叶子统计与画像组件 (Leaf Value Records，与运行时领域 Dataclass 一一映射)
      │    ├── DistributionStatsRecord     <-> bench.common.DistributionStats
      │    ├── MemoryProfileRecord         <-> core.types.MemoryProfile
      │    ├── PackingInfoRecord           <-> core.types.BatchScheduleInfo[DistributionStats]
      │    ├── EfficiencyMetricsRecord     <-> bench.common.EfficiencyMetrics
      │    └── ScheduleMetricsRecord       <-> engine.types.ScheduleMetrics[DistributionStats]
      │
      ├── 第二层：单用例记录层 (Case Level)
      │    └── BaseCaseRecord              # 共享 WorkloadSpec 核心字段 + StepAggregatedMetrics 公共指标 + 叶子组件
      │         ├── LockstepCaseRecord     <-> bench.lockstep.types.LockstepCaseResult
      │         │                              (扩展 ttft_ms, tpot_ms, op_latency_breakdown)
      │         └── SchedulerCaseRecord    <-> bench.scheduler.types.SchedulerCaseResult
      │                                        (扩展 case_id, max_running_requests, arrival_times_ms, schedule 等)
      │
      └── 第三层：顶层文件信封层 (Envelope Level) + 判别式联合 (Discriminated Union)
           └── BenchmarkEnvelopeBase       # 共享顶层元数据 (schema_version, tag, git, env, config, correctness, suite)
                ├── LockstepBenchmarkRecord   # engine="lockstep",  cases: list[LockstepCaseRecord]
                └── SchedulerBenchmarkRecord  # engine="scheduler", cases: list[SchedulerCaseRecord]
```

通过 `AnyBenchmarkRecord`（`Field(discriminator="engine")`），
反序列化任意评测 JSON 时可根据 `engine` 自动分发校验为对应的强类型子类。
"""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from mini_infra.bench.common import (
    SCHEMA_VERSION,
    CorrectnessReport,
    HardwarePeakSpecs,
    ModelCostProfile,
)

if TYPE_CHECKING:
    from mini_infra.bench.lockstep.types import LockstepCaseResult
    from mini_infra.bench.scheduler.types import SchedulerCaseResult


# ============================================================================
# 基础能力基类：属性访问 + 只读字典协议兼容
# ============================================================================


class _DictAccessModel(BaseModel):
    """提供属性访问与只读字典下标兼容 (`rec['key']`, `rec.get('key')`) 的 Pydantic 基类。"""

    model_config = ConfigDict(extra="allow")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if self.__pydantic_extra__ and key in self.__pydantic_extra__:
            return self.__pydantic_extra__[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            val = self[key]
            return default if val is None and key not in self.model_fields_set else val
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if hasattr(self, key):
            return getattr(self, key) is not None or key in self.model_fields_set
        return bool(self.__pydantic_extra__ and key in self.__pydantic_extra__)

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()


# ============================================================================
# 第一层：叶子统计与画像组件（与运行时领域 Dataclass 一一对应）
# ============================================================================


class DistributionStatsRecord(_DictAccessModel):
    """测量样本统计分布 Schema（对应 `bench.common.DistributionStats`）。"""

    median: float
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    max: float = 0.0


class MemoryProfileRecord(_DictAccessModel):
    """显存监控剖析 Schema，单位 MiB（对应 `core.types.MemoryProfile.to_dict()`）。"""

    weights_mb: float | None = None
    baseline_allocated_mb: float = 0.0
    prefill_allocated_mb: float | None = None
    peak_allocated_mb: float = 0.0
    dynamic_peak_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    fragmentation_mb: float = 0.0
    kv_cache_mb: float = 0.0


class PackingInfoRecord(_DictAccessModel):
    """批处理打包（Padding 效率）与槽位利用率 Schema（对应 `core.types.BatchScheduleInfo[DistributionStats]`）。"""

    padding_efficiency: DistributionStatsRecord
    slot_utilization: DistributionStatsRecord
    running_capacity_utilization: DistributionStatsRecord | None = None


class EfficiencyMetricsRecord(_DictAccessModel):
    """算力利用率 (MFU) 与显存带宽利用率 (MBU) Schema（对应 `bench.common.EfficiencyMetrics`）。"""

    prefill_mfu: float | None = None
    decode_mfu: float | None = None
    decode_mbu: float | None = None


class ScheduleMetricsRecord(_DictAccessModel):
    """Scheduler 专属队列与步进调度统计 Schema（对应 `engine.types.ScheduleMetrics[DistributionStats]`）。"""

    request_count: int
    max_running_requests: int
    makespan_ms: DistributionStatsRecord
    busy_ms: DistributionStatsRecord
    cache_management_ms: DistributionStatsRecord
    cache_management_fraction: DistributionStatsRecord
    queue_ms_p50: DistributionStatsRecord
    queue_ms_p95: DistributionStatsRecord
    ttft_ms_p50: DistributionStatsRecord
    ttft_ms_p95: DistributionStatsRecord
    end_to_end_ms_p50: DistributionStatsRecord
    end_to_end_ms_p95: DistributionStatsRecord
    itl_ms_p50: DistributionStatsRecord
    itl_ms_p95: DistributionStatsRecord
    itl_ms_max: DistributionStatsRecord


# ============================================================================
# 第二层：单用例记录层（共享 BaseCaseRecord + Lockstep / Scheduler 专属扩展）
# ============================================================================


class BaseCaseRecord(_DictAccessModel):
    """Lockstep 与 Scheduler 共享的单 Case 核心记录基类。

    聚合了：
    1. `engine.types.WorkloadSpec` 的通用负载描述（mode, batch_size, prompt_len, max_new_tokens, repeats, warmup）
    2. `core.types.StepAggregatedMetrics[DistributionStats]` 的跨场景公共吞吐与耗时指标
    3. 叶子画像组件（memory, packing, efficiency）
    """

    mode: str
    batch_size: int
    prompt_len: int | list[int]
    max_new_tokens: int | list[int]
    repeats: int = 1
    warmup: int = 0
    scale: str | None = None
    seed: int | None = None
    e2e_wall_ms: DistributionStatsRecord
    e2e_model_ms: DistributionStatsRecord | None = None
    throughput_tok_s: DistributionStatsRecord
    throughput_req_s: DistributionStatsRecord
    gpu_time_ratio: DistributionStatsRecord | None = None
    effective_prompt_tok_s: DistributionStatsRecord | None = None
    padded_prompt_tok_s: DistributionStatsRecord | None = None
    memory: MemoryProfileRecord
    packing: PackingInfoRecord | None = None
    efficiency: EfficiencyMetricsRecord | None = None


class LockstepCaseRecord(BaseCaseRecord):
    """Lockstep 单用例评测结果记录（对应 `bench.lockstep.types.LockstepCaseResult.to_record_dict()`）。"""

    ttft_ms: DistributionStatsRecord | None = None
    tpot_ms: DistributionStatsRecord | None = None
    op_latency_breakdown: dict[str, Any] | None = None


class SchedulerCaseRecord(BaseCaseRecord):
    """Scheduler 单用例评测结果记录（对应 `bench.scheduler.types.SchedulerCaseResult.to_record_dict()`）。"""

    case_id: str
    max_running_requests: int
    arrival_times_ms: float | list[float] = 0.0
    schedule: ScheduleMetricsRecord
    repeat_samples: list[dict[str, Any]] = Field(default_factory=list)
    sample_event_trace: list[dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# 第三层：顶层文件信封层 + 判别式联合（Discriminated Union）
# ============================================================================


class BenchmarkEnvelopeBase(_DictAccessModel):
    """统一顶层 Benchmark 信封基类（Lockstep 与 Scheduler 共享全部顶层元数据）。"""

    schema_version: int = SCHEMA_VERSION
    tag: str
    release_feature: str | None = None
    mode: str
    timestamp: str
    git: dict[str, Any]
    env: dict[str, Any]
    peak_specs: dict[str, float | None]
    model_cost: dict[str, Any] | None = None
    config: dict[str, Any]
    correctness: dict[str, Any]
    suite: dict[str, Any] | None = None


class LockstepBenchmarkRecord(BenchmarkEnvelopeBase):
    """Lockstep 基准测试完整记录信封。"""

    engine: Literal["lockstep"] = "lockstep"
    cases: list[LockstepCaseRecord]


class SchedulerBenchmarkRecord(BenchmarkEnvelopeBase):
    """Scheduler 基准测试完整记录信封。"""

    engine: Literal["scheduler"] = "scheduler"
    cases: list[SchedulerCaseRecord]


AnyBenchmarkRecord = Annotated[
    LockstepBenchmarkRecord | SchedulerBenchmarkRecord,
    Field(discriminator="engine"),
]

_BENCHMARK_RECORD_ADAPTER: TypeAdapter[LockstepBenchmarkRecord | SchedulerBenchmarkRecord] = TypeAdapter(
    AnyBenchmarkRecord
)


def build_benchmark_record(
    args: Any,
    model: torch.nn.Module,
    mode_label: str,
    cases: Sequence["LockstepCaseResult | SchedulerCaseResult"],
    correctness: CorrectnessReport | dict[str, Any],
    env: dict[str, Any],
    peaks: HardwarePeakSpecs | dict[str, float | None],
    cost: ModelCostProfile | dict[str, Any] | None,
    git: dict[str, Any],
    warmup: int,
    repeats: int,
    suite_info: dict[str, Any] | None = None,
    engine: Literal["lockstep", "scheduler"] | None = None,
) -> LockstepBenchmarkRecord | SchedulerBenchmarkRecord:
    """统一构建 Lockstep 或 Scheduler 标准评测结果信封（由 Pydantic 校验并返回模型）。"""
    if engine is None:
        if cases and hasattr(cases[0], "schedule_metrics"):
            engine = "scheduler"
        else:
            engine = "lockstep"

    record_cases = [c.to_record_dict() for c in cases]
    peaks_dict = peaks.to_dict() if hasattr(peaks, "to_dict") else peaks
    cost_dict = cost.to_dict() if cost is not None and hasattr(cost, "to_dict") else cost

    if isinstance(correctness, dict):
        correctness_dict = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in correctness.items()}
    elif hasattr(correctness, "to_dict"):
        correctness_dict = correctness.to_dict()
    else:
        correctness_dict = correctness

    try:
        dtype_str = str(next(model.parameters()).dtype)
    except StopIteration:
        dtype_str = "unknown"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "tag": args.tag,
        "release_feature": getattr(args, "release_feature", None),
        "mode": mode_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git,
        "env": env,
        "peak_specs": peaks_dict,
        "model_cost": cost_dict,
        "config": {
            "warmup": warmup,
            "repeats": repeats,
            "model": getattr(args, "model", None),
            "revision": getattr(args, "revision", None),
            "dtype": dtype_str,
            "tokenizer_included": False,
            "profile_mode": str(args.profile_mode) if getattr(args, "profile_mode", None) else None,
        },
        "correctness": correctness_dict,
        "cases": record_cases,
    }
    if suite_info is not None:
        payload["suite"] = suite_info

    return _BENCHMARK_RECORD_ADAPTER.validate_python(payload)


def _check_single_case_constraint(
    rec: LockstepBenchmarkRecord | SchedulerBenchmarkRecord,
    source: str | Path,
) -> None:
    """校验 result-history 约束：单 Case，或同版本同模式下按 small/medium/large 各存 1 个 Case。"""
    if not rec.cases:
        raise ValueError(f"基准记录 {source} 不包含任何 case (single-case 约束)。")
    if len(rec.cases) == 1:
        return
    scales = [c.scale for c in rec.cases]
    modes = {c.mode for c in rec.cases}
    if (
        rec.suite is None
        and len(modes) == 1
        and all(s in ("small", "medium", "large") for s in scales)
        and len(set(scales)) == len(scales)
    ):
        return
    raise ValueError(
        f"基准记录 {source} 必须只包含 1 个 case（或同模式下按 small/medium/large 各 1 个 case，single-case 约束），"
        f"实际包含 {len(rec.cases)} 个 cases。"
    )


def validate_benchmark_record(
    data: Any,
    *,
    require_single_case: bool = False,
    source: str | Path = "<memory>",
) -> LockstepBenchmarkRecord | SchedulerBenchmarkRecord:
    """将 dict 或已解析对象校验为强类型 BenchmarkRecord，可选强制检查单 Case 约束。"""
    if isinstance(data, LockstepBenchmarkRecord | SchedulerBenchmarkRecord):
        rec = data
    else:
        rec = _BENCHMARK_RECORD_ADAPTER.validate_python(data)

    if require_single_case:
        _check_single_case_constraint(rec, source)
    return rec


def load_benchmark_record(
    path: Path,
    *,
    require_single_case: bool = False,
) -> LockstepBenchmarkRecord | SchedulerBenchmarkRecord:
    """从 JSON 文件加载并校验 BenchmarkRecord，可选强制检查单 Case 约束。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    raw_text = path.read_text(encoding="utf-8")
    rec = _BENCHMARK_RECORD_ADAPTER.validate_json(raw_text)
    if require_single_case:
        _check_single_case_constraint(rec, path)
    return rec


__all__ = [
    "AnyBenchmarkRecord",
    "BaseCaseRecord",
    "BenchmarkEnvelopeBase",
    "DistributionStatsRecord",
    "EfficiencyMetricsRecord",
    "LockstepBenchmarkRecord",
    "LockstepCaseRecord",
    "MemoryProfileRecord",
    "PackingInfoRecord",
    "ScheduleMetricsRecord",
    "SchedulerBenchmarkRecord",
    "SchedulerCaseRecord",
    "build_benchmark_record",
    "load_benchmark_record",
    "validate_benchmark_record",
]
