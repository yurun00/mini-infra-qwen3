"""Benchmark 评测结果统一 Pydantic v2 Schema 与多态信封（Lockstep & Scheduler 共享，纯 CPU 零 PyTorch 依赖）。

核心架构设计：**三层结构 + 判别式联合（Discriminated Union）**

```
pydantic.BaseModel
 └── _RecordModel                      # 基础能力基类：强类型属性访问 + 只读字典下标兼容 (rec["k"])
      ├── 第一层：叶子统计与画像组件 (Leaf Value Records，与运行时领域 Dataclass 一一映射)
      │    ├── DistributionStatsRecord     <-> bench.stats.DistributionStats
      │    ├── MemoryProfileRecord         <-> engine.metrics.MemoryProfile
      │    ├── PackingInfoRecord           <-> engine.metrics.BatchScheduleInfo[DistributionStats]
      │    ├── EfficiencyMetricsRecord     <-> bench.efficiency.EfficiencyMetrics
      │    └── ScheduleMetricsRecord       <-> engine.metrics.ScheduleMetrics[DistributionStats]
      │
      ├── 第二层：单用例记录层 (Case Level)
      │    └── BaseCaseRecord              # 共享 ResolvedWorkloadSpec + 公共指标 + 叶子画像
      │         ├── LockstepCaseRecord     <-> bench.lockstep.types.LockstepCaseResult
      │         │                              (扩展 ttft_ms, tpot_ms, op_latency_breakdown)
      │         └── SchedulerCaseRecord    <-> bench.scheduler.types.SchedulerCaseResult
      │                                        (扩展 workload_id, max_running_requests,
      │                                         arrival_times_ms, schedule, paged_cache_config 等)
      │
      └── 第三层：顶层文件信封层 (Envelope Level) + 判别式联合 (Discriminated Union)
           └── BenchmarkEnvelopeBase       # 共享顶层元数据 (schema_version, tag, git, env, config, correctness, suite)
                ├── LockstepBenchmarkRecord   # engine="lockstep",  cases: list[LockstepCaseRecord]
                └── SchedulerBenchmarkRecord  # engine="scheduler", cases: list[SchedulerCaseRecord]
```

通过 `AnyBenchmarkRecord`（`Field(discriminator="engine")`），
反序列化任意评测 JSON 时可根据 `engine` 自动分发校验为对应的强类型子类。
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


# ============================================================================
# 基础能力基类
# ============================================================================


class _RecordModel(BaseModel):
    """Benchmark 记录 Pydantic 公共基类（支持序列化导出为字典）。"""

    model_config = ConfigDict(extra="allow")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


# ============================================================================
# 第一层：叶子统计与画像组件（与运行时领域 Dataclass 一一对应）
# ============================================================================


class DistributionStatsRecord(_RecordModel):
    """测量样本统计分布 Schema（对应 `bench.stats.DistributionStats`）。"""

    median: float
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    max: float = 0.0


class PagedCacheProfileRecord(_RecordModel):
    """Paged KV Cache 专属 physical block pool 画像 Schema（对应 `engine.metrics.PagedCacheProfile.to_dict()`）。"""

    block_size: int = 0
    bytes_per_block: int = 0
    num_blocks: int = 0
    peak_used_blocks: int = 0
    block_cold_allocation_count: int = 0
    block_reuse_count: int = 0
    block_release_count: int = 0


class KVCacheProfileRecord(_RecordModel):
    """KV Cache 画像 Schema（对应 `engine.metrics.KVCacheProfile.to_dict()`）。"""

    peak_live_kv_cache_mb: float = 0.0
    peak_kv_cache_pool_mb: float = 0.0
    cache_realign_moved_bytes: int = 0
    accessed_kv_cache_tokens: int = 0
    paged_cache: PagedCacheProfileRecord | None = None


class MemoryProfileRecord(_RecordModel):
    """显存监控剖析 Schema，单位 MiB（对应 `engine.metrics.MemoryProfile.to_dict()`）。"""

    weights_mb: float | None = None
    baseline_allocated_mb: float = 0.0
    prefill_allocated_mb: float | None = None
    peak_allocated_mb: float = 0.0
    dynamic_peak_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    fragmentation_mb: float = 0.0
    kv_cache: KVCacheProfileRecord = Field(default_factory=KVCacheProfileRecord)


class PackingInfoRecord(_RecordModel):
    """批处理打包（Padding 效率）与 slot_utilization Schema（对应 `BatchScheduleInfo[DistributionStats]`）。"""

    padding_efficiency: DistributionStatsRecord
    slot_utilization: DistributionStatsRecord
    running_capacity_utilization: DistributionStatsRecord | None = None


class EfficiencyMetricsRecord(_RecordModel):
    """算力利用率 (MFU) 与显存带宽利用率 (MBU) Schema（对应 `bench.efficiency.EfficiencyMetrics`）。"""

    prefill_mfu: float | None = None
    decode_mfu: float | None = None
    decode_mbu: float | None = None


class ScheduleMetricsRecord(_RecordModel):
    """Scheduler 专属队列与步进调度统计 Schema（对应 `engine.metrics.ScheduleMetrics[DistributionStats]`）。"""

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


class BaseCaseRecord(_RecordModel):
    """Lockstep 与 Scheduler 共享的单 Case 核心记录基类。

    聚合了：
    1. `bench.spec.common.ResolvedWorkloadSpec` 的通用负载描述
       （engine_config, num_total_requests, prompt_len, max_new_tokens, repeats, warmup）
    2. `engine.metrics.StepAggregatedMetrics[DistributionStats]` 的跨场景公共吞吐与耗时指标
    3. 叶子画像组件（memory, packing, efficiency）
    """

    engine_config: str
    num_total_requests: int
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


class PagedCacheConfigRecord(_RecordModel):
    """Paged KV Cache 配置记录。"""

    block_size: int = 16
    num_blocks: int | None = None


class CaseKeyRecord(_RecordModel):
    """用例身份键记录（对应 `bench.cases.CaseKey.to_dict()`）。"""

    workload_id: str
    caching_method: str
    batching_method: str
    flags: list[str] = Field(default_factory=list)
    max_running_requests: int
    block_size: int | None = None
    num_blocks: int | None = None
    canonical_id: str | None = None


class SchedulerCaseRecord(BaseCaseRecord):
    """Scheduler 单用例评测结果记录（对应 `bench.scheduler.types.SchedulerCaseResult.to_record_dict()`）。"""

    workload_id: str
    max_running_requests: int
    arrival_times_ms: float | list[float] = 0.0
    schedule: ScheduleMetricsRecord
    repeat_samples: list[dict[str, Any]] = Field(default_factory=list)
    sample_event_trace: list[dict[str, Any]] = Field(default_factory=list)
    paged_cache_config: PagedCacheConfigRecord | None = None


class ComparisonResultRecord(_RecordModel):
    """通用基准指标相对对比统计。"""

    baseline_key: CaseKeyRecord
    candidate_key: CaseKeyRecord
    metric_path: str
    metric_name: str
    kind: str
    baseline_value: float
    candidate_value: float
    ratio: float
    relative_delta: float
    speedup: float | None = None
    gain: float | None = None
    reduction: float | None = None
    formatted: str | None = None


# ============================================================================
# 第三层：顶层文件信封层 + 判别式联合（Discriminated Union）
# ============================================================================


class BenchmarkEnvelopeBase(_RecordModel):
    """统一顶层 Benchmark 信封基类（Lockstep 与 Scheduler 共享全部顶层元数据）。"""

    schema_version: int = SCHEMA_VERSION
    tag: str
    release_feature: str | None = None
    engine_config: str
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
    comparisons: list[ComparisonResultRecord] = Field(default_factory=list)


AnyBenchmarkRecord = Annotated[
    LockstepBenchmarkRecord | SchedulerBenchmarkRecord,
    Field(discriminator="engine"),
]

__all__ = [
    "AnyBenchmarkRecord",
    "BaseCaseRecord",
    "BenchmarkEnvelopeBase",
    "CaseKeyRecord",
    "ComparisonResultRecord",
    "DistributionStatsRecord",
    "EfficiencyMetricsRecord",
    "KVCacheProfileRecord",
    "LockstepBenchmarkRecord",
    "LockstepCaseRecord",
    "MemoryProfileRecord",
    "PackingInfoRecord",
    "PagedCacheConfigRecord",
    "PagedCacheProfileRecord",
    "SCHEMA_VERSION",
    "ScheduleMetricsRecord",
    "SchedulerBenchmarkRecord",
    "SchedulerCaseRecord",
]
