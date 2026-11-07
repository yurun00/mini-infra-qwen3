"""Continuous Batching 受控实验套件配置注册表。

定义 4 个代表性评测场景：
1. fixed-vs-continuous-variable-output-length: 相同长度 prompt、交替长短预算、同时到达。
   观察动态退出与槽位利用率提升。
2. fixed-vs-continuous-uniform-output-length: 相同长度 prompt、统一均值输出预算、同时到达（对照基准）。
   两策略耗时差距应收窄。
3. fixed-vs-continuous-staggered-arrivals: 错峰到达请求。Continuous 随到随接，Fixed 必须等整批结束才能开新波次。
4. fixed-vs-continuous-prefill-interference: 长 Prompt 在中途插入，引起持续的计算与缓存搬移开销。

以及 Dense 与 Paged KV Cache 的 3 个对比套件：
5. dense-vs-paged-long-short-stress: 长短 Prompt 混杂并发下的显存与延迟基准对比。
6. dense-vs-paged-length-ratio: 扫描长短 Prompt 长度比。
7. dense-vs-paged-block-size: 扫描 Paged 物理块大小（8/16/32/64）。
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch

from mini_infra.bench.workloads import (
    make_alternating_budgets,
    make_arrival_times,
    make_long_short_specs,
    make_sequences,
)
from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.core.request import RequestSpec, make_request_specs
from mini_infra.engine import BatchingMethod, CachingMethod, FeatureFlag, Mode

CaseDimensionValue = str | CachingMethod | BatchingMethod | frozenset[FeatureFlag] | int | PagedCacheConfig | None


class CaseDimension(str, Enum):
    """用例身份键（CaseKey）的正交特征维度枚举。"""

    WORKLOAD_ID = "workload_id"
    CACHING_METHOD = "caching_method"
    BATCHING_METHOD = "batching_method"
    FLAGS = "flags"
    MAX_RUNNING_REQUESTS = "max_running_requests"
    PAGED_CACHE_CONFIG = "paged_cache_config"


@dataclass(frozen=True)
class CaseKey:
    """用例全局唯一身份键：由工作负载与系统配置正交维度完全确定。"""

    workload_id: str
    mode: Mode
    max_running_requests: int
    paged_cache_config: PagedCacheConfig | None = None

    def value(self, dim: CaseDimension | str) -> CaseDimensionValue:
        d = dim if isinstance(dim, CaseDimension) else CaseDimension(dim)
        if d is CaseDimension.WORKLOAD_ID:
            return self.workload_id
        if d is CaseDimension.CACHING_METHOD:
            return self.mode.caching_method
        if d is CaseDimension.BATCHING_METHOD:
            return self.mode.batching_method
        if d is CaseDimension.FLAGS:
            return self.mode.flags
        if d is CaseDimension.MAX_RUNNING_REQUESTS:
            return self.max_running_requests
        if d is CaseDimension.PAGED_CACHE_CONFIG:
            return self.paged_cache_config
        raise ValueError(f"未知维度: {dim}")

    def project(self, dims: Iterable[CaseDimension | str]) -> tuple[CaseDimensionValue, ...]:
        return tuple(self.value(d) for d in dims)

    @property
    def display_name(self) -> str:
        parts = [f"[{self.workload_id}]", str(self.mode), f"mrr={self.max_running_requests}"]
        if self.paged_cache_config is not None:
            parts.append(f"block={self.paged_cache_config.block_size}")
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.display_name

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "workload_id": self.workload_id,
            "mode": str(self.mode),
            "max_running_requests": self.max_running_requests,
        }
        if self.paged_cache_config is not None:
            data["paged_cache_config"] = {
                "block_size": self.paged_cache_config.block_size,
                "num_blocks": self.paged_cache_config.num_blocks,
            }
        return data


class MetricKind(str, Enum):
    """指标对比相对统计模式。"""

    SPEEDUP = "speedup"  # baseline / candidate（如耗时/makespan 加速比，数值越小越好）
    GAIN = "gain"  # (candidate - baseline) / baseline（如吞吐量增益，数值越大越好）
    REDUCTION = "reduction"  # (baseline - candidate) / baseline（如显存/token 削减，数值越小越好）


@dataclass(frozen=True)
class ComparisonMetric:
    """指定参与对比的指标点分路径与期望统计模式。"""

    path: str
    kind: MetricKind
    name: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.path.rsplit(".", 1)[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind.value,
            "name": self.display_name,
        }


@dataclass(frozen=True)
class ComparisonSpec:
    """Suite 级泛型对比规范配置。"""

    metrics: tuple[ComparisonMetric, ...]
    group_by: tuple[CaseDimension, ...]
    baseline: Mapping[CaseDimension, CaseDimensionValue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "group_by": [d.value for d in self.group_by],
            "baseline": {
                d.value if isinstance(d, CaseDimension) else str(d): v.value if hasattr(v, "value") else str(v)
                for d, v in self.baseline.items()
            },
        }


FIXED_VS_CONTINUOUS_COMPARISON_METRICS: tuple[ComparisonMetric, ...] = (
    ComparisonMetric("schedule_metrics.makespan_ms", MetricKind.SPEEDUP, "makespan"),
)

DENSE_VS_PAGED_COMPARISON_METRICS: tuple[ComparisonMetric, ...] = (
    ComparisonMetric("schedule_metrics.makespan_ms", MetricKind.SPEEDUP, "makespan"),
    ComparisonMetric("metrics.throughput_tok_s", MetricKind.GAIN, "throughput"),
    ComparisonMetric("memory.kv_cache.accessed_kv_cache_tokens", MetricKind.REDUCTION, "KV-read"),
    ComparisonMetric("memory.kv_cache.peak_kv_cache_pool_mb", MetricKind.REDUCTION, "pool"),
)


@dataclass(frozen=True)
class SchedulerSuiteCase:
    """Scheduler 评测最小单用例配置：包含用例身份键（CaseKey）对应的全量维度与请求规格。"""

    suite_name: str
    mode: Mode
    max_running_requests: int
    specs: tuple[RequestSpec, ...]
    workload_id: str
    paged_cache_config: PagedCacheConfig | None = None

    def __post_init__(self) -> None:
        if not self.workload_id:
            raise ValueError("workload_id 不能为空")
        if self.mode.caching_method is CachingMethod.PAGED and self.paged_cache_config is None:
            raise ValueError(f"Paged 模式 (mode={self.mode}) 必须提供 paged_cache_config")
        if self.mode.caching_method is not CachingMethod.PAGED and self.paged_cache_config is not None:
            raise ValueError(f"非 Paged 模式 (mode={self.mode}) 不得提供 paged_cache_config")

    @property
    def key(self) -> CaseKey:
        return CaseKey(
            workload_id=self.workload_id,
            mode=self.mode,
            max_running_requests=self.max_running_requests,
            paged_cache_config=self.paged_cache_config,
        )

    @property
    def request_count(self) -> int:
        return len(self.specs)

    @property
    def prompt_lengths(self) -> tuple[int, ...]:
        return tuple(len(s.token_ids) for s in self.specs)

    @property
    def output_budgets(self) -> tuple[int, ...]:
        return tuple(s.max_new_tokens for s in self.specs)

    @property
    def arrival_times_ms(self) -> tuple[float, ...]:
        return tuple(s.arrival_ms for s in self.specs)

    def to_spec_dict(self) -> dict[str, Any]:
        """导出用例负载规格字典。"""
        data: dict[str, Any] = {
            "suite_name": self.suite_name,
            "workload_id": self.workload_id,
            "mode": str(self.mode),
            "max_running_requests": self.max_running_requests,
            "request_count": self.request_count,
            "prompt_lengths": list(self.prompt_lengths),
            "output_budgets": list(self.output_budgets),
            "arrival_times_ms": list(self.arrival_times_ms),
        }
        if self.paged_cache_config is not None:
            data["paged_cache_config"] = {
                "block_size": self.paged_cache_config.block_size,
                "num_blocks": self.paged_cache_config.num_blocks,
            }
        return data


@runtime_checkable
class SchedulerSuite(Protocol):
    """Scheduler 评测套件统一接口协议。"""

    name: str
    warmup: int
    repeats: int
    max_running_requests: int
    interleave: tuple[CaseDimension, ...]
    comparison: ComparisonSpec | None

    def modes(self) -> list[Mode]: ...

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]: ...

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]: ...

    def to_dict(self) -> dict[str, Any]: ...


def _serialize_interleave(interleave: Sequence[CaseDimension]) -> list[str]:
    """序列化 interleave 特征维度列表供 JSON 结果记录。"""
    return [d.value for d in interleave]


def _expand_suite_cases_by_modes(
    suite: SchedulerSuite, specs: Sequence[RequestSpec], workload_id: str
) -> list[SchedulerSuiteCase]:
    """根据 Suite 的 modes() 列表与请求规格展开出具体的用例列表。"""
    specs_tuple = tuple(specs)
    return [
        SchedulerSuiteCase(
            suite_name=suite.name,
            mode=mode,
            max_running_requests=suite.max_running_requests,
            specs=specs_tuple,
            workload_id=workload_id,
        )
        for mode in suite.modes()
    ]


@dataclass(frozen=True)
class FixedVsContinuousVariableOutputLengthSuite:
    """固定批处理与连续批处理下变长输出（Variable Output Length）的对比实验套件。"""

    name: str = "fixed-vs-continuous-variable-output-length"
    max_running_requests: int = 4
    request_count: int = 8
    prompt_length: int = 128
    short_output: int = 8
    long_output: int = 32
    batching_methods: tuple[BatchingMethod, ...] = (BatchingMethod.FIXED, BatchingMethod.CONTINUOUS)
    cache_methods: tuple[CachingMethod, ...] = (CachingMethod.DENSE,)
    interleave: tuple[CaseDimension, ...] = (CaseDimension.BATCHING_METHOD,)
    warmup: int = 1
    repeats: int = 3
    comparison: ComparisonSpec | None = ComparisonSpec(
        metrics=FIXED_VS_CONTINUOUS_COMPARISON_METRICS,
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.BATCHING_METHOD: BatchingMethod.FIXED},
    )

    def __post_init__(self) -> None:
        if self.max_running_requests < 1 or self.request_count < 1 or self.prompt_length < 1:
            raise ValueError("max_running_requests, request_count, prompt_length 必须大于 0")
        if self.short_output < 1 or self.long_output < 1:
            raise ValueError("输出预算必须大于 0")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("warmup 不能小于 0，repeats 必须大于 0")
        if not self.batching_methods:
            raise ValueError("batching_methods 不能为空")
        if not self.cache_methods:
            raise ValueError("cache_methods 不能为空")

    def modes(self) -> list[Mode]:
        return [Mode(cm, batching_method=bm) for bm in self.batching_methods for cm in self.cache_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        lengths = [self.prompt_length] * self.request_count
        sequences = make_sequences(lengths, vocab_size)
        budgets = make_alternating_budgets(self.request_count, self.short_output, self.long_output)
        return make_request_specs(sequences, budgets)

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        return _expand_suite_cases_by_modes(
            self, self.build_specs(vocab_size, device=device), workload_id="variable_output_length"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
        if self.comparison is not None:
            data["comparison"] = self.comparison.to_dict()
        return data


@dataclass(frozen=True)
class FixedVsContinuousUniformOutputLengthSuite:
    """固定批处理与连续批处理下等长输出（Uniform Output Length）的对照实验套件。"""

    name: str = "fixed-vs-continuous-uniform-output-length"
    max_running_requests: int = 4
    request_count: int = 8
    prompt_length: int = 128
    output_budget: int = 20
    batching_methods: tuple[BatchingMethod, ...] = (BatchingMethod.FIXED, BatchingMethod.CONTINUOUS)
    cache_methods: tuple[CachingMethod, ...] = (CachingMethod.DENSE,)
    interleave: tuple[CaseDimension, ...] = (CaseDimension.BATCHING_METHOD,)
    warmup: int = 1
    repeats: int = 3
    comparison: ComparisonSpec | None = ComparisonSpec(
        metrics=FIXED_VS_CONTINUOUS_COMPARISON_METRICS,
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.BATCHING_METHOD: BatchingMethod.FIXED},
    )

    def __post_init__(self) -> None:
        if self.max_running_requests < 1 or self.request_count < 1 or self.prompt_length < 1:
            raise ValueError("参数必须大于 0")
        if self.output_budget < 1:
            raise ValueError("output_budget 必须大于 0")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("warmup 不能小于 0，repeats 必须大于 0")
        if not self.batching_methods:
            raise ValueError("batching_methods 不能为空")
        if not self.cache_methods:
            raise ValueError("cache_methods 不能为空")

    def modes(self) -> list[Mode]:
        return [Mode(cm, batching_method=bm) for bm in self.batching_methods for cm in self.cache_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        lengths = [self.prompt_length] * self.request_count
        sequences = make_sequences(lengths, vocab_size)
        budgets = [self.output_budget] * self.request_count
        return make_request_specs(sequences, budgets)

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        return _expand_suite_cases_by_modes(
            self, self.build_specs(vocab_size, device=device), workload_id="uniform_output_length"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
        if self.comparison is not None:
            data["comparison"] = self.comparison.to_dict()
        return data


@dataclass(frozen=True)
class FixedVsContinuousStaggeredArrivalsSuite:
    """固定批处理与连续批处理下错峰到达（Staggered Arrivals）的对比实验套件。"""

    name: str = "fixed-vs-continuous-staggered-arrivals"
    max_running_requests: int = 4
    request_count: int = 8
    prompt_length: int = 128
    short_output: int = 8
    long_output: int = 32
    arrival_interval_ms: float = 150.0
    batching_methods: tuple[BatchingMethod, ...] = (BatchingMethod.FIXED, BatchingMethod.CONTINUOUS)
    cache_methods: tuple[CachingMethod, ...] = (CachingMethod.DENSE,)
    interleave: tuple[CaseDimension, ...] = (CaseDimension.BATCHING_METHOD,)
    warmup: int = 1
    repeats: int = 3
    comparison: ComparisonSpec | None = ComparisonSpec(
        metrics=FIXED_VS_CONTINUOUS_COMPARISON_METRICS,
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.BATCHING_METHOD: BatchingMethod.FIXED},
    )

    def __post_init__(self) -> None:
        if self.max_running_requests < 1 or self.request_count < 1 or self.prompt_length < 1:
            raise ValueError("参数必须大于 0")
        if self.short_output < 1 or self.long_output < 1:
            raise ValueError("输出预算必须大于 0")
        if self.arrival_interval_ms < 0:
            raise ValueError("arrival_interval_ms 不能小于 0")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("warmup 不能小于 0，repeats 必须大于 0")
        if not self.batching_methods:
            raise ValueError("batching_methods 不能为空")
        if not self.cache_methods:
            raise ValueError("cache_methods 不能为空")

    def modes(self) -> list[Mode]:
        return [Mode(cm, batching_method=bm) for bm in self.batching_methods for cm in self.cache_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        lengths = [self.prompt_length] * self.request_count
        sequences = make_sequences(lengths, vocab_size)
        budgets = make_alternating_budgets(self.request_count, self.short_output, self.long_output)
        arrivals = make_arrival_times(self.request_count, self.arrival_interval_ms)
        return make_request_specs(sequences, budgets, arrivals)

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        return _expand_suite_cases_by_modes(
            self, self.build_specs(vocab_size, device=device), workload_id="staggered_arrivals"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
        if self.comparison is not None:
            data["comparison"] = self.comparison.to_dict()
        return data


@dataclass(frozen=True)
class FixedVsContinuousPrefillInterferenceSuite:
    """固定批处理与连续批处理下 Prefill 插入干扰（Prefill Interference）的对比实验套件。"""

    name: str = "fixed-vs-continuous-prefill-interference"
    max_running_requests: int = 4
    prompt_length: int = 128
    interference_prompt_length: int = 512
    short_output: int = 8
    long_output: int = 32
    interference_arrival_ms: float = 250.0
    batching_methods: tuple[BatchingMethod, ...] = (BatchingMethod.FIXED, BatchingMethod.CONTINUOUS)
    cache_methods: tuple[CachingMethod, ...] = (CachingMethod.DENSE,)
    interleave: tuple[CaseDimension, ...] = (CaseDimension.BATCHING_METHOD,)
    warmup: int = 1
    repeats: int = 3
    comparison: ComparisonSpec | None = ComparisonSpec(
        metrics=FIXED_VS_CONTINUOUS_COMPARISON_METRICS,
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.BATCHING_METHOD: BatchingMethod.FIXED},
    )

    def __post_init__(self) -> None:
        if self.max_running_requests < 3:
            raise ValueError("fixed-vs-continuous-prefill-interference 套件要求 max_running_requests 至少为 3")
        if self.prompt_length < 1 or self.interference_prompt_length < 1:
            raise ValueError("prompt_length 必须大于 0")
        if self.short_output < 1 or self.long_output < 1:
            raise ValueError("输出预算必须大于 0")
        if self.interference_arrival_ms < 0:
            raise ValueError("interference_arrival_ms 不能小于 0")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("warmup 不能小于 0，repeats 必须大于 0")
        if not self.batching_methods:
            raise ValueError("batching_methods 不能为空")
        if not self.cache_methods:
            raise ValueError("cache_methods 不能为空")

    def modes(self) -> list[Mode]:
        return [Mode(cm, batching_method=bm) for bm in self.batching_methods for cm in self.cache_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        lengths = [self.prompt_length, self.prompt_length, self.interference_prompt_length]
        sequences = make_sequences(lengths, vocab_size)
        budgets = [self.long_output, self.long_output, self.short_output]
        arrivals = [0.0, 0.0, self.interference_arrival_ms]
        return make_request_specs(sequences, budgets, arrivals)

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        return _expand_suite_cases_by_modes(
            self, self.build_specs(vocab_size, device=device), workload_id="prefill_interference"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
        if self.comparison is not None:
            data["comparison"] = self.comparison.to_dict()
        return data


@dataclass(frozen=True)
class DenseVsPagedLongShortStressSuite:
    """Dense 与 Paged KV Cache 在长短请求混杂压力负载下的对比评测套件。"""

    name: str = "dense-vs-paged-long-short-stress"
    max_running_requests: int = 8
    short_prompt_length: int = 128
    long_prompt_length: int = 2048
    initial_short_requests: int = 7
    followup_short_requests: int = 9
    short_output: int = 96
    followup_output: int = 64
    long_output: int = 8
    long_arrival_ms: float = 200.0
    followup_arrival_ms: float = 201.0
    block_size: int = 16
    batching_methods: tuple[BatchingMethod, ...] = (BatchingMethod.CONTINUOUS,)
    cache_methods: tuple[CachingMethod, ...] = (CachingMethod.DENSE, CachingMethod.PAGED)
    interleave: tuple[CaseDimension, ...] = (CaseDimension.CACHING_METHOD, CaseDimension.PAGED_CACHE_CONFIG)
    warmup: int = 1
    repeats: int = 3
    comparison: ComparisonSpec | None = ComparisonSpec(
        metrics=DENSE_VS_PAGED_COMPARISON_METRICS,
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.CACHING_METHOD: CachingMethod.DENSE},
    )

    def __post_init__(self) -> None:
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests 必须大于 0")
        if self.initial_short_requests < 1 or self.followup_short_requests < 1:
            raise ValueError("请求数量必须大于 0")
        if self.initial_short_requests >= self.max_running_requests:
            raise ValueError("initial_short_requests 必须小于 max_running_requests")
        if self.short_prompt_length < 1 or self.long_prompt_length < 1:
            raise ValueError("prompt_length 必须大于 0")
        if self.short_output < 1 or self.followup_output < 1 or self.long_output < 1:
            raise ValueError("输出预算必须大于 0")
        if self.long_arrival_ms < 0 or self.followup_arrival_ms < self.long_arrival_ms:
            raise ValueError("到达时间非法")
        if self.block_size < 1:
            raise ValueError("block_size 必须大于 0")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("warmup 不能小于 0，repeats 必须大于 0")
        if not self.batching_methods:
            raise ValueError("batching_methods 不能为空")
        if not self.cache_methods:
            raise ValueError("cache_methods 不能为空")

    def modes(self) -> list[Mode]:
        return [Mode(cm, batching_method=bm) for bm in self.batching_methods for cm in self.cache_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        return make_long_short_specs(
            vocab_size=vocab_size,
            short_prompt_length=self.short_prompt_length,
            long_prompt_length=self.long_prompt_length,
            initial_short_requests=self.initial_short_requests,
            followup_short_requests=self.followup_short_requests,
            short_output=self.short_output,
            followup_output=self.followup_output,
            long_output=self.long_output,
            long_arrival_ms=self.long_arrival_ms,
            followup_arrival_ms=self.followup_arrival_ms,
            max_running_requests=self.max_running_requests,
        )

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        specs = tuple(self.build_specs(vocab_size, device=device))
        dense_mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
        paged_mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
        workload_id = "long_short_stress"
        return [
            SchedulerSuiteCase(
                suite_name=self.name,
                mode=dense_mode,
                max_running_requests=self.max_running_requests,
                specs=specs,
                workload_id=workload_id,
            ),
            SchedulerSuiteCase(
                suite_name=self.name,
                mode=paged_mode,
                max_running_requests=self.max_running_requests,
                specs=specs,
                workload_id=workload_id,
                paged_cache_config=PagedCacheConfig(block_size=self.block_size),
            ),
        ]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
        if self.comparison is not None:
            data["comparison"] = self.comparison.to_dict()
        return data


@dataclass(frozen=True)
class DenseVsPagedLengthRatioSuite:
    """Dense 与 Paged KV Cache 在不同长 Prompt 长度下的对比评测套件（扫描长/短长度比）。"""

    name: str = "dense-vs-paged-length-ratio"
    max_running_requests: int = 8
    short_prompt_length: int = 128
    long_prompt_lengths: tuple[int, ...] = (128, 512, 2048, 3072)
    initial_short_requests: int = 7
    followup_short_requests: int = 9
    short_output: int = 96
    followup_output: int = 64
    long_output: int = 8
    long_arrival_ms: float = 200.0
    followup_arrival_ms: float = 201.0
    block_size: int = 16
    batching_methods: tuple[BatchingMethod, ...] = (BatchingMethod.CONTINUOUS,)
    cache_methods: tuple[CachingMethod, ...] = (CachingMethod.DENSE, CachingMethod.PAGED)
    interleave: tuple[CaseDimension, ...] = (CaseDimension.CACHING_METHOD, CaseDimension.PAGED_CACHE_CONFIG)
    warmup: int = 1
    repeats: int = 3
    comparison: ComparisonSpec | None = ComparisonSpec(
        metrics=DENSE_VS_PAGED_COMPARISON_METRICS,
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.CACHING_METHOD: CachingMethod.DENSE},
    )

    def __post_init__(self) -> None:
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests 必须大于 0")
        if self.initial_short_requests < 1 or self.followup_short_requests < 1:
            raise ValueError("请求数量必须大于 0")
        if self.initial_short_requests >= self.max_running_requests:
            raise ValueError("initial_short_requests 必须小于 max_running_requests")
        if self.short_prompt_length < 1:
            raise ValueError("short_prompt_length 必须大于 0")
        if not self.long_prompt_lengths or any(length < 1 for length in self.long_prompt_lengths):
            raise ValueError("long_prompt_lengths 必须包含大于 0 的整数")
        if self.short_output < 1 or self.followup_output < 1 or self.long_output < 1:
            raise ValueError("输出预算必须大于 0")
        if self.long_arrival_ms < 0 or self.followup_arrival_ms < self.long_arrival_ms:
            raise ValueError("到达时间非法")
        if self.block_size < 1:
            raise ValueError("block_size 必须大于 0")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("warmup 不能小于 0，repeats 必须大于 0")
        if not self.batching_methods:
            raise ValueError("batching_methods 不能为空")
        if not self.cache_methods:
            raise ValueError("cache_methods 不能为空")

    def modes(self) -> list[Mode]:
        return [Mode(cm, batching_method=bm) for bm in self.batching_methods for cm in self.cache_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        long_len = self.long_prompt_lengths[0] if self.long_prompt_lengths else 2048
        return make_long_short_specs(
            vocab_size=vocab_size,
            short_prompt_length=self.short_prompt_length,
            long_prompt_length=long_len,
            initial_short_requests=self.initial_short_requests,
            followup_short_requests=self.followup_short_requests,
            short_output=self.short_output,
            followup_output=self.followup_output,
            long_output=self.long_output,
            long_arrival_ms=self.long_arrival_ms,
            followup_arrival_ms=self.followup_arrival_ms,
            max_running_requests=self.max_running_requests,
        )

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        dense_mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
        paged_mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
        suite_cases: list[SchedulerSuiteCase] = []
        for long_len in self.long_prompt_lengths:
            workload_id = f"length_ratio_{long_len}"
            specs = tuple(
                make_long_short_specs(
                    vocab_size=vocab_size,
                    short_prompt_length=self.short_prompt_length,
                    long_prompt_length=long_len,
                    initial_short_requests=self.initial_short_requests,
                    followup_short_requests=self.followup_short_requests,
                    short_output=self.short_output,
                    followup_output=self.followup_output,
                    long_output=self.long_output,
                    long_arrival_ms=self.long_arrival_ms,
                    followup_arrival_ms=self.followup_arrival_ms,
                    max_running_requests=self.max_running_requests,
                )
            )
            suite_cases.append(
                SchedulerSuiteCase(
                    suite_name=self.name,
                    mode=dense_mode,
                    max_running_requests=self.max_running_requests,
                    specs=specs,
                    workload_id=workload_id,
                )
            )
            suite_cases.append(
                SchedulerSuiteCase(
                    suite_name=self.name,
                    mode=paged_mode,
                    max_running_requests=self.max_running_requests,
                    specs=specs,
                    workload_id=workload_id,
                    paged_cache_config=PagedCacheConfig(block_size=self.block_size),
                )
            )
        return suite_cases

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
        data["long_prompt_lengths"] = list(self.long_prompt_lengths)
        if self.comparison is not None:
            data["comparison"] = self.comparison.to_dict()
        return data


@dataclass(frozen=True)
class DenseVsPagedBlockSizeSuite:
    """Dense 与不同 block_size 的 Paged KV Cache 的对比评测套件（扫描块大小）。"""

    name: str = "dense-vs-paged-block-size"
    max_running_requests: int = 8
    short_prompt_length: int = 128
    long_prompt_length: int = 2048
    initial_short_requests: int = 7
    followup_short_requests: int = 9
    short_output: int = 96
    followup_output: int = 64
    long_output: int = 8
    long_arrival_ms: float = 200.0
    followup_arrival_ms: float = 201.0
    block_sizes: tuple[int, ...] = (8, 16, 32, 64)
    batching_methods: tuple[BatchingMethod, ...] = (BatchingMethod.CONTINUOUS,)
    cache_methods: tuple[CachingMethod, ...] = (CachingMethod.DENSE, CachingMethod.PAGED)
    interleave: tuple[CaseDimension, ...] = (CaseDimension.CACHING_METHOD, CaseDimension.PAGED_CACHE_CONFIG)
    warmup: int = 1
    repeats: int = 3
    comparison: ComparisonSpec | None = ComparisonSpec(
        metrics=DENSE_VS_PAGED_COMPARISON_METRICS,
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.CACHING_METHOD: CachingMethod.DENSE},
    )

    def __post_init__(self) -> None:
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests 必须大于 0")
        if self.initial_short_requests < 1 or self.followup_short_requests < 1:
            raise ValueError("请求数量必须大于 0")
        if self.initial_short_requests >= self.max_running_requests:
            raise ValueError("initial_short_requests 必须小于 max_running_requests")
        if self.short_prompt_length < 1 or self.long_prompt_length < 1:
            raise ValueError("prompt_length 必须大于 0")
        if self.short_output < 1 or self.followup_output < 1 or self.long_output < 1:
            raise ValueError("输出预算必须大于 0")
        if self.long_arrival_ms < 0 or self.followup_arrival_ms < self.long_arrival_ms:
            raise ValueError("到达时间非法")
        if not self.block_sizes or any(bs < 1 for bs in self.block_sizes):
            raise ValueError("block_sizes 必须包含大于 0 的整数")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("warmup 不能小于 0，repeats 必须大于 0")
        if not self.batching_methods:
            raise ValueError("batching_methods 不能为空")
        if not self.cache_methods:
            raise ValueError("cache_methods 不能为空")

    def modes(self) -> list[Mode]:
        return [Mode(cm, batching_method=bm) for bm in self.batching_methods for cm in self.cache_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        return make_long_short_specs(
            vocab_size=vocab_size,
            short_prompt_length=self.short_prompt_length,
            long_prompt_length=self.long_prompt_length,
            initial_short_requests=self.initial_short_requests,
            followup_short_requests=self.followup_short_requests,
            short_output=self.short_output,
            followup_output=self.followup_output,
            long_output=self.long_output,
            long_arrival_ms=self.long_arrival_ms,
            followup_arrival_ms=self.followup_arrival_ms,
            max_running_requests=self.max_running_requests,
        )

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        specs = tuple(self.build_specs(vocab_size, device=device))
        dense_mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
        paged_mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
        workload_id = "block_size_sweep"
        suite_cases: list[SchedulerSuiteCase] = [
            SchedulerSuiteCase(
                suite_name=self.name,
                mode=dense_mode,
                max_running_requests=self.max_running_requests,
                specs=specs,
                workload_id=workload_id,
            )
        ]
        for bs in self.block_sizes:
            suite_cases.append(
                SchedulerSuiteCase(
                    suite_name=self.name,
                    mode=paged_mode,
                    max_running_requests=self.max_running_requests,
                    specs=specs,
                    workload_id=workload_id,
                    paged_cache_config=PagedCacheConfig(block_size=bs),
                )
            )
        return suite_cases

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
        data["block_sizes"] = list(self.block_sizes)
        if self.comparison is not None:
            data["comparison"] = self.comparison.to_dict()
        return data


SCHEDULER_SUITES: dict[str, SchedulerSuite] = {
    "fixed-vs-continuous-variable-output-length": FixedVsContinuousVariableOutputLengthSuite(),
    "fixed-vs-continuous-uniform-output-length": FixedVsContinuousUniformOutputLengthSuite(),
    "fixed-vs-continuous-staggered-arrivals": FixedVsContinuousStaggeredArrivalsSuite(),
    "fixed-vs-continuous-prefill-interference": FixedVsContinuousPrefillInterferenceSuite(),
    "dense-vs-paged-long-short-stress": DenseVsPagedLongShortStressSuite(),
    "dense-vs-paged-length-ratio": DenseVsPagedLengthRatioSuite(),
    "dense-vs-paged-block-size": DenseVsPagedBlockSizeSuite(),
}


def get_scheduler_suite(name: str) -> SchedulerSuite:
    try:
        return SCHEDULER_SUITES[name]
    except KeyError as err:
        raise ValueError(f"未知 scheduler suite: {name}，可选: {sorted(SCHEDULER_SUITES)}") from err


get_suite = get_scheduler_suite
SUITES = SCHEDULER_SUITES


def get_all_scheduler_suites() -> Sequence[SchedulerSuite]:
    return list(SCHEDULER_SUITES.values())


__all__ = [
    "CaseDimension",
    "CaseDimensionValue",
    "CaseKey",
    "ComparisonMetric",
    "ComparisonSpec",
    "DENSE_VS_PAGED_COMPARISON_METRICS",
    "DenseVsPagedBlockSizeSuite",
    "DenseVsPagedLengthRatioSuite",
    "DenseVsPagedLongShortStressSuite",
    "FIXED_VS_CONTINUOUS_COMPARISON_METRICS",
    "FixedVsContinuousPrefillInterferenceSuite",
    "FixedVsContinuousStaggeredArrivalsSuite",
    "FixedVsContinuousUniformOutputLengthSuite",
    "FixedVsContinuousVariableOutputLengthSuite",
    "MetricKind",
    "SCHEDULER_SUITES",
    "SUITES",
    "SchedulerSuite",
    "SchedulerSuiteCase",
    "get_all_scheduler_suites",
    "get_scheduler_suite",
    "get_suite",
]
