"""Continuous Batching 受控实验套件配置注册表。

定义 4 个代表性评测场景：
1. fixed-vs-continuous-variable-output-length: 相同长度 prompt、交替长短预算、同时到达。
   观察动态退出与槽位利用率提升。
2. fixed-vs-continuous-uniform-output-length: 相同长度 prompt、统一均值输出预算、同时到达（对照基准）。
   两策略耗时差距应收窄。
3. fixed-vs-continuous-staggered-arrivals: 错峰到达请求。Continuous 随到随接，Fixed 必须等整批结束才能开新波次。
4. fixed-vs-continuous-prefill-interference: 长 Prompt 在中途插入，引起持续的计算与缓存搬移开销。
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
from mini_infra.engine import BatchingMethod, CachingMethod, FeatureDimension, FeatureFlag, Mode

INTERLEAVE_DIMENSIONS: dict[str, FeatureDimension] = {
    "batching_method": BatchingMethod,
    "caching_method": CachingMethod,
}


@dataclass(frozen=True)
class SchedulerSuiteCase:
    """Scheduler 评测最小单用例配置：包含用例 ID、套件名、运行模式（Mode）、最大并发数与请求规格。"""

    id: str
    suite_name: str
    mode: Mode
    max_running_requests: int
    specs: tuple[RequestSpec, ...]
    workload_id: str | None = None
    paged_cache_config: PagedCacheConfig | None = None

    def __post_init__(self) -> None:
        if self.mode.caching_method is CachingMethod.PAGED and self.paged_cache_config is None:
            raise ValueError(f"Paged 模式 (mode={self.mode}) 必须提供 paged_cache_config")
        if self.mode.caching_method is not CachingMethod.PAGED and self.paged_cache_config is not None:
            raise ValueError(f"非 Paged 模式 (mode={self.mode}) 不得提供 paged_cache_config")

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
            "case_id": self.id,
            "suite_name": self.suite_name,
            "mode": str(self.mode),
            "max_running_requests": self.max_running_requests,
            "request_count": self.request_count,
            "prompt_lengths": list(self.prompt_lengths),
            "output_budgets": list(self.output_budgets),
            "arrival_times_ms": list(self.arrival_times_ms),
        }
        if self.workload_id is not None:
            data["workload_id"] = self.workload_id
        if self.paged_cache_config is not None:
            data["block_size"] = self.paged_cache_config.block_size
        return data


@runtime_checkable
class SchedulerSuite(Protocol):
    """Scheduler 评测套件统一接口协议。"""

    name: str
    warmup: int
    repeats: int
    max_running_requests: int
    interleave: tuple[FeatureDimension, ...]

    def modes(self) -> list[Mode]: ...

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]: ...

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]: ...

    def to_dict(self) -> dict[str, Any]: ...


def _serialize_interleave(interleave: Sequence[FeatureDimension]) -> list[str]:
    """序列化 interleave 特征维度列表供 JSON 结果记录。"""
    names = {dim: name for name, dim in INTERLEAVE_DIMENSIONS.items()}
    return [d.value if isinstance(d, FeatureFlag) else names[d] for d in interleave]


def _case_id_for_mode(mode: Mode) -> str:
    if mode.batching_method == BatchingMethod.FIXED:
        return "fixed"
    if mode.batching_method == BatchingMethod.CONTINUOUS:
        return "continuous"
    return str(mode)


def _expand_suite_cases_by_modes(suite: SchedulerSuite, specs: Sequence[RequestSpec]) -> list[SchedulerSuiteCase]:
    """根据 Suite 的 modes() 列表与请求规格展开出具体的用例列表。"""
    specs_tuple = tuple(specs)
    return [
        SchedulerSuiteCase(
            id=_case_id_for_mode(mode),
            suite_name=suite.name,
            mode=mode,
            max_running_requests=suite.max_running_requests,
            specs=specs_tuple,
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
    interleave: tuple[FeatureDimension, ...] = (BatchingMethod,)
    warmup: int = 1
    repeats: int = 3

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
        return _expand_suite_cases_by_modes(self, self.build_specs(vocab_size, device=device))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
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
    interleave: tuple[FeatureDimension, ...] = (BatchingMethod,)
    warmup: int = 1
    repeats: int = 3

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
        return _expand_suite_cases_by_modes(self, self.build_specs(vocab_size, device=device))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
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
    interleave: tuple[FeatureDimension, ...] = (BatchingMethod,)
    warmup: int = 1
    repeats: int = 3

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
        return _expand_suite_cases_by_modes(self, self.build_specs(vocab_size, device=device))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
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
    interleave: tuple[FeatureDimension, ...] = (BatchingMethod,)
    warmup: int = 1
    repeats: int = 3

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
        return _expand_suite_cases_by_modes(self, self.build_specs(vocab_size, device=device))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
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
    interleave: tuple[FeatureDimension, ...] = (CachingMethod,)
    warmup: int = 1
    repeats: int = 3

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
        specs = tuple(self.build_specs(vocab_size=vocab_size, device=device))
        dense_mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
        paged_mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
        return [
            SchedulerSuiteCase(
                id="dense",
                suite_name=self.name,
                mode=dense_mode,
                max_running_requests=self.max_running_requests,
                specs=specs,
            ),
            SchedulerSuiteCase(
                id="paged",
                suite_name=self.name,
                mode=paged_mode,
                max_running_requests=self.max_running_requests,
                specs=specs,
                paged_cache_config=PagedCacheConfig(block_size=self.block_size),
            ),
        ]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["cache_methods"] = [str(m) for m in self.cache_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
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
    interleave: tuple[FeatureDimension, ...] = (CachingMethod,)
    warmup: int = 1
    repeats: int = 3

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
                    id="dense",
                    suite_name=self.name,
                    mode=dense_mode,
                    max_running_requests=self.max_running_requests,
                    specs=specs,
                    workload_id=workload_id,
                )
            )
            suite_cases.append(
                SchedulerSuiteCase(
                    id="paged",
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
    interleave: tuple[FeatureDimension, ...] = (CachingMethod,)
    warmup: int = 1
    repeats: int = 3

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
        specs = tuple(self.build_specs(vocab_size=vocab_size, device=device))
        dense_mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
        paged_mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
        suite_cases: list[SchedulerSuiteCase] = [
            SchedulerSuiteCase(
                id="dense",
                suite_name=self.name,
                mode=dense_mode,
                max_running_requests=self.max_running_requests,
                specs=specs,
                workload_id="block_size_sweep",
            )
        ]
        for bs in self.block_sizes:
            suite_cases.append(
                SchedulerSuiteCase(
                    id=f"paged-block-{bs}",
                    suite_name=self.name,
                    mode=paged_mode,
                    max_running_requests=self.max_running_requests,
                    specs=specs,
                    workload_id="block_size_sweep",
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
    "DenseVsPagedBlockSizeSuite",
    "DenseVsPagedLengthRatioSuite",
    "DenseVsPagedLongShortStressSuite",
    "FixedVsContinuousPrefillInterferenceSuite",
    "FixedVsContinuousStaggeredArrivalsSuite",
    "FixedVsContinuousUniformOutputLengthSuite",
    "FixedVsContinuousVariableOutputLengthSuite",
    "INTERLEAVE_DIMENSIONS",
    "SCHEDULER_SUITES",
    "SUITES",
    "SchedulerSuite",
    "SchedulerSuiteCase",
    "get_all_scheduler_suites",
    "get_scheduler_suite",
    "get_suite",
]
