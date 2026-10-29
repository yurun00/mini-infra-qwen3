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
    make_sequences,
)
from mini_infra.core.request import RequestSpec, make_request_specs
from mini_infra.engine import BatchingMethod, FeatureDimension, FeatureFlag, Mode


@dataclass(frozen=True)
class SchedulerSuiteCase:
    """Scheduler 评测最小单用例配置：包含用例 ID、套件名、运行模式（Mode）、最大并发数与请求规格。"""

    id: str
    suite_name: str
    mode: Mode
    max_running_requests: int
    specs: tuple[RequestSpec, ...]

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
        return {
            "case_id": self.id,
            "suite_name": self.suite_name,
            "mode": str(self.mode),
            "max_running_requests": self.max_running_requests,
            "request_count": self.request_count,
            "prompt_lengths": list(self.prompt_lengths),
            "output_budgets": list(self.output_budgets),
            "arrival_times_ms": list(self.arrival_times_ms),
        }


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
    return [d.value if isinstance(d, FeatureFlag) else "batching_method" for d in interleave]


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
    kv_cache: bool = True
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

    def modes(self) -> list[Mode]:
        flags = {FeatureFlag.KV_CACHE} if self.kv_cache else set()
        return [Mode(flags, batching_method=bm) for bm in self.batching_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        lengths = [self.prompt_length] * self.request_count
        sequences = [t.tolist() for t in make_sequences(lengths, vocab_size, device=device)]
        budgets = make_alternating_budgets(self.request_count, self.short_output, self.long_output)
        return make_request_specs(sequences, budgets)

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        return _expand_suite_cases_by_modes(self, self.build_specs(vocab_size, device=device))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
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
    kv_cache: bool = True
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

    def modes(self) -> list[Mode]:
        flags = {FeatureFlag.KV_CACHE} if self.kv_cache else set()
        return [Mode(flags, batching_method=bm) for bm in self.batching_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        lengths = [self.prompt_length] * self.request_count
        sequences = [t.tolist() for t in make_sequences(lengths, vocab_size, device=device)]
        budgets = [self.output_budget] * self.request_count
        return make_request_specs(sequences, budgets)

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        return _expand_suite_cases_by_modes(self, self.build_specs(vocab_size, device=device))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
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
    kv_cache: bool = True
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

    def modes(self) -> list[Mode]:
        flags = {FeatureFlag.KV_CACHE} if self.kv_cache else set()
        return [Mode(flags, batching_method=bm) for bm in self.batching_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        lengths = [self.prompt_length] * self.request_count
        sequences = [t.tolist() for t in make_sequences(lengths, vocab_size, device=device)]
        budgets = make_alternating_budgets(self.request_count, self.short_output, self.long_output)
        arrivals = make_arrival_times(self.request_count, self.arrival_interval_ms)
        return make_request_specs(sequences, budgets, arrivals)

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        return _expand_suite_cases_by_modes(self, self.build_specs(vocab_size, device=device))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
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
    kv_cache: bool = True
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

    def modes(self) -> list[Mode]:
        flags = {FeatureFlag.KV_CACHE} if self.kv_cache else set()
        return [Mode(flags, batching_method=bm) for bm in self.batching_methods]

    def build_specs(self, vocab_size: int, device: torch.device | None = None) -> list[RequestSpec]:
        lengths = [self.prompt_length, self.prompt_length, self.interference_prompt_length]
        sequences = [t.tolist() for t in make_sequences(lengths, vocab_size, device=device)]
        budgets = [self.long_output, self.long_output, self.short_output]
        arrivals = [0.0, 0.0, self.interference_arrival_ms]
        return make_request_specs(sequences, budgets, arrivals)

    def cases(self, vocab_size: int, device: torch.device | None = None) -> list[SchedulerSuiteCase]:
        return _expand_suite_cases_by_modes(self, self.build_specs(vocab_size, device=device))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        data["interleave"] = _serialize_interleave(self.interleave)
        return data


SCHEDULER_SUITES: dict[str, SchedulerSuite] = {
    "fixed-vs-continuous-variable-output-length": FixedVsContinuousVariableOutputLengthSuite(),
    "fixed-vs-continuous-uniform-output-length": FixedVsContinuousUniformOutputLengthSuite(),
    "fixed-vs-continuous-staggered-arrivals": FixedVsContinuousStaggeredArrivalsSuite(),
    "fixed-vs-continuous-prefill-interference": FixedVsContinuousPrefillInterferenceSuite(),
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
    "FixedVsContinuousPrefillInterferenceSuite",
    "FixedVsContinuousStaggeredArrivalsSuite",
    "FixedVsContinuousUniformOutputLengthSuite",
    "FixedVsContinuousVariableOutputLengthSuite",
    "SCHEDULER_SUITES",
    "SUITES",
    "SchedulerSuite",
    "SchedulerSuiteCase",
    "get_all_scheduler_suites",
    "get_scheduler_suite",
    "get_suite",
]
