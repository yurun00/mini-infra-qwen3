"""Scheduler 评测套件：由 `benchmarks/specs/scheduler/*.yaml` 声明，这里只负责把声明绑定到具体请求。

套件的全部配置（被测系统、请求负载、sweep、comparison）都在 YAML 中；新增 suite 只需新增一个 YAML 文件：

    benchmarks/specs/scheduler/<name>.yaml        # name 必须与文件名一致
    benchmarks/specs/scheduler/_bases/*.yaml      # 仅供 extends 复用，不可直接运行

`SchedulerSuite` 持有已校验的 `LoadedSuiteSpec` 与展开后的 `CaseSpec`；
`cases(vocab_size)` 把用例规格绑定到合成 token，得到 run_suite 消费的 `SchedulerCase` 列表。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_infra.bench.cases import CaseDimension, Comparison, RequestPayload, SchedulerCase
from mini_infra.bench.spec import (
    LoadedSuiteSpec,
    ResolvedWorkloadSpec,
    discover_suite_specs,
    get_suite_spec,
    load_suite_spec,
)
from mini_infra.bench.suite import BenchmarkSuite


@dataclass(frozen=True)
class SchedulerSuite(BenchmarkSuite[SchedulerCase]):
    """一个已加载的 Scheduler 评测套件（YAML 声明 + 展开后的用例规格）。"""

    @property
    def interleave(self) -> tuple[CaseDimension, ...]:
        return tuple(self.loaded.spec.execution.interleave)

    @property
    def comparison(self) -> Comparison | None:
        spec = self.loaded.spec.comparison
        return None if spec is None else Comparison.from_spec(spec)

    def cases(self, vocab_size: int) -> list[SchedulerCase]:
        """把用例规格绑定到合成 token；同一 workload 下的各 case 共享同一份请求规格。"""
        payloads_by_spec: dict[ResolvedWorkloadSpec, RequestPayload] = {}
        cases: list[SchedulerCase] = []
        for cs in self.case_specs:
            spec = cs.workload
            if spec not in payloads_by_spec:
                payloads_by_spec[spec] = RequestPayload.from_spec(spec, vocab_size)
            assert cs.max_running_requests is not None
            cases.append(
                SchedulerCase(
                    payload=payloads_by_spec[spec],
                    mode=cs.mode,
                    max_running_requests=cs.max_running_requests,
                    paged_cache_config=cs.paged_cache_config,
                    suite_name=self.name,
                )
            )
        return cases


def _suite_from_loaded(loaded: LoadedSuiteSpec) -> SchedulerSuite:
    return SchedulerSuite(loaded=loaded)


def load_scheduler_suite(path: Path, overrides: Mapping[str, Any] | None = None) -> SchedulerSuite:
    """从任意 YAML 路径加载 Scheduler 套件。"""
    loaded = load_suite_spec(path, overrides)
    if loaded.spec.engine != "scheduler":
        raise ValueError(f"{path}: engine 必须是 scheduler，实际为 {loaded.spec.engine}")
    return _suite_from_loaded(loaded)


def list_scheduler_suite_names() -> list[str]:
    return sorted(discover_suite_specs("scheduler"))


def get_scheduler_suite(name: str, overrides: Mapping[str, Any] | None = None) -> SchedulerSuite:
    return _suite_from_loaded(get_suite_spec("scheduler", name, overrides=overrides))


def get_all_scheduler_suites() -> Sequence[SchedulerSuite]:
    return [get_scheduler_suite(name) for name in list_scheduler_suite_names()]


__all__ = [
    "SchedulerSuite",
    "get_all_scheduler_suites",
    "get_scheduler_suite",
    "list_scheduler_suite_names",
    "load_scheduler_suite",
]
