"""Scheduler 评测套件：由 `benchmarks/specs/scheduler/*.yaml` 声明，这里只负责把声明绑定到具体请求。

套件的全部配置（被测系统、请求负载、sweep、comparison）都在 YAML 中；新增 suite 只需新增一个 YAML 文件：

    benchmarks/specs/scheduler/<name>.yaml        # name 必须与文件名一致
    benchmarks/specs/scheduler/_bases/*.yaml      # 仅供 extends 复用，不可直接运行

`SchedulerSuite` 持有已校验的 `LoadedSuiteSpec` 与展开后的 `CaseBlueprint`；
`cases(vocab_size)` 把蓝图绑定到合成 token，得到 run_suite 消费的 `SchedulerSuiteCase` 列表。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_infra.bench.scheduler.cases import CaseDimension, ComparisonSpec, SchedulerSuiteCase
from mini_infra.bench.spec import CaseBlueprint, LoadedSuiteSpec, ResolvedWorkload, discover_suite_specs, get_suite_spec
from mini_infra.bench.spec.loader import load_suite_spec
from mini_infra.bench.workloads import make_sequences
from mini_infra.core.request import RequestSpec, make_request_specs
from mini_infra.engine import Mode


@dataclass(frozen=True)
class SchedulerSuite:
    """一个已加载的 Scheduler 评测套件（YAML 声明 + 展开后的用例蓝图）。"""

    loaded: LoadedSuiteSpec
    blueprints: tuple[CaseBlueprint, ...]

    @property
    def name(self) -> str:
        return self.loaded.spec.name

    @property
    def warmup(self) -> int:
        return self.loaded.spec.execution.warmup

    @property
    def repeats(self) -> int:
        return self.loaded.spec.execution.repeats

    @property
    def interleave(self) -> tuple[CaseDimension, ...]:
        return tuple(self.loaded.spec.execution.interleave)

    @property
    def comparison(self) -> ComparisonSpec | None:
        config = self.loaded.spec.comparison
        return None if config is None else config.to_runtime()

    def modes(self) -> list[Mode]:
        """按展开顺序去重后的运行模式列表。"""
        return list(dict.fromkeys(bp.mode for bp in self.blueprints))

    def cases(self, vocab_size: int) -> list[SchedulerSuiteCase]:
        """把用例蓝图绑定到合成 token；同一 workload 下的各 case 共享同一份请求规格。"""
        specs_by_workload: dict[ResolvedWorkload, tuple[RequestSpec, ...]] = {}
        cases: list[SchedulerSuiteCase] = []
        for bp in self.blueprints:
            workload = bp.workload
            if workload not in specs_by_workload:
                specs_by_workload[workload] = tuple(
                    make_request_specs(
                        make_sequences(workload.prompt_lens, vocab_size),
                        workload.output_budgets,
                        workload.arrival_ms,
                    )
                )
            assert bp.max_running_requests is not None
            cases.append(
                SchedulerSuiteCase(
                    suite_name=self.name,
                    mode=bp.mode,
                    max_running_requests=bp.max_running_requests,
                    specs=specs_by_workload[workload],
                    workload_id=workload.workload_id,
                    paged_cache_config=bp.paged_cache_config,
                )
            )
        return cases

    def to_dict(self) -> dict[str, Any]:
        """写入结果 JSON `suite` 块：合并后的完整 YAML 声明及其内容哈希。"""
        return self.loaded.info()


def _suite_from_loaded(loaded: LoadedSuiteSpec) -> SchedulerSuite:
    return SchedulerSuite(loaded=loaded, blueprints=tuple(loaded.blueprints()))


def load_scheduler_suite(path: Path) -> SchedulerSuite:
    """从任意 YAML 路径加载 Scheduler 套件。"""
    loaded = load_suite_spec(path)
    if loaded.spec.engine != "scheduler":
        raise ValueError(f"{path}: engine 必须是 scheduler，实际为 {loaded.spec.engine}")
    return _suite_from_loaded(loaded)


def list_scheduler_suite_names() -> list[str]:
    return sorted(discover_suite_specs("scheduler"))


def get_scheduler_suite(name: str) -> SchedulerSuite:
    return _suite_from_loaded(get_suite_spec("scheduler", name))


def get_all_scheduler_suites() -> Sequence[SchedulerSuite]:
    return [get_scheduler_suite(name) for name in list_scheduler_suite_names()]


__all__ = [
    "SchedulerSuite",
    "get_all_scheduler_suites",
    "get_scheduler_suite",
    "list_scheduler_suite_names",
    "load_scheduler_suite",
]
