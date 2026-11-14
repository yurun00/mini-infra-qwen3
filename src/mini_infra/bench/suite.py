"""评测套件通用容器（BenchmarkSuite）与套件加载、发现及查询服务。"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_infra.bench.cases import (
    BenchmarkCase,
    CaseDimension,
)
from mini_infra.bench.comparison import Comparison
from mini_infra.bench.spec.common.schema import SuiteEngine
from mini_infra.bench.spec.suite.loader import (
    LoadedSuiteSpec,
    discover_suite_specs,
    get_suite_spec,
    load_suite_spec,
)
from mini_infra.engine import EngineConfig


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """评测套件通用容器：持有自包含的执行期属性与不可变用例元组（阶段二 Execution Plane）。"""

    name: str
    engine: SuiteEngine
    cases: tuple[BenchmarkCase, ...]
    warmup: int
    repeats: int
    interleave: tuple[CaseDimension, ...] = ()
    comparison: Comparison | None = None
    spec_info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("suite name 不能为空")
        if self.warmup < 0:
            raise ValueError(f"warmup 必须 >= 0，实际为 {self.warmup}")
        if self.repeats <= 0:
            raise ValueError(f"repeats 必须 > 0，实际为 {self.repeats}")
        if not isinstance(self.cases, tuple):
            object.__setattr__(self, "cases", tuple(self.cases))
        if not isinstance(self.interleave, tuple):
            object.__setattr__(self, "interleave", tuple(self.interleave))

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[BenchmarkCase]:
        return iter(self.cases)

    def __getitem__(self, index: int) -> BenchmarkCase:
        return self.cases[index]

    def engine_configs(self) -> list[EngineConfig]:
        """按用例展开顺序返回去重后的 EngineConfig 列表。"""
        return list(dict.fromkeys(c.engine_config for c in self.cases))

    def to_dict(self) -> dict[str, Any]:
        """写入结果 JSON `suite` 块：合并后的完整 YAML 声明及其内容哈希（或执行画像字典）。"""
        if self.spec_info is not None:
            return dict(self.spec_info)
        info: dict[str, Any] = {
            "name": self.name,
            "engine": self.engine,
            "warmup": self.warmup,
            "repeats": self.repeats,
        }
        if self.interleave:
            info["interleave"] = [d.value for d in self.interleave]
        if self.comparison is not None:
            info["comparison"] = self.comparison.to_dict()
        return info

    @classmethod
    def from_spec(cls, loaded: LoadedSuiteSpec) -> "BenchmarkSuite":
        """从已校验并展开的 LoadedSuiteSpec 降级构建不可变 BenchmarkSuite 实体。"""
        engine = loaded.spec.engine
        cases: list[BenchmarkCase] = []
        for cs in loaded.cases:
            cases.append(
                BenchmarkCase(
                    profile=cs.workload.to_profile(),
                    batching_method=cs.batching_method,
                    kv_cache_config=cs.caching_spec.kv_cache_config,
                    max_running_requests=cs.max_running_requests,
                    scheduler_config=cs.scheduler_spec.scheduler_config,
                    suite_name=loaded.spec.name,
                    engine=engine,
                )
            )
        comp = loaded.spec.comparison
        return cls(
            name=loaded.spec.name,
            engine=engine,
            cases=tuple(cases),
            warmup=loaded.spec.execution.warmup,
            repeats=loaded.spec.execution.repeats,
            interleave=tuple(loaded.spec.execution.interleave),
            comparison=None if comp is None else Comparison.from_spec(comp),
            spec_info=loaded.info(),
        )


def load_benchmark_suite(
    path: Path,
    overrides: Mapping[str, Any] | None = None,
    expected_engine: SuiteEngine | None = None,
) -> BenchmarkSuite:
    """从任意 YAML 路径加载 BenchmarkSuite。若提供 expected_engine 则校验引擎一致性。"""
    loaded = load_suite_spec(path, overrides)
    if expected_engine is not None and loaded.spec.engine != expected_engine:
        raise ValueError(f"{path}: engine 必须是 {expected_engine}，实际为 {loaded.spec.engine}")
    return BenchmarkSuite.from_spec(loaded)


def list_benchmark_suite_names(engine: SuiteEngine) -> list[str]:
    """列出指定引擎内置的所有 suite 名称。专供测试使用（Test-only）。"""
    return sorted(discover_suite_specs(engine))


def get_benchmark_suite(
    engine: SuiteEngine,
    name: str,
    overrides: Mapping[str, Any] | None = None,
) -> BenchmarkSuite:
    """获取指定引擎内置的命名套件。专供测试使用（Test-only）。"""
    return BenchmarkSuite.from_spec(get_suite_spec(engine, name, overrides=overrides))


def get_all_benchmark_suites(engine: SuiteEngine) -> Sequence[BenchmarkSuite]:
    """获取指定引擎内置的全部套件。专供测试使用（Test-only）。"""
    return [get_benchmark_suite(engine, name) for name in list_benchmark_suite_names(engine)]


__all__ = [
    "BenchmarkSuite",
    "get_all_benchmark_suites",
    "get_benchmark_suite",
    "list_benchmark_suite_names",
    "load_benchmark_suite",
]
