"""评测套件通用容器（BenchmarkSuite）。"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from mini_infra.bench.cases import (
    BenchmarkCase,
    CaseDimension,
    Comparison,
)
from mini_infra.bench.spec.common.schema import SuiteEngine
from mini_infra.bench.spec.suite.expand import CaseSpec
from mini_infra.bench.spec.suite.loader import LoadedSuiteSpec
from mini_infra.engine import Mode


@dataclass(frozen=True)
class BenchmarkSuite:
    """评测套件通用容器：持有已校验加载的 LoadedSuiteSpec 与不可变的用例元组。"""

    loaded: LoadedSuiteSpec
    cases: tuple[BenchmarkCase, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self._build_cases()))

    def _build_cases(self) -> Sequence[BenchmarkCase]:
        """由用例规格构建具体 BenchmarkCase 实体。"""
        engine = self.engine
        cases: list[BenchmarkCase] = []
        for cs in self.case_specs:
            mrr = cs.workload.num_total_requests if engine == "lockstep" else cs.max_running_requests
            assert mrr is not None
            paged_config = None if engine == "lockstep" else cs.paged_cache_config
            cases.append(
                BenchmarkCase(
                    profile=cs.workload.to_profile(),
                    mode=cs.mode,
                    max_running_requests=mrr,
                    paged_cache_config=paged_config,
                    suite_name=self.name,
                    engine=engine,
                )
            )
        return tuple(cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[BenchmarkCase]:
        return iter(self.cases)

    def __getitem__(self, index: int) -> BenchmarkCase:
        return self.cases[index]

    @property
    def name(self) -> str:
        return self.loaded.spec.name

    @property
    def engine(self) -> SuiteEngine:
        return self.loaded.spec.engine

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
    def comparison(self) -> Comparison | None:
        spec = self.loaded.spec.comparison
        return None if spec is None else Comparison.from_spec(spec)

    @property
    def case_specs(self) -> tuple[CaseSpec, ...]:
        return self.loaded.cases

    def modes(self) -> list[Mode]:
        """按展开顺序去重后的运行模式列表。"""
        return list(dict.fromkeys(cs.mode for cs in self.case_specs))

    def to_dict(self) -> dict[str, Any]:
        """写入结果 JSON `suite` 块：合并后的完整 YAML 声明及其内容哈希。"""
        return self.loaded.info()


__all__ = [
    "BenchmarkSuite",
]
