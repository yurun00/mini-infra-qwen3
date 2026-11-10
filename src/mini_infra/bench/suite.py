"""评测套件通用抽象基类（BenchmarkSuite）。"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from mini_infra.bench.spec.suite.expand import CaseSpec
from mini_infra.bench.spec.suite.loader import LoadedSuiteSpec
from mini_infra.engine import Mode

CaseT = TypeVar("CaseT")


@dataclass(frozen=True)
class BenchmarkSuite(ABC, Generic[CaseT]):
    """评测套件通用抽象基类：持有已校验加载的 LoadedSuiteSpec 与不可变的用例元组。"""

    loaded: LoadedSuiteSpec
    cases: tuple[CaseT, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self._build_cases()))

    @abstractmethod
    def _build_cases(self) -> Sequence[CaseT]:
        """由子类实现用例规格到具体 Case 实体的映射。"""
        ...

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[CaseT]:
        return iter(self.cases)

    def __getitem__(self, index: int) -> CaseT:
        return self.cases[index]

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
    "CaseT",
]
