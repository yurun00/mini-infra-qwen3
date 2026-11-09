"""评测套件通用抽象基类（BenchmarkSuite）。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from mini_infra.bench.spec.suite.expand import CaseSpec
from mini_infra.bench.spec.suite.loader import LoadedSuiteSpec
from mini_infra.engine import Mode

CaseT = TypeVar("CaseT")


@dataclass(frozen=True)
class BenchmarkSuite(ABC, Generic[CaseT]):
    """评测套件通用抽象基类：持有已校验加载的 LoadedSuiteSpec，提供标准元数据访问与懒绑定 cases 接口。"""

    loaded: LoadedSuiteSpec

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

    @abstractmethod
    def cases(self, vocab_size: int) -> list[CaseT]:
        """将用例规格绑定到指定 vocab_size 的合成 Token 载荷，返回可执行用例列表。"""
        ...


__all__ = [
    "BenchmarkSuite",
    "CaseT",
]
