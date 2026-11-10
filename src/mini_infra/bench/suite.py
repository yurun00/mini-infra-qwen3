"""评测套件通用容器（BenchmarkSuite）。"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from mini_infra.bench.cases import (
    BenchmarkCase,
    CaseDimension,
    Comparison,
)
from mini_infra.bench.spec.common.schema import SuiteEngine
from mini_infra.bench.spec.suite.loader import LoadedSuiteSpec
from mini_infra.engine import Mode


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

    def modes(self) -> list[Mode]:
        """按用例展开顺序去重后的运行模式列表。"""
        return list(dict.fromkeys(c.mode for c in self.cases))

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
            mrr = cs.workload.num_total_requests if engine == "lockstep" else cs.max_running_requests
            assert mrr is not None
            paged_config = None if engine == "lockstep" else cs.paged_cache_config
            cases.append(
                BenchmarkCase(
                    profile=cs.workload.to_profile(),
                    mode=cs.mode,
                    max_running_requests=mrr,
                    paged_cache_config=paged_config,
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


__all__ = [
    "BenchmarkSuite",
]
