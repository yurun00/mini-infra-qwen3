"""统一评测套件加载、发现与查询服务（bench.suites）。

消除引擎分立的加载器，统一由 BenchmarkSuite 加载 YAML 规格并校验绑定。
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mini_infra.bench.spec import (
    LoadedSuiteSpec,
    SuiteEngine,
    discover_suite_specs,
    get_suite_spec,
    load_suite_spec,
)
from mini_infra.bench.suite import BenchmarkSuite


def suite_from_loaded(loaded: LoadedSuiteSpec) -> BenchmarkSuite:
    """从已解析的 LoadedSuiteSpec 构造不可变 BenchmarkSuite 实体。"""
    return BenchmarkSuite.from_spec(loaded)


def load_benchmark_suite(
    path: Path,
    overrides: Mapping[str, Any] | None = None,
    expected_engine: SuiteEngine | None = None,
) -> BenchmarkSuite:
    """从任意 YAML 路径加载 BenchmarkSuite。若提供 expected_engine 则校验引擎一致性。"""
    loaded = load_suite_spec(path, overrides)
    if expected_engine is not None and loaded.spec.engine != expected_engine:
        raise ValueError(f"{path}: engine 必须是 {expected_engine}，实际为 {loaded.spec.engine}")
    return suite_from_loaded(loaded)


def list_benchmark_suite_names(engine: SuiteEngine) -> list[str]:
    """列出指定引擎内置的所有 suite 名称。"""
    return sorted(discover_suite_specs(engine))


def get_benchmark_suite(
    engine: SuiteEngine,
    name: str,
    overrides: Mapping[str, Any] | None = None,
) -> BenchmarkSuite:
    """获取指定引擎内置的命名套件。"""
    return suite_from_loaded(get_suite_spec(engine, name, overrides=overrides))


def get_all_benchmark_suites(engine: SuiteEngine) -> Sequence[BenchmarkSuite]:
    """获取指定引擎内置的全部套件。"""
    return [get_benchmark_suite(engine, name) for name in list_benchmark_suite_names(engine)]


__all__ = [
    "get_all_benchmark_suites",
    "get_benchmark_suite",
    "list_benchmark_suite_names",
    "load_benchmark_suite",
    "suite_from_loaded",
]
