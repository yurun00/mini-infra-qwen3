"""Scheduler 评测套件：由 `benchmarks/specs/scheduler/*.yaml` 声明。

套件的全部配置（被测系统、请求负载、sweep、comparison）都在 YAML 中；新增 suite 只需新增一个 YAML 文件：

    benchmarks/specs/scheduler/<name>.yaml        # name 必须与文件名一致
    benchmarks/specs/scheduler/_bases/*.yaml      # 仅供 extends 复用，不可直接运行
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mini_infra.bench.spec import (
    LoadedSuiteSpec,
    discover_suite_specs,
    get_suite_spec,
    load_suite_spec,
)
from mini_infra.bench.suite import BenchmarkSuite


def _suite_from_loaded(loaded: LoadedSuiteSpec) -> BenchmarkSuite:
    return BenchmarkSuite(loaded=loaded)


def load_scheduler_suite(path: Path, overrides: Mapping[str, Any] | None = None) -> BenchmarkSuite:
    """从任意 YAML 路径加载 Scheduler 套件。"""
    loaded = load_suite_spec(path, overrides)
    if loaded.spec.engine != "scheduler":
        raise ValueError(f"{path}: engine 必须是 scheduler，实际为 {loaded.spec.engine}")
    return _suite_from_loaded(loaded)


def list_scheduler_suite_names() -> list[str]:
    return sorted(discover_suite_specs("scheduler"))


def get_scheduler_suite(name: str, overrides: Mapping[str, Any] | None = None) -> BenchmarkSuite:
    return _suite_from_loaded(get_suite_spec("scheduler", name, overrides=overrides))


def get_all_scheduler_suites() -> Sequence[BenchmarkSuite]:
    return [get_scheduler_suite(name) for name in list_scheduler_suite_names()]


__all__ = [
    "get_all_scheduler_suites",
    "get_scheduler_suite",
    "list_scheduler_suite_names",
    "load_scheduler_suite",
]
