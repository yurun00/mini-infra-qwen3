"""Suite YAML 加载：安全解析、extends 深合并、校验、发现与内容哈希。"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mini_infra.bench.common import SPECS_DIR
from mini_infra.bench.spec.common.overrides import apply_overrides
from mini_infra.bench.spec.common.raw import load_raw_spec, spec_sha256
from mini_infra.bench.spec.suite.expand import CaseSpec, expand_suite_spec
from mini_infra.bench.spec.suite.schema import SuiteEngine, SuiteSpec


@dataclass(frozen=True)
class LoadedSuiteSpec:
    """已校验并已展开的 suite 声明：`cases` 为全部 sweep 点展开后的有序用例规格。"""

    spec: SuiteSpec
    path: Path
    raw: dict[str, Any]
    sha256: str
    cases: tuple[CaseSpec, ...]
    overrides: dict[str, Any] = field(default_factory=dict)

    def info(self) -> dict[str, Any]:
        """写入结果 JSON `suite` 块的可复现描述：合并后的完整声明与其内容哈希。"""
        info: dict[str, Any] = {
            "name": self.spec.name,
            "engine": self.spec.engine,
            "spec_sha256": self.sha256,
            "spec": self.raw,
        }
        if self.overrides:
            info["overrides"] = self.overrides
        return info


def load_suite_spec(path: Path, overrides: Mapping[str, Any] | None = None) -> LoadedSuiteSpec:
    """加载、校验并展开一份 suite YAML；非法负载在加载期暴露。

    overrides（点分路径 → 值）作用于 extends 合并后的原始声明，`spec_sha256` 基于覆盖后的声明计算。
    """
    raw = load_raw_spec(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    spec = SuiteSpec.model_validate(raw)
    if path.stem != spec.name:
        raise ValueError(f"{path}: 文件名 {path.stem!r} 必须与 name {spec.name!r} 一致")
    return LoadedSuiteSpec(
        spec=spec,
        path=path,
        raw=raw,
        sha256=spec_sha256(raw),
        cases=tuple(expand_suite_spec(spec)),
        overrides=dict(overrides or {}),
    )


def discover_suite_specs(engine: SuiteEngine, specs_dir: Path | None = None) -> dict[str, Path]:
    """发现 `<specs_dir>/<engine>/*.yaml`（不递归，因此 `_bases/` 中的继承基底不会被当作可运行 suite）。"""
    root = (specs_dir if specs_dir is not None else SPECS_DIR) / engine
    return {path.stem: path for path in sorted(root.glob("*.yaml"))}


def get_suite_spec(
    engine: SuiteEngine,
    name: str,
    specs_dir: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> LoadedSuiteSpec:
    """按名称加载内置 suite。"""
    available = discover_suite_specs(engine, specs_dir)
    if name not in available:
        raise ValueError(f"未知 {engine} suite: {name}，可选: {sorted(available)}")
    loaded = load_suite_spec(available[name], overrides)
    if loaded.spec.engine != engine:
        raise ValueError(f"{loaded.path}: engine 为 {loaded.spec.engine}，与所在目录 {engine} 不一致")
    return loaded


__all__ = [
    "LoadedSuiteSpec",
    "discover_suite_specs",
    "get_suite_spec",
    "load_suite_spec",
]
