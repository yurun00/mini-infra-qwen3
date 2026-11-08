"""Suite YAML 加载：安全解析（拒绝重复 key）、extends 深合并、校验、发现与内容哈希。"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mini_infra.bench.common import SPECS_DIR
from mini_infra.bench.spec.expand import CaseBlueprint, expand_suite_spec
from mini_infra.bench.spec.schema import SuiteEngine, SuiteSpec

_BASES_DIRNAME = "_bases"


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader 的变体：同一映射内出现重复 key 时报错（PyYAML 默认静默取最后一个）。"""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(None, None, f"映射中出现重复 key: {key!r}", key_node.start_mark)
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.load(f, Loader=_UniqueKeyLoader)  # noqa: S506 - 自定义 SafeLoader 子类
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 顶层必须是映射，实际为 {type(data).__name__}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """映射递归合并；非映射值（含 list）由 override 整体替换。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_raw_spec(path: Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """读取 YAML 并递归解析 extends（相对当前文件目录），返回合并后的原始映射。"""
    resolved = path.resolve()
    if resolved in _stack:
        chain = " -> ".join(p.name for p in (*_stack, resolved))
        raise ValueError(f"extends 出现循环: {chain}")
    data = _read_yaml_mapping(resolved)
    parent = data.pop("extends", None)
    if parent is None:
        return data
    if not isinstance(parent, str):
        raise ValueError(f"{resolved}: extends 必须是相对路径字符串")
    return deep_merge(load_raw_spec(resolved.parent / parent, (*_stack, resolved)), data)


@dataclass(frozen=True)
class LoadedSuiteSpec:
    """已校验的 suite 声明及其来源信息。"""

    spec: SuiteSpec
    path: Path
    raw: dict[str, Any]
    sha256: str

    def blueprints(self) -> list[CaseBlueprint]:
        return expand_suite_spec(self.spec)

    def info(self) -> dict[str, Any]:
        """写入结果 JSON `suite` 块的可复现描述：合并后的完整声明与其内容哈希。"""
        return {
            "name": self.spec.name,
            "engine": self.spec.engine,
            "spec_sha256": self.sha256,
            "spec": self.raw,
        }


def load_suite_spec(path: Path) -> LoadedSuiteSpec:
    """加载并校验一份 suite YAML；同时试展开全部 sweep 点，使非法负载在加载期暴露。"""
    raw = load_raw_spec(path)
    spec = SuiteSpec.model_validate(raw)
    if path.stem != spec.name:
        raise ValueError(f"{path}: 文件名 {path.stem!r} 必须与 name {spec.name!r} 一致")
    expand_suite_spec(spec)
    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return LoadedSuiteSpec(spec=spec, path=path, raw=raw, sha256=hashlib.sha256(canonical.encode()).hexdigest())


def discover_suite_specs(engine: SuiteEngine, specs_dir: Path | None = None) -> dict[str, Path]:
    """发现 `<specs_dir>/<engine>/*.yaml`（不递归，因此 `_bases/` 中的继承基底不会被当作可运行 suite）。"""
    root = (specs_dir if specs_dir is not None else SPECS_DIR) / engine
    return {path.stem: path for path in sorted(root.glob("*.yaml"))}


def get_suite_spec(engine: SuiteEngine, name: str, specs_dir: Path | None = None) -> LoadedSuiteSpec:
    """按名称加载内置 suite。"""
    available = discover_suite_specs(engine, specs_dir)
    if name not in available:
        raise ValueError(f"未知 {engine} suite: {name}，可选: {sorted(available)}")
    return load_suite_spec(available[name])


__all__ = [
    "LoadedSuiteSpec",
    "deep_merge",
    "discover_suite_specs",
    "get_suite_spec",
    "load_raw_spec",
    "load_suite_spec",
]
