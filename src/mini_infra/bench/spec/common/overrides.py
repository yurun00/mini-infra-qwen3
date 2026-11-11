"""命令行覆盖：`--set a.b.c=value`（value 按 YAML 解析）作用于 extends 合并后的原始声明。"""

import copy
from collections.abc import Iterable, Mapping
from typing import Any

import yaml


def parse_set_args(raw: Iterable[str]) -> dict[str, Any]:
    """把 ['execution.repeats=5', 'sweep.workload.n=[1,2]'] 解析为 {点分路径: YAML 值}。"""
    overrides: dict[str, Any] = {}
    for item in raw:
        path, sep, text = item.partition("=")
        path = path.strip()
        if not sep or not path or any(not part for part in path.split(".")):
            raise ValueError(f"--set 需要 key.path=value 形式，实际为 {item!r}")
        try:
            overrides[path] = yaml.safe_load(text)
        except yaml.YAMLError as err:
            raise ValueError(f"--set {path}: 值 {text!r} 不是合法 YAML: {err}") from err
    return overrides


def apply_overrides(raw: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """返回应用覆盖后的深拷贝：沿点分路径逐层下钻（缺失的中间层自动创建），末端整体替换。"""
    result = copy.deepcopy(dict(raw))
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        node: dict[str, Any] = result
        for depth, key in enumerate(parents):
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                where = ".".join(parents[: depth + 1])
                raise ValueError(f"--set {path}: {where} 不是映射（实际为 {type(child).__name__}），无法继续下钻")
            node = child
        node[leaf] = copy.deepcopy(value)
    return result


__all__ = ["apply_overrides", "parse_set_args"]
