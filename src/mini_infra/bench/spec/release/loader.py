"""Release 标准负载的加载与发现。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_infra.bench.common import SPECS_DIR
from mini_infra.bench.spec.common.raw import _sha256, load_raw_spec
from mini_infra.bench.spec.common.schema import SuiteEngine
from mini_infra.bench.spec.release.expand import (
    ResolvedReleaseScale,
    expand_release_workloads,
    release_spec_default_mode,
    resolve_release_spec,
)
from mini_infra.bench.spec.release.schema import ReleaseScale, ReleaseSpec
from mini_infra.bench.workloads import Workload, WorkloadScale
from mini_infra.engine import Mode


@dataclass(frozen=True)
class LoadedReleaseSpec:
    """已校验并已展开的 release 标准负载声明：`scales` 为默认 seed 下三档展开后的负载。"""

    spec: ReleaseSpec
    path: Path
    raw: dict[str, Any]
    sha256: str
    scales: dict[ReleaseScale, ResolvedReleaseScale]

    def resolve(self, seed: int | None = None) -> dict[ReleaseScale, ResolvedReleaseScale]:
        """以指定 seed 重新展开三档负载（seed 缺省即 `scales`，取 default_seed）。"""
        return self.scales if seed is None else resolve_release_spec(self.spec, seed)

    def to_workloads(self, *, seed: int | None = None) -> dict[WorkloadScale, Workload]:
        """转换为运行期三档 Workload。"""
        return expand_release_workloads(self.spec, seed=seed)


def load_release_spec(path: Path) -> LoadedReleaseSpec:
    """加载、校验并展开一份 release YAML（默认 seed）；非法负载在加载期暴露。"""
    raw = load_raw_spec(path)
    spec = ReleaseSpec.model_validate(raw)
    if path.stem != spec.engine:
        raise ValueError(f"{path}: 文件名 {path.stem!r} 必须与 engine {spec.engine!r} 一致")
    return LoadedReleaseSpec(
        spec=spec,
        path=path,
        raw=raw,
        sha256=_sha256(raw),
        scales=resolve_release_spec(spec),
    )


def get_release_spec(engine: SuiteEngine, specs_dir: Path | None = None) -> LoadedReleaseSpec:
    """按引擎加载内置 release 标准负载 `<specs_dir>/release/<engine>.yaml`。"""
    path = (specs_dir if specs_dir is not None else SPECS_DIR) / "release" / f"{engine}.yaml"
    if not path.exists():
        raise ValueError(f"未找到 {engine} 的 release 负载声明: {path}")
    return load_release_spec(path)


def release_default_mode(engine: SuiteEngine, specs_dir: Path | None = None) -> Mode:
    """release YAML 声明的默认运行模式（`bench release` 可通过命令行覆盖）。"""
    return release_spec_default_mode(get_release_spec(engine, specs_dir=specs_dir).spec)


def release_workloads(
    engine: SuiteEngine,
    *,
    seed: int | None = None,
    specs_dir: Path | None = None,
) -> dict[WorkloadScale, Workload]:
    """三档标准负载（small / medium / large）；seed 缺省取 YAML 的 default_seed。"""
    loaded = get_release_spec(engine, specs_dir=specs_dir)
    return loaded.to_workloads(seed=seed)


__all__ = [
    "LoadedReleaseSpec",
    "get_release_spec",
    "load_release_spec",
    "release_default_mode",
    "release_workloads",
]
