"""Release 标准负载展开：把 ReleaseSpec 展开为确定性负载与运行期 Workload（纯 CPU）。"""

from dataclasses import dataclass

from mini_infra.bench.spec.common.workload import ResolvedWorkload, resolve_workload_body
from mini_infra.bench.spec.release.schema import RELEASE_SCALES, ReleaseScale, ReleaseSpec
from mini_infra.bench.workloads import Workload, WorkloadScale
from mini_infra.engine import Mode


@dataclass(frozen=True)
class ResolvedReleaseScale:
    """一档展开后的 release 负载。"""

    scale: ReleaseScale
    seed: int
    max_running_requests: int | None
    workload: ResolvedWorkload


def resolve_release_spec(spec: ReleaseSpec, seed: int | None = None) -> dict[ReleaseScale, ResolvedReleaseScale]:
    """展开三档负载；seed 缺省取 default_seed。"""
    effective_seed = spec.default_seed if seed is None else seed
    resolved: dict[ReleaseScale, ResolvedReleaseScale] = {}
    for scale in RELEASE_SCALES:
        scale_spec = spec.scales[scale]
        body = {**scale_spec.workload, "seed": effective_seed, "seed_salt": scale_spec.seed_salt}
        workload = resolve_workload_body(
            f"{spec.engine}_{scale}", body, {}, context=f"release {spec.engine}/{scale} (seed={effective_seed})"
        )
        resolved[scale] = ResolvedReleaseScale(
            scale=scale,
            seed=effective_seed,
            max_running_requests=scale_spec.max_running_requests,
            workload=workload,
        )
    return resolved


def release_spec_default_mode(spec: ReleaseSpec) -> Mode:
    """Release YAML 声明的默认运行模式。"""
    default = spec.default_mode
    return Mode(default.caching_method, batching_method=default.batching_method)


def expand_release_workloads(
    spec: ReleaseSpec, *, seed: int | None = None, mode: Mode | None = None
) -> dict[WorkloadScale, Workload]:
    """把 ReleaseSpec 转换为三档运行时 Workload；seed 缺省取 default_seed，mode 缺省取 default_mode。"""
    target_mode = mode or release_spec_default_mode(spec)
    scales = resolve_release_spec(spec, seed=seed)
    workloads: dict[WorkloadScale, Workload] = {}
    for scale, resolved in scales.items():
        workloads[scale] = Workload(
            mode=target_mode,
            num_total_requests=resolved.workload.num_requests,
            prompt_len=resolved.workload.prompt_len,
            max_new_tokens=resolved.workload.max_new_tokens,
            repeats=spec.execution.repeats,
            warmup=spec.execution.warmup,
            seed=resolved.seed,
            scale=scale,
            max_running_requests=resolved.max_running_requests,
            arrival_times_ms=resolved.workload.arrival_ms if spec.engine == "scheduler" else None,
        )
    return workloads


__all__ = [
    "ResolvedReleaseScale",
    "expand_release_workloads",
    "release_spec_default_mode",
    "resolve_release_spec",
]
