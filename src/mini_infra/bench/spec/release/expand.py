"""Release 标准负载展开：把 ReleaseSpec 展开为确定性负载与运行期规格（纯 CPU）。"""

from dataclasses import dataclass

from mini_infra.bench.spec.common.checks import run_check
from mini_infra.bench.spec.common.schema import SchedulerEngineSpec
from mini_infra.bench.spec.common.workload import ResolvedWorkloadSpec, resolve_workload_body
from mini_infra.bench.spec.release.schema import RELEASE_SCALES, ReleaseScale, ReleaseSpec
from mini_infra.engine import Mode


@dataclass(frozen=True)
class ResolvedReleaseScale:
    """一档展开后的 release 负载。"""

    scale: ReleaseScale
    seed: int
    max_running_requests: int | None
    workload: ResolvedWorkloadSpec


def resolve_release_spec(spec: ReleaseSpec, seed: int | None = None) -> dict[ReleaseScale, ResolvedReleaseScale]:
    """展开三档负载；seed 缺省取 default_seed。"""
    effective_seed = spec.default_seed if seed is None else seed
    resolved: dict[ReleaseScale, ResolvedReleaseScale] = {}
    base_mrr = (
        spec.system.engine_mode.max_running_requests
        if isinstance(spec.system.engine_mode, SchedulerEngineSpec)
        else None
    )
    for scale in RELEASE_SCALES:
        scale_spec = spec.scales[scale]
        effective_mrr = scale_spec.max_running_requests or base_mrr
        body = {**scale_spec.workload, "seed": effective_seed, "seed_salt": scale_spec.seed_salt}
        workload = resolve_workload_body(
            f"{spec.engine}_{scale}", body, {}, context=f"release {spec.engine}/{scale} (seed={effective_seed})"
        )
        for check in spec.checks:
            run_check(
                check,
                context=f"release {spec.engine}/{scale}",
                max_running_requests=effective_mrr,
                output_budgets=workload.output_budgets,
                arrival_ms=workload.arrival_ms,
            )
        resolved[scale] = ResolvedReleaseScale(
            scale=scale,
            seed=effective_seed,
            max_running_requests=effective_mrr,
            workload=workload,
        )
    return resolved


def release_spec_default_mode(spec: ReleaseSpec) -> Mode:
    """Release YAML 声明的默认运行模式。"""
    return Mode(spec.system.caching_method.method, batching_method=spec.system.batching_method)


def expand_release_scales(spec: ReleaseSpec, *, seed: int | None = None) -> dict[ReleaseScale, ResolvedWorkloadSpec]:
    """把 ReleaseSpec 转换为三档运行时规格 ResolvedWorkloadSpec；seed 缺省取 default_seed。"""
    scales = resolve_release_spec(spec, seed=seed)
    return {scale: resolved.workload for scale, resolved in scales.items()}


__all__ = [
    "ResolvedReleaseScale",
    "expand_release_scales",
    "release_spec_default_mode",
    "resolve_release_spec",
]
