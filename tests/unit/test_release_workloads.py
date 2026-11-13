"""release 标准负载 YAML 与旧手写 `make_*_workload` 的 parity，以及 log_uniform / poisson 分布的确定性（纯 CPU）。"""

import json
import random
from pathlib import Path
from typing import Any

import pytest

from mini_infra.bench.spec import RELEASE_SCALES, ResolvedWorkloadSpec, get_release_spec
from mini_infra.bench.spec.common import sample_log_uniform_lengths, sample_poisson_arrivals
from mini_infra.bench.spec.release import release_workloads

GOLDEN = json.loads((Path(__file__).parent / "data" / "release_workloads_golden.json").read_text(encoding="utf-8"))


def _encode(w: ResolvedWorkloadSpec, *, engine: str, scale: str, seed: int) -> dict[str, Any]:
    def as_list(x: Any) -> Any:
        return list(x) if isinstance(x, tuple) else x

    loaded = get_release_spec(engine)
    resolved = loaded.resolve(seed)[scale]
    return {
        "engine_config": str(loaded.default_engine_config),
        "num_total_requests": w.num_total_requests,
        "prompt_len": as_list(w.prompt_lens),
        "max_new_tokens": as_list(w.output_budgets),
        "seed": seed,
        "scale": scale,
        "max_running_requests": resolved.max_running_requests,
        "arrival_times_ms": as_list(w.arrival_ms) if engine == "scheduler" else None,
        "warmup": loaded.spec.execution.warmup,
        "repeats": loaded.spec.execution.repeats,
    }


@pytest.mark.parametrize("engine", ["lockstep", "scheduler"])
@pytest.mark.parametrize("seed", [42, 7])
def test_release_yaml_matches_legacy_golden(engine, seed):
    actual = {
        scale: _encode(w, engine=engine, scale=scale, seed=seed)
        for scale, w in release_workloads(engine, seed=seed).items()
    }
    assert actual == GOLDEN[engine][str(seed)]


@pytest.mark.parametrize("engine", ["lockstep", "scheduler"])
def test_default_seed_is_the_standard_seed(engine):
    assert get_release_spec(engine).spec.default_seed == GOLDEN["default_seed"] == 42
    default = {s: _encode(w, engine=engine, scale=s, seed=42) for s, w in release_workloads(engine).items()}
    assert default == GOLDEN[engine]["42"]
    assert set(default) == set(RELEASE_SCALES)


def test_release_default_modes():
    assert str(get_release_spec("lockstep").default_engine_config) == "dense-kv-cache+fixed-batching"
    assert str(get_release_spec("scheduler").default_engine_config) == "dense-kv-cache+continuous-batching"


def test_release_seed_changes_workload():
    base = release_workloads("scheduler", seed=42)["medium"]
    other = release_workloads("scheduler", seed=43)["medium"]
    assert base.prompt_lens != other.prompt_lens
    assert base.arrival_ms != other.arrival_ms


def test_release_workloads_pure_payload():
    """验证 release_workloads 返回的 ResolvedWorkloadSpec 仅包含流量载荷，不含模式或执行参数。"""
    w = release_workloads("scheduler")["small"]
    assert not hasattr(w, "mode")
    assert not hasattr(w, "repeats")
    assert not hasattr(w, "warmup")
    assert not hasattr(w, "scale")
    assert w.workload_id == "scheduler_small"


def test_samplers_are_deterministic_and_within_bounds():
    lens = sample_log_uniform_lengths(random.Random(1), 32, 16, 64)
    assert lens == sample_log_uniform_lengths(random.Random(1), 32, 16, 64)
    assert len(lens) == 32 and all(16 <= x <= 64 for x in lens)
    arrivals = sample_poisson_arrivals(random.Random(1), 10, 3, 50.0)
    assert arrivals == sample_poisson_arrivals(random.Random(1), 10, 3, 50.0)
    assert arrivals[:3] == (0.0, 0.0, 0.0)
    assert list(arrivals) == sorted(arrivals)
    gaps = [b - a for a, b in zip(arrivals[3:], arrivals[4:])]
    assert all(g >= 0 for g in gaps)
    # 突发数超过请求数时全部同时到达
    assert sample_poisson_arrivals(random.Random(1), 3, 10, 50.0) == (0.0, 0.0, 0.0)
