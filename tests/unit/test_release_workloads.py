"""release 标准负载 YAML 与旧手写 `make_*_workload` 的 parity，以及 log_uniform / poisson 分布的确定性（纯 CPU）。"""

import json
import random
from pathlib import Path
from typing import Any

import pytest

from mini_infra.bench.spec import get_release_spec
from mini_infra.bench.spec.common import sample_log_uniform_lengths, sample_poisson_arrivals
from mini_infra.bench.spec.release import release_default_mode, release_workloads
from mini_infra.bench.workloads import WORKLOAD_SCALES, WorkloadSpec

GOLDEN = json.loads((Path(__file__).parent / "data" / "release_workloads_golden.json").read_text(encoding="utf-8"))


def _encode(w: WorkloadSpec) -> dict[str, Any]:
    def as_list(x: Any) -> Any:
        return list(x) if isinstance(x, tuple) else x

    return {
        "mode": str(w.mode),
        "num_total_requests": w.num_total_requests,
        "prompt_len": as_list(w.prompt_len),
        "max_new_tokens": as_list(w.max_new_tokens),
        "seed": w.seed,
        "scale": w.scale,
        "max_running_requests": w.max_running_requests,
        "arrival_times_ms": as_list(w.arrival_times_ms),
        "warmup": w.warmup,
        "repeats": w.repeats,
    }


@pytest.mark.parametrize("engine", ["lockstep", "scheduler"])
@pytest.mark.parametrize("seed", [42, 7])
def test_release_yaml_matches_legacy_golden(engine, seed):
    actual = {scale: _encode(w) for scale, w in release_workloads(engine, seed=seed).items()}
    assert actual == GOLDEN[engine][str(seed)]


@pytest.mark.parametrize("engine", ["lockstep", "scheduler"])
def test_default_seed_is_the_standard_seed(engine):
    assert get_release_spec(engine).spec.default_seed == GOLDEN["default_seed"] == 42
    default = {s: _encode(w) for s, w in release_workloads(engine).items()}
    assert default == GOLDEN[engine]["42"]
    assert set(default) == set(WORKLOAD_SCALES)


def test_release_default_modes():
    assert str(release_default_mode("lockstep")) == "dense-kv-cache+fixed-batching"
    assert str(release_default_mode("scheduler")) == "dense-kv-cache+continuous-batching"


def test_release_seed_changes_workload():
    base = release_workloads("scheduler", seed=42)["medium"]
    other = release_workloads("scheduler", seed=43)["medium"]
    assert base.prompt_len != other.prompt_len
    assert base.arrival_times_ms != other.arrival_times_ms


def test_release_mode_override_keeps_workload_identical():
    from mini_infra.engine import BatchingMethod, CachingMethod, Mode

    paged = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
    a = release_workloads("scheduler")["small"]
    b = release_workloads("scheduler", mode=paged)["small"]
    assert b.mode == paged
    assert (a.prompt_len, a.max_new_tokens, a.arrival_times_ms) == (b.prompt_len, b.max_new_tokens, b.arrival_times_ms)
    # WorkloadSpec 相等比较忽略 repeats / warmup / seed，但包含 mode
    assert a != b


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
