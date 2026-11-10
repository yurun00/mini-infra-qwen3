"""bench.spec：YAML suite 声明的加载、校验、展开，以及与旧手写 suite 的 parity 快照对照（纯 CPU）。"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mini_infra.bench.common import SPECS_DIR
from mini_infra.bench.lockstep.suites import get_lockstep_suite
from mini_infra.bench.scheduler.suites import get_scheduler_suite
from mini_infra.bench.spec import (
    deep_merge,
    discover_suite_specs,
    expand_suite_spec,
    get_suite_spec,
    load_raw_spec,
    load_suite_spec,
)
from mini_infra.engine import BatchingMethod, CachingMethod

GOLDEN = json.loads((Path(__file__).parent / "data" / "suite_cases_golden.json").read_text(encoding="utf-8"))
VOCAB = GOLDEN["vocab_size"]


def _digest(token_ids: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()[:16]


def _scheduler_entry(name: str) -> dict[str, Any]:
    suite = get_scheduler_suite(name)
    cases = [
        {
            "workload_id": c.workload_id,
            "mode": str(c.mode),
            "max_running_requests": c.max_running_requests,
            "paged_cache_config": (
                None
                if c.paged_cache_config is None
                else {"block_size": c.paged_cache_config.block_size, "num_blocks": c.paged_cache_config.num_blocks}
            ),
            "specs": [
                [s.request_idx, _digest(s.token_ids), len(s.token_ids), s.max_new_tokens, s.arrival_ms]
                for s in c.profile.to_payload(VOCAB)
            ],
        }
        for c in suite.cases
    ]
    assert suite.comparison is not None
    return {
        "warmup": suite.warmup,
        "repeats": suite.repeats,
        "max_running_requests": suite.loaded.spec.system.max_running_requests,
        "interleave": [d.value for d in suite.interleave],
        "modes": [str(m) for m in suite.modes()],
        "comparison": suite.comparison.to_dict(),
        "cases": cases,
    }


def _lockstep_entry(name: str) -> dict[str, Any]:
    suite = get_lockstep_suite(name)
    return {
        "warmup": suite.warmup,
        "repeats": suite.repeats,
        "modes": [str(m) for m in suite.modes()],
        "cases": [
            {
                "mode": str(cs.mode),
                "num_total_requests": cs.workload.num_total_requests,
                "prompt_len": (
                    list(cs.workload.prompt_len)
                    if isinstance(cs.workload.prompt_len, tuple)
                    else cs.workload.prompt_len
                ),
                "max_new_tokens": (
                    list(cs.workload.max_new_tokens)
                    if isinstance(cs.workload.max_new_tokens, tuple)
                    else cs.workload.max_new_tokens
                ),
            }
            for cs in suite.case_specs
        ],
    }


def test_builtin_specs_cover_golden_suites():
    assert set(discover_suite_specs("scheduler")) == set(GOLDEN["scheduler"])
    assert set(discover_suite_specs("lockstep")) == set(GOLDEN["lockstep"])


@pytest.mark.parametrize("name", sorted(GOLDEN["scheduler"]))
def test_scheduler_yaml_matches_legacy_golden(name):
    expected = GOLDEN["scheduler"][name]
    actual = _scheduler_entry(name)
    expected_cmp = expected.pop("comparison")
    actual_cmp = actual.pop("comparison")
    assert actual == expected
    assert actual_cmp == expected_cmp


@pytest.mark.parametrize("name", sorted(GOLDEN["lockstep"]))
def test_lockstep_yaml_matches_legacy_golden(name):
    assert _lockstep_entry(name) == GOLDEN["lockstep"][name]


def test_bases_are_not_discoverable_or_loadable_as_suites():
    assert "fixed_vs_continuous" not in discover_suite_specs("scheduler")
    with pytest.raises(ValueError):
        load_suite_spec(SPECS_DIR / "scheduler" / "_bases" / "fixed_vs_continuous.yaml")


def test_get_suite_spec_unknown_name():
    with pytest.raises(ValueError, match="未知 scheduler suite"):
        get_suite_spec("scheduler", "non-existent")


def test_suite_info_is_json_serializable_and_hash_is_stable():
    loaded = get_suite_spec("scheduler", "dense-vs-paged-block-size")
    info = loaded.info()
    json.dumps(info)
    assert info["name"] == "dense-vs-paged-block-size"
    assert info["spec_sha256"] == get_suite_spec("scheduler", "dense-vs-paged-block-size").sha256
    assert info["spec"]["sweep"] == {"system": {"block_size": [8, 16, 32, 64]}}


# --------------------------------------------------------------------------- 合成 YAML 的行为测试

_SCHEDULER_BASE = """
name: demo
engine: scheduler
system:
  max_running_requests: 4
  batching_method: [continuous-batching]
  caching_method: [dense-kv-cache, paged-kv-cache]
workload:
  id: w
  count: 2
  prompt_len: 8
  output_len: 4
"""


def _write(tmp_path: Path, text: str, name: str = "demo") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_block_size_sweep_only_expands_paged_modes(tmp_path):
    text = _SCHEDULER_BASE + "sweep:\n  system:\n    block_size: [4, 8]\n"
    cases = load_suite_spec(_write(tmp_path, text)).cases
    assert [(cs.mode.caching_method, cs.paged_cache_config and cs.paged_cache_config.block_size) for cs in cases] == [
        (CachingMethod.DENSE, None),
        (CachingMethod.PAGED, 4),
        (CachingMethod.PAGED, 8),
    ]


def test_workload_cartesian_sweep_and_type_preserving_interpolation(tmp_path):
    text = """
name: demo
engine: scheduler
system:
  max_running_requests: 4
  batching_method: [fixed-batching]
  caching_method: [dense-kv-cache]
workload:
  vars: {n: 1, plen: 8}
  id: "w_${n}_${plen}"
  count: "${n}"
  prompt_len: "${plen}"
  output_len: 4
sweep:
  workload:
    n: [1, 2]
    plen: [8, 16]
"""
    case_specs = expand_suite_spec(load_suite_spec(_write(tmp_path, text)).spec)
    assert [
        (cs.workload.workload_id, cs.workload.num_total_requests, cs.workload.prompt_lens[0]) for cs in case_specs
    ] == [
        ("w_1_8", 1, 8),
        ("w_1_16", 1, 16),
        ("w_2_8", 2, 8),
        ("w_2_16", 2, 16),
    ]


def test_scalar_length_is_broadcast_and_list_is_per_request(tmp_path):
    text = """
name: demo
engine: lockstep
system:
  batching_method: [fixed-batching]
  caching_method: [dense-kv-cache]
workload:
  id: w
  count: 2
  prompt_len: [8, 8]
  output_len: 4
"""
    (cs,) = load_suite_spec(_write(tmp_path, text)).cases
    assert cs.workload.prompt_len == (8, 8)
    assert cs.workload.max_new_tokens == 4


def test_extends_deep_merges_mappings_and_replaces_lists(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "system:\n  max_running_requests: 4\n  batching_method: [fixed-batching, continuous-batching]\n"
        "  caching_method: [dense-kv-cache]\nengine: scheduler\n",
        encoding="utf-8",
    )
    child = _write(
        tmp_path,
        "name: demo\nextends: base.yaml\nsystem:\n  batching_method: [continuous-batching]\n"
        "workload: {id: w, count: 1, prompt_len: 8, output_len: 4}\n",
    )
    raw = load_raw_spec(child)
    assert raw["system"] == {
        "max_running_requests": 4,
        "batching_method": ["continuous-batching"],
        "caching_method": ["dense-kv-cache"],
    }
    assert "extends" not in raw
    assert load_suite_spec(child).spec.system.batching_method == [BatchingMethod.CONTINUOUS]


def test_extends_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="循环"):
        load_raw_spec(tmp_path / "a.yaml")


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}, "c": [1]}
    override = {"a": {"d": 2}, "c": [2]}
    assert deep_merge(base, override) == {"a": {"b": 1, "d": 2}, "c": [2]}
    assert base == {"a": {"b": 1}, "c": [1]}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda t: t.replace("count: 2", "count: 2\n  count: 3"), "重复 key"),
        (lambda t: t.replace("name: demo", "name: Demo_X"), "只能包含"),
        (lambda t: t + "unknown_field: 1\n", "Extra inputs"),
        (lambda t: t.replace("prompt_len: 8", "prompt_len: 0"), "greater than 0"),
        (lambda t: t.replace("prompt_len: 8", "prompt_len: true"), "布尔"),
        (lambda t: t.replace("prompt_len: 8", "prompt_len: [8]"), "必须等于 count"),
        (lambda t: t.replace("prompt_len: 8", "prompt_len: {ladder: {max: 8}}"), "count == 4"),
        (lambda t: t.replace("  max_running_requests: 4\n", ""), "max_running_requests"),
        (lambda t: t + "sweep:\n  workload:\n    nope: [1]\n", "未在 workload.vars 中声明"),
        (lambda t: t + "sweep:\n  system:\n    block_size: []\n", "at least 1"),
        (
            lambda t: (
                t.replace("[dense-kv-cache, paged-kv-cache]", "[dense-kv-cache]")
                + "sweep:\n  system:\n    block_size: [8]\n"
            ),
            "block_size",
        ),
        (lambda t: t.replace("output_len: 4", "output_len: ${missing}"), "未定义的变量"),
        (lambda t: t + "checks:\n  - min_max_running_requests: 8\n", "要求 max_running_requests >= 8"),
    ],
)
def test_invalid_scheduler_specs_are_rejected_at_load(tmp_path, mutation, message):
    with pytest.raises(Exception, match=message):
        load_suite_spec(_write(tmp_path, mutation(_SCHEDULER_BASE)))


def test_filename_must_match_name(tmp_path):
    with pytest.raises(ValueError, match="文件名"):
        load_suite_spec(_write(tmp_path, _SCHEDULER_BASE, name="other"))


def test_free_batch_slot_check_counts_t0_arrivals(tmp_path):
    text = (
        _SCHEDULER_BASE.replace("max_running_requests: 4", "max_running_requests: 2")
        + "checks: [free_batch_slot_at_t0]\n"
    )
    with pytest.raises(ValueError, match="空闲 batch slot"):
        load_suite_spec(_write(tmp_path, text))


@pytest.mark.parametrize(
    "extra",
    [
        "system_mrr",
        "paged",
        "comparison",
        "interleave",
    ],
)
def test_lockstep_rejects_scheduler_only_features(tmp_path, extra):
    base = """
name: demo
engine: lockstep
system:
  {system_extra}batching_method: [fixed-batching]
  caching_method: [{caching}]
workload: {{id: w, count: 1, prompt_len: 8, output_len: 4}}
{tail}"""
    system_extra = "max_running_requests: 4\n  " if extra == "system_mrr" else ""
    caching = "paged-kv-cache" if extra == "paged" else "dense-kv-cache"
    tail = {
        "comparison": "comparison:\n  group_by: [workload_id]\n  baseline: {batching_method: none}\n"
        "  metrics: [{path: a.b, kind: speedup}]\n",
        "interleave": "execution: {interleave: [batching_method]}\n",
    }.get(extra, "")
    with pytest.raises(Exception, match="lockstep"):
        load_suite_spec(_write(tmp_path, base.format(system_extra=system_extra, caching=caching, tail=tail)))


def test_length_and_arrival_expansion_edge_cases():
    from mini_infra.bench.spec.common import GroupSpec, expand_arrival, expand_length

    def group(**kwargs: Any) -> GroupSpec:
        return GroupSpec.model_validate({"count": 4, "prompt_len": 8, "output_len": 4, **kwargs})

    assert expand_length(
        GroupSpec.model_validate({"count": 3, "prompt_len": {"alternating": [10, 20]}, "output_len": 1}).prompt_len, 3
    ) == (10, 20, 10)
    ladder = group(prompt_len={"ladder": {"max": 4}}, output_len={"ladder": {"max": 8, "floor": 2}})
    assert expand_length(ladder.prompt_len, 4) == (1, 1, 2, 4)
    assert expand_length(ladder.output_len, 4) == (2, 2, 4, 8)
    # 与历史 make_variable_prompt_lengths / make_variable_output_budgets 的取值一致
    for max_len, floor, expected in [
        (128, 1, (16, 32, 64, 128)),
        (1, 1, (1, 1, 1, 1)),
        (64, 2, (8, 16, 32, 64)),
        (2, 2, (2, 2, 2, 2)),
    ]:
        assert expand_length(group(prompt_len={"ladder": {"max": max_len, "floor": floor}}).prompt_len, 4) == expected
    assert expand_arrival(group(arrival_ms={"interval": 150}).arrival_ms, 4) == (0.0, 150.0, 300.0, 450.0)
    assert expand_arrival(group(arrival_ms=5).arrival_ms, 2) == (5.0, 5.0)
    assert expand_arrival(group(arrival_ms=[0, 1, 2, 3]).arrival_ms, 4) == (0.0, 1.0, 2.0, 3.0)


def test_negative_arrival_and_unknown_length_kind_are_rejected():
    from mini_infra.bench.spec.common import GroupSpec

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        GroupSpec.model_validate({"count": 1, "prompt_len": 8, "output_len": 4, "arrival_ms": -1})
    with pytest.raises(ValueError):
        GroupSpec.model_validate({"count": 1, "prompt_len": {"zipf": 1}, "output_len": 4})
