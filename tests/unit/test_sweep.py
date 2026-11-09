"""bench sweep：--set 覆盖、批量运行汇总、CPU 上跨 suite 共享环境的执行，以及随机分布 / release 声明校验（纯 CPU）。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mini_infra.bench.lockstep.suites import load_lockstep_suite
from mini_infra.bench.scheduler.suites import get_scheduler_suite, load_scheduler_suite
from mini_infra.bench.schema import validate_benchmark_record
from mini_infra.bench.spec import (
    ReleaseSpec,
    apply_overrides,
    get_release_spec,
    load_release_spec,
    load_suite_spec,
    parse_set_args,
)
from mini_infra.bench.sweep import SuiteRunSummary, format_sweep_summary, run_sweep
from mini_infra.cli import main
from mini_infra.cli.commands.bench.lockstep import LockstepRunEnv, execute_lockstep_suite
from mini_infra.cli.commands.bench.scheduler import SchedulerRunEnv, execute_scheduler_suite
from mini_infra.verification.smoke import build_tiny_model

_SCHEDULER_YAML = """
name: {name}
engine: scheduler
execution: {{warmup: 0, repeats: 1, interleave: [batching_method]}}
system:
  max_running_requests: 4
  batching_method: [fixed-batching, continuous-batching]
  caching_method: [dense-kv-cache]
workload: {{id: w, count: 2, prompt_len: 4, output_len: {{alternating: [2, 3]}}}}
"""

_LOCKSTEP_YAML = """
name: {name}
engine: lockstep
execution: {{warmup: 0, repeats: 1}}
system: {{batching_method: [fixed-batching], caching_method: [dense-kv-cache]}}
workload: {{id: w, count: 2, prompt_len: 4, output_len: 2}}
"""


def _write(tmp_path: Path, name: str, template: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(template.format(name=name), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- --set 覆盖


def test_parse_set_args_uses_yaml_values():
    parsed = parse_set_args(["execution.repeats=5", "sweep.workload.n=[1, 2]", "workload.id=abc", "x.y=null"])
    assert parsed == {"execution.repeats": 5, "sweep.workload.n": [1, 2], "workload.id": "abc", "x.y": None}


@pytest.mark.parametrize("bad", ["novalue", "=1", "a..b=1", "a.=1", ".a=1"])
def test_parse_set_args_rejects_malformed(bad):
    with pytest.raises(ValueError, match="key.path=value"):
        parse_set_args([bad])


def test_apply_overrides_creates_parents_replaces_leaf_and_does_not_mutate():
    raw = {"execution": {"warmup": 1, "repeats": 3}, "sweep": None}
    out = apply_overrides(raw, {"execution.repeats": 9, "system.paged.block_size": 32})
    assert out == {"execution": {"warmup": 1, "repeats": 9}, "sweep": None, "system": {"paged": {"block_size": 32}}}
    assert raw == {"execution": {"warmup": 1, "repeats": 3}, "sweep": None}
    with pytest.raises(ValueError, match="不是映射"):
        apply_overrides(raw, {"execution.repeats.deeper": 1})


def test_load_suite_spec_with_overrides_records_them_and_changes_hash(tmp_path):
    path = _write(tmp_path, "demo", _SCHEDULER_YAML)
    base = load_suite_spec(path)
    over = load_suite_spec(path, {"execution.repeats": 7})
    assert over.spec.execution.repeats == 7 and base.spec.execution.repeats == 1
    assert over.sha256 != base.sha256
    assert over.info()["overrides"] == {"execution.repeats": 7}
    assert "overrides" not in base.info()


def test_invalid_override_is_rejected_at_load(tmp_path):
    path = _write(tmp_path, "demo", _SCHEDULER_YAML)
    with pytest.raises(ValueError, match="未在 workload.vars 中声明"):
        load_suite_spec(path, {"sweep.workload.nope": [1, 2]})
    with pytest.raises(ValueError, match="greater than 0"):
        load_suite_spec(path, {"execution.repeats": 0})


def test_override_sweep_on_builtin_suite():
    suite = get_scheduler_suite("dense-vs-paged-length-ratio", {"sweep.workload.long_prompt_len": [256, 512]})
    assert [c.workload_id for c in suite.cases(vocab_size=100)] == [f"length_ratio_{n}" for n in (256, 256, 512, 512)]


# --------------------------------------------------------------------------- run_sweep / 汇总


def test_run_sweep_runs_in_order_and_formats_summary(tmp_path):
    order: list[str] = []
    targets = [load_suite_spec(_write(tmp_path, n, _SCHEDULER_YAML)) for n in ("aa", "bb")]

    def execute(loaded):
        order.append(loaded.spec.name)
        return SuiteRunSummary(
            name=loaded.spec.name,
            engine="scheduler",
            num_cases=2,
            output=tmp_path / f"{loaded.spec.name}.json",
            comparison_lines=(f"[{loaded.spec.name}] speedup 1.50x",) if loaded.spec.name == "bb" else (),
        )

    started: list[tuple[int, int]] = []
    summaries = run_sweep(targets, execute, on_start=lambda i, n, _: started.append((i, n)))
    assert order == ["aa", "bb"] and started == [(1, 2), (2, 2)]
    assert all(s.wall_s >= 0 for s in summaries)
    text = format_sweep_summary(summaries)
    assert "共 2 个 suite" in text and "aa" in text and "bb" in text
    assert text.count("--- Comparisons") == 1 and "speedup 1.50x" in text
    assert format_sweep_summary([]) == "sweep 未执行任何 suite。"


def test_run_sweep_stops_on_first_error(tmp_path):
    targets = [load_suite_spec(_write(tmp_path, n, _SCHEDULER_YAML)) for n in ("aa", "bb")]
    seen: list[str] = []

    def execute(loaded):
        seen.append(loaded.spec.name)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_sweep(targets, execute)
    assert seen == ["aa"]


# --------------------------------------------------------------------------- CLI（dry-run）


def test_sweep_cli_dry_run_multiple_suites_and_overrides(capsys):
    argv = [
        "bench", "sweep", "--engine", "scheduler", "--tag", "t", "--dry-run",
        "--suite", "fixed-vs-continuous-uniform-output-length", "dense-vs-paged-block-size",
        "--set", "execution.warmup=0", "--set", "execution.repeats=1",
    ]  # fmt: skip
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Suite: fixed-vs-continuous-uniform-output-length" in out and "Suite: dense-vs-paged-block-size" in out
    assert "共 2 步" in out and "共 5 步" in out


def test_sweep_cli_dry_run_all_and_spec(tmp_path, capsys):
    assert main(["bench", "sweep", "--engine", "lockstep", "--tag", "t", "--dry-run", "--all"]) == 0
    out = capsys.readouterr().out
    assert out.count("=== [Dry Run] Suite:") == 3

    path = _write(tmp_path, "demo-lock", _LOCKSTEP_YAML)
    assert main(["bench", "sweep", "--engine", "lockstep", "--tag", "t", "--dry-run", "--spec", str(path)]) == 0
    assert "Suite: demo-lock" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "只能指定"),
        (["--all", "--suite", "dense-vs-paged-block-size"], "只能指定"),
        (["--suite", "nope"], "未知 scheduler suite"),
        (["--all", "--output", "x.json"], "--output-dir"),
        (["--suite", "dense-vs-paged-block-size", "dense-vs-paged-length-ratio", "--output", "x.json"], "--output-dir"),
        (["--suite", "dense-vs-paged-block-size", "--output-dir", "d", "--output", "x.json"], "不能同时指定"),
    ],
)
def test_sweep_cli_rejects_bad_selection(extra, message):
    with pytest.raises(SystemExit, match=message):
        main(["bench", "sweep", "--engine", "scheduler", "--tag", "t", "--dry-run", *extra])


def test_sweep_cli_single_suite_accepts_output_dry_run(capsys):
    argv = [
        "bench",
        "sweep",
        "--engine",
        "scheduler",
        "--tag",
        "t",
        "--dry-run",
        "--suite",
        "dense-vs-paged-block-size",
        "--output",
        "custom.json",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Suite: dense-vs-paged-block-size" in out


def test_sweep_cli_rejects_duplicate_suite_names(tmp_path):
    (tmp_path / "a").mkdir()
    a = _write(tmp_path / "a", "same", _SCHEDULER_YAML)
    (tmp_path / "b").mkdir()
    b = _write(tmp_path / "b", "same", _SCHEDULER_YAML)
    with pytest.raises(SystemExit, match="必须唯一"):
        main(["bench", "sweep", "--engine", "scheduler", "--tag", "t", "--dry-run", "--spec", str(a), str(b)])


# --------------------------------------------------------------------------- CPU 上共享环境执行


def _args(**kw):
    base = {
        "tag": "test-tag",
        "model": "dummy",
        "revision": "main",
        "allow_mismatch": False,
        "warmup": None,
        "repeats": None,
        "interleave": None,
        "profile_mode": None,
        "tensorboard_dir": None,
    }
    return SimpleNamespace(**{**base, **kw})


def _env(cls, model, modes):
    # 预填 correctness_cache：执行函数命中缓存时不会调用 tokenizer / check_correctness
    return cls(
        model=model,
        tokenizer=None,
        device=torch.device("cpu"),
        vocab=model.config.vocab_size,
        env={"gpu": "CPU"},
        peaks={"bandwidth_gbs": None, "tflops": None},
        cost=None,
        git={"commit": "abc", "dirty": False},
        correctness_cache={
            m: {"checked": True, "passed": True, "match_ratio": 1.0, "updated_golden": False} for m in modes
        },
    )


def test_execute_scheduler_suites_share_env_and_write_valid_records(tmp_path):
    model = build_tiny_model()
    suites = [load_scheduler_suite(_write(tmp_path, n, _SCHEDULER_YAML)) for n in ("aa", "bb")]
    run_env = _env(SchedulerRunEnv, model, suites[0].modes())

    summaries = []
    for suite in suites:
        output = tmp_path / "out" / f"{suite.name}.json"
        summary = execute_scheduler_suite(run_env, _args(), suite, output)
        assert summary.output == output and summary.num_cases == 2
        record = validate_benchmark_record(json.loads(output.read_text(encoding="utf-8")), source=output)
        assert record.suite is not None and record.suite["name"] == suite.name
        summaries.append(summary)

    assert len(run_env.correctness_cache) == 2  # 两个 suite 共用同一份缓存，没有新增条目
    assert "aa" in format_sweep_summary(summaries)


def test_execute_scheduler_suite_refuses_failed_correctness(tmp_path):
    model = build_tiny_model()
    suite = load_scheduler_suite(_write(tmp_path, "aa", _SCHEDULER_YAML))
    run_env = _env(SchedulerRunEnv, model, suite.modes())
    first_mode = suite.modes()[0]
    run_env.correctness_cache[first_mode] = {"checked": True, "passed": False, "match_ratio": 0.5}
    with pytest.raises(SystemExit, match="正确性检查失败"):
        execute_scheduler_suite(run_env, _args(), suite, tmp_path / "x.json")
    assert not (tmp_path / "x.json").exists()


def test_execute_lockstep_suite_writes_valid_record(tmp_path):
    model = build_tiny_model()
    suite = load_lockstep_suite(_write(tmp_path, "demo-lock", _LOCKSTEP_YAML))
    run_env = _env(LockstepRunEnv, model, suite.modes())
    output = tmp_path / "out" / "demo-lock.json"
    summary = execute_lockstep_suite(run_env, _args(), suite, output)
    assert summary.engine == "lockstep" and summary.num_cases == 1 and summary.comparison_lines == ()
    record = validate_benchmark_record(json.loads(output.read_text(encoding="utf-8")), source=output)
    assert record.suite is not None and record.suite["engine"] == "lockstep"


def test_sweep_cli_single_suite_with_output_writes_to_exact_file(tmp_path, monkeypatch):
    import mini_infra.cli.commands.bench.sweep as sweep_cmd

    model = build_tiny_model()
    spec = _write(tmp_path, "single-suite", _SCHEDULER_YAML)
    suite = load_scheduler_suite(spec)
    run_env = _env(SchedulerRunEnv, model, suite.modes())
    monkeypatch.setattr(sweep_cmd, "prepare_scheduler_env", lambda args: run_env)

    target_output = tmp_path / "custom" / "my_result.json"
    argv = [
        "bench",
        "sweep",
        "--engine",
        "scheduler",
        "--tag",
        "t",
        "--spec",
        str(spec),
        "--output",
        str(target_output),
    ]
    assert main(argv) == 0
    assert target_output.exists()
    record = validate_benchmark_record(json.loads(target_output.read_text(encoding="utf-8")), source=target_output)
    assert record.suite is not None and record.suite["name"] == "single-suite"


# --------------------------------------------------------------------------- 随机分布与 release 声明校验


def test_suite_with_seeded_distributions_is_deterministic(tmp_path):
    text = """
name: seeded
engine: scheduler
system: {max_running_requests: 4, batching_method: [continuous-batching], caching_method: [dense-kv-cache]}
workload:
  id: w
  seed: 7
  seed_salt: 3
  count: 12
  prompt_len: {log_uniform: {min: 8, max: 64}}
  output_len: {log_uniform: {min: 2, max: 16}}
  arrival_ms: {poisson: {initial_burst: 3, mean_interval_ms: 40}}
"""
    path = tmp_path / "seeded.yaml"
    path.write_text(text, encoding="utf-8")
    (a,) = [cs.workload for cs in load_suite_spec(path).cases[:1]]
    (b,) = [cs.workload for cs in load_suite_spec(path, {}).cases[:1]]
    assert a == b
    assert len(a.prompt_lens) == 12 and all(8 <= x <= 64 for x in a.prompt_lens)
    assert a.arrival_ms[:3] == (0.0, 0.0, 0.0) and list(a.arrival_ms) == sorted(a.arrival_ms)
    other = load_suite_spec(path, {"workload.seed": 8}).cases[0].workload
    assert other.prompt_lens != a.prompt_lens


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("  prompt_len: {log_uniform: {min: 8, max: 64}}\n", "必须提供 workload.seed"),
        ("  seed: 1\n  prompt_len: {log_uniform: {min: 64, max: 8}}\n", "min <= max"),
        (
            "  seed: 1\n  prompt_len: 8\n  arrival_ms: {poisson: {initial_burst: 0, mean_interval_ms: 5}}\n",
            "greater than 0",
        ),
        (
            "  seed: 1\n  prompt_len: 8\n  arrival_ms: {poisson: {initial_burst: 2, mean_interval_ms: 0}}\n",
            "greater than 0",
        ),
    ],
)
def test_seeded_distribution_validation(tmp_path, line, message):
    text = (
        "name: bad\nengine: scheduler\n"
        "system: {max_running_requests: 4, batching_method: [continuous-batching], caching_method: [dense-kv-cache]}\n"
        "workload:\n  id: w\n  count: 4\n  output_len: 4\n" + line
    )
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_suite_spec(path)


def test_builtin_release_specs_are_valid():
    for engine in ("lockstep", "scheduler"):
        spec = get_release_spec(engine).spec
        assert spec.engine == engine and set(spec.scales) == {"small", "medium", "large"}


def _release_raw(engine="scheduler", **override):
    scale = {"seed_salt": 1, "max_running_requests": 4, "workload": {"count": 4, "prompt_len": 8, "output_len": 4}}
    system = (
        {"batching_method": ["continuous-batching"], "caching_method": ["dense-kv-cache"]}
        if engine == "scheduler"
        else {"batching_method": ["fixed-batching"], "caching_method": ["dense-kv-cache"]}
    )
    raw = {
        "engine": engine,
        "default_seed": 42,
        "system": system,
        "scales": {"small": scale, "medium": scale, "large": scale},
    }
    raw.update(override)
    return raw


def test_release_spec_validation():
    ReleaseSpec.model_validate(_release_raw())
    with pytest.raises(ValueError, match="恰好包含"):
        ReleaseSpec.model_validate(_release_raw(scales={"small": _release_raw()["scales"]["small"]}))
    no_mrr = {"seed_salt": 1, "workload": {"count": 4, "prompt_len": 8, "output_len": 4}}
    with pytest.raises(ValueError, match="必须提供 max_running_requests"):
        ReleaseSpec.model_validate(_release_raw(scales={"small": no_mrr, "medium": no_mrr, "large": no_mrr}))
    # 允许在 system 顶层提供默认 max_running_requests 供各 scale 继承
    system_with_mrr = {
        "batching_method": ["continuous-batching"],
        "caching_method": ["dense-kv-cache"],
        "max_running_requests": 4,
    }
    inherited = ReleaseSpec.model_validate(
        _release_raw(system=system_with_mrr, scales={"small": no_mrr, "medium": no_mrr, "large": no_mrr})
    )
    assert inherited.system.max_running_requests == 4
    with pytest.raises(ValueError, match="不支持 max_running_requests"):
        ReleaseSpec.model_validate(_release_raw(engine="lockstep"))
    with pytest.raises(ValueError, match="interleave"):
        ReleaseSpec.model_validate(_release_raw(execution={"interleave": ["batching_method"]}))


def _write_release(
    tmp_path: Path,
    name: str,
    workload: str = "{count: 4, prompt_len: 8, output_len: 4}",
    checks: str = "[]",
) -> Path:
    path = tmp_path / f"{name}.yaml"
    scale = f"{{seed_salt: 1, max_running_requests: 4, workload: {workload}}}"
    path.write_text(
        f"engine: {name}\ndefault_seed: 42\n"
        "system: {batching_method: [continuous-batching], caching_method: [dense-kv-cache]}\n"
        f"checks: {checks}\n"
        f"scales: {{small: {scale}, medium: {scale}, large: {scale}}}\n",
        encoding="utf-8",
    )
    return path


def test_load_release_spec_expands_eagerly_like_load_suite_spec(tmp_path):
    loaded = load_release_spec(_write_release(tmp_path, "scheduler"))
    assert set(loaded.scales) == {"small", "medium", "large"}
    assert loaded.scales["small"].workload.num_requests == 4 and loaded.scales["small"].seed == 42
    assert loaded.resolve() is loaded.scales
    assert loaded.resolve(7)["small"].seed == 7
    assert len(loaded.sha256) == 64

    # 语义守卫生效：违反检查在加载期抛出异常
    (tmp_path / "check_failed").mkdir()
    guard_bad = _write_release(tmp_path / "check_failed", "scheduler", checks="[{min_output_len: 10}]")
    with pytest.raises(ValueError, match="要求每个请求的输出预算 >= 10"):
        load_release_spec(guard_bad)

    # 非法负载在加载期暴露（而不是等到 release_workloads 调用时）
    (tmp_path / "bad").mkdir()
    bad = _write_release(tmp_path / "bad", "scheduler", "{count: 4}")
    with pytest.raises(ValueError, match="release scheduler/small"):
        load_release_spec(bad)


def test_load_release_spec_requires_filename_to_match_engine(tmp_path):
    path = _write_release(tmp_path, "scheduler")
    wrong = path.rename(tmp_path / "other.yaml")
    with pytest.raises(ValueError, match="必须与 engine"):
        load_release_spec(wrong)


def test_get_release_spec_unknown_engine_dir(tmp_path):
    with pytest.raises(ValueError, match="未找到"):
        get_release_spec("scheduler", specs_dir=tmp_path)
    (tmp_path / "release").mkdir()
    _write_release(tmp_path / "release", "scheduler")
    assert get_release_spec("scheduler", specs_dir=tmp_path).spec.engine == "scheduler"
