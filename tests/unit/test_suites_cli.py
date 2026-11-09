"""bench suites 子命令、--spec 参数、lockstep --dry-run 与 suite JSON Schema 的单元测试（CPU 可运行）。"""

import argparse
import json
from pathlib import Path

import pytest

from mini_infra.bench.common import BENCH_ARTIFACTS_DIR, REPO_ROOT, SUITES_DIR
from mini_infra.bench.spec import suite_json_schema
from mini_infra.cli import main
from mini_infra.cli.commands.bench.suite_source import suite_output_path

_SCHEDULER_YAML = """
name: demo-spec
engine: scheduler
execution: {warmup: 0, repeats: 1, interleave: [batching_method]}
system:
  max_running_requests: 4
  batching_method: [fixed-batching, continuous-batching]
  caching_method: [dense-kv-cache]
workload: {id: w, count: 2, prompt_len: 8, output_len: 4}
"""


def test_committed_json_schema_is_up_to_date():
    committed = json.loads((BENCH_ARTIFACTS_DIR / "specs" / "suite.schema.json").read_text(encoding="utf-8"))
    assert committed == suite_json_schema(), (
        "运行 `mini-infra bench suites schema --output benchmarks/specs/suite.schema.json`"
    )


def test_json_schema_shape():
    schema = suite_json_schema()
    assert schema["required"] == ["name"]
    assert {"name", "engine", "extends", "system", "workload", "sweep", "comparison", "checks"} <= set(
        schema["properties"]
    )
    assert "WorkloadTemplate" not in schema["$defs"] and "GroupSpec" in schema["$defs"]


def test_suites_list_and_show(capsys):
    assert main(["bench", "suites", "list"]) == 0
    out = capsys.readouterr().out
    assert "dense-vs-paged-block-size" in out and "serial-vs-fixed-batch-throughput" in out
    assert "_bases" not in out and "dense_vs_paged" not in out

    assert main(["bench", "suites", "list", "--engine", "lockstep"]) == 0
    out = capsys.readouterr().out
    assert "serial-vs-fixed-batch-throughput" in out and "dense-vs-paged" not in out

    assert main(["bench", "suites", "show", "dense-vs-paged-block-size"]) == 0
    out = capsys.readouterr().out
    assert "block_size:" in out and "共 5 个" in out
    assert "paged-kv-cache+continuous-batching | mrr=8 | block=64" in out


def test_suites_show_accepts_path_and_rejects_unknown(tmp_path, capsys):
    path = tmp_path / "demo-spec.yaml"
    path.write_text(_SCHEDULER_YAML, encoding="utf-8")
    assert main(["bench", "suites", "show", str(path)]) == 0
    assert "共 2 个" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="未知 suite"):
        main(["bench", "suites", "show", "nope"])


def test_suites_schema_stdout_and_file(tmp_path, capsys):
    assert main(["bench", "suites", "schema"]) == 0
    assert json.loads(capsys.readouterr().out) == suite_json_schema()
    out = tmp_path / "nested" / "schema.json"
    assert main(["bench", "suites", "schema", "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == suite_json_schema()


def test_suite_and_spec_and_all_are_mutually_exclusive(tmp_path):
    spec = tmp_path / "demo-spec.yaml"
    for engine in ("scheduler", "lockstep"):
        base = ["bench", "sweep", "--engine", engine, "--tag", "t", "--dry-run"]
        with pytest.raises(SystemExit, match="必须且只能指定"):
            main(base)
        with pytest.raises(SystemExit, match="必须且只能指定"):
            main([*base, "--suite", "x", "--spec", str(spec)])
        with pytest.raises(SystemExit, match="必须且只能指定"):
            main([*base, "--suite", "x", "--all"])


def test_scheduler_suite_dry_run_with_spec(tmp_path, capsys):
    path = tmp_path / "demo-spec.yaml"
    path.write_text(_SCHEDULER_YAML, encoding="utf-8")
    assert main(["bench", "sweep", "--engine", "scheduler", "--tag", "dev", "--dry-run", "--spec", str(path)]) == 0
    out = capsys.readouterr().out
    assert "=== [Dry Run] Suite: demo-spec ===" in out and "suite=demo-spec" in out


def test_lockstep_suite_dry_run_uses_yaml_execution_defaults(capsys):
    argv = [
        "bench",
        "sweep",
        "--engine",
        "lockstep",
        "--tag",
        "dev",
        "--dry-run",
        "--suite",
        "uniform-vs-variable-prompt-padding",
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "=== [Dry Run] Suite: uniform-vs-variable-prompt-padding ===" in out
    assert "共 2 个用例，每个 warmup=1 repeats=3" in out
    assert "prompt=16/32/64/128" in out

    assert main([*argv, "--warmup", "0", "--repeats", "5"]) == 0
    assert "warmup=0 repeats=5" in capsys.readouterr().out


def test_lockstep_suite_dry_run_with_spec(tmp_path, capsys):
    path = tmp_path / "demo-lockstep.yaml"
    path.write_text(
        "name: demo-lockstep\nengine: lockstep\n"
        "system: {batching_method: [fixed-batching], caching_method: [dense-kv-cache]}\n"
        "workload: {id: w, count: 2, prompt_len: 8, output_len: 4}\n",
        encoding="utf-8",
    )
    assert main(["bench", "sweep", "--engine", "lockstep", "--tag", "dev", "--dry-run", "--spec", str(path)]) == 0
    assert "Suite: demo-lockstep" in capsys.readouterr().out


def test_suite_output_path_routing(tmp_path):
    def ns(**kw):
        return argparse.Namespace(**{"output": None, "spec": None, "tag": "dev", **kw})

    kw = {"experimental": False}
    assert suite_output_path(ns(), "lockstep", "s", "abc", **kw) == SUITES_DIR / "lockstep" / "s" / "abc_dev.json"
    assert suite_output_path(ns(), "scheduler", "s", "abc", **kw) == SUITES_DIR / "scheduler" / "s" / "abc_dev.json"
    # --spec 的实验结果默认进入被 .gitignore 忽略的 tmp/，不污染受控的 benchmarks/
    experimental = suite_output_path(ns(spec=Path("x.yaml")), "scheduler", "s", "abc", experimental=True)
    assert experimental == REPO_ROOT / "tmp" / "suites" / "scheduler" / "s" / "abc_dev.json"
    explicit = tmp_path / "o.json"
    assert suite_output_path(ns(output=explicit), "lockstep", "s", "abc", experimental=True) == explicit


def test_committed_release_json_schema_is_up_to_date():
    from mini_infra.bench.spec import release_json_schema

    committed = json.loads((BENCH_ARTIFACTS_DIR / "specs" / "release.schema.json").read_text(encoding="utf-8"))
    hint = "运行 `mini-infra bench suites schema --kind release --output benchmarks/specs/release.schema.json` 重新生成"
    assert committed == release_json_schema(), hint
    assert set(committed["properties"]["scales"]["properties"]) == {"small", "medium", "large"}
    assert "required" not in committed and "GroupSpec" in committed["$defs"]


def test_suites_schema_kind_release(capsys):
    from mini_infra.bench.spec import release_json_schema

    assert main(["bench", "suites", "schema", "--kind", "release"]) == 0
    assert json.loads(capsys.readouterr().out) == release_json_schema()
