"""lockstep 模式下 suites（YAML 声明 → BenchmarkSuite → BenchmarkCase）与 run_suite 接入的单元测试（CPU 可运行）。"""

import json
from pathlib import Path

import pytest
import torch

from mini_infra.bench.cases import (
    BenchmarkCase,
    WorkloadProfile,
)
from mini_infra.bench.lockstep.run_case import run_case
from mini_infra.bench.lockstep.suites import (
    get_all_lockstep_suites,
    get_lockstep_suite,
    list_lockstep_suite_names,
    load_lockstep_suite,
)
from mini_infra.bench.spec.suite.loader import load_suite_spec
from mini_infra.bench.suite import BenchmarkSuite
from mini_infra.engine import BatchingMethod, CachingMethod, Mode
from mini_infra.verification.smoke import build_tiny_model


def _tiny_suite(tmp_path: Path, name: str, body: str) -> BenchmarkSuite:
    path = tmp_path / f"{name}.yaml"
    path.write_text(f"name: {name}\n{body}", encoding="utf-8")
    return load_lockstep_suite(path)


def _run_cases(suite: BenchmarkSuite):
    model = build_tiny_model()
    device = torch.device("cpu")
    return [
        run_case(
            model=model,
            device=device,
            case=case,
            warmup=suite.warmup,
            repeats=suite.repeats,
        )
        for case in suite.cases
    ]


def test_lockstep_suites_registry():
    assert list_lockstep_suite_names() == [
        "serial-vs-fixed-batch-throughput",
        "uniform-vs-variable-budget-slots",
        "uniform-vs-variable-prompt-padding",
    ]
    for suite in get_all_lockstep_suites():
        assert isinstance(suite, BenchmarkSuite)
        assert suite.engine == "lockstep"
        assert isinstance(suite.cases, tuple)
        assert len(suite) == len(suite.cases)
        assert list(suite) == list(suite.cases)
        assert get_lockstep_suite(suite.name).name == suite.name
    with pytest.raises(ValueError, match="未知 lockstep suite"):
        get_lockstep_suite("nope")


def test_load_lockstep_suite_rejects_other_engine(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text(
        "name: demo\nsystem: {engine_mode: {kind: scheduler, max_running_requests: 2}, batching_method: fixed-batching,"
        " caching_method: dense-kv-cache}\nworkload: {id: w, count: 1, prompt_len: 8, output_len: 4}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="engine 必须是 lockstep"):
        load_lockstep_suite(path)


def test_serial_vs_fixed_batch_throughput_cases_nest_batch_size_outside_mode():
    suite = get_lockstep_suite("serial-vs-fixed-batch-throughput")
    assert (suite.warmup, suite.repeats) == (1, 3)
    cases = suite.cases
    assert len(cases) == 5 * 2
    assert [(c.profile.num_total_requests, c.mode.batching_method) for c in cases[:4]] == [
        (1, BatchingMethod.NONE),
        (1, BatchingMethod.FIXED),
        (2, BatchingMethod.NONE),
        (2, BatchingMethod.FIXED),
    ]
    assert [c.profile.num_total_requests for c in cases[::2]] == [1, 2, 4, 8, 16]
    assert all(
        c.profile.prompt_lens == (128,) * c.profile.num_total_requests
        and c.profile.output_budgets == (32,) * c.profile.num_total_requests
        and c.mode.use_kv_cache
        for c in cases
    )
    assert [m.batching_method for m in suite.modes()] == [BatchingMethod.NONE, BatchingMethod.FIXED]


def test_serial_vs_fixed_batch_throughput_info_is_json_serializable():
    data = get_lockstep_suite("serial-vs-fixed-batch-throughput").to_dict()
    assert data["name"] == "serial-vs-fixed-batch-throughput"
    assert data["engine"] == "lockstep"
    assert data["spec"]["sweep"]["workload"]["num_requests"] == [1, 2, 4, 8, 16]
    json.dumps(data)


def test_uniform_vs_variable_prompt_padding_cases():
    cases = get_lockstep_suite("uniform-vs-variable-prompt-padding").cases
    assert [(c.profile.num_total_requests, c.profile.prompt_lens, c.profile.output_budgets) for c in cases] == [
        (4, (128, 128, 128, 128), (32, 32, 32, 32)),
        (4, (16, 32, 64, 128), (32, 32, 32, 32)),
    ]
    assert all(c.mode == Mode(CachingMethod.DENSE, batching_method=BatchingMethod.FIXED) for c in cases)


def test_uniform_vs_variable_budget_slots_cases():
    cases = get_lockstep_suite("uniform-vs-variable-budget-slots").cases
    assert [(c.profile.num_total_requests, c.profile.prompt_lens, c.profile.output_budgets) for c in cases] == [
        (4, (128, 128, 128, 128), (64, 64, 64, 64)),
        (4, (128, 128, 128, 128), (8, 16, 32, 64)),
    ]
    assert all(c.mode == Mode(CachingMethod.DENSE, batching_method=BatchingMethod.FIXED) for c in cases)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("output_len: 32", "output_len: 1", "输出预算 >= 2"),
        ("num_requests: [1, 2]", "num_requests: []", "不能为空列表"),
        ("num_requests: [1, 2]", "num_requests: [0]", "greater than 0"),
        ("repeats: 1", "repeats: 0", "greater than 0"),
        ("warmup: 0", "warmup: -1", "greater than or equal to 0"),
        ("batching_method: [none, fixed-batching]", "batching_method: []", "at least 1"),
    ],
)
def test_lockstep_suite_rejects_invalid_config(tmp_path, old, new, message):
    body = """execution: {warmup: 0, repeats: 1}
system:
  engine_mode: lockstep
  batching_method: none
  caching_method: dense-kv-cache
workload:
  vars: {num_requests: 1}
  id: "b_${num_requests}"
  count: "${num_requests}"
  prompt_len: 4
  output_len: 32
sweep:
  system:
    batching_method: [none, fixed-batching]
  workload:
    num_requests: [1, 2]
checks: [{min_output_len: 2}]
"""
    assert old in body
    with pytest.raises(ValueError, match=message):
        _tiny_suite(tmp_path, "demo", body.replace(old, new))


def test_parse_args_sweep_lockstep():
    from mini_infra.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["bench", "sweep", "--engine", "lockstep", "--tag", "t", "--suite", "serial-vs-fixed-batch-throughput"]
    )
    assert args.suite == ["serial-vs-fixed-batch-throughput"]
    assert args.engine == "lockstep"

    # 无法识别的参数直接报错
    for invalid in (["--prompt-lens", "32"], ["--batch-size", "2"], ["--no-kv-cache"]):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "bench",
                    "sweep",
                    "--engine",
                    "lockstep",
                    "--tag",
                    "t",
                    "--suite",
                    "serial-vs-fixed-batch-throughput",
                    *invalid,
                ]
            )


def test_lockstep_suite_subcommand_removed():
    from mini_infra.cli import build_parser

    parser = build_parser()
    # 验证旧的 bench lockstep suite 子命令已被彻底移除（由 bench sweep 统一接管）
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "lockstep", "suite", "--tag", "t"])


def test_serial_vs_fixed_batch_throughput_runs_end_to_end_on_cpu(tmp_path):
    """用缩小的 throughput 配置，跑通 serial 与 batch 两条路径的所有用例。"""
    tiny = _tiny_suite(
        tmp_path,
        "tiny-throughput",
        """execution: {warmup: 0, repeats: 1}
system:
  engine_mode: lockstep
  batching_method: none
  caching_method: dense-kv-cache
workload:
  vars: {num_requests: 1}
  id: "b_${num_requests}"
  count: "${num_requests}"
  prompt_len: 4
  output_len: 2
sweep:
  system:
    batching_method: [none, fixed-batching]
  workload:
    num_requests: [1, 2]
""",
    )
    results = _run_cases(tiny)

    assert [(r.key.mode, r.workload.num_total_requests) for r in results] == [
        (Mode(CachingMethod.DENSE), 1),
        (Mode(CachingMethod.DENSE, batching_method=BatchingMethod.FIXED), 1),
        (Mode(CachingMethod.DENSE), 2),
        (Mode(CachingMethod.DENSE, batching_method=BatchingMethod.FIXED), 2),
    ]
    for r in results:
        assert set(r.workload.prompt_lens) == {4} and set(r.workload.output_budgets) == {2}
        assert r.to_record_dict()["prompt_len"] == 4 and r.to_record_dict()["max_new_tokens"] == 2
        assert r.metrics.throughput_req_s.median > 0
        assert r.metrics.throughput_tok_s.median > 0


def test_uniform_vs_variable_prompt_padding_runs_end_to_end_on_cpu(tmp_path):
    """用缩小的 padding 配置，跑通 uniform 与 variable 两条用例并验证 Padding 有效率。"""
    tiny = _tiny_suite(
        tmp_path,
        "tiny-padding",
        """execution: {warmup: 0, repeats: 1}
system: {engine_mode: lockstep, batching_method: fixed-batching, caching_method: dense-kv-cache}
workload:
  vars: {scenario: uniform, prompt_len: [4, 4, 4, 4]}
  id: "${scenario}"
  count: 4
  prompt_len: "${prompt_len}"
  output_len: 2
sweep:
  workload:
    - {scenario: uniform, prompt_len: [4, 4, 4, 4]}
    - {scenario: variable, prompt_len: {ladder: {max: 4, floor: 1}}}
""",
    )
    res_homo, res_mixed = _run_cases(tiny)

    # uniform: [4, 4, 4, 4]，padding efficiency 应该为 100%
    assert res_homo.workload.num_total_requests == 4
    assert res_homo.workload.prompt_lens == (4, 4, 4, 4)
    assert res_homo.packing is not None and res_homo.packing.padding_efficiency.mean == 1.0

    # variable: [max(1, 4//8), max(1, 4//4), max(1, 4//2), 4] = [1, 1, 2, 4]
    assert res_mixed.workload.prompt_lens == (1, 1, 2, 4)
    assert res_mixed.packing is not None and res_mixed.packing.padding_efficiency.mean == (1 + 1 + 2 + 4) / 16
    assert res_mixed.packing.padding_efficiency.mean < 1.0


def test_uniform_vs_variable_budget_slots_runs_end_to_end_on_cpu(tmp_path):
    """用缩小的 slots 配置，跑通 uniform 与 variable 两条用例并验证 batch slot 利用率。"""
    tiny = _tiny_suite(
        tmp_path,
        "tiny-slots",
        """execution: {warmup: 0, repeats: 1}
system: {engine_mode: lockstep, batching_method: fixed-batching, caching_method: dense-kv-cache}
workload:
  vars: {scenario: uniform, output_len: [4, 4, 4, 4]}
  id: "${scenario}"
  count: 4
  prompt_len: [4, 4, 4, 4]
  output_len: "${output_len}"
sweep:
  workload:
    - {scenario: uniform, output_len: [4, 4, 4, 4]}
    - {scenario: variable, output_len: {ladder: {max: 4, floor: 2}}}
""",
    )
    res_uniform, res_variable = _run_cases(tiny)

    # uniform: [4, 4, 4, 4]，slot utilization 应该为 100%
    assert res_uniform.workload.output_budgets == (4, 4, 4, 4)
    assert res_uniform.packing is not None and res_uniform.packing.slot_utilization.mean == 1.0

    # variable: [max(2, 4//8), max(2, 4//4), max(2, 4//2), 4] = [2, 2, 2, 4]，slot utilization = 10 / 16
    assert res_variable.workload.prompt_lens == (4, 4, 4, 4)
    assert res_variable.workload.output_budgets == (2, 2, 2, 4)
    assert res_variable.packing is not None and res_variable.packing.slot_utilization.mean == (2 + 2 + 2 + 4) / 16
    assert res_variable.packing.slot_utilization.mean < 1.0


def test_run_suite_drives_cases_in_order(tmp_path):
    from mini_infra.bench.lockstep.run_suite import run_suite

    tiny = _tiny_suite(
        tmp_path,
        "tiny-run-suite",
        """execution: {warmup: 0, repeats: 1}
system: {engine_mode: lockstep, batching_method: fixed-batching, caching_method: dense-kv-cache}
workload:
  vars: {num_requests: 1}
  id: "b_${num_requests}"
  count: "${num_requests}"
  prompt_len: 4
  output_len: 2
sweep:
  workload:
    num_requests: [1, 2]
""",
    )
    seen: list[BenchmarkCase] = []
    model = build_tiny_model()
    results = run_suite(
        model,
        torch.device("cpu"),
        tiny,
        on_case_progress=lambda _n, _total, case: seen.append(case),
    )
    assert [c.profile.num_total_requests for c in seen] == [1, 2]
    assert [r.workload.num_total_requests for r in results] == [1, 2]


def test_benchmark_case_and_suite_unification(tmp_path):
    profile = WorkloadProfile(
        workload_id="test_profile",
        num_total_requests=3,
        prompt_lens=(16, 16, 16),
        output_budgets=(8, 8, 8),
        arrival_ms=(0.0, 0.0, 0.0),
    )
    mode = Mode(caching_method=CachingMethod.DENSE, batching_method=BatchingMethod.FIXED)

    # 1. 缺省 max_running_requests 自动推导为 num_total_requests
    case = BenchmarkCase(profile=profile, mode=mode, engine="lockstep")
    assert case.max_running_requests == 3
    assert case.key.max_running_requests == 3
    assert case.workload_id == "test_profile"
    assert case.engine == "lockstep"

    # 2. 显式提供 max_running_requests
    scheduler_case = BenchmarkCase(profile=profile, mode=mode, max_running_requests=2, engine="scheduler")
    assert scheduler_case.max_running_requests == 2
    assert scheduler_case.engine == "scheduler"

    # 3. BenchmarkSuite 实体容器直接实例化与解析
    spec_path = tmp_path / "generic-suite.yaml"
    spec_path.write_text(
        """name: generic-suite
execution: {warmup: 1, repeats: 2}
system: {engine_mode: lockstep, batching_method: fixed-batching, caching_method: dense-kv-cache}
workload:
  id: "w3"
  count: 3
  prompt_len: 16
  output_len: 8
""",
        encoding="utf-8",
    )
    loaded = load_suite_spec(spec_path)
    suite = BenchmarkSuite(loaded=loaded)
    assert suite.name == "generic-suite"
    assert suite.engine == "lockstep"
    assert len(suite) == 1
    assert isinstance(suite[0], BenchmarkCase)
    assert suite[0].max_running_requests == 3
