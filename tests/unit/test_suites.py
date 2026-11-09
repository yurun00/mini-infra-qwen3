"""lockstep.suites（YAML 声明 → LockstepSuite → LockstepCase）与 run_suite 接入的单元测试（CPU 可运行）。"""

import json
from pathlib import Path

import pytest
import torch

from mini_infra.bench.cases import LockstepCase
from mini_infra.bench.lockstep.run_case import run_case
from mini_infra.bench.lockstep.suites import (
    LockstepSuite,
    get_all_lockstep_suites,
    get_lockstep_suite,
    list_lockstep_suite_names,
    load_lockstep_suite,
)
from mini_infra.engine import BatchingMethod, CachingMethod, Mode
from mini_infra.verification.smoke import build_tiny_model


def _tiny_suite(tmp_path: Path, name: str, body: str) -> LockstepSuite:
    path = tmp_path / f"{name}.yaml"
    path.write_text(f"name: {name}\nengine: lockstep\n{body}", encoding="utf-8")
    return load_lockstep_suite(path)


def _run_cases(suite: LockstepSuite):
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
        for case in suite.cases(vocab_size=model.config.vocab_size)
    ]


def test_lockstep_suites_registry():
    assert list_lockstep_suite_names() == [
        "serial-vs-fixed-batch-throughput",
        "uniform-vs-variable-budget-slots",
        "uniform-vs-variable-prompt-padding",
    ]
    for suite in get_all_lockstep_suites():
        assert isinstance(suite, LockstepSuite)
        assert get_lockstep_suite(suite.name).name == suite.name
    with pytest.raises(ValueError, match="未知 lockstep suite"):
        get_lockstep_suite("nope")


def test_load_lockstep_suite_rejects_other_engine(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text(
        "name: demo\nengine: scheduler\nsystem: {max_running_requests: 2, batching_method: [fixed-batching],"
        " caching_method: [dense-kv-cache]}\nworkload: {id: w, count: 1, prompt_len: 8, output_len: 4}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="engine 必须是 lockstep"):
        load_lockstep_suite(path)


def test_serial_vs_fixed_batch_throughput_cases_nest_batch_size_outside_mode():
    suite = get_lockstep_suite("serial-vs-fixed-batch-throughput")
    assert (suite.warmup, suite.repeats) == (1, 3)
    cases = suite.cases(vocab_size=100)
    assert len(cases) == 5 * 2
    assert [(c.payload.request_count, c.mode.batching_method) for c in cases[:4]] == [
        (1, BatchingMethod.NONE),
        (1, BatchingMethod.FIXED),
        (2, BatchingMethod.NONE),
        (2, BatchingMethod.FIXED),
    ]
    assert [c.payload.request_count for c in cases[::2]] == [1, 2, 4, 8, 16]
    assert all(
        c.payload.prompt_lengths == (128,) * c.payload.request_count
        and c.payload.output_budgets == (32,) * c.payload.request_count
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
    cases = get_lockstep_suite("uniform-vs-variable-prompt-padding").cases(vocab_size=100)
    assert [(c.payload.request_count, c.payload.prompt_lengths, c.payload.output_budgets) for c in cases] == [
        (4, (128, 128, 128, 128), (32, 32, 32, 32)),
        (4, (16, 32, 64, 128), (32, 32, 32, 32)),
    ]
    assert all(c.mode == Mode(CachingMethod.DENSE, batching_method=BatchingMethod.FIXED) for c in cases)


def test_uniform_vs_variable_budget_slots_cases():
    cases = get_lockstep_suite("uniform-vs-variable-budget-slots").cases(vocab_size=100)
    assert [(c.payload.request_count, c.payload.prompt_lengths, c.payload.output_budgets) for c in cases] == [
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
  batching_method: [none, fixed-batching]
  caching_method: [dense-kv-cache]
workload:
  id: "b_${num_requests}"
  vars: {num_requests: 1}
  count: "${num_requests}"
  prompt_len: 4
  output_len: 32
sweep:
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
  batching_method: [none, fixed-batching]
  caching_method: [dense-kv-cache]
workload:
  id: "b_${num_requests}"
  vars: {num_requests: 1}
  count: "${num_requests}"
  prompt_len: 4
  output_len: 2
sweep:
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
        assert r.workload.prompt_len == 4 and r.workload.max_new_tokens == 2
        assert r.metrics.throughput_req_s.median > 0
        assert r.metrics.throughput_tok_s.median > 0


def test_uniform_vs_variable_prompt_padding_runs_end_to_end_on_cpu(tmp_path):
    """用缩小的 padding 配置，跑通 uniform 与 variable 两条用例并验证 Padding 有效率。"""
    tiny = _tiny_suite(
        tmp_path,
        "tiny-padding",
        """execution: {warmup: 0, repeats: 1}
system: {batching_method: [fixed-batching], caching_method: [dense-kv-cache]}
workload:
  id: "${scenario}"
  vars: {scenario: uniform, prompt_len: [4, 4, 4, 4]}
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
    assert res_homo.workload.prompt_len == (4, 4, 4, 4)
    assert res_homo.packing is not None and res_homo.packing.padding_efficiency.mean == 1.0

    # variable: [max(1, 4//8), max(1, 4//4), max(1, 4//2), 4] = [1, 1, 2, 4]
    assert res_mixed.workload.prompt_len == (1, 1, 2, 4)
    assert res_mixed.packing is not None and res_mixed.packing.padding_efficiency.mean == (1 + 1 + 2 + 4) / 16
    assert res_mixed.packing.padding_efficiency.mean < 1.0


def test_uniform_vs_variable_budget_slots_runs_end_to_end_on_cpu(tmp_path):
    """用缩小的 slots 配置，跑通 uniform 与 variable 两条用例并验证 batch slot 利用率。"""
    tiny = _tiny_suite(
        tmp_path,
        "tiny-slots",
        """execution: {warmup: 0, repeats: 1}
system: {batching_method: [fixed-batching], caching_method: [dense-kv-cache]}
workload:
  id: "${scenario}"
  vars: {scenario: uniform, output_len: [4, 4, 4, 4]}
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
    assert res_uniform.workload.max_new_tokens == (4, 4, 4, 4)
    assert res_uniform.packing is not None and res_uniform.packing.slot_utilization.mean == 1.0

    # variable: [max(2, 4//8), max(2, 4//4), max(2, 4//2), 4] = [2, 2, 2, 4]，slot utilization = 10 / 16
    assert res_variable.workload.prompt_len == (4, 4, 4, 4)
    assert res_variable.workload.max_new_tokens == (2, 2, 2, 4)
    assert res_variable.packing is not None and res_variable.packing.slot_utilization.mean == (2 + 2 + 2 + 4) / 16
    assert res_variable.packing.slot_utilization.mean < 1.0


def test_run_suite_drives_cases_in_order(tmp_path):
    from mini_infra.bench.lockstep.run_suite import run_suite

    tiny = _tiny_suite(
        tmp_path,
        "tiny-run-suite",
        """execution: {warmup: 0, repeats: 1}
system: {batching_method: [fixed-batching], caching_method: [dense-kv-cache]}
workload:
  id: "b_${num_requests}"
  vars: {num_requests: 1}
  count: "${num_requests}"
  prompt_len: 4
  output_len: 2
sweep:
  workload:
    num_requests: [1, 2]
""",
    )
    seen: list[LockstepCase] = []
    model = build_tiny_model()
    results = run_suite(
        model,
        torch.device("cpu"),
        tiny,
        model.config.vocab_size,
        on_case_progress=lambda _n, _total, case: seen.append(case),
    )
    assert [c.payload.request_count for c in seen] == [1, 2]
    assert [r.workload.num_total_requests for r in results] == [1, 2]
