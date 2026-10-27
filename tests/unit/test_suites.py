"""suites.py 配置与 run_suite.py 接入的单元测试（CPU 可运行）。"""

import pytest
import torch

from mini_infra.bench.batch.run_case import run_case
from mini_infra.bench.batch.suites import (
    SUITES,
    SerialVsFixedBatchThroughputSuite,
    Suite,
    UniformVsVariableBudgetSlotsSuite,
    UniformVsVariablePromptPaddingSuite,
    get_suite,
)
from mini_infra.engine import BatchingMethod
from mini_infra.verification.smoke import build_tiny_model


def test_serial_vs_fixed_batch_throughput_defaults():
    suite = get_suite("serial-vs-fixed-batch-throughput")
    assert suite.batch_sizes == (1, 2, 4, 8, 16)
    assert suite.prompt_length == 128
    assert suite.max_new_tokens == 32
    assert (suite.warmup, suite.repeats) == (1, 3)
    assert suite.batching_methods == (BatchingMethod.NONE, BatchingMethod.FIXED)


def test_serial_vs_fixed_batch_throughput_cases_nest_batch_size_outside_mode():
    cases = get_suite("serial-vs-fixed-batch-throughput").cases()
    assert len(cases) == 5 * 2
    assert [(c.batch_size, c.mode.batching_method) for c in cases[:4]] == [
        (1, BatchingMethod.NONE),
        (1, BatchingMethod.FIXED),
        (2, BatchingMethod.NONE),
        (2, BatchingMethod.FIXED),
    ]
    assert all(c.prompt_len == 128 and c.max_new_tokens == 32 and c.mode.kv_cache for c in cases)


def test_serial_vs_fixed_batch_throughput_to_dict_is_json_serializable():
    import json

    data = get_suite("serial-vs-fixed-batch-throughput").to_dict()
    assert data["batching_methods"] == ["none", "fixed-batching"]
    json.dumps(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_sizes": ()},
        {"batch_sizes": (0,)},
        {"prompt_length": 0},
        {"max_new_tokens": 1},
        {"repeats": 0},
        {"warmup": -1},
        {"batching_methods": ()},
    ],
)
def test_serial_vs_fixed_batch_throughput_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        SerialVsFixedBatchThroughputSuite(**kwargs)


def test_get_suite_unknown_name():
    with pytest.raises(ValueError):
        get_suite("nope")


def test_parse_args_suite():
    from mini_infra.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["bench", "batch", "suite", "--tag", "t", "--suite", "serial-vs-fixed-batch-throughput"])
    assert args.suite == "serial-vs-fixed-batch-throughput"

    # 无法识别的参数直接报错
    for invalid in (["--prompt-lens", "32"], ["--batch-size", "2"], ["--no-kv-cache"]):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["bench", "batch", "suite", "--tag", "t", "--suite", "serial-vs-fixed-batch-throughput", *invalid]
            )

    # 未知 suite 报错
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "batch", "suite", "--tag", "t", "--suite", "unknown"])


def test_parse_args_requires_suite():
    # run_suite 缺少 --suite 时直接报错退出
    from mini_infra.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "batch", "suite", "--tag", "t"])


def test_registered_suites_run_end_to_end_on_cpu():
    """用缩小的 throughput 配置，跑通 serial 与 batch 两条路径的所有用例。"""
    tiny = SerialVsFixedBatchThroughputSuite(batch_sizes=(1, 2), prompt_length=4, max_new_tokens=2, warmup=0, repeats=1)
    model = build_tiny_model()
    vocab = model.config.vocab_size

    results = [
        run_case(
            model,
            case.mode,
            case.prompt_len,
            case.max_new_tokens,
            case.batch_size,
            tiny.warmup,
            tiny.repeats,
            vocab,
            torch.device("cpu"),
        )
        for case in tiny.cases()
    ]

    assert [(r.workload.mode, r.workload.batch_size) for r in results] == [
        ("kv-cache", 1),
        ("kv-cache+fixed-batching", 1),
        ("kv-cache", 2),
        ("kv-cache+fixed-batching", 2),
    ]
    for r in results:
        assert r.workload.prompt_len == 4 and r.workload.max_new_tokens == 2
        assert r.throughput.throughput_req_s.median > 0
        assert r.throughput.throughput_tok_s.median > 0


def test_registry_contains_serial_vs_fixed_batch_throughput():
    assert "serial-vs-fixed-batch-throughput" in SUITES
    suite = SUITES["serial-vs-fixed-batch-throughput"]
    assert isinstance(suite, Suite)


def test_uniform_vs_variable_prompt_padding_defaults():
    suite = get_suite("uniform-vs-variable-prompt-padding")
    assert suite.batch_size == 4
    assert suite.max_prompt_length == 128
    assert suite.max_new_tokens == 32
    assert (suite.warmup, suite.repeats) == (1, 3)
    assert suite.kv_cache is True
    assert suite.batching_method == BatchingMethod.FIXED
    assert suite.name == "uniform-vs-variable-prompt-padding"


def test_uniform_vs_variable_prompt_padding_cases():
    suite = get_suite("uniform-vs-variable-prompt-padding")
    cases = suite.cases()
    assert len(cases) == 2

    # Case 0: uniform (等长)
    homo = cases[0]
    assert homo.batch_size == 4
    assert homo.prompt_len == (128, 128, 128, 128)
    assert homo.max_new_tokens == 32
    assert homo.mode.batching_method == BatchingMethod.FIXED
    assert homo.mode.kv_cache is True

    # Case 1: variable (不等长)
    mixed = cases[1]
    assert mixed.batch_size == 4
    assert mixed.prompt_len == (16, 32, 64, 128)
    assert mixed.max_new_tokens == 32
    assert mixed.mode.batching_method == BatchingMethod.FIXED
    assert mixed.mode.kv_cache is True


def test_uniform_vs_variable_prompt_padding_to_dict_is_json_serializable():
    import json

    data = get_suite("uniform-vs-variable-prompt-padding").to_dict()
    assert data["batch_size"] == 4
    assert data["max_prompt_length"] == 128
    assert data["batching_method"] == "fixed-batching"
    json.dumps(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 2},
        {"batch_size": 8},
        {"max_prompt_length": 0},
        {"max_new_tokens": 1},
        {"repeats": 0},
        {"warmup": -1},
        {"batching_method": BatchingMethod.NONE},
    ],
)
def test_uniform_vs_variable_prompt_padding_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        UniformVsVariablePromptPaddingSuite(**kwargs)


def test_registry_contains_uniform_vs_variable_prompt_padding():
    assert "uniform-vs-variable-prompt-padding" in SUITES
    assert "padding" not in SUITES
    assert "fixed-batch-padding" not in SUITES
    assert isinstance(get_suite("uniform-vs-variable-prompt-padding"), Suite)


def test_uniform_vs_variable_prompt_padding_runs_end_to_end_on_cpu():
    """用缩小的 padding 配置，跑通 uniform 与 variable 两条用例并验证 Padding 有效率。"""
    tiny = UniformVsVariablePromptPaddingSuite(max_prompt_length=4, max_new_tokens=2, warmup=0, repeats=1)
    model = build_tiny_model()
    vocab = model.config.vocab_size

    results = [
        run_case(
            model,
            case.mode,
            case.prompt_len,
            case.max_new_tokens,
            case.batch_size,
            tiny.warmup,
            tiny.repeats,
            vocab,
            torch.device("cpu"),
        )
        for case in tiny.cases()
    ]

    assert len(results) == 2
    res_homo, res_mixed = results

    # uniform: [4, 4, 4, 4]，padding efficiency 应该为 100%
    assert res_homo.workload.batch_size == 4
    assert res_homo.workload.prompt_len == (4, 4, 4, 4)
    assert res_homo.packing is not None and res_homo.packing.padding_efficiency == 1.0

    # variable: [max(1, 4//8), max(1, 4//4), max(1, 4//2), 4] = [1, 1, 2, 4]
    assert res_mixed.workload.batch_size == 4
    assert res_mixed.workload.prompt_len == (1, 1, 2, 4)
    assert res_mixed.packing is not None and res_mixed.packing.padding_efficiency == (1 + 1 + 2 + 4) / 16
    assert res_mixed.packing.padding_efficiency < 1.0


def test_uniform_vs_variable_budget_slots_defaults():
    suite = get_suite("uniform-vs-variable-budget-slots")
    assert suite.batch_size == 4
    assert suite.prompt_length == 128
    assert suite.max_new_tokens == 64
    assert (suite.warmup, suite.repeats) == (1, 3)
    assert suite.kv_cache is True
    assert suite.batching_method == BatchingMethod.FIXED
    assert suite.name == "uniform-vs-variable-budget-slots"


def test_uniform_vs_variable_budget_slots_cases():
    cases = get_suite("uniform-vs-variable-budget-slots").cases()
    assert len(cases) == 2

    # Case 0: uniform budget
    homo = cases[0]
    assert homo.batch_size == 4
    assert homo.prompt_len == (128, 128, 128, 128)
    assert homo.max_new_tokens == (64, 64, 64, 64)
    assert homo.mode.batching_method == BatchingMethod.FIXED
    assert homo.mode.kv_cache is True

    # Case 1: variable length budget
    mixed = cases[1]
    assert mixed.batch_size == 4
    assert mixed.prompt_len == (128, 128, 128, 128)
    assert mixed.max_new_tokens == (8, 16, 32, 64)
    assert mixed.mode.batching_method == BatchingMethod.FIXED
    assert mixed.mode.kv_cache is True


def test_uniform_vs_variable_budget_slots_to_dict_is_json_serializable():
    import json

    data = get_suite("uniform-vs-variable-budget-slots").to_dict()
    assert data["batch_size"] == 4
    assert data["prompt_length"] == 128
    assert data["max_new_tokens"] == 64
    assert data["batching_method"] == "fixed-batching"
    json.dumps(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 2},
        {"batch_size": 8},
        {"prompt_length": 0},
        {"max_new_tokens": 1},
        {"repeats": 0},
        {"warmup": -1},
        {"batching_method": BatchingMethod.NONE},
    ],
)
def test_uniform_vs_variable_budget_slots_rejects_invalid_config(kwargs):
    with pytest.raises(ValueError):
        UniformVsVariableBudgetSlotsSuite(**kwargs)


def test_registry_contains_uniform_vs_variable_budget_slots():
    assert "uniform-vs-variable-budget-slots" in SUITES
    assert isinstance(get_suite("uniform-vs-variable-budget-slots"), Suite)


def test_uniform_vs_variable_budget_slots_runs_end_to_end_on_cpu():
    """用缩小的 slots 配置，跑通 uniform 与 variable 两条用例并验证槽位利用率。"""
    tiny = UniformVsVariableBudgetSlotsSuite(prompt_length=4, max_new_tokens=4, warmup=0, repeats=1)
    model = build_tiny_model()
    vocab = model.config.vocab_size

    results = [
        run_case(
            model,
            case.mode,
            case.prompt_len,
            case.max_new_tokens,
            case.batch_size,
            tiny.warmup,
            tiny.repeats,
            vocab,
            torch.device("cpu"),
        )
        for case in tiny.cases()
    ]

    assert len(results) == 2
    res_uniform, res_variable = results

    # uniform: [4, 4, 4, 4]，slot utilization 应该为 100%
    assert res_uniform.workload.batch_size == 4
    assert res_uniform.workload.prompt_len == (4, 4, 4, 4)
    assert res_uniform.workload.max_new_tokens == (4, 4, 4, 4)
    assert res_uniform.packing is not None and res_uniform.packing.slot_utilization == 1.0

    # variable: [max(2, 4//8), max(2, 4//4), max(2, 4//2), 4] = [2, 2, 2, 4]
    # slot utilization = (2 + 2 + 2 + 4) / 16 = 0.625
    assert res_variable.workload.batch_size == 4
    assert res_variable.workload.prompt_len == (4, 4, 4, 4)
    assert res_variable.workload.max_new_tokens == (2, 2, 2, 4)
    assert res_variable.packing is not None and res_variable.packing.slot_utilization == (2 + 2 + 2 + 4) / 16
    assert res_variable.packing.slot_utilization < 1.0
