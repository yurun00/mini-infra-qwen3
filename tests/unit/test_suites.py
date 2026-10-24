"""suites.py 配置与 run_suite.py 接入的单元测试（CPU 可运行）。"""

from unittest.mock import patch

import pytest
import torch

from mini_infra.bench.run_case import run_case
from mini_infra.bench.run_suite import parse_args
from mini_infra.bench.suites import (
    SUITES,
    SerialVsFixedBatchThroughputSuite,
    Suite,
    get_suite,
)
from mini_infra.generation import BatchingMethod
from mini_infra.testing.smoke_test import build_tiny_model


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
    with patch("sys.argv", ["mini-infra-bench-suite", "--tag", "t", "--suite", "serial-vs-fixed-batch-throughput"]):
        args = parse_args()
    assert args.suite == "serial-vs-fixed-batch-throughput"

    # 无法识别的参数直接报错
    for invalid in (["--prompt-lens", "32"], ["--batch-size", "2"], ["--no-kv-cache"]):
        with patch(
            "sys.argv",
            ["mini-infra-bench-suite", "--tag", "t", "--suite", "serial-vs-fixed-batch-throughput", *invalid],
        ):
            with pytest.raises(SystemExit):
                parse_args()

    # 未知 suite 报错
    with patch("sys.argv", ["mini-infra-bench-suite", "--tag", "t", "--suite", "unknown"]):
        with pytest.raises(SystemExit):
            parse_args()


def test_parse_args_requires_suite():
    # run_suite 缺少 --suite 时直接报错退出
    with patch("sys.argv", ["mini-infra-bench-suite", "--tag", "t"]):
        with pytest.raises(SystemExit):
            parse_args()


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

    assert [(r["mode"], r["batch_size"]) for r in results] == [
        ("kv-cache", 1),
        ("kv-cache+fixed-batching", 1),
        ("kv-cache", 2),
        ("kv-cache+fixed-batching", 2),
    ]
    for r in results:
        assert r["prompt_len"] == 4 and r["max_new_tokens"] == 2
        assert r["throughput_req_s"]["median"] > 0
        assert r["throughput_tok_s"]["median"] > 0


def test_registry_contains_serial_vs_fixed_batch_throughput():
    assert "serial-vs-fixed-batch-throughput" in SUITES
    suite = SUITES["serial-vs-fixed-batch-throughput"]
    assert isinstance(suite, Suite)
