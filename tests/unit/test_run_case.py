"""run_case.py 广播规则与单用例执行测试（CPU 可运行）。"""

from unittest.mock import patch

import pytest
import torch

from mini_infra.bench.run_case import parse_args, resolve_case_specs, run_case
from mini_infra.generation import Mode
from mini_infra.testing.smoke_test import build_tiny_model


def test_resolve_case_specs_scalars():
    bs, pl, mnt = resolve_case_specs([32], [16])
    assert bs == 1
    assert pl == 32
    assert mnt == 16


def test_resolve_case_specs_scalar_broadcast_with_explicit_batch_size():
    bs, pl, mnt = resolve_case_specs([128], [32], batch_size=4)
    assert bs == 4
    assert pl == 128
    assert mnt == 32


def test_resolve_case_specs_variable_prompt_inferred_batch_size():
    bs, pl, mnt = resolve_case_specs([16, 32, 64, 128], [32])
    assert bs == 4
    assert pl == (16, 32, 64, 128)
    assert mnt == 32


def test_resolve_case_specs_variable_output_inferred_batch_size():
    bs, pl, mnt = resolve_case_specs([128], [8, 16, 32, 64])
    assert bs == 4
    assert pl == 128
    assert mnt == (8, 16, 32, 64)


def test_resolve_case_specs_both_variable_inferred_batch_size():
    bs, pl, mnt = resolve_case_specs([16, 32], [8, 16])
    assert bs == 2
    assert pl == (16, 32)
    assert mnt == (8, 16)


def test_resolve_case_specs_shape_mismatches():
    # prompt 与 output 长度不相等且均 > 1
    with pytest.raises(ValueError, match="不匹配"):
        resolve_case_specs([16, 32], [8, 16, 32])

    # 与显式 batch_size 不匹配
    with pytest.raises(ValueError, match="既不为 1，也不等于指定的 batch_size"):
        resolve_case_specs([16, 32], [8], batch_size=3)

    with pytest.raises(ValueError, match="既不为 1，也不等于指定的 batch_size"):
        resolve_case_specs([16], [8, 16], batch_size=3)

    # 长度非法
    with pytest.raises(ValueError):
        resolve_case_specs([], [8])

    with pytest.raises(ValueError):
        resolve_case_specs([16], [1])  # max_new_tokens 至少为 2


def test_parse_args_run_case():
    with patch("sys.argv", ["mini-infra-bench-case", "--tag", "dev", "--prompt-lens", "32", "--max-new-tokens", "16"]):
        args = parse_args()
    assert args.tag == "dev"
    assert args.prompt_lens == [32]
    assert args.max_new_tokens == [16]
    assert args.batch_size is None


def test_run_case_on_cpu():
    model = build_tiny_model()
    vocab = model.config.vocab_size
    mode = Mode()

    # 1. 单用例标量执行
    res_scalar = run_case(
        model=model,
        mode=mode,
        prompt_len=4,
        max_new_tokens=2,
        batch_size=1,
        warmup=0,
        repeats=1,
        vocab=vocab,
        device=torch.device("cpu"),
    )
    assert res_scalar["batch_size"] == 1
    assert res_scalar["prompt_len"] == 4
    assert res_scalar["max_new_tokens"] == 2
    assert res_scalar["throughput_tok_s"]["median"] > 0
    assert res_scalar["throughput_req_s"]["median"] > 0

    # 2. 变长 prompt 执行
    res_var = run_case(
        model=model,
        mode=mode,
        prompt_len=[2, 4],
        max_new_tokens=2,
        batch_size=2,
        warmup=0,
        repeats=1,
        vocab=vocab,
        device=torch.device("cpu"),
    )
    assert res_var["batch_size"] == 2
    assert res_var["prompt_len"] == [2, 4]
    assert res_var["max_new_tokens"] == 2
