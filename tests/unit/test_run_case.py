"""run_case.py 广播规则与单用例执行测试（CPU 可运行）。"""

import pytest
import torch

from mini_infra.bench.cases import BenchmarkCase, WorkloadProfile
from mini_infra.bench.lockstep.run_case import broadcast_case_lengths, run_case, run_case_once
from mini_infra.core.request import RequestSpec
from mini_infra.engine import Mode
from mini_infra.verification.smoke import build_tiny_model


def test_broadcast_case_lengths_scalars():
    bs, pl, mnt = broadcast_case_lengths([32], [16])
    assert bs == 1
    assert pl == 32
    assert mnt == 16


def test_broadcast_case_lengths_scalar_broadcast_with_explicit_batch_size():
    bs, pl, mnt = broadcast_case_lengths([128], [32], batch_size=4)
    assert bs == 4
    assert pl == 128
    assert mnt == 32


def test_broadcast_case_lengths_variable_prompt_inferred_batch_size():
    bs, pl, mnt = broadcast_case_lengths([16, 32, 64, 128], [32])
    assert bs == 4
    assert pl == (16, 32, 64, 128)
    assert mnt == 32


def test_broadcast_case_lengths_variable_output_inferred_batch_size():
    bs, pl, mnt = broadcast_case_lengths([128], [8, 16, 32, 64])
    assert bs == 4
    assert pl == 128
    assert mnt == (8, 16, 32, 64)


def test_broadcast_case_lengths_both_variable_inferred_batch_size():
    bs, pl, mnt = broadcast_case_lengths([16, 32], [8, 16])
    assert bs == 2
    assert pl == (16, 32)
    assert mnt == (8, 16)


def test_broadcast_case_lengths_shape_mismatches():
    # prompt 与 output 长度不相等且均 > 1
    with pytest.raises(ValueError, match="不匹配"):
        broadcast_case_lengths([16, 32], [8, 16, 32])

    # 与显式 batch_size 不匹配
    with pytest.raises(ValueError, match="既不为 1，也不等于指定的 batch_size"):
        broadcast_case_lengths([16, 32], [8], batch_size=3)

    with pytest.raises(ValueError, match="既不为 1，也不等于指定的 batch_size"):
        broadcast_case_lengths([16], [8, 16], batch_size=3)

    # 长度非法
    with pytest.raises(ValueError):
        broadcast_case_lengths([], [8])

    with pytest.raises(ValueError):
        broadcast_case_lengths([16], [1])  # max_new_tokens 至少为 2


def test_parse_args_run_case():
    from mini_infra.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["bench", "lockstep", "case", "--tag", "dev", "--prompt-lens", "32", "--max-new-tokens", "16"]
    )
    assert args.tag == "dev"
    assert args.kv_cache is True
    assert args.prompt_lens == [32]
    assert args.max_new_tokens == [16]
    assert args.batch_size is None
    assert args.warmup == 3
    assert args.repeats == 10


def test_run_case_on_cpu():
    model = build_tiny_model()
    mode = Mode()

    # 1. 单用例标量执行
    spec1 = RequestSpec(request_idx=0, token_ids=[1, 2, 3, 4], max_new_tokens=2, arrival_ms=0.0)
    case_scalar = BenchmarkCase(
        profile=WorkloadProfile.from_specs("s1", (spec1,)),
        mode=mode,
        engine="lockstep",
    )

    # 验证 run_case_once
    once_res = run_case_once(model, torch.device("cpu"), case_scalar)
    assert once_res.metrics.throughput_tok_s > 0

    res_scalar = run_case(
        model=model,
        device=torch.device("cpu"),
        case=case_scalar,
        warmup=0,
        repeats=1,
    )
    assert res_scalar.key.mode == mode
    assert isinstance(res_scalar.workload, WorkloadProfile)
    assert res_scalar.workload.num_total_requests == 1
    assert res_scalar.workload.prompt_lens == (4,)
    assert res_scalar.workload.output_budgets == (2,)
    assert res_scalar.metrics.throughput_tok_s.median > 0
    assert res_scalar.metrics.throughput_req_s.median > 0
    rec_scalar = res_scalar.to_record_dict()
    assert rec_scalar["num_total_requests"] == 1
    assert rec_scalar["prompt_len"] == 4
    assert rec_scalar["max_new_tokens"] == 2

    # 2. 变长 prompt 执行
    s1 = RequestSpec(request_idx=0, token_ids=[1, 2], max_new_tokens=2, arrival_ms=0.0)
    s2 = RequestSpec(request_idx=1, token_ids=[3, 4, 5, 6], max_new_tokens=2, arrival_ms=0.0)
    case_var = BenchmarkCase(
        profile=WorkloadProfile.from_specs("v1", (s1, s2)),
        mode=mode,
        engine="lockstep",
    )
    res_var = run_case(
        model=model,
        device=torch.device("cpu"),
        case=case_var,
        warmup=0,
        repeats=1,
    )
    assert res_var.key.mode == mode
    assert res_var.workload.num_total_requests == 2
    assert res_var.workload.prompt_lens == (2, 4)
    assert res_var.workload.output_budgets == (2, 2)
    rec_var = res_var.to_record_dict()
    assert rec_var["num_total_requests"] == 2
    assert rec_var["prompt_len"] == [2, 4]
    assert rec_var["max_new_tokens"] == 2
