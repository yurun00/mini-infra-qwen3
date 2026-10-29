"""`mini-infra run` 参数归一化、调度路径选择与调度器指标输出的 CPU 单元测试。"""

import argparse

import pytest
import torch

from mini_infra.cli.commands.run import (
    add_run_args,
    print_result,
    resolve_budgets,
    run_mode,
    shrink_budgets,
    uses_scheduler,
)
from mini_infra.engine import (
    BatchingMethod,
    FeatureFlag,
    GenerationOptions,
    LockstepResult,
    Mode,
    ScheduleResult,
)

CPU = torch.device("cpu")
KV = {FeatureFlag.KV_CACHE}
SEQS = [[3, 5, 7, 9], [11, 13], [2, 4, 6], [8, 10, 12, 14, 16]]


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(t) for t in token_ids)


def make_opts(**kwargs):
    return GenerationOptions(eos_token_id=-1, device=CPU, stop_on_eos=False, **kwargs)


def test_resolve_budgets():
    assert resolve_budgets([8], 4) == 8
    assert resolve_budgets([8, 24, 8, 24], 4) == [8, 24, 8, 24]
    with pytest.raises(ValueError, match="1 个值或与 prompts 数量一致"):
        resolve_budgets([8, 24], 4)
    with pytest.raises(ValueError, match="必须大于 0"):
        resolve_budgets([0], 4)
    assert shrink_budgets(32, 2) == 2
    assert shrink_budgets([1, 8, 24], 2) == [1, 2, 2]


def test_run_args_parse_per_request_budgets_and_max_running():
    parser = add_run_args(argparse.ArgumentParser())
    args = parser.parse_args(["--max-new-tokens", "8", "24", "8", "24", "--max-running-requests", "2"])
    assert args.max_new_tokens == [8, 24, 8, 24]
    assert args.max_running_requests == 2
    defaults = parser.parse_args([])
    assert defaults.max_new_tokens == [32]
    assert defaults.max_running_requests is None


def test_uses_scheduler_only_when_continuous_present():
    fixed = Mode(KV, batching_method=BatchingMethod.FIXED)
    cont = Mode(KV, batching_method=BatchingMethod.CONTINUOUS)
    assert uses_scheduler([fixed, cont])
    assert not uses_scheduler([fixed, Mode(KV)])


def test_run_mode_uses_scheduler_for_all_methods_when_via_scheduler(diverse_model):
    none_mode = Mode(KV, batching_method=BatchingMethod.NONE)
    fixed = Mode(KV, batching_method=BatchingMethod.FIXED)
    cont = Mode(KV, batching_method=BatchingMethod.CONTINUOUS)
    budgets = [3, 2, 4, 1]
    opts = make_opts(max_running_requests=2)

    none_sched = run_mode(none_mode, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=True)
    fixed_sched = run_mode(fixed, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=True)
    cont_sched = run_mode(cont, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=True)
    fixed_plain = run_mode(fixed, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=False)

    assert isinstance(none_sched, ScheduleResult)
    assert none_sched.batching_method is BatchingMethod.NONE
    assert none_sched.schedule_metrics.max_running_requests == 1
    assert isinstance(fixed_sched, ScheduleResult)
    assert fixed_sched.batching_method is BatchingMethod.FIXED
    assert fixed_sched.schedule_metrics.max_running_requests == 2
    assert isinstance(cont_sched, ScheduleResult)
    assert isinstance(fixed_plain, LockstepResult)
    tokens = [s.token_ids for s in fixed_sched.sequences]
    assert tokens == [s.token_ids for s in cont_sched.sequences]
    assert tokens == [s.token_ids for s in none_sched.sequences]


def test_run_mode_scheduler_rejects_profile_and_missing_kv(diverse_model):
    fixed = Mode(KV, batching_method=BatchingMethod.FIXED)
    with pytest.raises(ValueError, match="profile"):
        run_mode(fixed, diverse_model, SEQS, SEQS, 2, make_opts(profile=True), via_scheduler=True)
    no_kv = Mode(batching_method=BatchingMethod.FIXED)
    with pytest.raises(ValueError, match="KV Cache"):
        run_mode(no_kv, diverse_model, SEQS, SEQS, 2, make_opts(), via_scheduler=True)


def test_print_result_shows_schedule_metrics(diverse_model, capsys):
    cont = Mode(KV, batching_method=BatchingMethod.CONTINUOUS)
    res = run_mode(cont, diverse_model, SEQS, SEQS, [3, 2, 4, 1], make_opts(max_running_requests=2), True)
    print_result(cont, res, FakeTokenizer())
    out = capsys.readouterr().out
    for key in ("Makespan", "TTFT p50/p95", "ITL p50/p95/max", "Cache 管理耗时占比", "并发容量利用率"):
        assert key in out


def test_print_result_lockstep_has_no_schedule_metrics(diverse_model, capsys):
    fixed = Mode(KV, batching_method=BatchingMethod.FIXED)
    res = run_mode(fixed, diverse_model, SEQS, SEQS, 2, make_opts(), via_scheduler=False)
    print_result(fixed, res, FakeTokenizer())
    out = capsys.readouterr().out
    assert "Cache 管理耗时占比" not in out

