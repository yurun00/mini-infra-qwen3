"""`mini-infra run` 参数归一化、调度路径选择与调度器指标输出的 CPU 单元测试。"""

import argparse

import pytest
import torch

from mini_infra.cli.commands.run import (
    add_run_args,
    print_result,
    resolve_budgets,
    resolve_compare_feature,
    run_config,
    shrink_budgets,
    uses_scheduler,
)
from mini_infra.core import DenseCacheConfig, NoCacheConfig, PagedCacheConfig
from mini_infra.engine import (
    BatchingMethod,
    EngineConfig,
    FeatureFlag,
    GenerationResult,
    ScheduleResult,
)
from mini_infra.entrypoints import GenerationOptions

CPU = torch.device("cpu")
KV = DenseCacheConfig()
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
    fixed = EngineConfig(KV, batching_method=BatchingMethod.FIXED)
    cont = EngineConfig(KV, batching_method=BatchingMethod.CONTINUOUS)
    assert uses_scheduler([fixed, cont])
    assert not uses_scheduler([fixed, EngineConfig(KV)])


def test_run_config_uses_scheduler_for_all_methods_when_via_scheduler(diverse_model):
    none_cfg = EngineConfig(KV, batching_method=BatchingMethod.NONE, max_running_requests=2)
    fixed = EngineConfig(KV, batching_method=BatchingMethod.FIXED, max_running_requests=2)
    cont = EngineConfig(KV, batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2)
    budgets = [3, 2, 4, 1]
    opts = make_opts()

    none_sched = run_config(none_cfg, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=True)
    fixed_sched = run_config(fixed, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=True)
    cont_sched = run_config(cont, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=True)
    fixed_plain = run_config(fixed, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=False)

    assert isinstance(none_sched, ScheduleResult)
    assert none_sched.batching_method is BatchingMethod.NONE
    assert none_sched.schedule_metrics.max_running_requests == 1
    assert isinstance(fixed_sched, ScheduleResult)
    assert fixed_sched.batching_method is BatchingMethod.FIXED
    assert fixed_sched.schedule_metrics.max_running_requests == 2
    assert isinstance(cont_sched, ScheduleResult)
    assert isinstance(fixed_plain, GenerationResult)
    assert not isinstance(fixed_plain, ScheduleResult)
    tokens = [s.token_ids for s in fixed_sched.sequences]
    assert tokens == [s.token_ids for s in cont_sched.sequences]
    assert tokens == [s.token_ids for s in none_sched.sequences]


def test_run_config_scheduler_rejects_profile_and_missing_kv(diverse_model):
    fixed = EngineConfig(KV, batching_method=BatchingMethod.FIXED)
    with pytest.raises(ValueError, match="profile"):
        run_config(fixed, diverse_model, SEQS, SEQS, 2, make_opts(profile=True), via_scheduler=True)
    no_kv = EngineConfig(batching_method=BatchingMethod.FIXED)
    with pytest.raises(ValueError, match="KV Cache"):
        run_config(no_kv, diverse_model, SEQS, SEQS, 2, make_opts(), via_scheduler=True)


def test_print_result_shows_schedule_metrics(diverse_model, capsys):
    cont = EngineConfig(KV, batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2)
    res = run_config(cont, diverse_model, SEQS, SEQS, [3, 2, 4, 1], make_opts(), True)
    print_result(cont, res, FakeTokenizer())
    out = capsys.readouterr().out
    for key in (
        "Makespan",
        "TTFT p50/p95",
        "ITL p50/p95/max",
        "Cache 管理耗时占比",
        "并发容量利用率",
        "峰值 KV Cache Pool",
    ):
        assert key in out


def test_run_config_and_print_result_dense_vs_paged(diverse_model, capsys):
    dense_cfg = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2)
    paged_cfg = EngineConfig(
        PagedCacheConfig(block_size=4), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2
    )
    assert uses_scheduler([dense_cfg, paged_cfg])

    budgets = [3, 2, 4, 1]
    opts = make_opts()
    dense_res = run_config(dense_cfg, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=True)
    paged_res = run_config(paged_cfg, diverse_model, SEQS, SEQS, budgets, opts, via_scheduler=True)

    assert [s.token_ids for s in dense_res.sequences] == [s.token_ids for s in paged_res.sequences]
    print_result(paged_cfg, paged_res, FakeTokenizer())
    out = capsys.readouterr().out
    assert "峰值 KV Cache Pool" in out
    assert "Paged physical block pool: block_size=4" in out


def test_print_result_lockstep_has_no_schedule_metrics(diverse_model, capsys):
    fixed = EngineConfig(KV, batching_method=BatchingMethod.FIXED)
    res = run_config(fixed, diverse_model, SEQS, SEQS, 2, make_opts(), via_scheduler=False)
    print_result(fixed, res, FakeTokenizer())
    out = capsys.readouterr().out
    assert "Cache 管理耗时占比" not in out


def test_resolve_compare_feature():
    base = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS)

    configs, name = resolve_compare_feature(base, ["placeholder"])
    assert name == "placeholder"
    assert FeatureFlag.PLACEHOLDER not in configs[0].flags
    assert FeatureFlag.PLACEHOLDER in configs[1].flags

    configs, name = resolve_compare_feature(base, ["KVCacheConfig", "none", "paged-kv-cache"], block_size=8)
    assert name == "none vs paged-kv-cache"
    assert configs[0].kv_cache_config == NoCacheConfig()
    assert configs[1].kv_cache_config == PagedCacheConfig(block_size=8)

    configs, name = resolve_compare_feature(base, ["BatchingMethod", "fixed-batching", "continuous-batching"])
    assert name == "fixed-batching vs continuous-batching"
    assert configs[0].batching_method is BatchingMethod.FIXED
    assert configs[1].batching_method is BatchingMethod.CONTINUOUS

    for invalid in (
        [],
        ["unknown"],
        ["dense-kv-cache"],
        ["placeholder", "extra"],
        ["KVCacheConfig", "dense-kv-cache"],
        ["KVCacheConfig", "dense-kv-cache", "dense-kv-cache"],
        ["KVCacheConfig", "dense-kv-cache", "invalid"],
        ["BatchingMethod", "fixed", "continuous"],
    ):
        with pytest.raises(ValueError):
            resolve_compare_feature(base, invalid)
