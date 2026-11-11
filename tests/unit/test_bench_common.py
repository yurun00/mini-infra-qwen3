"""bench 共享基础设施子模块（paths / stats / efficiency / env / correctness / cases）的单元测试（CPU 可运行）。"""

import sys

import pytest

from mini_infra.bench.cases import DEFAULT_REPEATS, DEFAULT_WARMUP
from mini_infra.bench.correctness import CorrectnessReport
from mini_infra.bench.efficiency import EfficiencyMetrics, HardwarePeakSpecs, ModelCostProfile
from mini_infra.bench.format import format_lengths
from mini_infra.cli.commands.bench.args import add_common_args
from mini_infra.utils import percentile


@pytest.mark.parametrize(
    ("gpu_name", "expected_bandwidth", "expected_tflops"),
    [
        ("NVIDIA A100-SXM4-80GB", 1555.0, 312.0),  # 不能误命中 A10
        ("NVIDIA A10G", 600.0, 125.0),
        ("NVIDIA A10", 600.0, 125.0),
        ("NVIDIA L40S", 864.0, 362.0),  # 不能误命中 L4
        ("NVIDIA L4", 300.0, 121.0),
        ("Tesla T4", 320.0, 65.0),
        ("NVIDIA H100 80GB HBM3", 3350.0, 989.0),
        ("NVIDIA GeForce RTX 4090", 1008.0, 165.0),
    ],
)
def test_lookup_gpu_peak_specs_matches_exact_model(gpu_name, expected_bandwidth, expected_tflops):
    peaks = HardwarePeakSpecs.lookup(gpu_name)
    assert peaks == HardwarePeakSpecs(bandwidth_gbs=expected_bandwidth, tflops=expected_tflops)
    assert peaks.to_dict() == {"bandwidth_gbs": expected_bandwidth, "tflops": expected_tflops}


def test_lookup_gpu_peak_specs_unknown_gpu_and_overrides():
    assert HardwarePeakSpecs.lookup("Unknown GPU") == HardwarePeakSpecs(bandwidth_gbs=None, tflops=None)
    assert HardwarePeakSpecs.lookup("Unknown GPU", bandwidth_gbs=100.0, tflops=10.0) == HardwarePeakSpecs(
        bandwidth_gbs=100.0,
        tflops=10.0,
    )


def test_format_lengths_scalar_and_sequence():
    assert format_lengths(128) == "128"
    assert format_lengths((16, 32, 64)) == "16/32/64"
    assert format_lengths([8, 16]) == "8/16"


def test_percentile_supports_float_percent_and_matches_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == 2.0
    assert percentile(values, 95) == 4.0
    assert percentile(values, 99.5) == 4.0
    assert percentile([], 50) == 0.0


def test_add_common_args_includes_warmup_and_repeats():
    import argparse

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    default_args = parser.parse_args(["--tag", "test"])
    assert default_args.warmup == DEFAULT_WARMUP == 3
    assert default_args.repeats == DEFAULT_REPEATS == 10

    args = parser.parse_args(["--tag", "test", "--warmup", "2", "--repeats", "5"])
    assert args.tag == "test"
    assert args.warmup == 2
    assert args.repeats == 5


def test_repo_root_anchored_to_project_root():
    from mini_infra.bench.paths import (
        BENCH_ARTIFACTS_DIR,
        LOCKSTEP_RESULT_HISTORY_DIR,
        REPO_ROOT,
        RESULT_HISTORY_DIR,
        RESULTS_DIR,
        SCHEDULER_RESULT_HISTORY_DIR,
        SUITES_DIR,
    )

    assert REPO_ROOT.is_dir()
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert BENCH_ARTIFACTS_DIR == (REPO_ROOT / "benchmarks").resolve()
    assert RESULTS_DIR == BENCH_ARTIFACTS_DIR / "results"
    assert RESULT_HISTORY_DIR == RESULTS_DIR / "history"
    assert SUITES_DIR == RESULTS_DIR / "suites"
    assert LOCKSTEP_RESULT_HISTORY_DIR == RESULT_HISTORY_DIR / "lockstep"
    assert SCHEDULER_RESULT_HISTORY_DIR == RESULT_HISTORY_DIR / "scheduler"


def test_dataclasses_attributes_and_to_dict():
    # 1. HardwarePeakSpecs
    peaks = HardwarePeakSpecs(bandwidth_gbs=864.0, tflops=362.0)
    assert peaks.bandwidth_gbs == 864.0
    assert peaks.tflops == 362.0
    assert peaks.to_dict() == {"bandwidth_gbs": 864.0, "tflops": 362.0}
    with pytest.raises(TypeError):
        _ = peaks["bandwidth_gbs"]  # type: ignore[index]

    # 2. ModelCostProfile
    cost = ModelCostProfile(
        n_params=600_000_000,
        weight_bytes=1_200_000_000,
        kv_bytes_per_token=2048,
        attn_flop_coeff=16,
    )
    assert cost.n_params == 600_000_000
    assert cost.to_dict()["kv_bytes_per_token"] == 2048
    with pytest.raises(TypeError):
        _ = cost["n_params"]  # type: ignore[index]

    # 3. EfficiencyMetrics
    eff = EfficiencyMetrics(prefill_mfu=0.45, decode_mfu=0.32, decode_mbu=0.78)
    assert eff.prefill_mfu == 0.45
    assert eff.decode_mbu == 0.78
    assert eff.to_dict()["decode_mfu"] == 0.32
    with pytest.raises(TypeError):
        _ = eff["decode_mbu"]  # type: ignore[index]

    # 4. CorrectnessReport
    rep = CorrectnessReport(checked=True, passed=True, match_ratio=1.0, updated_golden=False)
    assert rep.checked is True
    assert rep.passed is True
    assert rep.to_dict() == {"checked": True, "passed": True, "match_ratio": 1.0, "updated_golden": False}
    with pytest.raises(TypeError):
        _ = rep["passed"]  # type: ignore[index]


def test_shared_modules_strictly_decoupled_from_engines():
    import mini_infra.bench.correctness as bench_correctness
    import mini_infra.bench.efficiency as bench_efficiency
    import mini_infra.bench.env as bench_env
    import mini_infra.bench.paths as bench_paths
    import mini_infra.bench.result.schema as bench_result_schema
    import mini_infra.bench.stats as bench_stats
    from mini_infra.bench.lockstep import (
        LockstepCaseResult,
        format_table,
    )
    from mini_infra.bench.result import build_benchmark_record, save_benchmark_record

    # 0. 旧上帝模块 bench.common 已彻底删除
    assert "mini_infra.bench.common" not in sys.modules

    # 1. 各共享基础设施模块不直接反向引入引擎结果类或表格展示钩子
    for mod in (bench_paths, bench_stats, bench_efficiency, bench_env, bench_correctness, bench_result_schema):
        assert "LockstepCaseResult" not in mod.__dict__
        assert "SchedulerCaseResult" not in mod.__dict__
        assert not hasattr(mod, "format_table")
        assert not hasattr(mod, "build_benchmark_record")
        assert not hasattr(mod, "add_common_args")

    # 2. paths、stats 与 result.schema 完全不依赖 torch
    assert "torch" not in bench_paths.__dict__
    assert "torch" not in bench_stats.__dict__
    assert "torch" not in bench_result_schema.__dict__

    # 3. 批处理展示与记录构建/保存在各自权威模块完整导出且可用
    assert callable(format_table)
    assert callable(getattr(LockstepCaseResult, "format_summary", None))
    assert callable(build_benchmark_record)
    assert callable(save_benchmark_record)
