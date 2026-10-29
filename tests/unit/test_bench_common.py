"""bench/common.py 共享辅助函数的单元测试（CPU 可运行）。"""

import pytest

from mini_infra.bench.common import format_lengths, resolve_peak_specs
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
def test_resolve_peak_specs_matches_exact_model(gpu_name, expected_bandwidth, expected_tflops):
    peaks = resolve_peak_specs(gpu_name)
    assert peaks == {"bandwidth_gbs": expected_bandwidth, "tflops": expected_tflops}


def test_resolve_peak_specs_unknown_gpu_and_overrides():
    assert resolve_peak_specs("Unknown GPU") == {"bandwidth_gbs": None, "tflops": None}
    assert resolve_peak_specs("Unknown GPU", bandwidth_gbs=100.0, tflops=10.0) == {
        "bandwidth_gbs": 100.0,
        "tflops": 10.0,
    }


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

    from mini_infra.bench.common import DEFAULT_REPEATS, DEFAULT_WARMUP, add_common_args

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
    from mini_infra.bench.common import BENCH_ARTIFACTS_DIR, REPO_ROOT

    assert REPO_ROOT.is_dir()
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert BENCH_ARTIFACTS_DIR == (REPO_ROOT / "benchmarks").resolve()


def test_dataclasses_attributes_and_dict_compatibility():
    from mini_infra.bench.common import (
        CorrectnessReport,
        EfficiencyMetrics,
        HardwarePeakSpecs,
        ModelCostProfile,
    )

    # 1. HardwarePeakSpecs
    peaks = HardwarePeakSpecs(bandwidth_gbs=864.0, tflops=362.0)
    assert peaks.bandwidth_gbs == 864.0
    assert peaks.tflops == 362.0
    assert peaks["bandwidth_gbs"] == 864.0
    assert peaks["tflops"] == 362.0
    assert peaks == {"bandwidth_gbs": 864.0, "tflops": 362.0}
    assert peaks.to_dict() == {"bandwidth_gbs": 864.0, "tflops": 362.0}

    # 2. ModelCostProfile
    cost = ModelCostProfile(
        n_params=600_000_000,
        weight_bytes=1_200_000_000,
        kv_bytes_per_token=2048,
        attn_flop_coeff=16,
    )
    assert cost.n_params == 600_000_000
    assert cost["n_params"] == 600_000_000
    assert cost.to_dict()["kv_bytes_per_token"] == 2048

    # 3. EfficiencyMetrics
    eff = EfficiencyMetrics(prefill_mfu=0.45, decode_mfu=0.32, decode_mbu=0.78)
    assert eff.prefill_mfu == 0.45
    assert eff["decode_mbu"] == 0.78
    assert eff.to_dict()["decode_mfu"] == 0.32

    # 4. CorrectnessReport
    rep = CorrectnessReport(checked=True, passed=True, match_ratio=1.0, updated_golden=False)
    assert rep.checked is True
    assert rep.passed is True
    assert rep["passed"] is True
    assert rep.to_dict() == {"checked": True, "passed": True, "match_ratio": 1.0, "updated_golden": False}


def test_common_strictly_decoupled_from_lockstep():
    import mini_infra.bench.common as bench_common
    from mini_infra.bench.lockstep import (
        build_benchmark_record,
        print_case_summary,
        print_table,
    )

    # 1. common.py 自身不直接反向引入 LockstepCaseResult，解除分层倒置
    assert "LockstepCaseResult" not in bench_common.__dict__

    # 2. common.py 没有任何批处理专属展示/记录函数与兼容钩子
    assert not hasattr(bench_common, "print_table")
    assert not hasattr(bench_common, "print_case_summary")
    assert not hasattr(bench_common, "build_benchmark_record")
    assert not hasattr(bench_common, "_format_case_row")

    # 3. 批处理展示与记录构建函数在 batch 域完整导出且可用
    assert callable(print_table)
    assert callable(print_case_summary)
    assert callable(build_benchmark_record)
