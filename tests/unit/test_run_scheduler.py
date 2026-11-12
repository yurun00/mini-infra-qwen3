"""bench.scheduler.run_case 与 run_suite 的单元测试（CPU 可运行，使用 tiny_model）。"""

from dataclasses import dataclass
from typing import Any

import torch

from mini_infra.bench.cases import BenchmarkCase, CaseDimension, WorkloadProfile
from mini_infra.bench.correctness import CorrectnessReport
from mini_infra.bench.efficiency import HardwarePeakSpecs
from mini_infra.bench.result import build_benchmark_record
from mini_infra.bench.scheduler import (
    extract_workload_profile,
    format_table,
    run_case,
    run_case_once,
    run_suite,
)
from mini_infra.bench.suite import load_benchmark_suite
from mini_infra.core import DenseCacheConfig, PagedCacheConfig, make_requests
from mini_infra.engine import BatchingMethod, CachingMethod, Mode
from mini_infra.verification.smoke import build_tiny_model


def test_parse_args_scheduler_sweep():
    from mini_infra.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "sweep",
            "--engine",
            "scheduler",
            "--tag",
            "dev",
            "--suite",
            "fixed-vs-continuous-variable-output-length",
            "--repeats",
            "2",
            "--interleave",
            "batching_method",
            "--dry-run",
        ]
    )
    assert args.tag == "dev"
    assert args.engine == "scheduler"
    assert args.suite == ["fixed-vs-continuous-variable-output-length"]
    assert args.repeats == 2
    assert args.interleave == ["batching_method"]
    assert args.dry_run is True


def test_scheduler_suite_subcommand_removed():
    import pytest

    from mini_infra.cli import build_parser

    parser = build_parser()
    # 验证旧的 bench scheduler suite 子命令已被彻底移除（由 bench sweep 统一接管）
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "scheduler", "suite", "--tag", "dev"])


def test_parse_args_scheduler_case():
    from mini_infra.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "scheduler",
            "case",
            "--tag",
            "dev",
            "--batching-method",
            "continuous-batching",
            "--max-running-requests",
            "2",
        ]
    )
    assert args.tag == "dev"
    assert args.caching_method == "dense-kv-cache"
    assert args.batching_method == "continuous-batching"
    assert args.block_size == 16
    assert args.num_blocks is None
    assert args.max_running_requests == 2

    paged_args = parser.parse_args(
        [
            "bench",
            "scheduler",
            "case",
            "--tag",
            "dev",
            "--caching-method",
            "paged-kv-cache",
            "--batching-method",
            "continuous-batching",
            "--block-size",
            "32",
            "--num-blocks",
            "128",
            "--max-running-requests",
            "4",
        ]
    )
    assert paged_args.caching_method == "paged-kv-cache"
    assert paged_args.block_size == 32
    assert paged_args.num_blocks == 128
    assert paged_args.max_running_requests == 4


def test_handle_scheduler_case_paged_on_cpu(tmp_path, monkeypatch):
    """验证 handle_case 在 paged-kv-cache 模式下透传 PagedCacheConfig 并落盘。"""
    import json

    from mini_infra.bench.correctness import CorrectnessReport
    from mini_infra.cli import build_parser
    from mini_infra.cli.commands.bench import scheduler as sched_cmd
    from mini_infra.core.block_manager import PagedCacheConfig

    tiny_model = build_tiny_model()
    captured_correctness_kwargs = {}

    def fake_check_correctness(model, tokenizer, mode, device, update_golden=False, kv_cache_config=None):
        captured_correctness_kwargs["mode"] = mode
        captured_correctness_kwargs["kv_cache_config"] = kv_cache_config
        return CorrectnessReport(checked=True, passed=True, match_ratio=1.0, updated_golden=False)

    monkeypatch.setattr(sched_cmd, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(sched_cmd, "resolve_model_directory", lambda model, rev: tmp_path)
    monkeypatch.setattr(sched_cmd, "Qwen3Tokenizer", lambda model_dir: object())
    monkeypatch.setattr(sched_cmd, "load_model", lambda model_dir, device: tiny_model)
    monkeypatch.setattr(sched_cmd, "check_correctness", fake_check_correctness)
    monkeypatch.setattr(sched_cmd, "gpu_info", lambda device: {"gpu": "CPU", "torch": torch.__version__})

    out_path = tmp_path / "paged_case.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "scheduler",
            "case",
            "--tag",
            "dev",
            "--caching-method",
            "paged-kv-cache",
            "--batching-method",
            "continuous-batching",
            "--block-size",
            "4",
            "--num-blocks",
            "32",
            "--request-count",
            "2",
            "--prompt-length",
            "4",
            "--output-budget",
            "2",
            "--max-running-requests",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--output",
            str(out_path),
        ]
    )
    ret = sched_cmd.handle_case(args)
    assert ret == 0
    assert captured_correctness_kwargs["mode"] == Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
    assert captured_correctness_kwargs["kv_cache_config"] == PagedCacheConfig(block_size=4, num_blocks=32)
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "paged-kv-cache+continuous-batching"
    assert saved["cases"][0]["memory"]["kv_cache"]["paged_cache"]["block_size"] == 4
    assert saved["cases"][0]["memory"]["kv_cache"]["paged_cache"]["num_blocks"] == 32


def test_handle_scheduler_case_default_output_to_case_dir(tmp_path, monkeypatch):
    """验证 handle_case 未指定 --output 时默认写入 CASE_DIR (benchmarks/case)。"""
    from mini_infra.bench.correctness import CorrectnessReport
    from mini_infra.cli import build_parser
    from mini_infra.cli.commands.bench import scheduler as sched_cmd

    tiny_model = build_tiny_model()
    monkeypatch.setattr(sched_cmd, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(sched_cmd, "resolve_model_directory", lambda model, rev: tmp_path)
    monkeypatch.setattr(sched_cmd, "Qwen3Tokenizer", lambda model_dir: object())
    monkeypatch.setattr(sched_cmd, "load_model", lambda model_dir, device: tiny_model)
    monkeypatch.setattr(
        sched_cmd,
        "check_correctness",
        lambda *args, **kwargs: CorrectnessReport(checked=True, passed=True, match_ratio=1.0, updated_golden=False),
    )
    monkeypatch.setattr(sched_cmd, "gpu_info", lambda device: {"gpu": "CPU", "torch": torch.__version__})

    fake_case_dir = tmp_path / "case"
    monkeypatch.setattr(sched_cmd, "CASE_DIR", fake_case_dir)

    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "scheduler",
            "case",
            "--tag",
            "test-tag",
            "--caching-method",
            "dense-kv-cache",
            "--batching-method",
            "continuous-batching",
            "--request-count",
            "2",
            "--prompt-length",
            "4",
            "--output-budget",
            "2",
            "--max-running-requests",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
        ]
    )
    ret = sched_cmd.handle_case(args)
    assert ret == 0
    saved_files = list(fake_case_dir.glob("*_test-tag.json"))
    assert len(saved_files) == 1
    assert saved_files[0].exists()


def test_run_scheduler_case_and_step_on_cpu():
    """验证原子 step 与单用例 run_case 仅执行单一策略。"""
    model = build_tiny_model()
    device = torch.device("cpu")
    requests = tuple(make_requests([[1, 2, 3], [4, 5]], [2, 3], [0.0, 10.0]))
    mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
    case = BenchmarkCase(
        profile=WorkloadProfile.from_requests("test-cont-case", requests),
        batching_method=BatchingMethod.CONTINUOUS,
        kv_cache_config=DenseCacheConfig(),
        max_running_requests=2,
        suite_name="test-cont-case",
        engine="scheduler",
    )

    # 1. 验证原子 run_case_once 与 extract_workload_profile
    single_res = run_case_once(model, device, case)
    assert single_res.metrics.e2e_wall_ms > 0
    assert len(single_res.sequences) == 2
    sw = extract_workload_profile(single_res, workload_id=case.workload_id)
    assert sw.num_total_requests == 2

    # 2. 验证多轮 run_case
    case_res = run_case(
        model=model,
        device=device,
        case=case,
        warmup=1,
        repeats=2,
    )

    assert case_res.key.mode == mode
    assert case_res.key.workload_id == "test-cont-case"
    assert case_res.mode == mode
    assert case_res.repeats == 2
    assert case_res.warmup == 1
    assert case_res.workload == sw
    assert case_res.metrics.e2e_wall_ms.mean > 0
    assert case_res.metrics.throughput_req_s.mean > 0
    assert case_res.schedule_metrics.ttft_ms_p95.mean > 0
    assert case_res.memory.kv_cache.peak_live_kv_cache_mb > 0
    assert case_res.packing is not None
    assert case_res.packing.slot_utilization.mean > 0
    assert case_res.packing.running_capacity_utilization is not None
    assert case_res.packing.running_capacity_utilization.mean > 0
    assert len(case_res.sample_events) > 0
    print(case_res.format_summary())


def test_run_scheduler_suite_on_cpu(tmp_path):
    """验证套件执行器 run_suite 支持 plan 驱动并导出统一 SchedulerBenchmarkRecord。"""
    model = build_tiny_model()
    device = torch.device("cpu")
    spec_path = tmp_path / "tiny-fixed-vs-continuous.yaml"
    spec_path.write_text(
        """
name: tiny-fixed-vs-continuous
execution: {warmup: 0, repeats: 2, interleave: [batching_method]}
system:
  engine_mode:
    kind: scheduler
    max_running_requests: 4
  batching_method: fixed-batching
  caching_spec: dense-kv-cache
sweep:
  system:
    batching_method: [fixed-batching, continuous-batching]
workload:
  id: variable_output_length
  count: 2
  prompt_len: 4
  output_len: {alternating: [2, 3]}
""",
        encoding="utf-8",
    )
    suite = load_benchmark_suite(spec_path, expected_engine="scheduler")

    cases = run_suite(
        model=model,
        device=device,
        suite=suite,
        vocab_size=model.config.vocab_size,
        warmup=0,
        repeats=2,
        interleave=(CaseDimension.BATCHING_METHOD,),
    )

    assert len(cases) == 2
    assert {c.key.mode.batching_method for c in cases} == {BatchingMethod.FIXED, BatchingMethod.CONTINUOUS}

    # 验证控制台表格打印无异常
    print(format_table(cases, suite_name=suite.name))

    # 验证完整结果构建
    class DummyArgs:
        tag = "test-tag"
        suite = "fixed-vs-continuous-variable-output-length"
        model = "dummy-model"
        revision = "main"

    record = build_benchmark_record(
        args=DummyArgs(),
        model=model,
        mode_label="suite(kv-cache+fixed-batching,kv-cache+continuous-batching)",
        cases=cases,
        correctness=CorrectnessReport(checked=True, passed=True, match_ratio=1.0, updated_golden=False),
        env={"gpu": "CPU"},
        peaks=HardwarePeakSpecs(bandwidth_gbs=None, tflops=None),
        cost=None,
        git={"commit": "abc", "dirty": False},
        warmup=0,
        repeats=2,
        suite_info=suite.to_dict(),
        engine="scheduler",
    )
    assert record.engine == "scheduler"
    assert record.suite is not None
    assert record.suite["name"] == "tiny-fixed-vs-continuous"
    assert record.suite["spec"]["workload"]["count"] == 2
    assert len(record.cases) == 2


def test_run_scheduler_paged_case_on_cpu():
    """验证 run_case_once 和 run_case 支持 Paged 缓存模式并正确采集 paged_cache 画像。"""
    model = build_tiny_model()
    device = torch.device("cpu")
    requests = tuple(make_requests([[1, 2, 3, 4], [5, 6]], [2, 3], [0.0, 5.0]))
    mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
    paged_config = PagedCacheConfig(block_size=4)

    case = BenchmarkCase(
        profile=WorkloadProfile.from_requests("stress_case", requests),
        batching_method=BatchingMethod.CONTINUOUS,
        kv_cache_config=paged_config,
        max_running_requests=2,
        suite_name="test-paged-case",
        engine="scheduler",
    )

    # 1. 单次执行
    single_res = run_case_once(model, device, case)
    assert single_res.metrics.e2e_wall_ms > 0
    assert len(single_res.sequences) == 2
    assert single_res.memory.kv_cache.paged_cache is not None
    assert single_res.memory.kv_cache.paged_cache.block_size == 4
    assert single_res.memory.kv_cache.paged_cache.peak_used_blocks > 0

    # 2. 多轮执行
    case_res = run_case(
        model=model,
        device=device,
        case=case,
        warmup=1,
        repeats=2,
    )
    assert case_res.key.mode == mode
    assert case_res.key.workload_id == "stress_case"
    assert case_res.mode == mode
    assert case_res.memory.kv_cache.paged_cache is not None
    assert case_res.memory.kv_cache.paged_cache.block_size == 4
    assert case_res.memory.kv_cache.paged_cache.block_cold_allocation_count > 0


def test_run_scheduler_suite_multi_workload_on_cpu():
    """验证 run_suite 在多 workload_id 情况下能正确归集每个 case 的独立测量结果。"""
    model = build_tiny_model()
    device = torch.device("cpu")
    req_w0 = tuple(make_requests([[1, 2], [3, 4]], [2, 2], [0.0, 0.0]))
    req_w1 = tuple(make_requests([[5, 6, 7], [8, 9, 10]], [1, 1], [0.0, 0.0]))

    dense_mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
    paged_mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
    paged_config = PagedCacheConfig(block_size=4)

    cases = [
        BenchmarkCase(
            profile=WorkloadProfile.from_requests("workload_0", req_w0),
            batching_method=BatchingMethod.CONTINUOUS,
            kv_cache_config=DenseCacheConfig(),
            max_running_requests=2,
            suite_name="multi-workload-suite",
            engine="scheduler",
        ),
        BenchmarkCase(
            profile=WorkloadProfile.from_requests("workload_0", req_w0),
            batching_method=BatchingMethod.CONTINUOUS,
            kv_cache_config=paged_config,
            max_running_requests=2,
            suite_name="multi-workload-suite",
            engine="scheduler",
        ),
        BenchmarkCase(
            profile=WorkloadProfile.from_requests("workload_1", req_w1),
            batching_method=BatchingMethod.CONTINUOUS,
            kv_cache_config=DenseCacheConfig(),
            max_running_requests=2,
            suite_name="multi-workload-suite",
            engine="scheduler",
        ),
        BenchmarkCase(
            profile=WorkloadProfile.from_requests("workload_1", req_w1),
            batching_method=BatchingMethod.CONTINUOUS,
            kv_cache_config=paged_config,
            max_running_requests=2,
            suite_name="multi-workload-suite",
            engine="scheduler",
        ),
    ]

    @dataclass
    class DummyMultiWorkloadSuite:
        cases: tuple[BenchmarkCase, ...]
        name: str = "multi-workload-suite"
        warmup: int = 0
        repeats: int = 2
        max_running_requests: int = 2
        interleave: tuple[CaseDimension, ...] = (
            CaseDimension.CACHING_METHOD,
            CaseDimension.BLOCK_SIZE,
        )
        comparison: Any = None

        def modes(self):
            return [dense_mode, paged_mode]

    suite = DummyMultiWorkloadSuite(cases=tuple(cases))
    results = run_suite(
        model=model,
        device=device,
        suite=suite,
        vocab_size=model.config.vocab_size,
        warmup=0,
        repeats=2,
    )

    assert len(results) == 4
    # 验证返回顺序与 cases 一致，且负载规格与各 case 匹配
    assert results[0].workload.prompt_lens == (2, 2)
    assert results[0].mode == dense_mode
    assert results[1].workload.prompt_lens == (2, 2)
    assert results[1].mode == paged_mode
    assert results[2].workload.prompt_lens == (3, 3)
    assert results[2].mode == dense_mode
    assert results[3].workload.prompt_lens == (3, 3)
    assert results[3].mode == paged_mode


def test_scheduler_suite_dry_run_main(capsys):
    """验证 bench sweep --dry-run 参数无需 GPU 与模型即可打印执行计划。"""
    from mini_infra.cli import main

    ret = main(
        [
            "bench",
            "sweep",
            "--engine",
            "scheduler",
            "--tag",
            "dev",
            "--dry-run",
            "--suite",
            "fixed-vs-continuous-variable-output-length",
            "--repeats",
            "2",
            "--interleave",
            "batching_method",
        ]
    )
    assert ret == 0

    captured = capsys.readouterr().out
    assert "=== [Dry Run] Suite: fixed-vs-continuous-variable-output-length ===" in captured
    assert "Scheduler 基准评测执行计划" in captured
    assert "warmup" in captured
    assert "repeat 1" in captured
