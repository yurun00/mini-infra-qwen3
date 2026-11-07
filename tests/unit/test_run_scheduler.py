"""bench.scheduler.run_case 与 run_suite 的单元测试（CPU 可运行，使用 tiny_model）。"""

import torch

from mini_infra.bench.scheduler import (
    build_benchmark_record,
    format_case_summary,
    format_table,
    run_case,
    run_case_once,
    run_suite,
)
from mini_infra.bench.scheduler.suites import FixedVsContinuousVariableOutputLengthSuite, SchedulerSuiteCase
from mini_infra.core.request import make_request_specs
from mini_infra.engine import BatchingMethod, CachingMethod, Mode
from mini_infra.verification.smoke import build_tiny_model


def test_parse_args_scheduler_suite():
    from mini_infra.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "scheduler",
            "suite",
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
    assert args.suite == "fixed-vs-continuous-variable-output-length"
    assert args.repeats == 2
    assert args.interleave == ["batching_method"]
    assert args.dry_run is True


def test_parse_args_scheduler_suite_validation():
    import pytest

    from mini_infra.cli import build_parser

    parser = build_parser()
    # 1. 验证 --suite 为必填项
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "scheduler", "suite", "--tag", "dev"])

    # 2. 验证已不再支持 'all'，一次只跑一个 suite
    with pytest.raises(SystemExit):
        parser.parse_args(["bench", "scheduler", "suite", "--tag", "dev", "--suite", "all"])


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
    assert args.batching_method == "continuous-batching"
    assert args.max_running_requests == 2


def test_run_scheduler_case_and_step_on_cpu():
    """验证原子 step 与单用例 run_case 仅执行单一策略。"""
    model = build_tiny_model()
    device = torch.device("cpu")
    specs = tuple(make_request_specs([[1, 2, 3], [4, 5]], [2, 3], [0.0, 10.0]))
    mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
    case = SchedulerSuiteCase(
        id="continuous",
        suite_name="test-cont-case",
        mode=mode,
        max_running_requests=2,
        specs=specs,
    )

    # 1. 验证原子 run_case_once 与 ScheduleResult.workload 属性
    single_res = run_case_once(model, device, case)
    assert single_res.step_agg_metrics.e2e_wall_ms > 0
    assert len(single_res.sequences) == 2
    assert single_res.workload.repeats == 1
    assert single_res.workload.warmup == 0

    # 2. 验证多轮 run_case（WorkloadSpec 相等比较忽略 repeats 与 warmup）
    case_res = run_case(
        model=model,
        device=device,
        case=case,
        warmup=1,
        repeats=2,
    )

    assert case_res.case_id == "continuous"
    assert case_res.workload.mode == mode
    assert case_res.workload.repeats == 2
    assert single_res.workload == case_res.workload
    assert case_res.metrics.e2e_wall_ms.mean > 0
    assert case_res.metrics.throughput_req_s.mean > 0
    assert case_res.schedule_metrics.ttft_ms_p95.mean > 0
    assert case_res.memory.kv_cache.peak_live_kv_cache_mb > 0
    assert case_res.packing is not None
    assert case_res.packing.slot_utilization.mean > 0
    assert case_res.packing.running_capacity_utilization is not None
    assert case_res.packing.running_capacity_utilization.mean > 0
    assert len(case_res.sample_events) > 0
    print(format_case_summary(case_res))


def test_run_scheduler_suite_on_cpu():
    """验证套件执行器 run_suite 支持 plan 驱动并导出统一 SchedulerBenchmarkRecord。"""
    model = build_tiny_model()
    device = torch.device("cpu")
    suite = FixedVsContinuousVariableOutputLengthSuite(
        request_count=2,
        prompt_length=4,
        short_output=2,
        long_output=3,
        warmup=0,
        repeats=2,
    )

    cases = run_suite(
        model=model,
        device=device,
        suite=suite,
        vocab_size=model.config.vocab_size,
        warmup=0,
        repeats=2,
        interleave=(BatchingMethod,),
    )

    assert len(cases) == 2
    assert {c.case_id for c in cases} == {"fixed", "continuous"}

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
        correctness={"check": True},
        env={"gpu": "CPU"},
        peaks={"bandwidth_gbs": None, "tflops": None},
        cost=None,
        git={"commit": "abc", "dirty": False},
        warmup=0,
        repeats=2,
        suite_info=suite.to_dict(),
        engine="scheduler",
    )
    assert record["engine"] == "scheduler"
    assert record["suite"]["request_count"] == 2
    assert len(record["cases"]) == 2


def test_run_scheduler_paged_case_on_cpu():
    """验证 run_case_once 和 run_case 支持 Paged 缓存模式并正确采集 paged_cache 画像。"""
    from mini_infra.core.block_manager import PagedCacheConfig

    model = build_tiny_model()
    device = torch.device("cpu")
    specs = tuple(make_request_specs([[1, 2, 3, 4], [5, 6]], [2, 3], [0.0, 5.0]))
    mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
    paged_config = PagedCacheConfig(block_size=4)

    case = SchedulerSuiteCase(
        id="paged",
        suite_name="test-paged-case",
        mode=mode,
        max_running_requests=2,
        specs=specs,
        workload_id="stress_case",
        paged_cache_config=paged_config,
    )

    # 1. 单次执行
    single_res = run_case_once(model, device, case)
    assert single_res.step_agg_metrics.e2e_wall_ms > 0
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
    assert case_res.case_id == "paged"
    assert case_res.workload.mode == mode
    assert case_res.memory.kv_cache.paged_cache is not None
    assert case_res.memory.kv_cache.paged_cache.block_size == 4
    assert case_res.memory.kv_cache.paged_cache.block_cold_allocation_count > 0


def test_run_scheduler_suite_multi_workload_on_cpu():
    """验证 run_suite 在多 workload_id 情况下能正确归集每个 case 的独立测量结果。"""
    from mini_infra.core.block_manager import PagedCacheConfig

    model = build_tiny_model()
    device = torch.device("cpu")
    spec_w0 = tuple(make_request_specs([[1, 2], [3, 4]], [2, 2], [0.0, 0.0]))
    spec_w1 = tuple(make_request_specs([[5, 6, 7], [8, 9, 10]], [1, 1], [0.0, 0.0]))

    dense_mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)
    paged_mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
    paged_config = PagedCacheConfig(block_size=4)

    cases = [
        SchedulerSuiteCase(
            id="dense",
            suite_name="multi-workload-suite",
            mode=dense_mode,
            max_running_requests=2,
            specs=spec_w0,
            workload_id="workload_0",
        ),
        SchedulerSuiteCase(
            id="paged",
            suite_name="multi-workload-suite",
            mode=paged_mode,
            max_running_requests=2,
            specs=spec_w0,
            workload_id="workload_0",
            paged_cache_config=paged_config,
        ),
        SchedulerSuiteCase(
            id="dense",
            suite_name="multi-workload-suite",
            mode=dense_mode,
            max_running_requests=2,
            specs=spec_w1,
            workload_id="workload_1",
        ),
        SchedulerSuiteCase(
            id="paged",
            suite_name="multi-workload-suite",
            mode=paged_mode,
            max_running_requests=2,
            specs=spec_w1,
            workload_id="workload_1",
            paged_cache_config=paged_config,
        ),
    ]

    class DummyMultiWorkloadSuite:
        name = "multi-workload-suite"
        warmup = 0
        repeats = 2
        max_running_requests = 2
        interleave = (CachingMethod,)

        def modes(self):
            return [dense_mode, paged_mode]

        def cases(self, vocab_size, device=None):
            return cases

    suite = DummyMultiWorkloadSuite()
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
    assert results[0].workload.prompt_len == 2
    assert results[0].workload.mode == dense_mode
    assert results[1].workload.prompt_len == 2
    assert results[1].workload.mode == paged_mode
    assert results[2].workload.prompt_len == 3
    assert results[2].workload.mode == dense_mode
    assert results[3].workload.prompt_len == 3
    assert results[3].workload.mode == paged_mode


def test_scheduler_suite_dry_run_main(capsys):
    """验证 --dry-run 参数无需 GPU 与模型即可打印执行计划。"""
    from mini_infra.cli import main

    ret = main(
        [
            "bench",
            "scheduler",
            "suite",
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
