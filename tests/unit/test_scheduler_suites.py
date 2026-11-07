"""bench.scheduler.suites 配置与规格构建的单元测试（CPU 可运行）。"""

import json

import pytest

from mini_infra.bench.scheduler.suites import (
    SCHEDULER_SUITES,
    CaseDimension,
    CaseKey,
    ComparisonMetric,
    ComparisonSpec,
    DenseVsPagedBlockSizeSuite,
    DenseVsPagedLengthRatioSuite,
    DenseVsPagedLongShortStressSuite,
    FixedVsContinuousPrefillInterferenceSuite,
    FixedVsContinuousStaggeredArrivalsSuite,
    FixedVsContinuousUniformOutputLengthSuite,
    FixedVsContinuousVariableOutputLengthSuite,
    MetricKind,
    SchedulerSuiteCase,
    get_all_scheduler_suites,
    get_scheduler_suite,
)
from mini_infra.bench.workloads import make_long_short_specs
from mini_infra.engine import BatchingMethod, CachingMethod, Mode


def test_scheduler_suites_registry():
    assert set(SCHEDULER_SUITES.keys()) == {
        "fixed-vs-continuous-variable-output-length",
        "fixed-vs-continuous-uniform-output-length",
        "fixed-vs-continuous-staggered-arrivals",
        "fixed-vs-continuous-prefill-interference",
        "dense-vs-paged-long-short-stress",
        "dense-vs-paged-length-ratio",
        "dense-vs-paged-block-size",
    }
    assert len(get_all_scheduler_suites()) == 7
    for name in SCHEDULER_SUITES:
        assert get_scheduler_suite(name).name == name

    with pytest.raises(ValueError, match="未知 scheduler suite"):
        get_scheduler_suite("non-existent")


def test_variable_output_length_suite():
    suite = FixedVsContinuousVariableOutputLengthSuite(request_count=4, prompt_length=16, short_output=2, long_output=6)
    specs = suite.build_specs(vocab_size=1000)
    assert len(specs) == 4
    assert [len(s.token_ids) for s in specs] == [16, 16, 16, 16]
    assert [s.max_new_tokens for s in specs] == [2, 6, 2, 6]
    assert [s.arrival_ms for s in specs] == [0.0, 0.0, 0.0, 0.0]
    json.dumps(suite.to_dict())

    with pytest.raises(ValueError):
        FixedVsContinuousVariableOutputLengthSuite(request_count=0)
    with pytest.raises(ValueError):
        FixedVsContinuousVariableOutputLengthSuite(short_output=0)
    with pytest.raises(ValueError):
        FixedVsContinuousVariableOutputLengthSuite(batching_methods=())


def test_uniform_output_length_suite():
    suite = FixedVsContinuousUniformOutputLengthSuite(request_count=3, prompt_length=10, output_budget=5)
    specs = suite.build_specs(vocab_size=1000)
    assert len(specs) == 3
    assert [len(s.token_ids) for s in specs] == [10, 10, 10]
    assert [s.max_new_tokens for s in specs] == [5, 5, 5]
    assert [s.arrival_ms for s in specs] == [0.0, 0.0, 0.0]
    json.dumps(suite.to_dict())

    with pytest.raises(ValueError):
        FixedVsContinuousUniformOutputLengthSuite(output_budget=0)
    with pytest.raises(ValueError):
        FixedVsContinuousUniformOutputLengthSuite(batching_methods=())


def test_staggered_arrivals_suite():
    suite = FixedVsContinuousStaggeredArrivalsSuite(
        request_count=3, prompt_length=8, short_output=2, long_output=4, arrival_interval_ms=50.0
    )
    specs = suite.build_specs(vocab_size=1000)
    assert len(specs) == 3
    assert [s.arrival_ms for s in specs] == [0.0, 50.0, 100.0]
    assert [s.max_new_tokens for s in specs] == [2, 4, 2]
    json.dumps(suite.to_dict())

    with pytest.raises(ValueError):
        FixedVsContinuousStaggeredArrivalsSuite(arrival_interval_ms=-1.0)
    with pytest.raises(ValueError):
        FixedVsContinuousStaggeredArrivalsSuite(batching_methods=())


def test_prefill_interference_suite():
    suite = FixedVsContinuousPrefillInterferenceSuite(
        prompt_length=10,
        interference_prompt_length=30,
        short_output=2,
        long_output=6,
        interference_arrival_ms=100.0,
    )
    specs = suite.build_specs(vocab_size=1000)
    assert len(specs) == 3
    assert [len(s.token_ids) for s in specs] == [10, 10, 30]
    assert [s.max_new_tokens for s in specs] == [6, 6, 2]
    assert [s.arrival_ms for s in specs] == [0.0, 0.0, 100.0]
    json.dumps(suite.to_dict())

    with pytest.raises(ValueError):
        FixedVsContinuousPrefillInterferenceSuite(max_running_requests=2)  # 至少 3
    with pytest.raises(ValueError):
        FixedVsContinuousPrefillInterferenceSuite(interference_arrival_ms=-10.0)
    with pytest.raises(ValueError):
        FixedVsContinuousPrefillInterferenceSuite(batching_methods=())
    with pytest.raises(ValueError):
        FixedVsContinuousPrefillInterferenceSuite(cache_methods=())


def test_suite_cases_expansion():
    fixed_suites = [s for s in get_all_scheduler_suites() if s.name.startswith("fixed-vs-continuous")]
    assert len(fixed_suites) == 4
    for suite in fixed_suites:
        assert suite.interleave == (CaseDimension.BATCHING_METHOD,)
        assert suite.comparison is not None
        modes = suite.modes()
        assert len(modes) == 2
        assert all(isinstance(m, Mode) for m in modes)
        assert all(m.use_kv_cache for m in modes)

        cases = suite.cases(vocab_size=500)
        assert len(cases) == 2
        assert all(isinstance(c, SchedulerSuiteCase) for c in cases)
        assert all(isinstance(c.mode, Mode) for c in cases)

        methods = {c.mode.batching_method for c in cases}
        assert methods == {BatchingMethod.FIXED, BatchingMethod.CONTINUOUS}
        assert cases[0].mode.batching_method == BatchingMethod.FIXED
        assert cases[1].mode.batching_method == BatchingMethod.CONTINUOUS
        assert cases[0].workload_id == cases[1].workload_id
        assert len(cases[0].workload_id) > 0
        assert cases[0].key == CaseKey(
            workload_id=cases[0].workload_id,
            mode=cases[0].mode,
            max_running_requests=cases[0].max_running_requests,
        )
        assert cases[1].key == CaseKey(
            workload_id=cases[1].workload_id,
            mode=cases[1].mode,
            max_running_requests=cases[1].max_running_requests,
        )
        assert len(cases[0].specs) == len(cases[1].specs)
        assert cases[0].suite_name == suite.name


def test_make_long_short_specs():
    specs = make_long_short_specs(vocab_size=1000)
    assert len(specs) == 17  # 7 + 1 + 9
    assert [s.request_idx for s in specs] == list(range(17))

    # 初始 7 个短请求
    for s in specs[:7]:
        assert len(s.token_ids) == 128
        assert s.max_new_tokens == 96
        assert s.arrival_ms == 0.0

    # 中途 1 个长请求
    long_s = specs[7]
    assert len(long_s.token_ids) == 2048
    assert long_s.max_new_tokens == 8
    assert long_s.arrival_ms == 200.0

    # 后续 9 个短请求
    for s in specs[8:]:
        assert len(s.token_ids) == 128
        assert s.max_new_tokens == 64
        assert s.arrival_ms == 201.0

    # 异常校验
    with pytest.raises(ValueError, match="vocab_size 必须大于 0"):
        make_long_short_specs(vocab_size=0)
    with pytest.raises(ValueError, match="prompt_length 必须大于 0"):
        make_long_short_specs(vocab_size=1000, short_prompt_length=0)
    with pytest.raises(ValueError, match="prompt_length 必须大于 0"):
        make_long_short_specs(vocab_size=1000, long_prompt_length=0)
    with pytest.raises(ValueError, match="请求数量必须大于 0"):
        make_long_short_specs(vocab_size=1000, initial_short_requests=0)
    with pytest.raises(ValueError, match="请求数量必须大于 0"):
        make_long_short_specs(vocab_size=1000, followup_short_requests=0)
    with pytest.raises(ValueError, match="max_running_requests 必须大于 0"):
        make_long_short_specs(vocab_size=1000, max_running_requests=0)
    with pytest.raises(ValueError, match="必须小于 max_running_requests"):
        make_long_short_specs(vocab_size=1000, initial_short_requests=8, max_running_requests=8)
    with pytest.raises(ValueError, match="输出预算必须大于 0"):
        make_long_short_specs(vocab_size=1000, short_output=0)
    with pytest.raises(ValueError, match="输出预算必须大于 0"):
        make_long_short_specs(vocab_size=1000, long_output=0)
    with pytest.raises(ValueError, match="输出预算必须大于 0"):
        make_long_short_specs(vocab_size=1000, followup_output=0)
    with pytest.raises(ValueError, match="long_arrival_ms 不能小于 0"):
        make_long_short_specs(vocab_size=1000, long_arrival_ms=-1.0)
    with pytest.raises(ValueError, match="不能小于 long_arrival_ms"):
        make_long_short_specs(vocab_size=1000, long_arrival_ms=300.0, followup_arrival_ms=200.0)


def test_dense_vs_paged_long_short_stress_suite():
    suite = DenseVsPagedLongShortStressSuite()
    assert suite.interleave == (CaseDimension.CACHING_METHOD, CaseDimension.PAGED_CACHE_CONFIG)
    assert suite.comparison is not None
    modes = suite.modes()
    assert len(modes) == 2
    assert [m.caching_method for m in modes] == [CachingMethod.DENSE, CachingMethod.PAGED]
    assert all(m.batching_method == BatchingMethod.CONTINUOUS for m in modes)

    specs = suite.build_specs(vocab_size=1000)
    assert len(specs) == 17

    cases = suite.cases(vocab_size=1000)
    assert len(cases) == 2
    assert cases[0].mode.caching_method == CachingMethod.DENSE
    assert cases[0].paged_cache_config is None
    assert cases[1].mode.caching_method == CachingMethod.PAGED
    assert cases[1].paged_cache_config is not None
    assert cases[1].paged_cache_config.block_size == 16
    assert cases[0].key.mode.caching_method == CachingMethod.DENSE
    assert cases[1].key.mode.caching_method == CachingMethod.PAGED

    data = suite.to_dict()
    json.dumps(data)
    assert data["name"] == "dense-vs-paged-long-short-stress"
    assert data["block_size"] == 16
    assert data["interleave"] == ["caching_method", "paged_cache_config"]
    assert data["comparison"] is not None

    with pytest.raises(ValueError):
        DenseVsPagedLongShortStressSuite(max_running_requests=0)
    with pytest.raises(ValueError):
        DenseVsPagedLongShortStressSuite(initial_short_requests=8, max_running_requests=8)
    with pytest.raises(ValueError):
        DenseVsPagedLongShortStressSuite(block_size=0)
    with pytest.raises(ValueError):
        DenseVsPagedLongShortStressSuite(cache_methods=())


def test_dense_vs_paged_length_ratio_suite():
    suite = DenseVsPagedLengthRatioSuite()
    assert suite.interleave == (CaseDimension.CACHING_METHOD, CaseDimension.PAGED_CACHE_CONFIG)
    assert suite.comparison is not None
    assert suite.long_prompt_lengths == (128, 512, 2048, 3072)

    specs = suite.build_specs(vocab_size=1000)
    assert len(specs) == 17

    cases = suite.cases(vocab_size=1000)
    # 4 个 workload，每个 2 cases -> 共 8 个 cases
    assert len(cases) == 8
    expected_workload_ids = [
        "length_ratio_128",
        "length_ratio_128",
        "length_ratio_512",
        "length_ratio_512",
        "length_ratio_2048",
        "length_ratio_2048",
        "length_ratio_3072",
        "length_ratio_3072",
    ]
    assert [c.workload_id for c in cases] == expected_workload_ids
    assert [c.mode.caching_method for c in cases] == [CachingMethod.DENSE, CachingMethod.PAGED] * 4

    # 验证各 workload 下长请求长度
    for i, long_len in enumerate([128, 512, 2048, 3072]):
        dense_case = cases[2 * i]
        paged_case = cases[2 * i + 1]
        assert len(dense_case.specs[7].token_ids) == long_len
        assert len(paged_case.specs[7].token_ids) == long_len
        assert paged_case.paged_cache_config.block_size == 16

    data = suite.to_dict()
    json.dumps(data)
    assert data["long_prompt_lengths"] == [128, 512, 2048, 3072]

    with pytest.raises(ValueError):
        DenseVsPagedLengthRatioSuite(long_prompt_lengths=())
    with pytest.raises(ValueError):
        DenseVsPagedLengthRatioSuite(long_prompt_lengths=(128, 0))


def test_dense_vs_paged_block_size_suite():
    suite = DenseVsPagedBlockSizeSuite()
    assert suite.interleave == (CaseDimension.CACHING_METHOD, CaseDimension.PAGED_CACHE_CONFIG)
    assert suite.comparison is not None
    assert suite.block_sizes == (8, 16, 32, 64)

    specs = suite.build_specs(vocab_size=1000)
    assert len(specs) == 17

    cases = suite.cases(vocab_size=1000)
    # 1 个 dense + 4 个 paged -> 共 5 个 cases
    assert len(cases) == 5
    assert all(c.workload_id == "block_size_sweep" for c in cases)
    assert cases[0].mode.caching_method == CachingMethod.DENSE
    assert cases[0].paged_cache_config is None

    for i, bs in enumerate([8, 16, 32, 64]):
        paged_case = cases[1 + i]
        assert paged_case.mode.caching_method == CachingMethod.PAGED
        assert paged_case.paged_cache_config.block_size == bs

    data = suite.to_dict()
    json.dumps(data)
    assert data["block_sizes"] == [8, 16, 32, 64]

    with pytest.raises(ValueError):
        DenseVsPagedBlockSizeSuite(block_sizes=())
    with pytest.raises(ValueError):
        DenseVsPagedBlockSizeSuite(block_sizes=(8, 0))


def test_scheduler_suite_case_paged_validation():
    from mini_infra.core.block_manager import PagedCacheConfig
    from mini_infra.core.request import RequestSpec
    from mini_infra.engine import CachingMethod

    spec = RequestSpec(request_idx=0, token_ids=[1, 2], max_new_tokens=2, arrival_ms=0.0)
    paged_mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
    dense_mode = Mode(CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS)

    # 1. Paged 模式不传 paged_cache_config 必须报错
    with pytest.raises(ValueError, match="Paged 模式 .* 必须提供 paged_cache_config"):
        SchedulerSuiteCase(
            workload_id="test",
            suite_name="test",
            mode=paged_mode,
            max_running_requests=2,
            specs=(spec,),
            paged_cache_config=None,
        )

    # 2. 非 Paged 模式传 paged_cache_config 必须报错
    with pytest.raises(ValueError, match="非 Paged 模式 .* 不得提供 paged_cache_config"):
        SchedulerSuiteCase(
            workload_id="test",
            suite_name="test",
            mode=dense_mode,
            max_running_requests=2,
            specs=(spec,),
            paged_cache_config=PagedCacheConfig(block_size=16),
        )

    # 3. workload_id 为空报错
    with pytest.raises(ValueError, match="workload_id 不能为空"):
        SchedulerSuiteCase(
            workload_id="",
            suite_name="test",
            mode=dense_mode,
            max_running_requests=2,
            specs=(spec,),
        )

    # 4. 正确传参合法构造
    case_paged = SchedulerSuiteCase(
        workload_id="workload_0",
        suite_name="test",
        mode=paged_mode,
        max_running_requests=2,
        specs=(spec,),
        paged_cache_config=PagedCacheConfig(block_size=16),
    )
    assert case_paged.workload_id == "workload_0"
    assert case_paged.paged_cache_config is not None
    assert case_paged.paged_cache_config.block_size == 16
    assert case_paged.key == CaseKey(
        workload_id="workload_0",
        mode=paged_mode,
        max_running_requests=2,
        paged_cache_config=PagedCacheConfig(block_size=16),
    )

    spec_dict = case_paged.to_spec_dict()
    assert spec_dict["workload_id"] == "workload_0"
    assert spec_dict["paged_cache_config"]["block_size"] == 16


def test_compute_comparisons():
    from mini_infra.bench.common import DistributionStats
    from mini_infra.bench.scheduler.format import (
        format_case_summary,
        format_comparison,
        format_table,
    )
    from mini_infra.bench.scheduler.types import (
        MetricComparison,
        SchedulerCaseResult,
        WorkloadSpec,
        compute_comparisons,
    )
    from mini_infra.core.block_manager import PagedCacheConfig
    from mini_infra.engine.metrics import (
        KVCacheProfile,
        MemoryProfile,
        PagedCacheProfile,
        ScheduleMetrics,
        StepAggregatedMetrics,
    )

    def dist(val: float) -> DistributionStats:
        return DistributionStats(median=val, mean=val, std=0.0, min=val, max=val, p90=val, p95=val, p99=val)

    def make_case(
        cm: CachingMethod,
        makespan: float,
        tok_s: float,
        pool_mb: float,
        accessed_toks: int,
        wid: str = "workload_A",
        bs: int | None = None,
        mrr: int = 8,
    ) -> SchedulerCaseResult:
        mode = Mode(cm, batching_method=BatchingMethod.CONTINUOUS)
        workload = WorkloadSpec(mode=mode, batch_size=8, prompt_len=128, max_new_tokens=64)
        paged_prof = (
            PagedCacheProfile(
                block_size=bs,
                bytes_per_block=4096,
                num_blocks=100,
                peak_used_blocks=50,
                block_cold_allocation_count=50,
                block_reuse_count=0,
                block_release_count=50,
            )
            if bs is not None
            else None
        )
        mem = MemoryProfile(
            kv_cache=KVCacheProfile(
                peak_live_kv_cache_mb=pool_mb * 0.8,
                peak_kv_cache_pool_mb=pool_mb,
                accessed_kv_cache_tokens=accessed_toks,
                paged_cache=paged_prof,
            )
        )
        step_metrics: StepAggregatedMetrics[DistributionStats] = StepAggregatedMetrics(
            e2e_wall_ms=dist(makespan),
            e2e_model_ms=dist(makespan * 0.9),
            throughput_tok_s=dist(tok_s),
            throughput_req_s=dist(10.0),
        )
        sched_metrics: ScheduleMetrics[DistributionStats] = ScheduleMetrics(
            request_count=8,
            max_running_requests=mrr,
            makespan_ms=dist(makespan),
            busy_ms=dist(makespan),
            cache_management_ms=dist(10.0),
            cache_management_fraction=dist(0.01),
            queue_ms_p50=dist(0.0),
            queue_ms_p95=dist(0.0),
            ttft_ms_p50=dist(20.0),
            ttft_ms_p95=dist(25.0),
            end_to_end_ms_p50=dist(makespan),
            end_to_end_ms_p95=dist(makespan),
            itl_ms_p50=dist(10.0),
            itl_ms_p95=dist(12.0),
            itl_ms_max=dist(15.0),
        )
        paged_cfg = PagedCacheConfig(block_size=bs) if bs is not None else None
        key = CaseKey(
            workload_id=wid,
            mode=mode,
            max_running_requests=mrr,
            paged_cache_config=paged_cfg,
        )
        return SchedulerCaseResult(
            key=key,
            workload=workload,
            metrics=step_metrics,
            schedule_metrics=sched_metrics,
            memory=mem,
        )

    dense_case = make_case(
        CachingMethod.DENSE,
        makespan=2000.0,
        tok_s=100.0,
        pool_mb=2000.0,
        accessed_toks=10000,
        wid="workload_A",
    )
    paged_16 = make_case(
        CachingMethod.PAGED,
        makespan=1600.0,
        tok_s=125.0,
        pool_mb=1000.0,
        accessed_toks=6000,
        wid="workload_A",
        bs=16,
    )
    paged_32 = make_case(
        CachingMethod.PAGED,
        makespan=1500.0,
        tok_s=133.33,
        pool_mb=1200.0,
        accessed_toks=7000,
        wid="workload_A",
        bs=32,
    )

    spec = ComparisonSpec(
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.CACHING_METHOD: CachingMethod.DENSE},
        metrics=(
            ComparisonMetric("schedule_metrics.makespan_ms", MetricKind.SPEEDUP, "makespan"),
            ComparisonMetric("metrics.throughput_tok_s", MetricKind.GAIN, "throughput"),
            ComparisonMetric("memory.kv_cache.accessed_kv_cache_tokens", MetricKind.REDUCTION, "KV-read"),
            ComparisonMetric("memory.kv_cache.peak_kv_cache_pool_mb", MetricKind.REDUCTION, "pool"),
        ),
    )

    cases = [dense_case, paged_16, paged_32]
    comparisons = compute_comparisons(cases, spec)
    # 2 candidate cases * 4 metrics = 8 comparisons
    assert len(comparisons) == 8

    # 验证 paged_16 的 makespan speedup
    c0 = comparisons[0]
    assert isinstance(c0, MetricComparison)
    assert c0.candidate_key == paged_16.key
    assert c0.baseline_key == dense_case.key
    assert c0.metric_name == "makespan"
    assert c0.kind == MetricKind.SPEEDUP
    assert c0.baseline_value == 2000.0
    assert c0.candidate_value == 1600.0
    assert c0.speedup == 1.25
    assert c0.ratio == 0.8
    assert c0.relative_delta == pytest.approx(-0.2)
    assert c0.formatted_value() == "speedup 1.25x"

    # 验证 paged_16 的 throughput gain
    c1 = comparisons[1]
    assert c1.metric_name == "throughput"
    assert c1.kind == MetricKind.GAIN
    assert c1.relative_delta == pytest.approx(0.25)
    assert c1.formatted_value() == "throughput +25.0%"

    # 验证 paged_16 的 KV-read reduction
    c2 = comparisons[2]
    assert c2.metric_name == "KV-read"
    assert c2.kind == MetricKind.REDUCTION
    assert c2.relative_delta == pytest.approx(-0.4)
    assert c2.formatted_value() == "KV-read -40.0%"

    # 验证 paged_16 的 pool reduction
    c3 = comparisons[3]
    assert c3.metric_name == "pool"
    assert c3.kind == MetricKind.REDUCTION
    assert c3.relative_delta == pytest.approx(-0.5)
    assert c3.formatted_value() == "pool -50.0%"

    # 验证 format_comparison
    comp_line = format_comparison(c0)
    assert "speedup 1.25x" in comp_line

    # 验证 baseline 找不到时报错
    bad_spec_no_baseline = ComparisonSpec(
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.CACHING_METHOD: CachingMethod.NONE},
        metrics=spec.metrics,
    )
    with pytest.raises(ValueError, match="未找到符合 baseline 约束"):
        compute_comparisons(cases, bad_spec_no_baseline)

    # 验证 baseline 匹配多个时报错
    dup_dense = make_case(
        CachingMethod.DENSE,
        makespan=1900.0,
        tok_s=105.0,
        pool_mb=2000.0,
        accessed_toks=10000,
        wid="workload_A",
        mrr=16,
    )
    with pytest.raises(ValueError, match="找到多个符合 baseline 约束"):
        compute_comparisons([dense_case, dup_dense, paged_16], spec)

    # 验证 format_case_summary 与 format_table
    summary_str = format_case_summary(paged_16)
    assert "[workload_A]" in summary_str
    assert "pool 1000.0 MB" in summary_str

    full_csv = format_table(cases, suite_name="test-suite")
    assert "workload_A" in full_csv
    assert "1000.00" in full_csv
