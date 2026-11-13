"""bench.scheduler.suites（YAML 声明 → BenchmarkSuite → BenchmarkCase）与用例层类型的单元测试（CPU 可运行）。"""

import json

import pytest

from mini_infra.bench.cases import (
    BenchmarkCase,
    CaseDimension,
    CaseKey,
    WorkloadProfile,
)
from mini_infra.bench.comparison import (
    Comparison,
    ComparisonMetric,
    ComparisonResult,
    MetricKind,
    compute_comparisons,
)
from mini_infra.bench.spec import get_suite_spec
from mini_infra.bench.suite import (
    BenchmarkSuite,
    get_all_benchmark_suites,
    get_benchmark_suite,
    list_benchmark_suite_names,
    load_benchmark_suite,
)
from mini_infra.core import DenseCacheConfig, KVCacheConfig, NoCacheConfig, PagedCacheConfig
from mini_infra.engine import BatchingMethod, EngineConfig

FIXED_VS_CONTINUOUS = [
    "fixed-vs-continuous-prefill-interference",
    "fixed-vs-continuous-staggered-arrivals",
    "fixed-vs-continuous-uniform-output-length",
    "fixed-vs-continuous-variable-output-length",
]


def test_scheduler_suites_registry():
    assert list_benchmark_suite_names("scheduler") == sorted(
        [
            "fixed-vs-continuous-variable-output-length",
            "fixed-vs-continuous-uniform-output-length",
            "fixed-vs-continuous-staggered-arrivals",
            "fixed-vs-continuous-prefill-interference",
            "dense-vs-paged-long-short-stress",
            "dense-vs-paged-length-ratio",
            "dense-vs-paged-block-size",
        ]
    )
    suites = get_all_benchmark_suites("scheduler")
    assert len(suites) == 7
    for suite in suites:
        assert isinstance(suite, BenchmarkSuite)
        assert isinstance(suite.cases, tuple)
        assert len(suite) == len(suite.cases)
        assert list(suite) == list(suite.cases)
        assert get_benchmark_suite("scheduler", suite.name).name == suite.name

    with pytest.raises(ValueError, match="未知 scheduler suite"):
        get_benchmark_suite("scheduler", "non-existent")


def test_load_scheduler_suite_rejects_other_engine(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text(
        "name: demo\nsystem: {engine_mode: lockstep, batching_method: fixed-batching, caching_spec: dense-kv-cache}\n"
        "workload: {id: w, count: 1, prompt_len: 8, output_len: 4}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="engine 必须是 scheduler"):
        load_benchmark_suite(path, expected_engine="scheduler")


@pytest.mark.parametrize("name", sorted(FIXED_VS_CONTINUOUS))
def test_fixed_vs_continuous_suites_expand_to_two_cases(name):
    suite = get_benchmark_suite("scheduler", name)
    assert suite.interleave == (CaseDimension.BATCHING_METHOD,)
    assert suite.comparison is not None
    configs = suite.engine_configs()
    assert [m.batching_method for m in configs] == [BatchingMethod.FIXED, BatchingMethod.CONTINUOUS]
    assert all(m.has_kv_cache for m in configs)

    cases = suite.cases
    assert len(cases) == 2
    assert all(isinstance(c, BenchmarkCase) and c.suite_name == name for c in cases)
    assert cases[0].workload_id == cases[1].workload_id
    assert cases[0].profile == cases[1].profile
    for case in cases:
        assert case.key == CaseKey(
            case.workload_id,
            caching_method=case.kv_cache_config.kind,
            batching_method=case.batching_method,
            max_running_requests=case.max_running_requests,
        )


def test_variable_output_length_suite_specs():
    suite = get_benchmark_suite("scheduler", "fixed-vs-continuous-variable-output-length")
    specs = suite.cases[0].profile.to_payload(vocab_size=1000)
    assert [len(s.token_ids) for s in specs] == [128] * 8
    assert [s.max_new_tokens for s in specs] == [8, 32] * 4
    assert {s.arrival_ms for s in specs} == {0.0}
    json.dumps(suite.to_dict())


def test_staggered_arrivals_suite_specs():
    suite = get_benchmark_suite("scheduler", "fixed-vs-continuous-staggered-arrivals")
    specs = suite.cases[0].profile.to_payload(vocab_size=1000)
    assert [s.arrival_ms for s in specs] == [i * 150.0 for i in range(8)]


def test_prefill_interference_suite_specs_and_min_running_check(tmp_path):
    suite = get_benchmark_suite("scheduler", "fixed-vs-continuous-prefill-interference")
    specs = suite.cases[0].profile.to_payload(vocab_size=1000)
    assert [len(s.token_ids) for s in specs] == [128, 128, 512]
    assert [s.max_new_tokens for s in specs] == [32, 32, 8]
    assert [s.arrival_ms for s in specs] == [0.0, 0.0, 250.0]

    spec = get_suite_spec("scheduler", "fixed-vs-continuous-prefill-interference")
    text = spec.path.read_text(encoding="utf-8")
    assert "min_max_running_requests: 3" in text
    base = spec.path.parent / "_bases" / "fixed_vs_continuous.yaml"
    path = tmp_path / "demo.yaml"
    path.write_text(
        f"name: demo\nextends: {base}\nsystem: {{engine_mode: {{kind: scheduler, max_running_requests: 2}}}}\n"
        "workload: {id: w, groups: [{count: 1, prompt_len: 8, output_len: 4}]}\n"
        "checks: [{min_max_running_requests: 3}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_running_requests >= 3"):
        load_benchmark_suite(path, expected_engine="scheduler")


def test_dense_vs_paged_long_short_stress_suite():
    suite = get_benchmark_suite("scheduler", "dense-vs-paged-long-short-stress")
    assert suite.interleave == (CaseDimension.CACHING_METHOD, CaseDimension.BLOCK_SIZE)
    assert suite.comparison is not None
    configs = suite.engine_configs()
    assert [m.kv_cache_config.kind for m in configs] == ["dense-kv-cache", "paged-kv-cache"]
    assert all(m.batching_method == BatchingMethod.CONTINUOUS for m in configs)

    cases = suite.cases
    assert len(cases) == 2
    assert cases[0].profile.num_total_requests == 17  # 7 + 1 + 9
    assert isinstance(cases[0].kv_cache_config, DenseCacheConfig)
    assert isinstance(cases[1].kv_cache_config, PagedCacheConfig)
    assert cases[1].kv_cache_config.block_size == 16

    p = cases[0].profile
    short_lens = p.prompt_lens[:7]
    short_budgets = p.output_budgets[:7]
    short_arrivals = p.arrival_ms[:7]
    assert all(
        plen == 128 and budget == 96 and arr == 0.0
        for plen, budget, arr in zip(short_lens, short_budgets, short_arrivals)
    )
    assert (p.prompt_lens[7], p.output_budgets[7], p.arrival_ms[7]) == (2048, 8, 200.0)
    followups = list(zip(p.prompt_lens[8:], p.output_budgets[8:], p.arrival_ms[8:]))
    assert all(plen == 128 and budget == 64 and arr == 201.0 for plen, budget, arr in followups)

    data = suite.to_dict()
    json.dumps(data)
    assert data["name"] == "dense-vs-paged-long-short-stress"
    assert data["engine"] == "scheduler"
    assert data["spec"]["execution"]["interleave"] == ["caching_method", "block_size"]


def test_dense_vs_paged_length_ratio_suite():
    suite = get_benchmark_suite("scheduler", "dense-vs-paged-length-ratio")
    cases = suite.cases
    assert [c.workload_id for c in cases] == [f"length_ratio_{n}" for n in (128, 512, 2048, 3072) for _ in range(2)]
    assert [c.kv_cache_config.kind for c in cases] == ["dense-kv-cache", "paged-kv-cache"] * 4
    for i, long_len in enumerate([128, 512, 2048, 3072]):
        dense_case, paged_case = cases[2 * i], cases[2 * i + 1]
        assert dense_case.profile.prompt_lens[7] == long_len
        assert paged_case.profile.prompt_lens[7] == long_len
        assert isinstance(paged_case.kv_cache_config, PagedCacheConfig) and paged_case.kv_cache_config.block_size == 16


def test_dense_vs_paged_block_size_suite():
    suite = get_benchmark_suite("scheduler", "dense-vs-paged-block-size")
    cases = suite.cases
    assert len(cases) == 5
    assert len(suite.engine_configs()) == 5
    assert all(c.workload_id == "block_size_sweep" for c in cases)
    assert cases[0].kv_cache_config.kind == "dense-kv-cache"
    assert isinstance(cases[0].kv_cache_config, DenseCacheConfig)
    for case, bs in zip(cases[1:], [8, 16, 32, 64]):
        assert case.kv_cache_config.kind == "paged-kv-cache"
        assert isinstance(case.kv_cache_config, PagedCacheConfig) and case.kv_cache_config.block_size == bs


def test_comparison_spec_from_yaml_is_typed():
    comparison = get_benchmark_suite("scheduler", "dense-vs-paged-block-size").comparison
    assert isinstance(comparison, Comparison)
    assert comparison.group_by == (CaseDimension.WORKLOAD_ID,)
    assert comparison.baseline == {CaseDimension.CACHING_METHOD: "dense-kv-cache"}
    assert [m._display_name for m in comparison.metrics] == ["makespan", "throughput", "KV-read", "pool"]
    assert all(isinstance(m, ComparisonMetric) and isinstance(m.kind, MetricKind) for m in comparison.metrics)


def test_scheduler_suite_case_paged_validation():
    from mini_infra.core.request import Request

    req = Request(request_idx=0, token_ids=[1, 2], max_new_tokens=2, arrival_ms=0.0)
    paged_config = EngineConfig(
        PagedCacheConfig(block_size=16), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2
    )

    # 1. workload_id 为空报错
    with pytest.raises(ValueError, match="workload_id 不能为空"):
        WorkloadProfile.from_requests("", (req,))

    # 2. PagedCacheConfig 通过 kv_cache_config 直接构建并派生 engine_config
    case_paged = BenchmarkCase(
        profile=WorkloadProfile.from_requests("workload_0", (req,)),
        batching_method=BatchingMethod.CONTINUOUS,
        kv_cache_config=PagedCacheConfig(block_size=16),
        max_running_requests=2,
        suite_name="test",
        engine="scheduler",
    )
    assert case_paged.workload_id == "workload_0"
    assert case_paged.engine_config == paged_config
    assert isinstance(case_paged.kv_cache_config, PagedCacheConfig)
    assert case_paged.kv_cache_config.block_size == 16
    assert case_paged.key == CaseKey(
        workload_id="workload_0",
        caching_method="paged-kv-cache",
        batching_method=BatchingMethod.CONTINUOUS,
        max_running_requests=2,
        block_size=16,
    )


def test_compute_comparisons():
    from mini_infra.bench.cases import WorkloadProfile
    from mini_infra.bench.scheduler.format import (
        format_table,
    )
    from mini_infra.bench.scheduler.types import (
        SchedulerCaseResult,
    )
    from mini_infra.bench.stats import DistributionStats
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
        kv_cfg: KVCacheConfig,
        makespan: float,
        tok_s: float,
        pool_mb: float,
        accessed_toks: int,
        wid: str = "workload_A",
        mrr: int = 8,
    ) -> SchedulerCaseResult:
        workload = WorkloadProfile(
            workload_id=wid,
            num_total_requests=8,
            prompt_lens=(128,) * 8,
            output_budgets=(64,) * 8,
            arrival_ms=(0.0,) * 8,
        )
        paged_prof = (
            PagedCacheProfile(
                block_size=kv_cfg.block_size,
                bytes_per_block=4096,
                num_blocks=100,
                peak_used_blocks=50,
                block_cold_allocation_count=50,
                block_reuse_count=0,
                block_release_count=50,
            )
            if isinstance(kv_cfg, PagedCacheConfig)
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
        key = CaseKey(
            workload_id=wid,
            caching_method=kv_cfg.kind,
            batching_method=BatchingMethod.CONTINUOUS,
            max_running_requests=mrr,
            block_size=kv_cfg.block_size if isinstance(kv_cfg, PagedCacheConfig) else None,
            num_blocks=kv_cfg.num_blocks if isinstance(kv_cfg, PagedCacheConfig) else None,
        )
        return SchedulerCaseResult(
            key=key,
            workload=workload,
            metrics=step_metrics,
            schedule_metrics=sched_metrics,
            memory=mem,
        )

    dense_case = make_case(
        DenseCacheConfig(),
        makespan=2000.0,
        tok_s=100.0,
        pool_mb=2000.0,
        accessed_toks=10000,
        wid="workload_A",
    )
    paged_16 = make_case(
        PagedCacheConfig(block_size=16),
        makespan=1600.0,
        tok_s=125.0,
        pool_mb=1000.0,
        accessed_toks=6000,
        wid="workload_A",
    )
    paged_32 = make_case(
        PagedCacheConfig(block_size=32),
        makespan=1500.0,
        tok_s=133.33,
        pool_mb=1200.0,
        accessed_toks=7000,
        wid="workload_A",
    )

    comparison = Comparison(
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.CACHING_METHOD: "dense-kv-cache"},
        metrics=(
            ComparisonMetric("schedule_metrics.makespan_ms", MetricKind.SPEEDUP, "makespan"),
            ComparisonMetric("metrics.throughput_tok_s", MetricKind.GAIN, "throughput"),
            ComparisonMetric("memory.kv_cache.accessed_kv_cache_tokens", MetricKind.REDUCTION, "KV-read"),
            ComparisonMetric("memory.kv_cache.peak_kv_cache_pool_mb", MetricKind.REDUCTION, "pool"),
        ),
    )

    cases = [dense_case, paged_16, paged_32]
    comparisons = compute_comparisons(cases, comparison)
    # 2 candidate cases * 4 metrics = 8 comparisons
    assert len(comparisons) == 8

    # 验证 paged_16 的 makespan speedup
    c0 = comparisons[0]
    assert isinstance(c0, ComparisonResult)
    assert c0.candidate_key == paged_16.key
    assert c0.baseline_key == dense_case.key
    assert c0.metric_name == "makespan"
    assert c0.kind == MetricKind.SPEEDUP
    assert c0.baseline_value == 2000.0
    assert c0.candidate_value == 1600.0
    assert c0._speedup == 1.25
    assert c0.ratio == 0.8
    assert c0.relative_delta == pytest.approx(-0.2)
    assert c0._formatted_value() == "speedup 1.25x"

    # 验证 paged_16 的 throughput gain
    c1 = comparisons[1]
    assert c1.metric_name == "throughput"
    assert c1.kind == MetricKind.GAIN
    assert c1.relative_delta == pytest.approx(0.25)
    assert c1._formatted_value() == "throughput +25.0%"

    # 验证 paged_16 的 KV-read reduction
    c2 = comparisons[2]
    assert c2.metric_name == "KV-read"
    assert c2.kind == MetricKind.REDUCTION
    assert c2.relative_delta == pytest.approx(-0.4)
    assert c2._formatted_value() == "KV-read -40.0%"

    # 验证 paged_16 的 pool reduction
    c3 = comparisons[3]
    assert c3.metric_name == "pool"
    assert c3.kind == MetricKind.REDUCTION
    assert c3.relative_delta == pytest.approx(-0.5)
    assert c3._formatted_value() == "pool -50.0%"

    # 验证 ComparisonResult.format_summary() 实例方法
    comp_line = c0.format_summary()
    assert "[wor_A]" in comp_line
    assert "pkc+cb" in comp_line
    assert "speedup 1.25x" in comp_line

    # 验证 baseline 找不到时报错
    bad_spec_no_baseline = Comparison(
        group_by=(CaseDimension.WORKLOAD_ID,),
        baseline={CaseDimension.CACHING_METHOD: "none"},
        metrics=comparison.metrics,
    )
    with pytest.raises(ValueError, match="未找到符合 baseline 约束"):
        compute_comparisons(cases, bad_spec_no_baseline)

    # 验证 baseline 匹配多个时报错
    dup_dense = make_case(
        DenseCacheConfig(),
        makespan=1900.0,
        tok_s=105.0,
        pool_mb=2000.0,
        accessed_toks=10000,
        wid="workload_A",
        mrr=16,
    )
    with pytest.raises(ValueError, match="找到多个符合 baseline 约束"):
        compute_comparisons([dense_case, dup_dense, paged_16], comparison)

    # 验证 SchedulerCaseResult.format_summary() 与 format_table
    summary_str = paged_16.format_summary()
    assert "[wor_A]" in summary_str
    assert "pkc+cb" in summary_str
    assert "pool 1000.0 MB" in summary_str

    full_csv = format_table(cases, suite_name="test-suite")
    assert "tes-sui" in full_csv
    assert "wor_A" in full_csv
    assert "dkc+cb" in full_csv
    assert "pkc+cb" in full_csv
    assert "1000.00" in full_csv


def test_format_table_abbreviations():
    from mini_infra.bench.cases import CaseKey
    from mini_infra.bench.scheduler.format import _abbreviate_suite_name

    def wid_abbr(wid: str) -> str:
        return CaseKey(
            wid,
            caching_method="dense-kv-cache",
            batching_method=BatchingMethod.CONTINUOUS,
            max_running_requests=16,
        ).abbreviated_workload_id

    def cfg_abbr(cfg: EngineConfig) -> str:
        return CaseKey(
            "w",
            caching_method=cfg.kv_cache_config.kind,
            batching_method=cfg.batching_method,
            max_running_requests=16,
            flags=cfg.flags,
        ).abbreviated_engine_config

    assert wid_abbr("variable_output_length") == "var_out_len"
    assert wid_abbr("uniform_output_length") == "uni_out_len"
    assert wid_abbr("staggered_arrivals") == "sta_arr"
    assert wid_abbr("prefill_interference") == "pre_int"
    assert wid_abbr("long_short_stress") == "lon_sho_str"
    assert wid_abbr("length_ratio_128") == "len_rat_128"
    assert wid_abbr("block_size_sweep") == "blo_siz_swe"
    assert wid_abbr("workload_A") == "wor_A"
    assert wid_abbr("") == ""

    assert _abbreviate_suite_name("dense-vs-paged-long-short-stress") == "den-vs-pag-lon-sho-str"
    assert _abbreviate_suite_name("dense-vs-paged-throughput") == "den-vs-pag-thr"
    assert _abbreviate_suite_name("test-suite") == "tes-sui"
    assert _abbreviate_suite_name("custom") == "cus"
    assert _abbreviate_suite_name("") == ""

    assert cfg_abbr(EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS)) == "dkc+cb"
    assert cfg_abbr(EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED)) == "dkc+fb"
    assert cfg_abbr(EngineConfig(PagedCacheConfig(), batching_method=BatchingMethod.CONTINUOUS)) == "pkc+cb"
    assert cfg_abbr(EngineConfig(NoCacheConfig())) == "nc"

    dense_key = CaseKey(
        "variable_output_length",
        caching_method="dense-kv-cache",
        batching_method=BatchingMethod.CONTINUOUS,
        max_running_requests=16,
    )
    assert dense_key.abbreviated_display_name == "[var_out_len] | dkc+cb | mrr=16"

    paged_key = CaseKey(
        "long_short_stress",
        caching_method="paged-kv-cache",
        batching_method=BatchingMethod.CONTINUOUS,
        max_running_requests=16,
        block_size=16,
        num_blocks=64,
    )
    assert paged_key.abbreviated_display_name == "[lon_sho_str] | pkc+cb | mrr=16 | blk-sz=16"
