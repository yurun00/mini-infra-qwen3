"""bench.scheduler.suites 配置与规格构建的单元测试（CPU 可运行）。"""

import json

import pytest

from mini_infra.bench.scheduler.suites import (
    SCHEDULER_SUITES,
    DenseVsPagedBlockSizeSuite,
    DenseVsPagedLengthRatioSuite,
    DenseVsPagedLongShortStressSuite,
    FixedVsContinuousPrefillInterferenceSuite,
    FixedVsContinuousStaggeredArrivalsSuite,
    FixedVsContinuousUniformOutputLengthSuite,
    FixedVsContinuousVariableOutputLengthSuite,
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
        assert suite.interleave == (BatchingMethod,)
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
        assert [c.id for c in cases] == ["fixed", "continuous"]
        assert cases[0].id == "fixed"
        assert cases[1].id == "continuous"
        assert cases[0].mode.batching_method == BatchingMethod.FIXED
        assert cases[1].mode.batching_method == BatchingMethod.CONTINUOUS
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
    assert suite.interleave == (CachingMethod,)
    modes = suite.modes()
    assert len(modes) == 2
    assert [m.caching_method for m in modes] == [CachingMethod.DENSE, CachingMethod.PAGED]
    assert all(m.batching_method == BatchingMethod.CONTINUOUS for m in modes)

    specs = suite.build_specs(vocab_size=1000)
    assert len(specs) == 17

    cases = suite.cases(vocab_size=1000)
    assert len(cases) == 2
    assert [c.id for c in cases] == ["dense", "paged"]
    assert cases[0].mode.caching_method == CachingMethod.DENSE
    assert cases[0].paged_cache_config is None
    assert cases[1].mode.caching_method == CachingMethod.PAGED
    assert cases[1].paged_cache_config is not None
    assert cases[1].paged_cache_config.block_size == 16

    data = suite.to_dict()
    json.dumps(data)
    assert data["name"] == "dense-vs-paged-long-short-stress"
    assert data["block_size"] == 16
    assert data["interleave"] == ["caching_method"]

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
    assert suite.interleave == (CachingMethod,)
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
    assert [c.id for c in cases] == ["dense", "paged"] * 4

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
    assert suite.interleave == (CachingMethod,)
    assert suite.block_sizes == (8, 16, 32, 64)

    specs = suite.build_specs(vocab_size=1000)
    assert len(specs) == 17

    cases = suite.cases(vocab_size=1000)
    # 1 个 dense + 4 个 paged -> 共 5 个 cases
    assert len(cases) == 5
    assert [c.id for c in cases] == ["dense", "paged-block-8", "paged-block-16", "paged-block-32", "paged-block-64"]
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
            id="paged",
            suite_name="test",
            mode=paged_mode,
            max_running_requests=2,
            specs=(spec,),
            paged_cache_config=None,
        )

    # 2. 非 Paged 模式传 paged_cache_config 必须报错
    with pytest.raises(ValueError, match="非 Paged 模式 .* 不得提供 paged_cache_config"):
        SchedulerSuiteCase(
            id="dense",
            suite_name="test",
            mode=dense_mode,
            max_running_requests=2,
            specs=(spec,),
            paged_cache_config=PagedCacheConfig(block_size=16),
        )

    # 3. 正确传参合法构造
    case_paged = SchedulerSuiteCase(
        id="paged",
        suite_name="test",
        mode=paged_mode,
        max_running_requests=2,
        specs=(spec,),
        workload_id="workload_0",
        paged_cache_config=PagedCacheConfig(block_size=16),
    )
    assert case_paged.workload_id == "workload_0"
    assert case_paged.paged_cache_config is not None
    assert case_paged.paged_cache_config.block_size == 16

    spec_dict = case_paged.to_spec_dict()
    assert spec_dict["workload_id"] == "workload_0"
    assert spec_dict["block_size"] == 16
