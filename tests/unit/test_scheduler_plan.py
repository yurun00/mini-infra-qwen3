"""bench.scheduler.plan 执行计划与特征调度算法的单元测试（纯 CPU）。"""

import pytest

from mini_infra.bench.cases import (
    BenchmarkCase,
    CaseDimension,
    CaseKey,
    WorkloadProfile,
)
from mini_infra.bench.scheduler.plan import (
    RunPhase,
    build_scheduler_run_plan,
    format_scheduler_run_plan,
)
from mini_infra.core import DenseCacheConfig, NoCacheConfig, PagedCacheConfig, Request
from mini_infra.engine import BatchingMethod


def make_dummy_case(
    batching_method: BatchingMethod,
    kv_cache: bool,
    suite_name: str = "test-suite",
    workload_id: str = "test_workload",
) -> BenchmarkCase:
    kv_cache_config = DenseCacheConfig() if kv_cache else NoCacheConfig()
    req = Request(request_idx=0, token_ids=[1, 2], max_new_tokens=2, arrival_ms=0.0)
    return BenchmarkCase(
        profile=WorkloadProfile.from_requests(workload_id, (req,)),
        batching_method=batching_method,
        kv_cache_config=kv_cache_config,
        max_running_requests=2,
        suite_name=suite_name,
        engine="scheduler",
    )


def test_parse_interleave_dimension():
    assert CaseDimension.from_str("batching_method") == CaseDimension.BATCHING_METHOD
    assert CaseDimension.from_str("caching_method") == CaseDimension.CACHING_METHOD
    assert CaseDimension.from_str("block_size") == CaseDimension.BLOCK_SIZE
    assert CaseDimension.from_str("num_blocks") == CaseDimension.NUM_BLOCKS
    assert CaseDimension.from_str("max_running_requests") == CaseDimension.MAX_RUNNING_REQUESTS
    assert CaseDimension.from_str("workload_id") == CaseDimension.WORKLOAD_ID

    with pytest.raises(ValueError, match="未知的 interleave 特征维度"):
        CaseDimension.from_str("batch_method")
    with pytest.raises(ValueError, match="未知的 interleave 特征维度"):
        CaseDimension.from_str("batching-method")
    with pytest.raises(ValueError, match="未知的 interleave 特征维度"):
        CaseDimension.from_str("kv-cache")
    with pytest.raises(ValueError, match="未知的 interleave 特征维度"):
        CaseDimension.from_str("none")
    with pytest.raises(ValueError, match="未知的 interleave 特征维度"):
        CaseDimension.from_str("unsupported_dimension")


def test_case_key_dimension_value():
    case = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=True)
    assert case.key.value(CaseDimension.BATCHING_METHOD) == BatchingMethod.CONTINUOUS
    assert case.key.value(CaseDimension.CACHING_METHOD) == "dense-kv-cache"
    assert case.key.value(CaseDimension.WORKLOAD_ID) == "test_workload"
    assert case.key.value(CaseDimension.MAX_RUNNING_REQUESTS) == 2
    assert case.key.value(CaseDimension.BLOCK_SIZE) is None
    assert case.key.value(CaseDimension.NUM_BLOCKS) is None
    assert case.key.canonical_id == "workload=test_workload/cache=dense-kv-cache/batch=continuous-batching/mrr=2"
    dim_map = case.key.as_dimension_map()
    assert dim_map[CaseDimension.WORKLOAD_ID] == "test_workload"
    assert dim_map[CaseDimension.CACHING_METHOD] == "dense-kv-cache"

    # 验证 to_dict 与 from_dict 的对偶恢复
    key_dict = case.key.to_dict()
    restored_key = CaseKey.from_dict(key_dict)
    assert restored_key == case.key
    assert restored_key.canonical_id == case.key.canonical_id

    # 验证带 block_size 与 num_blocks 时的 display_name 与恢复
    paged_key = CaseKey(
        "custom_wl",
        caching_method="paged-kv-cache",
        batching_method=BatchingMethod.CONTINUOUS,
        max_running_requests=16,
        block_size=16,
        num_blocks=64,
    )
    assert "paged-kv-cache(block_size=16,num_blocks=64)+continuous-batching+mrr=16" in paged_key.display_name
    restored_paged = CaseKey.from_dict(paged_key.to_dict())
    assert restored_paged == paged_key
    assert restored_paged.block_size == 16
    assert restored_paged.num_blocks == 64

    with pytest.raises(ValueError, match="is not a valid CaseDimension"):
        case.key.value("invalid_dimension")


def test_interleave_batching_method_unselected_kv_cache():
    """验证核心场景：选中 batching_method，未选中 kv-cache。

    4 种组合：
      c1: fixed, kv_cache=True
      c2: continuous, kv_cache=True
      c3: fixed, kv_cache=False
      c4: continuous, kv_cache=False

    预期行为：
      Group 0 (kv_cache=True): 连续跑完所有 repeat（fixed 与 continuous 轮换）
      Group 1 (kv_cache=False): 连续跑完所有 repeat（fixed 与 continuous 轮换）
    """
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    c2 = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=True)
    c3 = make_dummy_case(BatchingMethod.FIXED, kv_cache=False)
    c4 = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=False)
    cases = [c1, c2, c3, c4]

    plan = build_scheduler_run_plan(cases, warmup=1, repeats=3, interleave=(CaseDimension.BATCHING_METHOD,))

    # 总步数：每个 Group 有 2 cases * 1 warmup + 2 cases * 3 repeats = 8 步，共 2 Groups = 16 步
    assert len(plan) == 16

    # 拆解验证 Group 0 阶段
    g0_steps = [p for p in plan if p.group_id == 0]
    g1_steps = [p for p in plan if p.group_id == 1]
    assert len(g0_steps) == 8
    assert len(g1_steps) == 8

    # 验证 Group 0 全部排在 Group 1 之前（为选中的 feature 连续跑完所有 repeat）
    assert plan[:8] == g0_steps
    assert plan[8:] == g1_steps

    # Group 0 warmup 阶段
    assert g0_steps[0].phase == RunPhase.WARMUP and g0_steps[0].case == c1
    assert g0_steps[1].phase == RunPhase.WARMUP and g0_steps[1].case == c2

    # Group 0 measure 阶段：按 repeat 轮换顺序
    # Round 0 (偶数，正序): c1, c2
    assert g0_steps[2].repeat_idx == 0 and g0_steps[2].case == c1
    assert g0_steps[3].repeat_idx == 0 and g0_steps[3].case == c2
    # Round 1 (奇数，反序): c2, c1
    assert g0_steps[4].repeat_idx == 1 and g0_steps[4].case == c2
    assert g0_steps[5].repeat_idx == 1 and g0_steps[5].case == c1
    # Round 2 (偶数，正序): c1, c2
    assert g0_steps[6].repeat_idx == 2 and g0_steps[6].case == c1
    assert g0_steps[7].repeat_idx == 2 and g0_steps[7].case == c2

    # Group 1 验证同样的结构
    assert g1_steps[0].phase == RunPhase.WARMUP and g1_steps[0].case == c3
    assert g1_steps[1].phase == RunPhase.WARMUP and g1_steps[1].case == c4
    # Round 0 (偶数，正序): c3, c4
    assert g1_steps[2].repeat_idx == 0 and g1_steps[2].case == c3
    assert g1_steps[3].repeat_idx == 0 and g1_steps[3].case == c4
    # Round 1 (奇数，反序): c4, c3
    assert g1_steps[4].repeat_idx == 1 and g1_steps[4].case == c4
    assert g1_steps[5].repeat_idx == 1 and g1_steps[5].case == c3
    # Round 2 (偶数，正序): c3, c4
    assert g1_steps[6].repeat_idx == 2 and g1_steps[6].case == c3
    assert g1_steps[7].repeat_idx == 2 and g1_steps[7].case == c4


def test_interleave_none_pure_serial():
    """验证不交替时（interleave 为空），各用例按纯串行执行。"""
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    c2 = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=True)

    plan = build_scheduler_run_plan([c1, c2], warmup=1, repeats=2, interleave=())
    assert len(plan) == 6

    # c1 跑完 warmup 与 2 轮 repeats
    assert plan[0].case == c1 and plan[0].phase == RunPhase.WARMUP
    assert plan[1].case == c1 and plan[1].phase == RunPhase.MEASURE and plan[1].repeat_idx == 0
    assert plan[2].case == c1 and plan[2].phase == RunPhase.MEASURE and plan[2].repeat_idx == 1

    # c2 跑完 warmup 与 2 轮 repeats
    assert plan[3].case == c2 and plan[3].phase == RunPhase.WARMUP
    assert plan[4].case == c2 and plan[4].phase == RunPhase.MEASURE and plan[4].repeat_idx == 0
    assert plan[5].case == c2 and plan[5].phase == RunPhase.MEASURE and plan[5].repeat_idx == 1


def test_interleave_all_features():
    """验证全部特征交替时，所有用例处于同一个 Group 并每轮轮转。"""
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    c2 = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=True)
    c3 = make_dummy_case(BatchingMethod.FIXED, kv_cache=False)

    plan = build_scheduler_run_plan(
        [c1, c2, c3],
        warmup=0,
        repeats=2,
        interleave=(CaseDimension.BATCHING_METHOD, CaseDimension.CACHING_METHOD),
    )
    assert len(plan) == 6
    assert all(p.group_id == 0 for p in plan)

    # 3 个元素轮转：Round 0 为 [0, 1, 2]，Round 1 shift 1 为 [1, 2, 0]
    r0_cases = [p.case for p in plan[:3]]
    r1_cases = [p.case for p in plan[3:]]
    assert r0_cases == [c1, c2, c3]
    assert r1_cases == [c2, c3, c1]


def test_interleave_invalid_type():
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    with pytest.raises(TypeError, match="未知的 interleave 特征维度"):
        build_scheduler_run_plan([c1], warmup=1, repeats=2, interleave=("unknown_feature",))  # type: ignore[arg-type]


def test_format_scheduler_run_plan():
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    c2 = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=True)
    plan = build_scheduler_run_plan([c1, c2], warmup=1, repeats=2, interleave=(CaseDimension.BATCHING_METHOD,))
    output = format_scheduler_run_plan(plan)
    assert "共 6 步" in output
    assert "warmup" in output
    assert "repeat 1" in output


def test_empty_plan():
    assert build_scheduler_run_plan([], warmup=1, repeats=2) == []
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    assert build_scheduler_run_plan([c1], warmup=1, repeats=0) == []


def test_multi_workload_id_grouping_and_interleaving():
    req = Request(request_idx=0, token_ids=[1, 2], max_new_tokens=2, arrival_ms=0.0)

    # 两个不同的 workload，每个 workload 包含 dense 和 paged 两种 caching_method
    w0_dense = BenchmarkCase(
        profile=WorkloadProfile.from_requests("workload_0", (req,)),
        batching_method=BatchingMethod.CONTINUOUS,
        kv_cache_config=DenseCacheConfig(),
        max_running_requests=2,
        suite_name="test-multi-workload",
        engine="scheduler",
    )
    w0_paged = BenchmarkCase(
        profile=WorkloadProfile.from_requests("workload_0", (req,)),
        batching_method=BatchingMethod.CONTINUOUS,
        kv_cache_config=PagedCacheConfig(block_size=16),
        max_running_requests=2,
        suite_name="test-multi-workload",
        engine="scheduler",
    )
    w1_dense = BenchmarkCase(
        profile=WorkloadProfile.from_requests("workload_1", (req,)),
        batching_method=BatchingMethod.CONTINUOUS,
        kv_cache_config=DenseCacheConfig(),
        max_running_requests=2,
        suite_name="test-multi-workload",
        engine="scheduler",
    )
    w1_paged = BenchmarkCase(
        profile=WorkloadProfile.from_requests("workload_1", (req,)),
        batching_method=BatchingMethod.CONTINUOUS,
        kv_cache_config=PagedCacheConfig(block_size=16),
        max_running_requests=2,
        suite_name="test-multi-workload",
        engine="scheduler",
    )

    cases = [w0_dense, w0_paged, w1_dense, w1_paged]
    plan = build_scheduler_run_plan(
        cases,
        warmup=1,
        repeats=2,
        interleave=(CaseDimension.CACHING_METHOD, CaseDimension.BLOCK_SIZE),
    )

    # 每个 workload 对应 1 个 Group：2 cases * 1 warmup + 2 cases * 2 repeats = 6 步，共 2 Groups = 12 步
    assert len(plan) == 12

    g0_steps = [p for p in plan if p.group_id == 0]
    g1_steps = [p for p in plan if p.group_id == 1]
    assert len(g0_steps) == 6
    assert len(g1_steps) == 6
    assert all(p.case.workload_id == "workload_0" for p in g0_steps)
    assert all(p.case.workload_id == "workload_1" for p in g1_steps)

    # Group 0 内部交替
    assert g0_steps[0].phase == RunPhase.WARMUP and g0_steps[0].case == w0_dense
    assert g0_steps[1].phase == RunPhase.WARMUP and g0_steps[1].case == w0_paged
    # Repeat 1 (偶数轮，正序: dense -> paged)
    assert g0_steps[2].repeat_idx == 0 and g0_steps[2].case == w0_dense
    assert g0_steps[3].repeat_idx == 0 and g0_steps[3].case == w0_paged
    # Repeat 2 (奇数轮，反序: paged -> dense)
    assert g0_steps[4].repeat_idx == 1 and g0_steps[4].case == w0_paged
    assert g0_steps[5].repeat_idx == 1 and g0_steps[5].case == w0_dense

    formatted = format_scheduler_run_plan(plan)
    assert "[workload_0]" in formatted
    assert "[workload_1]" in formatted


def test_existing_suites_plan_golden_invariance():
    from mini_infra.bench.suite import get_all_benchmark_suites

    fixed_suites = [s for s in get_all_benchmark_suites("scheduler") if s.name.startswith("fixed-vs-continuous")]
    assert len(fixed_suites) == 4
    for suite in fixed_suites:
        cases = suite.cases
        plan = build_scheduler_run_plan(cases, warmup=1, repeats=3, interleave=suite.interleave)
        assert all(p.group_id == 0 for p in plan)
        assert len(plan) == 8  # 2 warmup + 2*3 measure
        assert all(p.case.workload_id == cases[0].workload_id for p in plan)
        formatted = format_scheduler_run_plan(plan)
        assert f"[{cases[0].workload_id}]" in formatted


def test_dense_vs_paged_suites_plan():
    from mini_infra.bench.suite import get_benchmark_suite

    # 1. dense-vs-paged-long-short-stress: 1 workload, 2 cases (dense, paged)
    stress_suite = get_benchmark_suite("scheduler", "dense-vs-paged-long-short-stress")
    cases = stress_suite.cases
    assert len(cases) == 2
    plan = build_scheduler_run_plan(cases, warmup=1, repeats=3, interleave=stress_suite.interleave)
    assert all(p.group_id == 0 for p in plan)
    assert len(plan) == 8

    # 2. dense-vs-paged-length-ratio: 4 workloads (length_ratio_128, 512, 2048, 3072), 8 cases, 4 groups
    ratio_suite = get_benchmark_suite("scheduler", "dense-vs-paged-length-ratio")
    cases = ratio_suite.cases
    assert len(cases) == 8
    plan = build_scheduler_run_plan(cases, warmup=1, repeats=3, interleave=ratio_suite.interleave)
    assert len(plan) == 32
    assert {p.group_id for p in plan} == {0, 1, 2, 3}
    formatted = format_scheduler_run_plan(plan)
    assert "[length_ratio_128]" in formatted
    assert "[length_ratio_3072]" in formatted

    # 3. dense-vs-paged-block-size: 1 workload (block_size_sweep), 5 cases (dense + 4 paged), 1 group
    bs_suite = get_benchmark_suite("scheduler", "dense-vs-paged-block-size")
    cases = bs_suite.cases
    assert len(cases) == 5
    plan = build_scheduler_run_plan(cases, warmup=1, repeats=3, interleave=bs_suite.interleave)
    assert all(p.group_id == 0 for p in plan)
    assert len(plan) == 20
    formatted = format_scheduler_run_plan(plan)
    assert "[block_size_sweep]" in formatted


def test_build_scheduler_run_plan_duplicate_case_key():
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True, workload_id="w0")
    c2 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True, workload_id="w0")
    with pytest.raises(ValueError, match="存在重复的用例身份键（CaseKey）"):
        build_scheduler_run_plan([c1, c2], warmup=1, repeats=2)


def test_build_scheduler_run_plan_inconsistent_specs():
    req1 = Request(request_idx=0, token_ids=[1, 2], max_new_tokens=2, arrival_ms=0.0)
    req2 = Request(request_idx=0, token_ids=[1, 2, 3], max_new_tokens=2, arrival_ms=0.0)
    c1 = BenchmarkCase(
        profile=WorkloadProfile.from_requests("w0", (req1,)),
        batching_method=BatchingMethod.FIXED,
        kv_cache_config=DenseCacheConfig(),
        max_running_requests=2,
        suite_name="s",
        engine="scheduler",
    )
    c2 = BenchmarkCase(
        profile=WorkloadProfile.from_requests("w0", (req2,)),
        batching_method=BatchingMethod.CONTINUOUS,
        kv_cache_config=DenseCacheConfig(),
        max_running_requests=2,
        suite_name="s",
        engine="scheduler",
    )
    with pytest.raises(ValueError, match="同一 workload_id 'w0' 下的用例请求规格（profile）不一致"):
        build_scheduler_run_plan([c1, c2], warmup=1, repeats=2)
