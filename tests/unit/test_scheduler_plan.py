"""bench.scheduler.plan 执行计划与特征调度算法的单元测试（纯 CPU）。"""

import pytest

from mini_infra.bench.scheduler.plan import (
    RunPhase,
    build_scheduler_run_plan,
    extract_case_feature,
    format_scheduler_run_plan,
    parse_interleave_dimension,
)
from mini_infra.bench.scheduler.suites import SchedulerSuiteCase
from mini_infra.core.request import RequestSpec
from mini_infra.engine import BatchingMethod, FeatureFlag, Mode


def make_dummy_case(
    batching_method: BatchingMethod,
    kv_cache: bool,
    suite_name: str = "test-suite",
    case_id: str | None = None,
) -> SchedulerSuiteCase:
    flags = {FeatureFlag.KV_CACHE} if kv_cache else set()
    mode = Mode(flags, batching_method=batching_method)
    spec = RequestSpec(request_index=0, token_ids=[1, 2], max_new_tokens=2, arrival_ms=0.0)
    cid = case_id or ("continuous" if batching_method == BatchingMethod.CONTINUOUS else "fixed")
    return SchedulerSuiteCase(
        id=cid,
        suite_name=suite_name,
        mode=mode,
        max_running_requests=2,
        specs=(spec,),
    )


def test_parse_interleave_dimension():
    assert parse_interleave_dimension("batching_method") is BatchingMethod
    assert parse_interleave_dimension("kv-cache") is FeatureFlag.KV_CACHE

    with pytest.raises(ValueError, match="未知的 interleave 特征"):
        parse_interleave_dimension("batch_method")
    with pytest.raises(ValueError, match="未知的 interleave 特征"):
        parse_interleave_dimension("batching-method")
    with pytest.raises(ValueError, match="未知的 interleave 特征"):
        parse_interleave_dimension("kv_cache")
    with pytest.raises(ValueError, match="未知的 interleave 特征"):
        parse_interleave_dimension("none")
    with pytest.raises(ValueError, match="未知的 interleave 特征"):
        parse_interleave_dimension("unsupported_dimension")


def test_extract_case_feature():
    case = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=True)
    assert extract_case_feature(case, BatchingMethod) == BatchingMethod.CONTINUOUS
    assert extract_case_feature(case, FeatureFlag.KV_CACHE) is True

    with pytest.raises(TypeError, match="未知的特征维度类型"):
        extract_case_feature(case, "batching_method")  # type: ignore[arg-type]


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

    plan = build_scheduler_run_plan(cases, warmup=1, repeats=3, interleave=(BatchingMethod,))

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
    assert g0_steps[2].repeat_index == 0 and g0_steps[2].case == c1
    assert g0_steps[3].repeat_index == 0 and g0_steps[3].case == c2
    # Round 1 (奇数，反序): c2, c1
    assert g0_steps[4].repeat_index == 1 and g0_steps[4].case == c2
    assert g0_steps[5].repeat_index == 1 and g0_steps[5].case == c1
    # Round 2 (偶数，正序): c1, c2
    assert g0_steps[6].repeat_index == 2 and g0_steps[6].case == c1
    assert g0_steps[7].repeat_index == 2 and g0_steps[7].case == c2

    # Group 1 验证同样的结构
    assert g1_steps[0].phase == RunPhase.WARMUP and g1_steps[0].case == c3
    assert g1_steps[1].phase == RunPhase.WARMUP and g1_steps[1].case == c4
    # Round 0 (偶数，正序): c3, c4
    assert g1_steps[2].repeat_index == 0 and g1_steps[2].case == c3
    assert g1_steps[3].repeat_index == 0 and g1_steps[3].case == c4
    # Round 1 (奇数，反序): c4, c3
    assert g1_steps[4].repeat_index == 1 and g1_steps[4].case == c4
    assert g1_steps[5].repeat_index == 1 and g1_steps[5].case == c3
    # Round 2 (偶数，正序): c3, c4
    assert g1_steps[6].repeat_index == 2 and g1_steps[6].case == c3
    assert g1_steps[7].repeat_index == 2 and g1_steps[7].case == c4


def test_interleave_none_pure_serial():
    """验证不交替时（interleave 为空），各用例按纯串行执行。"""
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    c2 = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=True)

    plan = build_scheduler_run_plan([c1, c2], warmup=1, repeats=2, interleave=())
    assert len(plan) == 6

    # c1 跑完 warmup 与 2 轮 repeats
    assert plan[0].case == c1 and plan[0].phase == RunPhase.WARMUP
    assert plan[1].case == c1 and plan[1].phase == RunPhase.MEASURE and plan[1].repeat_index == 0
    assert plan[2].case == c1 and plan[2].phase == RunPhase.MEASURE and plan[2].repeat_index == 1

    # c2 跑完 warmup 与 2 轮 repeats
    assert plan[3].case == c2 and plan[3].phase == RunPhase.WARMUP
    assert plan[4].case == c2 and plan[4].phase == RunPhase.MEASURE and plan[4].repeat_index == 0
    assert plan[5].case == c2 and plan[5].phase == RunPhase.MEASURE and plan[5].repeat_index == 1


def test_interleave_all_features():
    """验证全部特征交替时，所有用例处于同一个 Group 并每轮轮转。"""
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    c2 = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=True)
    c3 = make_dummy_case(BatchingMethod.FIXED, kv_cache=False)

    plan = build_scheduler_run_plan(
        [c1, c2, c3], warmup=0, repeats=2, interleave=(BatchingMethod, FeatureFlag.KV_CACHE)
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
        build_scheduler_run_plan([c1], warmup=1, repeats=2, interleave=("batching_method",))  # type: ignore[arg-type]


def test_format_scheduler_run_plan():
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    c2 = make_dummy_case(BatchingMethod.CONTINUOUS, kv_cache=True)
    plan = build_scheduler_run_plan([c1, c2], warmup=1, repeats=2, interleave=(BatchingMethod,))
    output = format_scheduler_run_plan(plan)
    assert "共 6 步" in output
    assert "warmup" in output
    assert "repeat 1" in output


def test_empty_plan():
    assert build_scheduler_run_plan([], warmup=1, repeats=2) == []
    c1 = make_dummy_case(BatchingMethod.FIXED, kv_cache=True)
    assert build_scheduler_run_plan([c1], warmup=1, repeats=0) == []
