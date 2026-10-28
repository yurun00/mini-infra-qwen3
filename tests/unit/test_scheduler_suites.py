"""bench.scheduler.suites 配置与规格构建的单元测试（CPU 可运行）。"""

import json

import pytest

from mini_infra.bench.scheduler.suites import (
    SCHEDULER_SUITES,
    FixedVsContinuousPrefillInterferenceSuite,
    FixedVsContinuousStaggeredArrivalsSuite,
    FixedVsContinuousUniformOutputLengthSuite,
    FixedVsContinuousVariableOutputLengthSuite,
    SchedulerSuiteCase,
    get_all_scheduler_suites,
    get_scheduler_suite,
)
from mini_infra.engine import BatchingMethod, FeatureFlag, Mode


def test_scheduler_suites_registry():
    assert set(SCHEDULER_SUITES.keys()) == {
        "fixed-vs-continuous-variable-output-length",
        "fixed-vs-continuous-uniform-output-length",
        "fixed-vs-continuous-staggered-arrivals",
        "fixed-vs-continuous-prefill-interference",
    }
    assert len(get_all_scheduler_suites()) == 4
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


def test_suite_cases_expansion():
    for suite in get_all_scheduler_suites():
        assert suite.interleave == (BatchingMethod,)
        modes = suite.modes()
        assert len(modes) == 2
        assert all(isinstance(m, Mode) for m in modes)
        assert all(m.is_enabled(FeatureFlag.KV_CACHE) for m in modes)

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
