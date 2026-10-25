"""Mode 与 FeatureFlag / BatchingMethod 的单元测试。"""

import pytest

from mini_infra.generation.mode import BatchingMethod, FeatureFlag, Mode


def test_feature_flag():
    assert FeatureFlag.from_str("kv-cache") == FeatureFlag.KV_CACHE
    assert FeatureFlag.from_str(FeatureFlag.KV_CACHE) == FeatureFlag.KV_CACHE
    assert str(FeatureFlag.KV_CACHE) == "kv-cache"

    with pytest.raises(ValueError, match="未知特性标志"):
        FeatureFlag.from_str("nonexistent-feature")


def test_batching_method():
    assert BatchingMethod.from_str(None) == BatchingMethod.NONE
    assert BatchingMethod.from_str("none") == BatchingMethod.NONE
    assert BatchingMethod.from_str("fixed-batching") == BatchingMethod.FIXED
    assert BatchingMethod.from_str("continuous-batching") == BatchingMethod.CONTINUOUS
    assert str(BatchingMethod.FIXED) == "fixed-batching"

    with pytest.raises(ValueError, match="未知批处理策略"):
        BatchingMethod.from_str("unknown-batching")


def test_mode_default():
    mode = Mode()
    assert not mode.flags
    assert mode.batching_method == BatchingMethod.NONE
    assert not mode.kv_cache
    assert not mode.fixed_batching
    assert not mode.continuous_batching
    assert not mode.is_batch
    assert str(mode) == "no-cache"


def test_mode_with_flag():
    mode = Mode()
    mode_with_kv = mode.with_flag(FeatureFlag.KV_CACHE, True)
    assert mode_with_kv.is_enabled(FeatureFlag.KV_CACHE)
    assert mode_with_kv.is_enabled("kv-cache")
    assert mode_with_kv.kv_cache
    assert str(mode_with_kv) == "kv-cache"

    mode_without_kv = mode_with_kv.with_flag(FeatureFlag.KV_CACHE, False)
    assert not mode_without_kv.is_enabled(FeatureFlag.KV_CACHE)
    assert not mode_without_kv.kv_cache


def test_mode_with_batching_method():
    mode = Mode({FeatureFlag.KV_CACHE})
    mode_fixed = mode.with_batching_method(BatchingMethod.FIXED)
    assert mode_fixed.fixed_batching
    assert mode_fixed.is_batch
    assert not mode_fixed.continuous_batching
    assert mode_fixed.batching_method == BatchingMethod.FIXED
    assert str(mode_fixed) == "kv-cache+fixed-batching"


def test_mode_to_dict():
    mode = Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.FIXED)
    d = mode.to_dict()
    assert d == {
        "kv-cache": True,
        "batching_method": "fixed-batching",
    }
    assert "batching" not in d  # 确保无历史兼容冗余键


def test_mode_contains():
    mode = Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.FIXED)
    assert FeatureFlag.KV_CACHE in mode
    assert "kv-cache" in mode
    assert BatchingMethod.FIXED in mode
    assert "fixed-batching" in mode
    assert BatchingMethod.CONTINUOUS not in mode
