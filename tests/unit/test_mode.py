"""Mode 与 FeatureFlag / CachingMethod / BatchingMethod 的单元测试。"""

import pytest

from mini_infra.engine.mode import BatchingMethod, CachingMethod, FeatureFlag, Mode


def test_feature_flag():
    assert FeatureFlag.from_str("placeholder") == FeatureFlag.PLACEHOLDER
    assert FeatureFlag.from_str(FeatureFlag.PLACEHOLDER) == FeatureFlag.PLACEHOLDER
    assert str(FeatureFlag.PLACEHOLDER) == "placeholder"

    with pytest.raises(ValueError, match="未知特性标志"):
        FeatureFlag.from_str("nonexistent-feature")


def test_caching_method():
    assert CachingMethod.from_str(None) == CachingMethod.NONE
    assert CachingMethod.from_str("none") == CachingMethod.NONE
    assert CachingMethod.from_str("dense-kv-cache") == CachingMethod.DENSE
    assert CachingMethod.from_str("paged-kv-cache") == CachingMethod.PAGED
    assert CachingMethod.from_str(CachingMethod.PAGED) == CachingMethod.PAGED
    assert str(CachingMethod.DENSE) == "dense-kv-cache"

    with pytest.raises(ValueError, match="未知 KV Cache 策略"):
        CachingMethod.from_str("unknown-kv-cache")


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
    assert mode.caching_method == CachingMethod.NONE
    assert mode.batching_method == BatchingMethod.NONE
    assert not mode.use_kv_cache
    assert not mode.is_batch
    assert str(mode) == "no-cache"


def test_mode_with_flag():
    mode = Mode().with_flag(FeatureFlag.PLACEHOLDER, True)
    assert mode.is_enabled(FeatureFlag.PLACEHOLDER)
    assert str(mode) == "placeholder"

    assert not mode.with_flag(FeatureFlag.PLACEHOLDER, False).is_enabled(FeatureFlag.PLACEHOLDER)


def test_mode_with_caching_method():
    mode = Mode(batching_method=BatchingMethod.FIXED)
    dense = mode.with_caching_method(CachingMethod.DENSE)
    assert dense.use_kv_cache
    assert dense.caching_method == CachingMethod.DENSE
    assert str(dense) == "dense-kv-cache+fixed-batching"

    paged = dense.with_caching_method("paged-kv-cache")
    assert paged.use_kv_cache
    assert str(paged) == "paged-kv-cache+fixed-batching"


def test_mode_with_batching_method():
    mode = Mode(CachingMethod.DENSE)
    mode_fixed = mode.with_batching_method(BatchingMethod.FIXED)
    assert mode_fixed.is_batch
    assert mode_fixed.batching_method == BatchingMethod.FIXED
    assert str(mode_fixed) == "dense-kv-cache+fixed-batching"


def test_mode_feature_dimensions_and_value():
    assert Mode.FEATURE_DIMENSIONS == (FeatureFlag.PLACEHOLDER, CachingMethod, BatchingMethod)

    mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
    assert mode.feature_value(FeatureFlag.PLACEHOLDER) is False
    assert mode.feature_value(CachingMethod) == CachingMethod.PAGED
    assert mode.feature_value(BatchingMethod) == BatchingMethod.CONTINUOUS

    mode_no_kv = Mode(batching_method=BatchingMethod.FIXED)
    assert mode_no_kv.feature_value(CachingMethod) == CachingMethod.NONE
    assert mode_no_kv.feature_value(BatchingMethod) == BatchingMethod.FIXED

    for invalid in ("unknown_feature", "batching_method", "dense-kv-cache"):
        with pytest.raises(TypeError, match="未知的特征维度类型"):
            mode.feature_value(invalid)


def test_mode_with_paged_kv_cache():
    mode = Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
    assert str(mode) == "paged-kv-cache+continuous-batching"
    assert mode.with_caching_method(CachingMethod.DENSE) == Mode(
        CachingMethod.DENSE, batching_method=BatchingMethod.CONTINUOUS
    )
