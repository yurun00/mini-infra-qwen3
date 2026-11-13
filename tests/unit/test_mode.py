"""EngineConfig 与 FeatureFlag / KVCacheConfig / BatchingMethod 的单元测试。"""

import pytest

from mini_infra.core import (
    DenseCacheConfig,
    NoCacheConfig,
    PagedCacheConfig,
    kv_cache_config_from_str,
    parse_kv_cache_kind,
)
from mini_infra.engine.config import BatchingMethod, EngineConfig, FeatureFlag, validate_engine_config


def test_feature_flag():
    assert FeatureFlag.from_str("placeholder") == FeatureFlag.PLACEHOLDER
    assert FeatureFlag.from_str(FeatureFlag.PLACEHOLDER) == FeatureFlag.PLACEHOLDER
    assert str(FeatureFlag.PLACEHOLDER) == "placeholder"

    with pytest.raises(ValueError, match="未知特性标志"):
        FeatureFlag.from_str("nonexistent-feature")


def test_kv_cache_kind_and_config_from_str():
    assert parse_kv_cache_kind(None) == "none"
    assert parse_kv_cache_kind("none") == "none"
    assert parse_kv_cache_kind("dense-kv-cache") == "dense-kv-cache"
    assert parse_kv_cache_kind("paged-kv-cache") == "paged-kv-cache"
    assert kv_cache_config_from_str(None) == NoCacheConfig()
    assert kv_cache_config_from_str("dense-kv-cache") == DenseCacheConfig()
    assert kv_cache_config_from_str("paged-kv-cache") == PagedCacheConfig()
    assert kv_cache_config_from_str("paged-kv-cache", block_size=8, num_blocks=32) == PagedCacheConfig(8, 32)

    with pytest.raises(ValueError, match="未知 KV Cache 策略"):
        parse_kv_cache_kind("unknown-kv-cache")


def test_batching_method():
    assert BatchingMethod.from_str(None) == BatchingMethod.NONE
    assert BatchingMethod.from_str("none") == BatchingMethod.NONE
    assert BatchingMethod.from_str("fixed-batching") == BatchingMethod.FIXED
    assert BatchingMethod.from_str("continuous-batching") == BatchingMethod.CONTINUOUS
    assert BatchingMethod.from_str(BatchingMethod.FIXED) == BatchingMethod.FIXED
    assert str(BatchingMethod.FIXED) == "fixed-batching"

    with pytest.raises(ValueError, match="未知批处理策略"):
        BatchingMethod.from_str("unknown-batching")


def test_engine_config_default():
    cfg = EngineConfig()
    assert not cfg.flags
    assert cfg.kv_cache_config == NoCacheConfig()
    assert cfg.batching_method == BatchingMethod.NONE
    assert cfg.max_running_requests is None
    assert not cfg.has_kv_cache
    assert not cfg.is_batch
    assert str(cfg) == "no-cache"


def test_engine_config_with_flag():
    cfg = EngineConfig().with_flag(FeatureFlag.PLACEHOLDER, True)
    assert FeatureFlag.PLACEHOLDER in cfg.flags
    assert str(cfg) == "placeholder+no-cache"

    assert FeatureFlag.PLACEHOLDER not in cfg.with_flag(FeatureFlag.PLACEHOLDER, False).flags


def test_engine_config_with_kv_cache_config():
    cfg = EngineConfig(batching_method=BatchingMethod.FIXED)
    dense = cfg.with_kv_cache_config(DenseCacheConfig())
    assert dense.has_kv_cache
    assert dense.kv_cache_config == DenseCacheConfig()
    assert str(dense) == "dense-kv-cache+fixed-batching"

    paged = dense.with_kv_cache_config(PagedCacheConfig(block_size=8))
    assert paged.has_kv_cache
    assert paged.kv_cache_config == PagedCacheConfig(block_size=8)
    assert str(paged) == "paged-kv-cache(block_size=8)+fixed-batching"


def test_engine_config_with_batching_method_and_max_running():
    cfg = EngineConfig(DenseCacheConfig(), max_running_requests=4)
    cfg_fixed = cfg.with_batching_method(BatchingMethod.FIXED)
    assert cfg_fixed.is_batch
    assert cfg_fixed.batching_method == BatchingMethod.FIXED
    assert cfg_fixed.max_running_requests == 4
    assert str(cfg_fixed) == "dense-kv-cache+fixed-batching+mrr=4"

    with pytest.raises(ValueError, match="max_running_requests 必须大于 0"):
        EngineConfig(DenseCacheConfig(), max_running_requests=0)


def test_engine_config_paged_block_size_inequality():
    paged_8 = EngineConfig(PagedCacheConfig(block_size=8), batching_method=BatchingMethod.CONTINUOUS)
    paged_16 = EngineConfig(PagedCacheConfig(block_size=16, num_blocks=64), batching_method=BatchingMethod.CONTINUOUS)
    assert paged_8 != paged_16
    assert str(paged_8) == "paged-kv-cache(block_size=8)+continuous-batching"
    assert str(paged_16) == "paged-kv-cache(block_size=16,num_blocks=64)+continuous-batching"
    assert paged_8.with_kv_cache_config(DenseCacheConfig()) == EngineConfig(
        DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS
    )


def test_validate_engine_config():
    validate_engine_config(EngineConfig(PagedCacheConfig(), batching_method=BatchingMethod.CONTINUOUS))
    with pytest.raises(ValueError, match="paged-kv-cache 仅支持"):
        validate_engine_config(EngineConfig(PagedCacheConfig(), batching_method=BatchingMethod.NONE))
    with pytest.raises(ValueError, match="continuous-batching 依赖 KV Cache"):
        validate_engine_config(EngineConfig(NoCacheConfig(), batching_method=BatchingMethod.CONTINUOUS))
