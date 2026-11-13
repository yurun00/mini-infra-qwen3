"""pytest 包装：复用 mini_infra.verification.smoke 的自检逻辑（CPU 可运行）。"""

from mini_infra.verification.smoke import (
    build_tiny_model,
    check_causal_mask,
    check_continuous_batching,
    check_fixed_batching,
    check_kv_cache_correctness,
    check_output_shape_and_tied_weights,
    check_paged_kv_cache,
    check_token_budget_scheduling,
)


def test_output_shape_and_tied_weights():
    check_output_shape_and_tied_weights(build_tiny_model())


def test_causal_mask_ignores_future_tokens():
    check_causal_mask(build_tiny_model())


def test_kv_cache_correctness():
    check_kv_cache_correctness(build_tiny_model())


def test_fixed_batching():
    check_fixed_batching(build_tiny_model())


def test_continuous_batching():
    check_continuous_batching(build_tiny_model())


def test_paged_kv_cache():
    check_paged_kv_cache(build_tiny_model())


def test_token_budget_scheduling():
    check_token_budget_scheduling(build_tiny_model())
