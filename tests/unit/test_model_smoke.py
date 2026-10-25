"""pytest 包装：复用 mini_infra.verification.smoke_test 的自检逻辑（CPU 可运行）。"""

from mini_infra.verification.smoke_test import (
    build_tiny_model,
    check_causal_mask,
    check_fixed_batching,
    check_kv_cache_correctness,
    check_output_shape_and_tied_weights,
)


def test_output_shape_and_tied_weights():
    check_output_shape_and_tied_weights(build_tiny_model())


def test_causal_mask_ignores_future_tokens():
    check_causal_mask(build_tiny_model())


def test_kv_cache_correctness():
    check_kv_cache_correctness(build_tiny_model())


def test_fixed_batching():
    check_fixed_batching(build_tiny_model())
