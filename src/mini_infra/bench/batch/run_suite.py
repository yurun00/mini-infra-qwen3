"""套件基准测试（run_suite）：执行整套 Batch BenchmarkSuite 的纯调度内核。

驱动 Suite 中的所有用例顺序执行，捕获 OOM 并收集结果。
"""

from collections.abc import Callable
from typing import Any

import torch

from mini_infra.bench.batch.run_case import run_case
from mini_infra.bench.batch.suites import Suite, SuiteCase
from mini_infra.bench.batch.types import CaseBenchmarkResult
from mini_infra.bench.common import ProfileMode
from mini_infra.profiling import TensorBoardOptions


def run_suite(
    model: torch.nn.Module,
    device: torch.device,
    suite: Suite,
    vocab: int,
    warmup: int | None = None,
    repeats: int | None = None,
    cost: dict[str, Any] | None = None,
    peaks: dict[str, float | None] | None = None,
    profile_mode: ProfileMode | None = None,
    tensorboard_options: TensorBoardOptions | None = None,
    on_case_progress: Callable[[int, int, SuiteCase], None] | None = None,
    on_case_result: Callable[[CaseBenchmarkResult], None] | None = None,
) -> list[CaseBenchmarkResult]:
    """按序执行 Suite 中的所有用例，返回评测结果列表。"""
    cases = suite.cases()
    warmup_val = warmup if warmup is not None else suite.warmup
    repeats_val = repeats if repeats is not None else suite.repeats

    case_results: list[CaseBenchmarkResult] = []
    for idx, case in enumerate(cases, 1):
        if on_case_progress is not None:
            on_case_progress(idx, len(cases), case)

        try:
            case_result = run_case(
                model=model,
                mode=case.mode,
                prompt_len=list(case.prompt_len) if isinstance(case.prompt_len, tuple) else case.prompt_len,
                max_new_tokens=(
                    list(case.max_new_tokens) if isinstance(case.max_new_tokens, tuple) else case.max_new_tokens
                ),
                batch_size=case.batch_size,
                warmup=warmup_val,
                repeats=repeats_val,
                vocab=vocab,
                device=device,
                cost=cost,
                peaks=peaks,
                profile_mode=profile_mode,
                tensorboard_options=tensorboard_options,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

        case_results.append(case_result)
        if on_case_result is not None:
            on_case_result(case_result)

    return case_results


__all__ = [
    "run_suite",
]
