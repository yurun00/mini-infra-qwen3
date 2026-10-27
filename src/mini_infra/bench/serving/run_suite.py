"""套件基准测试（run_suite）：一次调用执行整套 Serving 评测套件。

一个 ServingSuite 包含该场景下的完整用例集合。
run_suite.py 负责根据 Suite 与命令行参数生成执行计划（支持按特征 interleave），
并逐项调用 run_case 模块的原子 step 执行，最后汇总结算加速比与吞吐。
结果默认落盘至 benchmarks/suites/serving/<suite_name>/<commit>_<tag>.json。

用法示例：
    # 运行单一 Serving 套件
    mini-infra-bench-serving-suite --tag dev --suite fixed-vs-continuous-variable-output-length

    # 预览执行计划（无需 GPU 与模型）
    mini-infra-bench-serving-suite --dry-run --suite fixed-vs-continuous-variable-output-length
"""

import argparse
import sys
import time
from collections.abc import Sequence
from typing import Any

import torch

from mini_infra.bench.common import (
    BENCH_ARTIFACTS_DIR,
    add_common_args,
    check_correctness,
    git_info,
    gpu_info,
    model_cost,
    require_cuda,
    resolve_peak_specs,
    save_benchmark_record,
)
from mini_infra.bench.serving.plan import (
    RunPhase,
    build_serving_run_plan,
    format_serving_run_plan,
    parse_interleave_dimension,
)
from mini_infra.bench.serving.run_case import run_case_once
from mini_infra.bench.serving.suites import (
    SERVING_SUITES,
    ServingSuite,
    get_serving_suite,
)
from mini_infra.bench.serving.types import (
    ServingCaseResult,
    ServingSuiteResult,
    summarize_serving_runs,
)
from mini_infra.core.types import ScheduleResult
from mini_infra.engine import BatchingMethod, FeatureDimension, FeatureFlag, Mode
from mini_infra.model import Qwen3Tokenizer, load_model, resolve_model_directory

SCHEMA_VERSION = "1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-infra-qwen3 serving benchmark suite runner")
    add_common_args(p)
    p.add_argument(
        "--suite",
        choices=sorted(SERVING_SUITES),
        required=True,
        help="运行 suites.py 中注册的命名 Serving 套件（结果写入 benchmarks/suites/serving/<name>/）",
    )
    p.add_argument(
        "--interleave",
        nargs="*",
        default=None,
        help="选择交替轮换的特征维度（如 batching_method kv-cache，或 none 关闭交替；默认使用 Suite 的配置）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅生成并打印执行计划，不加载模型也不要求 CUDA",
    )
    return p.parse_args()


def parse_interleave_args(
    raw: Sequence[str] | None, default: tuple[FeatureDimension, ...]
) -> tuple[FeatureDimension, ...]:
    """将 CLI 字符串参数解析为强类型 FeatureDimension 元组。"""
    if raw is None:
        return default
    if not raw or "none" in raw:
        return ()
    return tuple(parse_interleave_dimension(f) for f in raw)


def run_suite(
    model: torch.nn.Module,
    device: torch.device,
    suite: ServingSuite,
    vocab_size: int,
    warmup: int | None = None,
    repeats: int | None = None,
    interleave: Sequence[FeatureDimension] | None = None,
) -> ServingSuiteResult:
    """运行一个 Serving 评测套件：根据计划驱动执行，支持选定特征在 repeat 间轮换交替。"""
    warmup_val = warmup if warmup is not None else suite.warmup
    repeats_val = repeats if repeats is not None else suite.repeats
    interleave_features = tuple(interleave) if interleave is not None else suite.interleave

    cases = suite.cases(vocab_size, device=device)
    mode_by_case_id = {case.id: str(case.mode) for case in cases}

    # 生成执行计划
    plan = build_serving_run_plan(cases, warmup=warmup_val, repeats=repeats_val, interleave=interleave_features)

    # 驱动执行计划，按用例 ID 归集测量结果
    collected: dict[str, list[ScheduleResult]] = {case.id: [] for case in cases}
    for planned_run in plan:
        res = run_case_once(model, device, planned_run.case)
        if planned_run.phase == RunPhase.MEASURE:
            collected[planned_run.case.id].append(res)

    case_results: dict[str, ServingCaseResult] = {
        cid: summarize_serving_runs(cid, mode_by_case_id[cid], runs) for cid, runs in collected.items()
    }

    # 计算加速比：若存在 fixed 与 continuous 对比，计算 makespan 加速比
    speedup = 1.0
    if "fixed" in case_results and "continuous" in case_results:
        f_mean = case_results["fixed"].makespan_ms.mean
        c_mean = case_results["continuous"].makespan_ms.mean
        speedup = f_mean / c_mean if c_mean > 0 else 1.0

    return ServingSuiteResult(
        suite_name=suite.name,
        cases=case_results,
        continuous_makespan_speedup=speedup,
    )


def print_serving_table(suite_results: Sequence[ServingSuiteResult]) -> None:
    """打印控制台标准 CSV 汇总表格。"""
    header = [
        "suite",
        "case_id",
        "mode",
        "makespan_ms_mean",
        "request_throughput_per_second_mean",
        "output_token_throughput_per_second_mean",
        "execution_slot_utilization_mean",
        "ttft_ms_p95_mean",
        "itl_ms_p95_mean",
        "cache_management_fraction_mean",
        "speedup_vs_fixed",
    ]
    print(",".join(header))
    for suite_res in suite_results:
        for cid, res in suite_res.cases.items():
            speedup_str = f"{suite_res.continuous_makespan_speedup:.2f}x" if "continuous" in cid else "1.00x"
            row = [
                suite_res.suite_name,
                cid,
                res.mode,
                f"{res.makespan_ms.mean:.3f}",
                f"{res.request_throughput_req_s.mean:.3f}",
                f"{res.output_token_throughput_tok_s.mean:.3f}",
                f"{res.execution_slot_utilization.mean * 100:.2f}%",
                f"{res.ttft_ms_p95.mean:.3f}",
                f"{res.itl_ms_p95.mean:.3f}",
                f"{res.cache_management_fraction.mean * 100:.2f}%",
                speedup_str,
            ]
            print(",".join(row))


def build_serving_record(
    args: argparse.Namespace,
    model: torch.nn.Module,
    suite: ServingSuite,
    suite_result: ServingSuiteResult,
    correctness: dict[str, Any],
    env: dict[str, Any],
    peaks: dict[str, float | None],
    cost: dict[str, Any] | None,
    git: dict[str, Any],
) -> dict[str, Any]:
    """构建 Serving 评测结果标准 JSON 记录。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "tag": args.tag,
        "benchmark_type": "serving",
        "suite": suite_result.suite_name,
        "suite_config": suite.to_dict(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git,
        "env": env,
        "peak_specs": peaks,
        "model_cost": cost,
        "config": {
            "model": args.model,
            "revision": args.revision,
            "dtype": str(next(model.parameters()).dtype),
            "decoding": "greedy, forced output budget, EOS disabled",
            "tokenizer_included": False,
        },
        "correctness": correctness,
        "continuous_makespan_speedup": suite_result.continuous_makespan_speedup,
        "cases": {cid: c.to_record_dict() for cid, c in suite_result.cases.items()},
    }


def main() -> None:
    args = parse_args()
    suite = get_serving_suite(args.suite)

    # 支持 --dry-run：仅生成并打印计划，无需 GPU / 模型
    if args.dry_run:
        cases = suite.cases(vocab_size=1000)
        interleave_features = parse_interleave_args(args.interleave, suite.interleave)
        warmup = args.warmup if args.warmup is not None else suite.warmup
        repeats = args.repeats if args.repeats is not None else suite.repeats
        plan = build_serving_run_plan(cases, warmup=warmup, repeats=repeats, interleave=interleave_features)
        print(f"=== [Dry Run] Suite: {suite.name} ===")
        print(format_serving_run_plan(plan))
        return

    device = require_cuda()
    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    vocab = model.config.vocab_size

    # 正确性门禁：分别对 fixed 与 continuous 验证 golden 输出
    check_modes = [
        Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.FIXED),
        Mode({FeatureFlag.KV_CACHE}, batching_method=BatchingMethod.CONTINUOUS),
    ]
    correctness = {str(m): check_correctness(model, tokenizer, m, device, update_golden=False) for m in check_modes}
    if any(c["passed"] is False for c in correctness.values()) and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info(device)
    peaks = resolve_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = model_cost(model)
    git = git_info()

    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    interleave_features = parse_interleave_args(args.interleave, suite.interleave)

    print(
        f"[bench-serving-suite] suite={suite.name} max_running={suite.max_running_requests} "
        f"warmup={warmup} repeats={repeats} ...",
        flush=True,
    )

    suite_res = run_suite(
        model=model,
        device=device,
        suite=suite,
        vocab_size=vocab,
        warmup=warmup,
        repeats=repeats,
        interleave=interleave_features,
    )

    for case_id, res in suite_res.cases.items():
        cache_info = (
            f" | cache {res.cache_management_fraction.mean * 100:.1f}%"
            if res.cache_management_fraction.mean > 0
            else ""
        )
        speedup_info = f" | speedup {suite_res.continuous_makespan_speedup:.2f}x" if "continuous" in case_id else ""
        print(
            f"  {case_id:<12}: makespan {res.makespan_ms.mean:.2f} ms | "
            f"{res.output_token_throughput_tok_s.mean:.1f} tok/s | {res.request_throughput_req_s.mean:.2f} req/s | "
            f"slot_util {res.execution_slot_utilization.mean * 100:.1f}% | "
            f"ttft_p95 {res.ttft_ms_p95.mean:.2f} ms | itl_p95 {res.itl_ms_p95.mean:.2f} ms"
            f"{cache_info}{speedup_info}"
        )

    record = build_serving_record(
        args=args,
        model=model,
        suite=suite,
        suite_result=suite_res,
        correctness=correctness,
        env=env,
        peaks=peaks,
        cost=cost,
        git=git,
    )

    serving_dir = BENCH_ARTIFACTS_DIR / "suites" / "serving"
    output_path = args.output or (serving_dir / args.suite / f"{git['commit']}_{args.tag}.json")
    save_benchmark_record(output_path, record, dirty=git["dirty"])

    print()
    print_serving_table([suite_res])


__all__ = [
    "build_serving_record",
    "main",
    "parse_args",
    "print_serving_table",
    "run_suite",
]

if __name__ == "__main__":
    main()
