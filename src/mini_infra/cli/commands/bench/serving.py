"""CLI 子命令：mini-infra bench serving (case / suite)

Serving 评测套件与用例执行入口。
"""

import argparse
import sys
import time
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
from mini_infra.bench.serving.plan import build_serving_run_plan, format_serving_run_plan
from mini_infra.bench.serving.run_case import print_single_case_table
from mini_infra.bench.serving.run_case import run_case as run_serving_case
from mini_infra.bench.serving.run_suite import (
    build_serving_record,
    parse_interleave_args,
    print_serving_table,
)
from mini_infra.bench.serving.run_suite import (
    run_suite as run_serving_suite,
)
from mini_infra.bench.serving.suites import (
    SERVING_SUITES,
    ServingSuiteCase,
    get_serving_suite,
)
from mini_infra.bench.workloads import make_arrival_times, make_sequences
from mini_infra.core.request import make_request_specs
from mini_infra.engine import BatchingMethod, FeatureFlag, Mode
from mini_infra.model import Qwen3Tokenizer, load_model, resolve_model_directory


def add_serving_case_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 serving case 参数。"""
    add_common_args(p)
    p.add_argument(
        "--batching-method",
        choices=[BatchingMethod.CONTINUOUS.value, BatchingMethod.FIXED.value],
        default=BatchingMethod.CONTINUOUS.value,
        help=f"执行的批处理调度策略（默认: {BatchingMethod.CONTINUOUS.value}）",
    )
    p.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="调度队列最大运行并发数（默认: 4）",
    )
    p.add_argument("--prompt-length", type=int, default=128, help="统一输入 Prompt 长度（默认: 128）")
    p.add_argument("--request-count", type=int, default=8, help="请求总数（默认: 8）")
    p.add_argument("--output-budget", type=int, default=20, help="输出 Token 预算（默认: 20）")
    p.add_argument("--arrival-interval-ms", type=float, default=0.0, help="错峰到达时间间隔 ms（默认: 0.0）")
    return p


def add_serving_suite_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 serving suite 参数。"""
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
    return p


def build_case_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行单用例 Serving 基准测试")
    add_serving_case_args(p)
    return p


def build_suite_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="运行预设 Serving 套件")
    add_serving_suite_args(p)
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 serving 组及 case/suite 子命令。"""
    serving_parser = subparsers.add_parser("serving", help="在线并发 Serving 调度基准测试")
    serving_subs = serving_parser.add_subparsers(dest="serving_subcommand", required=True)

    case_parser = serving_subs.add_parser("case", help="运行单用例 Serving 基准测试")
    add_serving_case_args(case_parser)
    case_parser.set_defaults(func=handle_case)

    suite_parser = serving_subs.add_parser("suite", help="运行预设 Serving 套件")
    add_serving_suite_args(suite_parser)
    suite_parser.set_defaults(func=handle_suite)
    return serving_parser


def handle_case(args: argparse.Namespace) -> int:
    device = require_cuda()

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    vocab = model.config.vocab_size

    method = BatchingMethod(args.batching_method)
    target_mode = Mode({FeatureFlag.KV_CACHE}, batching_method=method)

    correctness = check_correctness(model, tokenizer, target_mode, device, update_golden=False)
    if correctness["passed"] is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    lengths = [args.prompt_length] * args.request_count
    sequences = [t.tolist() for t in make_sequences(lengths, vocab, device=device)]
    budgets = [args.output_budget] * args.request_count
    arrivals = make_arrival_times(args.request_count, args.arrival_interval_ms)
    specs = tuple(make_request_specs(sequences, budgets, arrivals))
    max_running = args.max_running_requests or 4
    warmup = args.warmup if args.warmup is not None else 1
    repeats = args.repeats if args.repeats is not None else 3
    case_name = "custom"

    case_variant_id = "continuous" if method == BatchingMethod.CONTINUOUS else "fixed"
    case = ServingSuiteCase(
        id=case_variant_id,
        suite_name=case_name,
        mode=target_mode,
        max_running_requests=max_running,
        specs=specs,
    )

    print(
        f"[bench-serving-case] case={case_name} mode={target_mode} requests={len(specs)} "
        f"max_running={max_running} warmup={warmup} repeats={repeats} ...",
        flush=True,
    )

    res = run_serving_case(
        model=model,
        device=device,
        case=case,
        warmup=warmup,
        repeats=repeats,
    )
    print_single_case_table(case_name, res)

    if args.output:
        record: dict[str, Any] = {
            "schema_version": "1.0",
            "tag": args.tag,
            "benchmark_type": "serving-case",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git": git_info(),
            "env": gpu_info(device),
            "peak_specs": resolve_peak_specs(gpu_info(device)["gpu"], args.peak_bandwidth_gbs, args.peak_tflops),
            "model_cost": model_cost(model),
            "case": case.to_spec_dict(),
            "warmup": warmup,
            "repeats": repeats,
            "result": res.to_record_dict(),
        }
        save_benchmark_record(args.output, record, dirty=git_info()["dirty"])
    return 0


def handle_suite(args: argparse.Namespace) -> int:
    suite = get_serving_suite(args.suite)

    if args.dry_run:
        warmup = args.warmup if args.warmup is not None else suite.warmup
        repeats = args.repeats if args.repeats is not None else suite.repeats
        interleave_features = parse_interleave_args(args.interleave, suite.interleave)
        cases = suite.cases(vocab_size=100, device=torch.device("cpu"))
        plan = build_serving_run_plan(cases, warmup=warmup, repeats=repeats, interleave=interleave_features)
        print(f"=== [Dry Run] Suite: {args.suite} ===")
        print(format_serving_run_plan(plan))
        return 0

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

    suite_res = run_serving_suite(
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
    return 0
