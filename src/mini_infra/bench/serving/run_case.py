"""单用例 Serving 基准测试（run_case）：一次调用只执行正好一个 Case。

支持单独评测某一特定策略与模式下的 Serving 用例。
既可从已注册的 ServingSuite 提取负载规格，也可通过命令行指定请求参数。

用法示例：
    # 单独评测 fixed-vs-continuous-variable-output-length 套件的 continuous 策略
    mini-infra-bench-serving-case --tag dev --suite fixed-vs-continuous-variable-output-length \
        --batching-method continuous-batching

    # 单独评测自定义负载下的 fixed 批处理
    mini-infra-bench-serving-case --tag dev --batching-method fixed-batching --max-running-requests 4 \
        --request-count 8 --prompt-length 128 --output-budget 32
"""

import argparse
import sys
import time
from typing import Any

import torch

from mini_infra.bench.common import (
    add_common_args,
    check_correctness,
    git_info,
    gpu_info,
    model_cost,
    require_cuda,
    resolve_peak_specs,
    save_benchmark_record,
)
from mini_infra.bench.serving.suites import (
    SERVING_SUITES,
    ServingSuiteCase,
    get_serving_suite,
)
from mini_infra.bench.serving.types import (
    ServingCaseResult,
    summarize_serving_runs,
)
from mini_infra.bench.workloads import make_arrival_times, make_sequences
from mini_infra.core.request import make_request_specs
from mini_infra.core.types import ScheduleResult
from mini_infra.engine import BatchingMethod, FeatureFlag, Mode, run_scheduler
from mini_infra.model import Qwen3Tokenizer, load_model, resolve_model_directory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-infra-qwen3 serving single-case benchmark runner")
    add_common_args(p)
    p.add_argument(
        "--suite",
        choices=sorted(SERVING_SUITES),
        default=None,
        help="可选：使用已注册 ServingSuite 的负载规格（如 fixed-vs-continuous-variable-output-length）",
    )
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
        help="调度队列最大运行并发数（默认: suite 指定或 4）",
    )
    p.add_argument("--prompt-length", type=int, default=128, help="统一输入 Prompt 长度（默认: 128）")
    p.add_argument("--request-count", type=int, default=8, help="请求总数（默认: 8）")
    p.add_argument("--output-budget", type=int, default=20, help="输出 Token 预算（默认: 20）")
    p.add_argument("--arrival-interval-ms", type=float, default=0.0, help="错峰到达时间间隔 ms（默认: 0.0）")
    return p.parse_args()


def run_case_once(
    model: torch.nn.Module,
    device: torch.device,
    case: ServingSuiteCase,
) -> ScheduleResult:
    """执行单一 ServingSuiteCase 的单次调度运行（原子操作）。"""
    return run_scheduler(
        batching_method=case.mode.batching_method,
        model=model,
        request_specs=case.specs,
        max_running_requests=case.max_running_requests,
        eos_token_id=-1,
        device=device,
        stop_on_eos=False,
    )


def run_case(
    model: torch.nn.Module,
    device: torch.device,
    case: ServingSuiteCase,
    warmup: int = 1,
    repeats: int = 3,
) -> ServingCaseResult:
    """连续执行单个 Case（warmup + repeats），汇总并返回 ServingCaseResult。"""
    for _ in range(warmup):
        run_case_once(model, device, case)

    runs = [run_case_once(model, device, case) for _ in range(repeats)]
    return summarize_serving_runs(case.id, str(case.mode), runs)


def print_single_case_table(case_name: str, res: ServingCaseResult) -> None:
    """控制台打印单一 Serving 用例的指标结果。"""
    print(
        "  [%s | %s]: makespan %.2f ms | %.1f tok/s | %.2f req/s | "
        "slot_util %.1f%% | ttft_p95 %.2f ms | itl_p95 %.2f ms | cache %.1f%%"
        % (
            case_name,
            res.case_id,
            res.makespan_ms.mean,
            res.output_token_throughput_tok_s.mean,
            res.request_throughput_req_s.mean,
            res.execution_slot_utilization.mean * 100,
            res.ttft_ms_p95.mean,
            res.itl_ms_p95.mean,
            res.cache_management_fraction.mean * 100,
        )
    )


def main() -> None:
    args = parse_args()
    device = require_cuda()

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    vocab = model.config.vocab_size

    # 正确性检查
    method = BatchingMethod.from_str(args.batching_method)
    target_mode = Mode({FeatureFlag.KV_CACHE}, batching_method=method)
    corr = check_correctness(model, tokenizer, target_mode, device, update_golden=False)
    if corr.get("passed") is False and not args.allow_mismatch:
        sys.exit(f"正确性检查失败：{args.batching_method} 输出与 golden 不一致（--allow-mismatch 可忽略）")

    # 构建 ServingSuiteCase
    if args.suite is not None:
        suite = get_serving_suite(args.suite)
        specs = tuple(suite.build_specs(vocab, device=device))
        max_running = args.max_running_requests or suite.max_running_requests
        warmup = args.warmup if args.warmup is not None else suite.warmup
        repeats = args.repeats if args.repeats is not None else suite.repeats
        case_name = suite.name
    else:
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

    res = run_case(
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


__all__ = [
    "main",
    "parse_args",
    "print_single_case_table",
    "run_case",
    "run_case_once",
]

if __name__ == "__main__":
    main()
