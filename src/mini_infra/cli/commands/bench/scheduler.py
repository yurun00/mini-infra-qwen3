"""CLI 子命令：mini-infra bench scheduler case

队列调度器（Scheduler）单用例评测执行入口。套件执行请使用 `mini-infra bench sweep --engine scheduler`。
"""

import argparse
import sys
from dataclasses import replace

from mini_infra.bench.cases import BenchmarkCase, WorkloadProfile
from mini_infra.bench.correctness import check_correctness
from mini_infra.bench.efficiency import HardwarePeakSpecs, ModelCostProfile
from mini_infra.bench.env import git_info, gpu_info, require_cuda
from mini_infra.bench.paths import CASE_DIR
from mini_infra.bench.result import build_benchmark_record, save_benchmark_record
from mini_infra.bench.scheduler.format import format_table
from mini_infra.bench.scheduler.run_case import run_case
from mini_infra.bench.spec.common.lengths import make_arrival_times
from mini_infra.cli.commands.bench.args import add_common_args
from mini_infra.core import (
    BATCHING_POLICY_KINDS,
    DenseCacheConfig,
    PagedCacheConfig,
    PrefillFirstConfig,
    kv_cache_config_from_str,
    scheduler_config_from_str,
)
from mini_infra.engine import BatchingMethod, EngineConfig, Qwen3Tokenizer, validate_engine_config
from mini_infra.model import load_model, resolve_model_directory


def _add_scheduler_case_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 scheduler case 参数。"""
    add_common_args(p)
    p.add_argument(
        "--caching-method",
        choices=[DenseCacheConfig.kind, PagedCacheConfig.kind],
        default=DenseCacheConfig.kind,
        help=f"执行的 KV 缓存管理策略（默认: {DenseCacheConfig.kind}）",
    )
    p.add_argument(
        "--batching-method",
        choices=[BatchingMethod.CONTINUOUS.value, BatchingMethod.FIXED.value],
        default=BatchingMethod.CONTINUOUS.value,
        help=f"执行的批处理调度策略（默认: {BatchingMethod.CONTINUOUS.value}）",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Paged KV Cache 的 physical block 大小（默认: 16，仅 paged-kv-cache 时生效）",
    )
    p.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Paged KV Cache 的 physical block pool 总块数（可选，默认按负载自动推导，仅 paged-kv-cache 时生效）",
    )
    p.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="调度队列最大运行并发数（默认: 4）",
    )
    p.add_argument(
        "--batching-policy",
        choices=list(BATCHING_POLICY_KINDS),
        default=PrefillFirstConfig.kind,
        help="迭代级单步组批策略，token-budget 仅支持 continuous-batching（默认: %(default)s）",
    )
    p.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="token-budget 策略下单次 Prefill 的 padded prompt token 上限，须不小于 --max-running-requests",
    )
    p.add_argument("--prompt-length", type=int, default=128, help="统一输入 Prompt 长度（默认: 128）")
    p.add_argument("--prompt-lens", type=int, nargs="+", default=None, help="逐请求输入 Prompt 长度列表（可选）")
    p.add_argument("--request-count", type=int, default=8, help="请求总数（默认: 8）")
    p.add_argument("--output-budget", type=int, default=20, help="统一输出 Token 预算（默认: 20）")
    p.add_argument("--output-budgets", type=int, nargs="+", default=None, help="逐请求输出 Token 预算列表（可选）")
    p.add_argument("--arrival-interval-ms", type=float, default=0.0, help="错峰到达时间间隔 ms（默认: 0.0）")
    p.add_argument("--arrival-times-ms", type=float, nargs="+", default=None, help="逐请求到达时间列表 ms（可选）")
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 scheduler 组及 case 子命令。"""
    scheduler_parser = subparsers.add_parser("scheduler", help="队列调度器（Scheduler: Fixed vs Continuous）基准测试")
    scheduler_subs = scheduler_parser.add_subparsers(dest="scheduler_subcommand", required=True)

    case_parser = scheduler_subs.add_parser("case", help="运行单用例 Scheduler 基准测试")
    _add_scheduler_case_args(case_parser)
    case_parser.set_defaults(func=handle_case)
    return scheduler_parser


def handle_case(args: argparse.Namespace) -> int:
    device = require_cuda()

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)

    method = BatchingMethod(args.batching_method)
    max_running = args.max_running_requests or 4
    kv_cache_config = kv_cache_config_from_str(
        args.caching_method,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
    )
    scheduler_config = scheduler_config_from_str(args.batching_policy, token_budget=args.token_budget)
    target_config = EngineConfig(
        kv_cache_config,
        batching_method=method,
        max_running_requests=max_running,
        scheduler_config=scheduler_config,
    )
    validate_engine_config(target_config)

    correctness = check_correctness(
        model,
        tokenizer,
        target_config,
        device,
        update_golden=False,
    )
    if correctness.passed is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    req_count = len(args.prompt_lens) if args.prompt_lens is not None else args.request_count
    prompt_len: int | tuple[int, ...] = tuple(args.prompt_lens) if args.prompt_lens is not None else args.prompt_length
    max_new_tokens: int | tuple[int, ...] = (
        tuple(args.output_budgets) if args.output_budgets is not None else args.output_budget
    )
    arrival_ms = (
        tuple(args.arrival_times_ms)
        if args.arrival_times_ms is not None
        else tuple(make_arrival_times(req_count, args.arrival_interval_ms))
    )
    warmup = args.warmup if args.warmup is not None else 1
    repeats = args.repeats if args.repeats is not None else 3

    prompt_lens = (prompt_len,) * req_count if isinstance(prompt_len, int) else tuple(prompt_len)
    output_budgets = (max_new_tokens,) * req_count if isinstance(max_new_tokens, int) else tuple(max_new_tokens)
    workload_profile = WorkloadProfile(
        workload_id="custom",
        num_total_requests=req_count,
        prompt_lens=prompt_lens,
        output_budgets=output_budgets,
        arrival_ms=arrival_ms,
    )
    case_name = "custom"

    case = BenchmarkCase(
        profile=workload_profile,
        batching_method=method,
        kv_cache_config=kv_cache_config,
        max_running_requests=max_running,
        scheduler_config=scheduler_config,
        suite_name=case_name,
        engine="scheduler",
    )

    block_info = f" block_size={kv_cache_config.block_size}" if isinstance(kv_cache_config, PagedCacheConfig) else ""
    print(
        f"[bench-scheduler-case] case={case_name} mode={target_config}{block_info} "
        f"requests={workload_profile.num_total_requests} "
        f"max_running={max_running} warmup={warmup} repeats={repeats} ...",
        flush=True,
    )

    def on_case_step(phase: str, step_number: int, total_steps: int) -> None:
        count = workload_profile.num_total_requests
        print(
            f"  [{phase} {step_number}/{total_steps}] mode={target_config} requests={count} ...",
            flush=True,
        )

    result = run_case(
        model=model,
        device=device,
        case=case,
        warmup=warmup,
        repeats=repeats,
        on_step_progress=on_case_step,
    )
    if getattr(args, "scale", None) is not None or getattr(args, "seed", None) is not None:
        result = replace(
            result,
            scale=getattr(args, "scale", None),
            seed=getattr(args, "seed", None),
        )
    print(result.format_summary())

    env = gpu_info(device)
    peaks = HardwarePeakSpecs.lookup(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = ModelCostProfile.from_model(model)
    git = git_info()
    record = build_benchmark_record(
        args=args,
        model=model,
        engine_config_label=str(target_config),
        cases=[result],
        correctness=correctness,
        env=env,
        peaks=peaks,
        cost=cost,
        git=git,
        warmup=warmup,
        repeats=repeats,
        engine="scheduler",
    )
    output = args.output or CASE_DIR / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, is_dirty=bool(git["dirty"]))

    print()
    print(format_table([result], suite_name=case_name))
    return 0
