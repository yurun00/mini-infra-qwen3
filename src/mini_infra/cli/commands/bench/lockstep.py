"""CLI 子命令：mini-infra bench lockstep case

齐步（Lockstep：所有请求同时开始，无调度器）单用例基准测试执行入口。
套件执行请使用 `mini-infra bench sweep --engine lockstep`。
"""

import argparse
import sys
from dataclasses import replace

from mini_infra.bench.cases import BenchmarkCase, WorkloadProfile
from mini_infra.bench.correctness import check_correctness
from mini_infra.bench.efficiency import HardwarePeakSpecs, ModelCostProfile
from mini_infra.bench.env import git_info, gpu_info, require_cuda
from mini_infra.bench.format import format_lengths
from mini_infra.bench.lockstep.format import format_table
from mini_infra.bench.lockstep.run_case import broadcast_case_lengths, run_case
from mini_infra.bench.paths import LOCKSTEP_RESULT_HISTORY_DIR
from mini_infra.bench.result import build_benchmark_record, save_benchmark_record
from mini_infra.cli.commands.bench.args import add_common_args
from mini_infra.core import DenseCacheConfig, NoCacheConfig
from mini_infra.engine import BatchingMethod, Qwen3Tokenizer
from mini_infra.model import load_model, resolve_model_directory
from mini_infra.profiling import TensorBoardOptions


def _add_lockstep_case_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向 parser 添加 lockstep case 参数。"""
    add_common_args(p)
    p.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 KV Cache 特性（默认: True）",
    )
    p.add_argument(
        "--batching-method",
        default=BatchingMethod.NONE.value,
        choices=[BatchingMethod.NONE.value, BatchingMethod.FIXED.value],
        help="指定齐步批处理策略（默认: none）",
    )
    p.add_argument(
        "--batch-size", type=int, default=None, help="批处理大小；未指定时自动由 prompt_lens/max_new_tokens 推断"
    )
    p.add_argument(
        "--prompt-lens",
        type=int,
        nargs="+",
        default=[128],
        help="Prompt 长度（1 项表示等长，多项表示逐请求长度，默认: 128）",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[64],
        help="最大新生成 Token 预算（1 项表示等长，多项表示逐请求长度，默认: 64）",
    )
    p.add_argument("--update-golden", action="store_true", help="用当前输出覆盖 golden（仅在 baseline 上使用）")
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 lockstep 组及 case 子命令。"""
    lockstep_parser = subparsers.add_parser("lockstep", help="齐步（Lockstep：所有请求同时开始，无调度器）推理基准测试")
    lockstep_subs = lockstep_parser.add_subparsers(dest="lockstep_subcommand", required=True)

    case_parser = lockstep_subs.add_parser("case", help="运行单用例齐步（Lockstep）基准测试")
    _add_lockstep_case_args(case_parser)
    case_parser.set_defaults(func=_handle_case)
    return lockstep_parser


def _handle_case(args: argparse.Namespace) -> int:
    device = require_cuda()

    target_bs, resolved_prompt_len, resolved_max_new_tokens = broadcast_case_lengths(
        args.prompt_lens, args.max_new_tokens, args.batch_size
    )

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)

    kv_cache_config = DenseCacheConfig() if args.kv_cache else NoCacheConfig()
    batching_method = BatchingMethod.from_str(args.batching_method)

    plens = (resolved_prompt_len,) * target_bs if isinstance(resolved_prompt_len, int) else resolved_prompt_len
    budgets = (
        (resolved_max_new_tokens,) * target_bs if isinstance(resolved_max_new_tokens, int) else resolved_max_new_tokens
    )
    profile = WorkloadProfile(
        workload_id="custom",
        num_total_requests=target_bs,
        prompt_lens=plens,
        output_budgets=budgets,
        arrival_ms=(0.0,) * target_bs,
    )
    case = BenchmarkCase(
        profile=profile,
        batching_method=batching_method,
        kv_cache_config=kv_cache_config,
        suite_name="custom",
        engine="lockstep",
    )

    correctness = check_correctness(model, tokenizer, case.engine_config, device, args.update_golden)
    if correctness.passed is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info()
    peaks = HardwarePeakSpecs.lookup(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = ModelCostProfile.from_model(model)
    if peaks.bandwidth_gbs is None or peaks.tflops is None:
        print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
    tb_options = TensorBoardOptions(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None

    print(
        f"[bench-case] mode={case.engine_config} batch={case.profile.num_total_requests} "
        f"prompt={format_lengths(case.profile.prompt_lens)} "
        f"max_new_tokens={format_lengths(case.profile.output_budgets)} ...",
        flush=True,
    )

    result = run_case(
        model=model,
        device=device,
        case=case,
        warmup=args.warmup,
        repeats=args.repeats,
        cost=cost,
        peaks=peaks,
        profiling_mode=args.profiling_mode,
        tensorboard_options=tb_options,
    )
    if getattr(args, "scale", None) is not None or getattr(args, "seed", None) is not None:
        result = replace(
            result,
            scale=getattr(args, "scale", None),
            seed=getattr(args, "seed", None),
        )

    print(result.format_summary())

    git = git_info()
    record = build_benchmark_record(
        args=args,
        model=model,
        engine_config_label=str(case.engine_config),
        cases=[result],
        correctness=correctness,
        env=env,
        peaks=peaks,
        cost=cost,
        git=git,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    output = args.output or LOCKSTEP_RESULT_HISTORY_DIR / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, is_dirty=bool(git["dirty"]))
    print()
    print(format_table([result]))
    return 0
