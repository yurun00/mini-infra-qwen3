"""CLI 子命令：mini-infra run

手写模型单请求/批处理推理与特性对照。
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from mini_infra.core import make_request_specs
from mini_infra.engine import (
    BatchingMethod,
    FeatureFlag,
    GenerationOptions,
    GenerationResult,
    Mode,
    ScheduleResult,
    generate,
    run_scheduler,
    synchronize,
)
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Tokenizer,
    load_model,
    resolve_model_directory,
)
from mini_infra.profiling import TensorBoardOptions


def add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向指定 parser 添加 run 命令的所有参数。"""
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["用一句话解释 Prefill。", "用一句话解释 Decode。"],
        help="输入提示词列表（默认包含 Prefill 与 Decode 两条提示词）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[32],
        metavar="N",
        help="输出 Token 预算：提供 1 个值（所有请求共用）或与 prompts 数量一致的逐请求预算（默认: 32）",
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=None,
        help="调度器最大并发请求数，仅对 fixed/continuous 调度路径生效（默认: 不限，即 prompt 数量）",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 KV Cache 特性（默认: True）",
    )
    parser.add_argument(
        "--batching-method",
        default=BatchingMethod.NONE.value,
        choices=[m.value for m in BatchingMethod],
        help="指定基准批处理策略（默认: none）",
    )
    parser.add_argument(
        "--compare-feature",
        nargs="+",
        default=None,
        metavar="FEATURE",
        help="指定 1 个（与基准对比）或 2 个 feature（相互对比，如 fixed-batching continuous-batching）",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否启用底层 PyTorch Profiler 采集算子与 Kernel 耗时明细（默认: False）",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=10,
        help="展示耗时前 K 个的 CUDA Kernel（默认: 10）",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="导出 TensorBoard 格式性能剖析日志的目录（需配合 --profile 使用）",
    )
    return parser


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在根/上级 subparsers 中注册 run 子命令。"""
    parser = subparsers.add_parser("run", help="运行手写模型单请求/批处理推理与特性对照")
    add_run_args(parser)
    parser.set_defaults(func=handle_run)
    return parser


def resolve_feature(name: str) -> FeatureFlag | BatchingMethod:
    """将名称解析为 FeatureFlag 或作为 feature 之一的 BatchingMethod（兼容下划线与常见简写）。"""
    normalized = name.strip().replace("_", "-").lower()
    try:
        return FeatureFlag.from_str(normalized)
    except ValueError:
        pass
    try:
        return BatchingMethod.from_str(normalized)
    except ValueError:
        pass
    if normalized == "fixed":
        return BatchingMethod.FIXED
    if normalized == "continuous":
        return BatchingMethod.CONTINUOUS
    if normalized == "none":
        return BatchingMethod.NONE

    valid_features = [f.value for f in FeatureFlag] + [b.value for b in BatchingMethod]
    raise ValueError(f"--compare-feature 参数 '{name}' 未知。可选 feature: {valid_features}")


def resolve_compare_features_string(mode: Mode, compare_features: Sequence[str] | None) -> str:
    """根据 compare_feature 从 mode 中查找对应值并拼接为字符串。"""
    if not compare_features:
        return ""
    features: list[str] = []
    for f in compare_features:
        feat = resolve_feature(f)
        val = (
            mode.batching_method.value
            if isinstance(feat, BatchingMethod)
            else (feat.value if mode.is_enabled(feat) else f"no-{feat.value}")
        )
        if val not in features:
            features.append(val)
    return "_".join(features)


def resolve_budgets(max_new_tokens: Sequence[int], num_prompts: int) -> int | list[int]:
    """将 --max-new-tokens 归一化：1 个值广播为 int，与 prompts 等长则逐请求预算，否则报错。"""
    budgets = list(max_new_tokens)
    if any(b < 1 for b in budgets):
        raise ValueError("--max-new-tokens 必须大于 0")
    if len(budgets) == 1:
        return budgets[0]
    if len(budgets) == num_prompts:
        return budgets
    raise ValueError(
        f"--max-new-tokens 应提供 1 个值或与 prompts 数量一致（{num_prompts}），当前传入 {len(budgets)} 个"
    )


def shrink_budgets(budgets: int | list[int], limit: int) -> int | list[int]:
    """将预算截断到不超过 limit（用于 warmup）。"""
    if isinstance(budgets, int):
        return min(budgets, limit)
    return [min(b, limit) for b in budgets]


def uses_scheduler(modes: Sequence[Mode]) -> bool:
    """只要任一 mode 使用 continuous batching，所有结果都统一由 run_scheduler 产出（指标可比）。"""
    return any(m.batching_method is BatchingMethod.CONTINUOUS for m in modes)


def run_mode(
    mode: Mode,
    model: torch.nn.Module,
    input_ids,
    sequences: Sequence[Sequence[int]],
    budgets: int | list[int],
    options: GenerationOptions,
    via_scheduler: bool,
) -> GenerationResult:
    """执行单个 mode：via_scheduler 为 True 时统一由 run_scheduler 产出；否则沿用 generate。"""
    if via_scheduler:
        if not mode.is_enabled(FeatureFlag.KV_CACHE):
            raise ValueError("调度路径依赖 KV Cache，请启用 --kv-cache")
        if options.capture_logits or options.profile:
            raise ValueError("调度路径暂不支持 capture_logits/profile，请去掉 --profile 或 continuous 对比")
        specs = make_request_specs(sequences, budgets)
        concurrency = options.max_running_requests if options.max_running_requests is not None else len(specs)
        if concurrency < 1:
            raise ValueError("max_running_requests 必须大于 0")
        return run_scheduler(
            mode.batching_method,
            model,
            specs,
            concurrency,
            options.eos_token_id,
            options.device or torch.device("cuda"),
            pad_token_id=options.pad_token_id,
            stop_on_eos=options.stop_on_eos,
        )
    return generate(mode, model, input_ids, budgets, options=options)


def print_schedule_metrics(result: ScheduleResult) -> None:
    sm = result.schedule_metrics
    print("Makespan: %.2f ms" % sm.makespan_ms)
    print("排队延迟 p50/p95: %.2f / %.2f ms" % (sm.queue_ms_p50, sm.queue_ms_p95))
    print("TTFT p50/p95: %.2f / %.2f ms" % (sm.ttft_ms_p50, sm.ttft_ms_p95))
    print("ITL p50/p95/max: %.2f / %.2f / %.2f ms" % (sm.itl_ms_p50, sm.itl_ms_p95, sm.itl_ms_max))
    print("Cache 管理耗时占比: %.2f%%" % (sm.cache_management_fraction * 100))
    info = result.batch_info
    if info is not None and info.running_capacity_utilization is not None:
        print("并发容量利用率: %.2f%%" % (info.running_capacity_utilization * 100))


def print_result(mode: Mode, result, tokenizer: Qwen3Tokenizer, tensorboard_dir: Path | None = None) -> None:
    print("\n[%s]" % mode)
    for seq in result.sequences:
        print("请求 %d Token IDs: %s" % (seq.request_index, seq.token_ids))
        print("请求 %d 输出: %s" % (seq.request_index, tokenizer.decode(seq.token_ids, skip_special_tokens=True)))

    print("端到端 Wall-clock 延迟: %.2f ms" % result.metrics.e2e_wall_ms)
    print("端到端模型计算延迟: %.2f ms" % result.metrics.e2e_model_ms)
    print("请求吞吐: %.3f requests/s" % result.metrics.throughput_req_s)
    print("输出吞吐: %.3f tokens/s" % result.metrics.throughput_tok_s)

    # 细粒度执行与显存指标（包含 batch 及超集分位数分析）
    if result.metrics.ttft_ms is not None:
        print("Prefill 延迟: %.2f ms" % result.metrics.ttft_ms)
    if result.metrics.mean_tpot_ms is not None:
        print("平均 Decode 迭代延迟: %.2f ms" % result.metrics.mean_tpot_ms)
    if result.metrics.p50_tpot_ms is not None:
        print("P50 Decode 延迟: %.2f ms" % result.metrics.p50_tpot_ms)
    if result.metrics.p95_tpot_ms is not None:
        print("P95 Decode 延迟: %.2f ms" % result.metrics.p95_tpot_ms)
    if mode.is_batch and result.batch_info is not None:
        print("Padding 有效率: %.2f%%" % (result.batch_info.padding_efficiency * 100))
        print("槽位利用率: %.2f%%" % (result.batch_info.slot_utilization * 100))
    if isinstance(result, ScheduleResult):
        print_schedule_metrics(result)
    print("峰值显存: %.2f MiB" % result.memory.peak_allocated_mb)
    print("最终 KV Cache: %.2f MiB" % result.memory.kv_cache_mb)
    decode_times = result.trace.step_latencies_ms[1:]
    if decode_times:
        print("逐步 Decode 延迟(ms):", [round(value, 2) for value in decode_times])

    if result.op_breakdown:
        breakdown = result.op_breakdown
        print("\n--- [%s] 算子与模块耗时分析 (Profiler) ---" % mode)
        if breakdown.time_share_by_module:
            print("各模块耗时占比 (Module Scope):")
            for mod, share in breakdown.time_share_by_module.items():
                print("  %-20s: %6.2f%%" % (mod, share * 100))
        if breakdown.time_share_by_kernel_category:
            print("算子类型耗时占比 (Kernel Category):")
            for cat, share in breakdown.time_share_by_kernel_category.items():
                print("  %-20s: %6.2f%%" % (cat, share * 100))
        top_k = breakdown.top_kernels
        if top_k:
            print("Top-%d CUDA Kernels:" % len(top_k))
            for idx, k in enumerate(top_k, 1):
                print(
                    "  %2d. %-45s | %8.3f ms | %6.2f%% | %4d 次" % (idx, k.name[:45], k.cuda_ms, k.share * 100, k.calls)
                )
    if tensorboard_dir is not None:
        print("TensorBoard Trace 目录: %s" % tensorboard_dir)


def handle_run(args: argparse.Namespace) -> int:
    """执行 run 命令。"""
    if args.max_running_requests is not None and args.max_running_requests < 1:
        raise ValueError("--max-running-requests 必须大于 0")
    if args.warmup < 0:
        raise ValueError("--warmup 不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)

    prompts = args.prompts
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts]
    if len(sequences) == 1:
        input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
    else:
        input_ids = sequences

    budgets = resolve_budgets(args.max_new_tokens, len(sequences))
    model = load_model(model_directory, device)

    flags = {FeatureFlag.KV_CACHE} if args.kv_cache else set()
    base_mode = Mode(flags, batching_method=args.batching_method)

    compare_feature_name = None
    if args.compare_feature:
        items = [resolve_feature(x) for x in args.compare_feature]
        if len(items) == 1:
            item = items[0]
            if isinstance(item, FeatureFlag):
                mode_disable = base_mode.with_flag(item, False)
                mode_enable = base_mode.with_flag(item, True)
                compare_feature_name = item.value
            else:
                if item == BatchingMethod.NONE:
                    raise ValueError("不能将 'none' 作为单个对比 feature")
                mode_disable = base_mode.with_batching_method(BatchingMethod.NONE)
                mode_enable = base_mode.with_batching_method(item)
                compare_feature_name = item.value
            modes = (mode_disable, mode_enable)
        elif len(items) == 2:
            item1, item2 = items[0], items[1]
            if item1 == item2:
                raise ValueError(f"--compare-feature 传入了两个相同的对比 feature: '{item1.value}'")

            def apply_feature(mode: Mode, feature: FeatureFlag | BatchingMethod) -> Mode:
                if isinstance(feature, FeatureFlag):
                    return mode.with_flag(feature, True)
                return mode.with_batching_method(feature)

            mode1 = apply_feature(base_mode, item1)
            mode2 = apply_feature(base_mode, item2)
            compare_feature_name = f"{item1.value} vs {item2.value}"
            modes = (mode1, mode2)
        else:
            raise ValueError(
                f"--compare-feature 最多支持传入 2 个 feature，当前传入了 {len(items)} 个: {args.compare_feature}"
            )
    else:
        modes = (base_mode,)

    via_scheduler = uses_scheduler(modes)
    if args.max_running_requests is not None and not via_scheduler:
        print("[warn] --max-running-requests 仅对调度路径（含 continuous-batching）生效，当前对比未启用，将被忽略。")

    warmup_opts = GenerationOptions(
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        max_running_requests=args.max_running_requests,
    )
    for mode in modes:
        for _ in range(args.warmup):
            run_mode(mode, model, input_ids, sequences, shrink_budgets(budgets, 2), warmup_opts, via_scheduler)
        synchronize(device)

    if isinstance(input_ids, torch.Tensor):
        print(f"Prompt 数量: {input_ids.shape[0]}，Token 数: {input_ids.shape[1]}")
    else:
        prompt_lens = [len(s) for s in sequences]
        print(f"Prompt 数量: {len(sequences)}，各请求 Token 数: {prompt_lens}")

    if args.tensorboard_dir is not None and not args.profile:
        print("[warn] 指定了 --tensorboard-dir 但未开启 --profile，将不会采集并导出 TensorBoard 日志。")

    do_profile = args.profile
    results = {}
    for mode in modes:
        if do_profile and args.tensorboard_dir is not None:
            compare_features_str = resolve_compare_features_string(mode, args.compare_feature)
            tb_dir = args.tensorboard_dir / compare_features_str if compare_features_str else args.tensorboard_dir
            tb_options = TensorBoardOptions(log_dir=tb_dir)
        else:
            tb_dir = None
            tb_options = None

        run_opts = GenerationOptions(
            eos_token_id=tokenizer.eos_token_id,
            device=device,
            profile=do_profile,
            top_kernels=args.top_kernels,
            tensorboard_options=tb_options,
            max_running_requests=args.max_running_requests,
        )
        results[mode] = run_mode(mode, model, input_ids, sequences, budgets, run_opts, via_scheduler)
        print_result(mode, results[mode], tokenizer, tensorboard_dir=tb_dir)

    if len(modes) == 2:
        baseline = results[modes[0]]
        optimized = results[modes[1]]
        print(f"\n[特性 '{compare_feature_name}' 对照结论]")
        same_tokens = [s.token_ids for s in baseline.sequences] == [s.token_ids for s in optimized.sequences]
        print("逐请求 Token 一致:", same_tokens)
        baseline_decode = baseline.metrics.mean_tpot_ms
        optimized_decode = optimized.metrics.mean_tpot_ms
        if baseline_decode is not None and optimized_decode is not None and optimized_decode > 0:
            print("平均 Decode 加速比: %.2fx" % (baseline_decode / optimized_decode))
        if baseline.metrics.e2e_wall_ms > 0 and optimized.metrics.e2e_wall_ms > 0:
            print(
                "端到端 Wall-clock 耗时: %.2f ms -> %.2f ms"
                % (baseline.metrics.e2e_wall_ms, optimized.metrics.e2e_wall_ms)
            )
            print("Batch makespan 加速比: %.3fx" % (baseline.metrics.e2e_wall_ms / optimized.metrics.e2e_wall_ms))
        if isinstance(baseline, ScheduleResult) and isinstance(optimized, ScheduleResult):
            bm, om = baseline.schedule_metrics, optimized.schedule_metrics
            if bm.makespan_ms > 0 and om.makespan_ms > 0:
                print(
                    "调度 Makespan: %.2f ms -> %.2f ms（加速比 %.3fx）"
                    % (bm.makespan_ms, om.makespan_ms, bm.makespan_ms / om.makespan_ms)
                )
            print("TTFT p95: %.2f ms -> %.2f ms" % (bm.ttft_ms_p95, om.ttft_ms_p95))
    print("说明: 计时不包含 tokenizer、模型下载、模型加载和真实睡眠；默认 greedy decoding。")
    return 0
