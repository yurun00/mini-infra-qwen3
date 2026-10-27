import argparse
from pathlib import Path

import torch

from mini_infra.engine import (
    BatchingMethod,
    FeatureFlag,
    Mode,
    generate,
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


def parse_args():
    parser = argparse.ArgumentParser(description="运行手写模型单请求/固定批处理推理与对照")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["用一句话解释 Prefill。", "用一句话解释 Decode。"],
        help="输入提示词列表（默认包含 Prefill 与 Decode 两条提示词）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
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
    return parser.parse_args()


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


def resolve_compare_features_string(mode: Mode, compare_features: list[str] | None) -> str:
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


def print_result(mode, result, tokenizer, tensorboard_dir=None):
    print("\n[%s]" % mode)
    for seq in result.sequences:
        print("请求 %d Token IDs: %s" % (seq.request_index, seq.token_ids))
        print("请求 %d 输出: %s" % (seq.request_index, tokenizer.decode(seq.token_ids, skip_special_tokens=True)))

    print("端到端 Wall-clock 延迟: %.2f ms" % result.metrics.e2e_wall_ms)
    print("端到端模型计算延迟: %.2f ms" % result.metrics.e2e_model_ms)
    print("请求吞吐: %.3f requests/s" % result.metrics.throughput_req_s)
    print("输出吞吐: %.3f tokens/s" % result.metrics.throughput_tok_s)

    # 细粒度执行与显存指标（包含 batch 及超集分位数分析）
    print("Prefill 延迟: %.2f ms" % result.metrics.ttft_ms)
    print("平均 Decode 迭代延迟: %.2f ms" % result.metrics.mean_tpot_ms)
    print("P50 Decode 延迟: %.2f ms" % result.metrics.p50_tpot_ms)
    print("P95 Decode 延迟: %.2f ms" % result.metrics.p95_tpot_ms)
    if mode.batching_method == BatchingMethod.FIXED and result.batch_info is not None:
        print("Padding 有效率: %.2f%%" % (result.batch_info.padding_efficiency * 100))
        print("槽位利用率: %.2f%%" % (result.batch_info.slot_utilization * 100))
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


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于 0")
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

    for mode in modes:
        for _ in range(args.warmup):
            generate(
                mode,
                model,
                input_ids,
                min(args.max_new_tokens, 2),
                tokenizer.eos_token_id,
                device,
            )
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

        results[mode] = generate(
            mode,
            model,
            input_ids,
            args.max_new_tokens,
            tokenizer.eos_token_id,
            device,
            profile=do_profile,
            top_kernels=args.top_kernels,
            tensorboard_options=tb_options,
        )
        print_result(mode, results[mode], tokenizer, tensorboard_dir=tb_dir)

    if len(modes) == 2:
        baseline = results[modes[0]]
        optimized = results[modes[1]]
        print(f"\n[特性 '{compare_feature_name}' 对照结论]")
        same_tokens = [s.token_ids for s in baseline.sequences] == [s.token_ids for s in optimized.sequences]
        print("逐请求 Token 一致:", same_tokens)
        baseline_decode = baseline.metrics.mean_tpot_ms
        optimized_decode = optimized.metrics.mean_tpot_ms
        if optimized_decode > 0:
            print("平均 Decode 加速比: %.2fx" % (baseline_decode / optimized_decode))
        if baseline.metrics.e2e_wall_ms > 0 and optimized.metrics.e2e_wall_ms > 0:
            print(
                "端到端 Wall-clock 耗时: %.2f ms -> %.2f ms"
                % (baseline.metrics.e2e_wall_ms, optimized.metrics.e2e_wall_ms)
            )
            print("Batch makespan 加速比: %.3fx" % (baseline.metrics.e2e_wall_ms / optimized.metrics.e2e_wall_ms))
    print("说明: 计时不包含 tokenizer、模型下载和模型加载；默认 greedy decoding。")


if __name__ == "__main__":
    main()
