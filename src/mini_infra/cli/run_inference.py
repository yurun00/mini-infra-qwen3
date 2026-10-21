import argparse

import torch

from mini_infra.generation import (
    BatchingMethod,
    FeatureFlag,
    Mode,
    generate,
    synchronize,
)
from mini_infra.qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from mini_infra.qwen3_tokenizer import Qwen3Tokenizer


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


def print_result(mode, result, tokenizer):
    print("\n[%s]" % mode)
    for index, token_ids in enumerate(result.unpadded_output_token_ids):
        print("请求 %d Token IDs: %s" % (index, token_ids))
        print("请求 %d 输出: %s" % (index, tokenizer.decode(token_ids, skip_special_tokens=True)))

    print("端到端 Wall-clock 延迟: %.2f ms" % result.e2e_wall_ms)
    print("端到端模型计算延迟: %.2f ms" % result.e2e_model_ms)
    print("请求吞吐: %.3f requests/s" % result.request_throughput_per_second)
    print("输出吞吐: %.3f tokens/s" % result.throughput_tok_s)

    # 细粒度执行与显存指标（包含 batch 及超集分位数分析）
    print("Prefill 延迟: %.2f ms" % result.ttft_ms)
    print("平均 Decode 迭代延迟: %.2f ms" % result.tpot_ms)
    print("P50 Decode 延迟: %.2f ms" % result.p50_decode_step_ms)
    print("P95 Decode 延迟: %.2f ms" % result.p95_decode_step_ms)
    if mode.batching_method == BatchingMethod.FIXED:
        if "padding_efficiency" in result:
            print("Padding 有效率: %.2f%%" % (result["padding_efficiency"] * 100))
        if "slot_utilization" in result:
            print("槽位利用率: %.2f%%" % (result["slot_utilization"] * 100))
    print("峰值显存: %.2f MiB" % result.memory["peak_allocated_mb"])
    print("最终 KV Cache: %.2f MiB" % result.memory["kv_cache_mb"])
    decode_times = result.step_latencies_ms[1:]
    if decode_times:
        print("逐步 Decode 延迟(ms):", [round(value, 2) for value in decode_times])


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

    model = load_handwritten_model(model_directory, device)

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
                mode_disable = base_mode.with_batching(BatchingMethod.NONE)
                mode_enable = base_mode.with_batching(item)
                compare_feature_name = item.value
            modes = (mode_disable, mode_enable)
        elif len(items) == 2:
            item1, item2 = items[0], items[1]
            if item1 == item2:
                raise ValueError(f"--compare-feature 传入了两个相同的对比 feature: '{item1.value}'")

            def apply_feature(mode: Mode, feature: FeatureFlag | BatchingMethod) -> Mode:
                if isinstance(feature, FeatureFlag):
                    return mode.with_flag(feature, True)
                return mode.with_batching(feature)

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

    results = {}
    for mode in modes:
        results[mode] = generate(
            mode,
            model,
            input_ids,
            args.max_new_tokens,
            tokenizer.eos_token_id,
            device,
        )
        print_result(mode, results[mode], tokenizer)

    if len(modes) == 2:
        baseline = results[modes[0]]
        optimized = results[modes[1]]
        print(f"\n[特性 '{compare_feature_name}' 对照结论]")
        print("逐请求 Token 一致:", baseline.unpadded_output_token_ids == optimized.unpadded_output_token_ids)
        baseline_decode = baseline.tpot_ms
        optimized_decode = optimized.tpot_ms
        if optimized_decode > 0:
            print("平均 Decode 加速比: %.2fx" % (baseline_decode / optimized_decode))
        if baseline.e2e_wall_ms > 0 and optimized.e2e_wall_ms > 0:
            print("端到端 Wall-clock 耗时: %.2f ms -> %.2f ms" % (baseline.e2e_wall_ms, optimized.e2e_wall_ms))
            print("Batch makespan 加速比: %.3fx" % (baseline.e2e_wall_ms / optimized.e2e_wall_ms))
    print("说明: 计时不包含 tokenizer、模型下载和模型加载；默认 greedy decoding。")


if __name__ == "__main__":
    main()
