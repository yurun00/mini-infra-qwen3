"""运行手写 Qwen3-0.6B 单请求推理，支持 no-cache / KV Cache 对照。入口：mini-infra-run。"""

import argparse

import torch

from mini_infra.bench.generation import generate, percentile, synchronize
from mini_infra.qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from mini_infra.qwen3_tokenizer import Qwen3Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="运行并比较 KV Cache 推理")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--prompt", default="请用一句话介绍 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("no-cache", "kv-cache", "both"),
        default="both",
    )
    return parser.parse_args()


def print_result(mode, result, tokenizer):
    new_token_ids = result.new_tokens[0].tolist()
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    decode_times = result.step_latencies_ms[1:]
    print("\n[%s]" % mode)
    print("生成 Token IDs:", new_token_ids)
    print("生成文本:", generated_text)
    print("Prefill 延迟: %.2f ms" % result.ttft_ms)
    print("平均 Decode 延迟: %.2f ms/token" % result.tpot_ms)
    print("P50 Decode 延迟: %.2f ms/token" % percentile(decode_times, 50))
    print("P95 Decode 延迟: %.2f ms/token" % percentile(decode_times, 95))
    print("端到端模型计算延迟: %.2f ms" % result.e2e_ms)
    print("Prefill 后已分配显存: %.2f MiB" % result.memory["prefill_allocated_mb"])
    print("峰值已分配显存: %.2f MiB" % result.memory["peak_allocated_mb"])
    print("最终 KV Cache 数据量: %.2f MiB" % (result.cache_bytes / 1024**2))
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
    token_ids = tokenizer.encode_chat_prompt(args.prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    model = load_handwritten_model(model_directory, device)
    modes = ("no-cache", "kv-cache") if args.mode == "both" else (args.mode,)

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

    print("Prompt Token 数:", input_ids.shape[1])
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

    if len(results) == 2:
        baseline = results["no-cache"]
        optimized = results["kv-cache"]
        print("\n[对照结论]")
        print("逐 Token 输出一致:", torch.equal(baseline.new_tokens, optimized.new_tokens))
        baseline_decode = baseline.tpot_ms
        optimized_decode = optimized.tpot_ms
        if optimized_decode > 0:
            print("平均 Decode 加速比: %.2fx" % (baseline_decode / optimized_decode))
    print("说明: 计时不包含 tokenizer、模型下载和模型加载；默认 greedy decoding。")


if __name__ == "__main__":
    main()
