"""检查 Batch Size、排列、同批内容和 Padding 对 Decode 输出的影响。入口： mini-infra-compare-batch。"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mini_infra.generation.engine import generate
from mini_infra.generation.mode import BatchingMethod, FeatureFlag, Mode
from mini_infra.qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from mini_infra.qwen3_tokenizer import Qwen3Tokenizer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="固定 Batch 数值与请求隔离实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本")
    parser.add_argument("--prompt", default="用一句话解释固定批处理。")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="生成的最大 Token 数")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="正确性验证默认使用 float32，避免低精度路径差异干扰判断",
    )
    parser.add_argument(
        "--logits-atol",
        type=float,
        default=1e-4,
        help="允许的最大 Logits 绝对误差",
    )
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args(argv)


def synthetic_sequence(length: int, vocab_size: int, salt: int) -> list[int]:
    """生成确定性的伪随机 Token 序列（避免生成 0，0 为默认 pad_token_id）。"""
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def compare_trace(
    reference: Any,
    candidate: Any,
    target_index: int,
    reference_index: int = 0,
) -> dict[str, Any]:
    """逐步对比两个运行轨迹中指定槽位的 Logits 误差、生成 Token 与 Top-1/Top-2 Margin。"""
    step_count = min(len(reference["logits_trace"]), len(candidate["logits_trace"]))
    largest_error = 0.0
    largest_error_before_divergence = 0.0
    mean_error_sum = 0.0
    matching_steps = 0
    first_mismatch_step = None
    mismatch_margin = None
    per_step = []

    for step in range(step_count):
        reference_logits = reference["logits_trace"][step][reference_index]
        candidate_logits = candidate["logits_trace"][step][target_index]
        difference = (reference_logits.float() - candidate_logits.float()).abs()
        max_error = difference.max().item()
        mean_error = difference.mean().item()

        reference_token = int(reference_logits.argmax().item())
        candidate_token = int(candidate_logits.argmax().item())

        top_two = torch.topk(candidate_logits.float(), k=2).values
        margin = float((top_two[0] - top_two[1]).item())
        matches = reference_token == candidate_token
        matching_steps += int(matches)

        largest_error = max(largest_error, max_error)
        if first_mismatch_step is None:
            largest_error_before_divergence = max(largest_error_before_divergence, max_error)
        mean_error_sum += mean_error

        if not matches and first_mismatch_step is None:
            first_mismatch_step = step + 1
            mismatch_margin = margin

        per_step.append(
            {
                "step": step + 1,
                "reference_token": reference_token,
                "candidate_token": candidate_token,
                "matches": matches,
                "max_abs_error": max_error,
                "mean_abs_error": mean_error,
                "candidate_top1_top2_margin": margin,
            }
        )

    return {
        "steps": step_count,
        "top1_match_rate": matching_steps / step_count if step_count else 0.0,
        "sequence_matches": first_mismatch_step is None,
        "first_mismatch_step": first_mismatch_step,
        "first_mismatch_margin": mismatch_margin,
        "largest_logits_abs_error": largest_error,
        "largest_logits_abs_error_before_divergence": largest_error_before_divergence,
        "average_mean_logits_abs_error": (mean_error_sum / step_count if step_count else 0.0),
        "per_step": per_step,
    }


@torch.inference_mode()
def run_compare_batch(
    model: torch.nn.Module,
    target: list[int],
    vocab_size: int,
    max_new_tokens: int,
    eos_token_id: int,
    device: torch.device,
    dtype: str = "float32",
    logits_atol: float = 1e-4,
    output: str | Path | None = None,
    environment_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """执行 8 组变种与 2 组隔离对比，返回 (all_rows, failure_rows)。"""
    equal_b = synthetic_sequence(len(target), vocab_size, 11)
    equal_c = synthetic_sequence(len(target), vocab_size, 29)
    equal_d = synthetic_sequence(len(target), vocab_size, 47)
    short = synthetic_sequence(max(1, len(target) // 2), vocab_size, 61)
    long = synthetic_sequence(len(target) + 17, vocab_size, 79)

    variants = [
        ("single", [target], 0),
        ("pair_content_b", [target, equal_b], 0),
        ("pair_content_c", [target, equal_c], 0),
        ("pair_reordered", [equal_b, target], 1),
        ("quad", [target, equal_b, equal_c, equal_d], 0),
        ("quad_reversed", [equal_d, equal_c, equal_b, target], 3),
        ("pair_short_first", [short, target], 1),
        ("pair_long_second", [target, long], 0),
    ]

    fixed_batch_mode = Mode({FeatureFlag.KV_CACHE}, batching=BatchingMethod.FIXED)

    # 1. 运行 reference (单请求 batch baseline)
    reference = generate(
        fixed_batch_mode,
        model,
        [target],
        max_new_tokens,
        eos_token_id=eos_token_id,
        device=device,
        stop_on_eos=False,
        capture_logits=True,
    )

    rows: list[dict[str, Any]] = []
    variant_runs: dict[str, Any] = {}

    # 2. 依次运行各个变种并与 reference 对比
    for name, sequences, target_index in variants:
        candidate = (
            reference
            if name == "single"
            else generate(
                fixed_batch_mode,
                model,
                sequences,
                max_new_tokens,
                eos_token_id=eos_token_id,
                device=device,
                stop_on_eos=False,
                capture_logits=True,
            )
        )
        variant_runs[name] = candidate
        comparison = compare_trace(reference, candidate, target_index)
        comparison.update(
            {
                "variant": name,
                "batch_size": len(sequences),
                "target_index": target_index,
                "prompt_lengths": [len(sequence) for sequence in sequences],
            }
        )
        rows.append(comparison)
        print(
            "%s: batch=%d, top1=%.2f%%, sequence_match=%s, "
            "first_mismatch=%s, pre_divergence_max_error=%.6f, max_error=%.6f"
            % (
                name,
                len(sequences),
                comparison["top1_match_rate"] * 100,
                comparison["sequence_matches"],
                comparison["first_mismatch_step"],
                comparison["largest_logits_abs_error_before_divergence"],
                comparison["largest_logits_abs_error"],
            )
        )

    # 3. 隔离性对比（同批不同请求内容隔离、排列顺序隔离）
    isolation_pairs = [
        ("pair_content_isolation", "pair_content_b", "pair_content_c", 0),
        ("quad_order_isolation", "quad", "quad_reversed", 3),
    ]
    for name, reference_name, candidate_name, candidate_index in isolation_pairs:
        comparison = compare_trace(
            variant_runs[reference_name],
            variant_runs[candidate_name],
            candidate_index,
        )
        comparison.update(
            {
                "variant": name,
                "reference_variant": reference_name,
                "candidate_variant": candidate_name,
                "target_index": candidate_index,
            }
        )
        rows.append(comparison)
        print(
            "%s: top1=%.2f%%, sequence_match=%s, max_error=%.6f"
            % (
                name,
                comparison["top1_match_rate"] * 100,
                comparison["sequence_matches"],
                comparison["largest_logits_abs_error"],
            )
        )

    # 4. 可选写入 JSON 输出
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_data = {
            "environment": environment_metadata or {},
            "results": rows,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(dump_data, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)

    failures = [row for row in rows if not row["sequence_matches"] or row["largest_logits_abs_error"] > logits_atol]
    return rows, failures


def main(argv=None):
    args = parse_args(argv)
    if args.max_new_tokens < 1:
        raise ValueError("输出长度必须大于 0")
    if args.logits_atol < 0:
        raise ValueError("误差阈值不能小于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    torch.manual_seed(0)
    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    target = tokenizer.encode_chat_prompt(args.prompt)
    vocab_size = len(tokenizer.vocab)

    comparison_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = load_handwritten_model(model_directory, device, dtype=comparison_dtype)

    environment = {
        "gpu": torch.cuda.get_device_name(device),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "prompt_tokens": len(target),
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "thinking": False,
    }

    print("Prompt Token 数: %d" % len(target))
    print("对比 dtype: %s, 最大允许 Logits 绝对误差: %.6f" % (args.dtype, args.logits_atol))

    rows, failures = run_compare_batch(
        model=model,
        target=target,
        vocab_size=vocab_size,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        dtype=args.dtype,
        logits_atol=args.logits_atol,
        output=args.output,
        environment_metadata=environment,
    )

    if args.dtype == "float32" and failures:
        print("float32 请求隔离或数值误差检查未通过")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
