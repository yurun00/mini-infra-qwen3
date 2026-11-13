"""检查 Lockstep 齐步批处理中 Batch Size、排列、同批内容和 Padding 对输出的影响。

入口：mini-infra verify lockstep-batch。
"""

from dataclasses import asdict
from typing import Any

import torch

from mini_infra.core import DenseCacheConfig
from mini_infra.engine import (
    BatchingMethod,
    EngineConfig,
)
from mini_infra.entrypoints import GenerationOptions, generate
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)
from mini_infra.verification.parity._common import (
    compare_logits,
    dump_json,
    prepare_prompt_compare,
)


def _synthetic_sequence(length: int, vocab_size: int, salt: int) -> list[int]:
    """生成确定性的伪随机 Token 序列（避免生成 0，0 为默认 pad_token_id）。"""
    values = torch.arange(length, dtype=torch.long)
    return ((values * (7919 + salt) + 17 + salt) % (vocab_size - 1) + 1).tolist()


def _compare_trace(
    reference: Any,
    candidate: Any,
    target_idx: int,
    reference_idx: int = 0,
) -> dict[str, Any]:
    """逐步对比两个运行轨迹中指定 batch slot 的 Logits 误差、生成 Token 与 Top-1/Top-2 Margin。"""
    ref_trace = reference.trace.logits_trace
    cand_trace = candidate.trace.logits_trace
    step_count = min(len(ref_trace), len(cand_trace))
    largest_error = 0.0
    largest_error_before_divergence = 0.0
    mean_error_sum = 0.0
    matching_steps = 0
    first_mismatch_step = None
    mismatch_margin = None
    per_step = []

    for step in range(step_count):
        comparison = compare_logits(
            ref_trace[step][reference_idx],
            cand_trace[step][target_idx],
            step=step + 1,
            verbose=False,
        )
        matching_steps += int(comparison.matches)

        largest_error = max(largest_error, comparison.max_abs_error)
        if first_mismatch_step is None:
            largest_error_before_divergence = max(largest_error_before_divergence, comparison.max_abs_error)
        mean_error_sum += comparison.mean_abs_error

        if not comparison.matches and first_mismatch_step is None:
            first_mismatch_step = comparison.step
            mismatch_margin = comparison.top1_top2_margin

        per_step.append(asdict(comparison))

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
def _run_compare_lockstep_batch(
    model: torch.nn.Module,
    target: list[int],
    vocab_size: int,
    max_new_tokens: int,
    eos_token_id: int,
    device: torch.device,
    logits_atol: float = 1e-4,
    output: str | None = None,
    environment_metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """执行 8 组变种与 2 组隔离对比，返回 (all_rows, failure_rows)。"""
    equal_b = _synthetic_sequence(len(target), vocab_size, 11)
    equal_c = _synthetic_sequence(len(target), vocab_size, 29)
    equal_d = _synthetic_sequence(len(target), vocab_size, 47)
    short = _synthetic_sequence(max(1, len(target) // 2), vocab_size, 61)
    long = _synthetic_sequence(len(target) + 17, vocab_size, 79)

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

    fixed_batch_config = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED)
    gen_opts = GenerationOptions(
        eos_token_id=eos_token_id,
        device=device,
        stop_on_eos=False,
        capture_logits=True,
    )

    # 1. 运行 reference (单请求 batch baseline)
    reference = generate(
        fixed_batch_config,
        model,
        [target],
        max_new_tokens,
        options=gen_opts,
    )

    rows: list[dict[str, Any]] = []
    variant_runs: dict[str, Any] = {}

    # 2. 依次运行各个变种并与 reference 对比
    for name, sequences, target_idx in variants:
        candidate = (
            reference
            if name == "single"
            else generate(
                fixed_batch_config,
                model,
                sequences,
                max_new_tokens,
                options=gen_opts,
            )
        )
        variant_runs[name] = candidate
        comparison = _compare_trace(reference, candidate, target_idx)
        comparison.update(
            {
                "variant": name,
                "batch_size": len(sequences),
                "target_idx": target_idx,
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
    for name, reference_name, candidate_name, candidate_idx in isolation_pairs:
        comparison = _compare_trace(
            variant_runs[reference_name],
            variant_runs[candidate_name],
            candidate_idx,
        )
        comparison.update(
            {
                "variant": name,
                "reference_variant": reference_name,
                "candidate_variant": candidate_name,
                "target_idx": candidate_idx,
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
        dump_json(output, {"environment": environment_metadata or {}, "results": rows})

    failures = [row for row in rows if not row["sequence_matches"] or row["largest_logits_abs_error"] > logits_atol]
    return rows, failures


def execute_compare_lockstep_batch(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompt: str = "用一句话解释固定批处理。",
    max_new_tokens: int = 8,
    dtype: str = "float32",
    logits_atol: float = 1e-4,
    output: str | None = None,
) -> int:
    """运行齐步批处理数值与请求隔离对齐验证，返回退出码（0 成功，1 失败）。"""
    if max_new_tokens < 1:
        raise ValueError("输出长度必须大于 0")
    if logits_atol < 0:
        raise ValueError("误差阈值不能小于 0")

    device, tokenizer, model_instance, environment, target = prepare_prompt_compare(
        model, revision, dtype, prompt, max_new_tokens, logits_atol
    )

    rows, failures = _run_compare_lockstep_batch(
        model=model_instance,
        target=target,
        vocab_size=len(tokenizer.vocab),
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        logits_atol=logits_atol,
        output=output,
        environment_metadata=environment,
    )

    if dtype == "float32" and failures:
        print("float32 请求隔离或数值误差检查未通过")
        return 1
    return 0


__all__ = [
    "execute_compare_lockstep_batch",
]
