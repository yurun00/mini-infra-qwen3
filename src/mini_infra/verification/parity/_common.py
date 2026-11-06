"""Parity 验证共享工具：CUDA 模型加载、环境元数据、JSON 输出与逐步比对统计。

本模块只依赖原生 Python 与 PyTorch；严禁引入 transformers（Golden Reference 仅限
compare_transformers.py）。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mini_infra.engine import Qwen3Tokenizer
from mini_infra.model import load_model, resolve_model_directory

_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class _LogitComparison:
    """单步 logits 比较结果：误差、贪心 Token、一致性、Top-1/Top-2 Margin。"""

    step: int
    reference_token: int
    candidate_token: int
    matches: bool
    max_abs_error: float
    mean_abs_error: float
    top1_top2_margin: float


def _parse_dtype(name: str) -> torch.dtype:
    """把 CLI 精度名映射为 torch dtype。"""
    return _DTYPE_BY_NAME[name]


def _require_cuda() -> torch.device:
    """守卫 CUDA 可用性并固定随机种子，返回 cuda device。"""
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")
    torch.manual_seed(0)
    return torch.device("cuda")


def prepare_cuda_model(
    model: str,
    revision: str,
    dtype: str | None = None,
) -> tuple[torch.device, str, Qwen3Tokenizer, torch.nn.Module]:
    """守卫 CUDA、固定种子并加载模型，返回 (device, model_directory, tokenizer, model)。

    dtype 为 None 时按模型配置文件的 dtype 加载；否则按 CLI 指定的精度加载。
    """
    device = _require_cuda()
    model_directory = resolve_model_directory(model, revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    model_dtype = _parse_dtype(dtype) if dtype is not None else None
    model_instance = load_model(model_directory, device, dtype=model_dtype)
    return device, model_directory, tokenizer, model_instance


def build_environment(
    device: torch.device,
    *,
    model: str,
    revision: str,
    dtype: str,
    **extra: Any,
) -> dict[str, Any]:
    """构造环境元数据字典，extra 追加业务字段。"""
    environment: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda if torch.cuda.is_available() else None,
        "model": model,
        "revision": revision,
        "dtype": dtype,
    }
    environment.update(extra)
    return environment


def prepare_prompt_compare(
    model: str,
    revision: str,
    dtype: str,
    prompt: str,
    max_new_tokens: int,
    logits_atol: float,
) -> tuple[torch.device, Qwen3Tokenizer, torch.nn.Module, dict[str, Any], list[int]]:
    """单 Prompt 对齐入口的公共引导：加载模型、编码 Prompt、构造环境并打印表头。

    返回 (device, tokenizer, model, environment, token_ids)。
    """
    device, _, tokenizer, model_instance = prepare_cuda_model(model, revision, dtype)
    token_ids = tokenizer.encode_chat_prompt(prompt)
    environment = build_environment(
        device,
        model=model,
        revision=revision,
        dtype=dtype,
        prompt_tokens=len(token_ids),
        max_new_tokens=max_new_tokens,
        logits_atol=logits_atol,
        decoding="greedy",
        thinking=False,
    )
    print("Prompt Token 数: %d" % len(token_ids))
    print("对比 dtype: %s, 最大允许 Logits 绝对误差: %.6f" % (dtype, logits_atol))
    return device, tokenizer, model_instance, environment, token_ids


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    """将 payload 以 UTF-8、2 空格缩进写入 JSON 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print("JSON 结果已写入:", output_path)


def _logits_abs_error(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> tuple[float, float]:
    """计算两组 logits 的最大/平均绝对误差（先转 float32 再相减，避免低精度截断）。"""
    difference = (reference_logits.float() - candidate_logits.float()).abs()
    return difference.max().item(), difference.mean().item()


def _report_top_logits(title: str, logits: torch.Tensor, k: int = 3) -> None:
    """打印一组 logits 的前 k 大 Token 与数值，用于分歧诊断。"""
    values, indices = torch.topk(logits.reshape(-1), k=k)
    print("%s 前 %d 大 Logits:" % (title, k))
    for rank, (token_id, value) in enumerate(zip(indices.tolist(), values.tolist()), 1):
        print("  %d. token=%d, logit=%.6f" % (rank, token_id, value))


def compare_logits(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    step: int,
    reference_name: str = "reference",
    candidate_name: str = "candidate",
    extra: str | None = None,
    verbose: bool = True,
) -> _LogitComparison:
    """比较两组 last-token logits，打印单步结果并在分歧时打印 top-k 诊断，返回结构化比较记录。"""
    max_error, mean_error = _logits_abs_error(reference_logits, candidate_logits)
    reference_token = int(reference_logits.argmax(dim=-1).item())
    candidate_token = int(candidate_logits.argmax(dim=-1).item())

    top_two = torch.topk(candidate_logits.reshape(-1).float(), k=2).values
    margin = float((top_two[0] - top_two[1]).item())
    matches = reference_token == candidate_token

    if verbose:
        detail = " | %s" % extra if extra else ""
        print(
            "步骤 %02d | %s=%d | %s=%d | 最大误差=%.6f | 平均误差=%.6f%s"
            % (step, reference_name, reference_token, candidate_name, candidate_token, max_error, mean_error, detail)
        )
        if not matches:
            print(
                "\n[Token 不一致] 步骤 %02d: %s=%d, %s=%d"
                % (step, reference_name, reference_token, candidate_name, candidate_token)
            )
            _report_top_logits(reference_name, reference_logits)
            _report_top_logits(candidate_name, candidate_logits)

    return _LogitComparison(
        step=step,
        reference_token=reference_token,
        candidate_token=candidate_token,
        matches=matches,
        max_abs_error=max_error,
        mean_abs_error=mean_error,
        top1_top2_margin=margin,
    )


def run_stepwise_greedy_compare(
    step_fn: Callable[[], tuple[torch.Tensor, torch.Tensor, int, str | None]],
    *,
    max_new_tokens: int,
    eos_token_id: int | None,
    reference_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    """逐步贪心解码对齐框架：每步调用一次 step_fn 并比较两侧最后一位 logits。

    step_fn 的约定：
    - 无入参；每次被调用需推进自身状态（闭包持有自回归历史，把新 token 拼到输入序列末尾）。
    - 返回四元组 (reference_logits, candidate_logits, token_id, extra)：
      * reference_logits / candidate_logits：两侧本步最后一位 logits，形状 [vocab]（或 [1, vocab]）。
      * token_id：本步贪心选出的 token，追加到 generated_token_ids 并用于 EOS 判定；
        分歧时框架提前返回，此时两侧 token 必相等，故取驱动侧 argmax 即可。
      * extra：追加到单步打印行的可选诊断串，无则传 None。
    """
    generated_token_ids = []
    largest_error = 0.0
    mean_error_sum = 0.0

    for step in range(max_new_tokens):
        reference_logits, candidate_logits, token_id, extra = step_fn()
        comparison = compare_logits(
            reference_logits,
            candidate_logits,
            step=step + 1,
            reference_name=reference_name,
            candidate_name=candidate_name,
            extra=extra,
        )
        largest_error = max(largest_error, comparison.max_abs_error)
        mean_error_sum += comparison.mean_abs_error

        if not comparison.matches:
            return {
                "tokens_match": False,
                "first_mismatch_step": step + 1,
                "largest_error": largest_error,
                "average_mean_error": mean_error_sum / (step + 1),
                "token_ids": generated_token_ids,
            }

        generated_token_ids.append(token_id)
        if token_id == eos_token_id:
            break

    return {
        "tokens_match": True,
        "first_mismatch_step": None,
        "largest_error": largest_error,
        "average_mean_error": mean_error_sum / len(generated_token_ids),
        "token_ids": generated_token_ids,
    }


def _report_stepwise_summary(result: dict[str, Any], generated_text: str) -> None:
    """打印逐步比对的总览结果（逐 Token 一致性、误差统计、生成 Token 与文本）。"""
    print("\n逐 Token 一致:", result["tokens_match"])
    print("所有步骤最大 Logits 绝对误差: %.6f" % result["largest_error"])
    print("各步骤平均 Logits 误差的均值: %.6f" % result["average_mean_error"])
    print("生成 Token IDs:", result["token_ids"])
    print("生成文本:", generated_text)


def finalize_stepwise_compare(
    result: dict[str, Any],
    generated_text: str,
    *,
    logits_atol: float,
    output: str | None,
    environment: dict[str, Any],
) -> int:
    """打印逐步对齐结果、写可选 JSON，并按 Token 一致性与误差阈值返回退出码。"""
    _report_stepwise_summary(result, generated_text)
    if output:
        dump_json(output, {"environment": environment, "result": result})
    if not result["tokens_match"]:
        print("首个不一致步骤:", result["first_mismatch_step"])
        return 1
    if result["largest_error"] > logits_atol:
        print("Token 一致，但 Logits 误差超过设定阈值。")
        return 2
    return 0
