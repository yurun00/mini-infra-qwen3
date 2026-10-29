"""用真实 Qwen3 权重检查 Scheduler 批处理（Fixed 与 Continuous）的请求归属和输出一致性。

入口：mini-infra verify scheduler-batch。
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from mini_infra.core.request import make_request_specs
from mini_infra.engine import BatchingMethod, run_scheduler
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Tokenizer,
    load_model,
    resolve_model_directory,
)

DEFAULT_PROMPTS = [
    "用一句话解释 Continuous Batching。",
    "用一句话解释 Prefill。",
    "用一句话解释 Decode。",
    "用一句话解释 KV Cache。",
]


@torch.inference_mode()
def run_compare_scheduler_batch(
    model: torch.nn.Module,
    sequences: Sequence[Sequence[int]],
    budgets: Sequence[int],
    eos_token_id: int | None,
    device: torch.device,
    max_running_requests: int = 2,
    output: str | Path | None = None,
    environment_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """运行单请求隔离、固定批处理与连续批处理对比，返回比对统计结果字典。"""
    specs = make_request_specs(sequences, budgets)

    # 1. 串行调度基准（serialized baseline，并发槽位为 1）
    serialized = run_scheduler(
        BatchingMethod.NONE,
        model,
        specs,
        max_running_requests=1,
        eos_token_id=eos_token_id,
        device=device,
        stop_on_eos=False,
    )

    # 2. 固定批处理 (Fixed Batching)
    fixed = run_scheduler(
        BatchingMethod.FIXED,
        model,
        specs,
        max_running_requests=max_running_requests,
        eos_token_id=eos_token_id,
        device=device,
        stop_on_eos=False,
    )

    # 3. 连续批处理 (Continuous Batching)
    continuous = run_scheduler(
        BatchingMethod.CONTINUOUS,
        model,
        specs,
        max_running_requests=max_running_requests,
        eos_token_id=eos_token_id,
        device=device,
        stop_on_eos=False,
    )

    serialized_token_ids = [s.token_ids for s in serialized.sequences]
    fixed_token_ids = [s.token_ids for s in fixed.sequences]
    continuous_token_ids = [s.token_ids for s in continuous.sequences]
    admission_trace = [list(event.admitted) for event in continuous.events if event.admitted]

    result: dict[str, Any] = {
        "serialized_token_ids": serialized_token_ids,
        "fixed_token_ids": fixed_token_ids,
        "continuous_token_ids": continuous_token_ids,
        "fixed_matches_serialized": fixed_token_ids == serialized_token_ids,
        "continuous_matches_serialized": continuous_token_ids == serialized_token_ids,
        "fixed_matches_continuous": fixed_token_ids == continuous_token_ids,
        "continuous_admission_trace": admission_trace,
    }

    print("fixed 与串行单请求一致:", result["fixed_matches_serialized"])
    print("continuous 与串行单请求一致:", result["continuous_matches_serialized"])
    print("fixed 与 continuous 一致:", result["fixed_matches_continuous"])
    print("Continuous admission trace:", result["continuous_admission_trace"])

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(environment_metadata or {})
        if "gpu" not in env:
            env["gpu"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        if "pytorch" not in env:
            env["pytorch"] = torch.__version__
        if "cuda_runtime" not in env:
            env["cuda_runtime"] = torch.version.cuda if torch.cuda.is_available() else None
        payload = {
            "environment": env,
            "result": result,
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        print("JSON 结果已写入:", output_path)

    return result


def execute_compare_scheduler_batch(
    model: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_MODEL_REVISION,
    prompts: Sequence[str] | None = None,
    max_new_tokens: int = 8,
    max_running_requests: int = 2,
    dtype: str = "float32",
    output: str | None = None,
) -> int:
    """运行 Scheduler 批处理动态请求归属与输出一致性验证，返回退出码（0 成功，1 失败）。"""
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2")
    if max_running_requests < 1:
        raise ValueError("max_running_requests 必须大于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    device = torch.device("cuda")
    comparison_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype]

    model_directory = resolve_model_directory(model, revision)
    tokenizer = Qwen3Tokenizer(model_directory)
    prompts_to_use = list(prompts) if prompts is not None else DEFAULT_PROMPTS
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts_to_use]
    if len(prompts_to_use) == 4:
        budgets = [2, max_new_tokens, 3, max_new_tokens]
    else:
        budgets = [2 if i % 2 == 0 else max_new_tokens for i in range(len(prompts_to_use))]

    model_instance = load_model(model_directory, device, dtype=comparison_dtype)

    environment = {
        "gpu": torch.cuda.get_device_name(device),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "model": model,
        "revision": revision,
        "dtype": dtype,
        "max_new_tokens": max_new_tokens,
        "max_running_requests": max_running_requests,
        "decoding": "greedy, EOS disabled",
        "thinking": False,
    }

    result = run_compare_scheduler_batch(
        model=model_instance,
        sequences=sequences,
        budgets=budgets,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        max_running_requests=max_running_requests,
        output=output,
        environment_metadata=environment,
    )

    if dtype == "float32" and not result["continuous_matches_serialized"]:
        print("float32 动态请求归属或输出一致性检查失败")
        return 1
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Continuous Batching 正确性实验")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION, help="Hugging Face 模型版本")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="验证使用的精度（默认: float32）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8, help="生成的最大 Token 数（至少为 2，默认: 8）")
    parser.add_argument("--max-running-requests", type=int, default=2, help="调度器最大并发请求数（默认: 2）")
    parser.add_argument("--output", help="可选 JSON 输出路径")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """独立可执行入口。"""
    args = parse_args(argv)
    code = execute_compare_scheduler_batch(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        max_running_requests=args.max_running_requests,
        output=args.output,
    )
    if code != 0 and args.dtype == "float32":
        raise SystemExit("float32 动态请求归属或输出一致性检查失败")
    return code


# 保持向后兼容别名
run_compare_continuous_batching = run_compare_scheduler_batch
execute_compare_continuous_batching = execute_compare_scheduler_batch

__all__ = [
    "DEFAULT_PROMPTS",
    "execute_compare_continuous_batching",
    "execute_compare_scheduler_batch",
    "main",
    "parse_args",
    "run_compare_continuous_batching",
    "run_compare_scheduler_batch",
]


if __name__ == "__main__":
    main()
