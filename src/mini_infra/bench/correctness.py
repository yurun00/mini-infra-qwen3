"""基准评测输出黄金门禁（Golden Output Correctness Gate）。"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from mini_infra.bench.env import git_info
from mini_infra.bench.paths import GOLDEN_OUTPUT_PATH
from mini_infra.engine import EngineConfig, Qwen3Tokenizer
from mini_infra.entrypoints import GenerationOptions, generate

GOLDEN_PROMPT = "请用一句话介绍 KV Cache。"
GOLDEN_MAX_OUTPUT_TOKENS = 32


@dataclass(frozen=True)
class CorrectnessReport:
    """正确性门禁检查报告。"""

    # infra-legacy(naming): 字段名对齐已落盘 JSON schema (correctness.checked/passed/updated_golden)
    checked: bool
    passed: bool | None
    match_ratio: float | None
    updated_golden: bool = False

    def to_dict(self) -> dict[str, bool | float | None]:
        return {
            "checked": self.checked,
            "passed": self.passed,
            "match_ratio": self.match_ratio,
            "updated_golden": self.updated_golden,
        }


def check_correctness(
    model: torch.nn.Module,
    tokenizer: Qwen3Tokenizer,
    config: EngineConfig,
    device: torch.device,
    update_golden: bool = False,  # infra-legacy(naming): 对齐 CLI --update-golden 参数名
) -> CorrectnessReport:
    """正确性门禁：固定 prompt 的 greedy 输出必须与 golden 一致。"""
    ids = torch.tensor([tokenizer.encode_chat_prompt(GOLDEN_PROMPT)], device=device)
    options = GenerationOptions(device=device)
    res = generate(
        config,
        model,
        ids,
        GOLDEN_MAX_OUTPUT_TOKENS,
        options=options,
    )
    tokens = res.padded_output_tokens(0).tolist()

    if update_golden:
        GOLDEN_OUTPUT_PATH.write_text(
            json.dumps({"prompt": GOLDEN_PROMPT, "tokens": tokens, "commit": git_info()["commit"]}, indent=2),
            encoding="utf-8",
        )
        print(f"[golden] 已写入 {GOLDEN_OUTPUT_PATH.name}（{len(tokens)} tokens）")
        return CorrectnessReport(checked=True, passed=True, match_ratio=1.0, updated_golden=True)

    if not GOLDEN_OUTPUT_PATH.exists():
        print("[golden] 未找到 golden 文件，跳过正确性检查（先用 --update-golden 生成）")
        return CorrectnessReport(checked=False, passed=None, match_ratio=None, updated_golden=False)

    golden = json.loads(GOLDEN_OUTPUT_PATH.read_text(encoding="utf-8"))["tokens"]
    n = min(len(golden), len(tokens))
    matched = sum(a == b for a, b in zip(golden[:n], tokens[:n]))
    ratio = matched / n if n else 0.0
    is_passed = tokens == golden
    print(f"[golden] 匹配 {matched}/{n}，{'通过' if is_passed else '失败'}")
    return CorrectnessReport(checked=True, passed=is_passed, match_ratio=ratio, updated_golden=False)


__all__ = [
    "CorrectnessReport",
    "GOLDEN_MAX_OUTPUT_TOKENS",
    "GOLDEN_PROMPT",
    "check_correctness",
]
