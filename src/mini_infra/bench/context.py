"""评测执行上下文（BenchmarkExecutionContext）：持有单次或批量评测共享的模型与环境信息。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import torch

from mini_infra.bench.correctness import CorrectnessReport
from mini_infra.bench.efficiency import HardwarePeakSpecs, ModelCostProfile
from mini_infra.bench.env import git_info, gpu_info, require_cuda
from mini_infra.engine import Mode, Qwen3Tokenizer
from mini_infra.model import load_model, resolve_model_directory


@dataclass
class BenchmarkExecutionContext:
    """一次评测运行（单个 suite、单个 case 或 sweep 内多个 suite）共享的模型与环境信息。"""

    model: torch.nn.Module
    tokenizer: Qwen3Tokenizer | None
    device: torch.device
    vocab: int
    env: dict[str, Any]
    peaks: HardwarePeakSpecs
    cost: ModelCostProfile | None
    git: dict[str, Any]
    correctness_cache: dict[Mode, CorrectnessReport] = field(default_factory=dict)

    @property
    def commit(self) -> str:
        return str(self.git.get("commit", "unknown"))

    @property
    def dirty(self) -> bool:
        return bool(self.git.get("dirty", False))

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BenchmarkExecutionContext:
        """要求 CUDA、加载模型并采集环境信息，构建 BenchmarkExecutionContext。"""
        device = require_cuda()
        model_dir = resolve_model_directory(args.model, args.revision)
        tokenizer = Qwen3Tokenizer(model_dir)
        model = load_model(model_dir, device)
        env = gpu_info(device)
        peaks = HardwarePeakSpecs.lookup(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
        if peaks.bandwidth_gbs is None or peaks.tflops is None:
            print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            vocab=model.config.vocab_size,
            env=env,
            peaks=peaks,
            cost=ModelCostProfile.from_model(model),
            git=git_info(),
        )


__all__ = [
    "BenchmarkExecutionContext",
]
