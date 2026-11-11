"""硬件峰值规格、模型理论工作量画像与算力/显存带宽利用率（MFU/MBU）计算。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch

GPU_PEAK_SPECS = {
    "L4": (300.0, 121.0),
    "T4": (320.0, 65.0),
    "A10": (600.0, 125.0),
    "A100": (1555.0, 312.0),
    "H100": (3350.0, 989.0),
    "L40S": (864.0, 362.0),
    "4090": (1008.0, 165.0),
    "3090": (936.0, 71.0),
}


@dataclass(frozen=True)
class HardwarePeakSpecs:
    """GPU 峰值显存带宽 (GB/s) 与 FP16 峰值算力 (TFLOPS)。"""

    bandwidth_gbs: float | None
    tflops: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {"bandwidth_gbs": self.bandwidth_gbs, "tflops": self.tflops}

    @staticmethod
    def _match_name(gpu_name: str) -> tuple[float | None, float | None]:
        """按词边界匹配 GPU 型号，优先最长 key：避免 `A10` 命中 `A100`、`L4` 命中 `L40S`。"""
        best_key: str | None = None
        for key in GPU_PEAK_SPECS:
            if re.search(rf"(?<![0-9A-Za-z]){re.escape(key)}(?![0-9])", gpu_name):
                if best_key is None or len(key) > len(best_key):
                    best_key = key
        return GPU_PEAK_SPECS[best_key] if best_key else (None, None)

    @classmethod
    def lookup(
        cls,
        gpu_name: str,
        bandwidth_gbs: float | None = None,
        tflops: float | None = None,
    ) -> HardwarePeakSpecs:
        """推断或覆盖 GPU 峰值显存带宽 (GB/s) 与 FP16 峰值算力 (TFLOPS)。"""
        spec = cls._match_name(gpu_name)
        return cls(bandwidth_gbs=bandwidth_gbs or spec[0], tflops=tflops or spec[1])


@dataclass(frozen=True)
class ModelCostProfile:
    """理想工作量模型（与引擎实现无关）：用于计算 MBU/MFU。"""

    n_params: int
    weight_bytes: int
    kv_bytes_per_token: int
    attn_flop_coeff: int

    def to_dict(self) -> dict[str, int]:
        return {
            "n_params": self.n_params,
            "weight_bytes": self.weight_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "attn_flop_coeff": self.attn_flop_coeff,
        }

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> ModelCostProfile:
        """理想工作量模型（与引擎实现无关）：用于计算 MBU/MFU。

        weight_bytes：每个 decode step 至少读取一遍全部权重（tied embedding 只算一份，lm_head 会读它）。
        flops_per_token：2 * 参数量（线性层）；注意力部分另按上下文长度计算。
        """
        cfg = model.config
        params = list(model.parameters())
        elem = params[0].element_size()
        n_params = sum(p.numel() for p in params)
        return cls(
            n_params=n_params,
            weight_bytes=n_params * elem,
            kv_bytes_per_token=2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * elem,
            attn_flop_coeff=cfg.num_hidden_layers * cfg.num_attention_heads * cfg.head_dim,
        )


@dataclass(frozen=True)
class EfficiencyMetrics:
    """基于理想工作量与实测延迟计算的 prefill/decode 算力 (MFU) 与带宽利用率 (MBU)。"""

    prefill_mfu: float | None = None
    decode_mfu: float | None = None
    decode_mbu: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "prefill_mfu": self.prefill_mfu,
            "decode_mfu": self.decode_mfu,
            "decode_mbu": self.decode_mbu,
        }

    @classmethod
    def compute(
        cls,
        cost: ModelCostProfile,
        peaks: HardwarePeakSpecs,
        batch: int,
        prompt_len: int | Sequence[int],
        max_new_tokens: int | Sequence[int],
        ttft_ms: float,
        tpot_ms: float,
    ) -> EfficiencyMetrics:
        """基于理想工作量与实测延迟，计算 prefill/decode 的 MFU 与 decode 的 MBU（带宽利用率）。"""
        avg_out_len = (
            sum(max_new_tokens) / len(max_new_tokens)
            if isinstance(max_new_tokens, Sequence) and not isinstance(max_new_tokens, str)
            else max_new_tokens
        )

        n_params = cost.n_params
        attn_flop_coeff = cost.attn_flop_coeff
        weight_bytes = cost.weight_bytes
        kv_bytes_per_token = cost.kv_bytes_per_token

        tflops = peaks.tflops
        bandwidth_gbs = peaks.bandwidth_gbs

        if isinstance(prompt_len, Sequence) and not isinstance(prompt_len, str):
            avg_prompt_len = sum(prompt_len) / len(prompt_len)
            ctx = avg_prompt_len + avg_out_len / 2
            sum_prefill_flops = sum(2 * n_params * pl + 2 * attn_flop_coeff * pl**2 for pl in prompt_len)
        else:
            ctx = prompt_len + avg_out_len / 2
            sum_prefill_flops = batch * (2 * n_params * prompt_len + 2 * attn_flop_coeff * prompt_len**2)

        prefill_mfu = None
        decode_mfu = None
        decode_mbu = None

        if tflops:
            peak_flops = tflops * 1e12
            prefill_mfu = sum_prefill_flops / (ttft_ms / 1000) / peak_flops if ttft_ms else None
            dec = batch * (2 * n_params + 4 * attn_flop_coeff * ctx)
            decode_mfu = dec / (tpot_ms / 1000) / peak_flops if tpot_ms else None
        if bandwidth_gbs and tpot_ms:
            moved = weight_bytes + batch * ctx * kv_bytes_per_token
            decode_mbu = moved / (tpot_ms / 1000) / (bandwidth_gbs * 1e9)
        return cls(prefill_mfu=prefill_mfu, decode_mfu=decode_mfu, decode_mbu=decode_mbu)


__all__ = [
    "EfficiencyMetrics",
    "GPU_PEAK_SPECS",
    "HardwarePeakSpecs",
    "ModelCostProfile",
]
