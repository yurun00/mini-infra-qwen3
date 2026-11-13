"""基础模型执行器（BaseModelRunner）：持有底层模型与硬件配置。

只认裸张量与底层硬件配置（device / pad_token_id），对请求队列、生命周期与高阶批处理场景完全无感知。
单步执行由各专职 Runner 在调用点显式组合：`with StepTimer(...)` 圈定计时区间，区间内直接调模型前向，
再用 `greedy_sample(logits)` 采样。
"""

from collections.abc import Sequence
from typing import Any, NamedTuple

import torch

from mini_infra.device import StepTimer
from mini_infra.worker.kv_cache import KVCache
from mini_infra.worker.sampler import greedy_sample


class PrefillOutput(NamedTuple):
    """Prefill 单步产出：各请求首个 Token ID 列表、Prefill Cache 容器与模型耗时（ms）。"""

    next_tokens: list[int]
    prefill_cache: KVCache
    model_ms: float


def _left_pad_sequences(
    token_sequences: Sequence[Any], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把一组非空的一维 Token 序列左 Padding 为矩形 Tensor（内部物理张量辅助函数）。

    Returns:
        input_ids:      [B, L], long, contiguous
        attention_mask: [B, L], bool, contiguous
        position_ids:   [B, L], long, contiguous
        prompt_lengths: [B],    long, contiguous
    """
    if not token_sequences:
        raise ValueError("token_sequences 不能为空")
    prompt_lengths = []
    normalized = []
    for sequence in token_sequences:
        tensor = torch.as_tensor(sequence, dtype=torch.long, device=device)
        if tensor.ndim != 1 or tensor.numel() < 1:
            raise ValueError("每个 Prompt 必须是一维非空 Token 序列")
        normalized.append(tensor)
        prompt_lengths.append(tensor.numel())

    padded_prompt_length = max(prompt_lengths)
    batch_size = len(normalized)
    input_ids = torch.full((batch_size, padded_prompt_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, padded_prompt_length), dtype=torch.bool, device=device)
    for row_idx, sequence in enumerate(normalized):
        prompt_length = sequence.numel()
        input_ids[row_idx, padded_prompt_length - prompt_length :] = sequence
        attention_mask[row_idx, padded_prompt_length - prompt_length :] = True

    # 真实 Token 的逻辑位置从 0 开始；Padding 位置记为 0（会被 mask 掉）。
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.clamp_(min=0)
    return input_ids, attention_mask, position_ids, torch.tensor(prompt_lengths, dtype=torch.long, device=device)


class BaseModelRunner:
    """基础无状态模型执行器：只持有模型与硬件配置，不承载单步执行原语与请求批次状态。"""

    def __init__(self, model: torch.nn.Module, pad_token_id: int = 0):
        self.model = model
        self.device = next(model.parameters()).device
        self.pad_token_id = pad_token_id

    def run_prefill_step(self, prompt_sequences: Sequence[Sequence[int]]) -> PrefillOutput:
        """对一批新入场请求做左 Padding Prefill，返回各请求首个 Token 列表、Prefill Cache 与模型耗时。

        prefill_cache: 逐层 (key, value) 均为 [B, H_kv, L, D], model_dtype, contiguous
        """
        input_ids, attention_mask, position_ids, _ = _left_pad_sequences(
            prompt_sequences, self.pad_token_id, self.device
        )
        with StepTimer(self.device) as timer:
            logits, present_key_values = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
            next_token, _ = greedy_sample(logits)
        return PrefillOutput(
            next_tokens=next_token.squeeze(-1).tolist(),
            prefill_cache=KVCache(present_key_values),
            model_ms=timer.elapsed_ms,
        )


__all__ = [
    "BaseModelRunner",
    "PrefillOutput",
]
