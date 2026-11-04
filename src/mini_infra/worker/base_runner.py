"""基础模型执行器（BaseModelRunner）：负责底层张量前向与设备端采样。

只认裸张量与底层硬件配置（device / pad_token_id），对请求队列、生命周期与高阶批处理场景完全无感知。
"""

import time
from collections.abc import Sequence
from typing import Any, NamedTuple

import torch

from mini_infra.device import synchronize


class ForwardSampleOutput(NamedTuple):
    """底层张量单步前向与贪心采样产出（纯张量面内部值对象）：

    - 属性访问：.next_token (GPU Tensor), .present_key_values, .elapsed_ms, .event_pair, .last_logits
    - present_key_values 是模型返回的逐层 (key, value) 纯张量列表；
    - last_logits 仅在 capture_logits=True 时填充；CUDA 下 should_sync=False 时 event_pair 存放计时 Event，
      elapsed_ms 为 0。
    """

    next_token: torch.Tensor
    present_key_values: list[tuple[torch.Tensor, torch.Tensor]]
    elapsed_ms: float = 0.0
    event_pair: tuple[Any, Any] | None = None
    last_logits: torch.Tensor | None = None


def _left_pad_sequences(
    token_sequences: Sequence[Any], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把一组非空的一维 Token 序列左 Padding 为矩形 Tensor（内部物理张量辅助函数）。

    Returns:
        input_ids [B, L], attention_mask [B, L] (bool), position_ids [B, L], prompt_lengths [B]
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
    """基础无状态模型执行器：只负责底层张量前向与贪心采样，不持有任何请求批次状态。"""

    def __init__(self, model: torch.nn.Module, device: torch.device, pad_token_id: int = 0):
        self.model = model
        self.device = device
        self.pad_token_id = pad_token_id

    def _forward_and_sample(
        self,
        model_input: torch.Tensor,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        capture_logits: bool = False,
        should_sync: bool = False,
    ) -> ForwardSampleOutput:
        """底层单步张量前向推理与贪心采样（受保护底层物理原语）。

        - past_key_values 为 None 时做全量上下文计算（Prefill 或 No-Cache baseline），否则做增量 Decode；
        - should_sync=False（默认）：CUDA 下通过 Event 异步打点，供静态自回归循环避免每步 CPU-GPU 同步；
        - should_sync=True：执行前后硬件同步，直接计算耗时 ms，供调度器推进逻辑时钟。
        """
        if should_sync:
            synchronize(self.device)

        event_pair = None
        elapsed_ms = 0.0

        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            t0 = time.perf_counter()

        logits, present_key_values = self.model(
            model_input, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values
        )
        last_logits = logits[:, -1, :]
        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        if self.device.type == "cuda":
            end.record()
            if should_sync:
                synchronize(self.device)
                elapsed_ms = start.elapsed_time(end)
            else:
                event_pair = (start, end)
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000

        result_logits = last_logits if capture_logits else None
        return ForwardSampleOutput(
            next_token=next_token,
            present_key_values=present_key_values,
            elapsed_ms=elapsed_ms,
            event_pair=event_pair,
            last_logits=result_logits,
        )


__all__ = [
    "BaseModelRunner",
    "ForwardSampleOutput",
]
