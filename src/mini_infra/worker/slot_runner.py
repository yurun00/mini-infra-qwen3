"""连续批处理 batch slot 调度专职执行器。

负责管理基于 DenseSlotCache 的 Prefill 与增量 Decode 模型前向计算与计时。
"""

from collections.abc import Sequence
from typing import NamedTuple

import torch

from mini_infra.device import StepTimer
from mini_infra.model import DenseAttentionMetadata
from mini_infra.worker.base_runner import BaseModelRunner
from mini_infra.worker.dense_slot_cache import DenseSlotCache
from mini_infra.worker.sampler import greedy_sample


class SlotDecodeOutput(NamedTuple):
    """一步 batch slot Decode 的产出：按 batch slot 索引排列的下一个 Token ID 列表与模型耗时（ms）。"""

    batch_slot_tokens: list[int]
    model_ms: float


def _prepare_decode_inputs(
    slot_cache: DenseSlotCache,
    running_batch_slot_indices: Sequence[int],
    last_tokens: Sequence[int],
    positions: Sequence[int],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造 capacity 行的单步 Decode 输入（纯组装，不 sync、不计时）。

    空闲/已结束 batch slot 填 Padding 并整行 mask；先在 Host 侧组装列表再一次性传输，避免逐 batch slot 的小 kernel。

    Returns:
        input_ids:      [capacity, 1],            long, contiguous
        position_ids:   [capacity, 1],            long, contiguous
        attention_mask: [capacity, past_len + 1], bool, contiguous
    """
    if slot_cache.history_mask is None:
        raise RuntimeError("Cache 尚未分配，无法构造 Decode 输入")
    token_ids = [pad_token_id] * slot_cache.capacity
    pos_list = [0] * slot_cache.capacity
    active = [False] * slot_cache.capacity
    for batch_slot_idx, token_id, pos in zip(running_batch_slot_indices, last_tokens, positions):
        token_ids[batch_slot_idx] = token_id
        pos_list[batch_slot_idx] = pos
        active[batch_slot_idx] = True
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=slot_cache.device).unsqueeze(1)
    position_ids = torch.tensor(pos_list, dtype=torch.long, device=slot_cache.device).unsqueeze(1)
    active_mask = torch.tensor(active, dtype=torch.bool, device=slot_cache.device).unsqueeze(1)
    attention_mask = torch.cat((slot_cache.history_mask, active_mask), dim=1)
    return input_ids, position_ids, attention_mask


class SlotRunner(BaseModelRunner):
    """连续批处理 batch slot 调度场景专职执行器：无状态，驱动 DenseSlotCache 完成 Prefill 与 Decode。"""

    def prepare_decode_inputs(
        self,
        dense_slot_cache: DenseSlotCache,
        running_batch_slot_indices: Sequence[int],
        last_tokens: Sequence[int],
        positions: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构造单步 Decode 输入（纯组装，不 sync、不计时，供 engine 在测量区间边界自行计时）。

        Returns:
            input_ids:      [capacity, 1],            long, contiguous
            position_ids:   [capacity, 1],            long, contiguous
            attention_mask: [capacity, past_len + 1], bool, contiguous
        """
        return _prepare_decode_inputs(
            slot_cache=dense_slot_cache,
            running_batch_slot_indices=running_batch_slot_indices,
            last_tokens=last_tokens,
            positions=positions,
            pad_token_id=self.pad_token_id,
        )

    def run_decode_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        dense_slot_cache: DenseSlotCache,
    ) -> SlotDecodeOutput:
        """执行一步 Decode 前向与采样，并把新 KV 与本步 attention_mask 写回 batch slot。

        input_ids:      [capacity, 1],            long, contiguous
        attention_mask: [capacity, past_len + 1], bool, contiguous
        position_ids:   [capacity, 1],            long, contiguous
        """
        with StepTimer(self.device) as timer:
            logits, present_key_values = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                attn_metadata=DenseAttentionMetadata(tuple(dense_slot_cache.past_key_values)),
            )
            next_token, _ = greedy_sample(logits)
        dense_slot_cache.commit_decode_step(attention_mask, present_key_values)
        return SlotDecodeOutput(batch_slot_tokens=next_token.squeeze(-1).tolist(), model_ms=timer.elapsed_ms)


__all__ = [
    "SlotDecodeOutput",
    "SlotRunner",
]
