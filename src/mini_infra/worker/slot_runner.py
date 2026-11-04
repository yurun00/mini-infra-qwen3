"""连续批处理槽位调度专职执行器。

负责管理基于 DenseSlotCache 槽位池的 Prefill 与增量 Decode 模型前向计算与计时。
"""

import time
from collections.abc import Sequence
from typing import NamedTuple

import torch

from mini_infra.device import synchronize
from mini_infra.worker.base_runner import BaseModelRunner, _left_pad_sequences
from mini_infra.worker.dense_slot_cache import DenseSlotCache
from mini_infra.worker.kv_cache import KVCache


class SlotPrefillOutput(NamedTuple):
    """一步槽位 Prefill 的产出：各请求首个 Token ID 列表、Prefill Cache 容器与模型耗时（ms）。"""

    next_tokens: list[int]
    prefill_cache: KVCache
    model_ms: float


class SlotDecodeOutput(NamedTuple):
    """一步槽位 Decode 的产出：按槽位索引的下一个 Token ID 列表，以及模型前向与输入准备耗时（ms）。"""

    slot_tokens: list[int]
    model_ms: float
    prepare_ms: float


def _prepare_decode_inputs(
    slot_cache: DenseSlotCache,
    running_slot_indices: Sequence[int],
    last_tokens: Sequence[int],
    positions: Sequence[int],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """构造 capacity 行的单步 Decode 输入（内部物理张量辅助函数）。

    返回 (input_ids, position_ids, attention_mask, 准备耗时 ms)。
    空闲/已结束槽位填 Padding 并整行 mask；先在 Host 侧组装列表再一次性传输，避免逐槽位的小 kernel。
    """
    if slot_cache.history_mask is None:
        raise RuntimeError("Cache 尚未分配，无法构造 Decode 输入")
    synchronize(slot_cache.device)
    start = time.perf_counter()
    token_ids = [pad_token_id] * slot_cache.capacity
    pos_list = [0] * slot_cache.capacity
    active = [False] * slot_cache.capacity
    for slot_idx, token_id, pos in zip(running_slot_indices, last_tokens, positions):
        token_ids[slot_idx] = token_id
        pos_list[slot_idx] = pos
        active[slot_idx] = True
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=slot_cache.device).unsqueeze(1)
    position_ids = torch.tensor(pos_list, dtype=torch.long, device=slot_cache.device).unsqueeze(1)
    active_mask = torch.tensor(active, dtype=torch.bool, device=slot_cache.device).unsqueeze(1)
    attention_mask = torch.cat((slot_cache.history_mask, active_mask), dim=1)
    synchronize(slot_cache.device)
    return input_ids, position_ids, attention_mask, (time.perf_counter() - start) * 1000


class SlotRunner(BaseModelRunner):
    """连续批处理槽位调度场景专职执行器：无状态，驱动 DenseSlotCache 完成 Prefill 与 Decode。"""

    def step_prefill(self, prompt_sequences: Sequence[Sequence[int]]) -> SlotPrefillOutput:
        """对一批新入场请求做左 Padding Prefill，返回 (各请求首个 Token 列表, Prefill Cache, 模型耗时 ms)。"""
        input_ids, attention_mask, position_ids, _ = _left_pad_sequences(
            prompt_sequences, self.pad_token_id, self.device
        )
        step_out = self._forward_and_sample(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, should_sync=True
        )
        next_tokens = step_out.next_token.squeeze(-1).tolist()
        return SlotPrefillOutput(
            next_tokens=next_tokens,
            prefill_cache=KVCache(step_out.present_key_values),
            model_ms=step_out.elapsed_ms,
        )

    def step_decode(
        self,
        dense_slot_cache: DenseSlotCache,
        running_slot_indices: Sequence[int],
        last_tokens: Sequence[int],
        positions: Sequence[int],
    ) -> SlotDecodeOutput:
        """对 DenseSlotCache 的全部槽位做一步 Decode，并把返回的新 KV 与本步 attention_mask 一起提交回槽位。"""
        input_ids, position_ids, attention_mask, prepare_ms = _prepare_decode_inputs(
            slot_cache=dense_slot_cache,
            running_slot_indices=running_slot_indices,
            last_tokens=last_tokens,
            positions=positions,
            pad_token_id=self.pad_token_id,
        )
        step_out = self._forward_and_sample(
            input_ids,
            past_key_values=dense_slot_cache.past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            should_sync=True,
        )
        dense_slot_cache.commit_decode_step(attention_mask, step_out.present_key_values)
        next_tokens = step_out.next_token.squeeze(-1).tolist()
        return SlotDecodeOutput(slot_tokens=next_tokens, model_ms=step_out.elapsed_ms, prepare_ms=prepare_ms)


__all__ = [
    "SlotDecodeOutput",
    "SlotPrefillOutput",
    "SlotRunner",
]
