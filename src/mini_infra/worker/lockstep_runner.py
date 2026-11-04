"""齐步（Lockstep）场景专职执行器与批次数据容器。

负责管理静态齐步批次（所有请求同时开始与推进）的 GPU 输入组装、单步推理与下一步输入迭代。
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from mini_infra.worker.base_runner import BaseModelRunner, _left_pad_sequences
from mini_infra.worker.kv_cache import KVCache


class LockstepOutput(NamedTuple):
    """齐步（Lockstep）单步推理产出：下一个 Token ID 列表、耗时、计时 Event 与可选 Logits。"""

    next_tokens: list[int]
    elapsed_ms: float = 0.0
    event_pair: tuple[Any, Any] | None = None
    last_logits: torch.Tensor | None = None


def _prepare_prefill_inputs(
    input_ids: torch.Tensor | Sequence[Any],
    token_sequences: list[torch.Tensor] | Sequence[Sequence[int]],
    has_uniform_prompt_len: bool,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构建 Prefill 阶段直接喂给模型的初始输入张量（内部物理张量辅助函数）。"""
    if has_uniform_prompt_len:
        if isinstance(input_ids, torch.Tensor):
            input_ids_tensor = (input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)).to(
                device=device, dtype=torch.long
            )
        else:
            input_ids_tensor = torch.as_tensor(input_ids, dtype=torch.long, device=device)
            if input_ids_tensor.ndim == 1:
                input_ids_tensor = input_ids_tensor.unsqueeze(0)
        batch_size, padded_prompt_length = input_ids_tensor.shape
        attention_mask = torch.ones((batch_size, padded_prompt_length), dtype=torch.bool, device=device)
        position_ids = (
            torch.arange(padded_prompt_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )
        prompt_lengths = torch.full((batch_size,), padded_prompt_length, dtype=torch.long, device=device)
        return input_ids_tensor, attention_mask, position_ids, prompt_lengths

    return _left_pad_sequences(token_sequences, pad_token_id, device)


@dataclass
class LockstepInputBatch:
    """管理一次齐步（Lockstep）生成期间 GPU 侧的输入张量与 KV Cache 容器。"""

    device: torch.device
    pad_token_id: int
    padded_prompt_length: int
    prompt_lengths: list[int]
    curr_input_ids: torch.Tensor
    curr_position_ids: torch.Tensor
    attention_mask: torch.Tensor
    full_input_ids: torch.Tensor | None = None
    full_position_ids: torch.Tensor | None = None
    kv_cache: KVCache | None = None

    def get_model_inputs(
        self, use_cache: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sequence[tuple[torch.Tensor, torch.Tensor]] | None]:
        """获取当前前向所需的 (input_ids, attention_mask, position_ids, past_key_values)。"""
        model_input = self.curr_input_ids if use_cache else self.full_input_ids
        position_ids = self.curr_position_ids if use_cache else self.full_position_ids
        past_kvs = self.kv_cache.past_key_values if (use_cache and self.kv_cache is not None) else None
        assert model_input is not None and position_ids is not None
        return model_input, self.attention_mask, position_ids, past_kvs

    def update_kv_cache(self, present_key_values: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """更新当前物理 KV Cache 显存池。"""
        self.kv_cache = KVCache(present_key_values)

    @property
    def cache_bytes(self) -> int:
        return self.kv_cache.total_bytes if self.kv_cache is not None else 0


class LockstepRunner(BaseModelRunner):
    """齐步（Lockstep）场景专职执行器：无状态，驱动 LockstepInputBatch 完成单步推理与输入张量迭代。"""

    def init_input_batch(
        self,
        input_ids: torch.Tensor | Sequence[Any],
        token_sequences: list[torch.Tensor] | Sequence[Sequence[int]],
        has_uniform_prompt_len: bool,
        use_cache: bool,
    ) -> LockstepInputBatch:
        """构建并初始化一个齐步批次的 GPU 输入张量容器与 Cache 状态。"""
        input_ids_tensor, attention_mask, position_ids, prompt_lengths = _prepare_prefill_inputs(
            input_ids, token_sequences, has_uniform_prompt_len, self.pad_token_id, self.device
        )
        return LockstepInputBatch(
            device=self.device,
            pad_token_id=self.pad_token_id,
            padded_prompt_length=input_ids_tensor.shape[1],
            prompt_lengths=prompt_lengths.tolist(),
            curr_input_ids=input_ids_tensor,
            curr_position_ids=position_ids,
            attention_mask=attention_mask,
            full_input_ids=input_ids_tensor if not use_cache else None,
            full_position_ids=position_ids if not use_cache else None,
            kv_cache=None,
        )

    def step(
        self,
        batch: LockstepInputBatch,
        capture_logits: bool = False,
        use_cache: bool = True,
    ) -> LockstepOutput:
        """执行单步齐步前向推理：读取 batch 张量、执行前向、就地更新 batch.kv_cache 并返回采样 Token。"""
        model_input, attention_mask, position_ids, past_kvs = batch.get_model_inputs(use_cache)
        step_out = self._forward_and_sample(
            model_input=model_input,
            past_key_values=past_kvs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            capture_logits=capture_logits,
            should_sync=False,
        )
        if use_cache:
            batch.update_kv_cache(step_out.present_key_values)
        next_tokens = step_out.next_token.squeeze(-1).tolist()
        return LockstepOutput(
            next_tokens=next_tokens,
            elapsed_ms=step_out.elapsed_ms,
            event_pair=step_out.event_pair,
            last_logits=step_out.last_logits,
        )

    def prepare_next_inputs(
        self,
        batch: LockstepInputBatch,
        curr_active: Sequence[bool],
        next_positions: Sequence[int],
        next_tokens: Sequence[int],
        use_cache: bool = True,
    ) -> None:
        """根据当前槽位活跃状态与生成结果，为齐步批次准备下一步的模型输入张量。"""
        active_tensor = torch.tensor(curr_active, dtype=torch.bool, device=self.device).unsqueeze(1)
        step_tokens = torch.tensor(next_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        pos_tensor = torch.tensor(next_positions, dtype=torch.long, device=self.device).unsqueeze(1)
        pad_tensor = torch.full_like(step_tokens, self.pad_token_id)
        zero_pos = torch.zeros_like(pos_tensor)

        masked_tokens = torch.where(active_tensor, step_tokens, pad_tensor)
        masked_pos = torch.where(active_tensor, pos_tensor, zero_pos)

        batch.attention_mask = torch.cat((batch.attention_mask, active_tensor), dim=1)
        if use_cache:
            batch.curr_input_ids = masked_tokens
            batch.curr_position_ids = masked_pos
        else:
            assert batch.full_input_ids is not None and batch.full_position_ids is not None
            batch.full_input_ids = torch.cat((batch.full_input_ids, masked_tokens), dim=1)
            batch.full_position_ids = torch.cat((batch.full_position_ids, masked_pos), dim=1)


__all__ = [
    "LockstepInputBatch",
    "LockstepOutput",
    "LockstepRunner",
]
