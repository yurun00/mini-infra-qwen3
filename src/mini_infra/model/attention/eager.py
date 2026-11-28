"""Eager attention backend：显式物化 Score 与 Probability 矩阵的 baseline。

数值路径与引入 attention backend 抽象前的 Qwen3Attention._attend 逐算子一致（加性 attn_bias、float32 Softmax），
保证既有 Baseline 的可运行性与可对比性。
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mini_infra.model.attention.backend import (
    AttentionBackend,
    AttentionInputs,
    CausalPaddingMask,
)


def _repeat_key_value(hidden_states: torch.Tensor, repeat_count: int) -> torch.Tensor:
    """把较少的 KV 头按 GQA 分组复制到与 Query 头数相同。

    hidden_states: [batch_size, num_kv_heads, kv_len, head_dim], model_dtype, contiguous / non-contiguous
    Returns:       [batch_size, num_heads, kv_len, head_dim],    model_dtype,
                   repeat_count == 1 时保持原视图，否则 contiguous
    """
    if repeat_count == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeat_count, dim=1)


def _build_attn_bias(mask: CausalPaddingMask, dtype: torch.dtype) -> torch.Tensor:
    """把可见性描述符物化为加性注意力偏置（可见位 0.0，屏蔽位 dtype 最小值）。

    Returns:
        attn_bias: [batch_size, 1, query_len, kv_len], dtype, contiguous（对齐 num_heads 广播维）
    """
    key_valid = mask.key_valid
    device = key_valid.device
    batch_size, kv_len = key_valid.shape
    query_len = mask.query_valid.shape[1]
    # query_positions: [query_len], key_positions: [kv_len]
    # -> causal_mask: [1, kv_len] <= [query_len, 1] -> [query_len, kv_len], bool, contiguous
    query_positions = torch.arange(
        mask.causal_offset, mask.causal_offset + query_len, device=device
    )
    key_positions = torch.arange(kv_len, device=device)
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    # causal_mask.view: [1, 1, query_len, kv_len] & key_valid.view: [batch_size, 1, 1, kv_len]
    # -> allowed_mask: [batch_size, 1, query_len, kv_len], bool, contiguous
    allowed_mask = causal_mask.view(1, 1, query_len, kv_len) & key_valid.view(
        batch_size, 1, 1, kv_len
    )
    attn_bias = torch.zeros(allowed_mask.shape, dtype=dtype, device=device)
    attn_bias.masked_fill_(~allowed_mask, torch.finfo(dtype).min)
    return attn_bias


@dataclass(frozen=True)
class EagerAttentionInputs(AttentionInputs):
    """eager attention backend 单次前向共享的预计算输入。

    attn_bias: [batch_size, 1, query_len, kv_len], model_dtype, contiguous
    """

    attn_bias: torch.Tensor


class EagerAttentionBackend(AttentionBackend[EagerAttentionInputs]):
    """显式 QK^T -> 加性 attn_bias -> float32 softmax -> PV。"""

    kind = "eager"

    def prepare_inputs(
        self, mask: CausalPaddingMask, dtype: torch.dtype
    ) -> EagerAttentionInputs:
        return EagerAttentionInputs(attn_bias=_build_attn_bias(mask, dtype))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_inputs: EagerAttentionInputs,
        scaling: float,
    ) -> torch.Tensor:
        """对完整历史 K/V 做 GQA Attention，返回 o_proj 前的隐藏表示。

        query:       [batch_size, num_heads, query_len, head_dim],  model_dtype, contiguous
        key / value: [batch_size, num_kv_heads, kv_len, head_dim],  model_dtype, contiguous / non-contiguous
        attn_inputs: EagerAttentionInputs（含加性 attn_bias [batch_size, 1, query_len, kv_len]）
        Returns:     [batch_size, query_len, num_heads * head_dim], model_dtype, contiguous
        """
        batch_size, num_heads, query_len, _ = query.shape

        # 只在真正计算 Attention 前按 GQA 组展开为 [batch_size, num_heads, kv_len, head_dim]，避免 Cache 存冗余副本。
        num_key_value_groups = num_heads // key.shape[1]
        attention_key = _repeat_key_value(key, num_key_value_groups)
        attention_value = _repeat_key_value(value, num_key_value_groups)

        # transpose(2, 3): [batch_size, num_heads, head_dim, kv_len]（non-contiguous 视图，直接由 matmul 消费）
        # -> attention_scores: [batch_size, num_heads, query_len, kv_len], model_dtype, contiguous
        attention_scores = torch.matmul(query, attention_key.transpose(2, 3)) * scaling
        attention_scores = attention_scores + attn_inputs.attn_bias

        # Softmax 显式升维 float32 归一化后再转回 model_dtype，防止低精度指数溢出。
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_probs = attention_probs.to(query.dtype)
        # matmul -> [batch_size, num_heads, query_len, head_dim], model_dtype, contiguous
        attention_output = torch.matmul(attention_probs, attention_value)

        # transpose(1, 2): [batch_size, query_len, num_heads, head_dim]（non-contiguous）
        # -> contiguous() -> view: [batch_size, query_len, num_heads * head_dim], model_dtype, contiguous
        attention_output = attention_output.transpose(1, 2).contiguous()
        return attention_output.view(batch_size, query_len, -1)


# 无状态单例：模型前向与 Runner 的默认 attention backend。
EAGER_ATTENTION_BACKEND = EagerAttentionBackend()

__all__ = [
    "EAGER_ATTENTION_BACKEND",
    "EagerAttentionBackend",
    "EagerAttentionInputs",
]
