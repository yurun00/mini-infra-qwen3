"""Attention backend 抽象（对标 vLLM AttentionBackend / AttentionImpl）。

Attention 核心计算（QK^T -> mask -> softmax -> PV）与 Q/K/V 投影、RoPE、KV Cache 寻址解耦为可替换的
attention backend：模型只负责产出 Q/K/V 与「因果 + padding」可见性描述符 CausalPaddingMask，
backend 负责将描述符预计算为本步跨层复用的后端专属输入（prepare_inputs，每次前向只做一次）并执行核心计算（forward，逐层调用）。

- eager：预计算加性 attn_bias（EagerAttentionInputs），显式物化 Score / Probability 矩阵的 baseline（EagerAttentionBackend）；
- flash：持有 CausalPaddingMask（FlashAttentionInputs），在 Triton Kernel 内分块 Online Softmax，不物化完整 Score（FlashAttentionBackend）。
"""

import abc
from typing import ClassVar, Generic, Literal, NamedTuple, TypeVar

import torch

AttentionBackendKind = Literal["eager", "flash"]
ATTENTION_BACKEND_KINDS: tuple[AttentionBackendKind, ...] = ("eager", "flash")


def parse_attention_backend_kind(value: str) -> AttentionBackendKind:
    """将字符串解析为合法的 AttentionBackendKind。"""
    if value == "eager":
        return "eager"
    if value == "flash":
        return "flash"
    raise ValueError(
        f"未知 attention backend: {value}，可选: {list(ATTENTION_BACKEND_KINDS)}"
    )


class CausalPaddingMask(NamedTuple):
    """单次前向的「因果 + padding」可见性描述符（backend 无关，全部 Tensor 已在 device 上）。

    key_valid:     [batch_size, kv_len],    bool, contiguous / 切片视图；False 为 padding / 未写满 / null block 位置
    query_valid:   [batch_size, query_len], bool, contiguous / 切片视图；False 为 padding Query 行（输出无意义）
    causal_offset: Query 第 0 行在 Key 序列中的位置偏移（int，由形状推导，零 Host-Device 同步）

    Query 行 q 可见 Key 位置 k 当且仅当 key_valid[b, k] 且 k <= causal_offset + q。
    """

    key_valid: torch.Tensor
    query_valid: torch.Tensor
    causal_offset: int


class AttentionInputs:
    """单次前向由 attention backend.prepare_inputs 预计算、供全部层共享的后端专属输入基类。"""


InputsT = TypeVar("InputsT", bound=AttentionInputs)


class AttentionBackend(abc.ABC, Generic[InputsT]):
    """无状态 attention backend 基类：prepare_inputs 每次前向调用一次，forward 每层调用一次。"""

    kind: ClassVar[AttentionBackendKind]

    @abc.abstractmethod
    def prepare_inputs(self, mask: CausalPaddingMask, dtype: torch.dtype) -> InputsT:
        """基于 CausalPaddingMask 预计算本 backend 单次前向跨层复用的专属输入。"""

    @abc.abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_inputs: InputsT,
        scaling: float,
    ) -> torch.Tensor:
        """执行 GQA Attention 核心计算，返回 o_proj 前的隐藏表示。

        query:       [batch_size, num_heads, query_len, head_dim],  model_dtype, contiguous
        key / value: [batch_size, num_kv_heads, kv_len, head_dim],  model_dtype, 末维 stride 为 1（可为 transpose 视图）
        Returns:     [batch_size, query_len, num_heads * head_dim], model_dtype, contiguous
        """


__all__ = [
    "ATTENTION_BACKEND_KINDS",
    "AttentionBackend",
    "AttentionBackendKind",
    "AttentionInputs",
    "CausalPaddingMask",
    "parse_attention_backend_kind",
]
