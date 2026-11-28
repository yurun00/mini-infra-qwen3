"""Attention backend 注册与选择（对标 vLLM get_attn_backend）。

flash backend 依赖 Triton：其模块仅在 get_attn_backend("flash") 时延迟导入，
使 CPU-only 环境（未安装 Triton）仍可加载模型与 eager 路径。
"""

from mini_infra.model.attention.backend import (
    ATTENTION_BACKEND_KINDS,
    AttentionBackend,
    AttentionBackendKind,
    AttentionInputs,
    CausalPaddingMask,
    parse_attention_backend_kind,
)
from mini_infra.model.attention.eager import (
    EAGER_ATTENTION_BACKEND,
    EagerAttentionBackend,
    EagerAttentionInputs,
)


def get_attn_backend(kind: AttentionBackendKind) -> AttentionBackend:
    """按 AttentionBackendKind 返回无状态 attention backend 实例。"""
    if kind == "eager":
        return EAGER_ATTENTION_BACKEND
    from mini_infra.model.attention.flash_attention import FlashAttentionBackend

    return FlashAttentionBackend()


__all__ = [
    "ATTENTION_BACKEND_KINDS",
    "EAGER_ATTENTION_BACKEND",
    "AttentionBackend",
    "AttentionBackendKind",
    "AttentionInputs",
    "CausalPaddingMask",
    "EagerAttentionBackend",
    "EagerAttentionInputs",
    "get_attn_backend",
    "parse_attention_backend_kind",
]
