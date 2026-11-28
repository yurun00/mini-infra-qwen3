"""Qwen3 模型纯张量算子架构。"""

from mini_infra.model.attention import (
    ATTENTION_BACKEND_KINDS,
    AttentionBackend,
    AttentionBackendKind,
    AttentionInputs,
    CausalPaddingMask,
    get_attn_backend,
    parse_attention_backend_kind,
)
from mini_infra.model.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    Qwen3Config,
    model_dtype_from_config,
)
from mini_infra.model.loader import (
    load_model,
    resolve_model_directory,
)
from mini_infra.model.qwen3 import (
    DenseAttentionMetadata,
    PagedAttentionMetadata,
    PastKeyValue,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    RMSNorm,
    RotaryEmbedding,
)

__all__ = [
    "ATTENTION_BACKEND_KINDS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODEL_REVISION",
    "AttentionBackend",
    "AttentionBackendKind",
    "AttentionInputs",
    "CausalPaddingMask",
    "DenseAttentionMetadata",
    "PagedAttentionMetadata",
    "PastKeyValue",
    "Qwen3Attention",
    "Qwen3Config",
    "Qwen3DecoderLayer",
    "Qwen3ForCausalLM",
    "Qwen3MLP",
    "Qwen3Model",
    "RMSNorm",
    "RotaryEmbedding",
    "get_attn_backend",
    "load_model",
    "model_dtype_from_config",
    "parse_attention_backend_kind",
    "resolve_model_directory",
]
