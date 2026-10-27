"""使用基础 PyTorch 实现带显式 KVCache 容器的 Qwen3-0.6B。

本文件使用显式的 KVCache 容器对象管理历史键值缓存：
- 当 kv_cache 为 None 时，执行全量上下文计算（No-Cache baseline）；
- 当传入 kv_cache 对象时，Prefill 阶段自动建立 Cache，Decode 阶段自动增量更新。
模型直接返回 Logits 张量，彻底消除对 use_cache 布尔标记和繁琐 Tuple 解包的依赖。
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

from mini_infra.memory import KVCache
from mini_infra.model.config import DEFAULT_MODEL_REVISION, Qwen3Config, model_dtype_from_config


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)。

    与标准 LayerNorm 相比，RMSNorm 去除了均值中心化（re-centering）操作，
    只除以均方根并乘以可学习的缩放参数 weight，从而节省约 7% 的归一化计算开销。
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 平方和求均值使用 float32，防止 bfloat16 精度不足。
        input_dtype = hidden_states.dtype
        float_states = hidden_states.float()
        variance = float_states.pow(2).mean(dim=-1, keepdim=True)
        normalized = float_states * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (Rotary Position Embedding, RoPE)。

    通过将词向量拆分为二维子空间并施加复数旋转，将绝对位置信息以相对位置
    距离的形式编码进 Attention 的 Query 与 Key 中。
    """

    def __init__(self, head_dim: int, rope_theta: float):
        super().__init__()
        # 每两个维度共享一个旋转频率。inv_freq 不参与训练，所以注册为 buffer。
        dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (dimensions / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, output_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """计算指定位置索引的正弦与余弦编码。

        Args:
            position_ids: Token 位置索引张量，形状为 [batch_size, sequence_length]。
            output_dtype: 输出张量的数据类型（如 torch.bfloat16）。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - cosine: 余弦位置编码，形状为 [batch_size, sequence_length, head_dim]
                - sine:   正弦位置编码，形状为 [batch_size, sequence_length, head_dim]
        """
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq
        embedding = torch.cat((freqs, freqs), dim=-1)
        return embedding.cos().to(output_dtype), embedding.sin().to(output_dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """把向量的两半交换，并对原来的后半部分取负。"""
    half = hidden_states.shape[-1] // 2
    first_half = hidden_states[..., :half]
    second_half = hidden_states[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_embedding(
    query: torch.Tensor, key: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 RoPE 同时应用到 Query 和 Key。"""
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotated_query = query * cosine + rotate_half(query) * sine
    rotated_key = key * cosine + rotate_half(key) * sine
    return rotated_query, rotated_key


def repeat_key_value(hidden_states: torch.Tensor, repeat_count: int) -> torch.Tensor:
    """把较少的 KV 头复制到与 Query 头数相同。"""
    if repeat_count == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeat_count, dim=1)


class Qwen3Attention(nn.Module):
    """支持复用历史 K/V 的 Grouped Query Attention。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = config.head_dim**-0.5
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads

        query_size = config.num_attention_heads * config.head_dim
        key_value_size = config.num_key_value_heads * config.head_dim

        self.q_proj = nn.Linear(config.hidden_size, query_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, key_value_size, bias=False)
        self.o_proj = nn.Linear(query_size, config.hidden_size, bias=False)

        # Qwen3 会在每个 attention head 内单独归一化 Query 和 Key。
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        causal_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        query = query.view(batch_size, query_length, self.num_attention_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)

        key = self.k_proj(hidden_states)
        key = key.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        key = self.k_norm(key).transpose(1, 2)

        value = self.v_proj(hidden_states)
        value = value.view(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        value = value.transpose(1, 2)

        # Cache 中保存已经完成位置旋转、但尚未按 GQA 组展开的 K。
        query, key = apply_rotary_embedding(query, key, cosine, sine)
        if kv_cache is not None:
            key, value = kv_cache.update(key, value, layer_index)

        # 只在真正计算 Attention 前展开，避免 Cache 占用不必要的显存。
        attention_key = repeat_key_value(key, self.num_key_value_groups)
        attention_value = repeat_key_value(value, self.num_key_value_groups)

        attention_scores = torch.matmul(query, attention_key.transpose(2, 3)) * self.scaling
        attention_scores = attention_scores + causal_mask

        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_probs = attention_probs.to(query.dtype)
        attention_output = torch.matmul(attention_probs, attention_value)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, query_length, -1)
        return self.o_proj(attention_output)


class Qwen3MLP(nn.Module):
    """Qwen3 的 SwiGLU 前馈网络。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen3DecoderLayer(nn.Module):
    """一个完整的 Qwen3 Decoder Layer。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        causal_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_index: int = 0,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(
            normalized,
            cosine,
            sine,
            causal_mask,
            kv_cache=kv_cache,
            layer_index=layer_index,
        )
        hidden_states = residual + attention_output

        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normalized)
        return hidden_states


class Qwen3Model(nn.Module):
    """由 Embedding、多个 Decoder Layer 和最终 Norm 组成的主干网络。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def validate_kv_cache(self, kv_cache: KVCache | None, batch_size: int) -> int:
        """校验 Cache 层数、Batch 维、KV 头数、head_dim 以及各层长度一致。返回历史长度。"""
        if kv_cache is None or len(kv_cache) == 0:
            return 0
        if len(kv_cache) != len(self.layers):
            raise ValueError("KV Cache 层数与模型层数不一致")
        past_length = kv_cache.get_seq_length()
        expected = (batch_size, self.config.num_key_value_heads, past_length, self.config.head_dim)
        for layer_index, (key, value) in enumerate(kv_cache):
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(
                    "第 %d 层 KV Cache 形状错误: key=%s, value=%s, expected=%s"
                    % (layer_index, tuple(key.shape), tuple(value.shape), expected)
                )
        return past_length

    def prepare_position_id_and_allowed_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        kv_cache: KVCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids 必须是形状为 [batch, sequence] 的非空 Tensor")
        device = input_ids.device
        batch_size, query_length = input_ids.shape
        past_length = self.validate_kv_cache(kv_cache, batch_size)
        key_value_length = past_length + query_length

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, key_value_length), dtype=torch.bool, device=device)
        if tuple(attention_mask.shape) != (batch_size, key_value_length):
            raise ValueError(
                "attention_mask 应为 %s，实际为 %s" % ((batch_size, key_value_length), tuple(attention_mask.shape))
            )
        attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        if position_ids is None:
            position_ids = torch.arange(past_length, key_value_length, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        if tuple(position_ids.shape) != (batch_size, query_length):
            raise ValueError(
                "position_ids 应为 %s，实际为 %s" % ((batch_size, query_length), tuple(position_ids.shape))
            )
        position_ids = position_ids.to(device=device, dtype=torch.long)

        query_positions = torch.arange(past_length, key_value_length, device=device)
        key_positions = torch.arange(key_value_length, device=device)
        future = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        allowed = (~future).unsqueeze(0) & attention_mask.unsqueeze(1)
        return position_ids, allowed.unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        position_ids, allowed = self.prepare_position_id_and_allowed_mask(
            input_ids, attention_mask, position_ids, kv_cache
        )
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        causal_mask = torch.zeros(allowed.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        causal_mask.masked_fill_(~allowed, torch.finfo(hidden_states.dtype).min)

        for layer_index, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                cosine,
                sine,
                causal_mask,
                kv_cache=kv_cache,
                layer_index=layer_index,
            )

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 主干网络加上把隐藏状态映射为词表 Logits 的输出层。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.lm_head(hidden_states)


def resolve_model_directory(model_name_or_path: str, revision: str = DEFAULT_MODEL_REVISION) -> str:
    """返回本地模型目录；模型不存在时从 Hugging Face 下载固定版本。"""
    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())

    return snapshot_download(
        repo_id=model_name_or_path,
        revision=revision,
        max_workers=2,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ],
    )


def load_model(model_directory: str | Path, device: torch.device, dtype: torch.dtype | None = None) -> Qwen3ForCausalLM:
    """从 safetensors 权重构造 Qwen3 模型。"""
    config = Qwen3Config.from_model_directory(model_directory)
    model = Qwen3ForCausalLM(config)

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    state_dict = load_file(str(weight_path), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)

    allowed_missing_keys = {"lm_head.weight"}
    unexpected_missing = set(load_result.missing_keys) - allowed_missing_keys
    if unexpected_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "权重名称不匹配，missing="
            + str(sorted(unexpected_missing))
            + ", unexpected="
            + str(sorted(load_result.unexpected_keys))
        )

    model.tie_weights()
    model_dtype = model_dtype_from_config(config) if dtype is None else dtype
    if model_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("不支持的模型 dtype: " + str(model_dtype))

    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    model.eval()
    return model
