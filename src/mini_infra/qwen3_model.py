"""使用基础 PyTorch 实现带显式 KVCache 容器的 Qwen3-0.6B。

本文件使用显式的 KVCache 容器对象管理历史键值缓存：
- 当 kv_cache 为 None 时，执行全量上下文计算（No-Cache baseline）；
- 当传入 kv_cache 对象时，Prefill 阶段自动建立 Cache，Decode 阶段自动增量更新。
模型直接返回 Logits 张量，彻底消除对 use_cache 布尔标记和繁琐 Tuple 解包的依赖。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass
class Qwen3Config:
    """本期手写实现需要的模型配置。"""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    torch_dtype: str
    tie_word_embeddings: bool

    @classmethod
    def from_model_directory(cls, model_directory):
        config_path = Path(model_directory) / "config.json"
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = json.load(file)

        config = cls(
            vocab_size=raw_config["vocab_size"],
            hidden_size=raw_config["hidden_size"],
            intermediate_size=raw_config["intermediate_size"],
            num_hidden_layers=raw_config["num_hidden_layers"],
            num_attention_heads=raw_config["num_attention_heads"],
            num_key_value_heads=raw_config["num_key_value_heads"],
            head_dim=raw_config["head_dim"],
            rms_norm_eps=raw_config["rms_norm_eps"],
            rope_theta=raw_config["rope_theta"],
            torch_dtype=raw_config.get("torch_dtype", "bfloat16"),
            tie_word_embeddings=raw_config.get("tie_word_embeddings", True),
        )
        config.validate()
        return config

    def validate(self):
        """尽早拒绝本期尚未覆盖的模型变体，避免静默地产生错误结果。"""

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("Attention 头数必须能被 KV 头数整除")
        if not self.tie_word_embeddings:
            raise ValueError("本期实现只支持共享 Embedding 和 LM Head 权重的模型")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)。

    与标准 LayerNorm 相比，RMSNorm 去除了均值中心化（re-centering）操作，
    只除以均方根并乘以可学习的缩放参数 weight，从而节省约 7% 的归一化计算开销。
    """

    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
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

    def __init__(self, head_dim, rope_theta):
        super().__init__()

        # 每两个维度共享一个旋转频率。inv_freq 不参与训练，所以注册为 buffer。
        dimensions = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (rope_theta ** (dimensions / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, output_dtype):
        """计算指定位置索引的正弦与余弦编码。

        Args:
            position_ids: Token 位置索引张量，形状为 [batch_size, sequence_length]。
            output_dtype: 输出张量的数据类型（如 torch.bfloat16）。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - cosine: 余弦位置编码，形状为 [batch_size, sequence_length, head_dim]
                - sine:   正弦位置编码，形状为 [batch_size, sequence_length, head_dim]
        """
        # freqs 形状: [batch_size, sequence_length, head_dim // 2]
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq

        # Qwen3 将同一组频率分别应用到向量的前半部分和后半部分（与 rotate_half 对应）。
        # embedding, cosine 与 sine 形状均为: [batch_size, sequence_length, head_dim]
        embedding = torch.cat((freqs, freqs), dim=-1)
        return embedding.cos().to(output_dtype), embedding.sin().to(output_dtype)


def rotate_half(hidden_states):
    """把向量的两半交换，并对原来的后半部分取负。"""

    half = hidden_states.shape[-1] // 2
    first_half = hidden_states[..., :half]
    second_half = hidden_states[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_embedding(query, key, cosine, sine):
    """把 RoPE 同时应用到 Query 和 Key。

    Args:
        query: Query 张量，形状为 [batch_size, num_attention_heads, sequence_length, head_dim]。
        key: Key 张量，形状为 [batch_size, num_key_value_heads, sequence_length, head_dim]。
        cosine: 余弦位置编码，形状为 [batch_size, sequence_length, head_dim]。
        sine: 正弦位置编码，形状为 [batch_size, sequence_length, head_dim]。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - rotated_query: 旋转后的 Query，形状为 [batch_size, num_attention_heads, sequence_length, head_dim]。
            - rotated_key: 旋转后的 Key，形状为 [batch_size, num_key_value_heads, sequence_length, head_dim]。
    """
    # cosine / sine 原形状: [batch_size, sequence_length, head_dim]
    # 在第 1 维 (head 维度) 插入一个维度，变为 [batch_size, 1, sequence_length, head_dim]
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)

    # query 与 rotate_half(query) 形状均为 [batch_size, num_attention_heads, sequence_length, head_dim]
    # cosine / sine 在第 1 维 (head 维) 广播，运算后形状保持:
    # [batch_size, num_attention_heads, sequence_length, head_dim]
    rotated_query = query * cosine + rotate_half(query) * sine

    # key 与 rotate_half(key) 形状均为 [batch_size, num_key_value_heads, sequence_length, head_dim]
    # 同样在 head 维广播，运算后形状保持:
    # [batch_size, num_key_value_heads, sequence_length, head_dim]
    rotated_key = key * cosine + rotate_half(key) * sine

    return rotated_query, rotated_key


def repeat_key_value(hidden_states, repeat_count):
    """把较少的 KV 头复制到与 Query 头数相同。"""

    if repeat_count == 1:
        return hidden_states
    # 这里选择直观的 repeat_interleave；Paged Attention 等优化留到后续章节。
    return hidden_states.repeat_interleave(repeat_count, dim=1)


class KVCache:
    """动态 KV Cache 显式容器。

    在模型前向传播中用于累积并维护历史 Token 的 Key 和 Value 张量：
    - Prefill 阶段：模型将各层的初始 Key/Value 张量写入容器中；
    - Decode 阶段：每一步单 Token 推理时，将新计算的 Key/Value 原地追加到历史缓存中。
    """

    def __init__(self):
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_index: int):
        """将当前层新生成的 key_states 与 value_states 拼入缓存，并返回拼接后的全量张量。

        Args:
            key_states: [batch_size, num_kv_heads, seq_len, head_dim]
            value_states: [batch_size, num_kv_heads, seq_len, head_dim]
            layer_index: 解码层索引 (0 <= layer_index < num_layers)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 当前层拼接历史后的全量 (key, value)
        """
        if len(self.key_cache) <= layer_index:
            # Prefill 阶段：首次填充当前层
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # Decode 阶段：在序列维度（dim=2）追加新计算的 Key/Value
            self.key_cache[layer_index] = torch.cat([self.key_cache[layer_index], key_states], dim=2)
            self.value_cache[layer_index] = torch.cat([self.value_cache[layer_index], value_states], dim=2)
        return self.key_cache[layer_index], self.value_cache[layer_index]

    def get_seq_length(self, layer_index: int = 0) -> int:
        """获取已缓存的历史序列长度。"""
        if not self.key_cache or len(self.key_cache) <= layer_index:
            return 0
        return self.key_cache[layer_index].shape[2]

    def total_bytes(self) -> int:
        """计算当前所有层 Key/Value 张量实际占用的内存/显存字节数。"""
        return sum(
            k.numel() * k.element_size() + v.numel() * v.element_size()
            for k, v in zip(self.key_cache, self.value_cache)
        )

    def __len__(self) -> int:
        """已初始化的层数。"""
        return len(self.key_cache)

    def __iter__(self):
        """允许直接对每层的 (key, value) 进行遍历迭代。"""
        return iter(zip(self.key_cache, self.value_cache))

    def __getitem__(self, layer_index: int):
        """获取指定层的 (key, value)。"""
        return self.key_cache[layer_index], self.value_cache[layer_index]


def cache_size_bytes(kv_cache):
    """返回 KV Cache 容器实际承载的数据字节数。"""
    if kv_cache is None:
        return 0
    if isinstance(kv_cache, KVCache):
        return kv_cache.total_bytes()
    return sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in kv_cache)


class Qwen3Attention(nn.Module):
    """支持复用历史 K/V 的 Grouped Query Attention。"""

    def __init__(self, config):
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
        hidden_states,
        cosine,
        sine,
        causal_mask,
        kv_cache=None,
        layer_index=0,
    ):
        batch_size, query_length, _ = hidden_states.shape

        # 投影后拆分 head，形状变为 [batch, heads, query_length, head_dim]。
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

        # causal_mask 保证当前位置只能看到自己和之前的 Token。
        attention_scores = torch.matmul(query, attention_key.transpose(2, 3)) * self.scaling
        attention_scores = attention_scores + causal_mask

        # Softmax 使用 float32 计算，再转回模型 dtype，与参考实现保持一致。
        # 1. Softmax 权重 @ Value 矩阵相乘
        # attention_probs:  [batch_size, num_heads, query_length, key_value_length]
        # attention_value:  [batch_size, num_heads, key_value_length, head_dim]
        # attention_output: [batch_size, num_heads, query_length, head_dim]
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_probs = attention_probs.to(query.dtype)
        attention_output = torch.matmul(attention_probs, attention_value)

        # contiguous() 在显存中重新分配一块内存，对不连续的内存进行拷贝和重排，确保内存是连续的，view() 才能正常工作
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, query_length, -1)
        return self.o_proj(attention_output)


class Qwen3MLP(nn.Module):
    """Qwen3 的 SwiGLU 前馈网络。"""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen3DecoderLayer(nn.Module):
    """一个完整的 Qwen3 Decoder Layer。"""

    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        cosine,
        sine,
        causal_mask,
        kv_cache=None,
        layer_index=0,
    ):
        # Attention 子层：Pre-Norm -> Attention -> Residual。
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

        # MLP 子层：Pre-Norm -> MLP -> Residual。
        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normalized)
        return hidden_states


class Qwen3Model(nn.Module):
    """由 Embedding、多个 Decoder Layer 和最终 Norm 组成的主干网络。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def make_causal_mask(self, query_length, key_value_length, past_length, dtype, device):
        """创建适用于 Prefill 和带历史 Cache Decode 的因果 Mask。

        在 Prefill 阶段，query_length == key_value_length，生成下三角为 0、上三角为 -inf 的因果掩码；
        在 Decode 阶段，query_length 为 1，Key 包含历史 Token，最新 Token 能看到历史全部位置。
        """
        query_positions = torch.arange(past_length, past_length + query_length, device=device)
        key_positions = torch.arange(key_value_length, device=device)
        future_positions = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        mask = torch.zeros((query_length, key_value_length), dtype=dtype, device=device)
        mask.masked_fill_(future_positions, torch.finfo(dtype).min)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, kv_cache=None):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids 必须是形状为 [batch, sequence] 的非空 Tensor")

        hidden_states = self.embed_tokens(input_ids)
        batch_size, query_length = input_ids.shape
        past_length = kv_cache.get_seq_length() if kv_cache is not None else 0
        key_value_length = past_length + query_length

        position_ids = torch.arange(past_length, key_value_length, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)
        causal_mask = self.make_causal_mask(
            query_length,
            key_value_length,
            past_length,
            hidden_states.dtype,
            hidden_states.device,
        )

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

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        # Qwen3-0.6B 的输入 Embedding 与输出 LM Head 使用同一份权重。
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, kv_cache=None):
        hidden_states = self.model(input_ids, kv_cache=kv_cache)
        return self.lm_head(hidden_states)


def resolve_model_directory(model_name_or_path, revision=DEFAULT_MODEL_REVISION):
    """返回本地模型目录；模型不存在时从 Hugging Face 下载固定版本。"""

    local_path = Path(model_name_or_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())

    return snapshot_download(
        repo_id=model_name_or_path,
        revision=revision,
        # 测试服务器到 Hugging Face 的连接偶尔会被重置。降低并发数可以
        # 提高首次下载的稳定性，且不会影响模型加载和推理性能。
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


def model_dtype_from_config(config):
    """把配置文件中的 dtype 名称转换为 PyTorch dtype。"""

    supported_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if config.torch_dtype not in supported_dtypes:
        raise ValueError("不支持的模型 dtype: " + config.torch_dtype)
    return supported_dtypes[config.torch_dtype]


def load_handwritten_model(model_directory, device, dtype=None):
    """从 safetensors 权重构造手写 Qwen3 模型。"""

    config = Qwen3Config.from_model_directory(model_directory)
    model = Qwen3ForCausalLM(config)

    weight_path = Path(model_directory) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("没有找到模型权重: " + str(weight_path))

    state_dict = load_file(str(weight_path), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)

    # 共享权重只在 safetensors 中保存一份，所以 lm_head.weight 允许缺失。
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

    # model.to(dtype=...) 会连 float buffer 一起转换。RoPE 的频率必须保留
    # float32 精度，因此先保存一份，再在模型移动完成后放回目标设备。
    rotary_inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(device=device, dtype=model_dtype)
    model.model.rotary_emb.inv_freq = rotary_inv_freq.to(device)
    model.eval()
    return model
