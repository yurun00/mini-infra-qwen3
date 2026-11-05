"""使用基础 PyTorch 实现的 Qwen3-0.6B，历史键值以纯张量形式进出。

模型层与显存管理层彻底解耦：只接受/返回 torch.Tensor，不认识任何 Cache 容器对象。
前向由 __call__ / forward 承接，按 attn_metadata 类型分派执行路径（对标 vLLM attn_metadata 分派）：
- attn_metadata 为 None 时执行全量上下文计算（Prefill 或 No-Cache baseline）；
- attn_metadata 为 DenseAttentionMetadata 时走 Dense KV Cache 增量路径（逐层历史 key/value 拼在序列维左侧，
  返回新拼接的 present_key_values）；
- attn_metadata 为 PagedAttentionMetadata 时走 Paged KV Cache Decode 路径（寻址 slot_mapping / block_tables /
  context_lens 与逐层 physical block pool（kv_pools）都封装在 metadata 内，新 K/V 就地写入 physical block pool，
  按 block table gather 历史后 attend）。纯张量进出，零外部容器依赖。
"""

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mini_infra.model.config import Qwen3Config

# 单层历史键值：(key, value)，Dense 路径形状为 [batch_size, num_kv_heads, past_len, head_dim]，
# Paged 路径下为 physical block pool 切片 [num_blocks, block_size, num_kv_heads, head_dim]。
PastKeyValue = tuple[torch.Tensor, torch.Tensor]


class PagedAttentionMetadata(NamedTuple):
    """Paged Decode 单步所需的物理寻址 + 逐层存储元数据（对标 vLLM attn_metadata）。

    slot_mapping:  [batch_size],             long, contiguous；本步新 Token K/V 写入的
                                             slot = physical_block_id * block_size + offset
    block_tables:  [batch_size, max_blocks], long, contiguous；各请求 logical block -> physical block 映射
                                             （max_blocks 为当前 batch 内最长请求已分配块数，右侧以 null block 补齐）
    context_lens:  [batch_size],             long, contiguous；各请求含本步新 Token 在内的有效上下文长度
    kv_pools:      逐层 (key_pool, value_pool)，各 [num_blocks, block_size, num_kv_heads, head_dim]，model_dtype，
                   contiguous；取料阶段暂留空，由 PagedRunner 在前向前用 _replace 组装为 physical block pool。
    """

    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    kv_pools: tuple[PastKeyValue, ...] = ()


class DenseAttentionMetadata(NamedTuple):
    """Dense KV Cache Decode 单步所需的逐层历史键值（按层排序，与 Qwen3Model.layers 对齐，
    对标 vLLM attn_metadata 的 dense 形态）。

    past_key_values: 逐层 (key, value)，各 [batch_size, num_kv_heads, past_len, head_dim]，model_dtype，contiguous
    """

    past_key_values: tuple[PastKeyValue, ...]


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
        """对最后一维执行均方根归一化（内部升维 float32 计算方差，再转回输入精度）。

        hidden_states: [..., hidden_dim], model_dtype, contiguous
        Returns:       [..., hidden_dim], model_dtype, contiguous
        """
        # 平方和求均值使用 float32，防止 bfloat16 / float16 精度不足。
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
        """计算指定位置索引的正弦与余弦编码（内部以 float32 计算角度后转回 output_dtype）。

        position_ids: [batch_size, query_len], long, contiguous / expand 视图
        output_dtype: 目标输出精度（model_dtype）

        Returns:
            (cosine, sine): 各 [batch_size, query_len, head_dim], output_dtype, contiguous
        """
        # unsqueeze(-1): [batch_size, query_len, 1] * [head_dim // 2]
        #   -> [batch_size, query_len, head_dim // 2], float32, contiguous
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq
        # cat: [batch_size, query_len, head_dim], float32, contiguous
        embedding = torch.cat((freqs, freqs), dim=-1)
        return embedding.cos().to(output_dtype), embedding.sin().to(output_dtype)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """把末维向量的前后两半交换，并对原来的后半部分取负。

    hidden_states: [batch_size, num_heads, query_len, head_dim], model_dtype, contiguous / non-contiguous
    Returns:       [batch_size, num_heads, query_len, head_dim], model_dtype, contiguous
    """
    half = hidden_states.shape[-1] // 2
    # 末维切片产出 [batch_size, num_heads, query_len, head_dim // 2] 非连续视图，经 torch.cat 重排为 contiguous。
    first_half = hidden_states[..., :half]
    second_half = hidden_states[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)


def _apply_rotary_embedding(
    query: torch.Tensor, key: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 RoPE 同时应用到 Query 和 Key。

    query:   [batch_size, num_heads, query_len, head_dim],    model_dtype, non-contiguous（transpose 视图）
    key:     [batch_size, num_kv_heads, query_len, head_dim], model_dtype, non-contiguous（transpose 视图）
    cosine:  [batch_size, query_len, head_dim],               model_dtype, contiguous
    sine:    [batch_size, query_len, head_dim],               model_dtype, contiguous

    Returns:
        (rotated_query, rotated_key): 形状分别同 query / key，model_dtype，contiguous
    """
    # unsqueeze(1): [batch_size, 1, query_len, head_dim]，对齐 head 维广播。
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    # 逐元素乘加结果落地为新分配的 contiguous 张量。
    rotated_query = query * cosine + _rotate_half(query) * sine
    rotated_key = key * cosine + _rotate_half(key) * sine
    return rotated_query, rotated_key


def _repeat_key_value(hidden_states: torch.Tensor, repeat_count: int) -> torch.Tensor:
    """把较少的 KV 头按 GQA 分组复制到与 Query 头数相同。

    hidden_states: [batch_size, num_kv_heads, kv_len, head_dim], model_dtype, contiguous / non-contiguous
    Returns:       [batch_size, num_heads, kv_len, head_dim],    model_dtype,
                   repeat_count == 1 时保持原视图，否则 contiguous
    """
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

    def _project_qkv(
        self, hidden_states: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """投影并归一化 Q/K/V，且对 Q/K 施加 RoPE。

        hidden_states: [batch_size, query_len, hidden_size], model_dtype, contiguous
        cosine / sine: [batch_size, query_len, head_dim],    model_dtype, contiguous

        Returns:
            query: [batch_size, num_heads, query_len, head_dim],    model_dtype, contiguous
            key:   [batch_size, num_kv_heads, query_len, head_dim], model_dtype, contiguous（已旋转，未按 GQA 组展开）
            value: [batch_size, num_kv_heads, query_len, head_dim], model_dtype, non-contiguous（transpose(1, 2) 视图）
        """
        batch_size, query_len, _ = hidden_states.shape

        # q_proj -> view: [batch_size, query_len, num_heads, head_dim], contiguous
        # -> q_norm -> transpose(1, 2): [batch_size, num_heads, query_len, head_dim], non-contiguous
        query = self.q_proj(hidden_states)
        query = query.view(batch_size, query_len, self.num_attention_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)

        # k_proj -> view: [batch_size, query_len, num_kv_heads, head_dim], contiguous
        # -> k_norm -> transpose(1, 2): [batch_size, num_kv_heads, query_len, head_dim], non-contiguous
        key = self.k_proj(hidden_states)
        key = key.view(batch_size, query_len, self.num_key_value_heads, self.head_dim)
        key = self.k_norm(key).transpose(1, 2)

        # v_proj -> view: [batch_size, query_len, num_kv_heads, head_dim], contiguous
        # -> transpose(1, 2): [batch_size, num_kv_heads, query_len, head_dim], non-contiguous
        value = self.v_proj(hidden_states)
        value = value.view(batch_size, query_len, self.num_key_value_heads, self.head_dim)
        value = value.transpose(1, 2)

        # _apply_rotary_embedding 逐元素计算后，query 与 key 均为 [batch_size, *, query_len, head_dim], contiguous。
        query, key = _apply_rotary_embedding(query, key, cosine, sine)
        return query, key, value

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """对完整历史 K/V 做 GQA Attention，返回 o_proj 前的隐藏表示。

        query:       [batch_size, num_heads, query_len, head_dim],    model_dtype, contiguous
        key / value: [batch_size, num_kv_heads, kv_len, head_dim],    model_dtype, contiguous / non-contiguous
        causal_mask: [batch_size, 1, query_len, kv_len],              model_dtype, contiguous

        Returns:
            attention_output: [batch_size, query_len, num_heads * head_dim], model_dtype, contiguous
        """
        batch_size, _, query_len, _ = query.shape

        # 只在真正计算 Attention 前按 GQA 组展开为 [batch_size, num_heads, kv_len, head_dim]，避免 Cache 存冗余副本。
        attention_key = _repeat_key_value(key, self.num_key_value_groups)
        attention_value = _repeat_key_value(value, self.num_key_value_groups)

        # transpose(2, 3): [batch_size, num_heads, head_dim, kv_len]（non-contiguous 视图，直接由 matmul 消费）
        # -> attention_scores: [batch_size, num_heads, query_len, kv_len], model_dtype, contiguous
        attention_scores = torch.matmul(query, attention_key.transpose(2, 3)) * self.scaling
        attention_scores = attention_scores + causal_mask

        # Softmax 显式升维 float32 归一化后再转回 model_dtype，防止低精度指数溢出。
        attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
        attention_probs = attention_probs.to(query.dtype)
        # matmul -> [batch_size, num_heads, query_len, head_dim], model_dtype, contiguous
        attention_output = torch.matmul(attention_probs, attention_value)

        # transpose(1, 2): [batch_size, query_len, num_heads, head_dim]（non-contiguous）
        # -> contiguous() -> view: [batch_size, query_len, num_heads * head_dim], model_dtype, contiguous
        attention_output = attention_output.transpose(1, 2).contiguous()
        return attention_output.view(batch_size, query_len, -1)

    def _apply_paged_kv_pool(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_pool: PastKeyValue,
        metadata: PagedAttentionMetadata,
    ) -> PastKeyValue:
        """单步 Paged Decode：将新 K/V 就地写入 physical block pool，并按 block_tables 收集历史 K/V 视图。

        key / value:  [batch_size, num_kv_heads, 1, head_dim], model_dtype, key 为 contiguous / value 为 transpose 视图
        kv_pool:      (key_pool, value_pool)，各 [num_blocks, block_size, num_kv_heads, head_dim]，model_dtype，
                      contiguous，就地更新
        metadata:     PagedAttentionMetadata，含 slot_mapping [batch_size] (long) 与
                      block_tables [batch_size, max_blocks] (long)

        Returns:
            (history_key, history_value): 形状均为 [batch_size, num_kv_heads, max_history_len, head_dim]，model_dtype，
            作为非连续视图供 _attend 的 matmul 消费。
        """
        batch_size = key.shape[0]
        key_pool, value_pool = kv_pool
        pool_shape = (-1, self.num_key_value_heads, self.head_dim)
        # view -> [num_blocks * block_size, num_kv_heads, head_dim], model_dtype, contiguous；
        # squeeze(2): [batch_size, num_kv_heads, 1, head_dim] -> [batch_size, num_kv_heads, head_dim], contiguous，
        # 按 slot 写入 physical block pool。
        key_pool.view(pool_shape).index_copy_(0, metadata.slot_mapping, key.squeeze(2))
        value_pool.view(pool_shape).index_copy_(0, metadata.slot_mapping, value.squeeze(2))

        # 按 block table 取块：[batch_size, max_blocks, block_size, num_kv_heads, head_dim]（gather 产出，contiguous）
        #   -> view 为 [batch_size, max_blocks * block_size, num_kv_heads, head_dim]（contiguous）
        #   -> transpose 为 [batch_size, num_kv_heads, max_history_len, head_dim]（非连续视图，直接喂 matmul）。
        max_history_len = metadata.block_tables.shape[1] * key_pool.shape[1]
        history_shape = (batch_size, max_history_len, self.num_key_value_heads, self.head_dim)
        history_key = key_pool[metadata.block_tables].view(history_shape).transpose(1, 2)
        history_value = value_pool[metadata.block_tables].view(history_shape).transpose(1, 2)
        return history_key, history_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        causal_mask: torch.Tensor,
        past_key_value: PastKeyValue | None = None,
        kv_pool: PastKeyValue | None = None,
        attn_metadata: PagedAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue]:
        """执行单层自注意力（按入参分派 No-Cache / Dense KV Cache / Paged KV Cache 路径）。

        hidden_states:  [batch_size, query_len, hidden_size], model_dtype, contiguous
        cosine / sine:  [batch_size, query_len, head_dim],    model_dtype, contiguous
        causal_mask:    [batch_size, 1, query_len, kv_len],   model_dtype, contiguous
        past_key_value: Dense 历史 (key, value)，各 [batch_size, num_kv_heads, past_len, head_dim]，
                        model_dtype，contiguous
        kv_pool:        Paged 单层 physical block pool (key_pool, value_pool)，
                        各 [num_blocks, block_size, num_kv_heads, head_dim]，model_dtype，contiguous

        Returns:
            attention_output:  [batch_size, query_len, hidden_size], model_dtype, contiguous
            present_key_value: Dense / No-Cache 下为 (key, value)（各 [batch_size, num_kv_heads, kv_len, head_dim]），
                               Paged 下原样返回 kv_pool 引用
        """
        query, key, value = self._project_qkv(hidden_states, cosine, sine)

        if attn_metadata is not None:
            assert kv_pool is not None
            key, value = self._apply_paged_kv_pool(key, value, kv_pool, attn_metadata)
            present_key_value = kv_pool
        elif past_key_value is not None:
            past_key, past_value = past_key_value
            # cat(dim=2): [batch_size, num_kv_heads, past_len + query_len, head_dim], model_dtype, contiguous
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
            present_key_value = (key, value)
        else:
            present_key_value = (key, value)

        attention_output = self.o_proj(self._attend(query, key, value, causal_mask))
        return attention_output, present_key_value


class Qwen3MLP(nn.Module):
    """Qwen3 的 SwiGLU 前馈网络。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 SwiGLU 前馈映射：down_proj(silu(gate_proj(x)) * up_proj(x))。

        hidden_states: [batch_size, query_len, hidden_size], model_dtype, contiguous
        Returns:       [batch_size, query_len, hidden_size], model_dtype, contiguous
        """
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen3DecoderLayer(nn.Module):
    """一个完整的 Qwen3 Decoder Layer（Pre-Norm + Self-Attention + Pre-Norm + SwiGLU MLP）。"""

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
        past_key_value: PastKeyValue | None = None,
        kv_pool: PastKeyValue | None = None,
        attn_metadata: PagedAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue]:
        """执行单层 Decoder 前向。

        hidden_states: [batch_size, query_len, hidden_size], model_dtype, contiguous
        cosine / sine: [batch_size, query_len, head_dim],    model_dtype, contiguous
        causal_mask:   [batch_size, 1, query_len, kv_len],   model_dtype, contiguous

        Returns:
            hidden_states:     [batch_size, query_len, hidden_size], model_dtype, contiguous
            present_key_value: 当前层更新后的 PastKeyValue
        """
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention_output, present_key_value = self.self_attn(
            normalized,
            cosine,
            sine,
            causal_mask,
            past_key_value=past_key_value,
            kv_pool=kv_pool,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + attention_output

        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(normalized)
        return hidden_states, present_key_value


class Qwen3Model(nn.Module):
    """由 Embedding、多个 Decoder Layer 和最终 Norm 组成的主干网络。"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def _validate_past_key_values(self, past_key_values: Sequence[PastKeyValue] | None, batch_size: int) -> int:
        """校验历史键值的层数、Batch 维、KV 头数、head_dim 以及各层长度一致。返回历史长度 past_len。"""
        if past_key_values is None:
            return 0
        if len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values 层数与模型层数不一致")
        past_len = past_key_values[0][0].shape[2]
        expected = (batch_size, self.config.num_key_value_heads, past_len, self.config.head_dim)
        for layer_idx, (key, value) in enumerate(past_key_values):
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError(
                    "第 %d 层 KV Cache 形状错误: key=%s, value=%s, expected=%s"
                    % (layer_idx, tuple(key.shape), tuple(value.shape), expected)
                )
        return past_len

    def _prepare_position_ids_and_allowed_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Sequence[PastKeyValue] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """校验 Prefill / Dense Decode 输入并构造位置索引与 4D 布尔可见掩码。

        input_ids:      [batch_size, query_len],            long, contiguous
        attention_mask: [batch_size, past_len + query_len], bool / int, contiguous（None 时默认全 True）
        position_ids:   [batch_size, query_len],            long, contiguous
                        （None 时默认 past_len .. past_len + query_len - 1）

        Returns:
            position_ids: [batch_size, query_len],         long, contiguous / expand 视图
            allowed_mask: [batch_size, 1, query_len, kv_len], bool, contiguous
        """
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids 必须是形状为 [batch, sequence] 的非空 Tensor")
        device = input_ids.device
        batch_size, query_len = input_ids.shape
        past_len = self._validate_past_key_values(past_key_values, batch_size)
        kv_len = past_len + query_len

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, kv_len), dtype=torch.bool, device=device)
        if tuple(attention_mask.shape) != (batch_size, kv_len):
            raise ValueError("attention_mask 应为 %s，实际为 %s" % ((batch_size, kv_len), tuple(attention_mask.shape)))
        attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        if position_ids is None:
            position_ids = torch.arange(past_len, kv_len, device=device)
            # unsqueeze(0).expand: [1, query_len] -> [batch_size, query_len], long, non-contiguous 广播视图
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        if tuple(position_ids.shape) != (batch_size, query_len):
            raise ValueError("position_ids 应为 %s，实际为 %s" % ((batch_size, query_len), tuple(position_ids.shape)))
        position_ids = position_ids.to(device=device, dtype=torch.long)

        # query_positions: [query_len], key_positions: [kv_len]
        # -> future: [1, kv_len] > [query_len, 1] -> [query_len, kv_len], bool, contiguous
        query_positions = torch.arange(past_len, kv_len, device=device)
        key_positions = torch.arange(kv_len, device=device)
        future = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        # (~future).unsqueeze(0): [1, query_len, kv_len] & attention_mask.unsqueeze(1): [batch_size, 1, kv_len]
        # -> allowed: [batch_size, query_len, kv_len], bool, contiguous
        allowed = (~future).unsqueeze(0) & attention_mask.unsqueeze(1)
        # unsqueeze(1): [batch_size, 1, query_len, kv_len], bool, contiguous（对齐 num_heads 广播维）
        return position_ids, allowed.unsqueeze(1)

    def _build_paged_inputs(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        metadata: PagedAttentionMetadata,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """校验 Paged Decode 输入并构造广播加性掩码。

        input_ids:    [batch_size, 1], long, contiguous / 切片视图
        position_ids: [batch_size, 1], long, contiguous / 切片视图
        metadata:     PagedAttentionMetadata（含 block_tables [batch_size, max_blocks] 与 context_lens [batch_size]）
        dtype:        model_dtype

        Returns:
            position_ids: [batch_size, 1],                    long,  保持输入视图
            causal_mask:  [batch_size, 1, 1, max_history_len], dtype, contiguous
        """
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("Paged Decode 的 input_ids 必须是形状为 [batch, 1] 的 Tensor")
        if position_ids.ndim != 2 or position_ids.shape[1] != 1:
            raise ValueError("Paged Decode 的 position_ids 必须是形状为 [batch, 1] 的 Tensor")
        kv_pools = metadata.kv_pools
        if len(kv_pools) != len(self.layers):
            raise ValueError("kv_pools 层数与模型层数不一致")

        # 加性掩码 [batch_size, 1, 1, max_history_len]（max_history_len = max_blocks * block_size）：
        # 仅前 context_lens[b] 个有效位置为 0.0，其余（尾块未写满位置、右侧补齐的 null block）填 dtype 最小值屏蔽。
        # block_tables: [batch_size, max_blocks], key_pool: [num_blocks, block_size, num_kv_heads, head_dim]
        max_history_len = metadata.block_tables.shape[1] * kv_pools[0][0].shape[1]
        # positions: [max_history_len], long, contiguous（0 .. max_history_len - 1）
        positions = torch.arange(max_history_len, device=input_ids.device)
        # visible: [1, max_history_len] < [batch_size, 1] -> [batch_size, max_history_len], bool, contiguous
        visible = positions.unsqueeze(0) < metadata.context_lens.unsqueeze(1)
        # causal_mask: [batch_size, max_history_len], model_dtype, contiguous
        causal_mask = torch.zeros(visible.shape, dtype=dtype, device=input_ids.device)
        causal_mask.masked_fill_(~visible, torch.finfo(dtype).min)
        # view -> [batch_size, 1, 1, max_history_len], model_dtype, contiguous
        # （对齐 _attend 中 attention_scores [batch_size, num_heads, 1, max_history_len] 的广播维度）
        causal_mask = causal_mask.view(visible.shape[0], 1, 1, max_history_len)
        return position_ids, causal_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attn_metadata: DenseAttentionMetadata | PagedAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, list[PastKeyValue]]:
        """执行主干网络前向，返回最终归一化隐藏状态与逐层键值状态。

        input_ids:      [batch_size, query_len],            long, contiguous / 切片视图
        attention_mask: [batch_size, past_len + query_len], bool / int, contiguous（Paged 下忽略）
        position_ids:   [batch_size, query_len],            long, contiguous / 切片视图
        attn_metadata:  None（Prefill / No-Cache）| DenseAttentionMetadata | PagedAttentionMetadata

        Returns:
            hidden_states:      [batch_size, query_len, hidden_size], model_dtype, contiguous
            present_key_values: 长度为 num_hidden_layers 的 PastKeyValue 列表
        """
        hidden_states = self.embed_tokens(input_ids)
        if isinstance(attn_metadata, PagedAttentionMetadata):
            if position_ids is None:
                raise ValueError("Paged Decode 必须提供 position_ids")
            position_ids, causal_mask = self._build_paged_inputs(
                input_ids, position_ids, attn_metadata, hidden_states.dtype
            )
        else:
            past_key_values = None if attn_metadata is None else attn_metadata.past_key_values
            position_ids, allowed = self._prepare_position_ids_and_allowed_mask(
                input_ids, attention_mask, position_ids, past_key_values
            )
            causal_mask = torch.zeros(allowed.shape, dtype=hidden_states.dtype, device=hidden_states.device)
            causal_mask.masked_fill_(~allowed, torch.finfo(hidden_states.dtype).min)

        cosine, sine = self.rotary_emb(position_ids, hidden_states.dtype)

        present_key_values: list[PastKeyValue] = []
        is_paged = isinstance(attn_metadata, PagedAttentionMetadata)
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_past = None
            layer_kv_pool = None
            if is_paged:
                layer_kv_pool = attn_metadata.kv_pools[layer_idx]
            elif attn_metadata is not None:
                layer_past = attn_metadata.past_key_values[layer_idx]
            hidden_states, present_key_value = decoder_layer(
                hidden_states,
                cosine,
                sine,
                causal_mask,
                past_key_value=layer_past,
                kv_pool=layer_kv_pool,
                attn_metadata=attn_metadata if is_paged else None,
            )
            present_key_values.append(present_key_value)

        return self.norm(hidden_states), present_key_values


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
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attn_metadata: DenseAttentionMetadata | PagedAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, list[PastKeyValue]]:
        """执行因果语言模型前向，返回词表 Logits 与逐层 present_key_values。

        input_ids:      [batch_size, query_len],            long, contiguous / 切片视图
        attention_mask: [batch_size, past_len + query_len], bool / int, contiguous（Paged 下忽略）
        position_ids:   [batch_size, query_len],            long, contiguous / 切片视图
        attn_metadata:  None（Prefill / No-Cache）| DenseAttentionMetadata | PagedAttentionMetadata

        Returns:
            logits:             [batch_size, query_len, vocab_size], model_dtype, contiguous
            present_key_values: 长度为 num_hidden_layers 的 PastKeyValue 列表
        """
        hidden_states, present_key_values = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
        )
        return self.lm_head(hidden_states), present_key_values


__all__ = [
    "DenseAttentionMetadata",
    "PagedAttentionMetadata",
    "PastKeyValue",
    "Qwen3Attention",
    "Qwen3DecoderLayer",
    "Qwen3ForCausalLM",
    "Qwen3MLP",
    "Qwen3Model",
    "RMSNorm",
    "RotaryEmbedding",
]
