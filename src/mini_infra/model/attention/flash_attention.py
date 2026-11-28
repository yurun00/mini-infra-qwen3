"""Flash attention backend：Triton 手写 FlashAttention 前向 Kernel（inference-only）。

每个 program 负责一个 (batch, query head, BLOCK_M 个 Query 行) Tile，按 BLOCK_N 分块流式读取 K/V，
以 Online Softmax 维护每行的最大值 m、归一化和 l 与输出累加 acc（全部 float32）：

    m_new = max(m, rowmax(S_tile))
    alpha = exp(m - m_new)
    P     = exp(S_tile - m_new)
    l     = alpha * l + rowsum(P)
    acc   = alpha * acc + P @ V_tile
    O     = acc / l

完整 Score / Probability 矩阵从不写入 GPU 全局显存：计算量仍为二次方，降低的是中间显存规模与全局显存读写量。

相对 eager baseline 的结构性差异（对标 vLLM FLASH_ATTN backend）：
- 原生 GQA：Query head h 直接读取 KV head h // num_kv_groups，不做 repeat_interleave 显存展开；
- 任意 stride：K/V 只要求末维连续，Paged Decode 按 block table gather 出的 transpose 视图可直接消费，零 contiguous() 拷贝；
- 因果 Tile 跳过：Query Tile 只遍历 k <= causal_offset + 本 Tile 末行 的 Key 分块（Full Prefill 约省一半计算）；
- 输出按 [batch, query_len, num_heads, head_dim] 布局直接写出，o_proj 前无需 transpose + contiguous 拷贝。

实现边界：CUDA；bfloat16 / float16 / float32（float32 走 IEEE 精度 tl.dot，供 float32 Parity）；head_dim 为不小于 16
的 2 的幂；forward-only；任意 batch / head / GQA 分组；Full Prefill、Q < K 的 Chunked Prefill 与 Decode；
因果 Mask、尾 Tile Mask 与逐序列 padding Mask（key_valid / query_valid）。
不含 backward、dropout、任意稠密 Mask、split-KV（Flash-Decoding）与直接按 block table 读取 physical block 的 Paged Kernel。
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from mini_infra.model.attention.backend import (
    AttentionBackend,
    AttentionInputs,
    CausalPaddingMask,
)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _flash_attention_forward_kernel(
    query,
    key,
    value,
    key_valid,
    query_valid,
    output,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_key_valid_b,
    stride_query_valid_b,
    stride_ob,
    stride_om,
    stride_oh,
    query_len,
    kv_len,
    causal_offset,
    softmax_scale_log2,
    NUM_HEADS: tl.constexpr,
    NUM_KV_GROUPS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    query_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1)
    batch_idx = batch_head_idx // NUM_HEADS
    head_idx = batch_head_idx % NUM_HEADS
    # 原生 GQA：同组 Query head 共享同一个 KV head，不展开 K/V。
    kv_head_idx = head_idx // NUM_KV_GROUPS

    query_offsets = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    # is_query_in_bounds: [BLOCK_M], bool (int1)；标记本 Tile 内未超出 query_len 尾边界的行。
    is_query_in_bounds = query_offsets < query_len

    # query_tile: [BLOCK_M, HEAD_DIM], model_dtype；整个 Key 循环内常驻寄存器 / 共享内存。
    query_tile = tl.load(
        query
        + batch_idx * stride_qb
        + head_idx * stride_qh
        + query_offsets[:, None] * stride_qm
        + dim_offsets[None, :],
        mask=is_query_in_bounds[:, None],
        other=0.0,
    )

    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    key_base = key + batch_idx * stride_kb + kv_head_idx * stride_kh
    value_base = value + batch_idx * stride_vb + kv_head_idx * stride_vh
    key_valid_base = key_valid + batch_idx * stride_key_valid_b

    # 因果 Tile 跳过：本 Tile 末行的全局位置为 causal_offset + (query_block_idx + 1) * BLOCK_M - 1，
    # 其后的 Key 分块对本 Tile 全部不可见，直接不读。
    key_end = tl.minimum(kv_len, causal_offset + (query_block_idx + 1) * BLOCK_M)
    for key_start in range(0, key_end, BLOCK_N):
        key_offsets = key_start + tl.arange(0, BLOCK_N)
        is_key_in_bounds = key_offsets < kv_len
        is_key_valid = tl.load(
            key_valid_base + key_offsets, mask=is_key_in_bounds, other=0
        ).to(tl.int1)
        # key_tile 以转置形态加载：[HEAD_DIM, BLOCK_N]，供 tl.dot(query_tile, key_tile) 直接得到 Score Tile。
        key_tile = tl.load(
            key_base + key_offsets[None, :] * stride_kn + dim_offsets[:, None],
            mask=is_key_in_bounds[None, :],
            other=0.0,
        )
        # scores: [BLOCK_M, BLOCK_N], float32；预乘 log2(e) 以便用 exp2 代替 exp。
        scores = (
            tl.dot(query_tile, key_tile, input_precision=INPUT_PRECISION)
            * softmax_scale_log2
        )
        is_allowed = is_key_valid[None, :] & (
            key_offsets[None, :] <= causal_offset + query_offsets[:, None]
        )
        scores = tl.where(is_allowed, scores, float("-inf"))

        new_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
        # 某行至今全部被屏蔽（如左 padding 覆盖整个 Key 分块）时 new_row_max 为 -inf：
        # 以 0 作指数基准，避免 (-inf) - (-inf) = NaN；此时 P 全为 0，alpha 作用于全零状态，结果不变。
        safe_row_max = tl.where(new_row_max == float("-inf"), 0.0, new_row_max)
        correction = tl.exp2(row_max - safe_row_max)
        probabilities = tl.exp2(scores - safe_row_max[:, None])
        row_sum = row_sum * correction + tl.sum(probabilities, axis=1)

        # value_tile: [BLOCK_N, HEAD_DIM], model_dtype
        value_tile = tl.load(
            value_base + key_offsets[:, None] * stride_vn + dim_offsets[None, :],
            mask=is_key_in_bounds[:, None],
            other=0.0,
        )
        accumulator = accumulator * correction[:, None] + tl.dot(
            probabilities.to(value_tile.dtype),
            value_tile,
            input_precision=INPUT_PRECISION,
        )
        row_max = new_row_max

    is_query_valid = tl.load(
        query_valid + batch_idx * stride_query_valid_b + query_offsets,
        mask=is_query_in_bounds,
        other=0,
    ).to(tl.int1)
    # padding Query 行（或无任何可见 Key 的行）row_sum 为 0：输出置 0，避免 0 / 0。
    safe_row_sum = tl.where(row_sum == 0.0, 1.0, row_sum)
    normalized = accumulator / safe_row_sum[:, None]
    normalized = tl.where(is_query_valid[:, None], normalized, 0.0)
    tl.store(
        output
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_offsets[:, None] * stride_om
        + dim_offsets[None, :],
        normalized.to(output.dtype.element_ty),
        mask=is_query_in_bounds[:, None],
    )


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_valid: torch.Tensor,
    query_valid: torch.Tensor,
    causal_offset: int,
) -> None:
    """算子 API 边界的形状 / dtype / 布局校验（纯 Python 元数据检查，零 Host-Device 同步）。"""
    if not query.is_cuda:
        raise RuntimeError("FlashAttention backend 只支持 CUDA Tensor")
    if query.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"FlashAttention 不支持 dtype={query.dtype}")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Q/K/V dtype 必须一致")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q/K/V 必须是 [batch, heads, seq_len, head_dim]")
    if key.shape != value.shape:
        raise ValueError("K/V 形状必须一致")
    batch_size, num_heads, query_len, head_dim = query.shape
    if key.shape[0] != batch_size or key.shape[3] != head_dim:
        raise ValueError("Q/K/V 的 batch 与 head_dim 必须一致")
    if num_heads % key.shape[1] != 0:
        raise ValueError("Query 头数必须能被 KV 头数整除")
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError(f"head_dim 必须是不小于 16 的 2 的幂，实际为 {head_dim}")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("Q/K/V 末维必须连续")
    if key_valid.dtype != torch.bool or query_valid.dtype != torch.bool:
        raise ValueError("key_valid / query_valid 必须是 bool Tensor")
    if tuple(key_valid.shape) != (batch_size, key.shape[2]):
        raise ValueError(f"key_valid 形状应为 {(batch_size, key.shape[2])}")
    if tuple(query_valid.shape) != (batch_size, query_len):
        raise ValueError(f"query_valid 形状应为 {(batch_size, query_len)}")
    if key_valid.stride(-1) != 1 or query_valid.stride(-1) != 1:
        raise ValueError("key_valid / query_valid 末维必须连续")
    if causal_offset < 0:
        raise ValueError("causal_offset 不能小于 0")


def _select_block_sizes(query_len: int, dtype: torch.dtype) -> tuple[int, int, int, int]:
    """按 Query 长度与 dtype 选取 (BLOCK_M, BLOCK_N, num_warps, num_stages)。

    Decode（query_len <= 16）用 tl.dot 允许的最小 BLOCK_M=16；float32 Tile 字节数翻倍，缩小 Tile 以控制共享内存。
    """
    if query_len <= 16:
        return (16, 64, 4, 2) if dtype != torch.float32 else (16, 32, 4, 2)
    if dtype == torch.float32:
        return 32, 32, 4, 2
    return 64, 64, 4, 2


def flash_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_valid: torch.Tensor,
    query_valid: torch.Tensor,
    causal_offset: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """执行手写 FlashAttention 前向。

    query:         [batch_size, num_heads, query_len, head_dim],  model_dtype, 末维连续（任意 stride）
    key / value:   [batch_size, num_kv_heads, kv_len, head_dim],  model_dtype, 末维连续（任意 stride，可为 transpose 视图）
    key_valid:     [batch_size, kv_len],                          bool, 末维连续
    query_valid:   [batch_size, query_len],                       bool, 末维连续
    causal_offset: Query 第 0 行在 Key 序列中的位置偏移（Query 行 q 只可见 k <= causal_offset + q）
    softmax_scale: Score 缩放系数，None 时为 1 / sqrt(head_dim)

    Returns:
        output: [batch_size, num_heads, query_len, head_dim], model_dtype, non-contiguous
                （底层为 contiguous 的 [batch_size, query_len, num_heads, head_dim] 缓冲的 transpose(1, 2) 视图；
                padding Query 行输出为 0）
    """
    _validate_inputs(query, key, value, key_valid, query_valid, causal_offset)
    batch_size, num_heads, query_len, head_dim = query.shape
    num_kv_heads, kv_len = key.shape[1], key.shape[2]
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    block_m, block_n, num_warps, num_stages = _select_block_sizes(query_len, query.dtype)

    # output_buffer: [batch_size, query_len, num_heads, head_dim], model_dtype, contiguous
    output_buffer = torch.empty(
        (batch_size, query_len, num_heads, head_dim),
        dtype=query.dtype,
        device=query.device,
    )
    grid = (triton.cdiv(query_len, block_m), batch_size * num_heads)
    _flash_attention_forward_kernel[grid](
        query,
        key,
        value,
        key_valid,
        query_valid,
        output_buffer,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key_valid.stride(0),
        query_valid.stride(0),
        output_buffer.stride(0),
        output_buffer.stride(1),
        output_buffer.stride(2),
        query_len,
        kv_len,
        causal_offset,
        # exp2 比 exp 更贴合 GPU SFU：把自然指数的 scale 换底到以 2 为底。
        scale * math.log2(math.e),
        NUM_HEADS=num_heads,
        NUM_KV_GROUPS=num_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        # float32 禁用 TF32，保证 float32 Parity 的 IEEE 精度；16 位输入走 Tensor Core 原生精度。
        INPUT_PRECISION="ieee" if query.dtype == torch.float32 else "tf32",
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # transpose(1, 2): [batch_size, num_heads, query_len, head_dim], non-contiguous 视图
    return output_buffer.transpose(1, 2)


@dataclass(frozen=True)
class FlashAttentionInputs(AttentionInputs):
    """flash attention backend 单次前向共享的预计算输入（持有 CausalPaddingMask，Kernel 内按 Tile 现算）。"""

    mask: CausalPaddingMask


class FlashAttentionBackend(AttentionBackend[FlashAttentionInputs]):
    """Triton FlashAttention backend：单步输入持有 CausalPaddingMask，不物化 4D Mask / Bias。"""

    kind = "flash"

    def prepare_inputs(
        self, mask: CausalPaddingMask, dtype: torch.dtype
    ) -> FlashAttentionInputs:
        return FlashAttentionInputs(mask=mask)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_inputs: FlashAttentionInputs,
        scaling: float,
    ) -> torch.Tensor:
        """query: [batch_size, num_heads, query_len, head_dim]；key / value: [batch_size, num_kv_heads, kv_len, head_dim]。

        Returns: [batch_size, query_len, num_heads * head_dim], model_dtype, contiguous
        """
        batch_size, _, query_len, _ = query.shape
        mask = attn_inputs.mask
        output = flash_attention_forward(
            query,
            key,
            value,
            mask.key_valid,
            mask.query_valid,
            mask.causal_offset,
            softmax_scale=scaling,
        )
        # transpose(1, 2) 还原为底层 contiguous 的 [batch_size, query_len, num_heads, head_dim]，
        # view 为 [batch_size, query_len, num_heads * head_dim]，零拷贝。
        return output.transpose(1, 2).view(batch_size, query_len, -1)


__all__ = [
    "FlashAttentionBackend",
    "FlashAttentionInputs",
    "flash_attention_forward",
]
