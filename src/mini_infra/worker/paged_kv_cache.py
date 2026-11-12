"""Paged KV Cache physical block pool：持有逐层 K/V 物理张量，只认 physical_block_id / slot 物理索引。

注意：本模块处于纯 GPU data plane，严禁 import RequestState、KVCacheManager 等 control plane 对象；
请求到物理块的映射（block table）由 core/block_manager.py 维护，并由 PagedRunner 在 Host 侧组装为张量后下发。
"""

import math

import torch

from mini_infra.worker.kv_cache import KVCache


class PagedKVCachePool:
    """逐层 physical block pool：每层 (key_pool, value_pool) 为 [num_blocks, block_size, num_kv_heads, head_dim]。

    - block 0 为 null block（永不分配），整池以 0 初始化：释放的块不清零（读取经 context_lens 屏蔽），
      但所有位置恒为有限值，避免 0 概率 × 未初始化 NaN 污染 Attention 输出。
    - slot = physical_block_id * block_size + offset，即 [num_blocks * block_size, ...] 展平视图的行号。
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.kv_pools: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.zeros(shape, dtype=dtype, device=device), torch.zeros(shape, dtype=dtype, device=device))
            for _ in range(num_layers)
        ]

    @property
    def bytes_per_block(self) -> int:
        """单个物理块跨全部层的 K+V 字节数。"""
        key_pool, _ = self.kv_pools[0]
        return 2 * len(self.kv_pools) * math.prod(key_pool.shape[1:]) * key_pool.element_size()

    @property
    def total_bytes(self) -> int:
        """整个 physical block pool（含 null block）占用的显存字节数。"""
        return self.num_blocks * self.bytes_per_block

    def write_prefill(
        self,
        prefill_cache: KVCache,
        token_rows: torch.Tensor,
        token_positions: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """把左 Padding Prefill 产生的有效 Token K/V 一次性散写入 slot（每层一次 index_copy_，无按请求循环）。

        prefill_cache:   逐层 (key, value)，各 [batch, num_kv_heads, padded_len, head_dim]
        token_rows:      [num_tokens] long，每个有效 Token 所在的 row
        token_positions: [num_tokens] long，每个有效 Token 在 padded_len 维的位置
        slot_mapping:    [num_tokens] long，对应的目标 slot（互不重复）
        """
        for (key_pool, value_pool), (key, value) in zip(self.kv_pools, prefill_cache):
            # 高级索引夹着切片：结果为 [num_tokens, num_kv_heads, head_dim]（contiguous）。
            key_pool.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
                0, slot_mapping, key[token_rows, :, token_positions, :]
            )
            value_pool.view(-1, self.num_kv_heads, self.head_dim).index_copy_(
                0, slot_mapping, value[token_rows, :, token_positions, :]
            )


__all__ = ["PagedKVCachePool"]
