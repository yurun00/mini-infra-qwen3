"""Paged KV Cache 专职执行器：Prefill 后散写 physical block pool，Decode 按 block table 寻址。

只认 physical_block_id 与裸张量（block table 由 control plane 以 list[list[int]] 下发），不感知 request_id / 调度状态。
"""

from collections.abc import Sequence
from typing import NamedTuple

import torch

from mini_infra.device import StepTimer
from mini_infra.model.qwen3 import PagedAttentionMetadata
from mini_infra.worker.base_runner import BaseModelRunner
from mini_infra.worker.kv_cache import KVCache
from mini_infra.worker.paged_kv_cache import PagedKVCachePool
from mini_infra.worker.sampler import greedy_sample

# 与 control plane FreeKVCacheBlockQueue 的约定：block 0 为 null block（永不分配），用于 block table 右侧补齐。
_NULL_BLOCK_ID = 0


class PagedDecodeOutput(NamedTuple):
    """一步 Paged Decode 的产出：按输入行次序的下一个 Token ID 列表与模型耗时（ms）。"""

    next_tokens: list[int]
    model_ms: float


class PagedRunner(BaseModelRunner):
    """Paged KV Cache 场景专职执行器：无状态，驱动 PagedKVCachePool 完成 Prefill 落盘与 Decode。"""

    def store_prefill(
        self,
        pool: PagedKVCachePool,
        prefill_cache: KVCache,
        block_tables: Sequence[Sequence[int]],
        cache_lengths: Sequence[int],
    ) -> None:
        """把 Prefill Cache 按各请求 block table 写入 physical block pool（不 sync、不计时，计时由调用方负责）。

        block_tables[i] 为第 i 行请求已分配的 physical_block_id 列表；
        prefill_cache 各行有效 Token 位于 padded_len 维末尾 cache_lengths[i] 个位置。
        """
        padded_length = prefill_cache.length
        block_size = pool.block_size
        token_rows: list[int] = []
        token_positions: list[int] = []
        slots: list[int] = []
        for row_idx, (block_table, cache_length) in enumerate(zip(block_tables, cache_lengths)):
            for token_idx in range(cache_length):
                token_rows.append(row_idx)
                token_positions.append(padded_length - cache_length + token_idx)
                slots.append(block_table[token_idx // block_size] * block_size + token_idx % block_size)
        # 单次 H2D：[3, num_tokens]。
        index_tensor = torch.tensor([token_rows, token_positions, slots], dtype=torch.long, device=self.device)
        pool.write_prefill(
            prefill_cache,
            token_rows=index_tensor[0],
            token_positions=index_tensor[1],
            slot_mapping=index_tensor[2],
        )

    def prepare_decode_inputs(
        self,
        block_tables: Sequence[Sequence[int]],
        last_tokens: Sequence[int],
        positions: Sequence[int],
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, PagedAttentionMetadata]:
        """Host 侧批量组装单步 Decode 输入并单次上卡（纯组装，不 sync、不计时）。

        只包含活跃请求（无 Padding 行）。block_tables 须已覆盖本步新 Token 的位置；positions[i] 为新 Token 的逻辑位置。

        Returns:
            input_ids:    [B, 1], long, non-contiguous 列切片视图
            position_ids: [B, 1], long, non-contiguous 列切片视图
            metadata:     PagedAttentionMetadata（slot_mapping [B], block_tables [B, max_blocks], context_lens [B]）
        """
        max_blocks = max(len(table) for table in block_tables)
        # [B, 4 + max_blocks]
        rows = [
            [token, position, table[position // block_size] * block_size + position % block_size, position + 1]
            + list(table)
            + [_NULL_BLOCK_ID] * (max_blocks - len(table))
            for table, token, position in zip(block_tables, last_tokens, positions)
        ]
        packed = torch.tensor(rows, dtype=torch.long, device=self.device)
        metadata = PagedAttentionMetadata(
            slot_mapping=packed[:, 2], block_tables=packed[:, 4:], context_lens=packed[:, 3]
        )
        return packed[:, 0:1], packed[:, 1:2], metadata

    def run_decode_step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        metadata: PagedAttentionMetadata,
        pool: PagedKVCachePool,
    ) -> PagedDecodeOutput:
        """执行一步 Paged Decode 前向与采样（新 K/V 由模型就地写入 physical block pool，耗时由 StepTimer 当场结算）。

        input_ids:    [B, 1], long, contiguous / 列切片视图
        position_ids: [B, 1], long, contiguous / 列切片视图
        metadata:     PagedAttentionMetadata（slot_mapping [B], block_tables [B, max_blocks], context_lens [B]）
        """
        with StepTimer(self.device) as timer:
            attn_metadata = metadata._replace(kv_pools=tuple(pool.kv_pools))
            logits, _ = self.model(
                input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )
            next_token, _ = greedy_sample(logits)
        return PagedDecodeOutput(next_tokens=next_token.squeeze(-1).tolist(), model_ms=timer.elapsed_ms)


__all__ = [
    "PagedDecodeOutput",
    "PagedRunner",
]
