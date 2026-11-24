"""Paged KV Cache 专职执行器：Prefill（整段 Prompt 或 chunk）后散写 physical block pool，Decode 按 block table 寻址。

只认 physical_block_id 与裸张量（block table 由 control plane 以 list[list[int]] 下发），不感知 request_id / 调度状态。
"""

from collections.abc import Sequence
from typing import NamedTuple

import torch

from mini_infra.device import StepTimer
from mini_infra.model.qwen3 import (
    DenseAttentionMetadata,
    PagedAttentionMetadata,
)
from mini_infra.worker.base_runner import BaseModelRunner, PrefillOutput
from mini_infra.worker.kv_cache import KVCache
from mini_infra.worker.paged_block_cache import PagedBlockCache
from mini_infra.worker.sampler import greedy_sample

# 与 control plane FreeKVCacheBlockQueue 的约定：block 0 为 null block（永不分配），用于 block table 右侧补齐。
_NULL_BLOCK_ID = 0


class PagedDecodeOutput(NamedTuple):
    """一步 Paged Decode 的产出：按输入行次序的下一个 Token ID 列表与模型耗时（ms）。"""

    next_tokens: list[int]
    model_ms: float


class PagedRunner(BaseModelRunner):
    """Paged KV Cache 场景专职执行器：无状态，驱动 PagedBlockCache 完成 Prefill 落盘与 Decode。"""

    def run_prefill_chunk_step(
        self,
        chunk_token_ids: Sequence[Sequence[int]],
        start_positions: Sequence[int],
        block_tables: Sequence[Sequence[int]],
        block_cache: PagedBlockCache,
    ) -> PrefillOutput:
        """对一批 Prompt 区间（整段 Prompt 或 chunk）做左 Padding Prefill，前序 chunk 的历史 K/V 经 block table 读回。

        chunk 行 i 为 Prompt 位置 [start_positions[i], start_positions[i] + len(chunk_token_ids[i]))：
        - input_ids / position_ids：chunk 部分左 Padding，真实 Token 的 position_ids 从 start_positions[i] 连续递增；
        - 历史部分：block_tables[i] 前 ceil(start / block_size) 个 physical block 取自 block_cache，
          右侧以 null block 补齐，attention_mask 只放行前 start 个位置，保证与整段 Prompt 一次性 Prefill 严格等价；
        - 全部行 start 为 0 时无历史，与整段 Prompt 的 Prefill 走同一前向路径。

        Host 侧把 input_ids / position_ids / start / chunk 长度 / 历史 block table 打包为 long 张量单次上卡（不 sync）。

        block_tables[i]: 请求 i 的 physical_block_id 列表（至少覆盖历史部分，可已含本 chunk 的块）
        next_tokens:     各行 chunk 末位 logits 的贪心 Token；仅 chunk 写满 Prompt 的行有意义
        prefill_cache:   逐层 (key, value) 均为 [B, H_kv, num_history_slots + padded_chunk_len, D]，
                         各行本 chunk 的有效 Token 位于末尾 len(chunk_token_ids[i]) 个位置
        """
        block_size = block_cache.block_size
        chunk_lens = [len(token_ids) for token_ids in chunk_token_ids]
        padded_chunk_len = max(chunk_lens)
        num_history_blocks = [
            (start + block_size - 1) // block_size for start in start_positions
        ]
        max_history_blocks = max(num_history_blocks)
        # Host 侧将每行异构元数据拍平为等宽整数行，合并为一次 H2D 后再切片视图，避免多次零碎上卡拷贝；
        # 每行布局（L = padded_chunk_len）：[input_ids(L) | position_ids(L) | start, chunk_len | 历史 block table]。
        rows = [
            [self.pad_token_id] * (padded_chunk_len - chunk_len)
            + list(token_ids)
            + [0] * (padded_chunk_len - chunk_len)
            + list(range(start, start + chunk_len))
            + [start, chunk_len]
            + list(table[:history_blocks])
            + [_NULL_BLOCK_ID] * (max_history_blocks - history_blocks)
            for token_ids, start, table, chunk_len, history_blocks in zip(
                chunk_token_ids,
                start_positions,
                block_tables,
                chunk_lens,
                num_history_blocks,
            )
        ]
        # [B, 2 * padded_chunk_len + 2 + max_history_blocks]，一次 H2D
        packed = torch.tensor(rows, dtype=torch.long, device=self.device)
        input_ids = packed[:, :padded_chunk_len]
        position_ids = packed[:, padded_chunk_len : 2 * padded_chunk_len]
        start_lens = packed[:, 2 * padded_chunk_len : 2 * padded_chunk_len + 1]
        chunk_len_column = packed[:, 2 * padded_chunk_len + 1 : 2 * padded_chunk_len + 2]
        history_block_ids = packed[:, 2 * padded_chunk_len + 2 :]

        chunk_positions = torch.arange(padded_chunk_len, device=self.device).unsqueeze(0)
        # attention_mask 的 chunk 部分：[B, padded_chunk_len]，左 Padding 位为 False
        attention_mask = chunk_positions >= padded_chunk_len - chunk_len_column
        attn_metadata = None
        with StepTimer(self.device) as timer:
            if max_history_blocks:
                history_positions = torch.arange(
                    max_history_blocks * block_size, device=self.device
                ).unsqueeze(0)
                # [B, max_history_blocks * block_size + padded_chunk_len]，bool, contiguous
                attention_mask = torch.cat(
                    (history_positions < start_lens, attention_mask), dim=1
                )
                attn_metadata = DenseAttentionMetadata(
                    past_key_values=block_cache.gather_blocks(history_block_ids)
                )
            logits, present_key_values = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )
            next_token, _ = greedy_sample(logits)
        return PrefillOutput(
            next_tokens=next_token.squeeze(-1).tolist(),
            prefill_cache=KVCache(present_key_values),
            model_ms=timer.elapsed_ms,
        )

    def store_prefill(
        self,
        block_cache: PagedBlockCache,
        prefill_cache: KVCache,
        block_tables: Sequence[Sequence[int]],
        chunk_lens: Sequence[int],
        start_positions: Sequence[int],
    ) -> None:
        """把 Prefill Cache 按各请求 block table 写入 physical block pool（不 sync、不计时，计时由调用方负责）。

        block_tables[i] 为第 i 行请求已分配的 physical_block_id 列表（须覆盖本 chunk 的全部位置）；
        prefill_cache 各行有效 Token 位于 padded_len 维末尾 chunk_lens[i] 个位置，
        写入请求的逻辑位置 start_positions[i] .. start_positions[i] + chunk_lens[i] - 1。
        """
        padded_length = prefill_cache.length
        block_size = block_cache.block_size
        token_rows: list[int] = []
        token_positions: list[int] = []
        slots: list[int] = []
        for row_idx, (block_table, chunk_len, start) in enumerate(
            zip(block_tables, chunk_lens, start_positions)
        ):
            for token_idx in range(chunk_len):
                position = start + token_idx
                token_rows.append(row_idx)
                token_positions.append(padded_length - chunk_len + token_idx)
                slots.append(
                    block_table[position // block_size] * block_size
                    + position % block_size
                )
        # 单次 H2D：[3, num_tokens]。
        index_tensor = torch.tensor(
            [token_rows, token_positions, slots],
            dtype=torch.long,
            device=self.device,
        )
        block_cache.write_prefill(
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
            [
                token,
                position,
                table[position // block_size] * block_size + position % block_size,
                position + 1,
            ]
            + list(table)
            + [_NULL_BLOCK_ID] * (max_blocks - len(table))
            for table, token, position in zip(block_tables, last_tokens, positions)
        ]
        packed = torch.tensor(rows, dtype=torch.long, device=self.device)
        metadata = PagedAttentionMetadata(
            slot_mapping=packed[:, 2],
            block_tables=packed[:, 4:],
            context_lens=packed[:, 3],
        )
        return packed[:, 0:1], packed[:, 1:2], metadata

    def run_decode_step(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        metadata: PagedAttentionMetadata,
        block_cache: PagedBlockCache,
    ) -> PagedDecodeOutput:
        """执行一步 Paged Decode 前向与采样（新 K/V 由模型就地写入 physical block pool，耗时由 StepTimer 当场结算）。

        input_ids:    [B, 1], long, contiguous / 列切片视图
        position_ids: [B, 1], long, contiguous / 列切片视图
        metadata:     PagedAttentionMetadata（slot_mapping [B], block_tables [B, max_blocks], context_lens [B]）
        """
        with StepTimer(self.device) as timer:
            attn_metadata = metadata._replace(kv_pools=tuple(block_cache.kv_pools))
            logits, _ = self.model(
                input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )
            next_token, _ = greedy_sample(logits)
        return PagedDecodeOutput(
            next_tokens=next_token.squeeze(-1).tolist(),
            model_ms=timer.elapsed_ms,
        )


__all__ = [
    "PagedDecodeOutput",
    "PagedRunner",
]
