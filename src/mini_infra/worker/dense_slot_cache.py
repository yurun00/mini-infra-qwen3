"""Dense 槽位 KV Cache：Continuous Batching 与固定批次的持久化批量 Cache。

注意：本模块处于纯 GPU 数据面，只认整数槽位索引与裸张量，严禁 import RequestState 等控制面生命周期对象。
"""

from collections.abc import Sequence

import torch

from mini_infra.worker.kv_cache import KVCache, cache_size_bytes


class DenseSlotCache:
    """持有 [capacity, kv_heads, length, head_dim] 形状的批量 KVCache 与对应的历史 Mask。

    序列维采用“右对齐”布局：各槽位真实历史位于序列维末尾，左侧为被 mask 的空位；
    Decode 时模型把新 Token 的 Key/Value 拼在序列维最右侧后整体返回，由 commit_decode_step 写回容器。
    """

    def __init__(self, capacity: int, device: torch.device):
        if capacity < 1:
            raise ValueError("capacity 必须大于 0")
        self._initial_capacity = capacity
        self.capacity = capacity
        self.device = device
        self.kv_cache: KVCache | None = None
        self.history_mask: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """当前底层批量 KVCache 的序列维长度（尚未分配时为 0）。"""
        return 0 if self.kv_cache is None else self.kv_cache.length

    @property
    def total_bytes(self) -> int:
        return cache_size_bytes(self.kv_cache)

    @property
    def past_key_values(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """导出底层批量 Cache 的纯张量视图，供模型前向直接消费。"""
        if self.kv_cache is None:
            raise RuntimeError("Cache 尚未分配，无法导出 past_key_values")
        return self.kv_cache.past_key_values

    def absorb_prefill(
        self,
        batch_slot_indices: Sequence[int],
        cache_lengths: Sequence[int],
        prefill_cache: KVCache,
    ) -> int:
        """把一批新请求的 Prefill Cache 写入指定 Batch 槽位，必要时分配/左侧扩容（不 sync、不计时，计时由调用方负责）。

        Args:
            batch_slot_indices: 本次入场请求绑定的 Batch 槽位索引列表。
            cache_lengths: 各入场请求的 Prefill 缓存长度（Token 数）。
            prefill_cache: 左 Padding Prefill 产生的 Cache，各行真实 Token 位于序列维末尾 cache_length 个位置。

        Returns:
            本次因左侧扩容搬移的历史 KV 字节数（首次分配或无需扩容时为 0）。
        """
        target_length = max(self.length, prefill_cache.length)
        moved_bytes = 0
        if self.kv_cache is None:
            self._allocate(prefill_cache, target_length)
        elif target_length > self.length:
            moved_bytes = self._grow_left(target_length - self.length)
        assert self.kv_cache is not None and self.history_mask is not None

        for row_idx, (batch_slot_idx, cache_length) in enumerate(zip(batch_slot_indices, cache_lengths)):
            start_pos = target_length - cache_length
            self.history_mask[batch_slot_idx].zero_()
            self.history_mask[batch_slot_idx, start_pos:] = True
            for layer_idx, (prefill_key, prefill_value) in enumerate(prefill_cache):
                key, value = self.kv_cache.key_cache[layer_idx], self.kv_cache.value_cache[layer_idx]
                key[batch_slot_idx].zero_()
                value[batch_slot_idx].zero_()
                key[batch_slot_idx, :, start_pos:, :] = prefill_key[row_idx, :, -cache_length:, :]
                value[batch_slot_idx, :, start_pos:, :] = prefill_value[row_idx, :, -cache_length:, :]
        return moved_bytes

    def adopt_prefill(
        self,
        batch_slot_indices: Sequence[int],
        cache_lengths: Sequence[int],
        prefill_cache: KVCache,
    ) -> None:
        """零拷贝接管当前波次左 Padding Prefill 产生的 Cache（Fixed Batching 路径），搬移耗时恒为 0。

        与 absorb_prefill 的拷贝语义相对：不分配、不复制，直接引用传入张量，仅按波次大小重置 capacity
        并重建 history_mask。接管后调用方应释放其引用，由本类独占持有。
        """
        self.capacity = len(batch_slot_indices)
        self.kv_cache = prefill_cache
        target_length = prefill_cache.length
        self.history_mask = torch.zeros((self.capacity, target_length), dtype=torch.bool, device=self.device)
        for batch_slot_idx, cache_length in zip(batch_slot_indices, cache_lengths):
            self.history_mask[batch_slot_idx, target_length - cache_length :] = True

    def commit_decode_step(
        self, attention_mask: torch.Tensor, present_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Decode 单步后，把模型返回的 present_key_values 写回批量 Cache，并将本步 Mask 记为新的历史 Mask。"""
        self.kv_cache = KVCache(present_key_values)
        self.history_mask = attention_mask

    def release(self, batch_slot_idx: int) -> None:
        """释放已结束请求的 Batch 槽位：只清 mask，K/V 数据留待下次入场时覆盖。"""
        if self.history_mask is not None and batch_slot_idx is not None:
            self.history_mask[batch_slot_idx].zero_()

    def reset(self) -> None:
        """没有任何运行中请求时丢弃整块 Cache，回收显存并恢复初始槽位容量。"""
        self.capacity = self._initial_capacity
        self.kv_cache = None
        self.history_mask = None

    def _allocate(self, prefill_cache: KVCache, length: int) -> None:
        new_cache = KVCache()
        for key, _ in prefill_cache:
            shape = (self.capacity, key.shape[1], length, key.shape[3])
            new_cache.key_cache.append(torch.zeros(shape, dtype=key.dtype, device=self.device))
            new_cache.value_cache.append(torch.zeros(shape, dtype=key.dtype, device=self.device))
        self.kv_cache = new_cache
        self.history_mask = torch.zeros((self.capacity, length), dtype=torch.bool, device=self.device)

    def _grow_left(self, padding_len: int) -> int:
        assert self.kv_cache is not None and self.history_mask is not None
        moved_bytes = cache_size_bytes(self.kv_cache)
        for layer_idx, (key, value) in enumerate(self.kv_cache):
            zeros = torch.zeros(
                (self.capacity, key.shape[1], padding_len, key.shape[3]), dtype=key.dtype, device=self.device
            )
            self.kv_cache.key_cache[layer_idx] = torch.cat((zeros, key), dim=2)
            self.kv_cache.value_cache[layer_idx] = torch.cat((zeros, value), dim=2)
        mask_padding = torch.zeros((self.capacity, padding_len), dtype=torch.bool, device=self.device)
        self.history_mask = torch.cat((mask_padding, self.history_mask), dim=1)
        return moved_bytes


__all__ = ["DenseSlotCache"]
