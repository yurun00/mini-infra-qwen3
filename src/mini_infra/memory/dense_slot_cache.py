"""Dense 槽位 KV Cache：Continuous Batching 的持久化批量 Cache。

本期尚未引入 Paged KV Cache。这里用固定数量（capacity）的 dense 槽位保存 KV Cache：
请求入场时把其 Prefill Cache 右对齐写入空槽位，Decode 时复用整个批量 Cache。
这种实现便于观察请求的动态加入与退出，也会如实暴露两类搬移代价：

1. 新请求的 Prompt 比当前 Cache 更长时，需要在序列维左侧补零扩容（整块 Cache 重新分配）；
2. 只要仍有请求在运行，Cache 长度就只增不减（已结束请求的槽位只清 mask，不回收显存）。
"""

import time
from collections.abc import Sequence

import torch

from mini_infra.core.request import RequestState
from mini_infra.device import synchronize
from mini_infra.memory.kv_cache import KVCache, cache_size_bytes


class DenseSlotCache:
    """持有 [capacity, kv_heads, length, head_dim] 形状的批量 KVCache 与对应的历史 Mask。

    序列维采用“右对齐”布局：各槽位真实历史位于序列维末尾，左侧为被 mask 的空位；
    Decode 时模型把新 Token 的 Key/Value 拼在序列维最右侧后整体返回，由本类写回容器。
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

    def absorb_prefill(self, states: Sequence[RequestState], prefill_cache: KVCache) -> float:
        """把一批新请求的 Prefill Cache 写入各自槽位，必要时分配/左侧扩容，返回搬移耗时（ms）。

        Args:
            states: 本次入场的请求，states[i] 对应 prefill_cache 的第 i 行，且已分配 slot_idx/cache_length。
            prefill_cache: 左 Padding Prefill 产生的 Cache，各行真实 Token 位于序列维末尾 cache_length 个位置。
        """
        synchronize(self.device)
        start = time.perf_counter()
        target_length = max(self.length, prefill_cache.length)
        if self.kv_cache is None:
            self._allocate(prefill_cache, target_length)
        elif target_length > self.length:
            self._grow_left(target_length - self.length)
        assert self.kv_cache is not None and self.history_mask is not None

        for row_idx, state in enumerate(states):
            slot_idx, cache_length = state.slot_idx, state.cache_length
            start_pos = target_length - cache_length
            self.history_mask[slot_idx].zero_()
            self.history_mask[slot_idx, start_pos:] = True
            for layer_idx, (prefill_key, prefill_value) in enumerate(prefill_cache):
                key, value = self.kv_cache.key_cache[layer_idx], self.kv_cache.value_cache[layer_idx]
                key[slot_idx].zero_()
                value[slot_idx].zero_()
                key[slot_idx, :, start_pos:, :] = prefill_key[row_idx, :, -cache_length:, :]
                value[slot_idx, :, start_pos:, :] = prefill_value[row_idx, :, -cache_length:, :]
        synchronize(self.device)
        return (time.perf_counter() - start) * 1000

    def prepare_decode_inputs(
        self, running: Sequence[RequestState], pad_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """构造 capacity 行的单步 Decode 输入，返回 (input_ids, position_ids, attention_mask, 准备耗时 ms)。

        空闲/已结束槽位填 Padding 并整行 mask；先在 Host 侧组装列表再一次性传输，避免逐槽位的小 kernel。
        """
        if self.history_mask is None:
            raise RuntimeError("Cache 尚未分配，无法构造 Decode 输入")
        synchronize(self.device)
        start = time.perf_counter()
        token_ids = [pad_token_id] * self.capacity
        positions = [0] * self.capacity
        active = [False] * self.capacity
        for state in running:
            slot_idx = state.slot_idx
            token_ids[slot_idx] = state.generated[-1]
            positions[slot_idx] = state.cache_length
            active[slot_idx] = True
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        position_ids = torch.tensor(positions, dtype=torch.long, device=self.device).unsqueeze(1)
        active_mask = torch.tensor(active, dtype=torch.bool, device=self.device).unsqueeze(1)
        attention_mask = torch.cat((self.history_mask, active_mask), dim=1)
        synchronize(self.device)
        return input_ids, position_ids, attention_mask, (time.perf_counter() - start) * 1000

    def commit_decode_step(
        self, attention_mask: torch.Tensor, present_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Decode 单步后，把模型返回的 present_key_values 写回批量 Cache，并将本步 Mask 记为新的历史 Mask。"""
        self.kv_cache = KVCache(present_key_values)
        self.history_mask = attention_mask

    def rebuild_pool(self, states: Sequence[RequestState], prefill_cache: KVCache) -> None:
        """用当前波次左 Padding Prefill 产生的 Cache 重建整块物理池（Fixed Batching 零拷贝路径）。"""
        self.capacity = len(states)
        self.kv_cache = prefill_cache
        target_length = prefill_cache.length
        self.history_mask = torch.zeros((self.capacity, target_length), dtype=torch.bool, device=self.device)
        for state in states:
            self.history_mask[state.slot_idx, target_length - state.cache_length :] = True

    def release(self, state: RequestState) -> None:
        """释放已结束请求的槽位：只清 mask，K/V 数据留待下次入场时覆盖。"""
        if self.history_mask is not None and state.slot_idx is not None:
            self.history_mask[state.slot_idx].zero_()

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

    def _grow_left(self, padding_len: int) -> None:
        assert self.kv_cache is not None and self.history_mask is not None
        for layer_idx, (key, value) in enumerate(self.kv_cache):
            zeros = torch.zeros(
                (self.capacity, key.shape[1], padding_len, key.shape[3]), dtype=key.dtype, device=self.device
            )
            self.kv_cache.key_cache[layer_idx] = torch.cat((zeros, key), dim=2)
            self.kv_cache.value_cache[layer_idx] = torch.cat((zeros, value), dim=2)
        mask_padding = torch.zeros((self.capacity, padding_len), dtype=torch.bool, device=self.device)
        self.history_mask = torch.cat((mask_padding, self.history_mask), dim=1)
