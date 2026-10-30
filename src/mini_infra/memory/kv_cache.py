"""KV Cache 显式容器定义与显存计算。"""

from typing import Any

import torch


class KVCache:
    """动态 KV Cache 显式容器。

    在模型前向传播中用于累积并维护历史 Token 的 Key 和 Value 张量：
    - Prefill 阶段：模型将各层的初始 Key/Value 张量写入容器中；
    - Decode 阶段：每一步单 Token 推理时，将新计算的 Key/Value 原地追加到历史缓存中。
    """

    def __init__(self):
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    @property
    def length(self) -> int:
        """当前已缓存的历史序列长度（尚未分配时为 0）。"""
        return self.get_seq_length()

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

    def __getitem__(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的 (key, value)。"""
        return self.key_cache[layer_index], self.value_cache[layer_index]


def cache_size_bytes(kv_cache: Any) -> int:
    """返回 KV Cache 容器实际承载的数据字节数。"""
    if kv_cache is None:
        return 0
    if isinstance(kv_cache, KVCache):
        return kv_cache.total_bytes()
    return sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in kv_cache)
