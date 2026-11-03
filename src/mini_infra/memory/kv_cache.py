"""KV Cache 显式容器定义与显存计算。"""

from collections.abc import Sequence

import torch


class KVCache:
    """动态 KV Cache 显式容器。

    模型层只接受/返回纯张量（past_key_values / present_key_values），不认识本容器，
    因此缓存的生命周期完全由显存层与引擎层维护：
    - 前向前用 past_key_values 导出逐层 (key, value) 张量喂给模型；
    - 前向后用模型返回的 present_key_values 重建容器（Prefill 首次建立，Decode 逐层变长）。
    """

    def __init__(self, past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None):
        layers = () if past_key_values is None else past_key_values
        self.key_cache: list[torch.Tensor] = [key for key, _ in layers]
        self.value_cache: list[torch.Tensor] = [value for _, value in layers]

    @property
    def past_key_values(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """导出模型前向所需的逐层 (key, value) 纯张量列表。"""
        return list(zip(self.key_cache, self.value_cache))

    @property
    def length(self) -> int:
        """当前已缓存的历史序列长度（尚未分配时为 0）。"""
        if not self.key_cache:
            return 0
        return self.key_cache[0].shape[2]

    @property
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


def cache_size_bytes(kv_cache: KVCache | Sequence[tuple[torch.Tensor, torch.Tensor]] | None) -> int:
    """返回 KV Cache 容器实际承载的数据字节数。"""
    if kv_cache is None:
        return 0
    if isinstance(kv_cache, KVCache):
        return kv_cache.total_bytes
    return sum(key.numel() * key.element_size() + value.numel() * value.element_size() for key, value in kv_cache)
