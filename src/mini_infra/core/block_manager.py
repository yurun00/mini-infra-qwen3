"""Paged KV Cache 的逻辑块管理（Logical Block Allocator）：FreeKVCacheBlockQueue 与 KVCacheManager。

对标 vLLM V1 的 FreeKVCacheBlockQueue 与 KVCacheManager：
控制面只维护「请求 -> 物理 Block ID 列表（Block Table）」与空闲块队列，
物理显存（K/V 张量）由 worker/paged_kv_cache.py 持有，二者仅通过整数 physical_block_id 对接。

注意：本模块处于纯 CPU 控制面，严禁 import torch 或持有任何 GPU 张量。
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from mini_infra.core.request import RequestSpec

# vLLM 约定：block 0 为 null block，永不分配，仅用于 Block Table 右侧补齐，保证补齐位寻址合法。
NULL_BLOCK_ID = 0


@dataclass(frozen=True)
class PagedCacheConfig:
    """Paged KV Cache 配置。

    block_size: 每块容纳的 Token 数；num_blocks: 物理块池大小（含 null block），None 表示按工作负载自动推导。
    """

    block_size: int = 16
    num_blocks: int | None = None

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size 必须大于 0")
        if self.num_blocks is not None and self.num_blocks < 2:
            raise ValueError("num_blocks 至少为 2（block 0 为 null block，至少还需 1 个可分配块）")


def num_blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """容纳 num_tokens 个 Token 所需的物理块数（向上取整）。"""
    return (num_tokens + block_size - 1) // block_size


def required_pool_blocks(request_specs: Sequence[RequestSpec], max_running_requests: int, block_size: int) -> int:
    """保证任意 max_running_requests 个请求并发都不会耗尽的物理块池大小（含 null block）。

    请求生命周期内 KV Cache 最多持有 prompt_len + max_new_tokens - 1 个 Token：
    最后一个生成的 Token 不会再被前向消费，其 K/V 永远不会写入。取最长的 max_running_requests 个请求求和即为上界。
    """
    peak_tokens = sorted((spec.prompt_len + spec.max_new_tokens - 1 for spec in request_specs), reverse=True)
    return 1 + sum(num_blocks_for_tokens(tokens, block_size) for tokens in peak_tokens[:max_running_requests])


class FreeKVCacheBlockQueue:
    """物理块 ID 的空闲队列（FIFO 空闲队列，对齐 vLLM V1：先释放的先复用）。"""

    def __init__(self, num_blocks: int):
        if num_blocks < 2:
            raise ValueError("num_blocks 至少为 2（block 0 为 null block，至少还需 1 个可分配块）")
        self.num_blocks = num_blocks
        self._free_block_ids: deque[int] = deque(range(NULL_BLOCK_ID + 1, num_blocks))
        self._ever_allocated_block_ids: set[int] = set()
        self.allocation_count: int = 0
        self.reuse_count: int = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        """已分配块数（不含 null block）。"""
        return self.num_blocks - 1 - len(self._free_block_ids)

    def allocate(self, num_blocks: int) -> list[int]:
        """取出 num_blocks 个空闲块；调用方须先确认容量充足。"""
        if num_blocks > len(self._free_block_ids):
            raise RuntimeError(f"空闲块不足：需要 {num_blocks}，剩余 {len(self._free_block_ids)}")
        allocated = []
        for _ in range(num_blocks):
            block_id = self._free_block_ids.popleft()
            if block_id in self._ever_allocated_block_ids:
                self.reuse_count += 1
            else:
                self._ever_allocated_block_ids.add(block_id)
                self.allocation_count += 1
            allocated.append(block_id)
        return allocated

    def free(self, block_ids: Sequence[int]) -> None:
        self._free_block_ids.extend(block_ids)


class KVCacheManager:
    """请求级 Block Table 管理：按 Token 数按需追加物理块，请求结束时整体归还。"""

    def __init__(self, block_size: int, num_blocks: int):
        if block_size < 1:
            raise ValueError("block_size 必须大于 0")
        # 每个物理块容纳的 Token 数（如 16），决定逻辑块划分粒度与物理槽位偏移计算
        self.block_size = block_size
        # 物理块 ID 空闲队列，负责可用物理块（physical_block_id）的分配与回收
        self.free_block_queue = FreeKVCacheBlockQueue(num_blocks)
        # 请求级块表映射（request_idx -> 物理块号列表），列表下标即为逻辑块号（logical_block_idx）
        self._block_tables: dict[int, list[int]] = {}
        # 请求已分配缓存的有效 Token 总数（request_idx -> total_tokens），用于判断追加 Token 时是否跨越块边界
        self._num_tokens_by_request_idx: dict[int, int] = {}

    @property
    def num_used_blocks(self) -> int:
        return self.free_block_queue.num_used_blocks

    @property
    def block_allocation_count(self) -> int:
        return self.free_block_queue.allocation_count

    @property
    def block_reuse_count(self) -> int:
        return self.free_block_queue.reuse_count

    def block_table(self, request_idx: int) -> list[int]:
        """请求当前的 Block Table（逻辑块号 -> 物理块号），只读使用。"""
        return self._block_tables[request_idx]

    def num_admissible(self, prompt_lens: Sequence[int]) -> int:
        """按 FCFS 次序，当前空闲块最多能为前多少个请求分配 Prompt 所需的块。"""
        budget = self.free_block_queue.num_free_blocks
        admissible = 0
        for prompt_len in prompt_lens:
            need = num_blocks_for_tokens(prompt_len, self.block_size)
            if need > budget:
                break
            budget -= need
            admissible += 1
        return admissible

    def allocate_slots(self, request_idx: int, num_new_tokens: int) -> bool:
        """为请求新增 num_new_tokens 个 Token 的缓存位，必要时追加物理块；块不足时不做任何分配并返回 False。"""
        block_table = self._block_tables.get(request_idx, [])
        total_tokens = self._num_tokens_by_request_idx.get(request_idx, 0) + num_new_tokens
        extra_blocks = num_blocks_for_tokens(total_tokens, self.block_size) - len(block_table)
        if extra_blocks > self.free_block_queue.num_free_blocks:
            return False
        block_table.extend(self.free_block_queue.allocate(extra_blocks))
        self._block_tables[request_idx] = block_table
        self._num_tokens_by_request_idx[request_idx] = total_tokens
        return True

    def free(self, request_idx: int) -> None:
        """归还请求持有的全部物理块。"""
        self.free_block_queue.free(self._block_tables.pop(request_idx))
        del self._num_tokens_by_request_idx[request_idx]


__all__ = [
    "FreeKVCacheBlockQueue",
    "KVCacheManager",
    "NULL_BLOCK_ID",
    "PagedCacheConfig",
    "num_blocks_for_tokens",
    "required_pool_blocks",
]
