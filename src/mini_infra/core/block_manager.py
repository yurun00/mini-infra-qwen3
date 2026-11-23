"""Paged KV Cache 的逻辑块管理（Logical Block Allocator）与 prefix caching。

- control plane 只维护「请求 -> 物理 Block ID 列表（block table）」、physical block 引用计数与 free block queue；
  物理显存（K/V 张量）由 worker/paged_kv_cache.py 持有，二者仅通过整数 physical_block_id 对接。
- prefix caching：K/V 已写入 physical block pool 的满块以 block hash（父块哈希 + 本块 Token ID 的链式哈希）
  登记为 cached block；新请求入场时沿 block hash 链匹配最长前缀命中（computed blocks），引用计数 +1 共享物理块，
  跳过这段 Prompt 的 Prefill 计算。
- 惰性淘汰（lazy eviction）：引用计数归零的块（含 cached block）回到 free block queue 尾部且保留 block hash，
  仍可被后续请求命中；只有从队首被重新分配时才真正淘汰其 block hash，free block queue 次序即 LRU 次序。

注意：本模块处于纯 CPU control plane，严禁 import torch 或持有任何 GPU 张量。
"""

import abc
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from mini_infra.core.request import Request, RequestState
from mini_infra.core.types import PagedCacheConfig

# vLLM 约定：block 0 为 null block，永不分配，仅用于 block table 右侧补齐，保证补齐位寻址合法。
NULL_BLOCK_ID = 0
# 首个逻辑块的父块哈希（block hash 链的根）
_ROOT_PARENT_HASH_VALUE = 0


def num_blocks_for_tokens(num_tokens: int, block_size: int) -> int:
    """容纳 num_tokens 个 Token 所需的物理块数（向上取整）。"""
    return (num_tokens + block_size - 1) // block_size


def infer_num_blocks_in_pool(requests: Sequence[Request], max_running_requests: int, block_size: int) -> int:
    """保证任意 max_running_requests 个请求并发都不会耗尽的 physical block pool 大小（含 null block）。

    请求生命周期内 KV Cache 最多持有 prompt_len + max_new_tokens - 1 个 Token：
    最后一个生成的 Token 不会再被前向消费，其 K/V 永远不会写入。取最长的 max_running_requests 个请求求和即为上界。
    prefix caching 下共享块只会降低实际占用，cached block 又可随时淘汰，该上界依然成立。
    """
    peak_tokens = sorted((req.prompt_len + req.max_new_tokens - 1 for req in requests), reverse=True)
    return 1 + sum(num_blocks_for_tokens(tokens, block_size) for tokens in peak_tokens[:max_running_requests])


class BlockHash(NamedTuple):
    """满块的 block hash：作为 cached block 索引的字典键。

    hash_value 由父块 hash_value 与本块 Token ID 链式求得，覆盖从 Prompt 开头到本块的整段前缀；
    token_ids 参与字典等值比较，使「末块 Token 不同但 hash_value 碰撞」不会误命中。
    """

    hash_value: int
    token_ids: tuple[int, ...]


def hash_block_tokens(parent_block_hash: BlockHash | None, token_ids: Sequence[int]) -> BlockHash:
    """计算一个满块的 block hash；parent_block_hash 为 None 表示首个逻辑块。"""
    block_token_ids = tuple(token_ids)
    parent_hash_value = _ROOT_PARENT_HASH_VALUE if parent_block_hash is None else parent_block_hash.hash_value
    return BlockHash(hash((parent_hash_value, block_token_ids)), block_token_ids)


@dataclass
class KVCacheBlock:
    """physical block 的 control plane 元数据。

    ref_cnt：引用该块的请求数（prefix caching 下可 > 1）；block_hash：已登记为 cached block 时的 block hash。
    """

    block_id: int
    ref_cnt: int = 0
    block_hash: BlockHash | None = None


@dataclass
class KVCacheStats:
    """Paged KV Cache 运行期纯统计指标（与分配/缓存控制态解耦，供画像采集使用）。"""

    peak_used_blocks: int = 0
    block_cold_allocation_count: int = 0
    block_reuse_count: int = 0
    block_release_count: int = 0
    prefix_cache_block_eviction_count: int = 0
    prefix_cache_prompt_tokens: int = 0
    prefix_cache_hit_tokens: int = 0
    _ever_allocated_block_ids: set[int] = field(default_factory=set, repr=False, compare=False)


class FreeKVCacheBlockQueue:
    """物理块 ID 的空闲队列（FIFO：先释放的先复用；prefix caching 下队首即 LRU 淘汰候选）。

    以 OrderedDict 维护次序，支持 O(1) 的队首取出、队尾追加与按 ID 中途摘除（空闲 cached block 被重新命中）。
    """

    def __init__(self, num_blocks: int, stats: KVCacheStats):
        if num_blocks < 2:
            raise ValueError("num_blocks 至少为 2（block 0 为 null block，至少还需 1 个可分配块）")
        self.num_blocks = num_blocks
        self._free_block_ids: OrderedDict[int, None] = OrderedDict.fromkeys(range(NULL_BLOCK_ID + 1, num_blocks))
        self.stats = stats

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_block_ids)

    @property
    def num_used_blocks(self) -> int:
        """已分配块数（不含 null block）。"""
        return self.num_blocks - 1 - len(self._free_block_ids)

    def allocate(self, num_blocks: int) -> list[int]:
        """从队首取出 num_blocks 个空闲块；调用方须先确认容量充足。"""
        if num_blocks > len(self._free_block_ids):
            raise RuntimeError(f"空闲块不足：需要 {num_blocks}，剩余 {len(self._free_block_ids)}")
        allocated = []
        for _ in range(num_blocks):
            block_id, _ = self._free_block_ids.popitem(last=False)
            if block_id in self.stats._ever_allocated_block_ids:
                self.stats.block_reuse_count += 1
            else:
                self.stats._ever_allocated_block_ids.add(block_id)
                self.stats.block_cold_allocation_count += 1
            allocated.append(block_id)
        self.stats.peak_used_blocks = max(self.stats.peak_used_blocks, self.num_used_blocks)
        return allocated

    def free(self, block_ids: Sequence[int]) -> None:
        """按给定次序追加到队尾。"""
        self._free_block_ids.update(dict.fromkeys(block_ids))
        self.stats.block_release_count += len(block_ids)

    def remove(self, block_id: int) -> None:
        """从队列中途摘除一个空闲块（空闲 cached block 被 prefix cache 命中而重新引用）。"""
        del self._free_block_ids[block_id]
        self.stats.peak_used_blocks = max(self.stats.peak_used_blocks, self.num_used_blocks)


class BlockPool:
    """physical block 元数据池：引用计数、free block queue 与 cached block 索引。

    可分配 physical block（1..num_blocks-1，block 0 为永不分配的 null block）的状态转换：

        Free/Uncached  ──allocate()──▶  Active/Uncached  ──cache()──▶  Active/Cached ◀─┐
        (ref=0, hash=∅) ◀────free()──── (ref=1, hash=∅)                (ref≥1, hash≠∅) │ touch() / free(ref>1)
                                              ▲                           │   ▲        └─┘
                                      allocate() 淘汰               free()│   │touch()
                                      (清 hash, ref=1)            (ref→0) │   │(移出队)
                                              │                           ▼   │
                                              └─────────────────────── Free/Cached
                                                                      (ref=0, hash≠∅)

    不变量：
    - block 在 free block queue 中 <=> ref_cnt == 0（Active 块绝不会被淘汰）；
    - block.block_hash is not None <=> _cached_block_id_by_hash[block.block_hash] == block.block_id；
    - Active/Uncached 为请求私有块，恒有 ref_cnt == 1（未写满、未启用 prefix caching 或重复 hash）。
    """

    def __init__(self, num_blocks: int):
        self.stats = KVCacheStats()
        self.free_block_queue = FreeKVCacheBlockQueue(num_blocks, stats=self.stats)
        self.blocks = [KVCacheBlock(block_id) for block_id in range(num_blocks)]
        self._cached_block_id_by_hash: dict[BlockHash, int] = {}

    @property
    def num_cached_blocks(self) -> int:
        """当前登记在 prefix cache 中的 cached block 数（含引用计数为 0、可被淘汰的块）。"""
        return len(self._cached_block_id_by_hash)

    def get_cached_block(self, block_hash: BlockHash) -> int | None:
        """按 block hash 查找 cached block 的 physical_block_id；未命中返回 None。"""
        return self._cached_block_id_by_hash.get(block_hash)

    def allocate(self, num_blocks: int) -> list[int]:
        """从 free block queue 队首取出 num_blocks 个块（引用计数置 1）；取到仍带 block hash 的块即淘汰其登记。"""
        block_ids = self.free_block_queue.allocate(num_blocks)
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.block_hash is not None:
                # 惰性淘汰：Free/Cached 仅在被重新分配时清除 block hash，直接转为 Active/Uncached
                del self._cached_block_id_by_hash[block.block_hash]
                block.block_hash = None
                self.stats.prefix_cache_block_eviction_count += 1
            block.ref_cnt = 1
        return block_ids

    def touch(self, block_ids: Sequence[int]) -> None:
        """引用 prefix cache 命中的 cached block：引用计数 +1，原本空闲的块移出 free block queue（不再可被淘汰）。"""
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.ref_cnt == 0:
                # 维护不变量：ref_cnt > 0 的块不在 free block queue 中，防止被淘汰
                self.free_block_queue.remove(block_id)
            block.ref_cnt += 1

    def free(self, block_ids: Iterable[int]) -> None:
        """引用计数 -1；归零的块按给定次序追加到 free block queue 尾部（保留 block hash，惰性淘汰）。"""
        freed_block_ids = []
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                # 归零后保留 block_hash，按调用方给定次序入队尾（Free/Cached 可继续命中或惰性淘汰）
                freed_block_ids.append(block_id)
        self.free_block_queue.free(freed_block_ids)

    def cache(self, block_id: int, block_hash: BlockHash) -> None:
        """把 K/V 已写满的 physical block 登记为 cached block。

        同一步内多个请求各自算出相同内容的块时，保留先登记者；后者不登记，作为私有块随请求释放。
        """
        if block_hash in self._cached_block_id_by_hash:
            return
        self.blocks[block_id].block_hash = block_hash
        self._cached_block_id_by_hash[block_hash] = block_id


class BaseKVCacheManager(abc.ABC):
    """IterationScheduler 依赖的逻辑块管理抽象契约（Paged 实际管理与 Dense 空对象共享）。"""

    __slots__ = ()

    @abc.abstractmethod
    def num_admissible(self, requests: Sequence[Request], prefilling_states: Sequence[RequestState]) -> int:
        """按 FCFS 次序返回 requests 中逻辑块容量可接纳的最长前缀长度（prefilling_states 为需预留后续块的在途请求）。"""
        ...

    @abc.abstractmethod
    def num_computed_tokens(self, request: Request) -> int:
        """新请求入场前可复用的已缓存 Prompt Token 数。"""
        ...

    @abc.abstractmethod
    def allocate_computed_blocks(self, request: Request) -> int:
        """新请求入场时引用已缓存块并建立初始映射，返回命中 Token 数。"""
        ...

    @abc.abstractmethod
    def allocate_slots(self, request_idx: int, num_new_tokens: int) -> bool:
        """为请求追加 num_new_tokens 个 Token 的缓存位；容量不足时不做任何分配并返回 False。"""
        ...

    @abc.abstractmethod
    def cache_blocks(self, state: RequestState) -> None:
        """本步 K/V 落盘后登记请求新写满的 cached block。"""
        ...

    @abc.abstractmethod
    def free(self, request_idx: int) -> None:
        """释放请求持有的全部逻辑块资源。"""
        ...


class NullKVCacheManager(BaseKVCacheManager):
    """无逻辑块管理器的空对象实现（Dense 路径）：不施加块容量约束，分配恒成功，登记与释放均为空操作。"""

    __slots__ = ()

    def num_admissible(self, requests: Sequence[Request], prefilling_states: Sequence[RequestState]) -> int:
        return len(requests)

    def num_computed_tokens(self, request: Request) -> int:
        return 0

    def allocate_computed_blocks(self, request: Request) -> int:
        return 0

    def allocate_slots(self, request_idx: int, num_new_tokens: int) -> bool:
        return True

    def cache_blocks(self, state: RequestState) -> None:
        return None

    def free(self, request_idx: int) -> None:
        return None


NULL_KV_CACHE_MANAGER = NullKVCacheManager()


class KVCacheManager(BaseKVCacheManager):
    """请求级 block table 管理：按 Token 数按需追加物理块，请求结束时整体释放引用。

    启用 prefix caching 时，额外负责 computed blocks 查找与入场引用（num_computed_tokens / allocate_computed_blocks）
    与 K/V 落盘后的满块登记（cache_blocks）。
    """

    def __init__(self, block_size: int, num_blocks: int, is_prefix_caching_enabled: bool = False):
        if block_size < 1:
            raise ValueError("block_size 必须大于 0")
        # 每个物理块容纳的 Token 数（如 16），决定逻辑块划分粒度与 slot 偏移计算
        self.block_size = block_size
        self.is_prefix_caching_enabled = is_prefix_caching_enabled
        # physical block 元数据池：引用计数、free block queue 与 cached block 索引
        self.block_pool = BlockPool(num_blocks)
        # 请求级 block table 映射（request_idx -> 物理块号列表），列表下标即为逻辑块号（logical_block_idx）
        self._block_tables: dict[int, list[int]] = {}
        # 请求已分配缓存的有效 Token 总数（request_idx -> total_tokens），用于判断追加 Token 时是否跨越块边界
        self._num_tokens_by_request_idx: dict[int, int] = {}
        # 请求满块的 block hash 缓存（request_idx -> 逐逻辑块 block hash），按需增量计算
        self._block_hashes_by_request_idx: dict[int, list[BlockHash]] = {}
        # 请求 block table 前缀中已登记（或命中）为 cached block 的块数，cache_blocks 只处理其后新写满的块
        self._num_cached_blocks_by_request_idx: dict[int, int] = {}

        # 运行期纯统计指标（与上方控制态解耦，与 BlockPool / FreeKVCacheBlockQueue 共享同一实例）
        self.stats = self.block_pool.stats

    @property
    def num_free_blocks(self) -> int:
        """free block queue 中的块数（含可被淘汰的空闲 cached block）。"""
        return self.block_pool.free_block_queue.num_free_blocks

    @property
    def num_used_blocks(self) -> int:
        """被至少一个请求引用的块数（不含 null block 与空闲 cached block）。"""
        return self.block_pool.free_block_queue.num_used_blocks

    def block_table(self, request_idx: int) -> list[int]:
        """请求当前的 Block Table（逻辑块号 -> 物理块号），只读使用。"""
        return self._block_tables[request_idx]

    def _get_computed_blocks(self, request: Request) -> list[int]:
        """prefix cache 对请求 Prompt 的最长前缀命中（computed blocks 的 physical_block_id 列表；只读，不改引用计数）。

        最多命中 (prompt_len - 1) // block_size 个整块：至少保留 1 个 Prompt Token 参与前向以产出首 Token logits。
        """
        if not self.is_prefix_caching_enabled:
            return []
        max_num_blocks = (request.prompt_len - 1) // self.block_size
        computed_block_ids: list[int] = []
        for block_hash in self._block_hashes(request.request_idx, request.token_ids, max_num_blocks):
            block_id = self.block_pool.get_cached_block(block_hash)
            if block_id is None:
                break
            computed_block_ids.append(block_id)
        return computed_block_ids

    def num_computed_tokens(self, request: Request) -> int:
        """请求入场时 prefix cache 可命中的 Token 数（恒为 block_size 的整数倍）。"""
        return len(self._get_computed_blocks(request)) * self.block_size

    def num_admissible(self, requests: Sequence[Request], prefilling_states: Sequence[RequestState]) -> int:
        """按 FCFS 次序，扣除在途 chunked prefill 预留块后的空闲块最多能为前多少个请求容纳 Prompt。

        prefilling_states：已入场但 Prompt 尚未写满的 chunked prefill 请求，先行预留其尚需追加的 Prompt 块数，
        保证后续 chunk 的块分配不会被新入场请求抢占而卡死。
        prefix cache 命中的块无需新分配；但其中引用计数为 0 的空闲 cached block 被引用后会离开 free block queue，
        同样消耗空闲额度（同批多个请求命中同一块只计一次）。
        """
        num_reserved_blocks = sum(
            num_blocks_for_tokens(s.request.prompt_len, self.block_size)
            - num_blocks_for_tokens(s.cache_length, self.block_size)
            for s in prefilling_states
        )
        budget = self.num_free_blocks - num_reserved_blocks
        touched_block_ids: set[int] = set()
        admissible = 0
        for request in requests:
            computed_block_ids = self._get_computed_blocks(request)
            num_evictable_computed_blocks = sum(
                1
                for block_id in computed_block_ids
                if self.block_pool.blocks[block_id].ref_cnt == 0 and block_id not in touched_block_ids
            )
            need = (
                num_blocks_for_tokens(request.prompt_len, self.block_size)
                - len(computed_block_ids)
                + num_evictable_computed_blocks
            )
            if need > budget:
                break
            budget -= need
            touched_block_ids.update(computed_block_ids)
            admissible += 1
        return admissible

    def allocate_computed_blocks(self, request: Request) -> int:
        """新请求入场：以 prefix cache 命中的 computed blocks 作为 block table 前缀并引用之，返回命中 Token 数。

        同一步的多个新请求须先全部调用本方法、再统一 allocate_slots：新块分配可能从队首淘汰空闲 cached block，
        先引用可保证同批后续请求已规划的命中不被淘汰。
        """
        computed_block_ids = self._get_computed_blocks(request)
        self.block_pool.touch(computed_block_ids)
        request_idx = request.request_idx
        num_computed_tokens = len(computed_block_ids) * self.block_size
        self._block_tables[request_idx] = computed_block_ids
        self._num_tokens_by_request_idx[request_idx] = num_computed_tokens
        self._num_cached_blocks_by_request_idx[request_idx] = len(computed_block_ids)
        if self.is_prefix_caching_enabled:
            self.stats.prefix_cache_prompt_tokens += request.prompt_len
            self.stats.prefix_cache_hit_tokens += num_computed_tokens
        return num_computed_tokens

    def allocate_slots(self, request_idx: int, num_new_tokens: int) -> bool:
        """为请求新增 num_new_tokens 个 Token 的缓存位，必要时追加物理块；块不足时不做任何分配并返回 False。"""
        block_table = self._block_tables.get(request_idx, [])
        total_tokens = self._num_tokens_by_request_idx.get(request_idx, 0) + num_new_tokens
        extra_blocks = num_blocks_for_tokens(total_tokens, self.block_size) - len(block_table)
        if extra_blocks > self.num_free_blocks:
            return False
        block_table.extend(self.block_pool.allocate(extra_blocks))
        self._block_tables[request_idx] = block_table
        self._num_tokens_by_request_idx[request_idx] = total_tokens
        return True

    def cache_blocks(self, state: RequestState) -> None:
        """本步 K/V 落盘后把请求前 state.cache_length 个 Token 中新写满的块登记为 cached block。

        Prefill 区间内的 Token 全部取自 Prompt；Decode 仅在恰好写满一个块时拼接 Prompt 与生成 Token。
        必须在本步前向与 K/V 落盘完成后调用：Prefill 的历史 K/V 在前向前读取，提前登记会让同批请求命中尚未写入的块。
        未启用 prefix caching 时无操作。
        """
        if not self.is_prefix_caching_enabled:
            return
        num_computed_tokens = state.cache_length
        if num_computed_tokens <= state.request.prompt_len:
            token_ids: Sequence[int] = state.request.token_ids
        elif num_computed_tokens % self.block_size == 0:
            token_ids = (*state.request.token_ids, *state.generated)
        else:
            return
        num_full_blocks = num_computed_tokens // self.block_size
        request_idx = state.request.request_idx
        num_cached_blocks = self._num_cached_blocks_by_request_idx.get(request_idx, 0)
        if num_full_blocks <= num_cached_blocks:
            return
        block_hashes = self._block_hashes(request_idx, token_ids, num_full_blocks)
        block_table = self._block_tables[request_idx]
        for block_idx in range(num_cached_blocks, num_full_blocks):
            self.block_pool.cache(block_table[block_idx], block_hashes[block_idx])
        self._num_cached_blocks_by_request_idx[request_idx] = num_full_blocks

    def free(self, request_idx: int) -> None:
        """释放请求对其全部物理块的引用（cached block 保留 block hash 留待后续命中）。

        逆序释放：尾部块（被共享的概率最低）先进入 free block queue，从而先被淘汰。
        """
        block_table = self._block_tables.pop(request_idx)
        del self._num_tokens_by_request_idx[request_idx]
        self._num_cached_blocks_by_request_idx.pop(request_idx, None)
        self._block_hashes_by_request_idx.pop(request_idx, None)
        self.block_pool.free(reversed(block_table))

    def _block_hashes(self, request_idx: int, token_ids: Sequence[int], num_full_blocks: int) -> list[BlockHash]:
        """请求前 num_full_blocks 个满块的 block hash（增量计算并缓存，已算部分不重复哈希）。"""
        block_hashes = self._block_hashes_by_request_idx.setdefault(request_idx, [])
        for block_idx in range(len(block_hashes), num_full_blocks):
            start = block_idx * self.block_size
            parent_block_hash = block_hashes[-1] if block_hashes else None
            block_hashes.append(hash_block_tokens(parent_block_hash, token_ids[start : start + self.block_size]))
        return block_hashes[:num_full_blocks]


__all__ = [
    "BaseKVCacheManager",
    "BlockHash",
    "BlockPool",
    "FreeKVCacheBlockQueue",
    "KVCacheBlock",
    "KVCacheManager",
    "KVCacheStats",
    "NULL_BLOCK_ID",
    "NULL_KV_CACHE_MANAGER",
    "NullKVCacheManager",
    "PagedCacheConfig",
    "hash_block_tokens",
    "infer_num_blocks_in_pool",
    "num_blocks_for_tokens",
]
