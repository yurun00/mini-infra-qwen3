"""Paged KV Cache 逻辑块管理（FreeKVCacheBlockQueue / KVCacheManager）的单元测试（纯 CPU control plane，无 torch）。"""

import pytest

from mini_infra.core import (
    NULL_BLOCK_ID,
    FreeKVCacheBlockQueue,
    KVCacheManager,
    PagedCacheConfig,
    Request,
    num_blocks_for_tokens,
    required_pool_blocks,
)


def test_num_blocks_for_tokens_rounds_up():
    assert [num_blocks_for_tokens(n, 4) for n in (1, 4, 5, 8, 9)] == [1, 1, 2, 2, 3]


def test_free_kv_cache_block_queue_never_hands_out_null_block_and_reuses_fifo():
    queue = FreeKVCacheBlockQueue(num_blocks=5)
    assert queue.num_free_blocks == 4 and queue.num_used_blocks == 0

    first = queue.allocate(3)
    assert first == [1, 2, 3] and NULL_BLOCK_ID not in first
    assert queue.num_free_blocks == 1 and queue.num_used_blocks == 3

    queue.free([2, 1])
    # FIFO：未分配的 4 先出，其后是先释放的 2、再释放的 1
    assert queue.allocate(3) == [4, 2, 1]
    with pytest.raises(RuntimeError, match="空闲块不足"):
        queue.allocate(1)


def test_free_kv_cache_block_queue_requires_at_least_one_allocatable_block():
    with pytest.raises(ValueError, match="num_blocks"):
        FreeKVCacheBlockQueue(num_blocks=1)


def test_allocate_slots_appends_blocks_only_when_crossing_a_block_boundary():
    manager = KVCacheManager(block_size=4, num_blocks=8)

    assert manager.allocate_slots(7, 5)  # prompt 5 -> 2 块
    assert manager.block_table(7) == [1, 2]
    for _ in range(3):  # 位置 5、6、7 仍在第 2 块内
        assert manager.allocate_slots(7, 1)
    assert manager.block_table(7) == [1, 2]
    assert manager.allocate_slots(7, 1)  # 第 9 个 Token 跨界 -> 第 3 块
    assert manager.block_table(7) == [1, 2, 3]
    assert manager.num_used_blocks == 3


def test_allocate_slots_failure_is_atomic():
    manager = KVCacheManager(block_size=2, num_blocks=4)  # 3 个可分配块
    assert manager.allocate_slots(0, 4)  # 占 2 块，剩 1 块
    assert not manager.allocate_slots(1, 4)  # 需要 2 块：失败且不残留任何状态
    assert manager.num_used_blocks == 2
    with pytest.raises(KeyError):
        manager.block_table(1)
    assert not manager.allocate_slots(0, 3)  # 增长到 7 Token 需 4 块 > 总量
    assert manager.block_table(0) == [1, 2]
    assert manager.allocate_slots(1, 2)  # 剩下的 1 块仍可用


def test_free_returns_blocks_and_allows_reuse():
    manager = KVCacheManager(block_size=2, num_blocks=4)
    manager.allocate_slots(0, 4)
    manager.allocate_slots(1, 2)
    assert manager.num_used_blocks == 3

    manager.free(0)
    assert manager.num_used_blocks == 1
    with pytest.raises(KeyError):
        manager.block_table(0)
    assert manager.allocate_slots(2, 4)
    assert sorted(manager.block_table(1) + manager.block_table(2)) == [1, 2, 3]


def test_num_admissible_is_fcfs_prefix_bounded_by_free_blocks():
    manager = KVCacheManager(block_size=4, num_blocks=6)  # 5 个可分配块
    assert manager.num_admissible([8, 4, 4]) == 3  # 2 + 1 + 1
    assert manager.num_admissible([8, 12, 1]) == 2  # 2 + 3 恰好用尽 5 块，第三个放不下
    assert manager.num_admissible([24, 4]) == 0  # 队首放不下则整体不入场（不越过队首，保持 FCFS）
    assert manager.num_admissible([]) == 0


def test_required_pool_blocks_sums_the_longest_concurrent_requests():
    requests = [
        Request(0, (1,) * 5, 4),  # 峰值 5+4-1=8 Token
        Request(1, (1,) * 3, 2),  # 4
        Request(2, (1,) * 9, 1),  # 9
    ]
    # 并发 2：最长两个是 9 与 8 -> ceil(9/4)+ceil(8/4)=3+2；另加 null block
    assert required_pool_blocks(requests, 2, 4) == 1 + 3 + 2
    # 并发不小于请求数：全部求和
    assert required_pool_blocks(requests, 8, 4) == 1 + 3 + 2 + 1


def test_paged_cache_config_validates():
    assert PagedCacheConfig().block_size == 16 and PagedCacheConfig().num_blocks is None
    with pytest.raises(ValueError, match="block_size"):
        PagedCacheConfig(block_size=0)
    with pytest.raises(ValueError, match="num_blocks"):
        PagedCacheConfig(num_blocks=1)


def test_block_manager_tracks_cold_allocation_reuse_release_and_peak():
    manager = KVCacheManager(block_size=4, num_blocks=5)  # 4 个可用块：1, 2, 3, 4
    assert manager.block_cold_allocation_count == 0
    assert manager.block_reuse_count == 0
    assert manager.block_release_count == 0
    assert manager.peak_used_blocks == 0

    # 首次分配 2 块 (1, 2)
    assert manager.allocate_slots(0, 8)
    assert manager.block_cold_allocation_count == 2
    assert manager.block_reuse_count == 0
    assert manager.peak_used_blocks == 2

    # 释放请求 0
    manager.free(0)
    assert manager.block_release_count == 2
    assert manager.num_used_blocks == 0

    # 再次分配 3 块（FIFO 先取冷块 3, 4，再复用热块 1）
    assert manager.allocate_slots(1, 12)
    assert manager.block_cold_allocation_count == 4
    assert manager.block_reuse_count == 1
    assert manager.peak_used_blocks == 3
