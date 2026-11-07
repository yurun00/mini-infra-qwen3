"""Paged KV Cache data plane 单元测试（CPU，tiny 模型，float32）：
验证 physical block pool 读写、Paged Decode 与 Dense Decode 的数值等价。
"""

import pytest
import torch

from mini_infra.core import KVCacheManager
from mini_infra.model import DenseAttentionMetadata
from mini_infra.worker import KVCache, PagedKVCachePool, PagedRunner
from mini_infra.worker.base_runner import _left_pad_sequences

CPU = torch.device("cpu")
BLOCK_SIZE = 4


@pytest.fixture
def model(diverse_model):
    return diverse_model


def make_pool(model, num_blocks=16, block_size=BLOCK_SIZE) -> PagedKVCachePool:
    config = model.config
    return PagedKVCachePool(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device=CPU,
    )


def prefill_cache(model, prompts: list[list[int]]) -> KVCache:
    input_ids, mask, positions, _ = _left_pad_sequences(prompts, 0, CPU)
    with torch.inference_mode():
        _, present = model(input_ids, attention_mask=mask, position_ids=positions)
    return KVCache(present)


def gather_request_kv(pool: PagedKVCachePool, layer: int, block_table: list[int], length: int):
    """按 Block Table 还原单请求的 [kv_heads, length, head_dim] K/V。"""
    key_pool, value_pool = pool.kv_pools[layer]
    key = torch.cat([key_pool[b] for b in block_table], dim=0)[:length]
    value = torch.cat([value_pool[b] for b in block_table], dim=0)[:length]
    return key.transpose(0, 1), value.transpose(0, 1)


def test_pool_layout_and_byte_accounting(model):
    config = model.config
    pool = make_pool(model, num_blocks=10)

    key_pool, value_pool = pool.kv_pools[0]
    assert len(pool.kv_pools) == config.num_hidden_layers
    assert (
        tuple(key_pool.shape)
        == (10, BLOCK_SIZE, config.num_key_value_heads, config.head_dim)
        == tuple(value_pool.shape)
    )
    assert not key_pool.any() and not value_pool.any()  # 全零初始化，保证读取到的脏数据恒为有限值
    expected_block_bytes = 2 * config.num_hidden_layers * BLOCK_SIZE * config.num_key_value_heads * config.head_dim * 4
    assert pool.bytes_per_block == expected_block_bytes
    assert pool.total_bytes == 10 * expected_block_bytes


def test_store_prefill_scatters_each_request_to_its_own_blocks(model):
    prompts = [[3, 5, 7, 9, 11, 13], [17, 19, 23]]  # 长度 6 与 3：左 Padding 到 6
    manager = KVCacheManager(BLOCK_SIZE, num_blocks=16)
    # 先让 request 5 占用一些块，保证请求的块不是从 1 开始的连续段
    manager.allocate_slots(5, 5)
    for idx, prompt in enumerate(prompts):
        manager.allocate_slots(idx, len(prompt))
    pool = make_pool(model)
    runner = PagedRunner(model, CPU)

    runner.store_prefill(
        pool,
        prefill_cache(model, prompts),
        [manager.block_table(i) for i in range(len(prompts))],
        [len(p) for p in prompts],
    )

    for idx, prompt in enumerate(prompts):
        reference = prefill_cache(model, [prompt])
        for layer, (ref_key, ref_value) in enumerate(reference):
            key, value = gather_request_kv(pool, layer, manager.block_table(idx), len(prompt))
            torch.testing.assert_close(key, ref_key[0])
            torch.testing.assert_close(value, ref_value[0])
    # 请求 5 的块与其他请求不相交，未被写入
    for block_id in manager.block_table(5):
        assert not pool.kv_pools[0][0][block_id].any()


def test_prepare_decode_inputs_builds_slot_mapping_and_null_padded_block_tables(model):
    runner = PagedRunner(model, CPU)
    # 请求 A 位于位置 5（context 6，块 [9, 4]）；请求 B 位于位置 2（context 3，块 [7]）
    input_ids, position_ids, metadata = runner.prepare_decode_inputs(
        block_tables=[[9, 4], [7]], last_tokens=[11, 22], positions=[5, 2], block_size=BLOCK_SIZE
    )

    assert input_ids.tolist() == [[11], [22]] and position_ids.tolist() == [[5], [2]]
    assert metadata.slot_mapping.tolist() == [4 * BLOCK_SIZE + 1, 7 * BLOCK_SIZE + 2]
    assert metadata.context_lens.tolist() == [6, 3]
    assert metadata.block_tables.tolist() == [[9, 4], [7, 0]]  # 0 为 null block 补齐


@pytest.mark.parametrize("block_size", [1, 3, 4, 16])
def test_paged_decode_logits_match_dense_decode(model, block_size):
    """长短不一的请求混合 Decode：逐步 Logits 与各请求独立 Dense Decode 的最大绝对误差 <= 1e-4，K/V 一致。"""
    prompts = [[3, 5, 7, 9, 11, 13, 15], [17, 19, 23], [29]]
    steps = 5
    manager = KVCacheManager(block_size, num_blocks=64)
    pool = make_pool(model, num_blocks=64, block_size=block_size)
    runner = PagedRunner(model, CPU)
    for idx, prompt in enumerate(prompts):
        manager.allocate_slots(idx, len(prompt))
    runner.store_prefill(
        pool,
        prefill_cache(model, prompts),
        [manager.block_table(i) for i in range(len(prompts))],
        [len(p) for p in prompts],
    )

    # 各请求独立的 Dense 参考：单请求 Prefill + 逐步 Decode
    dense_caches = [prefill_cache(model, [p]).past_key_values for p in prompts]
    tokens = [[(i * 13 + 2) % 90 + 1] for i in range(len(prompts))]
    lengths = [len(p) for p in prompts]

    with torch.inference_mode():
        for _ in range(steps):
            for idx in range(len(prompts)):
                assert manager.allocate_slots(idx, 1)
            input_ids, position_ids, metadata = runner.prepare_decode_inputs(
                [manager.block_table(i) for i in range(len(prompts))],
                last_tokens=[t[-1] for t in tokens],
                positions=lengths,
                block_size=block_size,
            )
            paged_logits, _ = model(
                input_ids,
                position_ids=position_ids,
                attn_metadata=metadata._replace(kv_pools=tuple(pool.kv_pools)),
            )

            for idx in range(len(prompts)):
                dense_logits, dense_caches[idx] = model(
                    torch.tensor([[tokens[idx][-1]]]),
                    position_ids=torch.tensor([[lengths[idx]]]),
                    attn_metadata=DenseAttentionMetadata(tuple(dense_caches[idx])),
                )
                assert (paged_logits[idx] - dense_logits[0]).abs().max().item() <= 1e-4
                lengths[idx] += 1
                tokens[idx].append(int(paged_logits[idx, -1].argmax()))
                assert tokens[idx][-1] == int(dense_logits[0, -1].argmax())

    for idx in range(len(prompts)):
        for layer, (dense_key, dense_value) in enumerate(dense_caches[idx]):
            key, value = gather_request_kv(pool, layer, manager.block_table(idx), lengths[idx])
            torch.testing.assert_close(key, dense_key[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(value, dense_value[0], atol=1e-5, rtol=1e-5)


def test_paged_decode_ignores_stale_data_in_recycled_blocks(model):
    """释放不清零的块被新请求复用后，块内尾部残留的旧数据必须被 context_lens 屏蔽，不得污染新请求。"""
    block_size = 4
    manager = KVCacheManager(block_size, num_blocks=8)
    pool = make_pool(model, num_blocks=8, block_size=block_size)
    runner = PagedRunner(model, CPU)

    def decode_once(request_idx: int, prompt: list[int]) -> torch.Tensor:
        manager.allocate_slots(request_idx, len(prompt))
        runner.store_prefill(pool, prefill_cache(model, [prompt]), [manager.block_table(request_idx)], [len(prompt)])
        manager.allocate_slots(request_idx, 1)
        input_ids, position_ids, metadata = runner.prepare_decode_inputs(
            [manager.block_table(request_idx)], [7], [len(prompt)], block_size
        )
        with torch.inference_mode():
            logits, _ = model(
                input_ids,
                position_ids=position_ids,
                attn_metadata=metadata._replace(kv_pools=tuple(pool.kv_pools)),
            )
            return logits

    decode_once(0, [3, 5, 7, 9, 11, 13, 15])  # 占满并写脏 2 个块
    manager.free(0)
    recycled = decode_once(1, [17, 19])  # 复用其中的块，块内第 3 个 slot 起全是旧请求的数据

    fresh_manager_pool = make_pool(model, num_blocks=8, block_size=block_size)
    fresh = KVCacheManager(block_size, num_blocks=8)
    fresh.allocate_slots(1, 2)
    runner.store_prefill(fresh_manager_pool, prefill_cache(model, [[17, 19]]), [fresh.block_table(1)], [2])
    fresh.allocate_slots(1, 1)
    input_ids, position_ids, metadata = runner.prepare_decode_inputs([fresh.block_table(1)], [7], [2], block_size)
    with torch.inference_mode():
        expected, _ = model(
            input_ids,
            position_ids=position_ids,
            attn_metadata=metadata._replace(kv_pools=tuple(fresh_manager_pool.kv_pools)),
        )

    torch.testing.assert_close(recycled, expected, atol=1e-6, rtol=0)
