"""PagedBlockCache data plane 单元测试（CPU，tiny 模型，float32）：
验证 physical block pool 读写、Paged Decode 与 Dense Decode 的数值等价。
"""

import pytest
import torch

from mini_infra.core import KVCacheManager
from mini_infra.model import DenseAttentionMetadata
from mini_infra.worker import KVCache, PagedBlockCache, PagedRunner
from mini_infra.worker.base_runner import _left_pad_sequences

CPU = torch.device("cpu")
BLOCK_SIZE = 4


@pytest.fixture
def model(diverse_model):
    return diverse_model


def make_block_cache(model, num_blocks=16, block_size=BLOCK_SIZE) -> PagedBlockCache:
    config = model.config
    return PagedBlockCache(
        num_layers=config.num_hidden_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device=CPU,
    )


def prefill_cache(model, prompts: list[list[int]]) -> KVCache:
    input_ids, attention_mask, position_ids, _ = _left_pad_sequences(prompts, 0, CPU)
    with torch.inference_mode():
        _, present = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    return KVCache(present)


def gather_request_kv(block_cache: PagedBlockCache, layer_idx: int, block_table: list[int], length: int):
    """按 Block Table 还原单请求的 [kv_heads, length, head_dim] K/V。"""
    key_pool, value_pool = block_cache.kv_pools[layer_idx]
    key = torch.cat([key_pool[b] for b in block_table], dim=0)[:length]
    value = torch.cat([value_pool[b] for b in block_table], dim=0)[:length]
    return key.transpose(0, 1), value.transpose(0, 1)


def test_pool_layout_and_byte_accounting(model):
    config = model.config
    block_cache = make_block_cache(model, num_blocks=10)

    key_pool, value_pool = block_cache.kv_pools[0]
    assert len(block_cache.kv_pools) == config.num_hidden_layers
    assert (
        tuple(key_pool.shape)
        == (10, BLOCK_SIZE, config.num_key_value_heads, config.head_dim)
        == tuple(value_pool.shape)
    )
    assert not key_pool.any() and not value_pool.any()  # 全零初始化，保证读取到的脏数据恒为有限值
    expected_block_bytes = 2 * config.num_hidden_layers * BLOCK_SIZE * config.num_key_value_heads * config.head_dim * 4
    assert block_cache.bytes_per_block == expected_block_bytes
    assert block_cache.total_bytes == 10 * expected_block_bytes


def test_store_prefill_scatters_each_request_to_its_own_blocks(model):
    prompts = [[3, 5, 7, 9, 11, 13], [17, 19, 23]]  # 长度 6 与 3：左 Padding 到 6
    manager = KVCacheManager(BLOCK_SIZE, num_blocks=16)
    # 先让 request 5 占用一些块，保证请求的块不是从 1 开始的连续段
    manager.allocate_slots(5, 5)
    for request_idx, prompt in enumerate(prompts):
        manager.allocate_slots(request_idx, len(prompt))
    block_cache = make_block_cache(model)
    runner = PagedRunner(model)

    runner.store_prefill(
        block_cache,
        prefill_cache(model, prompts),
        [manager.block_table(i) for i in range(len(prompts))],
        [len(p) for p in prompts],
        [0] * len(prompts),
    )

    for request_idx, prompt in enumerate(prompts):
        reference = prefill_cache(model, [prompt])
        for layer_idx, (ref_key, ref_value) in enumerate(reference):
            key, value = gather_request_kv(block_cache, layer_idx, manager.block_table(request_idx), len(prompt))
            torch.testing.assert_close(key, ref_key[0])
            torch.testing.assert_close(value, ref_value[0])
    # 请求 5 的块与其他请求不相交，未被写入
    for block_id in manager.block_table(5):
        assert not block_cache.kv_pools[0][0][block_id].any()


def test_prepare_decode_inputs_builds_slot_mapping_and_null_padded_block_tables(model):
    runner = PagedRunner(model)
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
    block_cache = make_block_cache(model, num_blocks=64, block_size=block_size)
    runner = PagedRunner(model)
    for request_idx, prompt in enumerate(prompts):
        manager.allocate_slots(request_idx, len(prompt))
    runner.store_prefill(
        block_cache,
        prefill_cache(model, prompts),
        [manager.block_table(i) for i in range(len(prompts))],
        [len(p) for p in prompts],
        [0] * len(prompts),
    )

    # 各请求独立的 Dense 参考：单请求 Prefill + 逐步 Decode
    dense_caches = [prefill_cache(model, [p]).past_key_values for p in prompts]
    tokens = [[(i * 13 + 2) % 90 + 1] for i in range(len(prompts))]
    lengths = [len(p) for p in prompts]

    with torch.inference_mode():
        for _ in range(steps):
            for request_idx in range(len(prompts)):
                assert manager.allocate_slots(request_idx, 1)
            input_ids, position_ids, metadata = runner.prepare_decode_inputs(
                [manager.block_table(i) for i in range(len(prompts))],
                last_tokens=[t[-1] for t in tokens],
                positions=lengths,
                block_size=block_size,
            )
            paged_logits, _ = model(
                input_ids,
                position_ids=position_ids,
                attn_metadata=metadata._replace(kv_pools=tuple(block_cache.kv_pools)),
            )

            for request_idx in range(len(prompts)):
                dense_logits, dense_caches[request_idx] = model(
                    torch.tensor([[tokens[request_idx][-1]]]),
                    position_ids=torch.tensor([[lengths[request_idx]]]),
                    attn_metadata=DenseAttentionMetadata(tuple(dense_caches[request_idx])),
                )
                assert (paged_logits[request_idx] - dense_logits[0]).abs().max().item() <= 1e-4
                lengths[request_idx] += 1
                tokens[request_idx].append(int(paged_logits[request_idx, -1].argmax()))
                assert tokens[request_idx][-1] == int(dense_logits[0, -1].argmax())

    for request_idx in range(len(prompts)):
        for layer_idx, (dense_key, dense_value) in enumerate(dense_caches[request_idx]):
            key, value = gather_request_kv(
                block_cache, layer_idx, manager.block_table(request_idx), lengths[request_idx]
            )
            torch.testing.assert_close(key, dense_key[0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(value, dense_value[0], atol=1e-5, rtol=1e-5)


def test_paged_decode_ignores_stale_data_in_recycled_blocks(model):
    """释放不清零的块被新请求复用后，块内尾部残留的旧数据必须被 context_lens 屏蔽，不得污染新请求。"""
    block_size = 4
    manager = KVCacheManager(block_size, num_blocks=8)
    block_cache = make_block_cache(model, num_blocks=8, block_size=block_size)
    runner = PagedRunner(model)

    def decode_once(request_idx: int, prompt: list[int]) -> torch.Tensor:
        manager.allocate_slots(request_idx, len(prompt))
        runner.store_prefill(
            block_cache, prefill_cache(model, [prompt]), [manager.block_table(request_idx)], [len(prompt)], [0]
        )
        manager.allocate_slots(request_idx, 1)
        input_ids, position_ids, metadata = runner.prepare_decode_inputs(
            [manager.block_table(request_idx)], [7], [len(prompt)], block_size
        )
        with torch.inference_mode():
            logits, _ = model(
                input_ids,
                position_ids=position_ids,
                attn_metadata=metadata._replace(kv_pools=tuple(block_cache.kv_pools)),
            )
            return logits

    decode_once(0, [3, 5, 7, 9, 11, 13, 15])  # 占满并写脏 2 个块
    manager.free(0)
    recycled = decode_once(1, [17, 19])  # 复用其中的块，块内第 3 个 slot 起全是旧请求的数据

    fresh_block_cache = make_block_cache(model, num_blocks=8, block_size=block_size)
    fresh = KVCacheManager(block_size, num_blocks=8)
    fresh.allocate_slots(1, 2)
    runner.store_prefill(fresh_block_cache, prefill_cache(model, [[17, 19]]), [fresh.block_table(1)], [2], [0])
    fresh.allocate_slots(1, 1)
    input_ids, position_ids, metadata = runner.prepare_decode_inputs([fresh.block_table(1)], [7], [2], block_size)
    with torch.inference_mode():
        expected, _ = model(
            input_ids,
            position_ids=position_ids,
            attn_metadata=metadata._replace(kv_pools=tuple(fresh_block_cache.kv_pools)),
        )

    torch.testing.assert_close(recycled, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("block_size", [1, 3, 4, 16])
def test_chunked_prefill_matches_one_shot_prefill_kv_and_first_token(model, block_size):
    """分轮 chunk Prefill（同批各行 start 不同）后，逐请求 K/V 与整段 Prefill 一致，末 chunk 首 Token 一致。"""
    prompts = [[(j * 5 + 3) % 80 + 1 for j in range(11)], [(j * 7 + 2) % 80 + 1 for j in range(6)]]
    # 每轮 (请求, start, end)；第 2 轮两行 start 分别为 5 与 2，覆盖历史长度不等的左 Padding 与 null block 补齐
    rounds = [[(0, 0, 5), (1, 0, 2)], [(0, 5, 8), (1, 2, 6)], [(0, 8, 11)]]
    manager = KVCacheManager(block_size, num_blocks=64)
    block_cache = make_block_cache(model, num_blocks=64, block_size=block_size)
    runner = PagedRunner(model)
    first_tokens: dict[int, int] = {}

    with torch.inference_mode():
        for chunks in rounds:
            for request_idx, start, end in chunks:
                assert manager.allocate_slots(request_idx, end - start)
            block_tables = [manager.block_table(request_idx) for request_idx, _, _ in chunks]
            starts = [start for _, start, _ in chunks]
            out = runner.run_prefill_chunk_step(
                [prompts[request_idx][start:end] for request_idx, start, end in chunks],
                starts,
                block_tables,
                block_cache,
            )
            runner.store_prefill(
                block_cache, out.prefill_cache, block_tables, [end - start for _, start, end in chunks], starts
            )
            for (request_idx, _, end), token in zip(chunks, out.next_tokens):
                if end == len(prompts[request_idx]):
                    first_tokens[request_idx] = token

    for request_idx, prompt in enumerate(prompts):
        reference = prefill_cache(model, [prompt])
        input_ids, attention_mask, position_ids, _ = _left_pad_sequences([prompt], 0, CPU)
        with torch.inference_mode():
            logits, _ = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        assert first_tokens[request_idx] == int(logits[0, -1].argmax())
        for layer_idx, (ref_key, ref_value) in enumerate(reference):
            key, value = gather_request_kv(block_cache, layer_idx, manager.block_table(request_idx), len(prompt))
            torch.testing.assert_close(key, ref_key[0], atol=1e-5, rtol=0)
            torch.testing.assert_close(value, ref_value[0], atol=1e-5, rtol=0)


def test_chunk_prefill_with_no_history_is_identical_to_whole_prompt_prefill(model):
    """全部行 start 为 0：与 BaseModelRunner.run_prefill_step 的整段 Prompt Prefill 逐位一致（baseline 路径不变）。"""
    prompts = [[3, 5, 7, 9, 11, 13], [17, 19, 23]]
    runner = PagedRunner(model)
    block_cache = make_block_cache(model)
    block_tables = [[1, 2], [3]]

    with torch.inference_mode():
        chunk_out = runner.run_prefill_chunk_step(prompts, [0, 0], block_tables, block_cache)
        whole_out = runner.run_prefill_step(prompts)

    assert chunk_out.next_tokens == whole_out.next_tokens
    for (chunk_key, chunk_value), (whole_key, whole_value) in zip(chunk_out.prefill_cache, whole_out.prefill_cache):
        assert torch.equal(chunk_key, whole_key) and torch.equal(chunk_value, whole_value)


def test_gather_blocks_returns_per_layer_history_views(model):
    block_cache = make_block_cache(model, num_blocks=6, block_size=BLOCK_SIZE)
    for key_pool, value_pool in block_cache.kv_pools:
        key_pool.copy_(torch.arange(key_pool.numel(), dtype=torch.float32).view_as(key_pool))
        value_pool.copy_(key_pool + 0.5)
    block_ids = torch.tensor([[2, 4], [5, 0]], dtype=torch.long)

    gathered = block_cache.gather_blocks(block_ids)

    config = model.config
    assert len(gathered) == config.num_hidden_layers
    key, value = gathered[0]
    assert tuple(key.shape) == (2, config.num_key_value_heads, 2 * BLOCK_SIZE, head_dim := config.head_dim)
    assert tuple(value.shape) == (2, config.num_key_value_heads, 2 * BLOCK_SIZE, head_dim)
    key_pool, value_pool = block_cache.kv_pools[0]
    torch.testing.assert_close(key[0].transpose(0, 1), torch.cat([key_pool[2], key_pool[4]], dim=0))
    torch.testing.assert_close(value[1].transpose(0, 1), torch.cat([value_pool[5], value_pool[0]], dim=0))
