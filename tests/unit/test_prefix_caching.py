"""prefix caching 的单元与端到端测试。

1. control plane（纯 CPU，无 torch）：block hash 链、computed blocks 查找与 prompt_len - 1 上限、引用计数共享、
   惰性 LRU 淘汰（尾块先淘汰）、入场准入对空闲 cached block 的计费；
2. 端到端（CPU tiny 模型，float32，贪心解码）：启用 prefix caching 后各 batching policy / block size / 并发度下
   逐请求输出与串行 generate、未启用 prefix caching 逐位一致，且命中部分确实跳过 Prefill 计算。
"""

import pytest

from mini_infra.core import (
    BlockHash,
    ChunkedPrefillConfig,
    DenseCacheConfig,
    KVCacheManager,
    PagedCacheConfig,
    PrefillFirstConfig,
    Request,
    SchedulePhase,
    SchedulerConfig,
    TokenBudgetConfig,
    hash_block_tokens,
    kv_cache_config_from_str,
    make_requests,
)
from mini_infra.engine import BatchingMethod, EngineConfig, run_scheduler
from mini_infra.entrypoints import GenerationOptions, generate

# ---------------------------------------------------------------------------
# control plane：KVCacheManager / BlockPool
# ---------------------------------------------------------------------------


def _prefill(manager: KVCacheManager, request: Request) -> list[int]:
    """模拟一次入场 + 整段 Prompt Prefill + K/V 落盘后登记 cached block，返回 block table。"""
    manager.allocate_computed_blocks(request)
    assert manager.allocate_slots(request.request_idx, request.prompt_len - manager.num_computed_tokens(request))
    manager.cache_blocks(request.request_idx, request.token_ids, request.prompt_len)
    return manager.block_table(request.request_idx)


def test_hash_block_tokens_chains_parent_and_compares_token_ids():
    root = hash_block_tokens(None, [1, 2])
    assert root == hash_block_tokens(None, (1, 2))
    assert isinstance(root, BlockHash) and root.token_ids == (1, 2)
    # 同一块 Token、不同父块：hash 不同（block hash 覆盖整段前缀）
    assert hash_block_tokens(root, [3, 4]) != hash_block_tokens(hash_block_tokens(None, [9, 9]), [3, 4])
    assert hash_block_tokens(root, [3, 4]) == hash_block_tokens(hash_block_tokens(None, [1, 2]), [3, 4])
    # hash_value 碰撞但 Token 不同也不会被视作同一 block hash
    assert BlockHash(7, (1,)) != BlockHash(7, (2,))


def test_get_computed_blocks_matches_longest_full_block_prefix_and_keeps_last_token():
    manager = KVCacheManager(block_size=4, num_blocks=8, is_prefix_caching_enabled=True)
    owner = Request(0, tuple(range(1, 11)), 4)  # 10 Token：2 个满块 + 1 个半块
    assert manager._get_computed_blocks(owner) == []
    assert _prefill(manager, owner) == [1, 2, 3]
    assert manager.block_pool.num_cached_blocks == 2  # 半块不登记

    shares_two_blocks = Request(1, (*range(1, 9), 50, 51, 52), 4)
    assert manager._get_computed_blocks(shares_two_blocks) == [1, 2]
    assert manager.num_computed_tokens(shares_two_blocks) == 8
    # Prompt 恰为 2 个满块：最多命中 (8 - 1) // 4 = 1 块，保留最后一个 Token 前向计算首 Token logits
    assert manager._get_computed_blocks(Request(2, tuple(range(1, 9)), 4)) == [1]
    # 首块不同则整条链断开；第二块不同只命中首块
    assert manager._get_computed_blocks(Request(3, (99, *range(2, 11)), 4)) == []
    assert manager._get_computed_blocks(Request(4, (1, 2, 3, 4, 9, 9, 9, 9, 1), 4)) == [1]


def test_computed_blocks_are_shared_by_reference_count_and_survive_owner_release():
    manager = KVCacheManager(block_size=4, num_blocks=8, is_prefix_caching_enabled=True)
    _prefill(manager, Request(0, tuple(range(1, 11)), 4))
    sharer = Request(1, (*range(1, 9), 50, 51, 52), 4)

    assert manager.allocate_computed_blocks(sharer) == 8
    assert manager.block_table(1) == [1, 2]
    assert [manager.block_pool.blocks[b].ref_cnt for b in (1, 2, 3)] == [2, 2, 1]
    assert manager.allocate_slots(1, 3)
    assert manager.block_table(1) == [1, 2, 4]
    assert manager.num_used_blocks == 4
    assert (manager.stats.prefix_cache_prompt_tokens, manager.stats.prefix_cache_hit_tokens) == (10 + 11, 8)

    manager.free(0)  # 共享块仍被请求 1 引用：只有私有块 3 回到 free block queue
    assert manager.num_used_blocks == 3
    manager.free(1)
    assert manager.num_used_blocks == 0
    # 引用计数归零后 cached block 保留 block hash（惰性淘汰），仍可被后续请求命中
    assert manager.block_pool.num_cached_blocks == 2
    assert manager._get_computed_blocks(Request(2, (*range(1, 9), 7), 1)) == [1, 2]


def test_lazy_eviction_reuses_lru_free_blocks_and_evicts_tail_blocks_first():
    manager = KVCacheManager(block_size=2, num_blocks=4, is_prefix_caching_enabled=True)  # 3 个可分配块
    assert _prefill(manager, Request(0, (1, 2, 3, 4, 5), 1)) == [1, 2, 3]
    manager.free(0)  # 逆序释放：free block queue 次序为 3, 2, 1

    assert manager.allocate_computed_blocks(Request(1, (7, 8, 9), 1)) == 0
    assert manager.allocate_slots(1, 3)  # 取队首 3（无 block hash）与 2（淘汰其 block hash）
    assert manager.block_table(1) == [3, 2]
    assert manager.stats.prefix_cache_block_eviction_count == 1
    assert manager.block_pool.num_cached_blocks == 1
    # 根块 1 仍保留：前缀 (1, 2) 命中，第二块已被淘汰
    assert manager._get_computed_blocks(Request(2, (1, 2, 3, 4, 6), 1)) == [1]


def test_num_admissible_charges_free_cached_hits_once():
    manager = KVCacheManager(block_size=2, num_blocks=5, is_prefix_caching_enabled=True)  # 4 个可分配块
    _prefill(manager, Request(0, (1, 2, 3, 4, 5), 1))
    manager.free(0)  # 4 块全空闲，其中块 1、2 为 cached block
    first = Request(1, (1, 2, 3, 4, 9), 1)  # 命中 [1, 2]：新块 1 + 引用 2 个空闲 cached block = 3
    second = Request(2, (1, 2, 3, 4, 8), 1)  # 同样命中 [1, 2]，但已被 first 计费：只需新块 1
    third = Request(3, (7, 7, 7), 1)  # 需 2 块

    assert manager.num_admissible([first, second, third], 0) == 2
    assert manager.num_admissible([first, second], 1) == 1


def test_disabled_manager_never_hits_or_caches():
    manager = KVCacheManager(block_size=2, num_blocks=5)
    request = Request(0, (1, 2, 3, 4, 5), 1)
    _prefill(manager, request)
    manager.free(0)
    assert manager.block_pool.num_cached_blocks == 0
    assert manager._get_computed_blocks(Request(1, (1, 2, 3, 4, 5), 1)) == []
    assert (manager.stats.prefix_cache_prompt_tokens, manager.stats.prefix_cache_hit_tokens) == (0, 0)


def test_prefix_caching_config_requires_paged_kv_cache():
    assert kv_cache_config_from_str("paged-kv-cache", is_prefix_caching_enabled=True) == PagedCacheConfig(
        is_prefix_caching_enabled=True
    )
    assert str(PagedCacheConfig(block_size=8, is_prefix_caching_enabled=True)) == (
        "paged-kv-cache(block_size=8,prefix_caching)"
    )
    assert str(PagedCacheConfig(block_size=8)) == "paged-kv-cache(block_size=8)"
    with pytest.raises(ValueError, match="prefix caching 仅支持 paged-kv-cache"):
        kv_cache_config_from_str("dense-kv-cache", is_prefix_caching_enabled=True)


# ---------------------------------------------------------------------------
# 端到端：run_scheduler
# ---------------------------------------------------------------------------

SHARED_PREFIX = [(j * 5) % 90 + 1 for j in range(10)]


@pytest.fixture
def model(diverse_model):
    return diverse_model


def shared_prefix_prompts(suffix_lens: list[int]) -> list[list[int]]:
    return [SHARED_PREFIX + [(i * 11 + j * 7) % 90 + 1 for j in range(n)] for i, n in enumerate(suffix_lens)]


def serial_outputs(model, prompts, budgets) -> list[list[int]]:
    opts = GenerationOptions(eos_token_id=-1, stop_on_eos=False)
    return [
        generate(EngineConfig(DenseCacheConfig()), model, [prompt], budget, options=opts).sequences[0].token_ids
        for prompt, budget in zip(prompts, budgets)
    ]


PREFILL_FIRST = PrefillFirstConfig()


def run_paged(model, requests, capacity, policy=PREFILL_FIRST, method=BatchingMethod.CONTINUOUS, **config):
    return run_scheduler(
        EngineConfig(
            PagedCacheConfig(**config),
            batching_method=method,
            scheduler_config=SchedulerConfig(max_running_requests=capacity, batching_policy=policy),
        ),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )


def _num_padded_prefill_tokens(result) -> int:
    return sum(e.num_scheduled_tokens for e in result.events if e.phase is SchedulePhase.PREFILL)


POLICIES = {
    "prefill-first": PrefillFirstConfig(),
    "token-budget": TokenBudgetConfig(16),
    "chunked-prefill": ChunkedPrefillConfig(8),
}


@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("block_size", [1, 3, 4])
@pytest.mark.parametrize("policy", list(POLICIES))
def test_prefix_caching_outputs_match_serial_and_skip_cached_prefill(model, policy, block_size, capacity):
    prompts = shared_prefix_prompts([3, 5, 2, 6])
    budgets = [4, 3, 5, 2]
    requests = make_requests(prompts, budgets, [0.0, 0.0, 0.0001, 0.0002])

    enabled = run_paged(
        model, requests, capacity, POLICIES[policy], block_size=block_size, is_prefix_caching_enabled=True
    )
    disabled = run_paged(model, requests, capacity, POLICIES[policy], block_size=block_size)

    expected = serial_outputs(model, prompts, budgets)
    assert [s.token_ids for s in enabled.sequences] == expected
    assert [s.token_ids for s in disabled.sequences] == expected
    # 首个请求冷启动无命中；后续请求命中公共前缀的整块（命中长度恒为 block_size 的整数倍）
    cached = [s.prefix_cache_hit_tokens for s in enabled.sequences]
    assert cached[0] == 0 and sum(cached) > 0
    assert all(n % block_size == 0 and n <= len(SHARED_PREFIX) for n in cached)
    assert all(s.prefix_cache_hit_tokens == 0 for s in disabled.sequences)
    # 命中部分不再计入 Prefill 计费 Token；画像统计与逐请求命中一致
    assert _num_padded_prefill_tokens(enabled) < _num_padded_prefill_tokens(disabled)
    profile = enabled.memory.kv_cache.paged_cache
    assert profile is not None
    assert profile.prefix_cache_hit_tokens == sum(cached)
    assert profile.prefix_cache_prompt_tokens == sum(len(p) for p in prompts)
    # 每个请求恰入场一次（入场首 chunk 起点为命中长度而非 0）
    assert sorted(i for e in enabled.events for i in e.admitted_request_ids) == list(range(len(requests)))


def test_identical_prompt_hits_all_but_the_last_token_block(model):
    prompt = shared_prefix_prompts([2])[0]  # 12 Token
    requests = make_requests([prompt, prompt], [3, 3])

    result = run_paged(model, requests, 1, block_size=4, is_prefix_caching_enabled=True)

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, [prompt, prompt], [3, 3])
    # 最多命中 (12 - 1) // 4 = 2 块：最后一个满块须重算以产出首 Token logits
    assert [s.prefix_cache_hit_tokens for s in result.sequences] == [0, 8]


def test_running_request_blocks_are_shared_while_it_decodes(model):
    """后到请求命中仍在 Decode 的请求已写满的 Prompt 块：共享 physical block，峰值占用低于未启用时。"""
    prompts = shared_prefix_prompts([2, 3])
    budgets = [12, 4]
    requests = make_requests(prompts, budgets, [0.0, 0.0001])

    enabled = run_paged(model, requests, 2, block_size=2, is_prefix_caching_enabled=True)
    disabled = run_paged(model, requests, 2, block_size=2)

    assert [s.token_ids for s in enabled.sequences] == serial_outputs(model, prompts, budgets)
    assert enabled.sequences[1].prefix_cache_hit_tokens == len(SHARED_PREFIX)
    assert enabled.memory.kv_cache.paged_cache.peak_used_blocks < disabled.memory.kv_cache.paged_cache.peak_used_blocks


def test_generated_token_blocks_are_cached_for_multi_turn_prompts(model):
    """多轮对话：下一轮 Prompt = 上一轮 Prompt + 输出 + 新问题，命中范围覆盖上一轮 Decode 写满的块。"""
    first_prompt = shared_prefix_prompts([0])[0][:6]
    first_output = serial_outputs(model, [first_prompt], [10])[0]
    second_prompt = first_prompt + first_output + [7, 8, 9]  # 19 Token
    prompts = [first_prompt, second_prompt]
    requests = make_requests(prompts, [10, 3])

    result = run_paged(model, requests, 1, block_size=4, is_prefix_caching_enabled=True)

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, [10, 3])
    # 首轮写入 K/V 的 Token 为 6 + 9 = 15 个（最后一个生成 Token 不前向）：3 个满块 = 12 Token，超出首轮 Prompt
    assert result.sequences[1].prefix_cache_hit_tokens == 12


def test_tight_pool_evicts_cached_blocks_and_stays_correct(model):
    groups = [[(g * 13 + j * 3) % 90 + 1 for j in range(8)] for g in range(3)]
    prompts = [groups[g] + [g + 1, g + 2] for g in (0, 1, 2, 0)]  # A, B, C, A
    budgets = [3, 3, 3, 3]
    requests = make_requests(prompts, budgets)
    expected = serial_outputs(model, prompts, budgets)

    # 自动推导的最小 pool（并发 1：仅 3 个可分配块）：B、C 入场即淘汰 A 的 cached block，A 再次到来未命中
    tight = run_paged(model, requests, 1, block_size=4, is_prefix_caching_enabled=True)
    assert [s.token_ids for s in tight.sequences] == expected
    assert tight.memory.kv_cache.paged_cache.prefix_cache_block_eviction_count > 0
    assert tight.sequences[3].prefix_cache_hit_tokens == 0

    # 更大的 pool 可同时保留三组前缀：A 再次到来命中其 8 Token 前缀
    roomy = run_paged(model, requests, 1, block_size=4, num_blocks=16, is_prefix_caching_enabled=True)
    assert [s.token_ids for s in roomy.sequences] == expected
    assert roomy.sequences[3].prefix_cache_hit_tokens == 8


def test_prefix_caching_with_fixed_batching_matches_serial(model):
    prompts = shared_prefix_prompts([3, 4, 2, 5])
    budgets = [3, 2, 4, 3]
    requests = make_requests(prompts, budgets)

    result = run_paged(model, requests, 2, method=BatchingMethod.FIXED, block_size=4, is_prefix_caching_enabled=True)

    assert [s.token_ids for s in result.sequences] == serial_outputs(model, prompts, budgets)
    # 第二波请求命中第一波写入的公共前缀块
    assert [s.prefix_cache_hit_tokens for s in result.sequences] == [0, 0, 8, 8]
