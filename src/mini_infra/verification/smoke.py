"""快速自检：用很小的随机模型检查核心代码，不需要下载真实权重，CPU 数秒内完成。

入口：mini-infra verify smoke。同一份逻辑也被 tests/unit/test_model_smoke.py 复用。
"""

from collections.abc import Sequence

import torch

from mini_infra.core import (
    DenseCacheConfig,
    KVCacheManager,
    PagedCacheConfig,
    SchedulePhase,
    SchedulerConfig,
    TokenBudgetConfig,
    make_requests,
)
from mini_infra.engine import (
    BatchingMethod,
    EngineConfig,
    ScheduleEvent,
    run_scheduler,
)
from mini_infra.entrypoints import GenerationOptions, generate
from mini_infra.model import DenseAttentionMetadata, Qwen3Config, Qwen3ForCausalLM
from mini_infra.worker import KVCache, PagedKVCachePool, PagedRunner, cache_size_bytes
from mini_infra.worker.base_runner import _left_pad_sequences


def build_tiny_model():
    torch.manual_seed(0)
    # 缩小所有维度，使自检可以在 CPU 上几秒内完成。
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        torch_dtype="float32",
        tie_word_embeddings=True,
    )
    config.validate()
    return Qwen3ForCausalLM(config).eval()


def check_output_shape_and_tied_weights(model):
    with torch.inference_mode():
        logits, _ = model(torch.tensor([[1, 2, 3]]))

    assert logits.shape == (1, 3, model.config.vocab_size)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def check_causal_mask(model):
    # 两个输入只有最后一个 Token 不同。若 causal mask 正确，前两个位置
    # 的输出不应该受最后一个 Token 影响。
    with torch.inference_mode():
        first, _ = model(torch.tensor([[1, 2, 3]]))
        second, _ = model(torch.tensor([[1, 2, 4]]))

    assert torch.allclose(first[:, :2, :], second[:, :2, :], atol=1e-6)


def check_kv_cache_correctness(model):
    """验证 Dense KV Cache 增量路径与全量重算路径输出完全一致。"""
    prompt = torch.tensor([[1, 2, 3]])
    config = model.config

    # attn_metadata=None 即 Prefill：Logits 与全量重算天然一致，同时产出可复用的历史键值。
    with torch.inference_mode():
        baseline_logits, present = model(prompt)
    cache = KVCache(present)

    assert len(cache) == config.num_hidden_layers
    assert cache.length == 3
    expected_shape = (1, config.num_key_value_heads, 3, config.head_dim)
    for key, value in cache:
        assert tuple(key.shape) == expected_shape
        assert tuple(value.shape) == expected_shape

    expected_bytes = (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * 3
        * torch.tensor([], dtype=torch.float32).element_size()
    )
    assert cache_size_bytes(cache) == expected_bytes
    assert cache.total_bytes == expected_bytes
    assert baseline_logits.shape == (1, 3, config.vocab_size)

    # 单步 Decode 验证：把上一步的 present_key_values 包成 DenseAttentionMetadata 喂回去即可增量追加。
    next_input = torch.tensor([[4]])
    full_context = torch.cat((prompt, next_input), dim=1)
    with torch.inference_mode():
        full_logits, _ = model(full_context)
        cached_logits, present = model(next_input, attn_metadata=DenseAttentionMetadata(tuple(cache.past_key_values)))
    cache = KVCache(present)

    assert torch.allclose(full_logits[:, -1, :], cached_logits[:, -1, :], atol=1e-5)
    assert cache.length == 4


def check_fixed_batching(model):
    """验证左 Padding、请求隔离、completion_step 指标与非法 Mask 拒绝。"""
    device = torch.device("cpu")
    config = model.config
    request_a = [3, 5, 7, 9]
    request_b = [11, 13]
    request_c = [17, 19]

    padded, mask, positions, lengths = _left_pad_sequences([request_a, request_b], 0, device)
    assert padded.tolist() == [[3, 5, 7, 9], [0, 0, 11, 13]]
    assert mask.tolist() == [[True] * 4, [False, False, True, True]]
    assert positions.tolist() == [[0, 1, 2, 3], [0, 0, 0, 1]]
    assert lengths.tolist() == [4, 2]
    print("左 Padding、Position IDs、请求隔离、排列不变性检查通过")

    fixed_batching_config = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED)
    logits_opts = GenerationOptions(eos_token_id=-1, stop_on_eos=False, capture_logits=True)
    single = generate(fixed_batching_config, model, [request_a], 4, options=logits_opts)
    paired = generate(fixed_batching_config, model, [request_b, request_a], [4, 4], options=logits_opts)
    replaced = generate(fixed_batching_config, model, [request_c, request_a], [4, 4], options=logits_opts)

    assert single.sequences[0].token_ids == paired.sequences[1].token_ids == replaced.sequences[1].token_ids
    for step in range(4):
        assert torch.allclose(single.trace.logits_trace[step][0], paired.trace.logits_trace[step][1], atol=1e-5)
        assert torch.allclose(paired.trace.logits_trace[step][1], replaced.trace.logits_trace[step][1], atol=1e-5)

    base_opts = GenerationOptions(eos_token_id=-1, stop_on_eos=False)
    batched = generate(fixed_batching_config, model, [request_a, request_b], [2, 4], options=base_opts)
    assert batched.trace.num_active_tokens_per_step == [2, 2, 1, 1]
    assert [s.completion_step for s in batched.sequences] == [2, 4]
    assert abs(batched.batch_info.slot_utilization - 0.75) < 1e-12
    expected_bytes = 2 * config.num_hidden_layers * 2 * config.num_key_value_heads * config.head_dim * 7 * 4
    assert abs(batched.memory.kv_cache.peak_live_kv_cache_mb - expected_bytes / (1024**2)) < 1e-12
    print("Batched KV Cache、completion_step 和利用率检查通过")

    # 串行 baseline（单请求 generate 路径）与 Batch 输出的 Token 应完全一致。
    serial = generate(
        EngineConfig(DenseCacheConfig()),
        model,
        [request_a, request_b],
        [2, 4],
        options=base_opts,
    )
    assert [s.token_ids for s in serial.sequences] == [s.token_ids for s in batched.sequences]
    print("Batch 与串行输出一致性检查通过")

    # 非法 Cache batch 维和错误长度的 Attention Mask 必须被拒绝。
    prefill_ids = torch.tensor([request_a])
    with torch.inference_mode():
        _, present = model(prefill_ids)
        try:
            model(
                prefill_ids[:, -1:],
                attention_mask=torch.ones((1, 1), dtype=torch.bool),
                attn_metadata=DenseAttentionMetadata(tuple(present)),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("错误长度的 Attention Mask 未被拒绝")


def check_continuous_batching(model):
    """验证 Continuous Batching 动态加入退出、FCFS 队列调度与串行 baseline 输出完全一致。"""
    sequences = [
        [3, 5, 7, 9],
        [11, 13],
        [17, 19, 23],
        [29, 31, 37, 41, 43],
    ]
    budgets = [2, 4, 3, 2]

    # 1. 统一 generate 接口分发验证（多请求异构预算）
    continuous_config = EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS)
    serial_config = EngineConfig(DenseCacheConfig())
    base_opts = GenerationOptions(eos_token_id=-1, stop_on_eos=False)

    cont_gen = generate(continuous_config, model, sequences, budgets, options=base_opts)
    serial_gen = generate(serial_config, model, sequences, budgets, options=base_opts)
    assert [s.token_ids for s in cont_gen.sequences] == [s.token_ids for s in serial_gen.sequences]

    # 2. 队列调度引擎对比：固定批次 (Fixed, 容量 2)、连续批次 (Continuous, 容量 2) 与串行 (Serialized, 容量 1)
    requests = make_requests(sequences, budgets)
    fixed = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.FIXED, max_running_requests=2),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )
    continuous = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )
    serialized = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.NONE, max_running_requests=1),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )

    assert [s.token_ids for s in fixed.sequences] == [s.token_ids for s in serialized.sequences]
    assert [s.token_ids for s in continuous.sequences] == [s.token_ids for s in serialized.sequences]

    # 3. 动态退场与 batch slot 插队轨迹 (Dynamic Admission Trace)：Req 0 完成后 Req 2 动态插队，随后 Req 3 插队
    prefill_admissions = [list(e.admitted_request_ids) for e in continuous.events if e.phase == SchedulePhase.PREFILL]
    assert prefill_admissions[0] == [0, 1]
    assert prefill_admissions[1] == [2]
    assert prefill_admissions[2] == [3]

    # 4. batch slot 有效利用率优势对比
    assert continuous.batch_info.slot_utilization > fixed.batch_info.slot_utilization

    # 5. 请求时序指标逻辑自洽性校验 (queue_ms >= 0, ttft_ms >= queue_ms, end_to_end_ms >= ttft_ms)
    assert all(
        rm.queue_ms >= 0 and rm.ttft_ms >= queue_ms_val and rm.end_to_end_ms >= rm.ttft_ms
        for rm in continuous.request_metrics
        for queue_ms_val in [rm.queue_ms]
    )

    # 6. 错峰到达时序门禁 (Staggered Arrival Timeline)
    staggered_requests = make_requests(sequences[:2], [4, 2], arrival_times_ms=[0.0, 5.0])
    staggered = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2),
        model,
        staggered_requests,
        -1,
        stop_on_eos=False,
    )
    assert [len(s.token_ids) for s in staggered.sequences] == [4, 2]
    assert staggered.request_metrics[1].admitted_ms >= 5.0

    # 7. FCFS 高并发排队保序 (FIFO Queue Ordering under Concurrency Limit)
    same_arrival_requests = make_requests([sequences[0]] * 12, 1)
    same_arrival = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=3),
        model,
        same_arrival_requests,
        -1,
        stop_on_eos=False,
    )
    flattened_admissions = [request_idx for event in same_arrival.events for request_idx in event.admitted_request_ids]
    assert flattened_admissions == list(range(12))

    print("Continuous Batching 动态加入退出、FCFS 与输出一致性检查通过")
    print("固定批次、Continuous Batching 与串行单请求输出一致")


def check_paged_kv_cache(model):
    """验证 Paged KV Cache 的 physical block pool 跨界读写/释放复用与 Continuous Batching 端到端一致性。"""
    device = torch.device("cpu")
    config = model.config
    block_size = 4
    # num_blocks=4: 1 个 null block (0) + 3 个可分配 physical block (1, 2, 3)
    manager = KVCacheManager(block_size=block_size, num_blocks=4)
    pool = PagedKVCachePool(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_size=block_size,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device=device,
    )
    runner = PagedRunner(model)

    # 1. Prefill 跨界分配与散写还原（6 个 token，block_size=4 -> 2 个 physical block）
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    expected_layers = []
    for layer_idx in range(config.num_hidden_layers):
        key = torch.arange(num_kv_heads * 6 * head_dim, dtype=torch.float32).view(1, num_kv_heads, 6, head_dim)
        key = key + layer_idx * 1000.0
        value = key + 500.0
        expected_layers.append((key, value))

    assert manager.allocate_slots(0, 6)
    old_block_ids = list(manager.block_table(0))
    assert len(old_block_ids) == 2
    assert manager.num_used_blocks * block_size == 8
    runner.store_prefill(pool, KVCache(expected_layers), [old_block_ids], [6])

    for layer_idx, (exp_key, exp_val) in enumerate(expected_layers):
        key_pool, val_pool = pool.kv_pools[layer_idx]
        gathered_k = torch.cat([key_pool[b] for b in old_block_ids], dim=0)[:6].transpose(0, 1).unsqueeze(0)
        gathered_v = torch.cat([val_pool[b] for b in old_block_ids], dim=0)[:6].transpose(0, 1).unsqueeze(0)
        assert torch.equal(gathered_k, exp_key)
        assert torch.equal(gathered_v, exp_val)

    # 2. Decode 单步追加寻址与就地写池（位置 6 -> 写入第 2 个 physical block 的 offset 2）
    assert manager.allocate_slots(0, 1)
    assert len(manager.block_table(0)) == 2
    input_ids, position_ids, metadata = runner.prepare_decode_inputs(
        block_tables=[manager.block_table(0)],
        last_tokens=[7],
        positions=[6],
        block_size=block_size,
    )
    assert input_ids.tolist() == [[7]]
    assert position_ids.tolist() == [[6]]
    assert metadata.slot_mapping.tolist() == [old_block_ids[1] * block_size + 2]
    assert metadata.context_lens.tolist() == [7]
    runner.run_decode_step(input_ids, position_ids, metadata, pool)

    # 3. 释放请求 0 并验证 free block queue 的 FIFO 复用与脏尾部隔离
    manager.free(0)
    assert manager.num_used_blocks == 0
    assert manager.block_release_count == 2
    # 此时 free block queue 为 [3, 1, 2]；先分配 1 个块给占位请求 99 消费冷块 3，再为请求 1 分配 2 个 token
    assert manager.allocate_slots(99, 2)
    assert manager.block_table(99) == [3]
    assert manager.allocate_slots(1, 2)
    assert manager.block_table(1)[0] == min(old_block_ids)
    assert manager.block_reuse_count == 1

    short_layers = [(k[:, :, :2, :], v[:, :, :2, :]) for k, v in expected_layers]
    runner.store_prefill(pool, KVCache(short_layers), [manager.block_table(1)], [2])
    for layer_idx, (exp_key, exp_val) in enumerate(short_layers):
        key_pool, val_pool = pool.kv_pools[layer_idx]
        b0 = manager.block_table(1)[0]
        assert torch.equal(key_pool[b0, :2].transpose(0, 1).unsqueeze(0), exp_key)
        assert torch.equal(val_pool[b0, :2].transpose(0, 1).unsqueeze(0), exp_val)

    # 4. Dense 与 Paged Continuous Batching 端到端对齐与块复用指标验证
    sequences = [
        [3, 5, 7],
        [11, 13, 17, 19, 23, 29],
        [31, 37],
        [41, 43, 47, 53, 59],
    ]
    budgets = [5, 2, 4, 3]
    arrivals = [0.0, 0.0, 0.1, 0.2]
    requests = make_requests(sequences, budgets, arrivals)
    dense = run_scheduler(
        EngineConfig(DenseCacheConfig(), batching_method=BatchingMethod.CONTINUOUS, max_running_requests=2),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )
    paged = run_scheduler(
        EngineConfig(
            PagedCacheConfig(block_size=block_size),
            batching_method=BatchingMethod.CONTINUOUS,
            max_running_requests=2,
        ),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )
    paged_gen = generate(
        EngineConfig(PagedCacheConfig(block_size=block_size), batching_method=BatchingMethod.CONTINUOUS),
        model,
        sequences,
        budgets,
        options=GenerationOptions(
            eos_token_id=-1,
            stop_on_eos=False,
        ),
    )

    assert [s.token_ids for s in dense.sequences] == [s.token_ids for s in paged.sequences]
    assert [s.token_ids for s in dense.sequences] == [s.token_ids for s in paged_gen.sequences]
    assert paged.memory.kv_cache.paged_cache is not None
    assert paged.memory.kv_cache.paged_cache.block_reuse_count > 0
    assert paged.memory.kv_cache.paged_cache.peak_used_blocks > 0
    assert dense.memory.kv_cache.accessed_kv_cache_tokens > 0
    assert paged.memory.kv_cache.accessed_kv_cache_tokens > 0
    admissions = [request_idx for event in paged.events for request_idx in event.admitted_request_ids]
    assert admissions == [0, 1, 2, 3]

    print("Physical block 跨界写入、读取、释放和复用检查通过")
    print("Dense 与 Paged Continuous Batching 输出一致")


def _num_padded_prompt_tokens(events: Sequence[ScheduleEvent]) -> int:
    return sum(e.num_scheduled_tokens for e in events if e.phase is SchedulePhase.PREFILL)


def check_token_budget_scheduling(model):
    """验证 token-budget 调度：与 prefill-first 逐请求输出一致，Prefill burst 被预算拆分并由 decode protection 隔开。"""
    token_budget = 6
    sequences = [
        [3, 5, 7],
        [11, 13, 17, 19],
        [23, 29, 31, 37, 41, 43],
        [47, 53],
        [59, 61, 67, 71, 73],
    ]
    requests = make_requests(sequences, [5, 5, 1, 1, 3], [0.0, 0.0, 0.1, 0.1, 0.1])
    base_config = EngineConfig(
        PagedCacheConfig(block_size=4),
        batching_method=BatchingMethod.CONTINUOUS,
        max_running_requests=4,
    )
    prefill_first = run_scheduler(base_config, model, requests, -1, stop_on_eos=False)
    token_budget_run = run_scheduler(
        base_config.with_scheduler_config(SchedulerConfig(TokenBudgetConfig(token_budget))),
        model,
        requests,
        -1,
        stop_on_eos=False,
    )

    assert [s.token_ids for s in token_budget_run.sequences] == [s.token_ids for s in prefill_first.sequences]
    assert prefill_first.schedule_metrics.max_consecutive_prefill_steps > 1
    assert token_budget_run.schedule_metrics.max_consecutive_prefill_steps == 1
    assert token_budget_run.schedule_metrics.num_oversize_prefill_steps == 0  # 本负载不触发 oversize prefill
    budget_prefills = [e for e in token_budget_run.events if e.phase is SchedulePhase.PREFILL]
    assert all(e.num_scheduled_tokens <= token_budget for e in budget_prefills)
    assert _num_padded_prompt_tokens(token_budget_run.events) <= _num_padded_prompt_tokens(prefill_first.events)
    paged_profile = token_budget_run.memory.kv_cache.paged_cache
    assert paged_profile is not None and paged_profile.block_reuse_count > 0

    print("token-budget 调度：预算拆分、decode protection 与 prefill-first 输出一致性检查通过")


def run_smoke_test():
    model = build_tiny_model()
    check_output_shape_and_tied_weights(model)
    check_causal_mask(model)
    check_kv_cache_correctness(model)
    check_fixed_batching(model)
    check_continuous_batching(model)
    check_paged_kv_cache(model)
    check_token_budget_scheduling(model)


__all__ = [
    "build_tiny_model",
    "check_causal_mask",
    "check_continuous_batching",
    "check_fixed_batching",
    "check_kv_cache_correctness",
    "check_output_shape_and_tied_weights",
    "check_paged_kv_cache",
    "check_token_budget_scheduling",
    "run_smoke_test",
]


if __name__ == "__main__":
    run_smoke_test()
