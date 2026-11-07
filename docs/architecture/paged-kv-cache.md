# Paged KV Cache（physical block pool + block table）

[`DenseSlotCache`](continuous-batching.md) 为每个 batch slot 预留 `[capacity, H_kv, length, D]` 的稠密矩形，`length` 取全局最长序列：短请求被迫携带 Padding，长请求入场触发整块左侧扩容（`torch.cat` 拷贝），Decode 恒以 `capacity` 行执行。Paged KV Cache 参考 vLLM V1，把显存切成定长 physical block，请求只持有「logical block → physical block 号」的映射（block table）。

## 一、分层与职责

| 层 | 模块 | 职责 |
|---|---|---|
| ③ Core（CPU） | [`core/block_manager.py`](../../src/mini_infra/core/block_manager.py) | `FreeKVCacheBlockQueue`（free block queue，FIFO 空闲队列，block 0 为 null block）、`KVCacheManager`（请求级 block table、按需追加 physical block、整体归还）。纯 Python，无 `torch`。 |
| ④ Worker（GPU） | [`worker/paged_kv_cache.py`](../../src/mini_infra/worker/paged_kv_cache.py) | `PagedKVCachePool`：逐层 physical block pool，只认 `physical_block_id` / physical slot（token 级物理位）。 |
| ④ Worker（GPU） | [`worker/paged_runner.py`](../../src/mini_infra/worker/paged_runner.py) | `PagedRunner`：Host 侧批量组装 `slot_mapping` / `block_tables` / `context_lens`，单次 H2D；Prefill 后散写 physical block pool。 |
| ⑤ Model | [`model/qwen3.py`](../../src/mini_infra/model/qwen3.py) | `forward`（支持 `attn_metadata=PagedAttentionMetadata`）：只接收裸张量（physical block pool + `PagedAttentionMetadata`），就地写入新 K/V、按 block table 读取历史。 |
| ② Engine | [`engine/paged_schedule_backend.py`](../../src/mini_infra/engine/paged_schedule_backend.py) | `PagedScheduleBackend`：与 Dense 路径共享统一调度主循环（三级队列 + FCFS/Prefill 优先规则），仅替换 KV 存储与读取方式。 |

## 二、张量契约

```text
key_pool / value_pool: [num_blocks, block_size, num_kv_heads, head_dim], model dtype, contiguous（每层一对，vLLM 布局）
slot_mapping:          [batch]              long   physical_block_id * block_size + offset（本步新 Token 的写入 physical slot）
block_tables:          [batch, max_blocks]  long   logical block -> physical block，右侧以 null block(0) 补齐
context_lens:          [batch]              long   含本步新 Token 的有效上下文长度
```

- physical block pool 整池零初始化，释放的 block **不清零**：读取经 `context_lens` 构造的加性掩码屏蔽，且池内恒为有限值，避免 `0 × NaN`。
- Decode 只执行活跃请求，无 Padding 行（`physical_batch_size == num_active_requests`）；请求结束仅归还 physical block 号，不移动其他请求数据。
- 按 block table 取块（`key_pool[block_tables]`）的结果天然连续，`view` + `transpose` 以非连续视图直接喂 `matmul`，不触发 `contiguous()`。
- Prefill 沿用 Dense 路径的左 Padding 前向（数值同源），再每层一次 `index_copy_` 把有效 Token 散写入各请求的 physical block；该开销计入 `cache_management_ms`。

## 三、使用

```python
Mode(CachingMethod.PAGED, batching_method=BatchingMethod.CONTINUOUS)
GenerationOptions(paged_cache_config=PagedCacheConfig(block_size=16, num_blocks=None))  # None：按工作负载推导
```

```bash
# 1. Dense batch slot vs Paged（同一调度策略，唯一变量是 KV 存储）
mini-infra run --batching-method continuous-batching --compare-feature CachingMethod dense-kv-cache paged-kv-cache --kv-block-size 16

# 2. 单用例基准测试（指定 block size 并以连续批处理驱动）
mini-infra bench scheduler case --tag dev --caching-method paged-kv-cache --batching-method continuous-batching --block-size 16 --max-running-requests 4

# 3. 运行 Dense vs Paged 受控对比套件（自动评估 makespan_speedup、output_throughput_gain、accessed_kv_cache_tokens_reduction 与 peak_kv_cache_pool_reduction）
mini-infra bench scheduler suite --tag dev --suite dense-vs-paged-long-short-stress
mini-infra bench scheduler suite --tag dev --suite dense-vs-paged-length-ratio
mini-infra bench scheduler suite --tag dev --suite dense-vs-paged-block-size
```

默认 physical block pool 大小为「最长的 `max_running_requests` 个请求的峰值 Token（`prompt + max_new - 1`）所需 physical block 数之和 + null block」，保证不会耗尽。

## 四、与 vLLM 的差异（已知限制）

- 未实现抢占（Preemption）与 Prefix Caching：显式指定过小的 `num_blocks` 时，入场按「Prompt 所需 physical block」受限（FCFS 前缀），但并发增长耗尽 physical block pool 会抛出 `RuntimeError`，不会静默出错。
- 当前仅支持 `continuous-batching` 与 `fixed-batching`；None 批处理与 `paged-kv-cache` 组合会被拒绝。
- Attention 为纯 PyTorch 的「按 block table gather 后做稠密 Attention」，无融合 Paged Attention Kernel；每层会拷贝一次上下文。
- 历史发版基准数据（`benchmarks/result-history/`）对齐受控发版计划维护，`bench scheduler` 已完整集成 Paged KV Cache 单用例与对比套件。

## 五、验证

```bash
pytest tests/unit/test_block_manager.py tests/unit/test_paged_kv_cache.py tests/unit/test_paged_scheduler.py
mini-infra verify scheduler-batch   # float32：serialized / fixed / continuous / paged 逐 Token 一致
```
