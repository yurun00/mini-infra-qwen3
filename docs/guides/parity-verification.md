# 数值对齐与等价性验证指南 (Parity Verification Guide)

本指南汇总了 [`src/mini_infra/verification/parity/`](../../src/mini_infra/verification/parity) 目录下的全部精度对齐与正确性测试指令。

在推理底座的演进过程中，每一次引入新特性（如 KV Cache、固定批处理、连续批处理调度器）都必须经过严格的**数值等价性（Parity）**与**请求隔离（Request Isolation）**验证，确保性能优化不以牺牲模型输出精度为代价。

---

## 一、测试全景速查表

| 验证模块 | 源码文件 | CLI 指令 (`mini-infra verify ...`) | Pytest 指令 | 默认精度 | 核心检验目标 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **1. 参考实现对齐** | `compare_transformers.py` | `mini-infra verify transformers` | `pytest tests/gpu/parity/test_transformers_parity.py` | FP16/BF16 | 验证手写 Qwen3 与 Hugging Face 官方分词、Logits 与输出 100% 对齐 |
| **2. KV Cache 等价** | `compare_cache.py` | `mini-infra verify cache` | `pytest tests/gpu/parity/test_cache_parity.py` | `float32` | 验证引入 KV Cache 增量 Decode 与无 Cache 全量重算的数值完全一致 |
| **3. Lockstep 批隔离** | `compare_batch_lockstep.py` | `mini-infra verify batch-lockstep` | `pytest tests/gpu/parity/test_lockstep_batch_parity.py` | `float32` | 验证固定批处理在不同 Batch Size、顺序与 Padding 长度下的严格请求隔离 |
| **4. Scheduler 批调度** | `compare_batch_scheduler.py` | `mini-infra verify batch-scheduler` | `pytest tests/gpu/parity/test_scheduler_batch_parity.py` | `float32` | 验证 Continuous、Paged KV Cache 与 Fixed 调度下的动态 batch slot / block table 绑定、Mask 管理与串行基准对齐 |
| **5. Paged KV Cache 对照** | `compare_paged_cache_logits.py` / `compare_paged_cache_scheduler.py` | `mini-infra verify paged-cache-logits` / `paged-cache-scheduler` | `pytest tests/gpu/parity/test_paged_cache_parity.py -k logits` / `-k scheduler` | `float32` | 验证 Dense 与 Paged KV Cache 逐 Token Logits 误差在容差内、Greedy 输出一致，以及 Scheduler 环境下多 Block Size 输出一致性与入场轨迹 |
| **6. 迭代调度策略对照** | `compare_schedulers_token_budget.py` | `mini-infra verify schedulers-token-budget` | `pytest tests/gpu/parity/test_scheduler_batch_parity.py -k schedulers` | `float32` | 在错峰 Burst 负载下验证 `prefill-first` 与 `token-budget` 的逐请求输出一致性、单步轨迹与 decode protection（最大连续 Prefill 步数对比） |
| **7. Chunked Prefill 对照** | `compare_chunked_prefill.py` | `mini-infra verify chunked-prefill` | `pytest tests/gpu/parity/test_scheduler_batch_parity.py -k chunked_prefill` | `float32` | 在含超长 Prompt 的错峰负载下验证 `token-budget` 与 `chunked-prefill` 的逐请求输出一致性、首 Token Logits 误差（`atol <= 1e-4`）与硬预算轨迹 |
| **8. Prefix Caching 对照** | `compare_prefix_caching.py` | `mini-infra verify prefix-caching` | `pytest tests/gpu/parity/test_scheduler_batch_parity.py -k prefix_caching` | `float32` | 在共享长前缀负载下验证 prefix caching 关闭 / 开启的逐请求输出一致性、命中前缀后只算剩余 Prompt 的首 Token Logits 误差（`atol <= 1e-4`）与 Prefill 计算量下降 |

---

## 二、各项验证详细指令与参数

### 1. Hugging Face Transformers 对照测试

* **对应源码**：[`src/mini_infra/verification/parity/compare_transformers.py`](../../src/mini_infra/verification/parity/compare_transformers.py)
* **核心目标**：以 Hugging Face `transformers`（`AutoModelForCausalLM`）为黄金参考（Golden Truth），比对分词、Prefill 与逐步 Decode 下的 Logits 最大绝对误差，确保自研模型架构无算子缺陷。

#### CLI 运行指令
```bash
# 默认参数运行（生成 32 tokens）
mini-infra verify transformers

# 自定义提示词与最大绝对误差阈值
mini-infra verify transformers \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32 \
  --logits-atol 0.25 \
  --output tmp/parity_transformers.json
```

#### Pytest 运行指令
```bash
pytest tests/gpu/parity/test_transformers_parity.py
```

* **判定标准**：
  - 双方分词结果完全一致；
  - 逐步解码选出的 Greedy Token ID 完全一致；
  - 最终生成的文本完全一致；
  - 各步骤 Logits 的最大绝对误差小于 `--logits-atol`（默认 `0.25`）。

---

### 2. KV Cache 前后数值对齐测试

* **对应源码**：[`src/mini_infra/verification/parity/compare_cache.py`](../../src/mini_infra/verification/parity/compare_cache.py)
* **核心目标**：验证模型在开启 KV Cache（增量计算新 Token）与关闭 KV Cache（每一步将全部历史 Token 送入重新前向）两种执行路径下的数值等价性。

#### CLI 运行指令
```bash
# 默认以 float32 精度运行 32 tokens 对齐验证
mini-infra verify cache

# 自定义参数与高精度阈值
mini-infra verify cache \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32 \
  --dtype float32 \
  --logits-atol 1e-4 \
  --output tmp/parity_cache.json
```

#### Pytest 运行指令
```bash
pytest tests/gpu/parity/test_cache_parity.py
```

* **判定标准**：
  - 逐步比对两种路径的每一步 Logits 最大绝对误差小于 `1e-4`；
  - 生成的 Token 序列完全相同。

---

### 3. Lockstep 批处理请求隔离与数值对齐测试

* **对应源码**：[`src/mini_infra/verification/parity/compare_batch_lockstep.py`](../../src/mini_infra/verification/parity/compare_batch_lockstep.py)
* **核心目标**：验证齐步固定批处理（Fixed Batching）的**请求隔离（Request Isolation）**。不管一个请求与什么样的长短请求伴随组成 Batch、位于 Batch 的第几行、受到多长的左填充（Left Padding），其生成结果必须与该请求**完全单独隔离运行**时 100% 一致。

#### CLI 运行指令
```bash
# 默认运行（内置隔离矩阵比对：单请求基准、同批组合、逆序置换与不同 Padding）
mini-infra verify batch-lockstep

# 指定生成长度并输出详细测试 JSON
mini-infra verify batch-lockstep \
  --prompt "用一句话解释固定批处理。" \
  --max-new-tokens 8 \
  --dtype float32 \
  --output tmp/parity_lockstep.json
```

#### Pytest 运行指令
```bash
pytest tests/gpu/parity/test_lockstep_batch_parity.py
```

* **判定标准**：
  - 各置换组合中目标请求生成的 Token 序列与单请求独立运行完全一致；
  - Logits 最大绝对误差小于 `1e-4`。

---

### 4. scheduler 批调度一致性测试（Fixed vs Continuous vs Paged vs 串行）

* **对应源码**：[`src/mini_infra/verification/parity/compare_batch_scheduler.py`](../../src/mini_infra/verification/parity/compare_batch_scheduler.py)
* **核心目标**：验证在底层通用调度循环（三级队列 + `DenseScheduleBackend` / `PagedScheduleBackend`）下：
  1. **Fixed Batching（波次准入）** 与单请求串行调度（`max_running_requests=1`）输出完全一致；
  2. **Continuous Batching（空 batch slot 即准入，DenseSlotCache）** 与单请求串行调度输出完全一致；
  3. **Continuous Batching + Paged KV Cache（physical block pool + block table）** 与单请求串行调度输出完全一致；
  4. 检验长短请求混跑、短请求提前结束释放 batch slot / physical block、新请求动态 Prefill 入场、Cache 左侧扩容与 history mask / `context_lens` 掩码隔离等复杂调度时序下的正确性。

#### CLI 运行指令
```bash
# 默认参数运行（并发 batch slot 2，输出预算差异化测试）
mini-infra verify batch-scheduler

# 指定参数与导出 Admission 轨迹
mini-infra verify batch-scheduler \
  --max-new-tokens 8 \
  --max-running-requests 2 \
  --dtype float32 \
  --output tmp/parity_scheduler.json
```

#### Pytest 运行指令
```bash
# 1. GPU 真实 Qwen3 权重端到端对齐测试
pytest tests/gpu/parity/test_scheduler_batch_parity.py

# 2. CPU 快速单元逻辑测试（无 GPU 环境可用，覆盖队列出入与 batch slot 分配）
pytest tests/unit/test_compare_scheduler_batch.py
```

* **判定标准**：
  - 打印并断言：
    - `fixed 与串行单请求一致: True`
    - `continuous 与串行单请求一致: True`
    - `fixed 与 continuous 一致: True`
    - `paged 与串行单请求一致: True`
    - `paged 与 continuous 一致: True`
  - 打印动态入场轨迹（如 `Continuous admission trace: [[0, 1], [2], [3]]`）。

---

### 5. Paged KV Cache 对照（Logits + 调度器输出）

Paged KV Cache 提供两个互补的对齐验证：单 prompt 的逐步 Logits 对照，以及 Scheduler 环境下的多请求输出对照。

**① Logits 逐 Token 对照**

* **对应源码**：[`src/mini_infra/verification/parity/compare_paged_cache_logits.py`](../../src/mini_infra/verification/parity/compare_paged_cache_logits.py)
* **核心目标**：以 Dense KV Cache 为基准，逐 Token 比对同一 prompt 在 Paged KV Cache（physical block pool + block table 寻址）路径下每一步 Decode 的 Logits 与 Greedy Token，验证 Paged 物理块写盘、按块读盘与 block table 映射的数值等价性。

#### CLI 运行指令
```bash
# 默认参数运行（2 条内置 prompt，生成 8 tokens，block_size=16）
mini-infra verify paged-cache-logits

# 自定义参数与精度阈值
mini-infra verify paged-cache-logits \
  --prompts "用一句话解释 Paged KV Cache。" \
  --max-new-tokens 8 \
  --block-size 4 \
  --dtype float32 \
  --logits-atol 1e-4 \
  --output tmp/parity_paged_logits.json
```

#### Pytest 运行指令
```bash
pytest tests/gpu/parity/test_paged_cache_parity.py -k logits
```

* **判定标准**：
  - 每一步 Decode 选出的 Greedy Token 与 Dense 完全一致；
  - 各步 Logits 的最大绝对误差 ≤ `--logits-atol`（`float32` 默认 `1e-4`，`bfloat16` 默认 `1.0`）。

**② 调度器输出对照**

* **对应源码**：[`src/mini_infra/verification/parity/compare_paged_cache_scheduler.py`](../../src/mini_infra/verification/parity/compare_paged_cache_scheduler.py)
* **核心目标**：在 Scheduler 连续批处理环境下，比对 Dense 与多个 Block Size 的 Paged KV Cache 在动态入场/退场、block table 追加与复用下的逐请求输出一致性与入场轨迹。

#### CLI 运行指令
```bash
# 默认参数运行（4 条内置 prompt，并发 2，block_size ∈ {4, 16}）
mini-infra verify paged-cache-scheduler

# 自定义参数并导出 JSON
mini-infra verify paged-cache-scheduler \
  --max-new-tokens 8 \
  --block-sizes 4 16 \
  --max-running-requests 2 \
  --dtype float32 \
  --output tmp/parity_paged_scheduler.json
```

#### Pytest 运行指令
```bash
pytest tests/gpu/parity/test_paged_cache_parity.py -k scheduler
```

* **判定标准**：
  - `all_match_dense` 为 `True`（每个 Block Size 的 Paged 输出均与 Dense 一致）；
  - 每个 Block Size 的 `admission_trace`（动态入场轨迹）非空；
  - 每个 Block Size 的 `block_reuse_count`（物理块复用计数）存在且 ≥ 0。

---

### 6. 迭代调度策略对照（`prefill-first` vs `token-budget`）

* **对应源码**：[`src/mini_infra/verification/parity/compare_schedulers_token_budget.py`](../../src/mini_infra/verification/parity/compare_schedulers_token_budget.py)
* **核心目标**：在错峰 Burst 负载（6 条请求，`arrival_ms = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]`，`max_running_requests = 4`）下，比较 `prefill-first` 与 `token-budget` 的逐请求 Token 输出一致性、单步调度轨迹（`prefill_first_trace` / `token_budget_trace`，含 `is_oversize_prefill`）与聚合指标（特别是 `max_consecutive_prefill_steps`）。

#### CLI 运行指令
```bash
# 默认参数运行（6 条错峰请求，并发 4，token_budget=64，block_size=16）
mini-infra verify schedulers-token-budget

# 自定义预算与导出 JSON 轨迹
mini-infra verify schedulers-token-budget \
  --max-new-tokens 8 \
  --token-budget 64 \
  --block-size 16 \
  --max-running-requests 4 \
  --dtype float32 \
  --output tmp/parity_schedulers.json
```

#### Pytest 运行指令
```bash
# 1. GPU 真实权重对照
pytest tests/gpu/parity/test_scheduler_batch_parity.py -k schedulers

# 2. CPU 单元逻辑测试
pytest tests/unit/test_compare_schedulers.py
```

* **判定标准**：
  - `tokens_match` 为 `True`（`prefill-first` 与 `token-budget` 逐请求生成序列完全一致）；
  - 打印并导出双方 `max_consecutive_prefill_steps` 与完整调度事件轨迹。

---

### 7. Chunked Prefill 对照（`token-budget` vs `chunked-prefill`）

* **对应源码**：[`src/mini_infra/verification/parity/compare_chunked_prefill.py`](../../src/mini_infra/verification/parity/compare_chunked_prefill.py)
* **核心目标**：在包含超长 Prompt 的 4 条错峰负载（`arrival_ms = [0.0, 1.0, 1.0, 1.0]`，`token_budget = 32`，`max_running_requests = 4`）下，对照 `token-budget`（软预算、触发 oversize prefill）与 `chunked-prefill`（硬预算、Prompt 分块）的逐请求输出一致性、单步调度轨迹与聚合指标，并逐请求验证分块 Prefill 与整段 Prefill 的首 Token Logits 最大绝对误差。

#### CLI 运行指令
```bash
# 默认参数运行（4 条长短悬殊错峰请求，并发 4，token_budget=32，block_size=16）
mini-infra verify chunked-prefill

# 自定义预算与导出 JSON 结果
mini-infra verify chunked-prefill \
  --max-new-tokens 8 \
  --token-budget 32 \
  --block-size 16 \
  --max-running-requests 4 \
  --dtype float32 \
  --logits-atol 1e-4 \
  --output tmp/parity_chunked_prefill.json
```

#### Pytest 运行指令
```bash
# 1. GPU 真实权重对照
pytest tests/gpu/parity/test_scheduler_batch_parity.py -k chunked_prefill

# 2. CPU 单元逻辑测试
pytest tests/unit/test_compare_chunked_prefill.py
```

* **判定标准**：
  - `tokens_match` 为 `True`（`token-budget` 与 `chunked-prefill` 逐请求生成序列完全一致）；
  - `float32` 下 `max_first_token_logit_error <= logits_atol`（默认 `1e-4`）；
  - `chunked-prefill` 的 `num_oversize_prefill_steps == 0` 且 `max_prefill_step_tokens <= token_budget`。

---

### 8. Prefix Caching 对照（prefix caching 关闭 vs 开启）

* **对应源码**：[`src/mini_infra/verification/parity/compare_prefix_caching.py`](../../src/mini_infra/verification/parity/compare_prefix_caching.py)
* **核心目标**：对应参考实现 chapter_08 的 `run_inference.py` / `compare_prefix_cache.py`。以单组 4 条自然语言请求同时覆盖冷启动、整块对齐前缀命中（288 Token）、非对齐前缀截断（290 → 288 Token）与「不同首块 + 相同后续 17 个满块」防误命中（命中 0 并触发紧凑池惰性淘汰），唯一变量为 `PagedCacheConfig.is_prefix_caching_enabled`，直接在 `chunked-prefill` 调度循环中对照逐请求输出、首 Token Logits 最大绝对误差、命中 Token 数、实际 Prefill 计算量与排空后的 physical block 引用归零。

#### CLI 运行指令
```bash
# 默认参数运行
mini-infra verify prefix-caching

# 自定义参数与导出 JSON 结果
mini-infra verify prefix-caching \
  --max-new-tokens 8 \
  --token-budget 64 \
  --block-size 16 \
  --max-running-requests 1 \
  --dtype float32 \
  --logits-atol 1e-4 \
  --output tmp/parity_prefix_caching.json
```

#### Pytest 运行指令
```bash
# 1. GPU 真实权重对照（含三种 batching policy × 并发 1/2 与单请求串行的逐 Token 对齐）
pytest tests/gpu/parity/test_scheduler_batch_parity.py -k prefix_caching

# 2. CPU 单元逻辑测试
pytest tests/unit/test_compare_prefix_caching.py tests/unit/test_prefix_caching.py
```

* **判定标准**：
  - 启用 prefix caching 后至少一个请求命中（默认内置 4 条请求在单并发、`block_size=16` 下精确命中 `[0, 288, 288, 0]`）；
  - `float32` 下 `tokens_match` 为 `True`；
  - `max_first_token_logit_error <= logits_atol`（`float32` 默认 `1e-4`，`bfloat16` 默认 `1.0`），且运行排空后全部 physical block 引用计数归零。

---

## 三、一键运行全部验证

### 1. 使用 CLI 依次执行全部对齐命令
```bash
# 执行全部 9 项精度对齐命令
mini-infra verify transformers
mini-infra verify cache
mini-infra verify batch-lockstep
mini-infra verify paged-cache-logits
mini-infra verify paged-cache-scheduler
mini-infra verify batch-scheduler
mini-infra verify schedulers-token-budget
mini-infra verify chunked-prefill
mini-infra verify prefix-caching
```

### 2. 使用 Pytest 一键回归全部 GPU 对齐测试
```bash
# 运行 tests/gpu/parity 下的全部测试文件
pytest tests/gpu/parity/
```

### 3. CPU 环境快速冒烟自检
```bash
# 无 GPU 时验证调度器与各组件逻辑完备性
mini-infra verify smoke
pytest tests/unit/test_compare_scheduler_batch.py tests/unit/test_compare_schedulers.py
```

---

## 四、工程要点与常见问题 (FAQ)

### Q1: 为什么多数 Parity 验证默认使用 `float32`，而推理默认是 `bfloat16`？
> [!IMPORTANT]
> `bfloat16` 仅有 7 位尾数位（有效数字仅 2~3 位）。当不同批处理策略改变了 Batch 大小或 Tile 划分时，GPU Tensor Core 内部的 GEMM 累加顺序微调会带来 $\sim 10^{-3}$ 的浮点舍入差异。在 Greedy Decoding 临界点，极微小的浮点波动可能引起个别近义词的 Top-1 翻转（如“程序”与“参数”）。
> 
> 因此，**对齐验证默认强制使用 `float32`，以排除硬件低精度截断噪声，专注于验证调度逻辑、KV Cache 寻址与 Attention Mask 的绝对算法正确性**。

### Q2: 遇到 `ModuleNotFoundError: No module named 'transformers'` 怎么办？
Transformers 对照测试需要额外依赖，可通过以下命令安装：
```bash
pip install -e ".[compare]"
```
如果环境未安装，`pytest tests/gpu/parity/test_transformers_parity.py` 会自动跳过（skip），不会导致测试套件红灯。

