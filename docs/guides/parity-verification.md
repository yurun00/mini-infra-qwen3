# 数值对齐与等价性验证指南 (Parity Verification Guide)

本指南汇总了 [`src/mini_infra/verification/parity/`](../../src/mini_infra/verification/parity) 目录下的全部精度对齐与正确性测试指令。

在推理底座的演进过程中，每一次引入新特性（如 KV Cache、固定批处理、连续批处理调度器）都必须经过严格的**数值等价性（Parity）**与**请求隔离（Request Isolation）**验证，确保性能优化不以牺牲模型输出精度为代价。

---

## 一、测试全景速查表

| 验证模块 | 源码文件 | CLI 指令 (`mini-infra verify ...`) | Pytest 指令 | 默认精度 | 核心检验目标 |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **1. 参考实现对齐** | `compare_transformers.py` | `mini-infra verify transformers` | `pytest tests/gpu/parity/test_transformers_parity.py` | FP16/BF16 | 验证手写 Qwen3 与 Hugging Face 官方分词、Logits 与输出 100% 对齐 |
| **2. KV Cache 等价** | `compare_cache.py` | `mini-infra verify cache` | `pytest tests/gpu/parity/test_cache_parity.py` | `float32` | 验证引入 KV Cache 增量 Decode 与无 Cache 全量重算的数值完全一致 |
| **3. Lockstep 批隔离** | `compare_lockstep_batch.py` | `mini-infra verify lockstep-batch` | `pytest tests/gpu/parity/test_lockstep_batch_parity.py` | `float32` | 验证固定批处理在不同 Batch Size、顺序与 Padding 长度下的严格请求隔离 |
| **4. Scheduler 批调度** | `compare_scheduler_batch.py` | `mini-infra verify scheduler-batch` | `pytest tests/gpu/parity/test_scheduler_batch_parity.py` | `float32` | 验证 Continuous 与 Fixed 调度下的动态槽位绑定、Mask 管理与串行基准对齐 |

> [!NOTE]
> 所有命令既支持直接平铺调用（如 `mini-infra verify cache`），也支持语义分层调用（如 `mini-infra verify parity cache`）。

---

## 二、各项验证详细指令与参数

### 1. Hugging Face Transformers 对照测试

* **对应源码**：[`src/mini_infra/verification/parity/compare_transformers.py`](file:///home/yurun/projects/mini-infra-qwen3/src/mini_infra/verification/parity/compare_transformers.py)
* **核心目标**：以 Hugging Face `transformers`（`AutoModelForCausalLM`）为黄金参考（Golden Truth），比对分词、Prefill 与逐步 Decode 下的 Logits 最大绝对误差，确保自研模型架构无算子缺陷。

#### CLI 运行指令
```bash
# 默认参数运行（生成 32 tokens）
mini-infra verify transformers

# 自定义提示词与最大绝对误差阈值
mini-infra verify transformers \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32 \
  --logits-atol 0.25
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

* **对应源码**：[`src/mini_infra/verification/parity/compare_cache.py`](file:///home/yurun/projects/mini-infra-qwen3/src/mini_infra/verification/parity/compare_cache.py)
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
  --logits-atol 1e-4
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

* **对应源码**：[`src/mini_infra/verification/parity/compare_lockstep_batch.py`](file:///home/yurun/projects/mini-infra-qwen3/src/mini_infra/verification/parity/compare_lockstep_batch.py)
* **核心目标**：验证齐步固定批处理（Fixed Batching）的**请求隔离（Request Isolation）**。不管一个请求与什么样的长短请求伴随组成 Batch、位于 Batch 的第几行、受到多长的左填充（Left Padding），其生成结果必须与该请求**完全单独隔离运行**时 100% 一致。

#### CLI 运行指令
```bash
# 默认运行（内置隔离矩阵比对：单请求基准、同批组合、逆序置换与不同 Padding）
mini-infra verify lockstep-batch

# 指定生成长度并输出详细测试 JSON
mini-infra verify lockstep-batch \
  --prompt "用一句话解释固定批处理。" \
  --max-new-tokens 8 \
  --dtype float32 \
  --output tmp/parity_lockstep.json
```
*(注：支持向后兼容别名 `mini-infra verify batch`)*

#### Pytest 运行指令
```bash
pytest tests/gpu/parity/test_lockstep_batch_parity.py
```

* **判定标准**：
  - 各置换组合中目标请求生成的 Token 序列与单请求独立运行完全一致；
  - Logits 最大绝对误差小于 `1e-4`。

---

### 4. Scheduler 批调度一致性测试（Fixed vs Continuous vs 串行）

* **对应源码**：[`src/mini_infra/verification/parity/compare_scheduler_batch.py`](file:///home/yurun/projects/mini-infra-qwen3/src/mini_infra/verification/parity/compare_scheduler_batch.py)
* **核心目标**：验证在底层通用调度循环（三级队列 + ModelRunner + DenseSlotCache 槽位管理）下：
  1. **Fixed Batching（波次准入）** 与单请求串行调度（`max_running_requests=1`）输出完全一致；
  2. **Continuous Batching（空槽即准入）** 与单请求串行调度输出完全一致；
  3. 检验长短请求混跑、短请求提前命中 EOS 释放槽位、新请求动态 Prefill 入场、Cache 左侧扩容与历史 Mask 隔离等复杂调度时序下的正确性。

#### CLI 运行指令
```bash
# 默认参数运行（并发槽位 2，输出预算差异化测试）
mini-infra verify scheduler-batch

# 指定参数与导出 Admission 轨迹
mini-infra verify scheduler-batch \
  --max-new-tokens 8 \
  --max-running-requests 2 \
  --dtype float32 \
  --output tmp/parity_scheduler.json
```
*(注：支持向后兼容别名 `mini-infra verify continuous`)*

#### Pytest 运行指令
```bash
# 1. GPU 真实 Qwen3 权重端到端对齐测试
pytest tests/gpu/parity/test_scheduler_batch_parity.py

# 2. CPU 快速单元逻辑测试（无 GPU 环境可用，覆盖队列出入与槽位分配）
pytest tests/unit/test_compare_scheduler_batch.py
```

* **判定标准**：
  - 打印并断言：
    - `fixed 与串行单请求一致: True`
    - `continuous 与串行单请求一致: True`
    - `fixed 与 continuous 一致: True`
  - 打印动态入场轨迹（如 `Continuous admission trace: [[0, 1], [2], [3]]`）。

---

## 三、一键运行全部验证

### 1. 使用 CLI 依次执行全部对齐命令
```bash
# 执行全部 4 项精度对齐命令
mini-infra verify transformers
mini-infra verify cache
mini-infra verify lockstep-batch
mini-infra verify scheduler-batch
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
pytest tests/unit/test_compare_scheduler_batch.py
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

