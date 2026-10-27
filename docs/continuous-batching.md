# 连续批处理与 Serving 基准测试指南 (Continuous Batching & Serving Benchmark Guide)

在多用户、动态并发的真实在线大模型服务（LLM Serving）场景中，传统的**固定批处理（Fixed Batching）**存在严重的效率瓶颈。本项目在保留原生 PyTorch 简洁性与显式语义的同时，实现了工业界核心的 **Continuous Batching（连续批处理 / 迭代级调度）**，并配套了针对在线并发场景的受控评测基准套件 `mini-infra-bench-serving`。

---

## 一、为什么需要 Continuous Batching？

在固定批处理模式下，一批请求（Wave）必须同时进场，并等待该批中最长请求生成结束才能整体释放显存并接入下一批请求。这会导致两大核心痛点：

1. **队头阻塞（Head-of-Line Blocking）与计算空泡**：
   * 当长短请求混合时（如短请求输出 8 Tokens，长请求输出 64 Tokens），短请求在第 8 步即结束，但在后续 56 步中该槽位只能持续计算无用的 Padding Token，计算与显存槽位利用率急剧下降。
2. **动态到达（Dynamic Arrival）无法即时响应**：
   * 在线请求随时间持续到达。固定批处理无法将新到达的请求动态插入正在运行的批次中，新请求必须在外部队列等待当前整个批次全部结束，导致排队延迟与首字延迟（TTFT）恶化。

**Continuous Batching** 将调度颗粒度从“请求级（Request-level）”下沉到“迭代级（Iteration-level）”：
* **即时释放**：任何请求一旦生成结束（达到最大 Token 预算或命中 EOS），立即归还槽位。
* **即时入场**：每一步 Decode 前检查空闲槽位与就绪队列，将新到达的请求 Prefill 激活并动态组批，最大化 GPU 计算利用率。

---

## 二、架构设计与核心实现

本项目 Continuous Batching 实现遵循工业级模块化 Serving 引擎分层（显存管理位于 `src/mini_infra/memory/`，调度核心位于 `src/mini_infra/core/`，执行引擎位于 `src/mini_infra/engine/`），保持零三方 C++ 扩展、纯 PyTorch 原生张量实现的架构原则：

### 1. 显存管理：`DenseSlotCache`

与单次生成的单体 `KVCache` 不同，Continuous Batching 需要在一个固定的并发容量（`max_running_requests`）下管理多个动态进出的请求：

* **静态槽位预分配**：在初始化时预分配 `[capacity, num_kv_heads, max_length, head_dim]` 的稠密 Key/Value 缓存张量，避免自回归过程中发生频繁的动态内存申请与显存碎片化。
* **右对齐与历史掩码（Right-Aligned & History Mask）**：
  * 请求进场时，其 Prompt 的 KV 被放置在槽位右侧，并由 `DenseSlotCache.mask` 统一管理历史有效位置（类似因果掩码的批扩展）。
  * 当更长 Prompt 入场导致槽位全局长度扩展时，自动对所有活动槽位的历史掩码左侧补 0，无需搬移 KV 缓存本身。
* **槽位即时复用**：请求结束后，`slots.release(slot_index)` 在常数时间内清空该槽位掩码与状态，使得后续请求可以立即无缝复用该物理空间。

### 2. 调度器与时钟模拟：`ArrivalQueue` & `run_scheduler`

* **FCFS 就绪队列**：维护逻辑时间线 `arrival_ms`。队列中的请求在 `clock_ms >= arrival_ms` 时就绪，严格按照先来先服务（First-Come-First-Served）原则出队。
* **Prefill 优先级策略**：
  * 若当前有空闲槽位且队列中有就绪请求，优先执行新请求的 Prefill（可多请求拼接左 Padding 批处理），将其写入 `DenseSlotCache`，并产生第 1 个输出 Token。
  * 若当前没有空闲槽位或无就绪请求，则针对所有活跃槽位并发执行一步单 Token Decode。
* **零静默回退（Zero Silent Fallback）**：
  * 若用户通过 `Mode(batching_method=BatchingMethod.CONTINUOUS)` 调用，但未开启 `KV_CACHE` 或使用了不支持的特性，引擎会显式抛出 `ValueError`，杜绝任何静默回退到单请求或固定批处理的行为。

---

## 三、Serving 基准测试套件 (`mini-infra-bench-serving`)

为了科学量化 Continuous Batching 在不同在线负载模式下的收益与开销，项目提供了独立的 Serving 评测入口 `mini-infra-bench-serving`（独立于单请求的 `mini-infra-bench-case` / `mini-infra-bench-suite`）。

### 1. 四大受控实验套件 (Suites)

| 套件名称 | 负载特征 | 核心评测目的 |
| :--- | :--- | :--- |
| `fixed-vs-continuous-variable-output-length` | $t=0$ 同时到达，8/64 交替输出预算 | **量化长短请求混合下的槽位复用加速比**：短请求结束后槽位立即让渡给等待队列中的请求，显著缩短总完成时间（Makespan）。 |
| `fixed-vs-continuous-uniform-output-length` | $t=0$ 同时到达，输出预算完全一致（32 Tokens） | **对照基线（Negative Control）**：所有请求几乎同时结束，理论上 Continuous 与 Fixed 性能接近，用于排查调度框架本身的额外常数开销。 |
| `fixed-vs-continuous-staggered-arrivals` | 到达时间均匀错开（如 $0, 20, 40, 60\dots$ ms） | **量化动态插入收益**：评测连续批处理能否在 Decode 循环中平滑吸纳新入场请求，降低排队延迟。 |
| `fixed-vs-continuous-prefill-interference` | 高频突发到达（Burst Arrivals） | **量化 Prefill 打断开销**：当新请求入场执行计算密集型 Prefill 时，正在 Decode 的老请求步延迟（ITL P95）受干扰程度。 |

### 2. 核心 Serving 评估指标

* **Makespan (ms)**：处理完全部批次请求的总耗时。
* **Continuous Makespan Speedup**：连续批处理相比固定批处理的总执行加速比（$\text{Makespan}_{\text{fixed}} / \text{Makespan}_{\text{continuous}}$）。
* **Execution Slot Utilization (%)**：有效生成的 Token 数除以 GPU 实际执行的槽位数，精确反映计算空泡（Padding）比例。
* **TTFT P50 / P95 (ms)**：首字延迟（Time to First Token）分位数，包含排队等待时间。
* **ITL P50 / P95 (ms)**：Token 间延迟（Inter-Token Latency），反映解码过程的流畅度与抗打断性。
* **Cache Management Overhead (%)**：稠密槽位状态维护与掩码重对齐在整个循环中所占的时间比例。

---

## 四、快速上手与命令行指南

### 1. 运行单用例独立评测（Single-Case Runner）

```bash
# 单独评测 fixed-vs-continuous-variable-output-length 负载在 Continuous Batching 下的表现（不跑对比）
mini-infra-bench-serving-case --tag dev --suite fixed-vs-continuous-variable-output-length --batching-method continuous-batching
```

### 2. 运行单一套件对比测试（Suite Runner）

```bash
# 在 GPU 上评测长短请求混合场景下的连续批处理加速比（自动对比 Fixed 与 Continuous）
mini-infra-bench-serving-suite --tag dev --suite fixed-vs-continuous-variable-output-length --warmup 1 --repeats 3
```

输出示例：
```text
[bench-serving-suite] suite=fixed-vs-continuous-variable-output-length max_running=4 warmup=1 repeats=3 ...
  Fixed     : makespan 2784.10 ms | 57.5 tok/s | 2.87 req/s | slot_util 62.5% | ttft_p95 1431.17 ms | itl_p95 51.22 ms
  Continuous: makespan 2416.91 ms | 66.2 tok/s | 3.31 req/s | slot_util 74.1% | ttft_p95 1082.18 ms | itl_p95 86.73 ms | cache 0.9% | speedup 1.15x
结果已写入 benchmarks/suites/serving/fixed-vs-continuous-variable-output-length/<commit>_dev.json
```

### 3. 运行特定受控 Serving 套件

```bash
# 运行指定的受控评测套件，生成 Serving 评测报告
mini-infra-bench-serving-suite --tag v0.3.0 --suite fixed-vs-continuous-variable-output-length
```

### 4. 热漂移规避（Thermal Drift Mitigation）

为消除连续基准测试中 GPU 升温与功耗降频对公平性带来的偏差，`mini-infra-bench-serving` 内建了交替执行机制（Run Alternation）：在奇数轮次按 `Fixed -> Continuous` 顺序执行，在偶数轮次按 `Continuous -> Fixed` 顺序执行，并在每次重复之间强制进行垃圾回收与显存重置。

---

## 五、正确性守门与自动化测试

项目遵循严格的正确性验证标准，包含以下层级的自动化检验：

1. **CPU 单元测试**：
   * `pytest tests/unit/test_scheduler.py`：验证到达队列 FCFS 出队、时间窗口推进与空队列边界。
   * `pytest tests/unit/test_slot_cache.py`：验证槽位分配、右对齐扩展、掩码更新与释放。
   * `pytest tests/unit/test_continuous.py`：在随机小模型上验证与串行 `generate` 逐位一致，覆盖变长 Prompt、EOS 提前截断、容量溢出排队等边缘场景。
2. **GPU 真实权重等价性测试**：
   * `pytest tests/gpu/parity/test_continuous_parity.py`：在真实的 Qwen3-0.6B 权重上，验证 Continuous Batching 和 Fixed Batching 的生成结果与单请求串行生成 100% 逐 Token 严格对齐。
3. **Smoke 快速冒烟**：
   * `mini-infra-smoke` 已集成 `check_continuous_batching`，数秒内完成 CPU 端全链路自检。
