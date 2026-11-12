# AI Inference Infra Engineering Rules (Core & Always-On)

`mini-infra-qwen3` 推理与调度引擎核心规则。子域细则由 `.agents/rules/` 按需挂载。

---

## 一、Agent 工程行为准则

1. **契约与不变量**
   - **张量契约**：明确接口输入/输出的维度语义（如 `[B, H_kv, S, D]`）、`dtype`、设备位置与内存连续性。
   - **状态不变量**：保持调度队列、batch slot / block table 索引、序列长度与 Attention Mask 之间的守恒关系。
   - **瓶颈意识**：明确改动落在 Compute-Bound（Prefill）还是 Memory-Bandwidth / Overhead-Bound（Decode 与调度准备）。

2. **正交演进与最小爆炸半径**
   - 单次改动保持正交，不得混杂架构重构、新特性与评测基准变更。
   - **基线可对照性**：新增优化必须经模式或策略抽象接入，**不得破坏既有 Baseline 路径的可运行性与可对比性**。
   - **零黑盒依赖**：核心运行时（`model/`、`worker/`、`core/`、`engine/`、`entrypoints/`）只用原生 Python 与 PyTorch，**严禁引入 `transformers`、`tokenizers` 等外部封装库**（`transformers` 仅限 `verification/parity/` 与 `tests/gpu/parity/` 作 Golden Reference）。

3. **YAGNI：禁废弃代码与 dead code**
   - **严禁保留兼容旧逻辑的废弃代码**：废弃 member field、过时 method name、兼容性 property、无用参数默认值、forwarding wrapper。
   - **严禁新增 dead code**：无调用点的函数/方法、无读取点的字段。
   - **测试专有代码归位与标注**：仅供 `tests/` 调用的代码优先迁入测试文件；确需留在生产代码中的，必须在注释或 docstring 显式标注 `Test-only`，严禁伪装为通用业务接口。

4. **结果指标变更的端到端联动守恒**
   - 增删改任何结果指标（延迟、吞吐、显存统计、调度事件字段等）必须同步更新：① Pydantic Schema（`bench/result/schema.py`）与受控基准数据（`benchmarks/results/history/`、`benchmarks/golden/`）；② 绘图脚本（`src/mini_infra/bench/plotting/`），严防 KeyError、字段取空或图表损坏。

5. **增量守门，存量债务显式标记**
   - **增量 100% 达标**：新建文件、新函数与本次 Diff 区域必须遵守全部规则；**无用户明确指令不得发起超范围大面积重构**，仅在直接触碰或任务明确要求时推进收敛（且须守住基线可对照性）。
   - **债务标记契约**：存量历史债务以注释显式声明，遇到即跳过告警且不擅自修改——单行/局部用 `# infra-legacy: <reason>`（或 `# infra-legacy(<rule>): <reason>`），文件顶部用 `# infra-legacy-file: <reason>`。

6. **边界校验，内部信任**
   - 只在 `entrypoints/` 与外部 API 边界做入参校验；内部模块信任自身契约，**严禁加 `None` 兜底、`try/except` 吞异常或冗余校验**。

---

## 二、系统架构与不变量

单向依赖栈，对齐现代推理引擎（如 vLLM V1）的前后端解耦、控制流（CPU）与数据流（GPU）分离：

- **分层由依赖决定，不由目录名决定**：模块只要 `import torch` 并持有张量即属 data plane；**只能依赖下游层，严禁反向依赖**，新增模块先声明归属层。
- **输入协议共享层（`inputs/`）**：`PromptInput` 等纯 Python 输入协议类型，**不 `import torch`、不持有张量、不含 tokenize**，供 `entrypoints/`（①）与 `engine/`（②）共同依赖。
- **栈下公共设施（`device.py`、`profiling/`、`utils.py`）**：位于五层栈之下的横切基础设施，栈内任意层可依赖，但**严禁反向依赖栈内任何模块**。

```text
① Entrypoints  (接入层)         用户级入口（generate / GenerationOptions）：参数校验、按 Mode 分发、结果聚合
      ↓
② Engine       (协调层)         驱动 Schedule -> Execute -> Process Outputs 主循环；维护生命周期与时钟；承载 Tokenizer 与输入规约
      ↓
③ Core         (control plane)  Request State Machine、Queue Scheduler、Batch Slot / Logical Block Allocator
      ↓
④ Worker       (data plane)     Input Tensor 组装、Physical KV Cache Pool 读写、Forward 触发与 Device Sampler
      ↓
⑤ Model        (compute plane)  Pure Tensor-In, Tensor-Out：无状态神经网络，仅消费与产出裸 Tensor
```

**各层职责与硬性不变量**

- **① `entrypoints/`**：用户级协议解析与入参校验、调用输入规约、按 `Mode` 分发引擎并聚合跨请求结果。**严禁执行算子计算、分配物理显存、或参与调度排队与 Cache 寻址**。
- **② `engine/`**：承担 `EngineCore` 编排职责，串联 Core（获取调度计划）与 Worker（触发执行），推进请求状态流转、EOS 判定与打点计时（兼含纯 CPU 的 `tokenizer.py` 与 `preprocessing.py`）。**严禁分配物理 KV 显存、拼装模型输入 Tensor 或直接调用底层算子；严禁被 `core/`、`worker/`、`model/` 反向依赖**。
- **③ `core/`**：**100% 纯 Python / CPU 实现，零 `torch` 依赖**。维护请求状态机（`WAITING -> RUNNING -> FINISHED`）、三级调度队列（`ArrivalQueue -> WaitingQueue -> RunningQueue`）与 KV Cache 逻辑映射（batch slot 分配与 `KVCacheManager` block table）。只操作原生整数索引与元数据，**严禁持有 Tensor、分配物理显存、触发 CUDA 同步或调用模型前向**。
- **④ `worker/`**：GPU 算力与物理显存池专职执行者。Runner 在 Host 侧将调度计划批量组装为 GPU 输入 Tensor、驱动 Model Forward 并在设备端采样（Sampler 属本层，不属 `model/`）；Cache Pool 按物理索引（`batch_slot_idx` / `physical_block_id`）原地读写 KV Cache。**只认物理索引与 Tensor，不感知 `request_id` 或业务请求对象；严禁掺杂排队策略或请求状态机；物理池严禁承担输入组装**。
- **⑤ `model/`**：无状态张量计算图（Pure Tensor-In, Tensor-Out），实现 Transformer、Norm、RoPE、Attention 与 MLP 算子。对调度、请求状态与外部 Cache 容器**完全无感知**，**严禁 `import` 或 `isinstance` 任何外部 Cache 容器**，KV 历史仅以裸 Tensor 进出；分词等 CPU 文本处理严格隔离在外。

**严格请求隔离**：无论单独运行、与任意长短请求拼 Batch、处于任意 batch slot / physical block、经历动态抢占/扩缩容或不同长度 Padding，有效 Token 的 `position_ids` 与因果可见范围（Attention Mask / block table 寻址）必须与独立运行时**严格等价**，绝不允许跨请求数值污染。

### 概念统一命名（Concept Naming Registry）

全仓每个架构概念只有一个英文规范名（与代码标识符对齐）；文档与注释**严禁使用中文别名或近义混写**，**不同概念严禁共用一名**。新增概念先登记再写代码。

| 规范名 | 代码锚点 | 定义 |
|---|---|---|
| batch slot | `RequestState.batch_slot_idx`、`RunningQueue.free_batch_slots`、`DenseSlotCache` 首维、`LockstepState` | Dense / Lockstep 路径的请求级常驻并发位：并发容量空间里的一个常驻 batch 行，一个活跃请求独占一个 |
| slot（physical slot） | `PagedAttentionMetadata.slot_mapping`、`physical_block_id * block_size + offset` | Paged 路径的 token 级物理存储位：展平 physical block pool 的行号，本步新 token K/V 的写入地址 |
| physical block pool | `PagedKVCachePool` | 逐层 `(key_pool, value_pool)` 物理块张量池，布局 `[num_blocks, block_size, H_kv, D]` |
| free block queue | `FreeKVCacheBlockQueue` | 物理块 ID 的 FIFO 空闲队列；只存整数 ID，不持有显存张量 |
| block table | `block_tables` 张量 / `KVCacheManager` | 请求级「logical block → physical block」映射，右侧以 null block（`0`）补齐 |
| physical block / logical block / null block | `physical_block_id` / 逻辑块号 / `NULL_BLOCK_ID = 0` | 物理块 = 池内实际显存块；逻辑块 = 某请求的第 k 个块；null block = 永不分配、仅用于补齐寻址 |
| row | `row_idx`（如 `PagedRunner` Prefill 散写行） | 单步前向临时 batch 张量的行号；与 batch slot（常驻并发位）不可混用 |
| attention metadata | `Qwen3Model.forward` 的 `attn_metadata` / `DenseAttentionMetadata` / `PagedAttentionMetadata` | 模型前向的注意力寻址与存储描述符，按类型分派：`DenseAttentionMetadata.past_key_values` 承载逐层 dense 历史键值；`PagedAttentionMetadata` 承载 paged 寻址（`slot_mapping` / `block_tables` / `context_lens`）与逐层 physical block pool（`kv_pools`） |
| sampler | `worker/sampler.py` 的 `greedy_sample` | data plane 设备端贪心采样算子：只吃 logits 张量、吐 token 张量，不感知请求状态与计时 |
| history mask | `DenseSlotCache.history_mask` | 标记各 batch slot 内历史 token 有效位置（右对齐）的布尔掩码 |
| scheduler | `ArrivalQueue` / `WaitingQueue` / `RunningQueue` / `run_scheduler` | 调度器概念词统一小写；`Lockstep` 因对应 `LockstepState` / `LockstepRunner` 类保留大写 |
| control plane / data plane / compute plane | 分层图 ③④⑤ | 模块 `import torch` 且持有张量即 data plane；纯 CPU 决策即 control plane；纯张量算子计算即 compute plane |

**命名与反混用红线**

- **禁中文别名**：physical block pool 不写「物理块池 / 块池」，block table 不写「页表」，batch slot 不写「请求槽位 / 执行槽」；文档与注释用规范名，首次出现可附代码标识符。
- **三个「行/位」语义严禁互混**：`slot_mapping` 的 token 级 slot（写入地址）≠ dense 的 batch slot（请求并发位）≠ 单步前向的 row（临时 batch 行）。

---

## 三、必跑验证门禁

改动完成后、宣告任务结束前，必须通过以下秒级 CPU 闭环（无需 GPU 或下载权重）：

```bash
ruff check . && ruff format --check .
mini-infra verify smoke && pytest -m "not gpu"
```

改动涉及算子、KV Cache、批处理或调度器逻辑且当前环境具备 GPU 时，追加遵循 `.agents/rules/parity-and-benchmarking.md` 执行 `float32` 数值等价性验证（`pytest -m gpu`）与零性能退化门禁（`mini-infra bench check-regression`）。
