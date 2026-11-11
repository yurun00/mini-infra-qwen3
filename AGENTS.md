# AI Inference Infra Engineering Rules (Core & Always-On)

`mini-infra-qwen3` 推理与调度引擎核心规则。子域细则由 `.agents/rules/` 按需挂载。

---

## 一、Agent 工程行为准则

1. **契约与不变量**
   - **张量契约**：接口输入/输出的维度语义（如 `[B, H_kv, S, D]`）、`dtype`、设备位置、内存连续性。
   - **状态不变量**：调度队列、batch slot / block table 索引、序列长度与 Attention Mask 之间的守恒关系。
   - **瓶颈意识**：改动落在 Compute-Bound（Prefill）还是 Memory-Bandwidth / Overhead-Bound（Decode 与调度准备）。

2. **正交演进与最小爆炸半径**
   - 单次改动保持正交，不得混合重构、新特性与评测基准变更。
   - **基线可对照性**：新增优化必须经模式或策略抽象接入，**不得破坏既有 Baseline 路径的可运行性与可对比性**。
   - **零黑盒依赖**：核心运行时（`model/`、`worker/`、`core/`、`engine/`、`entrypoints/`）只用原生 Python 与 PyTorch，**严禁引入 `transformers`、`tokenizers` 等外部封装库**（`transformers` 仅限 `verification/parity/` 与 `tests/gpu/parity/` 作 Golden Reference）。

3. **YAGNI：禁废弃代码与 dead code**
   - 严禁为兼容旧逻辑保留废弃代码：废弃 member field、过时 method name、兼容性 property、无用参数默认值、forwarding wrapper。
   - 严禁新增 dead code：无调用点的函数/方法、无读取点的字段。
   - **测试专有代码显式标注**：生产代码中仅供测试（`tests/`）调用、未被任何生产链路引用的函数/方法/类，必须在 docstring 中显式注明使用边界（如 `专供测试使用 / Test-only`），严禁伪装为通用业务接口。

4. **结果指标变更的端到端联动守恒**
   - **增删改任何结果指标**（延迟、吞吐、显存统计、调度事件字段等）必须同步下游：
     1. **Schema 与 JSON 数据**：Pydantic 模型（`bench/schema.py`）、基准数据文件（`benchmarks/results/history/`、`benchmarks/golden/`）。
     2. **绘图脚本**：`src/mini_infra/bench/plotting/`，防止指标增删改名导致 KeyError、字段取空或图表损坏。

5. **增量守门，存量债务显式标记**
   - **增量 100% 达标**：新建文件、新函数、本次修改的 Diff 区域必须遵守全部架构不变量与工程规则。
   - **无用户明确指令不得发起超范围的大面积重构**。
   - **债务标记契约**：以注释显式声明历史债务，遇到即跳过告警且不擅自修改：
     - 单行或局部代码块：`# infra-legacy: <豁免原因或收敛计划>` 或 `# infra-legacy(<rule>): <reason>`
     - 文件级（置于文件顶部）：`# infra-legacy-file: <豁免原因>`
   - **仅直接触碰该文件/函数、或任务明确要求重构时推进收敛**，且必须守住基线可对照性。

6. **边界校验，内部信任**
   - 只在 `entrypoints/` 与外部 API 边界做入参校验；内部模块信任自身契约，**严禁加 `None` 兜底、`try/except` 吞异常或冗余校验**。

---

## 二、系统架构与不变量

单向依赖栈，对齐现代高性能大模型推理引擎（如 vLLM V1）的前后端解耦、控制流（CPU）与数据流（GPU）分离：

- **分层由依赖决定，不由目录名决定**：模块只要 `import torch` 并持有张量即属 data plane，无论物理位于哪个包。
- **只能依赖下游层，严禁反向依赖**；新增模块先声明归属哪一层。
- **输入协议共享层（`inputs/`）**：`PromptInput` 等纯 Python 输入协议类型，**不 `import torch`、不持有张量、不含 tokenize**；被 `entrypoints/`（①）与 `engine/`（②）共同依赖。tokenize 与其规约实现仍属 `engine/`。
- **栈下公共设施（`device.py`、`profiling/`、`utils.py`）**：位于五层栈**之下**的横切基础设施，不属 ①-⑤ 任何一层。栈内任意层均可依赖，但它们**严禁反向依赖栈内任何模块**（否则单向依赖栈成环）。

```text
① Entrypoints  (接入层)      协议解析、参数校验、Tokenize / Detokenize、结果聚合
      ↓
② Engine       (协调层)      驱动 Schedule -> Execute -> Process Outputs 主循环，维护生命周期与时钟
      ↓
③ Core         (control plane)  Request State Machine、Queue Scheduler、Batch Slot / Logical Block Allocator
      ↓
④ Worker       (data plane)     Input Tensor 组装、Physical KV Cache Pool 读写、Forward 触发与 Device Sampler
      ↓
⑤ Model        (compute plane)  Pure Tensor-In, Tensor-Out：无状态神经网络，仅消费与产出裸 Tensor
```

**各层职责与硬性不变量**

- **① `entrypoints/`（Entrypoints）**
  - **核心职责**：协议解析与入参校验；Tokenize / Detokenize；跨请求结果聚合与格式化输出。
  - **硬性不变量**：纯 Python 原生对象出入。**严禁执行 Tensor 操作、分配物理显存、或参与调度排队与 Cache 寻址**。

- **② `engine/`（Orchestrator）**
  - **核心职责**：承担 `EngineCore` 角色，驱动 `Schedule -> Execute -> Process Outputs` 主循环；串联 Core（获取调度计划）与 Worker（触发执行）；推进请求状态流转、EOS 判定与打点计时。
  - **硬性不变量**：编排者而非执行者。**严禁分配物理显存、拼装 GPU Tensor 或直接调用模型底层算子**；**严禁被下游层（`core/`、`worker/`、`model/`）反向依赖**。

- **③ `core/`（Control Plane: Scheduler & Logical Allocator）**
  - **核心职责**：纯 CPU 逻辑决策中枢。维护 Request State Machine（`WAITING -> RUNNING -> FINISHED`）；执行排队优先级与抢占调度；管理 KV Cache 逻辑映射（维护 batch slot 分配与 block table，负责逻辑 Token 序列到 physical block 的映射）。
  - **硬性不变量**：**100% 纯 Python / CPU 实现，零 `torch` GPU 依赖**。只操作原生整数索引与元数据，**严禁持有 GPU Tensor、分配物理显存、触发 CUDA 同步或调用模型前向**。

- **④ `worker/`（Data Plane: Model Runner & Physical KV Cache Pool）**
  - **核心职责**：GPU 算力与物理显存池专职执行者。在 Host 侧将调度计划批量组装为 GPU 输入 Tensor（`input_ids`、`position_ids`、`slot_mapping` / `block_tables`）；按物理索引（显存池切片 / 物理块号：`batch_slot_idx` / `physical_block_id`）原地读写 KV Cache；触发 Model Forward 并在设备端完成采样（Sampler）。
  - **硬性不变量**：只认**物理索引（显存池切片 / 物理块号下标，不感知 `request_id` 或业务请求对象）与 Tensor**。**严禁掺杂业务排队策略、请求优先级或生命周期状态机**；物理池**严禁承担输入组装**（输入组装归属 Runner）；Sampler 属本层，不属 `model/`。

- **⑤ `model/`（Compute Plane: Pure Model Executor）**
  - **核心职责**：无状态张量计算图（Pure Tensor-In, Tensor-Out）。实现 Transformer 网络架构、Norm、RoPE、Attention Kernel 与 MLP 等底层数学算子。
  - **硬性不变量**：无状态计算。对上层调度、请求状态与外部 Cache 容器管理**完全无感知**，**严禁 import 或 `isinstance` 任何外部 Cache 容器**，KV 历史仅以裸 Tensor 形式进出；分词等 CPU 文本处理严格隔离在外。

**严格请求隔离**：无论单独运行、与任意长短请求拼 Batch、处于任意 physical slot / page、经历动态抢占/扩缩容或不同长度 Padding，其有效 Token 的 `position_ids` 与因果可见范围（Attention Mask / block table 寻址）必须与独立运行时**严格等价**，绝不允许跨请求数值污染。

### 概念统一命名（Concept Naming Registry）

**目的**：全仓每个架构概念只有一个英文规范名（与代码标识符对齐）；文档与注释严禁使用中文别名或近义混写；**不同概念严禁共用一名**。下表是权威对照，新增概念先登记再写代码。

| 规范名 | 代码锚点 | 定义 |
|---|---|---|
| batch slot | `RequestState.batch_slot_idx`、`RunningQueue.free_batch_slots`、`DenseSlotCache` 首维 | Dense 路径的请求级常驻并发位：并发容量（`capacity`）空间里的一个 batch 行，一个活跃请求独占一个 |
| slot（physical slot） | `PagedAttentionMetadata.slot_mapping`、`physical_block_id * block_size + offset` | Paged 路径的 token 级物理存储位：展平 physical block pool 的行号，本步新 token K/V 的写入地址（vLLM 同义） |
| physical block pool | `PagedKVCachePool` | 逐层 (key_pool, value_pool) 物理块张量池，布局 `[num_blocks, block_size, H_kv, D]` |
| free block queue | `FreeKVCacheBlockQueue` | 物理块 ID 的 FIFO 空闲队列；只存整数 ID，不持有显存张量 |
| block table | `block_tables` 张量 / `KVCacheManager` | 请求级「logical block → physical block」映射，右侧以 null block（0）补齐 |
| physical block / logical block / null block | `physical_block_id` / 逻辑块号 / `NULL_BLOCK_ID = 0` | 物理块 = 池内实际显存块；逻辑块 = 某请求的第 k 个块；null block = 永不分配、仅用于补齐寻址 |
| row | `row_idx`（如 `PagedRunner` Prefill 散写行） | 单步前向临时 batch 张量的行号；与 batch slot（常驻并发位）不可混用 |
| attention metadata | `Qwen3Model.forward` 的 `attn_metadata` / `DenseAttentionMetadata` / `PagedAttentionMetadata` | 模型前向的注意力寻址与存储描述符，按类型分派执行路径：`DenseAttentionMetadata.past_key_values` 承载逐层 dense 历史键值；`PagedAttentionMetadata` 承载 paged 寻址（`slot_mapping` / `block_tables` / `context_lens`）与逐层 physical block pool（`kv_pools`） |
| sampler | `worker/sampler.py` 的 `greedy_sample` | data plane 的设备端贪心采样算子：只吃 logits 张量、吐 token 张量，不感知请求状态与计时 |
| history mask | `DenseSlotCache.history_mask` | 标记各 batch slot 内历史 token 有效位置（右对齐）的布尔掩码 |
| scheduler | `ArrivalQueue` / `RunningQueue` / `run_scheduler` | 调度器概念词统一小写；`Lockstep` 因对应 `LockstepState` / `LockstepRunner` 类保留大写 |
| control plane / data plane / compute plane | 分层图 ③④⑤ | 模块 `import torch` 且持有张量即 data plane；纯 CPU 决策即 control plane；纯张量算子计算即 compute plane |

**命名与反混用红线**

- 概念词只允许表内英文规范名，**禁止中文别名**：physical block pool 不写「物理块池 / 块池」，block table 不写「页表」，batch slot 不写「请求槽位 / 执行槽」。
- **三个「行/位」语义严禁互混**：`slot_mapping` 的 token 级 slot（写入地址）≠ dense 的 batch slot（请求并发位）≠ 单步前向的 row（临时 batch 行）。
- 文档与注释用规范名，首次出现可附代码标识符（如 physical block pool（`PagedKVCachePool`））。

---

## 三、必跑验证门禁

改动完成后、宣告任务结束前，必须通过以下秒级 CPU 闭环（无需 GPU 或下载权重）：

```bash
ruff check . && ruff format --check .
mini-infra verify smoke && pytest -m "not gpu"
```

改动涉及算子、KV Cache、批处理或调度器逻辑且当前环境具备 GPU 时，追加遵循 `.agents/rules/parity-and-benchmarking.md` 执行 `float32` 数值等价性验证（`pytest -m gpu`）与零性能退化门禁（`mini-infra bench check-regression`）。
