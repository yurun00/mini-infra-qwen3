# AI Inference Infra Engineering Rules (Core & Always-On)

`mini-infra-qwen3`：从零手写的高性能大模型推理与调度引擎。目标是**高可读、确定性、高硬件利用率、零精度妥协**的系统级 Python/PyTorch 代码。子域细则由 `.agents/rules/` 按需挂载。

---

## 一、Agent 工程行为准则

1. **契约与不变量先行**：动手前先明确并保持三项契约
   - **张量契约**：接口输入/输出的维度语义（如 `[B, H_kv, S, D]`）、`dtype`、设备位置、内存连续性。
   - **状态不变量**：调度队列、槽位/页表索引、序列长度与 Attention Mask 之间的守恒关系。
   - **瓶颈意识**：改动落在 Compute-Bound（Prefill）还是 Memory-Bandwidth / Overhead-Bound（Decode 与调度准备）。

2. **正交演进与最小爆炸半径**
   - 单次改动保持正交，不得把「架构重构」「新算子/新特性」「评测基准变更」混在一起。
   - **基线可对照性**：新增优化（新 Cache 布局、算子融合、调度策略）必须经模式或策略抽象接入，**不得破坏既有 Baseline 路径的可运行性与可对比性**。
   - **零黑盒依赖**：核心运行时（`model/`、`worker/`、`core/`、`engine/`、`entrypoints/`）只用原生 Python 与 PyTorch，**严禁引入 `transformers`、`tokenizers` 等外部封装库**（`transformers` 仅限 `verification/parity/` 与 `tests/gpu/parity/` 作 Golden Reference）。

3. **YAGNI：拒绝无声明的向后兼容**
   - 除非明确要求向下兼容，**严禁为兼容旧逻辑增加用不到的代码**：废弃 member field、过时 method name、兼容性 property、无用参数默认值、forwarding wrapper。
   - 杜绝死代码与认知噪音。

4. **结果指标变更的端到端联动守恒**
   - **增删改任何结果指标**（延迟、吞吐、显存统计、调度事件字段等）必须同步下游：
     1. **Schema 与 JSON 数据**：Pydantic 模型（`bench/schema.py`）、基准数据文件（`benchmarks/result-history/`、`benchmarks/data/`）。
     2. **绘图脚本**：`src/mini_infra/bench/plotting/`，防止指标增删改名导致 KeyError、字段取空或图表损坏。

5. **增量严格守门，存量债务豁免**
   - **增量 100% 达标**：新建文件、新函数、本次修改的 Diff 区域必须遵守全部架构不变量与工程规则。
   - **存量豁免**：未被本次任务触碰的历史代码即便违规也默认豁免；**无用户明确指令不得发起超范围的大面积重构**。
   - **债务标记契约**：以注释显式声明历史债务，Agent 遇到即跳过告警且不擅自修改
     - 单行或局部代码块：`# infra-legacy: <豁免原因或收敛计划>` 或 `# infra-legacy(<rule>): <reason>`
     - 文件级（置于文件顶部）：`# infra-legacy-file: <豁免原因>`
   - **触碰时顺手收敛（Boy Scout Rule）**：仅当改动直接触碰该文件/函数、或任务明确要求重构时推进收敛，且必须守住基线可对照性。

6. **边界校验，内部信任**
   - 只在 `entrypoints/` 与外部 API 边界做入参校验；内部模块信任自身契约，**严禁为不会发生的场景加 `None` 兜底、`try/except` 吞异常或冗余校验**。

---

## 二、系统架构与不变量

单向依赖栈，对齐现代高性能大模型推理引擎（如 vLLM V1）的前后端解耦、控制流（CPU）与数据流（GPU）分离：

- **分层由依赖决定，不由目录名决定**：模块只要 `import torch` 并持有张量即属数据面，无论物理位于哪个包。
- **只能依赖下游层，严禁反向依赖**；新增模块先声明归属哪一层。

```text
① Entrypoints  (接入层)      协议解析、参数校验、Tokenize / Detokenize、结果聚合
      ↓
② Engine       (协调层)      驱动 Schedule -> Execute -> Process Outputs 主循环，维护生命周期与时钟
      ↓
③ Core         (CPU 控制面)  Request State Machine、Queue Scheduler、Logical Slot/Block Allocator
      ↓
④ Worker       (GPU 数据面)  Input Tensor 组装、Physical KV Cache Pool 读写、Forward 触发与 Device Sampler
      ↓
⑤ Model        (算子计算面)  Pure Tensor-In, Tensor-Out：无状态神经网络，仅消费与产出裸 Tensor
```

**各层职责与硬性不变量**

- **① `entrypoints/`（Entrypoints）**
  - **核心职责**：协议解析与入参校验；Tokenize / Detokenize；跨请求结果聚合与格式化输出。
  - **硬性不变量**：纯 Python 原生对象出入。**严禁执行 Tensor 操作、分配物理显存、或参与调度排队与 Cache 寻址**。

- **② `engine/`（Orchestrator）**
  - **核心职责**：承担 `EngineCore` 角色，驱动 `Schedule -> Execute -> Process Outputs` 主循环；串联 Core（获取调度计划）与 Worker（触发执行）；推进请求状态流转、EOS 判定与打点计时。
  - **硬性不变量**：编排者而非执行者。**严禁分配物理显存、拼装 GPU Tensor 或直接调用模型底层算子**；**严禁被下游层（`core/`、`worker/`、`model/`）反向依赖**。

- **③ `core/`（Control Plane: Scheduler & Logical Allocator）**
  - **核心职责**：纯 CPU 逻辑决策中枢。维护 Request State Machine（`WAITING -> RUNNING -> FINISHED`）；执行排队优先级与抢占调度；管理 KV Cache 逻辑映射（维护 Slot Map / Block Table，负责逻辑 Token 序列到物理显存块的映射）。
  - **硬性不变量**：**100% 纯 Python / CPU 实现，零 `torch` GPU 依赖**。只操作原生整数索引与元数据，**严禁持有 GPU Tensor、分配物理显存、触发 CUDA 同步或调用模型前向**。

- **④ `worker/`（Data Plane: Model Runner & Physical KV Cache Pool）**
  - **核心职责**：GPU 算力与物理显存池专职执行者。在 Host 侧将调度计划批量组装为 GPU 输入 Tensor（`input_ids`、`position_ids`、`slot_mapping` / `block_tables`）；按物理索引（显存池切片/物理块号：`slot_idx`、`physical_block_id`）原地读写 KV Cache；触发 Model Forward 并在设备端完成采样（Sampler）。
  - **硬性不变量**：只认**物理索引（显存切片/物理块号下标，不感知 `request_id` 或业务请求对象）与 Tensor**。**严禁掺杂业务排队策略、请求优先级或生命周期状态机**；物理池**严禁承担输入组装**（输入组装归属 Runner）；Sampler 属本层，不属 `model/`。

- **⑤ `model/`（Compute Plane: Pure Model Executor）**
  - **核心职责**：无状态张量计算图（Pure Tensor-In, Tensor-Out）。实现 Transformer 网络架构、Norm、RoPE、Attention Kernel 与 MLP 等底层数学算子。
  - **硬性不变量**：无状态计算。对上层调度、请求状态与外部 Cache 容器管理**完全无感知**，**严禁 import 或 `isinstance` 任何外部 Cache 容器**，KV 历史仅以裸 Tensor 形式进出；分词等 CPU 文本处理严格隔离在外。

**严格请求隔离**：无论单独运行、与任意长短请求拼 Batch、处于任意物理 Slot/Page、经历动态抢占/扩缩容或不同长度 Padding，其有效 Token 的 `position_ids` 与因果可见范围（Attention Mask / Block Table 寻址）必须与独立运行时**严格等价**，绝不允许跨请求数值污染。

---

## 三、必跑验证门禁

改动完成后、宣告任务结束前，必须通过以下秒级 CPU 闭环（无需 GPU 或下载权重）：

```bash
ruff check . && ruff format --check .
mini-infra verify smoke && pytest -m "not gpu"
```

改动涉及算子、KV Cache、批处理或调度器逻辑且当前环境具备 GPU 时，追加遵循 `.agents/rules/parity-and-benchmarking.md` 执行 `float32` 数值等价性验证（`pytest -m gpu`）与零性能退化门禁（`mini-infra bench check-regression`）。
