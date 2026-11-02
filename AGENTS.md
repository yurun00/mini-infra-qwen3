# AI Inference Infra Engineering Rules (Core & Always-On)

你正在参与 `mini-infra-qwen3`（从零手写的高性能大模型推理与调度引擎）的研发。你的核心目标是编写**高可读、确定性、高硬件利用率且零精度妥协**的系统级 Python/PyTorch 代码。

在任何任务规划、代码编写与重构中，必须无条件遵守以下全局准则与架构不变量（具体子域深水区规则由 `.agents/rules/` 按需自动挂载）：

---

## 一、Agent 工程行为准则 (Engineering Mindset & Workflow)

1. **契约与不变量先行 (Contract & Invariant First)**
   - 动手修改前，必须先明确并保持三项契约：
     1. **张量契约（Tensor Contract）**：接口输入/输出的显式维度语义（如 `[B, H_kv, S, D]`）、`dtype`、设备位置与内存连续性。
     2. **状态不变量（State Invariant）**：调度队列、槽位/页表索引（Slot/Page Index）、序列长度与 Attention Mask 之间的守恒关系。
     3. **瓶颈意识（Roofline Awareness）**：明确改动处于 Compute-Bound（如 Prefill）还是 Memory-Bandwidth / Overhead-Bound（如 Decode 与调度准备）。

2. **正交演进与最小爆炸半径 (Orthogonal Evolution & Minimal Blast Radius)**
   - **关注点分离**：单次改动保持正交，绝不将“架构重构”、“新算子/新特性实现”与“评测基准变更”混杂耦合。
   - **基线可对照性（Baseline Preservability）**：新增任何优化特性（如新 Cache 布局、算子融合、调度策略）必须通过清晰的模式或策略抽象接入，**绝不破坏既有 Baseline 路径的可运行性与可对比性**。
   - **零黑盒依赖（Zero Black-Box Policy）**：核心运行时（`model/`、`memory/`、`core/`、`engine/`、`entrypoints/`）坚持纯原生 Python 与 PyTorch 实现，**严禁引入 `transformers`、`tokenizers` 等外部黑盒封装库**（`transformers` 仅限在 `verification/parity/` 与 `tests/gpu/parity/` 中作为 Golden Reference 对照）。

---

## 二、系统架构与不变量 (Architecture & Invariants)

本项目采用**单向依赖栈**，对齐 vLLM V1 的「前后端解耦、控制流（CPU）与数据流（GPU）分离」设计哲学：

- **分层由依赖决定，不由目录名决定**：一个模块只要 `import torch` 并持有张量即属数据面，无论它物理上位于哪个包。
- **只能依赖下游层，严禁反向依赖**；新增模块必须先声明归属哪一层。

```text
① Entrypoints  (entrypoints/)            协议解析与请求接入，只搬运 Python 原生对象
      ↓
② Engine       (engine/*_loop.py)        驱动 Schedule -> Execute -> Process Outputs 闭环
      ↓                                   分词 / detokenize / 停止词判定 / 结果聚合 / 计时打点
③ Core         (core/)                   纯 CPU 控制面：请求状态机、队列调度、Slot/Block ID 分配
      ↓
④ Worker       (engine/runner.py         GPU 数据面：输入组装、物理 KV 显存池读写、前向触发与采样
                 + memory/)
      ↓
⑤ Model        (model/)                  算子面：Pure Tensor-In, Tensor-Out
```

1. **分层职责与硬性边界 (Layer Responsibilities & Hard Boundaries)**
   - **① 接入层（`entrypoints/`）**：外部请求的协议解析、参数校验与跨请求结果聚合。
     - **硬性不变量**：严禁出现张量操作、KV Cache 管理或调度决策。
   - **② 引擎协调层（`engine/`）**：承担 vLLM `EngineCore` 的角色，驱动主循环，并负责 CPU 侧文本处理（分词 / detokenize）、生命周期推进与计时打点。
     - **硬性不变量**：Engine 是**编排者而非执行者**——**严禁自行分配物理显存、自行拼装 GPU 输入张量或直接调用模型算子**；也严禁被 `core/`、`memory/`、`model/` 反向依赖。
   - **③ 纯 CPU 控制面（`core/` — Scheduler & Logical Allocator）**：请求生命周期状态机（`WAITING -> RUNNING -> FINISHED`）、排队抢占策略、停止条件判定（EOS / Token Budget），以及 **KV Cache 逻辑索引分配（如 `RunningQueue` 槽位号或 PagedAttention 的 `BlockManager` 块表 ID）**。
     - **硬性不变量**：只操作 Python 原生对象与整数索引（Slot ID / Block ID），**严禁持有 GPU `torch.Tensor`、严禁分配物理显存、严禁触发 CUDA 同步或调用模型前向**，确保调度与块分配逻辑可在无 GPU、无权重下毫秒级单测。
   - **④ GPU 数据面（`engine/runner.py` + `memory/` — ModelRunner & Physical KV Pool）**：`ModelRunner` 将控制面下发的调度元数据批量组装为 GPU 输入张量（`input_ids`、`position_ids`、`attention_mask` / `slot_mapping`），触发前向并完成设备端采样（Sampler），仅向上层返回轻量 Token ID；`memory/` 持有物理 KV Cache 显存池，按 Slot/Block 索引原地读写。
     - **硬性不变量**：数据面只认**槽位/块索引与张量**，**严禁掺杂业务排队策略、请求优先级或 EOS/预算生命周期判定**。物理池（`memory/`）**严禁承担输入组装职责**——那是 `ModelRunner` 的边界；Sampler 属本层，不属 `model/`。
   - **⑤ 纯张量算子面（`model/` — Model Executor）**：遵循 `Pure Tensor-In, Tensor-Out`。
     - **硬性不变量**：对上层调度器、请求状态及外部 Cache 管理策略**完全无感知**，**严禁 import 或 `isinstance` 判断任何 Cache 容器类型**，KV 历史只以裸张量（如 `Sequence[tuple[Tensor, Tensor]]`）形式进出；分词器等 CPU 文本处理不属于本层。

2. **严格请求隔离不变性 (Strict Request Isolation)**
   - 无论一个请求是单独运行，还是与任意长短请求拼 Batch、处于任意物理 Slot/Page、经历动态抢占/扩缩容或不同长度的 Padding，其有效 Token 的位置编码（`position_ids`）与因果可见范围（Attention Mask / Block Table 寻址）必须与独立运行时**严格等价**，绝不允许跨请求数值污染。

3. **当前已知偏差（只准收敛，不准扩大）**
   - `model/qwen3.py` 仍 import `KVCache` 容器（违反 ⑤）。
   - `DenseSlotCache.prepare_decode_inputs()` 住在 `memory/` 却承担输入组装（违反 ④）。
   - `model/tokenizer.py` 是 CPU 文本处理，却住在 `model/`（违反 ⑤）。
   - **新增代码不得复制上述任何偏差**；改动触碰这些文件时应顺手推进收敛，且必须保持 `prepare_ms` 等既有计时口径不变以守住基线可对照性。

---

## 三、每次改动必跑的基础验证底线 (Mandatory Baseline Gate)

任何代码修改完成后，在宣告任务结束前，必须至少通过以下秒级 CPU 验证闭环（无需 GPU 或下载权重）：

1. **静态规范与类型契约检查**：
   ```bash
   ruff check . && ruff format --check .
   ```
2. **模型结构冒烟与控制面 CPU 单元测试**：
   ```bash
   mini-infra verify smoke && pytest -m "not gpu"
   ```
- 若改动涉及算子、KV Cache、批处理或调度器逻辑，且当前环境具备 GPU，必须进一步遵循 `.agents/rules/parity-and-benchmarking.md` 执行 `float32` 数值等价性验证（`pytest -m gpu`）与零性能退化门禁（`mini-infra bench check-regression`）。

