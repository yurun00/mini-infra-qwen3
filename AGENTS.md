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
   - **零黑盒依赖**：核心运行时（`model/`、`memory/`、`core/`、`engine/`、`entrypoints/`）只用原生 Python 与 PyTorch，**严禁引入 `transformers`、`tokenizers` 等外部封装库**（`transformers` 仅限 `verification/parity/` 与 `tests/gpu/parity/` 作 Golden Reference）。

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

---

## 二、系统架构与不变量

单向依赖栈，对齐 vLLM V1 的前后端解耦、控制流（CPU）与数据流（GPU）分离：

- **分层由依赖决定，不由目录名决定**：模块只要 `import torch` 并持有张量即属数据面，无论物理位于哪个包。
- **只能依赖下游层，严禁反向依赖**；新增模块先声明归属哪一层。

```text
① Entrypoints  (entrypoints/)     协议解析与请求接入，只搬运 Python 原生对象
      ↓
② Engine       (engine/*_loop.py) 驱动 Schedule -> Execute -> Process Outputs 闭环
      ↓                            分词 / detokenize / 停止词判定 / 结果聚合 / 计时打点
③ Core         (core/)             纯 CPU 控制面：请求状态机、队列调度、Slot/Block ID 分配
      ↓
④ Worker       (engine/runner.py   GPU 数据面：输入组装、物理 KV 显存池读写、前向触发与采样
                 + memory/)
      ↓
⑤ Model        (model/)            算子面：Pure Tensor-In, Tensor-Out
```

**各层职责与硬性不变量**

- **① `entrypoints/`（接入层）**：协议解析、参数校验、跨请求结果聚合。**严禁张量操作、KV Cache 管理、调度决策**。
- **② `engine/`（引擎协调层）**：承担 vLLM `EngineCore` 角色，驱动主循环与 CPU 侧文本处理（分词 / detokenize）、生命周期推进、计时打点。**编排者而非执行者**——**严禁自行分配物理显存、拼装 GPU 输入张量或直接调用模型算子**；严禁被 `core/`、`memory/`、`model/` 反向依赖。
- **③ `core/`（纯 CPU 控制面：Scheduler & Logical Allocator）**：请求生命周期状态机（`WAITING -> RUNNING -> FINISHED`）、排队抢占策略、停止条件判定（EOS / Token Budget）、KV Cache 逻辑索引分配（`RunningQueue` 槽位号 / PagedAttention `BlockManager` 块表 ID）。只操作 Python 原生对象与整数索引（Slot ID / Block ID），**严禁持有 GPU `torch.Tensor`、分配物理显存、触发 CUDA 同步或调用模型前向**，确保无 GPU、无权重下可毫秒级单测。
- **④ `engine/runner.py` + `memory/`（GPU 数据面：ModelRunner & Physical KV Pool）**：`ModelRunner` 将控制面下发的调度元数据批量组装为 GPU 输入张量（`input_ids`、`position_ids`、`attention_mask` / `slot_mapping`），触发前向并完成设备端采样，只向上返回轻量 Token ID；`memory/` 持有物理 KV Cache 显存池，按 Slot/Block 索引原地读写。数据面只认**槽位/块索引与张量**，**严禁掺杂业务排队策略、请求优先级或 EOS/预算生命周期判定**；物理池**严禁承担输入组装**（那是 `ModelRunner` 的边界）；Sampler 属本层，不属 `model/`。
- **⑤ `model/`（纯张量算子面：Model Executor）**：`Pure Tensor-In, Tensor-Out`。对上层调度器、请求状态、外部 Cache 管理策略**完全无感知**，**严禁 import 或 `isinstance` 判断任何 Cache 容器类型**，KV 历史只以裸张量（如 `Sequence[tuple[Tensor, Tensor]]`）形式进出；分词器等 CPU 文本处理不属本层。

**严格请求隔离**：无论单独运行、与任意长短请求拼 Batch、处于任意物理 Slot/Page、经历动态抢占/扩缩容或不同长度 Padding，其有效 Token 的 `position_ids` 与因果可见范围（Attention Mask / Block Table 寻址）必须与独立运行时**严格等价**，绝不允许跨请求数值污染。

---

## 三、必跑验证门禁

改动完成后、宣告任务结束前，必须通过以下秒级 CPU 闭环（无需 GPU 或下载权重）：

```bash
ruff check . && ruff format --check .
mini-infra verify smoke && pytest -m "not gpu"
```

改动涉及算子、KV Cache、批处理或调度器逻辑且当前环境具备 GPU 时，追加遵循 `.agents/rules/parity-and-benchmarking.md` 执行 `float32` 数值等价性验证（`pytest -m gpu`）与零性能退化门禁（`mini-infra bench check-regression`）。
