# 固定批处理与左填充机制 (Fixed Batching & Left-Padding Architecture)

在推理系统从单请求单序列向多并发演进的过程中，**固定批处理（Fixed Batching / Lockstep Batching）** 是最基础且至关重要的批处理形态。本项目在 [`src/mini_infra/engine/preprocessing.py`](../../src/mini_infra/engine/preprocessing.py) 与 [`src/mini_infra/engine/lockstep_loop.py`](../../src/mini_infra/engine/lockstep_loop.py) 中，完整实现了基于**左填充（Left Padding）** 的因果解码对齐、动态完成槽位管理与严格的请求隔离机制。

---

## 一、为什么需要固定批处理？

### 1. 单请求推理解码的算力浪费
现代 GPU（如 NVIDIA L4/A100/H100）具备数万个 CUDA Core 与庞大的 Tensor Core 矩阵乘单元。
* 在单请求（Batch Size = 1）解码阶段，GEMV（矩阵-向量乘法）的计算强度（Arithmetic Intensity）极低，GPU 大部分执行单元处于空闲状态，吞吐（Tokens/s）受限于显存读取带宽（Memory-Bound）。
* **固定批处理**：将多个独立请求打包为一个二维张量 `[B, S]`，使单步解码由 GEMV 提升为 GEMM（矩阵-矩阵乘法）。单步前向时间仅增加少量常数开销，而单位时间内生成的 Token 总量（吞吐）成倍提升。

### 2. 齐步执行模型（Lockstep）
固定批处理遵循齐步（Lockstep）语义：
* **同时进场**：批次内的 $B$ 个请求在 $t=0$ 同时输入模型；
* **单步 Prefill**：整批 Prompt 一次性完成注意力预填充并初始化 KV Cache；
* **齐步 Decode**：每一步前向传播同时驱动所有未结束的请求向前解码 1 个 Token，直到批次内**所有请求均达到预算或命中 EOS**，才整体释放显存。

---

## 二、核心机制：左填充（Left Padding）

在自回归解码器（Decoder-only）架构中，批处理面临的首要挑战是：**不同请求的输入 Prompt 长度参差不齐**。

```mermaid
flowchart TD
    subgraph 错误示范: 右填充 (Right Padding)
        R1["Req 1: [tok1, tok2, tok3, tok4] -> 下一位置为第 4 列"]
        R2["Req 2: [tok1, tok2, pad , pad ] -> 下一位置为第 2 列 (错位!)"]
    end

    subgraph 正确方案: 左填充 (Left Padding)
        L1["Req 1: [tok1, tok2, tok3, tok4] -> 生成边界对齐在最右侧"]
        L2["Req 2: [pad , pad , tok1, tok2] -> 生成边界对齐在最右侧 (完美对齐!)"]
    end
```

### 1. 为什么自回归模型必须使用左填充？
* **右填充（Right Padding）的失效**：
  在编码器（如 BERT）或一次性分类任务中右填充很常见。但在自回归生成中，每个请求的新 Token 都在自身序列的最末端产生。如果右填充，每个请求当前最新有效 Token 的物理列索引各不相同（请求 1 在第 4 列，请求 2 在第 2 列），后续增量 Decode 步无法以统一的单列张量 `[B, 1]` 与 KV Cache 进行规整拼接。
* **左填充（Left Padding）的天然优势**：
  在左侧填充占位符后，**所有请求的最后一个有效 Token 天然对齐在张量的最右边界（列索引为 $L_{\max}-1$）**。后续自回归生成时，模型仅需在最右侧统一追加新 Token，所有请求在物理张量与自注意力缓存中的生成边界完全对齐。

### 2. `left_pad_sequences` 的张量构造实现
在 [`src/mini_infra/engine/preprocessing.py`](../../src/mini_infra/engine/preprocessing.py#L9) 中，提供了严谨的张量构造逻辑：

```python
def left_pad_sequences(token_sequences, pad_token_id, device):
    # 1. 计算最大序列长度并初始化张量
    max_length = max(lengths)
    batch_size = len(token_sequences)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)

    # 2. 靠右放置真实 Token 序列，左侧保留 pad_token_id
    for row, sequence in enumerate(token_sequences):
        length = sequence.numel()
        input_ids[row, max_length - length :] = sequence
        attention_mask[row, max_length - length :] = True

    # 3. 构造相对位置编码 Position IDs
    # 真实 Token 逻辑位置严格从 0 开始递增；左侧 Padding 位置 clamp 为 0
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.clamp_(min=0)
    return input_ids, attention_mask, position_ids, torch.tensor(lengths)
```

#### 逐值示例
输入两个 Prompt：`[3, 5, 7, 9]`（长度 4）与 `[11, 13]`（长度 2），左填充输出如下：
* **`input_ids`**：
  $$\begin{bmatrix} 3 & 5 & 7 & 9 \\ 0 & 0 & 11 & 13 \end{bmatrix}$$
* **`attention_mask`**：
  $$\begin{bmatrix} 1 & 1 & 1 & 1 \\ 0 & 0 & 1 & 1 \end{bmatrix}$$
* **`position_ids`**：
  $$\begin{bmatrix} 0 & 1 & 2 & 3 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
通过将左侧 Padding 处的 `attention_mask` 置为 `0`（False），并在自注意力计算中加入 $-\infty$ 偏置，确保模型注意力完全屏蔽填充位。

---

## 三、请求隔离（Request Isolation）与排列不变性

批处理最严苛的质量红线是**请求隔离**：批次内的任何请求，其输出绝对不能受到同批次其他请求的存在、Prompt 内容、长短或排列顺序的影响。

### 1. 因果掩码隔离保证
批处理下的因果掩码为三维张量 `[B, 1, S, S]`。对于批中的第 $i$ 个请求：
$$M_{i, j, k} = \begin{cases} 0, & \text{if } j \ge k \text{ 且 } \text{mask}_{i, k} = \text{True} \\ -\infty, & \text{otherwise} \end{cases}$$
请求 $A$ 的 Key/Value 只能被请求 $A$ 自身的 Query 看到，彻底杜绝批内跨请求的信息泄露。

### 2. 排列不变性（Permutation Invariance）验证
在 [`src/mini_infra/verification/parity/compare_lockstep_batch.py`](../../src/mini_infra/verification/parity/compare_lockstep_batch.py) 中，系统针对请求隔离进行了完备验证：
* 将单请求独立执行结果作为 **Golden Truth**；
* 将请求 $A$ 分别放置在 Batch 的第 0 行、第 1 行；
* 将请求 $A$ 分别与极短 Prompt、极长 Prompt 混批运行；
* **判定标准**：无论 Batch Size 为何、顺序为何、Padding 长度为何，请求 $A$ 的生成结果与单请求结果 **100% 逐 Token 严格一致**，且单步 Logits 最大绝对误差小于 $10^{-5}$。

---

## 四、齐步执行状态机与度量指标

在 [`src/mini_infra/engine/lockstep_loop.py`](../../src/mini_infra/engine/lockstep_loop.py) 中，`LockstepState` 向量化状态机驱动自回归循环：

### 1. 动态退出与槽位空转
每个请求拥有独立的输出预算（`budgets: torch.Tensor`）：
* 某请求生成结束时，`curr_active[req_idx]` 被置为 `False`；
* 记录该请求的实际完成步数（`SequenceOutput.completion_step`）；
* 已完成的请求在后续 Decode 步中将以 Dummy Token 占位，其槽位进入**空转（Idle）状态**，直到批内所有请求全部完成。

### 2. 核心效率指标
* **填充效率（Padding Efficiency）**：
  衡量 Prefill 阶段有效 Prompt Token 占总输入矩形面积的比例：
  $$\text{Padding Efficiency} = \frac{\sum_{i=1}^B \text{prompt\_len}_i}{B \times \max_{i=1}^B(\text{prompt\_len}_i)}$$
* **执行槽位利用率（Slot Utilization）**：
  衡量 Decode 阶段有效生成的 Token 数占总执行槽位行数的比例：
  $$\text{Slot Utilization} = \frac{\sum_{i=1}^B \text{output\_tokens}_i}{\sum_{t=1}^{T} \text{executed\_batch\_size}_t} = \frac{\sum_{i=1}^B \text{output\_tokens}_i}{B \times \max_{i=1}^B(\text{output\_tokens}_i)}$$

---

## 五、固定批处理的局限性与向 Continuous Batching 演进

固定批处理相比单请求串行大幅提升了 GPU 算力利用率，但在在线服务化（Serving）场景下暴露出难以克服的结构性瓶颈：

1. **计算空泡（Bubble）与木桶效应**：
   * 若批次中请求 1 输出 2 个 Token，请求 2 输出 64 个 Token，批次必须持续执行 64 步。前者的槽位在后 62 步完全处于闲置浪费状态（槽位利用率甚至不足 50%）。
2. **队头阻塞与排队恶化**：
   * 批次锁定了全部显存与槽位，外部新到达的请求无法在当前批次执行途中加入，只能等待前一波整批结束，造成极大的排队与首字延迟（TTFT）。

为了打破固定批次的同步约束，系统进一步演进出将调度细化到每一步步级粒度的解决方案：详见 [连续批处理架构与调度原理 (continuous-batching.md)](continuous-batching.md)。

