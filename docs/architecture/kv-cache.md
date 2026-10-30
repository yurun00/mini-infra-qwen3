# KV Cache 架构与显式状态管理 (KV Cache & Memory Architecture)

在大语言模型（LLM）的自回归生成（Autoregressive Generation）过程中，**KV Cache（键值缓存）** 是推理系统最核心的基础设施之一。本项目遵循显式、透明、零侵入的设计原则，手写实现了原生的 `KVCache` 容器与 RoPE 偏移机制，消除全量重复计算的同时保证数值精度 100% 严格对齐。

---

## 一、为什么需要 KV Cache？

### 1. 自回归生成的计算冗余
标准的因果语言模型（Causal LM）遵循逐 Token 生成的范式：
$$P(x_t \mid x_{<t}) = \text{Softmax}\left(\frac{Q_t K_{\le t}^T}{\sqrt{d_k}} + M_{\le t}\right) V_{\le t}$$

如果每一步解码（Decode）都将当前全部历史序列输入模型重新计算（即 **全量重算 / 无 Cache 模式**）：
- 在生成第 $N$ 个 Token 时，前 $N-1$ 个历史 Token 的 Query、Key、Value 都会被重新投影并前向计算一次。
- **时间复杂度危机**：
  - 单步计算复杂度由序列长度线性增加：第 $t$ 步的计算量为 $O(t)$。
  - 生成 $N$ 个 Token 的累计计算复杂度达到 **$O(N^2)$**（若考虑 Attention 矩阵则是 **$O(N^3)$**）。
  - 模型执行延迟随生成长度呈抛物线恶化。

### 2. KV Cache 的核心洞察
由于因果注意力掩码（Causal Mask）的单向性：
- 历史 Token 的表示不受后续未来 Token 的影响；
- 每一层 Attention 算子中，历史 Token 计算出的 **Key** 和 **Value** 向量是**完全不可变（Immutable）**的。
- 只有当前的最新 Token 会作为 **Query** 探寻所有历史键值对。

因此，**将历史各层的 Key 和 Value 向量缓存在 GPU 显存中，解码时仅将最新的单个 Token 输入模型计算一次单行 Query，并将其生成的单个 Key/Value 拼入缓存**，即可将单步解码的计算量恒定降为 $O(1)$（算子前向）与 $O(t)$（Attention 向量内积），端到端生成复杂度降为 **$O(N^2)$**。

```mermaid
flowchart LR
    subgraph 无 Cache (全量重算)
        A1["[x1, x2] -> 全量前向 -> 生成 x3"]
        A2["[x1, x2, x3] -> 全量重新前向 -> 生成 x4"]
        A3["[x1, x2, x3, x4] -> 全量重新前向 -> 生成 x5"]
        A1 --> A2 --> A3
    end
    
    subgraph KV Cache (增量解码)
        B1["Prefill: [x1, x2] -> 缓存 KV_1,2 -> 生成 x3"]
        B2["Decode: [x3] + 读取 KV_1,2 -> 追加 KV_3 -> 生成 x4"]
        B3["Decode: [x4] + 读取 KV_1,2,3 -> 追加 KV_4 -> 生成 x5"]
        B1 --> B2 --> B3
    end
```

---

## 二、架构设计与显式容器实现

本项目杜绝框架层隐藏的全局隐式全局变量，在 [`src/mini_infra/memory/kv_cache.py`](../../src/mini_infra/memory/kv_cache.py) 中构建了纯 Python/PyTorch 的显式数据容器：

### 1. 容器数据结构与张量维度
`KVCache` 类维护每层解耦的张量列表：
* `key_cache: list[torch.Tensor]`
* `value_cache: list[torch.Tensor]`

对于拥有 $L$ 个 Decoder 层的模型，容器容量为 $L$。在每一层中，Key 和 Value 张量的布局严格遵循：
$$\text{Shape: } [B, H_{\text{kv}}, S, D]$$
* $B$ (`batch_size`)：批大小。
* $H_{\text{kv}}$ (`num_key_value_heads`)：GQA（Grouped Query Attention）键值头数（如 Qwen3-0.6B 中为 2）。
* $S$ (`seq_len`)：当前已缓存的有效序列长度。
* $D$ (`head_dim`)：注意力头维度（如 Qwen3 中为 128）。

### 2. Prefill 与 Decode 的双模更新
在 [`KVCache.update()`](../../src/mini_infra/memory/kv_cache.py#L20) 中显式区分两阶段：
```python
def update(
    self, key_states: torch.Tensor, value_states: torch.Tensor, layer_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(self.key_cache) <= layer_index:
        # 1. Prefill 阶段：首次前向，将整段 Prompt 的全量 Key/Value 保存入池
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
    else:
        # 2. Decode 阶段：沿序列长度维度 (dim=2) 原地拼接当前步的单 Token Key/Value
        self.key_cache[layer_index] = torch.cat([self.key_cache[layer_index], key_states], dim=2)
        self.value_cache[layer_index] = torch.cat([self.value_cache[layer_index], value_states], dim=2)
    return self.key_cache[layer_index], self.value_cache[layer_index]
```

### 3. RoPE 旋转位置编码的位置偏置处理
在增量 Decode 步中，输入的 `input_ids` 维度仅为 `[B, 1]`。若直接应用默认的绝对位置序列 `arange(0, 1)`，会导致位置编码错误。

在 [`src/mini_infra/model/attention.py`](../../src/mini_infra/model/attention.py) 中，通过传入的 `position_ids` 或通过 `cache.get_seq_length()` 计算全局偏移：
$$\text{position\_id} = \text{cached\_length}$$
确保单步 Query/Key 能够正确旋转到序列中的绝对位置坐标。

---

## 三、显存理论建模与带宽分析

### 1. KV Cache 显存占用精确公式
单次请求的理论 KV Cache 显存容量由模型超参数与序列长度完全决定：
$$\text{KV Cache Bytes} = 2 \times L \times B \times H_{\text{kv}} \times D \times S \times \text{sizeof}(\text{dtype})$$

以 **Qwen3-0.6B**（FP16 / BF16，`sizeof=2` 字节）为例：
* $L = 28$
* $H_{\text{kv}} = 2$
* $D = 128$
* 单 Token 单批次占用的显存增量为：
  $$2 \times 28 \times 1 \times 2 \times 128 \times 1 \times 2 = 28,672 \text{ Bytes} \approx 28.0 \text{ KB / token}$$
* 若序列长度达到 4096，单个并发请求就需要消耗约 **112 MB** 纯 KV Cache 显存。

在 [`src/mini_infra/memory/kv_cache.py`](../../src/mini_infra/memory/kv_cache.py#L49) 中，通过 `total_bytes()` 实现纳秒级显存审计：
```python
def total_bytes(self) -> int:
    return sum(
        k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in zip(self.key_cache, self.value_cache)
    )
```

### 2. 计算强度与 Memory-Bound 特征
* **Prefill 阶段**：Prompt 整段前向，GEMM 矩阵乘法计算强度（FLOPs/Byte）高，属于 **Compute-Bound（计算受限）**。
* **Decode 阶段**：每一步仅输入 1 个 Token，但每层都需要从 HBM（显存）中完整读取此前积累的整个 KV Cache 张量到 SRAM。计算量极小（$O(D)$），访存量极大（$O(S \cdot D)$），属于典型的 **Memory-Bound（内存带宽受限）**。
* 系统优化核心聚焦于提升 **MBU（Model Bandwidth Utilization）**，详见 [算子性能剖析指南](../guides/profiling.md)。

---

## 四、等价性验证与正确性门禁

引入任何显存缓存机制，绝不能破坏模型输出数值等价性。

本项目在 [`src/mini_infra/verification/parity/compare_cache.py`](../../src/mini_infra/verification/parity/compare_cache.py) 中构建了严格的逐步比对门禁：
* 验证目标：
  1. 生成的 Token 序列在 `use_cache=False` 与 `use_cache=True` 下 **100% 逐 Token 严格相等**；
  2. 解码过程每一步的输出 Logits 最大绝对误差满足：
     $$\max | \text{logits}_{\text{cache}} - \text{logits}_{\text{baseline}} | < 10^{-5}$$
* 执行检验命令：
  ```bash
  mini-infra verify cache
  # 或通过 pytest：
  pytest tests/gpu/parity/test_cache_parity.py
  ```

---

## 五、局限性与向后演进

尽管基础 `KVCache` 彻底解决了自回归重复计算的渐进复杂度，但在多并发服务化场景下，简单的 `torch.cat` 动态拼接暴露出关键瓶颈：
1. **显存碎片化（Memory Fragmentation）**：随着序列自回归增长，频繁执行动态扩容与张量重分配会触发 PyTorch 缓存分配器的反复垃圾回收；
2. **多请求动态并发受限**：每个序列长度不同，难以直接进行批处理内存连续管理。

为了解决多并发问题，项目在 Chapter 03/04 演进出：
- **静态槽位管理**：[固定批处理与左填充机制 (fixed-batching.md)](fixed-batching.md)；
- **迭代级槽位复用**：[连续批处理架构与调度原理 (continuous-batching.md)](continuous-batching.md)；
- **非连续块显存虚拟化**：后续章节的 PagedAttention 机制。

