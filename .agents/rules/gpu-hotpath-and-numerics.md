---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/model/**/*.py,src/mini_infra/worker/**/*.py,src/mini_infra/engine/**/*.py,src/mini_infra/device.py"
description: "张量维度契约、GPU 热路径纪律、显存布局契约、浮点数值稳定性与双轨计时规范"
---

# GPU 热路径、显存管理与数值精度专项规则

修改模型算子（`model/`）、data plane 执行器与 KV Cache 管理（`worker/`）、执行引擎（`engine/`）或设备工具（`device.py`）时，遵守以下硬件与数值纪律。

---

## 一、GPU 热路径与显存纪律

1. **零隐式 Host-Device 同步**：模型 `forward()` 与自回归 Decode 循环热路径中**严禁**出现触发 CPU-GPU 强同步、阻塞流水线的操作——`tensor.item()`、`tensor.cpu()`、`tensor.numpy()`、`print(tensor)`，或基于 GPU 张量标量值的 Python `if / while` 分支。

2. **元数据 Host 侧批量组装后单次上卡**：单步调度所需的 `input_ids`、`position_ids`、physical slot / block table 映射（`slot_mapping` / `block_tables`）与活跃掩码（Active Mask）必须先在 CPU 侧按 control plane 下发的整数索引批量组装完毕，再经单次 H2D 构造为 GPU 张量。**严禁**在单步迭代内按 batch slot 或按 request 循环发射零碎的 GPU 张量切片写入与小 Kernel。

3. **显存生命周期与推理态**：所有面向推理与调度的执行入口必须显式装饰 `@torch.inference_mode()`（优于 `torch.no_grad()`，彻底关闭视图追踪与版本计数器开销）。临时缓冲区拷贝或绑定到持久化池后，及时释放大张量引用（如 `del prefill_cache`），避免虚高运行期峰值已分配显存（Peak Allocated Memory）。

4. **显存视图与原地操作**
   - 热路径中间结果**优先由预分配 Buffer 经 `out=` 参数承接**（如 `torch.matmul(a, b, out=buffer)`），避免每步 Decode 反复申请释放造成显存碎片与 Allocator 开销。
   - **优先 `view()` 而非 `reshape()` 做形状变换**：`view()` 在内存不连续时直接报错，能暴露隐藏的布局假设；`reshape()` 静默退化为拷贝，把问题藏到性能里。
   - **严禁在 KV Cache 读写路径触发隐式 `contiguous()` 显存拷贝**（包括 `transpose` / `permute` 之后直接喂给要求连续内存的算子）。缓存布局必须在设计阶段就与算子的访存模式对齐，不允许靠运行期拷贝兜底。

---

## 二、张量对齐与数值稳定性不变量

1. **张量维度契约强制声明**：所有核心方法（模型算子、Cache 读写、Runner 执行入口）必须在 Docstring 或注释头部用统一格式写明输入/输出的维度语义、`dtype` 与内存连续性：

   ```text
   hidden_states:  [batch, seq_len, hidden_dim],               bf16, contiguous
   key_cache:      [capacity, num_kv_heads, length, head_dim], bf16, contiguous
   attention_mask: [batch, past_len + query_len],              bool
   ```

   目的是**防止 PyTorch 广播掩盖维度错位**：广播会让 `[B, 1, D]` 与 `[B, S, D]` 的静默错配通过运行期检查，产出错误数值而不报错。方法内部按需补充关键中间张量的维度、类型与连续性注释；凡 `reshape` / `view` / `transpose` / 切片等改变内存布局的操作，必须就地注明结果形状与是否 `contiguous`。

2. **序列填充与缓存对齐契约**
   - **Padding 隔离**：批量填充（如 Left Padding）时，有效 Token 的 `position_ids` 必须严格从 `0` 连续递增，Padding 位必须通过 `attention_mask` 彻底屏蔽，严禁参与有效 Attention 权重归一化或污染 RoPE 相对位置距离。
   - **缓存布局守恒**：动态 batch slot 或分页缓存（Dense / Paged KV Cache）在扩容、迁移或 batch slot 复用时，必须同步更新历史掩码（history mask / block table）并清零脏数据，确保 free batch slot 与历史残留不会泄漏到活跃请求。

3. **低精度算子内部升维保护**：`bfloat16` / `float16` 推理下，凡涉及平方求和归约、方差倒数、指数归一化或旋转位置频率计算的算子（如 `RMSNorm`、`Softmax`、`RotaryEmbedding`）**必须在算子内部显式升维至 `torch.float32` 完成核心计算后，再转回目标 `dtype`**，严防低尾数位下的精度坍塌或溢出。

---

## 三、异步流水线与同步计时的边界

1. **吞吐执行态（异步打点）**：齐步自回归（Lockstep）等追求纯吞吐的执行循环在 CUDA 下使用 `PendingStepLatencies`（通过 `with latencies.measure_step():` 记录起止事件 `torch.cuda.Event(enable_timing=True)`），待循环结束后由其 `settle()` 一次性同步并批量结算；**严禁在每步 Decode 内部插入 `torch.cuda.synchronize()`** 破坏异步发射流水线。GPU 未排空前 Event 时间戳尚未写入，提前读 `elapsed_time` 会把 CPU 阻塞至该 Event 完成、打断 kernel 发射（表现为吞吐退化而非报错，属静默故障），故 **`synchronize` 必须内聚在 `settle()` 内，严禁在协调层手写裸 `elapsed_time` 结算或让裸 Event 对跨层流转**。

2. **离散调度仿真与阶段剖析态（显式同步）**：耗时数据用于推进离散事件调度器逻辑时钟（`clock_ms`）或精确剖析子阶段开销时，必须在测量区间前后显式调用 `synchronize(device)`。**正交计时拆解**：纯模型算子前向耗时（Model Forward Time）与系统管理开销（KV Cache 分配/扩容/拷贝、Decode 输入准备耗时）必须正交分离统计，禁止混为一谈。
