# Benchmark 性能评估与演进分析报告

基准版本：`baseline (no cache)*` (`a497386`)，GPU：NVIDIA L4，数值为 median，括号内为相对基准的加速比（>1 表示更快）。

| 版本 | commit | batch | prompt | output | TTFT (ms) | TPOT (ms) | 吞吐 (tok/s) | 显存 (MiB) | 正确性 |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| baseline (no cache)* | a497386 | 1 | 32 | 64 | 37.49 (1.00x) | 36.29 (1.00x) | 27.5 (1.00x) | 1201 | ✅ |
| baseline (no cache)* | a497386 | 1 | 128 | 64 | 47.04 (1.00x) | 45.86 (1.00x) | 21.8 (1.00x) | 1257 | ✅ |
| baseline (no cache)* | a497386 | 1 | 512 | 64 | 49.38 (1.00x) | 46.84 (1.00x) | 21.3 (1.00x) | 1480 | ✅ |
| baseline (no cache)* | a497386 | 1 | 1024 | 64 | 99.68 (1.00x) | 107.52 (1.00x) | 9.3 (1.00x) | 1778 | ✅ |
| v0.1.0 (no cache)* | 915b0d9 | 1 | 32 | 64 | 37.24 (1.01x) | 36.90 (0.98x) | 27.1 (0.98x) | 1201 | ✅ |
| v0.1.0 (no cache)* | 915b0d9 | 1 | 128 | 64 | 47.69 (0.99x) | 46.54 (0.99x) | 21.5 (0.99x) | 1257 | ✅ |
| v0.1.0 (no cache)* | 915b0d9 | 1 | 512 | 64 | 49.86 (0.99x) | 47.63 (0.98x) | 21.0 (0.98x) | 1480 | ✅ |
| v0.1.0 (no cache)* | 915b0d9 | 1 | 1024 | 64 | 101.14 (0.99x) | 107.61 (1.00x) | 9.3 (1.00x) | 1778 | ✅ |
| v0.2.0 (KV cache)* | 437d41d | 1 | 32 | 64 | 39.42 (0.95x) | 38.15 (0.95x) | 26.2 (0.95x) | 1158 | ✅ |
| v0.2.0 (KV cache)* | 437d41d | 1 | 128 | 64 | 49.47 (0.95x) | 47.55 (0.96x) | 21.0 (0.96x) | 1197 | ✅ |
| v0.2.0 (KV cache)* | 437d41d | 1 | 512 | 64 | 51.52 (0.96x) | 47.96 (0.98x) | 20.8 (0.98x) | 1350 | ✅ |
| v0.2.0 (KV cache)* | 437d41d | 1 | 1024 | 64 | 95.37 (1.05x) | 47.79 (2.25x) | 20.6 (2.21x) | 1556 | ✅ |

## 一、跨版本演化与扩展趋势 (Evolution & Scaling Across Releases)

![Evolution & Scaling](evolution_and_scaling.png)

## 二、当前版本深度剖析与优化指引 (Profiling & Next Steps)

![Profiling](profiling_and_optimization.png)

基于当前版本 `v0.2.0 (KV cache)*` 在长上下文（`prompt=1024`）下的基准测量与 profiling 结果，诊断如下：

### 1. 架构模块瓶颈诊断（改哪里？）
- 🔴 **Attention 模块耗时占比最高（60.1%）**：
  - **现状**：自回归 Decode 阶段重复计算全量上下文历史，计算与访存开销随序列长度急剧膨胀。
  - 👉 **优先优化行动**：**引入 KV Cache** 将每步全量重算解耦，预期可带来 5x~10x 以上吞吐跃升；后续进一步接入 **FlashAttention / SDPA 算子融合**。
- 🟡 **RMSNorm 频繁调用开销显著（耗时占比 15.3%）**：
  - **现状**：每个 Transformer 层多次独立的 Norm 小算子带来了大量的 Kernel 启动与显存往返开销。
  - 👉 **后续优化行动**：实施 **Fused RMSNorm**（将残差加 Add 与 RMSNorm 融合成单个 Kernel）。

### 2. 算子类型与访存诊断（怎么改？）
- 🔹 **GEMM 矩阵乘占 53.8%**，**Elementwise + Memory 搬运占 35.0%**。
  - 访存搬运和小算子占用了近三分之一的 GPU 时间，说明存在较多碎片化临时张量读写。
  - 👉 **技术路线**：结合 Triton 或 PyTorch Inductor 进行横向与纵向算子融合，减少 SRAM 与 HBM 之间的中间写回。

### 3. 硬件物理天花板差距
- 🔹 **Decode MBU（显存带宽利用率）仅为 9.2%**：
  - 当前 Decode 受限于重复计算而非饱和访存，距离理想的 Memory-bound 饱和线（60%~80%）仍有巨大空间。
- 🔹 **GPU 忙碌比例达到 100.0%**：当前几乎无 CPU 空泡，瓶颈纯在 GPU 端计算与访存。

### 4. 显存足迹与碎片诊断
- 🔹 在长序列下，纯动态增量（激活与状态）为 **410.8 MiB**，PyTorch Allocator 碎片与预留达到 **300.2 MiB**。
  - 👉 **优化行动**：进入长文本及多 Batching 场景后，应引入 **PagedAttention（分页显存管理）** 避免非连续预分配带来的显存碎片。

### 5. 当前 Top 8 热点 Kernel 明细
| 序号 | Kernel 名称 | 类别 | CUDA 耗时 (ms) | 耗时占比 | 调用次数 |
| :-: | :--- | :-: | :-: | :-: | :-: |
| 1 | `aten::mm` | gemm | 346.98 | 24.5% | 12608 |
| 2 | `std::enable_if<!(false), void>::type int` | gemm | 217.92 | 15.4% | 9492 |
| 3 | `std::enable_if<!(false), void>::type int` | gemm | 115.98 | 8.2% | 3591 |
| 4 | `aten::copy_` | memory | 99.58 | 7.0% | 21852 |
| 5 | `aten::cat` | other | 66.75 | 4.7% | 7240 |
| 6 | `aten::mul` | elementwise | 56.12 | 4.0% | 25280 |
| 7 | `void at::native::(anonymous namespace)::` | memory | 55.27 | 3.9% | 3472 |
| 8 | `void at::native::elementwise_kernel<128,` | memory | 42.06 | 3.0% | 3612 |


