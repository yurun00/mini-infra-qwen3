# Lockstep 齐步推理基准演进与性能剖析报告

基准版本：`baseline (no cache)*` (`a497386`)，GPU：NVIDIA L4，数值为 median，括号内为相对基准的变化比（>1 表示更快/更高）。

| 版本 | commit | batch | prompt | max_new_tokens | TTFT (ms) | TPOT (ms) | 吞吐 (tok/s) | 显存 (MiB) | 槽位利用率 | 正确性 |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| baseline (no cache)* | a497386 | 1 | 128 | 64 | 47.04 (1.00x) | 45.86 (1.00x) | 21.8 (1.00x) | 1257 | 100.0% | ✅ |
| v0.1.0 (no cache)* | 915b0d9 | 1 | 128 | 64 | 47.69 (0.99x) | 46.54 (0.99x) | 21.5 (0.99x) | 1257 | 100.0% | ✅ |
| v0.2.0 (KV cache)* | 437d41d | 1 | 128 | 64 | 49.47 (0.95x) | 47.55 (0.96x) | 21.0 (0.96x) | 1197 | 100.0% | ✅ |
| v0.3.0 (fixed batch) | 1c21386 | 1 | 128 | 64 | 56.88 (0.83x) | 55.91 (0.82x) | 17.9 (0.82x) | 1197 | 100.0% | ✅ |

## 一、跨版本演化与扩展趋势 (Evolution & Scaling Across Releases)

![Evolution & Scaling](./evolution_and_scaling.png)

## 二、当前版本深度剖析与优化指引 (Profiling & Next Steps)

![Profiling](./profiling_and_optimization.png)

基于当前版本 `v0.3.0 (fixed batch)`（`prompt=128`）下的基准测量与 profiling 结果，诊断如下：

### 1. 架构模块瓶颈诊断（改哪里？）
- 🔴 **Attention 模块耗时占比最高（63.8%）**：
  - 👉 **优先优化行动**：引入 KV Cache / FlashAttention 算子融合。
- 🟡 **RMSNorm 调用开销显著（耗时占比 16.1%）**：建议实施 Fused RMSNorm。

### 2. 算子类型与访存诊断（怎么改？）
- 🔹 **GEMM 矩阵乘占 65.3%**，**Elementwise + Memory 搬运占 28.7%**。

### 3. 硬件物理天花板差距
- 🔹 **Decode MBU（显存带宽利用率）为 7.2%**。
- 🔹 **GPU 忙碌比例达到 99.1%**：瓶颈主要在 GPU 端计算与访存。

### 4. 显存足迹与碎片诊断
- 🔹 纯动态增量为 **52.2 MiB**，Allocator 碎片与预留为 **46.7 MiB**。

### 5. 当前 Top 8 热点 Kernel 明细
| 序号 | Kernel 名称 | 类别 | CUDA 耗时 (ms) | 耗时占比 | 调用次数 |
| :-: | :--- | :-: | :-: | :-: | :-: |
| 1 | `aten::mm` | gemm | 333.42 | 31.7% | 12608 |
| 2 | `std::enable_if<!(false), void>::type int` | gemm | 214.83 | 20.4% | 9800 |
| 3 | `std::enable_if<!(false), void>::type int` | gemm | 115.08 | 10.9% | 3591 |
| 4 | `aten::mul` | elementwise | 45.23 | 4.3% | 25280 |
| 5 | `aten::copy_` | memory | 44.77 | 4.3% | 22109 |
| 6 | `void at::native::elementwise_kernel<128,` | elementwise | 23.68 | 2.3% | 10809 |
| 7 | `aten::add` | elementwise | 22.00 | 2.1% | 16255 |
| 8 | `void at::native::unrolled_elementwise_ke` | memory | 21.79 | 2.1% | 9088 |


