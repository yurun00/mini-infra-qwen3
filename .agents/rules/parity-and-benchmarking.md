---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/verification/**/*.py,src/mini_infra/bench/**/*.py,src/mini_infra/profiling/**/*.py,tests/**/*.py,benchmarks/**/*"
description: "确定性数值等价性验证（Parity）、假设驱动性能评测与基准数据防污染规范"
---

# 数值对齐验证 (Parity) 与性能评测基准专项规则

编写测试（`tests/`）、数值等价性校验（`verification/`）、性能基准与剖析（`bench/`、`profiling/`）或操作 `benchmarks/` 目录时，遵守以下验证与评测纪律。

---

## 一、确定性数值等价性验证铁律

1. **强制 `float32` 剥离硬件 GEMM 累加噪声**
   - `bfloat16` 仅有 7 位尾数位：Batch Size、Padding 长度或矩阵分块（Tiling）变化会改变 GPU Tensor Core 的 GEMM 累加顺序，引入 `~1e-3` 的合法浮点舍入噪声，足以在 Greedy Decoding 临界点翻转 Top-1 Token。
   - 验证新算子、KV Cache 机制、批处理隔离或动态调度器的**算法正确性**时（`verify cache`、`verify lockstep-batch`、`verify scheduler-batch`），**必须默认固定在 `dtype=torch.float32` + Greedy Decoding 下对照**，以排除硬件低精度截断噪声。

2. **零妥协通过标准**
   - **自洽等价性（Self-Parity）**：开启优化路径（KV Cache、Fixed/Continuous Batching、新算子）与朴素基线（无 Cache 全量重算、单请求串行独立运行）相比，`float32` 下逐步 Logits 最大绝对误差必须满足 `atol <= 1e-4`，且生成的 Greedy Token 序列必须 **100% 逐字完全一致**。
   - **跨维置换不变性**：批处理与调度测试必须覆盖不同 Batch Size、逆序置换、长短悬殊请求混跑、EOS 提前退场 batch slot 复用以及触发缓存动态扩容等边界工况。

---

## 二、假设驱动的性能评测与统计严谨性

1. **统计有效性**
   - 任何性能测量必须包含预热迭代（Warmup），消除 CUDA Context 初始化、JIT 编译与 PyTorch Caching Allocator 冷启动抖动。
   - 性能报告与回归判定必须基于多轮重复测量（Repeats）的**中位数与分位数（P50 / P95）**，严禁仅凭单次运行或易受极值污染的算术平均值（Mean）下结论。
   - 性能回归门禁（`check_regression`）必须同时结合相对退化阈值（如 `> 5%`）与 `3 * std` 噪声保护带，避免硬件微小波动造成误报。

2. **多维硬件与显存指标正交采集**
   - 评测结果需正交呈现延迟（TTFT / TPOT / Makespan / Queueing Delay）、吞吐（Tokens/s）与硬件效率（MBU 显存带宽利用率 / MFU 算力利用率）。
   - 显存指标必须严格区分：**峰值已分配显存（Peak Allocated）**、**缓存池预留显存（Peak Reserved）**、**KV Cache 物理占用字节**、**有效 Token 真实负载字节（衡量显存碎片率）**。

---

## 三、基准数据卫生与防污染红线

1. **严禁污染受控历史归档与黄金基准**
   - 日常开发、Debug 与调优实验运行基准测试时，输出文件**必须**定向至 `tmp/` 目录或命名为 `*.local.json`（已由 `.gitignore` 忽略）。
   - **绝对禁止**手动编辑、伪造或用非标准实验参数覆盖以下受版本控制的权威文件：
     - `benchmarks/result-history/{lockstep,scheduler}/*.json`
     - `benchmarks/data/reference.json`
     - `benchmarks/data/golden_output_tokens.json`
   - 正式历史归档仅允许在版本发布（Release SOP）或受控基准更新流程中，使用固定随机种子（`seed=42`）的标准工作负载生成。

2. **Schema 强类型契约同步**：评测 JSON 结构由 `src/mini_infra/bench/schema.py` 中的三层 Pydantic v2 模型与判别式联合（`AnyBenchmarkRecord`）严格守护。任何运行时领域统计指标的增删，必须同步更新对应的 Pydantic Record 与单元测试，严禁塞入未声明的散乱裸字典字段。
