---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/verification/**/*.py,src/mini_infra/bench/**/*.py,src/mini_infra/profiling/**/*.py,src/mini_infra/cli/commands/verify/**/*.py,src/mini_infra/cli/commands/bench/**/*.py,tests/**/*.py,benchmarks/**/*"
description: "确定性数值等价性验证（Parity）、假设驱动性能评测与基准数据防污染规范"
---

# 数值对齐验证 (Parity) 与性能评测基准专项规则

编写测试（`tests/`）、数值等价性校验（`verification/`、`cli/commands/verify/`）、性能基准与剖析（`bench/`、`profiling/`、`cli/commands/bench/`）或操作 `benchmarks/` 目录时，遵守以下验证与评测纪律。

---

## 一、确定性数值等价性验证铁律

1. **强制 `float32` 剥离硬件 GEMM 累加噪声**
   - `bfloat16` 仅 7 位尾数：Batch Size、Padding 长度或矩阵分块（Tiling）变化会改变 Tensor Core GEMM 累加顺序，引入 `~1e-3` 浮点舍入噪声并可能在 Greedy Decoding 临界点翻转 Top-1 Token。
   - 验证算子、KV Cache、批处理隔离或动态调度器的**算法正确性**时（`verify {cache,lockstep-batch,scheduler-batch,paged-cache,paged-cache-scheduler}`），**必须默认固定在 `dtype=torch.float32` + Greedy Decoding 下对照**。
2. **零妥协通过标准**
   - **自洽等价性（Self-Parity）**：优化路径（Dense / Paged KV Cache、Fixed / Continuous Batching、新算子）对照朴素基线（无 Cache 全量重算、单请求串行独立运行），`float32` 下逐步 Logits 最大绝对误差须满足 `atol <= 1e-4`，且 Greedy Token 序列 **100% 逐字一致**。
   - **跨维置换不变性**：批处理与调度测试必须覆盖不同 Batch Size、逆序置换、长短悬殊混跑、EOS 提前退场后 batch slot / physical block 复用及缓存动态扩容等边界工况。

---

## 二、假设驱动的性能评测与统计严谨性

1. **统计有效性**
   - 性能测量必须包含预热迭代（`warmup`），消除 CUDA Context 初始化、JIT 编译与 Caching Allocator 冷启动抖动。
   - 性能报告与回归判定必须基于多轮测量（`repeats`）的**中位数与分位数（P50 / P95）**，严禁仅凭单次运行或算术平均值（Mean）下结论。
   - 性能回归门禁（`check_regression`）必须同时结合相对退化阈值（如 `> 5%`）与 `3 * std` 噪声保护带，避免硬件微小波动误报。
2. **多维硬件与显存指标正交采集**
   - 结果需正交呈现延迟（`TTFT` / `TPOT` / `ITL` / `Makespan` / `Queueing`）、吞吐（`tok/s`、`req/s`）与硬件效率（`MBU` / `MFU`）。
   - 显存指标严格区分（对应 `MemoryProfile` / `KVCacheProfile`）：**峰值已分配显存（`peak_allocated_mb`）**、**Allocator 预留与碎片（`peak_reserved_mb` / `fragmentation_mb`）**、**KV Cache 物理池容量（`peak_kv_cache_pool_mb`）**与**活跃 KV 占用（`peak_live_kv_cache_mb`）**。

---

## 三、基准数据卫生与防污染红线

1. **严禁污染受控历史归档与黄金基准**
   - 日常开发与调优实验的基准输出**必须**写入 `tmp/`（如 `tmp/results/`）或命名为 `*.local.json`（已由 `.gitignore` 忽略）。
   - 以下受版本控制的权威资产**默认禁止手动编辑、伪造或用非标准参数覆盖**（仅在 Schema 迁移或使用固定随机种子 `seed=42` 的标准 Release SOP 中受控更新）：
     - `benchmarks/results/history/{lockstep,scheduler}/*.json`
     - `benchmarks/golden/{reference,golden_output_tokens}.json`
2. **Schema 强类型契约同步**：评测 JSON 由 `src/mini_infra/bench/result/schema.py` 的三层 Pydantic v2 模型与判别式联合（`AnyBenchmarkRecord`）守护；增删改任何统计指标必须同步更新 Pydantic Record、历史 JSON、绘图脚本与单测（见 `AGENTS.md` §一.4），严禁塞入未声明的裸字典字段。
