---
name: Performance Analysis
about: 深入分析推理与调度引擎的性能瓶颈（如 CPU 调度开销、显存带宽受限、算子执行时延、Prefill/Decode 分布等）
title: "[Perf] Analysis: <Target Component or Bottleneck>"
labels: ["type:performance", "type:analysis"]
assignees: ""
---

## 📌 问题描述 (Problem Description)

<!-- 
清晰阐述本次性能分析的背景、观察到的异常现象或分析动机：
1. 现象与痛点：例如吞吐未达预期、Decode 阶段 ITL 波动、GPU 算力利用率低、CPU 调度存在显著开销等。
2. 目标分析范围：明确涉及的引擎架构层（Control Plane / Data Plane / Compute Plane）与运行模式（如 Dense / Paged、Lockstep / Dynamic / Continuous）。
3. 理论预期：分析对象理论上属于 Compute-Bound（如长 Prefill GEMM）还是 Memory-Bandwidth / Host Overhead-Bound（如 Decode 算子、调度与 Tensor 拼装）。
-->

### 1. 现象与背景
- 

### 2. 分析目标与范围
- **受影响模块/模式**：
- **理论瓶颈预期**：[ ] Compute-Bound / [ ] Memory-Bandwidth-Bound / [ ] CPU Host Overhead-Bound

---

## 🖥️ 实验环境与测试负载 (Environment & Workload)

### 硬件与软件环境
- **Model**: `Qwen/Qwen3-0.6B`
- **Precision / Dtype**: `torch.bfloat16`
- **GPU**: NVIDIA L4 (24GB VRAM, PCIe Gen4)
- **CUDA / Driver**: 
- **PyTorch**: 

### 负载配置 (Workload Settings)
- **Batching Method**: <!-- e.g., lockstep / dynamic / continuous -->
- **Cache Mode**: <!-- e.g., dense / paged -->
- **Request Count / Concurrency**: 
- **Input Length (Prompt)**: 
- **Output Length (Budget)**: 

### 复现命令 (Reproduction Command)
```bash
# 执行性能评测或 Profiling 采样的具体 CLI 命令
mini-infra bench scheduler case --tag perf-analysis ...
```

---

## 🔬 分析方法 (Analysis Methodology)

<!-- 
说明采用的观测工具、度量指标与拆解逻辑：
1. 采集工具：PyTorch Profiler、Chrome Trace、mini-infra Profiling Snapshot、Nsight Systems 等。
2. 分解维度：
   - Control Plane (CPU)：调度队列推进、Logical Allocator（batch slot / block table 映射）耗时。
   - Data Plane (Host & GPU)：Input Tensor 组装、slot_mapping 收集、KV Cache 读写与 Sampler 耗时。
   - Compute Plane (GPU Kernel)：Attention、GEMM/Linear、RoPE、RMSNorm 细粒度耗时与显存带宽占用。
-->

- **使用的工具/手段**：
- **核心观测指标**：<!-- e.g., Step Time, GPU Kernel Duration, CPU Overhead, DRAM Bandwidth, SM Throughput -->
- **细粒度拆解链路**：
  1. **Control Plane** (CPU 调度与元数据准备):
  2. **Data Plane** (Host 侧输入组装与 KV 物理池寻址):
  3. **Compute Plane** (GPU Kernel 前向计算与设备端采样):

---

## 📊 结果输出 (Results & Findings)

<!-- 
展示 Profiling 采集到的数据、表格、Trace 关键点与图表：
-->

### 1. 耗时分解表 (Time Breakdown)

| 阶段 / 算子 | CPU 耗时 (ms) | GPU 耗时 (ms) | 占总前向耗时比例 (%) | 瓶颈属性 (Compute/Memory/Overhead) |
|---|---|---|---|---|
| Schedule & Allocator | | - | | Host Overhead |
| Input Tensor Assemble | | - | | Host Overhead |
| Prefill Attention | | | | |
| Decode Attention | | | | |
| MLP / Linear Projections | | | | |
| Sampler | | | | |
| **Total / Step** | | | 100% | |

### 2. Trace 与关键数据佐证
- **关键 Trace 截图 / 热点片段**：
  <!-- 可贴入 Chrome Trace / Nsys 截图或文字片段，例如：存在大段 GPU Idle 等待 Host Launch、算子未占满带宽等 -->
- **重要度量观测值**：
  - DRAM Throughput / Bandwidth 达成率：
  - GPU SM 利用率：
  - CPU Launch / Dispatch 延迟占比：

---

## 💡 分析结论 (Conclusions & Bottleneck Attribution)

<!-- 
对上述实验数据进行深入归纳与瓶颈定性：
1. 根本瓶颈定位：到底是受限于 Python/CPU 调度开销、GPU 算子 Launch 延迟、Memory-Bound 访存吞吐，还是计算单元占满？
2. 架构层面的成因：为什么会出现这一瓶颈？现有的数据结构或算法有何限制？
-->

### 1. 根因归因 (Root Cause)
- 

### 2. 瓶颈分类定性
- **主要瓶颈**：
- **次要瓶颈/潜在隐患**：

---

## 🛠️ 优化建议与行动项 (Optimization Proposals & Action Items)

<!-- 
根据分析结论提出的具体优化建议与跟踪计划：
-->

- [ ] **短期优化方案 (Quick Wins)**:
  - 
- [ ] **中长期架构改进 (Long-term Improvements)**:
  - 
- **预期收益评估**：
  - 预计吞吐 / 延迟改善：

---

## 📎 附件与产物 (Artifacts & References)

- **Trace 文件 / Snapshot 路径**：
- **相关 Issue / PR**：

