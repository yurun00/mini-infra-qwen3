# mini-infra-qwen3 技术文档中心 (Documentation)

欢迎查阅 **mini-infra-qwen3** 核心技术文档。本项目聚焦于从零构建现代化大语言模型（LLM）推理系统，涵盖模型手写、显式 KV Cache 状态管理、动态连续批处理调度（Continuous Batching）以及系统级性能评测与剖析。

---

## 📚 文档目录架构

### 1. 核心架构与原理 (Architecture)
深入剖析推理引擎的设计理念、状态流转与核心调度算法。
* [连续批处理架构与调度原理 (Continuous Batching)](architecture/continuous-batching.md)：详解 Iteration-level Scheduling、槽位动态管理、Prefill 插队与请求生命周期状态机。

### 2. 实践与操作指南 (Guides)
面向日常开发、调试与正确性验证的操作指引。
* [数值对齐与等价性验证指南 (Parity Verification)](guides/parity-verification.md)：涵盖 Hugging Face Transformers、显式 KV Cache、Lockstep 批处理与动态 Scheduler 批处理的四级严格精度验证方案。
* [算子性能剖析与 Trace 指南 (Profiling & Traces)](guides/profiling.md)：使用 PyTorch Profiler 采集算子细粒度耗时，导出 Chrome/Perfetto Trace，并利用 TensorBoard 进行 MBU/MFU 带宽利用率诊断。

### 3. 基准测试与性能评测 (Benchmarks)
面向服务化评测、吞吐/延迟测量与质量守门。
* [基准测试与性能评测全景指南 (Benchmarking Overview)](benchmarks/overview.md)：系统汇总 Lockstep 齐步测试、Scheduler 动态调度评测、历史演进趋势图绘制与 CI 零性能回归守护。

