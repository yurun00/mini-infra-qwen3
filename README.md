# mini-infra-qwen3

本项目采用 **特性驱动（Feature-Driven）** 的工程演进路线，以最精简、清晰的 Python 和 PyTorch 原生实现，从零解析现代大模型推理引擎的核心机制与优化演进。

---

## 特性全景 (Features)

### 已实现特性
- **手写 Qwen3-0.6B 模型架构**：基于纯原生 PyTorch 实现完整前向计算，不依赖 Transformers 模型代码，不引入黑盒优化，清晰展示每一层计算逻辑。
- **轻量原生 Tokenizer**：纯 Python + regex 实现的 Byte-level BPE Tokenizer，直接解析权重目录下的 `tokenizer.json`，内建 non-thinking Chat Template，不依赖 tokenizers / Transformers。
- **KV Cache 显存与计算优化**：实现原生显式 `KVCache` 容器，Prefill 阶段建立缓存，自回归 Decode 每步只计算增量 Token，消除历史重复前向计算；支持精确显存占用统计与逐步延迟分位数（P50/P95）分析。
- **单请求与固定批处理（Fixed Batching）自回归推理**：支持 greedy decoding 逐 Token 生成与左填充（Left Padding）对齐，支持 `--compare-feature` 对比特性标志（如 `kv-cache`）或批处理策略（如 `fixed-batching`）并自动计算加速比。
- **严格精度对照**：内建与 Hugging Face 参考实现的逐 Token 对齐程序，以及手写模型开启/关闭 KV Cache 的逐步 Logits 对齐工具。
- **轻量 CPU 快速自检**：使用随机生成的小尺寸配置模型，无需下载真实模型权重即可在数秒内验证因果掩码与网络计算。
- **Continuous Batching 调度与并发调度基准**：实现原生 `DenseSlotCache` 显存槽位管理与迭代级动态组批调度，短请求即时退出、新到达请求即时抢占空槽，消除队头阻塞；内建 `mini-infra bench scheduler` 受控评测套件，覆盖混合预算加速、到达抖动、首字延迟与 Prefill 打断分析。

### 规划中特性 (Roadmap)
- [x] **Continuous Batching 调度**：支持多请求动态并发与动态组批（DenseSlotCache + `mini-infra bench scheduler`）
- [ ] **PagedAttention 内存管理**：非连续显存分页分配与碎片消除
- [ ] **FlashAttention / SDPA 算子融合**：优化 Attention 显存与计算访存瓶颈
- [ ] **FlashAttention-Varlen**：针对可变长度输入的 FlashAttention  
- [ ] **Tensor Parallelism (TP)**：张量并行分布式推理

---

## 目录结构

```text
mini-infra-qwen3/
├── src/
│   └── mini_infra/
│       ├── __init__.py                # 统一导出核心 API：generate, GenerationResult, Mode, load_model 等
│       ├── device.py                  # 通用硬件设备同步与显存统计工具
│       ├── utils.py                   # 通用基础数学与统计工具（percentile 分位数计算等）
│       ├── model/                     # 统一模型与 Tokenizer 域
│       │   ├── config.py              # Qwen3Config 模型配置
│       │   ├── qwen3.py               # RMSNorm, RotaryEmbedding, Attention, MLP, Qwen3ForCausalLM
│       │   └── tokenizer.py           # Qwen3Tokenizer
│       ├── memory/                    # 显存与 KV Cache 管理体系
│       │   ├── kv_cache.py            # KVCache 键值缓存容器与显存统计
│       │   └── slot_cache.py          # DenseSlotCache 连续批处理槽位管理器
│       ├── core/                      # 调度与请求核心领域域（纯 CPU 逻辑，无 GPU/模型依赖）
│       │   ├── request.py             # RequestSpec, RequestState (append_token), RequestStatus 请求生命周期模型
│       │   ├── scheduler.py           # 三级队列（ArrivalQueue -> WaitingQueue -> RunningQueue: admit/step/retire）
│       │   └── types.py               # SequenceOutput, RequestMetrics, MemoryProfile, BatchScheduleInfo 跨层共享基础类型
│       ├── entrypoints/               # 用户级推理入口域（对标 vLLM entrypoints）
│       │   ├── offline.py             # 离线文本生成薄入口（Offline Entrypoint）：generate
│       │   └── types.py               # GenerationOptions, GenerationResult, aggregate_generation_results
│       ├── engine/                    # 推理与调度执行引擎域
│       │   ├── runner.py              # 模型执行器 ModelRunner（step_forward / prefill_requests / decode_slots）
│       │   ├── lockstep_loop.py       # 齐步自回归生成主循环（run_lockstep：所有请求同时开始，无调度器）与状态机 LockstepState
│       │   ├── scheduler_loop.py      # 请求队列调度主循环（run_scheduler：统一驱动 CONTINUOUS 与 FIXED）
│       │   ├── mode.py                # Mode、FeatureFlag 与 BatchingMethod 模式定义
│       │   ├── preprocessing.py       # Prompt 规约、左填充（Left Padding）与 Batch 张量初始化
│       │   └── types.py               # LockstepResult, ScheduleResult, ScheduleMetrics, ScheduleEvent 引擎层结果类型
│       ├── profiling/                 # 算子耗时剖析域
│       │   ├── tracer.py              # PyTorch Profiler 打点、Scope 注入与 ProfileCollector
│       │   ├── analyzer.py            # 耗时统计、Kernel 分类与跨 run 聚合
│       │   └── types.py               # OpLatencyBreakdown, KernelInfo, TensorBoardOptions
│       ├── cli/                       # 统一命令行门面层（mini-infra 统一主入口）
│       │   ├── main.py                # 根命令 mini-infra 注册与分发
│       │   └── commands/              # run / bench / verify 分级子命令实现
│       │       ├── run.py             # mini-infra run：单请求/批处理推理与对照
│       │       ├── verify/            # mini-infra verify (smoke / transformers / cache / lockstep-batch / scheduler-batch)
│       │       └── bench/             # mini-infra bench (lockstep / scheduler / plot / check-regression)
│       ├── verification/              # 正确性验证与数值等价性核心库
│       │   ├── smoke.py               # 快速自检算法实现（CPU）
│       │   └── parity/                # 精度与等价性对齐核心库
│       │       ├── compare_cache.py   # KV Cache 前后逐步 Logits 对齐
│       │       ├── compare_transformers.py # 与 Transformers 对齐
│       │       ├── compare_lockstep_batch.py # Lockstep 齐步批处理不同变种的请求隔离与数值对齐
│       │       └── compare_scheduler_batch.py # Scheduler 批处理（Fixed/Continuous）动态请求归属与输出一致性对齐
│       └── bench/                     # 性能基准测试与分析核心库
│           ├── common.py              # 共享基准基础设施、指标计算与目录常量
│           ├── workloads.py           # 合成 Prompt、错峰到达与长短预算序列生成工具
│           ├── regression.py          # 性能回归质量门禁核心逻辑
│           ├── lockstep/              # 齐步（Lockstep：所有请求同时开始，无调度器）基准测试
│           │   ├── types.py           # LockstepCaseResult, DistributionStats 等领域模型
│           │   ├── suites.py          # 齐步批处理套件定义
│           │   ├── display.py         # 终端表格格式化与基准记录封装
│           │   ├── run_case.py        # 单 case 基准测试执行引擎
│           │   └── run_suite.py       # 预设测试套件执行引擎
│           ├── scheduler/             # 队列调度器（Scheduler：Fixed vs Continuous Batching）基准测试
│           │   ├── types.py           # SchedulerCaseResult, SchedulerSuiteResult 等领域模型
│           │   ├── suites.py          # Scheduler 受控实验套件与 SchedulerSuiteCase
│           │   ├── plan.py            # 特征多维展开与基于 Feature 的 Interleave 执行计划生成器
│           │   ├── run_case.py        # 单用例评测执行引擎
│           │   └── run_suite.py       # Scheduler 套件计划执行引擎
│           └── plotting/              # 绘图工具包
│               ├── history.py         # 读取 result-history/ 生成演进趋势图
│               └── compare.py         # 对比独立 JSON 结果绘制多维对比图
├── tests/
│   ├── unit/                          # CPU 单元测试（pytest）
│   └── gpu/
│       └── parity/                    # GPU 精度与等价性对齐测试（pytest tests/gpu/parity/）
├── benchmarks/
│   ├── data/                          # golden_output_tokens.json、reference.json（需提交）
│   ├── suites/                        # 预设基准测试套件配置（YAML 及 serving 结果）
│   ├── result-history/                # 每个版本一份 <commit>_<tag>.json
│   └── reports/                       # SUMMARY.md、evolution_and_scaling.png、profiling_and_optimization.png（生成物）
├── docs/                              # 技术文档中心
│   ├── index.md                       # 文档导航主页
│   ├── architecture/                  # 系统架构原理（KV Cache、Fixed Batching、Continuous Batching）
│   ├── guides/                        # 实践指南（数值对齐验证、算子 Profiling 与 Trace）
│   └── benchmarks/                    # 性能基准与评测全景指南
├── .github/workflows/
│   ├── cpu-ci.yml                     # CPU 上 ruff + pytest
│   ├── pr-perf-regression.yml         # PR 提测：GPU runner 上的性能回归守门
│   └── release-benchmark.yml          # Tag 发布：GPU runner 上的全量基准与趋势图
├── .pre-commit-config.yaml            # ruff（commit）+ 性能回归（pre-push）
├── pyproject.toml                     # 依赖、入口点、ruff/pytest 配置的唯一来源
├── requirements.txt                   # 基础推理最小依赖
├── requirements-compare.txt           # 对齐评测扩展依赖
├── requirements-lock.txt              # 完整依赖锁文件
├── CONTRIBUTING.md                    # 贡献指南与日常开发工作流（CPU/GPU 测试、质量守门）
├── RELEASE.md                         # 特性发布与 Minor Release 标准流程 (SOP)
├── LICENSE                            # 开源许可证
└── README.md                          # 项目文档与使用指南
```

---

## 验证环境

- **实例规格**：GCP `g2-standard-4`
- **GPU**：NVIDIA L4 24GB
- **CPU**：4 vCPU (Intel Xeon @ 2.20GHz)
- **内存**：16 GiB
- **操作系统**：Ubuntu 22.04 LTS
- **Python**：3.10
- **GPU 驱动**：580.178.04
- **CUDA Runtime**：12.6
- **PyTorch**：2.7.1+cu126
- **Transformers**：4.51.3
- **模型 revision**：`c1899de289a04d12100db370d81485cdf75e47ca` (`Qwen/Qwen3-0.6B`)

---

## 快速上手

### 环境准备与安装

从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 安装完整锁版本依赖（推荐，用于精确复现）
python -m pip install -r requirements-lock.txt

# 或者只安装基础运行依赖
python -m pip install -r requirements.txt
```

安装项目（命令行入口依赖它）：
```bash
pip install -e .                    # 仅推理
pip install -e ".[compare,bench]"   # 加上 Transformers 对照与绘图
pip install -e ".[dev]"            # 开发：pytest、ruff、pre-commit
```

> **提示**：首次运行会从 Hugging Face 下载固定版本的 `Qwen/Qwen3-0.6B`。如果服务器访问 Hugging Face 受限，可设置镜像端点：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

---

### 快速自检

使用随机生成的小模型在 CPU 上快速验证模型结构与 Causal Mask 计算正确性（无需下载真实权重）：

```bash
mini-infra verify smoke
```

输出 `smoke test passed` 即表示自检通过。完整的 CPU 测试集（含 smoke test）可用 `pytest -m "not gpu"` 运行。

---

### 运行推理与计时

```bash
# 1. 比较指定特性的 enable 与 disable 对照（如 KV Cache 开启/关闭）
mini-infra run \
  --compare-feature kv-cache \
  --prompts "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32

# 2. 单模式或固定批处理（Fixed Batching）运行多条提示词
mini-infra run \
  --batching-method fixed-batching \
  --prompts "用一句话解释 Prefill。" "用一句话解释 Decode。" \
  --max-new-tokens 32

# 3. 对比两种批处理策略（如串行 none 与固定批处理 fixed-batching）
mini-infra run \
  --compare-feature none fixed-batching \
  --prompts "用一句话解释 Prefill。" "用一句话解释 Decode。" \
  --max-new-tokens 32

# 4. 对比 Fixed 与 Continuous Batching（逐请求输出预算 + 并发上限）
mini-infra run \
  --compare-feature fixed-batching continuous-batching \
  --prompts "用一句话解释 Prefill。" "用一句话解释 Decode。" "用一句话解释 KV Cache。" "用一句话解释固定批处理。" \
  --max-new-tokens 8 24 8 24 \
  --max-running-requests 2
```

`--max-new-tokens` 可传 1 个值（所有请求共用）或与 prompts 数量一致的逐请求预算。只要 `--batching-method` 或 `--compare-feature` 中出现 continuous-batching，所有 fixed/continuous 结果统一由 `run_scheduler` 产出（受 `--max-running-requests` 约束），并额外输出 Makespan、排队/TTFT/ITL 分位数与 Cache 管理耗时占比；该路径暂不支持 `--profile`。

输出包含生成 Token IDs、生成文本、Prefill 延迟、平均 Decode 延迟、P50/P95 步延迟、Prefill 后已分配显存、峰值显存与最终 KV Cache 数据量；在 `--compare-feature` 模式下自动给出两组生成的逐 Token 一致性校验及加速比分析。

---

## 性能评估与基准演进 (Benchmarks)

本项目采用严格的数据驱动与时间线演进追踪，每个特性的引入均经过全矩阵基准测试与性能诊断。

完整的多版本评测数据表、加速比对比与 Profiling 诊断详见 [SUMMARY.md](./benchmarks/reports/SUMMARY.md)。

### 1. 跨版本演化与扩展趋势 (Evolution & Scaling Across Releases)
![Evolution & Scaling](./benchmarks/reports/evolution_and_scaling.png)

### 2. 当前版本深度剖析与优化指引 (Profiling & Next Steps)
![Profiling](./benchmarks/reports/profiling_and_optimization.png)

---

## 工程规范与工作流指引 (Workflows & Guides)

为了保持项目首页高信噪比，各阶段详细的工程实践与操作指南已整理归档至专属文档：

* 🛠️ **[贡献与日常开发指南 (CONTRIBUTING.md)](CONTRIBUTING.md)**
  * 本地开发环境配置与 Pre-commit 守护安装
  * 代码规范检查（Ruff）与 CPU 快速冒烟自检（`mini-infra verify smoke`、pytest）
  * GPU 端到端精度对齐（[数值对齐与等价性验证指南 (docs/guides/parity-verification.md)](docs/guides/parity-verification.md)）
  * 零性能倒退守护（Zero-Regression Guard：`mini-infra bench check-regression`）与独立全量基准对比测量
* 📊 **[基准测试与性能评测指南 (docs/benchmarks/overview.md)](docs/benchmarks/overview.md)**
  * Lockstep 齐步基准（单用例与批处理吞吐/Padding/槽位扩展套件）
  * Scheduler 调度器基准（Continuous vs Fixed 动态准入与长短请求压测套件）
  * 跨版本演进时间线图（`bench plot history`）与独立对比分析（`bench plot compare`）
* 🔍 **[算子性能剖析与 Trace 指南 (docs/guides/profiling.md)](docs/guides/profiling.md)**
  * 使用 `mini-infra bench lockstep case` 采样生成详细性能诊断数据
  * 导出 Chrome / TensorBoard Trace（`*.pt.trace.json`）
  * 使用 **Perfetto Web UI** 秒级加载与 60fps 交互缩放查看 CUDA Stream 与 Kernel 耗时
  * 远程实例 TensorBoard 启动与 IDE / SSH 端口转发
  * 硬件带宽利用率（MBU）与显存碎片诊断思路
* 🚀 **[特性发布流程 SOP (RELEASE.md)](RELEASE.md)**
  * 核心特性发版 5 步走（质量自检 → 版本号升级 → 全量基准固化 → 代码提交 → Git Tag 打标）
  * 数据流约定与特性（Release Feature）命名规范
  * GitHub Actions 自动云端测试与评测产物归档机制

---

## 致谢与参考 (Acknowledgments)

- [SVDX-lab/miniinfra](https://github.com/SVDX-lab/miniinfra/)

