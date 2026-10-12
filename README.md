# mini-infra-qwen3

本项目实现是基于 https://github.com/SVDX-lab/miniinfra/ 《超轻量AI Infra教程—推理篇》的代码仓库，以最精简、清晰的 Python 和 PyTorch 原生实现深入解析大模型推理引擎的核心机制与优化演进。

本项目采用**特性驱动（Feature-Driven）**的工程演进路线，从最基础的原生模型计算与分词出发，逐步演进增加现代推理引擎的关键特性。

---

## 特性全景 (Features)

### 已实现特性
- **手写 Qwen3-0.6B 模型架构**：基于纯原生 PyTorch 实现完整前向计算，不依赖 Transformers 模型代码，不引入黑盒优化，清晰展示每一层计算逻辑。
- **轻量原生 Tokenizer**：纯 Python + regex 实现的 Byte-level BPE Tokenizer，直接解析权重目录下的 `tokenizer.json`，内建 non-thinking Chat Template，不依赖 tokenizers / Transformers。
- **单请求自回归推理**：支持 greedy decoding 逐 Token 生成，实现 Prefill 与 Decode 耗时及峰值显存测量。
- **严格精度对照**：内建与 Hugging Face 参考实现的逐 Token 对齐程序，验证手写分词器输入与模型 Logits 绝对误差（在 bfloat16 下达到 0 误差）。
- **轻量 CPU 快速自检**：使用随机生成的小尺寸配置模型，无需下载真实模型权重即可在数秒内验证因果掩码与网络计算。

### 规划中特性 (Roadmap)
- [ ] **KV Cache 支持**：Prefill 与自回归 Decode 计算解耦，避免重复计算历史上下文
- [ ] **FlashAttention / SDPA 算子融合**：优化 Attention 显存与计算访存瓶颈
- [ ] **Continuous Batching 调度**：支持多请求动态并发与动态组批
- [ ] **PagedAttention 内存管理**：非连续显存分页分配与碎片消除
- [ ] **Tensor Parallelism (TP)**：张量并行分布式推理

---

## 目录结构

```text
mini-infra-qwen3/
├── src/
│   └── mini_infra/
│       ├── __init__.py                # mini_infra 包导出入口
│       ├── qwen3_model.py             # Qwen3-0.6B 结构与前向算子手写实现
│       ├── qwen3_tokenizer.py         # 教学版 Byte-level BPE Tokenizer
│       ├── smoke.py                   # mini-infra-smoke：随机小模型快速自检（CPU）
│       ├── cli/
│       │   ├── infer.py               # mini-infra-run：单请求推理与计时
│       │   └── compare.py             # mini-infra-compare：与 Transformers 对齐
│       └── bench/
│           ├── run.py                 # mini-infra-bench：统计型 benchmark（含正确性门禁）
│           ├── plot.py                # mini-infra-bench-plot：加速比汇总表与趋势图
│           └── regression.py          # mini-infra-bench-check：性能回归检查
├── tests/
│   ├── unit/                          # CPU 单元测试（pytest）
│   └── gpu/                           # 需要 GPU/权重的测试（pytest -m gpu）
├── benchmarks/
│   ├── data/                          # golden_output_tokens.json、reference.json（需提交）
│   ├── results/                       # 每个版本一份 <commit>_<tag>.json
│   └── reports/                       # SUMMARY.md、trend.png（生成物）
├── .github/workflows/
│   ├── ci.yml                         # CPU 上 ruff + pytest
│   └── perf-regression.yml            # GPU runner 上的性能回归检查
├── .pre-commit-config.yaml            # ruff（commit）+ 性能回归（pre-push）
├── pyproject.toml                     # 依赖、入口点、ruff/pytest 配置的唯一来源
├── requirements.txt                   # 基础推理最小依赖
├── requirements-compare.txt           # 对齐评测扩展依赖
├── requirements-lock.txt              # 完整依赖锁文件
├── LICENSE                            # 开源许可证
└── README.md                          # 项目文档与使用指南
```

---

## 验证环境

- **GPU**：NVIDIA GeForce RTX 3080 Ti 12GB
- **CPU**：Intel Xeon Gold 6226R
- **内存**：31 GiB
- **操作系统**：Ubuntu 22.04
- **Python**：3.10
- **GPU 驱动**：595.80
- **CUDA Runtime**：12.6
- **PyTorch**：2.7.1+cu126
- **Transformers**：4.51.3
- **模型 revision**：`c1899de289a04d12100db370d81485cdf75e47ca` (`Qwen/Qwen3-0.6B`)

---

## 快速上手

### 1. 准备环境与安装依赖

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

### 2. 快速自检

使用随机生成的小模型在 CPU 上快速验证模型结构与 Causal Mask 计算正确性（无需下载真实权重）：

```bash
mini-infra-smoke
```

输出 `smoke test passed` 即表示自检通过。完整的 CPU 测试集（含 smoke test）可用 `pytest -m "not gpu"` 运行。

---

### 3. 运行推理与计时

```bash
mini-infra-run \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32
```

输出包含生成 Token IDs、生成文本、Prefill 延迟、平均 Decode 延迟和峰值显存占用（计时排除 tokenizer、模型下载与权重加载开销）。

---

### 4. 与 Transformers 对照测试

```bash
mini-infra-compare \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32
# 或作为 pytest GPU 用例：pytest -m gpu
```

对照程序会验证分词一致性，并在相同 greedy decoding 条件下逐步比较手写模型与 Transformers 的：
- 下一个 Token ID；
- Logits 最大绝对误差；
- Logits 平均绝对误差；
- 最终生成文本。

---

## 质量保证与性能发布工作流

本项目采用分层测试与性能工程管理策略：**日常提交保证高周转与零倒退（Zero-Regression），大版本/新特性发布时沉淀全矩阵性能对比与时间线演进图**。

### 1. 日常开发与提交（Normal Commits & PRs）
- **快速自检**：`mini-infra-smoke`（仅需 2 秒，CPU 快速验证网络前向与因果掩码）。
- **单元测试**：`pytest -m "not gpu"`（覆盖算子结构、输入输出形状与基准回归比对逻辑）。
- **零性能倒退守护（Zero-Regression Guard）**：
  ```bash
  mini-infra-bench-check    # 轻量级（约 1 分钟），TTFT / TPOT 中位数劣化 >5% 自动拦截报警
  ```
  - 本地：`pip install -e ".[dev]" && pre-commit install --hook-type pre-commit --hook-type pre-push`，commit 运行 ruff，push 前运行回归守护。
  - CI：PR 会在 GPU runner 上自动触发 [perf-regression.yml](.github/workflows/perf-regression.yml)。

---

### 2. 特性发布与 Minor Release（Feature Deliveries & Tags）
每当完成一个重大优化特性（如 KV Cache、FlashAttention 等，对应 `v0.2.0`、`v0.3.0` 等 Minor Release）时，执行完整的基准测量与跨版本时间线演进生成：

```bash
pip install -e ".[bench]"

# 1. 运行全量矩阵测试并记录新版本特性（以 KV Cache 为例）
mini-infra-bench --tag "v0.2.0-kv-cache"

# 2. 自动生成多版本演化表与对比图
mini-infra-bench-plot

# 3. 将新性能固化为后续日常开发的零退化底线
mini-infra-bench-check --update-reference
```

#### 生成产物：
- **`benchmarks/reports/SUMMARY.md`**：记录所有历史版本的详细数据与相对 Baseline 加速比。
- **`benchmarks/reports/timeline.png`（特性演进时间线图）**：横轴展示随着各个 Feature（Baseline $\to$ KV Cache $\to$ FlashAttention...）的引入，Decode 延迟逐步下降、吞吐逐步提升、加速比递增的演变趋势。
- **`benchmarks/reports/trend.png`（序列长度扩展曲线）**：各版本在不同 Prompt 长度（32 $\sim$ 1024+）下的延迟变化曲线。

#### CI 自动化：
推送版本 Tag（如 `git tag v0.2.0 && git push --tags`）或在 GitHub 上手动触发 [release-benchmark.yml](.github/workflows/release-benchmark.yml)，即可全自动执行全量 Benchmark 并归档报告产物。

---

## 3080 Ti 实测基准记录 (Baseline)

测试配置：bfloat16、greedy decoding、关闭 thinking，并强制 Transformers 使用 eager attention 和关闭 KV Cache。

| Prompt | 最大生成长度 | 实际对比 Token 数 | Token 是否一致 | Logits 最大绝对误差 |
| :--- | :---: | :---: | :---: | :---: |
| `请用一句话介绍 KV Cache。` | 32 | 32 | 是 | 0.000000 |
| `Explain prefill in one short sentence.` | 16 | 16 | 是 | 0.000000 |
| `计算 17 乘以 23，只给出结果。` | 16 | 12（遇到 EOS） | 是 | 0.000000 |

基准性能记录（以 `请用一句话介绍 KV Cache。`、19 输入 Token、32 输出 Token、2 次 warm-up 为例）：
- **Prefill 延迟**：34.43 ms
- **平均 Decode 延迟**：33.40 ms/token
- **端到端模型计算延迟**：1069.80 ms
- **峰值已分配显存**：1173.80 MiB

---

## 当前特性限制与已知边界

- **模型架构**：当前仅适配固定版本的 `Qwen/Qwen3-0.6B` dense 模型。
- **并发与批处理**：当前仅支持 `batch_size = 1`、无 padding 的单请求输入。
- **采样策略**：当前仅实现 greedy decoding。
- **对话模板**：默认关闭 thinking 模式；手写 Chat Template 当前覆盖纯文本 system/user 消息，不支持工具调用与多模态。
- **状态缓存**：当前尚未引入 KV Cache，每次 Decode 均重新计算完整前向上下文（作为后续引入 KV Cache 对比性能的原始基准）。
