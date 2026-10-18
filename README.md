# mini-infra-qwen3

本项目采用**特性驱动（Feature-Driven）**的工程演进路线，以最精简、清晰的 Python 和 PyTorch 原生实现，从零解析现代大模型推理引擎的核心机制与优化演进。

---

## 特性全景 (Features)

### 已实现特性
- **手写 Qwen3-0.6B 模型架构**：基于纯原生 PyTorch 实现完整前向计算，不依赖 Transformers 模型代码，不引入黑盒优化，清晰展示每一层计算逻辑。
- **轻量原生 Tokenizer**：纯 Python + regex 实现的 Byte-level BPE Tokenizer，直接解析权重目录下的 `tokenizer.json`，内建 non-thinking Chat Template，不依赖 tokenizers / Transformers。
- **KV Cache 显存与计算优化**：实现原生显式 `KVCache` 容器，Prefill 阶段建立缓存，自回归 Decode 每步只计算增量 Token，消除历史重复前向计算；支持精确显存占用统计与逐步延迟分位数（P50/P95）分析。
- **单请求自回归推理**：支持 greedy decoding 逐 Token 生成，支持 `--mode {both, kv-cache, no-cache}` 对照运行并自动计算加速比。
- **严格精度对照**：内建与 Hugging Face 参考实现的逐 Token 对齐程序，以及手写模型开启/关闭 KV Cache 的逐步 Logits 对齐工具。
- **轻量 CPU 快速自检**：使用随机生成的小尺寸配置模型，无需下载真实模型权重即可在数秒内验证因果掩码与网络计算。

### 规划中特性 (Roadmap)
- [ ] **FlashAttention / SDPA 算子融合**：优化 Attention 显存与计算访存瓶颈
- [ ] **FlashAttention-Varlen**：针对可变长度输入的 FlashAttention  
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
│       ├── cli/
│       │   └── run_inference.py       # mini-infra-run：单请求推理与计时
│       ├── testing/
│       │   ├── smoke_test.py          # mini-infra-smoke：随机小模型快速自检（CPU）
│       │   ├── compare_cache.py       # mini-infra-compare-cache：KV Cache 前后逐步 Logits 对齐
│       │   └── compare_with_transformers.py # mini-infra-compare：与 Transformers 对齐
│       └── bench/
│           ├── run.py                 # mini-infra-bench：全量基准测量（落盘至 result-history/，plot 唯一数据源）
│           ├── plot_history.py        # mini-infra-bench-plot-history：读取 result-history/ 生成汇总表与演进趋势图
│           ├── plot_compare.py        # mini-infra-bench-plot-compare：对比两个独立 JSON 结果（如不同 mode）并绘制多维对比图
│           └── regression.py          # mini-infra-bench-regression：轻量质量守门（仅 0/1 退出码，不参与画图）
├── tests/
│   ├── unit/                          # CPU 单元测试（pytest）
│   └── gpu/                           # 需要 GPU/权重的测试（pytest -m gpu）
├── benchmarks/
│   ├── data/                          # golden_output_tokens.json、reference.json（需提交）
│   ├── result-history/                # 每个版本一份 <commit>_<tag>.json
│   └── reports/                       # SUMMARY.md、trend.png（生成物）
├── .github/workflows/
│   ├── cpu-ci.yml                     # CPU 上 ruff + pytest
│   ├── pr-perf-regression.yml         # PR 提测：GPU runner 上的性能回归守门
│   └── release-benchmark.yml          # Tag 发布：GPU runner 上的全量基准与趋势图
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
mini-infra-smoke
```

输出 `smoke test passed` 即表示自检通过。完整的 CPU 测试集（含 smoke test）可用 `pytest -m "not gpu"` 运行。

---

### 运行推理与计时

```bash
# 默认 --mode both：同时运行 no-cache 与 kv-cache 进行端到端对照并自动输出加速比
mini-infra-run \
  --prompt "请用一句话介绍 KV Cache。" \
  --max-new-tokens 32

# 也可指定单模式运行：--mode no-cache 或 --mode kv-cache
mini-infra-run --mode kv-cache --max-new-tokens 64
```

输出包含生成 Token IDs、生成文本、Prefill 延迟、平均 Decode 延迟、P50/P95 延迟、逐步 Decode 耗时序列、Prefill 后已分配显存、峰值显存与最终 KV Cache 数据量；在 `both` 模式下自动给出两组生成的逐 Token 一致性校验及平均 Decode 加速比。

---

## 质量保证与性能发布工作流

本项目采用分层测试与性能工程管理策略：**日常提交保证高周转与零倒退（Zero-Regression），大版本/新特性发布时沉淀全矩阵性能对比与时间线演进图**。

### 日常开发与提交（Normal Commits & PRs）

在日常开发与 PR 提测过程中，通过分层检查保障代码质量、计算精度与性能不劣化：

#### 代码规范与快速自检 (CPU)
- **快速冒烟自检**：`mini-infra-smoke`（仅需 2 秒，CPU 快速验证模型结构与因果掩码）。
- **单元测试**：`pytest -m "not gpu"`（覆盖算子实现、张量形状与判定逻辑）。
- **代码规范检查**：`ruff check . && ruff format --check .`。

#### 端到端精度对齐测试 (GPU)
- **KV Cache 前后数值对齐**：
  ```bash
  mini-infra-compare-cache \
    --prompt "请用一句话介绍 KV Cache。" \
    --max-new-tokens 32
  # 或作为 pytest GPU 用例：pytest tests/gpu/test_e2e_compare_cache.py
  ```
  在 float32 精度下逐步比对同一手写模型在无 Cache 与有 Cache 两种执行路径下的 Logits 最大绝对误差与生成 Token，确保引入 KV Cache 之后数值完全等价。

- **与 Transformers 对照测试**：
  ```bash
  mini-infra-compare \
    --prompt "请用一句话介绍 KV Cache。" \
    --max-new-tokens 32
  # 或作为 pytest GPU 用例：pytest tests/gpu/test_e2e_transformers_parity.py
  ```
  对照程序会验证分词一致性，并在相同 greedy decoding 条件下逐步比较手写模型与 Hugging Face 参考实现的 Token ID、Logits 最大绝对误差与生成文本。

#### 零性能倒退守护（Zero-Regression Guard）
```bash
mini-infra-bench-regression    # 轻量级（约 1 分钟），TTFT / TPOT 中位数劣化 >5% 自动拦截报警
```
- **本地保护**：`pip install -e ".[dev]" && pre-commit install --hook-type pre-commit --hook-type pre-push`，commit 运行 ruff，push 前运行回归守护。
- **CI 自动化**：PR 会在 GPU runner 上自动触发 [pr-perf-regression.yml](.github/workflows/pr-perf-regression.yml)。

#### 独立全量基准测量（不污染发版历史）
在日常开发调优阶段，若想测量当前实现的完整全矩阵性能（TTFT、TPOT P50/P95、吞吐、MBU/MFU 硬件利用率与显存拆解），但**不想沉淀到 `benchmarks/result-history/` 污染正式发版的时间线图表**，可通过 `--output` 指定输出到 `tmp/` 临时文件（已被 `.gitignore` 忽略，不会上传到 GitHub），并使用 `mini-infra-bench-plot-compare` 绘制跨模式对比图：
```bash
# 1. 测量无 cache 模式结果
mini-infra-bench --tag dev --mode no-cache --output tmp/test_nocache.json

# 2. 测量有 KV Cache 模式结果
mini-infra-bench --tag dev --mode kv-cache --output tmp/test_kvcache.json

# 3. 对比两个 JSON 结果：终端打印对比表，并绘制 6 维度对比图（Decode 延迟、吞吐、加速比、TTFT、显存占用、端到端延迟）
mini-infra-bench-plot-compare tmp/test_nocache.json tmp/test_kvcache.json -o tmp/compare.png

# 快速试跑对比（缩小测试矩阵）
mini-infra-bench --tag dev --mode no-cache --prompt-lens 32 128 --output-lens 32 --repeats 3 --output tmp/test_nocache.json
mini-infra-bench --tag dev --mode kv-cache --prompt-lens 32 128 --output-lens 32 --repeats 3 --output tmp/test_kvcache.json
mini-infra-bench-plot-compare tmp/test_nocache.json tmp/test_kvcache.json -o tmp/compare.png
```

---

### 特性发布与 Minor Release 标准流程（Feature Deliveries & Tags）

每当完成一个核心特性演进（如 KV Cache、FlashAttention 等，对应 `v0.2.0`、`v0.3.0` 等 Minor Release）时，需按以下标准发布流程（SOP）执行基准测试、跨版本演进图生成与版本固化：

#### 步骤 1：质量与精度对齐自检
在准备发版前，确认代码规范与模型精度无误：
```bash
# 1. 语法规范与排版检查
ruff check . && ruff format --check .

# 2. CPU 快速自检与判定规则单测
pytest -m "not gpu"

# 3. GPU 端到端精度对齐（验证手写实现与 Hugging Face 绝对一致，bfloat16 达到 0 误差）
pytest -m gpu
```

#### 步骤 2：更新版本号与文档
1. 修改 `pyproject.toml` 中的版本号：
   ```toml
   [project]
   version = "0.2.0"  # 升级至目标版本号
   ```
2. 更新 `README.md` 中的 **特性全景 (Features)** 状态（勾选已完成特性）。

#### 步骤 3：全量基准测量与演进图表生成
在固定的 GPU 测试机上运行全量基准，沉淀该版本的性能数据并绘制时间线图：
```bash
# 安装绘图与基准依赖
pip install -e ".[bench]"

# 1. 运行全矩阵测试并记录新版本特性（以 KV Cache 为例，结果落盘至 benchmarks/result-history/）
mini-infra-bench --tag "v0.2.0" --release-feature "KV cache" --profile-ops --update-golden

# 2. 读取所有历史版本数据，重新生成多版本演化表与对比图
mini-infra-bench-plot-history

# 3. 将新性能固化为后续日常开发的零退化底线（更新 benchmarks/data/reference.json）
mini-infra-bench-regression --update-reference
```

> **注意（数据流与绘图约定）**：
> * **`mini-infra-bench-plot-history` 只消费 `mini-infra-bench`（`run.py`）的输出**：只有通过 `mini-infra-bench` 跑出的全矩阵数据才会归档至 `benchmarks/result-history/` 并被自动读入绘制成趋势图与演进时间线；
> * **`mini-infra-bench-regression` 仅用于 CI 拦截守门**：其生成的 `candidate.local.json` 为临时比对文件，不进入版本库亦不参与画图。

> **版本号与特性（Feature）标注约定**：
> * **Git Tag、pyproject.toml 与 `--tag`** 统一采用纯版本号（如 `v0.2.0`），严格遵循 SemVer 与 PEP 440 规范，保证 CI/CD 自动化构建一致性；
> * **`mini-infra-bench` 新增 `--release-feature` 参数**（如 `"KV cache"`），在生成的 JSON 结果中作为 `release_feature` 独立属性存储；`mini-infra-bench-plot-history` 会自动在 Markdown 报表与演进趋势图中将其渲染为 `v0.2.0 (KV cache)`。

#### 步骤 4：提交评测产物与版本代码
将代码修改与自动生成的基准产物一同纳入 Git 提交：
```bash
git add pyproject.toml README.md benchmarks/
git commit -m "release: v0.2.0 - add KV Cache support and update benchmark history"
```

#### 步骤 5：创建 Release Tag 并推送到远程
打上带注解的版本 Tag 并推送到远程仓库：
```bash
git tag -a v0.2.0 -m "Release v0.2.0: KV Cache support"
git push origin main --tags
```

> **CI 自动化机制**：
> 推送 `v*` 格式的 Tag 会在 GitHub Actions 上自动触发 [release-benchmark.yml](.github/workflows/release-benchmark.yml) 工作流：
> * **特性透传与解析**：工作流会自动提取并向 `--release-feature` 透传特性描述（按优先级自动获取：① 手动 `workflow_dispatch` 传参 $\rightarrow$ ② 匹配已有同 Tag 的历史 JSON 归档 $\rightarrow$ ③ 从 Git Annotated Tag 注释中解析，如 `Release v0.2.0: KV cache`）；
> * **云端产物归档**：在云端具备 GPU Runner 时，会自动在 GitHub Actions 归档带有当前 Commit 签名的全矩阵评测数据、Markdown 诊断报告与时间线大图 Artifact。

#### 发布产物清单：
- **`benchmarks/result-history/<commit>_<tag>.json`**：该版本全矩阵测试的原始性能数据（持久化存储）。
- **`benchmarks/data/reference.json`**：当前特性的轻量基准参考数据（作为日常 pre-push 零退化守门底线）。
- **`benchmarks/data/golden_output_tokens.json`**：固定 Prompt 下的 Golden 输出 Token 序列（用于版本发布与日常的正确性 0/1 门禁校验）。
- **`benchmarks/reports/SUMMARY.md`**：记录所有历史版本的详细数据、相对 Baseline 加速比以及深度 Profiling 诊断报告。
- **`benchmarks/reports/evolution_and_scaling.png`（跨版本演化与扩展趋势大图）**：包含版本演进时间线（Decode 延迟下降、吞吐提升、加速比递增曲线）以及不同 Prompt 序列长度（32 $\sim$ 1024+）下的扩展曲线。
- **`benchmarks/reports/profiling_and_optimization.png`（当前版本深度剖析与优化指引大图）**：包含 MBU/MFU 硬件天花板差距、Attention vs MLP 模块耗时归因、算子类型分布透视及显存碎片足迹剖析。

---

## 致谢与参考 (Acknowledgments)

- 本项目实现参考并基于 [SVDX-lab/miniinfra](https://github.com/SVDX-lab/miniinfra/) 《超轻量AI Infra教程—推理篇》。

