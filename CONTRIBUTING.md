# mini-infra-qwen3 贡献与日常开发指南

**mini-infra-qwen3** 的开发坚持 **特性驱动（Feature-Driven）** 的演进路线，采用最精简、清晰的原生 Python 和 PyTorch 实现。为了保证代码库的高周转与计算精度、性能零倒退，请遵循以下开发与提交流程。

---

## 一、开发环境准备

从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 安装完整锁版本依赖（用于精准复现）
pip install -r requirements-lock.txt

# 以可编辑模式安装全量开发与测试依赖
pip install -e ".[dev,compare,bench]"

# 安装 Git Hooks 守护（提交时自动检查代码规范，推送前自动运行轻量性能回归守门）
pre-commit install --hook-type pre-commit --hook-type pre-push
```

---

## 二、日常开发分层验证工作流

在提交 PR 或合并代码前，代码需依次通过以下分层验证：

### 1. 代码规范与快速自检 (CPU)
*无需真实模型权重或 GPU 算力，数秒内即可完成：*

* **冒烟自检**：
  ```bash
  mini-infra verify smoke
  ```
  使用随机生成的小模型在 CPU 上快速验证模型网络结构、前向传播与因果掩码（Causal Mask）计算正确性。
* **CPU 单元测试**：
  ```bash
  pytest -m "not gpu"
  ```
  覆盖算子实现、张量形状与判定逻辑。
* **代码格式与 Lint 检查**：
  ```bash
  ruff check . && ruff format --check .
  ```

---

### 2. 端到端精度对齐测试 (GPU)
*需要 GPU 环境，下载固定版本 `Qwen/Qwen3-0.6B`（若受限可配置 `export HF_ENDPOINT=https://hf-mirror.com`）。更详尽的指令与参数说明见 **[docs/parity-verification.md](docs/parity-verification.md)**：*

* **Scheduler 批调度（Fixed 与 Continuous）动态对齐**：
  ```bash
  mini-infra verify scheduler-batch \
    --max-new-tokens 8 \
    --max-running-requests 2
  # 或使用 pytest：pytest tests/gpu/parity/test_scheduler_batch_parity.py
  ```
  在 float32 下验证 Continuous Batching、Fixed Batching 与串行单请求生成 100% 逐 Token 严格对齐，覆盖动态槽位绑定、历史 Mask 管理与长短请求动态进出。

* **固定批处理请求隔离与数值对齐**：
  ```bash
  mini-infra verify lockstep-batch \
    --prompt "用一句话解释固定批处理。" \
    --max-new-tokens 8
  # 或使用 pytest：pytest tests/gpu/parity/test_lockstep_batch_parity.py
  ```
  验证不同 Batch Size、排列顺序、同批伴随内容和不同 Padding 长度下各个请求的生成 Token 与 Logits，严格保障请求隔离（Request Isolation）与数值不变性。

* **KV Cache 前后数值对齐**：
  ```bash
  mini-infra verify cache \
    --prompt "请用一句话介绍 KV Cache。" \
    --max-new-tokens 32
  # 或使用 pytest：pytest tests/gpu/parity/test_cache_parity.py
  ```
  在 float32 精度下逐步比对手写模型在无 Cache 与有 Cache 两种执行路径下的 Logits 最大绝对误差与生成 Token，确保引入 KV Cache 之后数值完全等价。

* **与 Hugging Face Transformers 对照测试**：
  ```bash
  mini-infra verify transformers \
    --prompt "请用一句话介绍 KV Cache。" \
    --max-new-tokens 32
  # 或使用 pytest：pytest tests/gpu/parity/test_transformers_parity.py
  ```
  验证分词一致性，并在相同 greedy decoding 条件下逐步比较手写模型与参考实现的 Token ID、Logits 最大绝对误差与生成文本。

---

### 3. 零性能倒退守护（Zero-Regression Guard）

```bash
mini-infra bench check-regression
```
* **机制**：轻量级性能回归守护测试（耗时约 1 分钟）。比对当前实现与 `benchmarks/data/reference.json` 中的底线基准，若 TTFT 或 TPOT 中位数劣化超过 5%，自动以非零状态码退出并告警。
* **本地自动化**：通过 `pre-commit` 注册在 `pre-push` hook 中，防止意外推送劣化性能的代码。
* **CI 自动化**：每个提交的 PR 会在 GPU Runner 上自动触发 [.github/workflows/pr-perf-regression.yml](.github/workflows/pr-perf-regression.yml)。

---

## 三、独立全量基准测量（不污染发版历史）

> 💡 更多 Lockstep、Scheduler 与各预设实验套件的详细命令样例见 **[docs/benchmarks.md](docs/benchmarks.md)**。

在日常性能调优阶段，若想测量当前实现的完整全矩阵性能（TTFT、TPOT P50/P95、吞吐、MBU/MFU 硬件利用率与显存拆解），但**不想沉淀到 `benchmarks/result-history/` 污染正式发版的时间线图表**：

可通过 `--output` 将结果输出到 `tmp/` 临时文件（已被 `.gitignore` 忽略），并使用 `mini-infra bench plot compare` 绘制跨模式对比图：

```bash
# 1. 测量无 cache 模式
mini-infra bench lockstep case --tag dev --no-kv-cache --output tmp/test_nocache.json

# 2. 测量有 KV Cache 模式
mini-infra bench lockstep case --tag dev --kv-cache --output tmp/test_kvcache.json

# 3. 对比两个 JSON 结果：终端打印对比表，并绘制 6 维度对比图（Decode 延迟、吞吐、加速比、TTFT、显存占用、端到端延迟）
mini-infra bench plot compare tmp/test_nocache.json tmp/test_kvcache.json -o tmp/compare.png

# 快速试跑对比（缩小测试矩阵）
mini-infra bench lockstep case --tag dev --no-kv-cache --prompt-lens 32 128 --max-new-tokens 32 --repeats 3 --output tmp/test_nocache.json
mini-infra bench lockstep case --tag dev --kv-cache --prompt-lens 32 128 --max-new-tokens 32 --repeats 3 --output tmp/test_kvcache.json
mini-infra bench plot compare tmp/test_nocache.json tmp/test_kvcache.json -o tmp/compare.png
```

---

## 四、深入指南与延伸

* **算子性能诊断与 Trace 查看**：如需深入分析 CUDA Kernel 耗时或使用 TensorBoard / Perfetto，请参阅 [docs/profiling.md](docs/profiling.md)。
* **版本发布 SOP**：核心特性演进完成后的发版流程（版本号管理、全量基准沉淀、Git Tag 打标），请参阅 [RELEASE.md](RELEASE.md)。
