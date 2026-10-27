# mini-infra-qwen3 特性发布流程指南 (Release SOP)

本项目采用 **特性驱动（Feature-Driven）** 的演进方式。每当完成一个核心特性演进（如 KV Cache、Fixed Batching、FlashAttention 等，对应 `v0.2.0`、`v0.3.0` 等 Minor Release）时，需按本标准发布流程（SOP）执行精度校验、全量基准测试、跨版本演进图生成与版本固化。

---

## 发布标准流程

### 步骤 1：质量与精度对齐自检
在准备发版前，确认代码风格与模型精度完全达标：

```bash
# 1. 语法规范与排版检查
ruff check . && ruff format --check .

# 2. CPU 快速自检与判定规则单测
pytest -m "not gpu"

# 3. GPU 端到端精度对齐（验证手写实现与 Hugging Face 绝对一致，bfloat16 达到 0 误差）
pytest -m gpu
```

---

### 步骤 2：更新版本号与文档
1. 修改 `pyproject.toml` 中的版本号：
   ```toml
   [project]
   version = "0.3.0"  # 升级至目标版本号
   ```
2. 更新 `README.md` 中的 **特性全景 (Features)** 状态（勾选本次已完成的特性）。

---

### 步骤 3：全量基准测量与演进图表生成
在固定的 GPU 测试机（如 GCP L4 实例）上运行全量基准，沉淀该版本的性能数据并绘制时间线图：

```bash
# 安装绘图与基准依赖
pip install -e ".[bench]"

# 1. 运行全矩阵测试并记录新版本特性（以 KV Cache 为例，结果落盘至 benchmarks/result-history/）
mini-infra bench batch case --tag "v0.2.0" --release-feature "KV cache" --profile-mode extra-profile --update-golden

# 2. 读取所有历史版本数据，重新生成多版本演化表与对比图
mini-infra bench plot history

# 3. 将新性能固化为后续日常开发的零退化底线（更新 benchmarks/data/reference.json）
mini-infra bench check-regression --update-reference
```

> ⚠️ **数据流与绘图约定**：
> * **`mini-infra bench plot history` 只消费 `mini-infra bench batch case` 的输出**：只有通过 `mini-infra bench batch case` 跑出的全矩阵数据才会归档至 `benchmarks/result-history/` 并被自动读入绘制成趋势图与演进时间线；
> * **`mini-infra bench check-regression` 仅用于 CI 拦截守门**：其生成的 `candidate.local.json` 为临时比对文件，不进入版本库亦不参与画图。

> 🏷️ **版本号与特性（Feature）标注约定**：
> * **Git Tag、pyproject.toml 与 `--tag`** 统一采用纯版本号（如 `v0.2.0`），严格遵循 SemVer 与 PEP 440 规范，保证 CI/CD 自动化构建一致性；
> * **`mini-infra bench batch case` 接收 `--release-feature` 参数**（如 `"KV cache"`），在生成的 JSON 结果中作为 `release_feature` 独立属性存储；`mini-infra bench plot history` 会自动在 Markdown 报表与演进趋势图中将其渲染为 `v0.2.0 (KV cache)`。

---

### 步骤 4：提交评测产物与版本代码
将代码修改与自动生成的基准产物一同纳入 Git 提交：

```bash
git add pyproject.toml README.md benchmarks/
git commit -m "release: v0.2.0 - add KV Cache support and update benchmark history"
```

---

### 步骤 5：创建 Release Tag 并推送到远程
打上带注解的版本 Tag 并推送到远程仓库：

```bash
git tag -a v0.2.0 -m "Release v0.2.0: KV Cache support"
git push origin main --tags
```

#### CI 自动化机制
推送 `v*` 格式的 Tag 会在 GitHub Actions 上自动触发 [.github/workflows/release-benchmark.yml](.github/workflows/release-benchmark.yml)：
* **特性透传与解析**：工作流会自动提取并向 `--release-feature` 透传特性描述（按优先级自动获取：① 手动 `workflow_dispatch` 传参 $\rightarrow$ ② 匹配已有同 Tag 的历史 JSON 归档 $\rightarrow$ ③ 从 Git Annotated Tag 注释中解析，如 `Release v0.2.0: KV cache`）；
* **参数对齐与客观守门**：CI 同样启用 `--profile-mode extra-profile` 获取完整的硬件剖析指标；但 CI **不带** `--update-golden`，而是严格以版本库中正式提交的 `golden_output_tokens.json` 为基准，独立校验云端推理正确性；
* **云端产物归档**：在云端具备 GPU Runner 时，会自动在 GitHub Actions 归档带有当前 Commit 签名的全矩阵评测数据、Markdown 诊断报告与时间线大图 Artifact。

---

## 发布产物清单与持久化说明

每次发布版本后，版本库中的以下文件将得到更新或沉淀：

* **`benchmarks/result-history/<commit>_<tag>.json`**：该版本全矩阵测试的原始性能数据（版本库持久化存储）。
* **`benchmarks/data/reference.json`**：当前特性的轻量基准参考数据（作为日常 pre-push 零退化守门底线）。
* **`benchmarks/data/golden_output_tokens.json`**：固定 Prompt 下的 Golden 输出 Token 序列（用于版本发布与日常的正确性 0/1 门禁校验）。
* **`benchmarks/reports/SUMMARY.md`**：记录所有历史版本的详细数据、相对 Baseline 加速比以及深度 Profiling 诊断报告。
* **`benchmarks/reports/evolution_and_scaling.png`**：跨版本演化与扩展趋势大图（包含 Decode 延迟下降、吞吐提升、加速比曲线及长 Prompt 扩展规律）。
* **`benchmarks/reports/profiling_and_optimization.png`**：当前版本深度剖析与优化指引大图（包含 MBU/MFU 硬件天花板差距、Attention vs MLP 耗时归因、算子分布及显存碎片剖析）。
