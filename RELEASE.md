# mini-infra-qwen3 特性发布流程指南 (Release SOP)

本项目采用 **特性驱动（Feature-Driven）** 的演进方式。每当完成一个核心特性演进（如 KV Cache、Fixed Batching、Continuous Batching 等，对应 `v0.2.0`、`v0.3.0`、`v0.4.0` 等 Minor Release）时，需按本标准发布流程（SOP）执行精度校验、全量基准测试、跨版本演进图生成与版本固化。

---

## 架构说明：两大 Benchmark 引擎分支

在运行基准测试前，需理解系统内部分流的两大 Benchmark 引擎：

1. **`mini-infra bench lockstep`（齐步推理基准，无队列调度器）**：
   * **涵盖范围**：包含单请求串行生成（`batch_size=1`，如 Baseline、KV Cache）以及固定批处理（Fixed Batching）。
   * **命名由来**：所有请求在同一逻辑时刻同步推进，不走三级队列与槽位生命周期管理，统一由 `generate` / `lockstep_loop` 驱动。
2. **`mini-infra bench scheduler`（调度器推理基准，含队列与槽位管理）**：
   * **涵盖范围**：涵盖 Continuous Batching 与 Fixed Batching 在服务化场景下的队列排队、动态准入、EOS 提前退场与 DenseSlotCache 槽位管理。
   * **命名由来**：由调度循环（`run_scheduler`）统一驱动，用于测量请求在动态到达场景下的 TTFT、ITL、Makespan 与并发容量利用率。

---

## 发布标准流程

### 步骤 1：质量与精度对齐自检
在准备发版前，确认代码风格与模型精度完全达标：

```bash
# 1. 语法规范与排版检查
ruff check . && ruff format --check .

# 2. CPU 快速自检与判定规则单测
pytest -m "not gpu"

# 3. GPU 端到端精度对齐（在 float32 精度下验证逐 Token 绝对 0 误差，排除硬件 bfloat16 GEMM 舍入抖动）
pytest -m gpu

# 或依次运行 4 项独立的数值对齐与请求隔离验证（详见 docs/guides/parity-verification.md）
mini-infra verify transformers
mini-infra verify cache
mini-infra verify lockstep-batch
mini-infra verify scheduler-batch
```

---

### 步骤 2：更新版本号与文档
1. 修改 `pyproject.toml` 中的版本号：
   ```toml
   [project]
   version = "0.4.0"  # 升级至目标版本号（例如 Continuous Batching 对应 0.4.0）
   ```
2. 更新 `README.md` 中的 **特性全景 (Features)** 状态（勾选本次已完成的特性）。

---

### 步骤 3：全量基准测量与演进图表生成
在固定的 GPU 测试机（如 GCP L4 实例）上运行基准测试，沉淀该版本的单标杆性能数据并绘制演进图：

```bash
# 安装绘图与基准依赖
pip install -e ".[bench]"

# 1a. 若为齐步推理特性（如 v0.2.0 KV Cache / v0.3.0 Fixed Batching），运行 lockstep case：
mini-infra bench lockstep case \
  --tag "v0.3.0" \
  --release-feature "fixed batching" \
  --batching-method fixed-batching \
  --profile-mode extra-profile

# 1b. 若为队列调度器特性（如 v0.4.0 Continuous Batching），运行 scheduler case（结果同样默认落盘至 benchmarks/result-history/）：
mini-infra bench scheduler case \
  --tag "v0.4.0" \
  --release-feature "continuous batching" \
  --batching-method continuous-batching

# 2. （可选）针对服务化调度器运行专属长短请求对比套件（落盘至 benchmarks/suites/scheduler/）
mini-infra bench scheduler suite \
  --tag "v0.4.0" \
  --suite fixed-vs-continuous-variable-output-length

# 3. 读取所有历史版本单 Case 数据，重新生成多版本演化表与对比图（自动感知是否跨越 lockstep <-> scheduler 边界）
mini-infra bench plot history

# 4. 将新性能固化为后续日常开发的零退化底线（更新 benchmarks/data/reference.json）
mini-infra bench check-regression --update-reference
```

> ⚠️ **数据流与绘图约定**：
> * **`benchmarks/result-history/*.json` 严格遵循单 Case 约束（`require_single_case=True`）**：每个归档文件只包含 1 个标杆 Case，由统一 Pydantic Schema（`LockstepBenchmarkRecord | SchedulerBenchmarkRecord`）校验；多 Case 套件评测结果统一存放在 `benchmarks/suites/` 下。
> * **`mini-infra bench plot history` 具备跨边界感知能力**：当历史记录同时包含 `lockstep` 与 `scheduler` 时，自动仅绘制跨边界共享指标面板（并将 Lockstep 的 `ttft_ms` 与 Scheduler 的 `schedule.ttft_ms_p50` 自动对齐）；
> * **`mini-infra bench check-regression` 仅用于 CI 拦截守门**：其生成的 `candidate.local.json` 为临时比对文件，不进入版本库亦不参与画图。

> 🏷️ **版本号与特性（Feature）标注约定**：
> * **Git Tag、pyproject.toml 与 `--tag`** 统一采用纯版本号（如 `v0.4.0`），严格遵循 SemVer 与 PEP 440 规范，保证 CI/CD 自动化构建一致性；
> * **`mini-infra bench lockstep case` 与 `mini-infra bench scheduler case` 均接收 `--release-feature` 参数**（如 `"continuous batching"` 或 `"KV cache"`），在生成的 JSON 结果中作为 `release_feature` 独立属性存储；`mini-infra bench plot history` 会自动在 Markdown 报表与演进趋势图中将其渲染为 `v0.4.0 (continuous batching)`。

---

### 步骤 4：提交评测产物与版本代码
将代码修改与自动生成的基准产物一同纳入 Git 提交：

```bash
git add pyproject.toml README.md benchmarks/
git commit -m "release: v0.4.0 - add Continuous Batching support and update benchmark history"
```

---

### 步骤 5：创建 Release Tag 并推送到远程
打上带注解的版本 Tag 并推送到远程仓库：

```bash
git tag -a v0.4.0 -m "Release v0.4.0: Continuous Batching support"
git push origin main --tags
```

#### CI 自动化机制
推送 `v*` 格式的 Tag 会在 GitHub Actions 上自动触发 [.github/workflows/release-benchmark.yml](.github/workflows/release-benchmark.yml)：
* **特性透传与解析**：工作流会自动提取并向 `--release-feature` 透传特性描述（按优先级自动获取：① 手动 `workflow_dispatch` 传参 $\rightarrow$ ② 匹配已有同 Tag 的历史 JSON 归档 $\rightarrow$ ③ 从 Git Annotated Tag 注释中解析，如 `Release v0.4.0: Continuous batching`）；
* **参数对齐与客观守门**：CI 同样启用 `--profile-mode extra-profile` 获取完整的硬件剖析指标；但 CI **不带** `--update-golden`，而是严格以版本库中正式提交的 `golden_output_tokens.json` 为基准，独立校验云端推理正确性；
* **云端产物归档**：在云端具备 GPU Runner 时，会自动在 GitHub Actions 归档带有当前 Commit 签名的全矩阵评测数据、Markdown 诊断报告与时间线大图 Artifact。

---

## 发布产物清单与持久化说明

每次发布版本后，版本库中的以下文件将得到更新或沉淀：

* **`benchmarks/result-history/<commit>_<tag>.json`**：该版本单标杆用例的标准评测记录（遵循 `require_single_case=True` 与统一 Pydantic Schema，版本库持久化存储）。
* **`benchmarks/data/reference.json`**：当前特性的轻量基准参考数据（作为日常 pre-push 零退化守门底线）。
* **`benchmarks/data/golden_output_tokens.json`**：固定 Prompt 下的 Golden 输出 Token 序列（用于版本发布与日常的正确性 0/1 门禁校验）。
* **`benchmarks/reports/SUMMARY.md`**：记录所有历史版本的详细数据、相对 Baseline 加速比以及深度 Profiling 诊断报告。
* **`benchmarks/reports/evolution_and_scaling.png`**：跨版本演化趋势大图（基于边界感知 `MetricPlotSpec` 自动绘制各版本延迟、吞吐、显存与槽位利用率演进曲线）。
* **`benchmarks/reports/profiling_and_optimization.png`**：当前版本深度剖析与优化指引大图（包含 MBU/MFU 硬件天花板差距、Attention vs MLP 耗时归因、算子分布及显存碎片剖析）。
