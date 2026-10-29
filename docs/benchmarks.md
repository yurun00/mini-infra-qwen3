# 基准测试与性能评测指南 (Benchmarking Guide)

本指南系统汇总了 [`src/mini_infra/bench/`](../src/mini_infra/bench/) 模块提供的全部基准测试、套件压测、图表绘制与性能回归守门指令，供日常性能调优与版本发版参考。

---

## 一、Benchmark 命令全景速查

| 一级分类 | 子命令 | 核心执行函数 / 模块 | 核心用途 | 默认输出路径 |
| :--- | :--- | :--- | :--- | :--- |
| **Lockstep 齐步** | `mini-infra bench lockstep case` | `run_case.py` | 运行单用例无调度器推理（单请求 Baseline/KV Cache 或固定批处理） | `benchmarks/result-history/` 或 `--output` |
| | `mini-infra bench lockstep suite` | `run_suite.py` | 运行预设齐步批处理对比套件（吞吐扫描、Padding 损耗、槽位浪费） | `benchmarks/suites/<suite>/` |
| **Scheduler 调度器** | `mini-infra bench scheduler case` | `run_case.py` | 运行单用例动态调度服务化推理（Fixed vs Continuous，支持并发与错峰到达） | 控制台输出或 `--output` |
| | `mini-infra bench scheduler suite` | `run_suite.py` | 运行预设调度器套件（长短请求混跑、流式到达、Prefill 干扰） | `benchmarks/suites/scheduler/<name>/` |
| **Plot 绘图** | `mini-infra bench plot history` | `plotting/history.py` | 读取所有历史发版数据，生成跨版本演进时间线图与深度 Profiling 报告 | `benchmarks/reports/` |
| | `mini-infra bench plot compare` | `plotting/compare.py` | 对比任意两个独立 JSON 实验结果，绘制柱状对比图与 Markdown 差异表 | `./benchmark_comparison.png` |
| **Regression 守门** | `mini-infra bench check-regression` | `regression.py` | 1 分钟轻量级冒烟比对，拦截 TTFT/TPOT 劣化超过 5% 的代码变更 | `benchmarks/candidate.local.json` |

---

## 二、Lockstep 齐步基准测试 (`bench lockstep`)

面向**无队列调度器**的推理底座（基于 `generate` / `lockstep_loop`），覆盖单请求与整批同步前向（Fixed Batching）。

### 1. 单用例测试 (`lockstep case`)
用于快速测量某个特定 Prompt 长度、批大小或预算配置下的精确延迟与显存。参数遵循 NumPy 式广播规则。

#### 常用命令样例
```bash
# 1. 基础单请求测试（batch=1, prompt=128, max_new_tokens=64，默认启用 KV Cache）
mini-infra bench lockstep case \
  --tag dev

# 2. 对比无 KV Cache（Baseline）模式
mini-infra bench lockstep case \
  --tag dev \
  --no-kv-cache

# 3. 固定批处理等长压测（batch=4, prompt=128, max_new_tokens=32）
mini-infra bench lockstep case \
  --tag dev \
  --batch-size 4 \
  --prompt-lens 128 \
  --max-new-tokens 32 \
  --batching-method fixed-batching

# 4. 变长输入测试（自动根据 4 个 Prompt 长度推断 batch_size=4）
mini-infra bench lockstep case \
  --tag dev \
  --prompt-lens 16 32 64 128 \
  --max-new-tokens 32 \
  --batching-method fixed-batching

# 5. 开启硬件级 Profiling（采集 MBU、MFU 及 CUDA 算子耗时归因）
mini-infra bench lockstep case \
  --tag profile \
  --profile-mode extra-profile \
  --output tmp/profile.json
```

---

### 2. 预设实验套件 (`lockstep suite`)
通过系统扫描关键超参数，探索硬件与批处理策略的扩展边界。

#### 注册套件清单
1. **`serial-vs-fixed-batch-throughput`**：扫描 Batch Size 1, 2, 4, 8, 16，对比单请求串行与固定批处理的吞吐与显存增长曲线。
2. **`uniform-vs-variable-prompt-padding`**：固定 Batch Size 4，对比等长 Prompt 与阶梯变长 Prompt（16, 32, 64, 128），测量 Padding 有效率与 Prefill 算力浪费。
3. **`uniform-vs-variable-budget-slots`**：固定 Batch Size 4，对比等长生成预算与阶梯变长预算（8, 16, 32, 64），测量短请求提前退出导致的槽位闲置浪费。

#### 常用命令样例
```bash
# 运行固定批处理吞吐扩展套件
mini-infra bench lockstep suite \
  --tag dev \
  --suite serial-vs-fixed-batch-throughput

# 运行变长 Prompt 输入的 Padding 损耗套件
mini-infra bench lockstep suite \
  --tag dev \
  --suite uniform-vs-variable-prompt-padding

# 运行变长输出预算的槽位利用率套件
mini-infra bench lockstep suite \
  --tag dev \
  --suite uniform-vs-variable-budget-slots
```

---

## 三、Scheduler 调度器基准测试 (`bench scheduler`)

面向**服务化与队列调度**架构（基于 `run_scheduler`），具备三级队列、动态准入、EOS 退出与 DenseSlotCache 槽位管理。

### 1. 单用例服务化测试 (`scheduler case`)
模拟带有并发上限（Concurrency Limit）与请求到达时序的服务端负载。

#### 常用命令样例
```bash
# 1. 连续批处理（Continuous Batching）标准单用例（并发上限 4，请求总数 8）
mini-infra bench scheduler case \
  --tag dev \
  --batching-method continuous-batching \
  --max-running-requests 4 \
  --request-count 8 \
  --prompt-length 128 \
  --output-budget 32

# 2. 相同负载下的固定批处理（Fixed Batching）波次对照组
mini-infra bench scheduler case \
  --tag dev \
  --batching-method fixed-batching \
  --max-running-requests 4 \
  --request-count 8 \
  --prompt-length 128 \
  --output-budget 32

# 3. 错峰到达测试（模拟每间隔 20ms 到达一个新请求）
mini-infra bench scheduler case \
  --tag dev \
  --batching-method continuous-batching \
  --arrival-interval-ms 20.0 \
  --max-running-requests 4 \
  --request-count 12
```

---

### 2. 预设调度器套件 (`scheduler suite`)
评估连续批处理在复杂线上工况相对固定批处理的加速比与调度优势。

#### 注册套件清单
1. **`fixed-vs-continuous-variable-output-length`**：长短请求混跑（输出长度极不均衡）。连续批处理在短请求退出后立即释放槽位并接纳新请求，展现最大吞吐优势。
2. **`fixed-vs-continuous-uniform-output-length`**：所有请求输出等长基准，作为无长尾效应时的底线对比。
3. **`fixed-vs-continuous-staggered-arrivals`**：请求按时间阶梯陆续到达，考察连续批处理的即到即调度能力 vs 固定批处理的攒批排队延迟。
4. **`fixed-vs-continuous-prefill-interference`**：运行中 Decode 槽位受新入场请求 Prefill 插入时的 ITL 抖动与打断开销。

#### 常用命令样例
```bash
# 1. 运行长短请求混跑对比套件（推荐：最能体现 Continuous Batching 优势的场景）
mini-infra bench scheduler suite \
  --tag v0.4.0 \
  --suite fixed-vs-continuous-variable-output-length

# 2. 运行流式错峰到达对比套件
mini-infra bench scheduler suite \
  --tag v0.4.0 \
  --suite fixed-vs-continuous-staggered-arrivals

# 3. 预览执行计划（无需 GPU，Dry-Run 模式毫秒级验证调度计划）
mini-infra bench scheduler suite \
  --tag test \
  --suite fixed-vs-continuous-variable-output-length \
  --dry-run
```

> [!TIP]
> **热漂移规避（Thermal Drift Mitigation）**：
> `bench scheduler suite` 默认启用了交替执行机制（Run Alternation，轮流执行 Fixed $\leftrightarrow$ Continuous），避免连续基准测试中 GPU 升温与功耗降频造成的不公平偏差。

---

## 四、性能图表绘制与对比分析 (`bench plot`)

### 1. 跨版本演化全景图 (`plot history`)
读取 `benchmarks/result-history/*.json` 下积累的所有历史版本数据，重新绘制时间线演进大图与 Markdown 汇总报表。

```bash
# 读取历史数据并生成汇总报表与两大演进图
mini-infra bench plot history

# 产出文件：
# 1. benchmarks/reports/SUMMARY.md (Markdown 全景汇总表)
# 2. benchmarks/reports/evolution_and_scaling.png (跨版本加速比与长度扩展曲线)
# 3. benchmarks/reports/profiling_and_optimization.png (当前最新版本的 MBU/MFU 与算子剖析)
```

---

### 2. 任意两版本独立对比图 (`plot compare`)
日常调优时，直接对比任意两个独立的测试结果 JSON（无需沉淀到正式历史目录）。

```bash
# 对比当前实验结果与 Baseline
mini-infra bench plot compare \
  benchmarks/result-history/a497386_baseline.json \
  tmp/profile.json \
  --output-figure tmp/comparison.png \
  --output-markdown tmp/comparison.md \
  --title "KV Cache vs Baseline 性能对比"
```

---

## 五、轻量性能回归守门 (`bench check-regression`)

作为 CI 门禁与 Pre-push Hook 的核心守门工具，耗时约 1 分钟，现场运行轻量 Benchmark 并与 `benchmarks/data/reference.json` 比对。

```bash
# 1. 日常快速守护检查（TTFT 或 TPOT 劣化超过 5% 时报错退出）
mini-infra bench check-regression

# 2. 严格硬件匹配（非同一型号 GPU 时直接报错拦截，用于云端 CI）
mini-infra bench check-regression --strict-hardware

# 3. 新特性发布时固化当前性能底线（更新 benchmarks/data/reference.json）
mini-infra bench check-regression --update-reference
```

---

## 六、测试数据产物与输出目录约定

为保持工作区整洁，所有 Benchmark 命令的输出落盘建议遵循以下规范：

1. **`benchmarks/result-history/`**：
   - 仅存放发版时固化的正式基准文件（格式为 `<commit>_<tag>.json`），由 `mini-infra bench plot history` 消费。
2. **`benchmarks/suites/`**：
   - 各预设套件（Suite）执行结果自动落盘目录（如 `benchmarks/suites/serial-vs-fixed-batch-throughput/`）。
3. **`tmp/`（推荐临时输出目录）**：
   - 日常单次测试、调优对比或 Profiling 产物建议输出到 `tmp/`（如 `--output tmp/test.json`）。该目录已纳入 `.gitignore`，不会污染版本控制。
