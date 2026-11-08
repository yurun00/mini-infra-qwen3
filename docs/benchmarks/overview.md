# 基准测试与性能评测指南 (Benchmarking Guide)

本指南系统汇总了 [`src/mini_infra/bench/`](../../src/mini_infra/bench/) 模块提供的全部基准测试、套件压测、图表绘制与性能回归守门指令，供日常性能调优与版本发版参考。

---

## 一、Benchmark 命令全景速查

| 一级分类 | 子命令 | 核心执行函数 / 模块 | 核心用途 | 默认输出路径 |
| :--- | :--- | :--- | :--- | :--- |
| **Lockstep 齐步** | `mini-infra bench lockstep case` | `lockstep/run_case.py` | 运行单用例无调度器推理（单请求 Baseline/KV Cache 或固定批处理） | `benchmarks/result-history/lockstep/` 或 `--output` |
| | `mini-infra bench lockstep suite` | `lockstep/run_suite.py` | 运行预设齐步批处理对比套件（吞吐扫描、Padding 损耗、batch slot 浪费） | `benchmarks/suites/lockstep/<suite>/` |
| **scheduler 调度器** | `mini-infra bench scheduler case` | `scheduler/run_case.py` | 运行单用例动态调度服务化推理（Fixed vs Continuous、Dense vs Paged，支持并发与错峰到达） | 控制台输出或 `--output` |
| | `mini-infra bench scheduler suite` | `scheduler/run_suite.py` | 运行预设调度器套件（长短请求混跑、流式到达、Prefill 干扰、Paged 显存/块大小对比） | `benchmarks/suites/scheduler/<name>/` |
| **Suite 检视** | `mini-infra bench suites list \| show \| schema` | `spec/` | 浏览内置 suite、查看合并后的 YAML 与展开用例、导出 JSON Schema（无需 GPU） | 标准输出 |
| **Release 固化** | `mini-infra bench release` | `cli/commands/bench/release.py` | 按标准多规模负载（small/medium/large）一键生成版本基准、报告与回归基线 | `benchmarks/result-history/{lockstep,scheduler}/` |
| **Plot 绘图** | `mini-infra bench plot history` | `plotting/history.py` | 读取所有历史发版数据，生成跨版本演进时间线图与深度 Profiling 报告 | `benchmarks/reports/{lockstep,scheduler}/` |
| | `mini-infra bench plot compare` | `plotting/compare.py` | 对比任意两个独立 JSON 实验结果，绘制柱状对比图与 Markdown 差异表 | `./benchmark_comparison.png` |
| **Regression 守门** | `mini-infra bench check-regression` | `check_regression.py` | 1 分钟轻量级冒烟比对，拦截 TTFT/TPOT 劣化超过 5% 的代码变更 | `benchmarks/candidate.local.json` |

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
3. **`uniform-vs-variable-budget-slots`**：固定 Batch Size 4，对比等长生成预算与阶梯变长预算（8, 16, 32, 64），测量短请求提前退出导致的 batch slot 闲置浪费。

#### 常用命令样例
```bash
# 预览用例列表（无需 GPU，Dry-Run）
mini-infra bench lockstep suite \
  --tag dev \
  --suite serial-vs-fixed-batch-throughput \
  --dry-run

# 运行固定批处理吞吐扩展套件
mini-infra bench lockstep suite \
  --tag dev \
  --suite serial-vs-fixed-batch-throughput

# 运行变长 Prompt 输入的 Padding 损耗套件
mini-infra bench lockstep suite \
  --tag dev \
  --suite uniform-vs-variable-prompt-padding

# 运行变长输出预算的 batch slot 利用率套件
mini-infra bench lockstep suite \
  --tag dev \
  --suite uniform-vs-variable-budget-slots
```

> [!NOTE]
> 套件的 `warmup` / `repeats` 取自 YAML 的 `execution`，命令行 `--warmup` / `--repeats` 仅在显式传入时覆盖。

---

## 三、scheduler 调度器基准测试 (`bench scheduler`)

面向**服务化与队列调度**架构（基于 `run_scheduler`），具备三级队列、动态准入、EOS 退出、`DenseSlotCache`（batch slot 管理）与 `PagedKVCachePool`（physical block pool 管理）。

### 1. 单用例服务化测试 (`scheduler case`)
模拟带有并发上限（Concurrency Limit）、请求到达时序与不同 KV Cache 存储方案的服务端负载。

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

# 4. Paged KV Cache 单用例测试（指定 physical block 大小为 32，块池自动按负载推导）
mini-infra bench scheduler case \
  --tag dev \
  --caching-method paged-kv-cache \
  --batching-method continuous-batching \
  --block-size 32 \
  --max-running-requests 4 \
  --request-count 8 \
  --prompt-length 128 \
  --output-budget 32
```

---

### 2. 预设调度器套件 (`scheduler suite`)
评估连续批处理在复杂线上工况相对固定批处理的加速比，以及 Paged KV Cache 相对 Dense 路径在显存占用与 KV 访问上的优化收益。

#### 注册套件清单

**Fixed vs Continuous 对比套件（默认交替维度 `interleave=(batching_method,)`）**：
1. **`fixed-vs-continuous-variable-output-length`**：长短请求混跑（输出长度极不均衡）。连续批处理在短请求退出后立即释放 batch slot 并接纳新请求，展现最大吞吐优势。
2. **`fixed-vs-continuous-uniform-output-length`**：所有请求输出等长基准，作为无长尾效应时的底线对比。
3. **`fixed-vs-continuous-staggered-arrivals`**：请求按时间阶梯陆续到达，考察连续批处理的即到即调度能力 vs 固定批处理的攒批排队延迟。
4. **`fixed-vs-continuous-prefill-interference`**：运行中 Decode batch slot 受新入场请求 Prefill 插入时的 ITL 抖动与打断开销。

**Dense vs Paged 对比套件（默认交替维度 `interleave=(caching_method, paged_cache_config)`）**：
5. **`dense-vs-paged-long-short-stress`**：长短 Prompt 混杂受控压力测试（long=2048），对比 Dense batch slot 与 Paged 物理块池的峰值显存、KV 读取数与调度延迟。
6. **`dense-vs-paged-length-ratio`**：长短 Prompt 长度比扫描（长 Prompt 取 128 / 512 / 2048 / 3072），评估非均匀 Prompt 长度分布下 Paged KV Cache 的显存与吞吐收益变化趋势。
7. **`dense-vs-paged-block-size`**：Paged 物理块大小扫描（8 / 16 / 32 / 64），对比不同 block size 下的显存开销与调度吞吐，共用同一 Dense 基线。

#### 常用命令样例
```bash
# 1. 运行长短请求混跑对比套件（推荐：最能体现 Continuous Batching 优势的场景）
mini-infra bench scheduler suite \
  --tag v0.4.0 \
  --suite fixed-vs-continuous-variable-output-length

# 2. 运行长短 Prompt 受控压力测试套件（Dense vs Paged 对比）
mini-infra bench scheduler suite \
  --tag v0.5.0 \
  --suite dense-vs-paged-long-short-stress

# 3. 运行 block size 扫描套件
mini-infra bench scheduler suite \
  --tag v0.5.0 \
  --suite dense-vs-paged-block-size

# 4. 自定义交替轮换维度（可选: workload_id, caching_method, batching_method, flags, max_running_requests, paged_cache_config）
mini-infra bench scheduler suite \
  --tag v0.5.0 \
  --suite dense-vs-paged-block-size \
  --interleave caching_method paged_cache_config

# 5. 预览执行计划（无需 GPU，Dry-Run 模式毫秒级验证调度计划）
mini-infra bench scheduler suite \
  --tag test \
  --suite dense-vs-paged-block-size \
  --dry-run
```

> [!TIP]
> **热漂移规避（Thermal Drift Mitigation）与声明式对比**：
> `bench scheduler suite` 内建了基于 `CaseDimension` 的交替执行机制（Run Alternation），在奇偶轮次反转对比维度的执行顺序，并在每次运行间强制调用 `gc.collect()` 与 `torch.cuda.empty_cache()`。套件运行结束后自动计算中位数相对指标（Makespan 加速比、吞吐增益、KV 读取削减、显存削减）并输出列对齐终端摘要。

---

## 四、性能图表绘制与对比分析 (`bench plot`)

### 1. 跨版本演化全景图 (`plot history`)
读取 `benchmarks/result-history/{lockstep,scheduler}/*.json` 下积累的所有历史版本数据，重新绘制时间线演进大图与 Markdown 汇总报表。

```bash
# 读取历史数据并生成汇总报表与两大演进图
mini-infra bench plot history

# 产出文件（按 lockstep 与 scheduler 分域生成）：
# 1. benchmarks/reports/{lockstep,scheduler}/SUMMARY.md (Markdown 全景汇总表)
# 2. benchmarks/reports/{lockstep,scheduler}/evolution_and_scaling.png (跨版本加速比与长度扩展曲线)
# 3. benchmarks/reports/{lockstep,scheduler}/profiling_and_optimization.png (当前最新版本的 MBU/MFU 与算子剖析)
```

---

### 2. 任意两版本独立对比图 (`plot compare`)
日常调优时，直接对比任意两个独立的测试结果 JSON（无需沉淀到正式历史目录）。

```bash
# 对比当前实验结果与 Baseline
mini-infra bench plot compare \
  benchmarks/result-history/lockstep/915b0d9_v0.1.0.json \
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

## 六、声明式 Suite（YAML）：新增与检视

所有 suite 均由 [`benchmarks/specs/{lockstep,scheduler}/<name>.yaml`](../../benchmarks/specs/) 声明；**新增 suite = 新增一个 YAML 文件，无需改 Python**。`name` 必须与文件名一致，`_bases/` 下的文件仅供 `extends` 复用、不可直接运行。

```yaml
# benchmarks/specs/scheduler/fixed-vs-continuous-staggered-arrivals.yaml
# yaml-language-server: $schema=../suite.schema.json
name: fixed-vs-continuous-staggered-arrivals
extends: _bases/fixed_vs_continuous.yaml     # 映射深合并，列表整体替换
workload:
  id: staggered_arrivals                     # → CaseKey.workload_id
  count: 8
  prompt_len: 128
  output_len: {alternating: [8, 32]}
  arrival_ms: {interval: 150}
```

| 配置块 | 作用 |
| :--- | :--- |
| `engine` | `scheduler` 或 `lockstep`，决定 system 允许的键与 case 类型 |
| `execution` | `warmup` / `repeats` / `interleave`（scheduler 交替维度，取值为 `CaseDimension`） |
| `system` | 被测系统：`batching_method[]` × `caching_method[]`（× `paged.block_size`）与 `max_running_requests`（仅 scheduler） |
| `workload` | `id`、`vars`，以及 `groups`（或单组简写 `count / prompt_len / output_len / arrival_ms`）；值中可用 `${var}` |
| `sweep` | `workload`：dict-of-lists 为笛卡尔积，list-of-dicts 为显式场景；`system`：`block_size` / `max_running_requests` |
| `comparison` | scheduler 专用：`group_by` / `baseline` / `metrics`（`speedup` \| `gain` \| `reduction`） |
| `checks` | 具名语义守卫：`free_batch_slot_at_t0`、`min_max_running_requests: N`、`min_output_len: N` |

**长度与到达分布**：`prompt_len` / `output_len` 取标量（constant；lockstep 中保持标量广播语义）、列表（逐请求 explicit，须等于 `count`）、`{alternating: [a, b]}`、`{ladder: {max: M, floor: F}}`（`count` 必须为 4，取 `[M//8, M//4, M//2, M]` 并以 `F` 为下限）；`arrival_ms` 取标量、列表或 `{interval: ms}`。

**展开顺序**即执行顺序：`sweep.workload`（外层）→ `max_running_requests` → `batching_method` → `caching_method` → `block_size`（仅 paged 模式展开，dense 只出现一次）。

### 检视与临时实验
```bash
mini-infra bench suites list                          # 列出全部内置 suite 与用例数
mini-infra bench suites show dense-vs-paged-block-size   # 合并后的 YAML + 展开后的用例列表
mini-infra bench suites schema --output benchmarks/specs/suite.schema.json   # 重新导出 JSON Schema（IDE 补全）

# 运行任意 YAML（临时实验；结果默认写入被 .gitignore 忽略的 tmp/suites/，不污染 benchmarks/）
mini-infra bench scheduler suite --tag dev --spec tmp/my-suite.yaml --dry-run
```

> [!NOTE]
> 结果 JSON 的 `suite` 块记录 `{name, engine, spec_sha256, spec}`：`spec` 是 `extends` 合并后的完整声明，`spec_sha256` 是其内容哈希，可据此复现与比对实验配置。

---

## 七、测试数据产物与输出目录约定

为保持工作区整洁，所有 Benchmark 命令的输出落盘建议遵循以下规范：

1. **`benchmarks/result-history/`**：
   - 仅存放发版时固化的正式基准文件（格式为 `<commit>_<tag>.json`），由 `mini-infra bench plot history` 消费。
2. **`benchmarks/suites/`**：
   - 各预设套件（Suite）执行结果自动落盘目录，按引擎分域：`benchmarks/suites/lockstep/<name>/` 与 `benchmarks/suites/scheduler/<name>/`。
   - 以 `--spec` 运行的临时 YAML 套件默认写入 `tmp/suites/<engine>/<name>/`。
3. **`tmp/`（推荐临时输出目录）**：
   - 日常单次测试、调优对比或 Profiling 产物建议输出到 `tmp/`（如 `--output tmp/test.json`）。该目录已纳入 `.gitignore`，不会污染版本控制。
