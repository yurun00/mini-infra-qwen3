---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/bench/**/*.py,src/mini_infra/cli/commands/bench/**/*.py,benchmarks/specs/**/*"
description: "评测系统（bench）四阶段生命周期、命名规范、规格/载荷解耦、动词约束与跨引擎对称性不变量"
---

# 基准评测系统（Bench System）架构规范与工程契约

评测体系（`src/mini_infra/bench/`）、评测 CLI（`src/mini_infra/cli/commands/bench/`）与评测规格声明（`benchmarks/specs/`）严格遵守以下架构不变量。

---

## 一、单向四阶段生命周期与权威类型矩阵

评测全流程遵循**单向四阶段生命周期**，跨阶段严禁反向依赖，类名后缀与职责严格对齐：

`[1. Spec 规格声明 (*Spec)]` ──(`from_spec` / `expand_*`)──→ `[2. Execution 评测执行 (*Suite, *Case, *Plan, *Context, Comparison)]` ──(`run_case` / `run_suite` / `run_sweep`)──→ `[3. Result 测量结果 (*Result)]` ──(`build_benchmark_record`)──→ `[4. Record 持久化契约 (*Record)]`

| 阶段 | 规范后缀 / 核心类型 | 核心职责与不变量契约 |
|---|---|---|
| **1. Spec** | `SuiteSpec`, `ReleaseSpec`, `CaseSpec`, `WorkloadSpec`, `ResolvedWorkloadSpec`, `ComparisonSpec`, `LoadedSuiteSpec` | 纯声明式意图（YAML / Pydantic）与边界校验；支持参数扫描展开与分布确定性抽样；纯 CPU，**严禁包含 Token 序列或设备状态**。 |
| **2. Execution** | `BenchmarkSuite` | 套件自包含不可变容器；`__post_init__` 固化 `cases: tuple[BenchmarkCase, ...]`，实现容器协议（`__len__`, `__iter__`, `__getitem__`），禁止运行时动态重生成用例。 |
| **2. Execution** | `BenchmarkCase` | 最小单用例执行实体；由正交身份键（`key: CaseKey`）与数值画像（`profile: WorkloadProfile`）唯一确定。 |
| **2. Execution** | `WorkloadProfile` | 全仓唯一标准流量画像（SSOT）；纯数字元组（`prompt_lens`, `output_budgets`, `arrival_ms`），与 GPU / 词表解耦，供 `*Case` 与 `*CaseResult` 共享，切断结果层对规格层的反向依赖。 |
| **2. Execution** | `PlannedRun`, `Comparison`, `BenchmarkExecutionContext` | 原子排期步（`PlannedRun`）、强类型对比配置（`Comparison`）与共享执行上下文（`BenchmarkExecutionContext`：持有已加载模型、Tokenizer、设备、硬件峰值与正确性缓存）。 |
| **3. Result** | `LockstepCaseResult`, `SchedulerCaseResult`, `ComparisonResult` | 多轮实测统计聚合对象（`DistributionStats`）与相对基线对比结果；保留原始 `key: CaseKey` 与 `workload: WorkloadProfile`。 |
| **4. Record** | `*CaseRecord`, `*BenchmarkRecord`, `AnyBenchmarkRecord` | Pydantic v2 持久化 JSON 信封模型（`bench/result/schema.py`）；严格版本化与判别式联合校验。 |

---

## 二、负载与规格三层解耦与按需局部材质化

`WorkloadSpec`（分布规则与 `seed`）──(`resolve_workload_body`)──→ `ResolvedWorkloadSpec`（确定性纯数字元组声明）──(`to_profile`)──→ `WorkloadProfile`（轻量数值画像 SSOT）──(`to_payload(vocab_size)`)──→ `tuple[Request, ...]`（执行期函数栈局部变量，即用即弃）

- **按需局部材质化（Materialize on Demand）**：`*Suite` 与 `*Case` **严禁持有具体 Token 序列或词表依赖**；具体 `token_ids` 仅在底层执行函数（`run_case_once` / `run_case` / `run_suite`）栈局部调用 `profile.to_payload(vocab_size)` 现场生成，严禁堆常驻。
- **批量驱动契约**：`run_sweep` 以 `Sequence[BenchmarkSuite]` 为唯一目标输入，严禁绕过 `BenchmarkSuite` 穿透消费 `LoadedSuiteSpec` 或维护对象反查字典。

---

## 三、跨引擎对称性不变量

Scheduler 与 Lockstep 两套执行引擎在评测体系中保持严格结构对称：

1. **身份与负载对称**：所有用例通过 `@property def key(self) -> CaseKey` 暴露正交特征向量；所有 `*CaseResult` 显式持有 `key: CaseKey` 与 `workload: WorkloadProfile`。
2. **三级执行内核对称**（`bench/{lockstep,scheduler}/run_{case,suite}.py`）：
   - **单次原子执行 `run_case_once(model, device, case, ...)`**：执行单用例全部请求序列的单次推理/调度，返回 `GenerationResult` / `ScheduleResult`。
   - **多轮测量聚合 `run_case(model, device, case, warmup, repeats, ...)`**：执行预热与多轮测量，聚合样本统计分布并返回 `*CaseResult`。
   - **套件编排驱动 `run_suite(model, device, suite, ...)`**：解析执行计划（支持按 `CaseDimension` interleave 交替执行），驱动套件内全部用例并汇总 `list[*CaseResult]`。

---

## 四、动词语义白名单

| 动词前缀 | 语义契约 | 适用场景与范式 |
|---|---|---|
| `expand_*` | 笛卡尔积展开或网格扫描 | 纯函数：`SuiteSpec` / `ReleaseSpec` 扫描空间展开为 `CaseSpec` 列表 |
| `resolve_*` | 模板插值、参数求解或分布确定性抽样 | 纯函数：模板变量解析与分布抽样（如 `resolve_workload_body`） |
| `to_*` / `from_*` | 实例显式转换 / 类静态工厂方法 | 跨层类型转换（`to_profile`, `to_payload`, `to_dict`）与逆向装配（`from_spec`, `from_runs`, `from_args`） |
| `build_*` | 自由组装函数 | 多对象聚合装配、执行计划生成（`build_scheduler_run_plan`）或持久化构建（`build_benchmark_record`） |
| `extract_*` / `lookup_*` | 属性提取或静态元数据查表 | 从执行结果提取画像（`extract_workload_profile`）或查询硬件峰值（`HardwarePeakSpecs.lookup`） |
| `run_*` | 驱动设备执行并产出测量数据 | `run_case_once`、`run_case`、`run_suite`、`run_sweep` |
| `format_*` | 报表格式化与文本呈现 | 纯展示：结构化结果转换为终端表格或摘要文本 |

---

## 五、存储空间与产物隔离契约

1. **受控声明与基线空间（Git 追踪）**：`benchmarks/specs/{lockstep,scheduler,release}/` 存放 YAML 规格声明；`benchmarks/results/history/{lockstep,scheduler}/` 存放发版历史基准；`benchmarks/golden/` 存放黄金门禁基线；`benchmarks/reports/{lockstep,scheduler}/` 存放生成的图表与报告。
2. **本地与实验产物隔离（`.gitignore` 忽略）**：内置套件与单用例默认输出（`benchmarks/results/suites/`、`benchmarks/case/`）及任何带 `--set` 覆盖或外部 `--spec` 的实验运行产物（自动路由至 `tmp/results/suites/`）均不入库，绝不污染受控资产。
