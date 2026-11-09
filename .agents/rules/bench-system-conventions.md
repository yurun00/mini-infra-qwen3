---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/bench/**/*.py,src/mini_infra/cli/commands/bench/**/*.py,benchmarks/specs/**/*"
description: "评测系统（bench）四阶段生命周期、理想命名规范、Spec 与 Payload 严格解耦、动词收敛与跨引擎对称性纪律"
---

# 基准评测系统（Bench System）架构命名与生命周期规范

评测体系（`src/mini_infra/bench/`）、评测 CLI 子命令（`src/mini_infra/cli/commands/bench/`）与评测规格声明（`benchmarks/specs/`）严格遵守以下理想状态架构命名与生命周期契约。

---

## 一、四阶段生命周期与后缀规范

评测体系从声明到持久化严格遵循**单向四阶段生命周期**。每个阶段的类名后缀严格对齐其语义职责：

```text
[阶段 1: 声明规格 (Specification Plane)]     后缀：*Spec
   - 纯声明式意图（YAML / Pydantic / 纯数据 Dataclass），定义参数分布、系统配置与扫描空间。纯 CPU、零设备与 Token 依赖。
   ↓ build / from_spec (跨阶段构建执行实体)
[阶段 2: 评测执行 (Execution Plane)]         后缀/核心类型：*Suite, *Case, *Payload, *Plan, Comparison
   - 载入内存的套件容器、已完成数值求解并绑定 Token 序列的用例实体，具备唯一身份键与执行步排期。
   ↓ run_case / run_suite
[阶段 3: 测量结果 (Measurement Plane)]       后缀：*Result
   - 运行时实测多轮汇总对象（Dataclass），聚合样本并计算 DistributionStats 统计分布画像。
   ↓ build_benchmark_record / serialize
[阶段 4: 落盘契约 (Persistence Plane)]       后缀：*Record
   - Pydantic v2 模型，声明强类型 JSON 信封与字段校验，守卫持久化数据结构。
```

### 生命周期规范类型权威对照表

| 阶段 | 规范类型名 | 模块归属 | 职责定义与契约要求 |
|---|---|---|---|
| **Spec** | `SuiteSpec` | `bench.spec.suite.schema` | 套件 YAML 根声明模型（Pydantic），描述系统、负载模板、扫描空间与对比基准 |
| **Spec** | `CaseSpec` | `bench.spec.suite.expand` | `SuiteSpec` 经 `sweep` 展开后的**单用例规格**（持有具体的 `ResolvedWorkloadSpec` 与系统配置） |
| **Spec** | `WorkloadSpec` | `bench.spec.common.schema` | YAML 负载分布规则模型（持有 `LadderLength`、`PoissonArrival` 等分布对象与随机种子） |
| **Spec** | `ResolvedWorkloadSpec` | `bench.spec.common.workload` | 抽样求值后的**确定性数值流量画像**（持有 `prompt_lens`、`output_budgets`、`arrival_ms` 等数值元组，无 Token） |
| **Spec** | `ComparisonSpec` | `bench.spec.suite.schema` | YAML 声明的对比指标、分组与基线配置 |
| **Execution** | `BenchmarkSuite[CaseT]` | `bench.common` | 套件通用泛型抽象基类，持有 `LoadedSuiteSpec`，规范 `cases(vocab_size)` 接口与公共元数据访问 |
| **Execution** | `SchedulerSuite` / `LockstepSuite` | `bench.{scheduler,lockstep}.suites` | 继承 `BenchmarkSuite` 的具体套件容器，管理所包含的 `CaseSpec` 列表并产出可执行用例 |
| **Execution** | `LoadedSuiteSpec` | `bench.spec.suite.loader` | 加载入内存的套件规格及展开结果容器（持有 `spec`、`path`、`sha256`、`cases: tuple[CaseSpec, ...]`） |
| **Execution** | `RequestPayload` | `bench.cases` | 已合成并绑定确定性具体 Token ID 序列的请求实体载荷（持有 `tuple[RequestSpec, ...]`） |
| **Execution** | `SchedulerCase` | `bench.cases` | Scheduler 评测最小单用例（持有 `CaseKey`、`RequestPayload`、`Mode`、并发度与缓存配置） |
| **Execution** | `LockstepCase` | `bench.cases` | Lockstep 评测最小单用例（持有 `CaseKey`、`RequestPayload`、`Mode`） |
| **Execution** | `PlannedRun` | `bench.scheduler.plan` | 最小执行步计划（持有目标用例、`phase: RunPhase`、`repeat_idx` 与分组 ID） |
| **Execution** | `Comparison` | `bench.cases` | 经 `Comparison.from_spec(spec)` 装配生成的强类型运行时对比配置，指导多用例结果分组与基准指标计算 |
| **Result** | `SchedulerCaseResult` | `bench.scheduler.types` | Scheduler 多轮测量聚合结果，持有 `key: CaseKey` 与 `DistributionStats` 统计画像 |
| **Result** | `LockstepCaseResult` | `bench.lockstep.types` | Lockstep 多轮测量聚合结果，持有 `key: CaseKey` 与 `DistributionStats` 统计画像 |
| **Result** | `MetricComparison` | `bench.scheduler.types` | 两个用例间基准指标的相对对比结果（Speedup、Gain、Reduction） |
| **Record** | `SchedulerCaseRecord` | `bench.schema` | Scheduler 单用例持久化 Pydantic 模型 |
| **Record** | `LockstepCaseRecord` | `bench.schema` | Lockstep 单用例持久化 Pydantic 模型 |
| **Record** | `SchedulerBenchmarkRecord` | `bench.schema` | Scheduler 完整评测结果 JSON 顶层信封 |
| **Record** | `LockstepBenchmarkRecord` | `bench.schema` | Lockstep 完整评测结果 JSON 顶层信封 |
| **Record** | `AnyBenchmarkRecord` | `bench.schema` | 顶层记录的多态判别式联合（Discriminated Union by `engine`） |

---

## 二、流量、规格与载荷的三层解耦

请求负载在系统中严格按照“分布规则 $\to$ 数值画像 $\to$ Token 载荷”三层单向推进，各层职责独立且单一：

```text
[层级 1: 规则声明]   WorkloadSpec            (持有 Ladder/Poisson 等分布类型与 seed，Pydantic 校验)
                          ↓ resolve_workload_body (根据 seed 与 count 进行确定性抽样)
[层级 2: 数值画像]   ResolvedWorkloadSpec    (持有 prompt_lens / arrival_ms 等纯数字元组，纯 CPU，无 Token)
                          ↓ RequestPayload.from_spec (根据 vocab_size 装配具体 Token 序列)
[层级 3: 数据载荷]   RequestPayload          (持有 RequestSpec(token_ids=[...])，包含具体 Token 整数序列)
```

1. **`WorkloadSpec`（声明校验层）**：
   - 物理位置：`mini_infra.bench.spec.common.schema`。
   - 核心职责：YAML 输入边界守卫，负责反序列化与静态结构校验，承载变量占位符与随机分布定义。
2. **`ResolvedWorkloadSpec`（数值画像层）**：
   - 物理位置：`mini_infra.bench.spec.common.workload`。
   - 核心职责：全仓唯一标准流量画像（SSOT）。变量插值与分布抽样完成后的确定性结果，仅由纯整数/浮点元组构成，与词表大小和 GPU 设备完全解耦。
3. **`RequestPayload`（数据载荷层）**：
   - 物理位置：`mini_infra.bench.cases`。
   - 核心职责：通过类方法 `RequestPayload.from_spec(spec, vocab_size)` 装配具体 `token_ids` 后的确定性载荷容器。可直接投递给推理引擎消费。

---

## 三、动词与函数命名白名单

评测体系内的函数与方法命名严格限制在以下动词语义空间内，使调用点清晰表达其所处生命周期：

| 动词前缀 | 语义契约 | 规范命名正例 |
|---|---|---|
| `expand_*` | 笛卡尔积展开、参数网格扫描展开或多规模展开 | `expand_suite_spec(spec: SuiteSpec) -> list[CaseSpec]`<br>`expand_release_scales(spec: ReleaseSpec, seed: int)` |
| `resolve_*` | 占位符字符串插值、分布数学抽样或字符串收窄为强类型枚举 | `resolve_workload_body(id, body, vars, context)` |
| `build_*` / `*.from_spec()` | 装配具体 Token 序列载荷，构建对比配置或基于算法编排执行计划 | `RequestPayload.from_spec(spec, vocab_size) -> RequestPayload`<br>`Comparison.from_spec(spec) -> Comparison`<br>`build_scheduler_run_plan(cases, warmup, repeats)` |
| `extract_*` / `lookup_*` | 点分路径属性反射查询，或硬件参数查表匹配 | `extract_metric_by_path(obj: Any, path: str) -> float`<br>`lookup_gpu_peak_specs(gpu_name: str) -> HardwarePeakSpecs` |
| `run_*` | 驱动用例或套件在设备上执行，测量耗时并产出 Result | `run_case_once(model, device, case) -> ScheduleResult`<br>`run_case(model, device, case, ...) -> CaseResult`<br>`run_suite(model, device, suite, ...) -> list[CaseResult]`<br>`run_sweep(targets, execute) -> list[SuiteRunSummary]` |
| `format_*` | 纯展示与报表格式化，将结构化数据转为终端文本或 Markdown | `format_case_summary(result) -> str`<br>`format_table(results) -> str`<br>`format_comparison(comp) -> str` |

---

## 四、跨引擎（Lockstep 与 Scheduler）对称性规范

两套推理执行引擎在评测体系中遵守完全一致的数据抽象与调用规范：

1. **`CaseKey` 正交身份键全覆盖**：
   - 所有用例定义（`SchedulerCase`、`LockstepCase`）统一通过 `@property def key(self) -> CaseKey` 暴露正交特征向量（`workload_id`、`mode`、`max_running_requests`、`paged_cache_config`）。
   - 所有多轮聚合测量结果（`SchedulerCaseResult`、`LockstepCaseResult`）统一显式包含 `key: CaseKey` 字段。
2. **套件抽象与分词时机统一**：
   - 套件容器统一继承泛型基类 `BenchmarkSuite[CaseT]`，持有单一真相源 `LoadedSuiteSpec`，消除重复属性代理并规范公共接口。
   - 两套引擎的用例均统一持有 `RequestPayload` 作为输入载荷。
   - 套件生成用例时，统一通过 `suite.cases(vocab_size)` 完成从 `ResolvedWorkloadSpec` 到 `RequestPayload` 的合成绑定。
3. **三级执行内核接口对称**：
   - **用例单次调度执行（Single Case Run）**：`run_case_once(model, device, case)` 负责该用例全部请求序列的单次完整调度与推理执行（处理完整批次直至所有请求完成），返回 `ScheduleResult` 或 `GenerationResult`。
   - **用例多轮重复测量（Multi-Repeat Measurement Loop）**：`run_case(model, device, case, warmup, repeats, ...)` 负责执行预热轮次与多次正式测量，归集测量样本并计算统计分布画像，返回 `CaseResult`。
   - **套件驱动编排（Suite Execution Driver）**：`run_suite(model, device, suite, ...)` 负责解析执行计划（支持按特征 interleave），驱动套件内所有用例的有序执行并汇总结果列表。

---

## 五、评测产物与文件系统目录规范

文件系统路径清晰映射配置、产物与历史基准的权威边界，彻底隔离“规格声明（Specs）”与“执行产物（Results）”：

1. **规格声明目录（Specifications / 输入空间）**：
   - `benchmarks/specs/` 专用于持久化受版本控制的 YAML 规格声明与 IDE JSON Schema。
   - `benchmarks/specs/scheduler/*.yaml`：Scheduler 套件声明。
   - `benchmarks/specs/lockstep/*.yaml`：Lockstep 套件声明。
   - `benchmarks/specs/release/*.yaml`：Release 标准三档负载声明。
2. **评测执行产物空间（Run Results / 输出空间）**：
   - 所有基准测试运行产生的 JSON 结果统一收敛在产物空间下（如 `benchmarks/results/`）：
     - `benchmarks/results/suites/<engine>/<suite_name>/<commit>_<tag>.json`：标准套件的执行产物。
     - `benchmarks/results/case/<commit>_<tag>.json`：单用例调试或探索性执行产物。
     - `benchmarks/results/history/<engine>/<commit>_<tag>.json`：受控版本发版基准（Release Baselines，单 Case / 三规模结构约束）。
   - 带有命令行 `--set` 覆盖或使用 `--spec` 传入外部临时文件的实验性运行，统一写入由 `.gitignore` 保护的 `tmp/results/`，绝不污染受控资产。
3. **可视化报告与黄金参考数据（Reports & Golden Reference）**：
   - `benchmarks/reports/<engine>/`：由绘图命令生成的 Markdown 总结与 PNG 图表。
   - `benchmarks/data/reference.json` 与 `benchmarks/data/golden_output_tokens.json`：固定随机种子的黄金回归门禁基线。
