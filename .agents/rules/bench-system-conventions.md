---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/bench/**/*.py,src/mini_infra/cli/commands/bench/**/*.py,benchmarks/specs/**/*"
description: "评测系统（bench）四阶段生命周期、命名规范、规格/载荷解耦、动词约束与跨引擎对称性不变量"
---

# 基准评测系统（Bench System）架构规范与工程契约

评测体系（`src/mini_infra/bench/`）、评测 CLI（`src/mini_infra/cli/commands/bench/`）与评测规格声明（`benchmarks/specs/`）严格遵守以下架构不变量。

---

## 一、单向四阶段生命周期与后缀规范

评测全流程严格遵循**单向四阶段生命周期**，跨阶段严禁反向依赖，类名后缀与职责严格对齐：

```text
[阶段 1: 规格声明 Specification Plane] (*Spec)
   - 纯声明式意图（YAML / Pydantic 模型）。定义分布规则、系统配置与参数扫描空间。纯 CPU、零设备与 Token 依赖。
   ↓ build / from_spec / expand (跨阶段构建执行实体)
[阶段 2: 评测执行 Execution Plane] (*Suite, *Case, *Plan, Comparison)
   - 载入内存的套件容器与用例实体。持有确定性数值画像，具备唯一身份键与执行步排期。
   ↓ run_case / run_suite / run_sweep
[阶段 3: 测量结果 Measurement Plane] (*Result)
   - 运行时多轮实测聚合对象。归集样本并计算统计分布画像与相对基准比对。
   ↓ build_benchmark_record / serialize
[阶段 4: 持久化契约 Persistence Plane] (*Record)
   - Pydantic v2 模型。声明强类型 JSON 信封结构与字段合法性校验，守卫落盘数据。
```

### 权威类型矩阵

| 阶段 | 规范后缀 / 核心类型 | 核心职责与不变量契约 |
|---|---|---|
| **Spec** | `SuiteSpec`, `CaseSpec`, `WorkloadSpec`, `ResolvedWorkloadSpec`, `ComparisonSpec` | 纯数据规格声明与边界校验；支持参数扫描展开与分布确定性抽样；严禁包含 Token 序列或设备状态。 |
| **Execution** | `BenchmarkSuite` | 套件不可变容器；初始化即固化 `cases: tuple[BenchmarkCase, ...]`，实现原生容器协议（`len`, `iter`, `[]`）；批处理调度的一等公民。 |
| **Execution** | `BenchmarkCase` | 最小单用例执行实体；由正交身份键（`key: CaseKey`）与数值画像（`profile: WorkloadProfile`）唯一确定。 |
| **Execution** | `WorkloadProfile` | 全仓唯一标准流量画像（SSOT）；纯数字元组，与 GPU / 词表解耦，专供 `*Case` 与 `*CaseResult` 共享。 |
| **Execution** | `PlannedRun`, `Comparison` | 执行编排（原子排期步）与对比配置（分组维度、指标基线）。 |
| **Result** | `*CaseResult`, `ComparisonResult` | 实测统计画像（`DistributionStats`）；保留原始 `key: CaseKey` 与 `workload: WorkloadProfile`。 |
| **Record** | `*CaseRecord`, `*BenchmarkRecord`, `AnyBenchmarkRecord` | 持久化 JSON 信封模型；严格版本化与落盘校验。 |

---

## 二、负载与规格三层解耦与按需局部材质化

负载数据在系统中严格按“分布规则 $\to$ 规格展开 $\to$ 数值画像”推进，Token 载荷仅在执行期栈局部按需装配：

```text
WorkloadSpec (规则声明: 分布类型与 seed)
  ↓ resolve_workload_body (确定性抽样求解)
ResolvedWorkloadSpec (确定性纯数字元组声明)
  ↓ to_profile (显式转换)
WorkloadProfile (轻量数值画像 SSOT, 供 Case / Result 共享)
  ↓ to_payload(vocab_size) (执行期现场装配)
tuple[RequestSpec, ...] (函数栈局部变量，即用即弃，严禁堆常驻)
```

1. **唯一真相源（SSOT）**：`WorkloadProfile` 是全仓唯一标准流量画像，彻底切断结果层对规格层的反向依赖。
2. **按需局部材质化（Materialize on Demand）**：`*Suite` 与 `*Case` 严禁持有具体 Token 序列或词表依赖；具体 `token_ids` 仅在底层执行函数（`run_case` / `run_suite`）栈局部调用 `profile.to_payload(vocab_size)` 生成。

---

## 三、套件容器与不可变契约

1. **自包含不可变容器**：
   - 套件容器统一为 `BenchmarkSuite`，构造时必须在 `__post_init__` 中将用例固化为不可变元组 `cases: tuple[BenchmarkCase, ...]`。
   - 实现标准容器协议（`__len__`, `__iter__`, `__getitem__`），禁止运行时动态重生成或通过方法调用生成用例。
2. **执行层统一第一公民**：
   - 批处理驱动（`run_sweep`）以 `Sequence[BenchmarkSuite]` 为唯一目标输入，严禁绕过 Suite 穿透消费 `LoadedSuiteSpec` 或维护对象反查字典。

---

## 四、跨引擎对称性不变量

Scheduler 与 Lockstep 两套执行引擎在评测体系中必须保持严格结构对称：

1. **身份凭证对称**：所有用例必须通过 `@property def key(self) -> CaseKey` 暴露正交特征向量；所有结果必须显式包含 `key: CaseKey` 与 `workload: WorkloadProfile`。
2. **三级执行内核对称**：
   - **单次原子执行**：`run_case_once(model, device, case, ...)` 负责该用例全部请求序列的单次完整调度与推理执行，返回单次执行结果。
   - **多轮测量聚合**：`run_case(model, device, case, warmup, repeats, ...)` 负责执行预热轮次与多次正式测量，归集测量样本并计算统计分布画像，返回 `CaseResult`。
   - **套件编排驱动**：`run_suite(model, device, suite, ...)` 负责解析执行计划（支持按特征 interleave），驱动套件内所有用例的有序执行并汇总结果列表。

---

## 五、动词语义白名单

评测体系内的函数与方法命名严格受限于以下动词前缀，表达明确的生命周期语义：

| 动词前缀 | 语义契约 | 适用场景与范式 |
|---|---|---|
| `expand_*` | 笛卡尔积展开或网格扫描 | 纯函数：`SuiteSpec` 扫描空间展开为 `CaseSpec` 列表。 |
| `resolve_*` | 字符串插值、参数求解或分布确定性抽样 | 纯函数：模板变量解析与分布抽样。 |
| `to_*` | 实例显式转换方法 | 跨层级或领域模型类型转换（如 `to_profile()`, `to_payload()`）。 |
| `from_*` | 类静态工厂方法 | 从源类型、规格声明或外部输入逆向装配当前类实例。 |
| `build_*` | 自由组装函数 | 多对象聚合装配、执行排期生成或持久化 Record 构建。 |
| `extract_*` / `lookup_*` | 属性反射、画像提取或静态元数据查表 | 从执行结果提取画像或查询硬件峰值规格。 |
| `run_*` | 驱动设备执行并产出测量数据 | 驱动单次执行（`run_case_once`）、多轮测量（`run_case`）、套件编排（`run_suite`）或批量扫描（`run_sweep`）。 |
| `format_*` | 报表格式化与文本呈现 | 纯展示：结构化数据转换为终端表格或 Markdown 文本。 |

---

## 六、存储空间与产物隔离契约

1. **输入与声明空间（受控）**：`benchmarks/specs/<engine>/` 专用于版本控制的 YAML 规格声明。
2. **受控产物空间（受控）**：`benchmarks/results/` 用于收敛标准套件与发版基准的 JSON 结果；`benchmarks/golden/` 存放黄金门禁基线。
3. **临时/实验空间（隔离）**：任何带有命令行 `--set` 覆盖或使用 `--spec` 传入外部文件的运行产物，统一写入由 `.gitignore` 保护的临时目录（如 `tmp/results/`），绝不污染受控资产。
