---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/**/*.py"
description: "标识符命名与类型标注纪律：概念唯一名、索引/耗时/布尔后缀、私有前缀、property 与方法一致性、落盘契约名联动改名、命名简短与方法名不重复参数语义、类型标注从严（尽量不用 Any）"
---

# 命名与类型标注纪律专项规则

在 `src/mini_infra/` 下新增或修改任何标识符（变量、参数、字段、方法、属性）时，遵守以下九条纪律；历史存量确需豁免时按 `AGENTS.md` §一.5 标注 `# infra-legacy: <reason>`。

---

## 一、同一概念全仓唯一名

- 同一语义在所有层、所有文件必须同名，**严禁**近义别名混用（如 `max_len` / `max_length` / `padded_length`；`req_idx` / `request_idx` / `idx` 统一为 `request_idx`）；改名时向同文件既有写法收敛，不引入第三种叫法。
- **同名严禁承载两种语义**，判据是追读写点而非看名字像不像（例：`DenseSlotCache.history_mask` 仅含历史 `[capacity, past_len]`，不得与含当前步的整张 `attention_mask` 同名；语义确实不同的两处不可为「统一名字」强行同名）。

## 二、整数索引统一 `_idx` 后缀，前缀必须说明索引空间

- 整数索引写作 `batch_slot_idx` / `block_idx` / `layer_idx` / `row_idx` / `request_idx`；同时涉及两个索引空间时均写全（`completion_step_by_request_idx`、`prompt_len_by_request_idx`）。
- **严禁**裸 `idx`、裸 `index` 及空间不明的缩写（`req_idx` 必须写全为 `request_idx`）；`i`、`j` 仅限 5 行以内的局部循环。
- **两个防反转判据**：
  - **1-based 进度序号用 `_number`**：`enumerate(..., 1)` 产出、从不用于寻址（全仓无 `x[它]`）、只进进度展示字符串的序号用 `_number` 且与总量成对（`case_number` / `total_cases`、`step_number` / `total_steps`），改 `_idx` 会引入 off-by-one。
  - **`_idx` 不得改回 `_index`**：`idx` 是行业标准缩写，展开只增行宽（本仓 `line-length = 120`）不增信息量；纪律在**前缀写清空间**。

## 三、时间量统一 `_ms` 后缀

- 耗时与时间戳字段必须以 `_ms` 结尾且统一为**毫秒**（`prefill_ms`、`admitted_ms`、`arrival_ms`、`clock_ms`、`step_latencies_ms`）。
- **严禁**无单位裸时间名（`elapsed`、`duration`、`latency`、`time_cost`）；确需秒/纳秒量纲时必须在名字中标明单位（`_s`、`_ns`）。

## 四、布尔量统一谓词前缀

- 布尔变量与参数必须以 `is_`、`has_`、`can_`、`should_` 开头（`has_uniform_prompt_len`、`can_admit`、`is_all_finished`），**严禁**用动词原形或名词做布尔名（`sync`、`cache`、`uniform`）。
- **唯一豁免登记**：`use_cache` 保留 HuggingFace / PyTorch 生态惯例名以维持外部对照可比性；新增布尔量不得再以「惯例」免检。

## 五、仅本模块调用者必须私有化

- 仅被本模块内部调用的函数/方法必须加 `_` 前缀（`_validate_past_key_values`、`_compute_batch_metrics`）；加前缀前须先全仓 grep 确认无跨模块调用，并**先从 `__init__.py` 的 re-export / `__all__` 摘除再私有化**。
- 公共 API 类（出现在 `__all__` 中）面向使用者的方法保持公开，不因「当前仅测试调用」而私有化。

## 六、同名 API 严禁方法 / 属性异质

- 同一概念在不同类上暴露时**必须同为 `@property` 或同为方法**（如 `total_bytes` 在各 Cache 类上统一为 `@property`）。
- 无参、无副作用、O(1) 或纯读取状态的取值器一律用 `@property`；需传参、有副作用或开销与调用次数成正比的一律用方法。
- 返回内部容器快照的集合视图属性（如 `requests`、`free_batch_slots`）必须返回不可变 **`tuple`**，不得返回 `list`；取出型方法（如 `pop_batch`）不受此限。

## 七、落盘契约名的改名必须端到端联动

- 进入 **JSON schema、Pydantic 字段、CLI flag / argparse dest、绘图取值 key** 的名字属对外契约，改名时必须先 grep `benchmarks/**/*.json` 与 `src/mini_infra/bench/plotting/`，连同全部读写点与已落盘数据一次改全（见 `AGENTS.md` §一.4）：
  - 有下游读者 / 历史产物 → 读写点与历史 JSON 联动全改（如 `tidy.py` 写入、`history.py` 读取的 `"order_idx"` / `"release_idx"`）；
  - 零读者且零历史产物 → 随内部标识符直接改（如 `compare_lockstep_batch.py` 诊断输出的 `"target_idx"`）。
- CLI 的 `--prompt-lens`（dest `prompt_lens`）与 engine 内部局部变量同名但非同一成员：**严禁全局替换**，逐点修改内部变量。

## 八、命名尽量简短；方法名不重复参数语义

- **宁短勿赘**：满足唯一名（§一）与索引前缀（§二）前提下，不携带上下文可推断的冗余词。
- **方法名不重复参数语义**：正例 `dense_slot_cache.release(batch_slot_idx)`、`free_block_queue.allocate(num_blocks)`；反例 `release_slot(batch_slot_idx)`、`allocate_blocks(num_blocks)`。

## 九、类型标注从严：尽量不用 `Any`

- 能用精确类型时**严禁**用 `typing.Any`（或 `-> dict[str, Any]`）逃避静态检查。`Any` 仅限四类边界：① 反序列化/外部输入（JSON、CLI/配置）；② 无类型第三方库交互；③ 无法用联合类型表达的异构值；④ 点分路径反射工具。
- **类型选择优先级**：精确类型/结构（标量、类、Pydantic、`TypedDict`） → 封闭联合（`T | None`、`A | B`、`Literal`） → 泛型与协议（`TypeVar`、`Generic`、`Protocol`、`Sequence`、`Mapping`） → 顶层黑盒（`object`） → 边界逃生（`Any`）。`@overload`、`TypeGuard`/`TypeIs`、`cast` 属类型收窄与分派手段。
