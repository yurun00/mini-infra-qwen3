---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/**/*.py"
description: "标识符命名纪律：概念唯一名、索引/耗时/布尔后缀、私有前缀、property 与方法一致性、落盘契约名冻结"
---

# 命名纪律专项规则

命名是架构资产：读者靠标识符判断一个值属于哪一层、什么量纲、是否可变。在 `src/mini_infra/` 下新增或修改任何标识符（变量、参数、字段、方法、属性）时，遵守以下七条。

---

## 一、同一概念全仓唯一名

- 同一语义在所有层、所有文件必须同名。**严禁**近义别名混用：`max_len` / `max_length` / `padded_length`；`req_idx` / `request_idx` / `idx`（按 §二 三者收敛为 `request_idx`）。
- 反之，**同名严禁承载两种语义**。

  > **反例（已修复）**：`memory/dense_slot_cache.py` 的 `history_mask` 形状 `[capacity, past_len]`、**只含历史**（活跃位由 `prepare_decode_inputs` 现场 `cat`）；`engine/lockstep_loop.py` 的同名变量是**喂给模型的整张 mask**（含当前步），已改名 `attention_mask`。判据是追读写点，不是看名字像不像：同名异义的两处**必须**改名到语义可辨；语义确实不同的两处**不可**为「统一名字」互相改名。

- 改名时向**同文件既有写法**收敛，不引入第三种叫法。

## 二、整数索引统一 `_idx` 后缀，前缀必须说明索引空间

- 槽位号、块号、层号、行号、请求序号写作 `slot_idx` / `block_idx` / `layer_idx` / `row_idx` / `request_idx`。
- **严禁**裸 `idx`、裸 `index`，以及不说明索引空间的缩写（`req_idx` 的 `req` 是 request 还是 requirement？必须写全 `request_idx`）。`i`、`j` 仅限 5 行以内的局部循环。
- 名字里同时出现两个索引空间时两个都写全：`completion_step_by_request_idx`、`prompt_len_by_request_idx`。
- **1-based 进度序号不是索引**，用 `_number` 且与总量成对出现：`case_number` / `total_cases`、`step_number` / `total_steps`。满足任一即不属本节管辖：① 来自 `enumerate(..., 1)`；② 从不用于寻址（全仓无任何 `x[它]`），只进入进度展示字符串。改成 `_idx` 会制造 off-by-one 陷阱——`cases[case_idx]` 天然差一。现存于 `bench/lockstep/run_suite.py`、`bench/scheduler/run_suite.py`、`cli/commands/bench/{lockstep,scheduler,release}.py`；改名前先确认它是否真的被用来寻址。
- **与 §三 `_ms` 的区别（避免再次反转）**：`_ms` 是单位符号，`idx` 是 `index` 的既定行业缩写；展开成 `_index` 只增加行宽（本仓 `line-length = 120`）而不增加信息量。纪律的价值在**前缀说明了索引到哪个空间**，不在后缀长短——不得以「后缀不够正式」为由把 `_idx` 改回 `_index`。

## 三、时间量统一 `_ms` 后缀

- 耗时与时间戳字段必须以 `_ms` 结尾且统一为**毫秒**：`prefill_ms`、`admitted_ms`、`arrival_ms`、`clock_ms`、`step_latencies_ms`。
- **严禁**无单位的裸时间名（`elapsed`、`duration`、`latency`、`time_cost`）。秒 / 纳秒量纲的值不得进入公共字段；确有必要时必须在名字里写明单位（`_ns`、`_s`）。

## 四、布尔量统一谓词前缀

- 布尔变量与参数必须以下列之一开头：`is_`、`has_`、`can_`、`should_`，使调用点读起来是一个断言：`has_uniform_prompt_len`、`should_sync`、`can_admit`、`is_all_finished`。
- **严禁**动词原形或名词做布尔名（`sync`、`cache`、`uniform`）——调用点 `sync=False` 看不出这是开关还是动作。
- **惯例名豁免登记**：`use_cache` 保持原名不改。它是 HuggingFace / PyTorch 生态的既定惯例名，改动会牺牲与外部对照实现的可比性。除此之外，新增布尔量不得再引用「惯例」作为免检理由。

## 五、仅本模块调用者必须私有化

- 函数 / 方法若**只被定义它的模块内部调用**，必须加 `_` 前缀（`_validate_past_key_values`、`_compute_batch_metrics`）。
- 判据是实际调用点，不是「以后可能有人用」。加前缀前必须先全仓 grep 确认无跨模块调用。
- 先从 `__init__.py` 的 re-export / `__all__` 中摘除，再私有化；顺序不可颠倒。
- 宿主类本身是公共 API（出现在某个 `__all__` 中）时，其**面向使用者的方法**保持公开，不因「当前只有测试调用」而私有化。

## 六、同名 API 严禁方法 / 属性异质

- 同一概念在不同类上暴露时**必须同为 `@property` 或同为方法**。
  > **反例（已修复）**：`DenseSlotCache.total_bytes` 曾是 property 而 `KVCache.total_bytes()` 曾是方法，调用方要靠记住类型来决定加不加括号。
- 无参、无副作用、O(1) 或纯读取既有状态的取值器一律用 `@property`；需要参数、有副作用或代价与调用次数成正比的一律用方法。
- 集合视图属性（返回内部容器快照的 `requests` / `free_slots` 之类）必须返回 **`tuple`**，不得返回 `list`：可变返回值会让调用方误以为改它能影响内部状态。取出型产物（`pop_batch` 等）不受此限。

## 七、落盘契约名冻结

- 凡进入 **JSON 落盘 schema、Pydantic 模型字段、CLI flag / argparse dest、绘图脚本取值的 key** 的名字属对外契约，**不得在命名重构中顺手改名**。完整清单与联动要求见 `AGENTS.md` 第一节第 4 条。
- 冻结判据是**存在下游读者或已落盘的历史产物**，不是「它恰好被写进过某个 dict」。改名前必须先 grep `benchmarks/**/*.json` 与 `src/mini_infra/bench/plotting/`：
  - 有读者 / 有历史产物 → 字符串字面量冻结，即使与按 §二 改名后的局部变量不同名也保持原样，并在赋值处加注释说明。现存例：`bench/plotting/tidy.py` 写入 DataFrame 的 `"order_idx"` / `"release_idx"`，被 `history.py` 的 `sort_values` / `groupby` 读取。
  - 零读者且零历史产物 → 不触发本条，随标识符一起改名。现存例：`verification/parity/compare_lockstep_batch.py` 的 `--output` 诊断 JSON（`"target_idx"`）。
- CLI 的 `--prompt-lens`（argparse dest `prompt_lens`）与 engine 内部的局部变量恰好同名但**不是同一个成员**：此类情况**禁止全局替换**，只能逐点改内部变量。

---

## 附：违规标记与豁免

历史存量确需保留原名时，按 `AGENTS.md` 第一节第 5 条用显式债务标记（`# infra-legacy: <原因>`）声明，不得静默违规。
