---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/**/*.py"
description: "标识符命名与类型标注纪律：概念唯一名、索引/耗时/布尔后缀、私有前缀、property 与方法一致性、落盘契约名联动改名、命名简短与方法名不重复参数语义、类型标注从严（尽量不用 Any）"
---

# 命名与类型标注纪律专项规则

在 `src/mini_infra/` 下新增或修改任何标识符（变量、参数、字段、方法、属性）时，遵守以下九条纪律。

---

## 一、同一概念全仓唯一名

- 同一语义在所有层、所有文件必须同名。**严禁**近义别名混用：`max_len` / `max_length` / `padded_length`；`req_idx` / `request_idx` / `idx`（按 §二 三者收敛为 `request_idx`）。
- 反之，**同名严禁承载两种语义**。

  > **反例（已修复）**：`worker/dense_slot_cache.py` 的 `history_mask` 形状 `[capacity, past_len]`、**只含历史**（活跃位由 `_prepare_decode_inputs` 现场 `cat`）；`engine/lockstep_loop.py` 的同名变量是**喂给模型的整张 mask**（含当前步），已改名 `attention_mask`。判据是追读写点，不是看名字像不像：同名异义的两处**必须**改名到语义可辨；语义确实不同的两处**不可**为「统一名字」互相改名。

- 改名时向**同文件既有写法**收敛，不引入第三种叫法。

## 二、整数索引统一 `_idx` 后缀，前缀必须说明索引空间

- 整数索引写作 `batch_slot_idx` / `block_idx` / `layer_idx` / `row_idx` / `request_idx`；前缀写清索引到哪个空间。名字里同时出现两个空间时两个都写全：`completion_step_by_request_idx`、`prompt_len_by_request_idx`。
- **严禁**裸 `idx`、裸 `index`，以及不说明空间的缩写（`req_idx` 的 `req` 是 request 还是 requirement？写全 `request_idx`）。`i`、`j` 仅限 5 行以内的局部循环。
- 两个防反转判据：
  - **1-based 进度序号不是索引**：`enumerate(..., 1)` 产出、从不用于寻址（全仓无 `x[它]`）、只进进度字符串的序号用 `_number` 且与总量成对（`case_number` / `total_cases`、`step_number` / `total_steps`），改成 `_idx` 会引入 off-by-one。改名前先确认它是否真的被寻址。
  - **`_idx` 不得改回 `_index`**：`idx` 是 `index` 的既定行业缩写，展开只增行宽不增信息量（本仓 `line-length = 120`）；纪律在**前缀写清空间**，不在后缀长短。

## 三、时间量统一 `_ms` 后缀

- 耗时与时间戳字段必须以 `_ms` 结尾且统一为**毫秒**：`prefill_ms`、`admitted_ms`、`arrival_ms`、`clock_ms`、`step_latencies_ms`。
- **严禁**无单位的裸时间名（`elapsed`、`duration`、`latency`、`time_cost`）。秒 / 纳秒量纲的值不得进入公共字段；确有必要时必须在名字里写明单位（`_ns`、`_s`）。

## 四、布尔量统一谓词前缀

- 布尔变量与参数必须以下列之一开头：`is_`、`has_`、`can_`、`should_`，使调用点读起来是一个断言：`has_uniform_prompt_len`、`can_admit`、`is_all_finished`。
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
- 集合视图属性（返回内部容器快照的 `requests` / `free_batch_slots` 之类）必须返回 **`tuple`**，不得返回 `list`：可变返回值会让调用方误以为改它能影响内部状态。取出型产物（`pop_batch` 等）不受此限。

## 七、落盘契约名的改名必须端到端联动

- 进入 **JSON schema、Pydantic 字段、CLI flag / argparse dest、绘图取值 key** 的名字属对外契约，**允许改名，但必须连同全部读写点与已落盘数据一次改全**（详见 `AGENTS.md` 第一节第 4 条）。
- 改名前 grep `benchmarks/**/*.json` 与 `src/mini_infra/bench/plotting/` 列读写点：写入点与读取点（`sort_values` / `groupby`）同步改全，`benchmarks/result-history/*.json` 已落盘字段名一并更新，严禁留旧名或半截改名。
- 有下游读者 → 联动改（例：`bench/plotting/tidy.py` 写入的 `"order_idx"` / `"release_idx"` 被 `history.py` 读取）；零读者零历史产物 → 随标识符改（例：`compare_lockstep_batch.py` 的 `--output` 诊断 JSON `"target_idx"`）。
- CLI 的 `--prompt-lens`（dest `prompt_lens`）与 engine 内部局部变量同名但**不是同一成员**：**禁全局替换**，逐点改内部变量。

## 八、命名尽量简短；方法名不重复参数语义

- **宁短勿赘**：满足唯一名（§一）与索引前缀（§二）前提下，不携带上下文中可推断的冗余信息。
- **方法名不重复参数语义**：参数名已表达语义时，方法名不再同义反复。
  - 正例：`dense_slot_cache.release(batch_slot_idx)`、`free_block_queue.allocate(num_blocks)`
  - 反例：`release_slot(batch_slot_idx)`、`allocate_blocks(num_blocks)`

## 九、类型标注从严：尽量不用 `Any`

- 能用精确类型（`str` / `int` / `Optional[T]` / `Union[...]` / `TypeVar` / `Protocol` / Pydantic 模型）时，**严禁**用 `typing.Any` 抹掉静态检查与 IDE 补全/重构。
- `Any` 仅限以下边界处使用，其余内部一律收窄：
  1. 反序列化 / 外部输入（JSON、外部 API、配置文件）；
  2. 与无类型第三方库交互；
  3. 值类型确实异构、无法用 `Union` 表达；
  4. 按点分路径反射取值的通用工具。
- 类型收窄与选择优先级（信息量由高到低）：
  精确类型/结构（标量/类/Pydantic/`TypedDict`） → 封闭变体（`Union`/`Optional`/`Literal`） → 泛型与协议（`TypeVar`/`Generic`/`Protocol`） → 顶层黑盒（`object`） → 边界逃生（`Any`）。
  - `@overload`、`TypeGuard`/`TypeIs` 与 `cast` 属类型收窄与签名分派手段，非值类型选项。
- **`object` 绝不等于 `Any`**：`object` 是全严格检查的顶层类型，禁止未经验证访问成员，强制下游经 `isinstance` 收窄；`Any` 彻底关闭检查并向外传染。
- `-> dict[str, Any]` 之类返回应优先收窄为具体值类型或强类型模型，不得以 `Any` 逃避声明契约。

---

## 附：违规标记与豁免

历史存量确需保留原名时，按 `AGENTS.md` 第一节第 5 条用显式债务标记（`# infra-legacy: <原因>`）声明，不得静默违规。
