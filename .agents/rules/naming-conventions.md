---
trigger: glob
alwaysApply: false
globs: "src/mini_infra/**/*.py"
description: "标识符命名纪律：概念唯一名、索引/耗时/布尔后缀、私有前缀、方法属性一致性与落盘契约名冻结"
---

# 命名纪律专项规则 (Naming Conventions)

命名是本项目的**架构资产**：读者靠标识符判断一个值属于哪一层、是什么量纲、是否可变。
命名不一致会让分层边界失真，也会让跨模块重构与基准数据同步出错。
在 `src/mini_infra/` 下新增或修改任何标识符（变量、参数、字段、方法、属性）时，必须遵守以下七条：

---

## 一、同一概念全仓唯一名 (One Concept, One Name)

- 表达同一个语义的东西，在所有层、所有文件里必须叫同一个名字。**严禁**在不同模块给同一概念起近义别名（`max_len` / `max_length` / `padded_length` 混用，`req_idx` / `request_idx` / `idx` 混用）。
- 反过来，**同一个名字严禁承载两种语义**。若两处同名却是不同概念，必须改名到语义可辨。

  > **反例（已在本仓真实发生，作为判据）**：`history_mask` 与 `attention_mask` 是两种不同东西——
  > `memory/dense_slot_cache.py` 的 `history_mask` 形状 `[capacity, past_len]`，**只含历史**，本步活跃位由 `prepare_decode_inputs` 现场 `cat` 上去；
  > 而 `engine/lockstep_loop.py` 的同名字段是**直接喂给模型的整张 mask**（含当前步）。
  > 判定方法是追读写点，不是看名字像不像。二者不可互相改名统一。

- 改名时向**同文件既有写法**收敛，而不是引入第三种叫法。

## 二、整数索引统一 `_index` 后缀

- 槽位号、块号、层号、行号、请求序号一律写作 `slot_index` / `block_index` / `layer_index` / `row_index` / `request_index`。
- **严禁** `idx` 及其变体（`req_idx`、`batch_idx`、`slot_idx`）。`idx` 不携带任何量纲信息，且无法区分「张量行号」与「请求序号」这两类完全不同的索引。
- 索引名必须说明**索引到哪个空间**。裸 `index`、`i`、`j` 只在 5 行以内的局部循环里可接受。

## 三、时间量统一 `_ms` 后缀

- 所有耗时、时间戳字段必须以 `_ms` 结尾并统一为**毫秒**：`prefill_ms`、`admitted_ms`、`arrival_ms`、`clock_ms`、`step_latencies_ms`。
- **严禁**出现无单位的裸时间名（`elapsed`、`duration`、`latency`、`time_cost`）。秒 / 纳秒量纲的值不得进入公共字段；如确有必要，必须在名字里写明单位（`_ns`、`_s`）。

## 四、布尔量统一谓词前缀

- 布尔变量与参数必须以下列之一开头：`is_`、`has_`、`can_`、`should_`，使调用点读起来是一个断言：`has_uniform_prompt_len`、`should_sync`、`can_admit`、`is_all_finished`。
- **严禁**动词原形或名词做布尔名（`sync`、`cache`、`uniform`）——调用点 `sync=False` 无法看出这是开关还是动作。
- **惯例名豁免登记**：`use_cache` 保持原名不改。它是 HuggingFace / PyTorch 生态的既定惯例名，改动会牺牲与外部对照实现的可比性。除此之外，新增布尔量不得再引用「惯例」作为免检理由。

## 五、仅本模块调用者必须私有化

- 一个函数 / 方法若**只被定义它的模块内部调用**，必须加 `_` 前缀（`_validate_past_key_values`、`_compute_batch_metrics`）。
- 判据是实际调用点，不是「以后可能有人用」。加前缀前必须先全仓 grep 确认无跨模块调用。
- 从 `__init__.py` 的 re-export / `__all__` 中摘除后，才可将其私有化；顺序不可颠倒。
- 宿主类本身是公共 API（出现在某个 `__all__` 中）时，其**面向使用者的方法**保持公开，不因「当前只有测试调用」而私有化。

## 六、同名 API 严禁方法 / 属性异质

- 同一个概念在不同类上暴露时，**必须同为 `@property` 或同为方法**。
  > **反例（本仓曾真实发生，已于命名收敛中修复）**：`DenseSlotCache.total_bytes` 曾是 property 而 `KVCache.total_bytes()` 曾是方法，调用方要靠记住类型来决定加不加括号。
- 无参、无副作用、O(1) 或纯读取既有状态的取值器，一律用 `@property`；需要参数、有副作用或代价与调用次数成正比的，一律用方法。
- 集合视图属性（返回内部容器快照的 `requests` / `free_slots` 之类）必须返回 **`tuple`**，不得返回 `list`：可变返回值会让调用方误以为改它能影响内部状态。取出型产物（`pop_batch` 等）不受此限。

## 七、落盘契约名冻结 (Frozen Schema Names)

- 凡进入 **JSON 落盘 schema、Pydantic 模型字段、CLI flag / argparse dest、绘图脚本取值的 key** 的名字，属于对外契约，**不得在命名重构中顺手改名**。
- 完整清单与联动要求见根目录 `AGENTS.md` 第一节第 4 条（结果指标变更的端到端联动守恒）。改名前必须先 grep `benchmarks/**/*.json` 与 `src/mini_infra/bench/plotting/` 确认零命中。
- 特别地：CLI 的 `--prompt-lens`（argparse dest `prompt_lens`）与 engine 内部的局部变量恰好同名但**不是同一个成员**。此类情况下**禁止全局替换**，只能逐点改内部变量。

---

## 附：违规标记与豁免

历史存量若确需保留原名，按 `AGENTS.md` 第一节第 5 条使用显式债务标记（`# infra-legacy: <原因>`）声明，不得静默违规。
