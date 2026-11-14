# 声明式评测规范（Suite & Release YAML Schema）

本手册定义 [`benchmarks/specs/`](../../benchmarks/specs/) 下声明式评测套件（`SuiteSpec`）与发版标准负载（`ReleaseSpec`）的形式化语法、字段语义、分布展开协议与跨字段约束。对应代码实现位于 [`src/mini_infra/bench/spec/`](../../src/mini_infra/bench/spec/)。

- **新增或修改套件 = 编写 YAML 文件，零 Python 胶水代码**。
- 所有外部输入在加载边界（`SuiteSpec` / `ReleaseSpec` + `WorkloadSpec`）一次性完成 Pydantic 严格校验（`extra="forbid"`），下游执行层直接信任强类型契约。

---

## 一、形式化语法（Extended Backus-Naur Form, EBNF）

以下使用 ISO/IEC 14977 EBNF 风格完整表达 Suite YAML 与 Release YAML 的语法结构（含语法糖简写与标准判别映射形态）：

```ebnf
(* ========================================================================== *)
(* 1. 顶层产生式：Suite YAML 与 Release YAML                                   *)
(* ========================================================================== *)

suite_spec        = "name:" suite_name
                    [ "extends:" path_str ]
                    [ "description:" str ]
                    [ "execution:" execution_spec ]
                    "system:" system_spec
                    "workload:" workload_template
                    [ "sweep:" sweep_spec ]
                    [ "comparison:" comparison_spec ]
                    [ "checks:" "[" check_spec { "," check_spec } "]" ] ;

release_spec      = [ "extends:" path_str ]
                    [ "description:" str ]
                    "default_seed:" int
                    "system:" system_spec
                    [ "execution:" release_exec_spec ]
                    "scales:" "{" "small:" scale_spec ","
                                  "medium:" scale_spec ","
                                  "large:" scale_spec "}"
                    [ "checks:" "[" check_spec { "," check_spec } "]" ] ;

scale_spec        = "{" "seed_salt:" int
                        [ "," "max_running_requests:" pos_int ] ","
                        "workload:" "{" workload_body "}" "}" ;

(* ========================================================================== *)
(* 2. 执行控制 (Execution)                                                     *)
(* ========================================================================== *)

execution_spec    = "{" [ "warmup:" non_neg_int ]
                        [ "repeats:" pos_int ]
                        [ "interleave:" "[" case_dim { "," case_dim } "]" ] "}" ;

release_exec_spec = "{" [ "warmup:" non_neg_int ]
                        [ "repeats:" pos_int ] "}" ;    (* release 禁止 interleave *)

(* ========================================================================== *)
(* 3. 被测系统基线 (System)                                                    *)
(* ========================================================================== *)

system_spec       = "{" "engine_mode:" engine_mode_spec ","
                        "batching_method:" batching_method
                        [ "," "caching_spec:" caching_spec ]
                        [ "," "scheduler_spec:" scheduler_spec ] "}" ;

engine_mode_spec  = "lockstep"
                  | "scheduler"
                  | "{" "kind:" "lockstep" "}"
                  | "{" "kind:" "scheduler" [ "," "max_running_requests:" pos_int ] "}" ;

batching_method   = "none" | "fixed-batching" | "continuous-batching" ;

caching_spec      = caching_method
                  | "{" "kind:" ("none" | "dense-kv-cache") "}"
                  | "{" "kind:" "paged-kv-cache"
                        [ "," "block_size:" pos_int ]
                        [ "," "num_blocks:" (pos_int_ge_2 | "null") ] "}" ;

caching_method    = "none" | "dense-kv-cache" | "paged-kv-cache" ;

scheduler_spec    = "{" [ "batching_policy:" batching_policy_spec ] "}" ;

batching_policy_spec
                  = batching_policy                      (* 仅 prefill-first 可用标量简写 *)
                  | "{" "kind:" "prefill-first" "}"
                  | "{" "kind:" "token-budget" "," "token_budget:" pos_int "}" ;

batching_policy   = "prefill-first" | "token-budget" ;

(* ========================================================================== *)
(* 4. 负载模板与请求组 (Workload & Distributions)                              *)
(*    注：suite 的 workload_template 允许在任意层级的字段值或键中使用 "${var}" *)
(*        占位符，由 expand 阶段对每个 sweep 点完成插值后再按 workload_body 校验 *)
(* ========================================================================== *)

workload_template = "{" "id:" str
                        [ "," "vars:" "{" { var_name ":" yaml_value } "}" ]
                        [ "," "seed:" int ]
                        [ "," "seed_salt:" int ] ","
                        workload_body "}" ;

workload_body     = "groups:" "[" group_spec { "," group_spec } "]"
                  | group_fields ;                       (* 单组简写：直接平铺 group_fields *)

group_spec        = "{" group_fields "}" ;

group_fields      = "count:" pos_int ","
                    "prompt_len:" length_spec ","
                    "output_len:" length_spec
                    [ "," "arrival_ms:" arrival_spec ] ;

(* 4.1 长度分布（支持简写与带 kind 判别字段的完整形式） *)
length_spec       = pos_int                              (* constant 简写 *)
                  | "[" pos_int { "," pos_int } "]"      (* explicit 简写，列表长度 == count *)
                  | "{" "alternating:" "[" pos_int "," pos_int "]" "}"
                  | "{" "ladder:" "{" "max:" pos_int [ "," "floor:" pos_int ] "}" "}"
                  | "{" "log_uniform:" "{" "min:" pos_int "," "max:" pos_int "}" "}"
                  | "{" "kind:" "constant" "," "value:" pos_int "}"
                  | "{" "kind:" "explicit" "," "values:" "[" pos_int { "," pos_int } "]" "}"
                  | "{" "kind:" "alternating" "," "values:" "[" pos_int "," pos_int "]" "}"
                  | "{" "kind:" "ladder" "," "max:" pos_int [ "," "floor:" pos_int ] "}"
                  | "{" "kind:" "log_uniform" "," "min:" pos_int "," "max:" pos_int "}" ;

(* 4.2 到达时间分布（默认 ConstantArrival(value=0.0)，即 0 ms 同时到达） *)
arrival_spec      = non_neg_float                        (* constant 简写 *)
                  | "[" non_neg_float { "," non_neg_float } "]"   (* explicit 简写，列表长度 == count *)
                  | "{" "interval:" non_neg_float "}"
                  | "{" "poisson:" "{" "initial_burst:" pos_int "," "mean_interval_ms:" pos_float "}" "}"
                  | "{" "kind:" "constant" [ "," "value:" non_neg_float ] "}"
                  | "{" "kind:" "explicit" "," "values:" "[" non_neg_float { "," non_neg_float } "]" "}"
                  | "{" "kind:" "interval" "," "interval_ms:" non_neg_float "}"
                  | "{" "kind:" "poisson" "," "initial_burst:" pos_int "," "mean_interval_ms:" pos_float "}" ;

(* ========================================================================== *)
(* 5. 参数扫描 (Sweep)                                                         *)
(* ========================================================================== *)

sweep_spec        = "{" [ "workload:" workload_sweep ]
                        [ "system:" system_sweep ] "}" ;

workload_sweep    = "{" { var_name ":" "[" yaml_value { "," yaml_value } "]" } "}"          (* dict-of-lists: 笛卡尔积 *)
                  | "[" ( "{" var_name ":" yaml_value { "," var_name ":" yaml_value } "}" )+ "]" ; (* list-of-dicts: 显式场景 *)

system_sweep      = "{" [ "max_running_requests:" "[" pos_int { "," pos_int } "]" ]
                        [ "batching_method:" "[" batching_method { "," batching_method } "]" ]
                        [ "caching_spec:" "[" caching_spec { "," caching_spec } "]" ]
                        [ "scheduler_spec:" "[" scheduler_spec { "," scheduler_spec } "]" ] "}" ;

(* ========================================================================== *)
(* 6. 对比规约与语义守卫 (Comparison & Checks)                                 *)
(* ========================================================================== *)

comparison_spec   = "{" "metrics:" "[" metric_spec { "," metric_spec } "]" ","
                        "group_by:" "[" case_dim { "," case_dim } "]" ","
                        "baseline:" "{" case_dim ":" (str | int) { "," case_dim ":" (str | int) } "}" "}" ;

metric_spec       = "{" "path:" str ","
                        "kind:" ("speedup" | "gain" | "reduction")
                        [ "," "name:" str ] "}" ;

check_spec        = "free_batch_slot_at_t0"
                  | "{" "min_max_running_requests:" pos_int "}"
                  | "{" "min_output_len:" pos_int "}" ;

(* ========================================================================== *)
(* 7. 基础词法与枚举终结符 (Terminals)                                         *)
(* ========================================================================== *)

case_dim          = "workload_id" | "caching_method" | "batching_method" | "flags"
                  | "max_running_requests" | "block_size" | "num_blocks"
                  | "batching_policy" | "token_budget" ;
suite_name        = /^[a-z0-9][a-z0-9-]*$/ ;             (* 必须与 YAML 文件名 stem 一致 *)
pos_int           = /^[1-9][0-9]*$/ ;                    (* 严格 int > 0，拒绝 bool 与 float *)
pos_int_ge_2      = /^([2-9]|[1-9][0-9]+)$/ ;            (* 严格 int >= 2 *)
non_neg_int       = /^(0|[1-9][0-9]*)$/ ;
non_neg_float     = (* float >= 0.0，拒绝 bool *) ;
pos_float         = (* float > 0.0，拒绝 bool *) ;
```

---

## 二、各配置块与跨字段语义约束

| 配置块 | 核心职责与跨字段约束 |
| :--- | :--- |
| `name` / `extends` | `name` 必须满足 `^[a-z0-9][a-z0-9-]*$` 且与 YAML 文件名一致；`_bases/` 下的基底文件仅供 `extends` 继承，不可直接作为 suite 运行。`extends` 按**映射深合并、列表整体替换**规则递归合并。 |
| `execution` | `warmup`（默认 `1`）、`repeats`（默认 `3`）、`interleave`（仅 `scheduler` 支持，指定奇偶轮次反转顺序的 `CaseDimension` 列表以规避热漂移）。 |
| `system` | 定义基线运行环境：`engine_mode`（`lockstep` 或 `scheduler`）、`batching_method`、`caching_spec`（默认 `dense-kv-cache`）、`scheduler_spec`（默认 `{batching_policy: prefill-first}`）。当引擎为 `scheduler` 时，必须在 `system.engine_mode.max_running_requests` 或 `sweep.system.max_running_requests` 中提供并发上限。`token-budget` 要求生效的 `batching_method` 全部为 `continuous-batching`，且 `token_budget` 不小于每个生效的 `max_running_requests`。 |
| `workload` | 定义负载模板：`id`（支持 `${var}` 插值生成各场景的 `workload_id`）、`vars`（默认变量表）、以及多组 `groups` 或单组平铺字段（`count` / `prompt_len` / `output_len` / `arrival_ms`，二者不可同时出现）。 |
| `sweep` | `sweep.workload` 引用的变量名必须预先在 `workload.vars` 中声明；`sweep.system` 仅显式列出的轴参与笛卡尔积展开（未列出的轴沿用 `system` 基线值）。`sweep.system.caching_spec` 的元素支持标量简写或带专属子字段（如 `block_size`、`num_blocks`）的 `caching_spec` 变体映射；`sweep.system.scheduler_spec` 的元素为 `SchedulerSpec` 映射（如 `[{batching_policy: prefill-first}, {batching_policy: {kind: token-budget, token_budget: 256}}]`），用例身份键展平为 `batching_policy` + `token_budget`（默认 `prefill-first` 不进入 `canonical_id` 与 `engine_config` 字符串）。 |
| `comparison` | 仅 `scheduler` 支持：按 `group_by` 维度将用例分组，在组内匹配 `baseline` 约束的唯一基线用例，沿 `metrics[].path` 提取均值计算 `speedup`（`baseline / candidate`）、`gain`（`(candidate - baseline) / baseline`）或 `reduction`（`(baseline - candidate) / baseline`）。 |
| `checks` | 展开期语义守卫：`free_batch_slot_at_t0`（要求 $t=0$ 到达请求数 $<$ `max_running_requests`）、`{min_max_running_requests: N}`、`{min_output_len: N}`（Lockstep Decode 测量至少要求 `min_output_len: 2`）。 |

> [!IMPORTANT]
> **Lockstep 引擎边界约束**：`lockstep` 套件禁止声明 `max_running_requests`、`paged-kv-cache`、`continuous-batching`、`token-budget`、`execution.interleave`、`comparison` 以及 scheduler 专属守卫（`free_batch_slot_at_t0`、`min_max_running_requests`）。

---

## 三、长度/到达分布与展开顺序

### 1. 确定性分布与随机分布
- **确定性长度分布（`prompt_len` / `output_len`）**：
  - `constant`：标量正整数 $L$（所有请求等长；在 `lockstep` 单组下保持标量广播语义）。
  - `explicit`：长度为 `count` 的正整数列表 $[L_0, L_1, \dots]$。
  - `alternating`：`{alternating: [a, b]}`，偶数请求取 $a$、奇数请求取 $b$。
  - `ladder`：`{ladder: {max: M, floor: F}}`，要求组内 `count == 4`，生成阶梯四元组 $[\max(F, \lfloor M/8 \rfloor), \max(F, \lfloor M/4 \rfloor), \max(F, \lfloor M/2 \rfloor), M]$。
- **确定性到达分布（`arrival_ms`）**：
  - `constant`：标量非负浮点数 $t$（默认 `0.0`，所有请求同时到达）。
  - `explicit`：长度为 `count` 的非负浮点数列表。
  - `interval`：`{interval: d}`，第 $i$ 个请求在 $i \times d\text{ ms}$ 到达。
- **确定性伪随机分布（需提供 `workload.seed`）**：
  - `{log_uniform: {min, max}}`：分层对数均匀分布（Stratified Log-Uniform）+ 洗牌，生成短/中请求占主体、长尾请求并存的偏态分布。
  - `{poisson: {initial_burst, mean_interval_ms}}`：前 `initial_burst` 个请求在 $0\text{ ms}$ 突发到达，后续请求间隔服从指数分布并截断在 $[5, 3 \times \text{mean\_interval\_ms}]\text{ ms}$。
  - **抽样确定性**：同一 `WorkloadSpec` 内所有随机分布共享同一个伪随机数生成器 `random.Random(seed * 1000 + seed_salt)`，严格按「`groups` 声明顺序 $\to$ `prompt_len` $\to$ `output_len` $\to$ `arrival_ms`」依次抽样。

### 2. 参数扫描展开顺序（Execution Order）
[`expand_suite_spec`](../../src/mini_infra/bench/spec/suite/expand.py) 按以下固定嵌套顺序展开为有序用例列表：

```text
sweep.workload（外层场景）
  └── max_running_requests
        └── batching_method
              └── caching_spec（元素为 CachingSpec，可包含不同 block_size 等变体配置）
                    └── scheduler_spec（元素为 SchedulerSpec，可包含不同 token_budget 等变体配置）
```

---

## 四、Release 标准负载规范（`benchmarks/specs/release/`）

版本发布使用的 `small` / `medium` / `large` 三档标准负载声明在 [`benchmarks/specs/release/{lockstep,scheduler}.yaml`](../../benchmarks/specs/release/)，由 `mini-infra bench release` 经 `mini_infra.bench.spec.release` 加载：
- 顶层声明 `default_seed`、`system`、`execution`、`checks` 与 `scales`（必须恰好包含 `small`、`medium`、`large` 三档）。
- 每档 `ScaleSpec` 声明独立的 `seed_salt`（避免三档抽样序列同构）、`workload` 以及可选的 `max_running_requests`。
- 运行 `mini-infra bench release --seed N` 时仅替换顶层 `default_seed`。

> [!WARNING]
> 修改 `benchmarks/specs/release/*.yaml` 会改变跨版本历史基准的可比性，仅允许在受控基准演进流程中变更（详见 [RELEASE.md](../../RELEASE.md)），并受 `tests/unit/test_release_workloads.py` 快照测试守门。

---

## 五、检视、IDE 补全与 `--set` 实验覆盖

```bash
# 1. 列出内置 suite 或查看合并继承与展开后的完整用例
mini-infra bench suites list
mini-infra bench suites show dense-vs-paged-block-size

# 2. 导出 JSON Schema（供 VS Code / JetBrains YAML 插件自动补全与静态校验）
mini-infra bench suites schema --kind suite --output benchmarks/specs/suite.schema.json
mini-infra bench suites schema --kind release --output benchmarks/specs/release.schema.json

# 3. 使用 --set KEY=VALUE 覆盖任意点分路径字段（VALUE 按 YAML 语法解析；结果自动隔离至 tmp/results/suites/）
mini-infra bench sweep --engine scheduler --tag dev \
  --suite dense-vs-paged-length-ratio \
  --set 'sweep.workload.long_prompt_len=[256, 1024]' \
  --set execution.repeats=5
```

