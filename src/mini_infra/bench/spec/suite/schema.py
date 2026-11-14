"""Suite YAML 声明式配置 schema（Pydantic v2，纯 CPU）。

YAML 是外部输入边界：这里一次性完成全部校验，下游（expand / suites / run_suite）只信任已校验的类型。

结构（与 docs 中的 YAML 设计一一对应）::

    name / engine / description
    execution:  warmup / repeats / interleave
    system:     engine_mode / batching_method / caching_spec / scheduler_spec
    workload:   vars / id / (groups | 单组简写字段)
    sweep:      workload（dict-of-lists 笛卡尔积，或 list-of-dicts 显式场景）/ system
    comparison: metrics / group_by / baseline
    checks:     具名语义守卫

完整 EBNF 形式化语法与字段语义规范见 `docs/benchmarks/spec-schema.md`。
"""

import re
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator

from mini_infra.bench.cases import CaseDimension
from mini_infra.bench.comparison import MetricKind
from mini_infra.bench.spec.common.schema import (
    CachingSpec,
    CheckSpec,
    ExecutionSpec,
    GroupSpec,
    MinMaxRunningRequestsCheck,
    PagedCacheSpec,
    PositiveInt,
    SchedulerEngineSpec,
    SchedulerSpec,
    SuiteEngine,
    SystemSpec,
    TokenBudgetSpec,
    _Strict,
)
from mini_infra.engine import BatchingMethod

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


# --------------------------------------------------------------------------- 负载模板


class WorkloadTemplate(_Strict):
    """含 ${var} 占位符的负载模板；每个 sweep 点插值后才构成具体 WorkloadSpec。"""

    vars: dict[str, Any] = Field(default_factory=dict)
    id: Annotated[str, Field(min_length=1)]
    body: dict[str, Any]


# --------------------------------------------------------------------------- 扫描


class SystemSweepSpec(_Strict):
    """系统层扫描轴：仅显式指定的字段会在 expand 阶段展开为多个候选点。"""

    max_running_requests: Annotated[list[PositiveInt], Field(min_length=1)] | None = None
    batching_method: Annotated[list[BatchingMethod], Field(min_length=1)] | None = None
    caching_spec: Annotated[list[CachingSpec], Field(min_length=1)] | None = None
    scheduler_spec: Annotated[list[SchedulerSpec], Field(min_length=1)] | None = None


class SweepSpec(_Strict):
    """dict-of-lists = 笛卡尔积；list-of-dicts（仅 workload）= 显式场景列表。"""

    workload: dict[str, list[Any]] | list[dict[str, Any]] = Field(default_factory=dict)
    system: SystemSweepSpec = Field(default_factory=SystemSweepSpec)

    @field_validator("workload")
    @classmethod
    def _non_empty_axes(cls, value: dict[str, list[Any]] | list[dict[str, Any]]) -> Any:
        if isinstance(value, dict):
            for key, values in value.items():
                if not values:
                    raise ValueError(f"sweep.workload.{key} 不能为空列表")
        elif any(not scenario for scenario in value):
            raise ValueError("sweep.workload 的每个场景必须至少覆盖一个变量")
        return value


# --------------------------------------------------------------------------- 对比与守卫


class MetricSpec(_Strict):
    path: Annotated[str, Field(min_length=1)]
    kind: MetricKind
    name: str | None = None


class ComparisonSpec(_Strict):
    metrics: Annotated[list[MetricSpec], Field(min_length=1)]
    group_by: Annotated[list[CaseDimension], Field(min_length=1)]
    baseline: Annotated[dict[CaseDimension, str | int], Field(min_length=1)]


# --------------------------------------------------------------------------- 顶层


class SuiteSpec(_Strict):
    """一份完整的 suite 声明。"""

    name: str
    description: str = ""
    execution: ExecutionSpec = ExecutionSpec()
    system: SystemSpec
    workload: WorkloadTemplate
    sweep: SweepSpec = SweepSpec()
    comparison: ComparisonSpec | None = None
    checks: list[CheckSpec] = Field(default_factory=list)

    @property
    def engine(self) -> SuiteEngine:
        return self.system.engine_mode.kind

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _NAME_PATTERN.fullmatch(value):
            raise ValueError(f"suite 名称 {value!r} 只能包含小写字母、数字与连字符")
        return value

    @field_validator("workload", mode="before")
    @classmethod
    def _split_workload(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        return {
            "vars": raw.get("vars", {}),
            "id": raw.get("id"),
            "body": {k: v for k, v in raw.items() if k not in ("id", "vars")},
        }

    @model_validator(mode="after")
    def _check_cross_fields(self) -> "SuiteSpec":
        """跨字段联合语义校验。

        校验项包括：
        1. sweep.workload 变量一致性：扫描的变量必须预先在 workload.vars 中声明，防止变量名拼写错误；
        2. 引擎特异性约束：
           - scheduler: 必须明确指定 max_running_requests（在 system 基线或 sweep.system 中）；
             token-budget 要求生效的 batching_method 全部为 continuous-batching，且 token_budget 不小于每个生效的
             max_running_requests（与 engine.config.validate_engine_config 同一约束，在展开前提前拒绝）；
           - lockstep: 严格限制仅走固定批次和 Dense 路径，禁止使用 scheduler 特性
             （如并发容量、PagedKV、Continuous Batching、token-budget 等）。
        """
        # 1. 检查 sweep.workload 变量是否存在于 workload.vars 中
        template_vars = self.workload.vars
        sweep_workload = self.sweep.workload
        swept_names = (
            set(sweep_workload)
            if isinstance(sweep_workload, dict)
            else {key for scenario in sweep_workload for key in scenario}
        )
        unknown = swept_names - template_vars.keys()
        if unknown:
            raise ValueError(f"sweep.workload 引用了未在 workload.vars 中声明的变量: {sorted(unknown)}")

        # 2. 计算实际生效的系统维度取值集合（sweep.system 显式覆盖基线 system）
        effective_caching_specs = (
            self.sweep.system.caching_spec if self.sweep.system.caching_spec is not None else [self.system.caching_spec]
        )
        effective_batching_methods = (
            self.sweep.system.batching_method
            if self.sweep.system.batching_method is not None
            else [self.system.batching_method]
        )
        effective_scheduler_specs = (
            self.sweep.system.scheduler_spec
            if self.sweep.system.scheduler_spec is not None
            else [self.system.scheduler_spec]
        )

        # 3. 根据引擎类别校验对应特性的支持边界
        if isinstance(self.system.engine_mode, SchedulerEngineSpec):
            self._check_scheduler(
                self.system.engine_mode,
                effective_batching_methods,
                effective_scheduler_specs,
            )
        else:
            self._check_lockstep(effective_caching_specs, effective_batching_methods, effective_scheduler_specs)
        return self

    def _check_scheduler(
        self,
        engine_mode: SchedulerEngineSpec,
        effective_batching_methods: list[BatchingMethod],
        effective_scheduler_specs: list[SchedulerSpec],
    ) -> None:
        if self.sweep.system.max_running_requests is not None:
            effective_mrrs = self.sweep.system.max_running_requests
        elif engine_mode.max_running_requests is not None:
            effective_mrrs = [engine_mode.max_running_requests]
        else:
            raise ValueError(
                "scheduler suite 必须提供 system.engine_mode.max_running_requests（或在 sweep.system 中扫描）"
            )

        token_budget_specs = [spec for spec in effective_scheduler_specs if isinstance(spec, TokenBudgetSpec)]
        if not token_budget_specs:
            return
        if any(method is not BatchingMethod.CONTINUOUS for method in effective_batching_methods):
            raise ValueError("token-budget 调度策略要求生效的 batching_method 全部为 continuous-batching")
        max_mrr = max(effective_mrrs)
        for spec in token_budget_specs:
            if spec.token_budget < max_mrr:
                raise ValueError(f"token_budget={spec.token_budget} 不能小于生效的 max_running_requests={max_mrr}")

    def _check_lockstep(
        self,
        effective_caching_specs: list[CachingSpec],
        effective_batching_methods: list[BatchingMethod],
        effective_scheduler_specs: list[SchedulerSpec],
    ) -> None:
        if self.sweep.system.max_running_requests is not None:
            raise ValueError("lockstep suite 不支持 max_running_requests")
        if any(isinstance(s, TokenBudgetSpec) for s in effective_scheduler_specs):
            raise ValueError("lockstep suite 不支持 token-budget 调度策略")
        if any(isinstance(c, PagedCacheSpec) for c in effective_caching_specs):
            raise ValueError("lockstep suite 不支持 paged-kv-cache")
        if BatchingMethod.CONTINUOUS in effective_batching_methods:
            raise ValueError("lockstep suite 不支持 continuous-batching")
        if self.execution.interleave:
            raise ValueError("lockstep suite 不支持 execution.interleave")
        if self.comparison is not None:
            raise ValueError("lockstep suite 不支持 comparison")
        for check in self.checks:
            if check == "free_batch_slot_at_t0" or isinstance(check, MinMaxRunningRequestsCheck):
                raise ValueError(f"lockstep suite 不支持检查 {check!r}")


def suite_json_schema() -> dict[str, Any]:
    """导出供 IDE 补全用的 JSON Schema（描述 YAML 文件形状，而非插值后的内部模型）。

    与内部模型的差异：允许 `extends`；`workload` 含 ${var} 占位符，无法静态强类型化，因此只约束 id / vars / groups
    的外形（单组字段的含义见 `$defs.GroupSpec`，仅作文档参考）；`required` 仅 name，因为其余字段可由 extends 继承。
    """
    schema = SuiteSpec.model_json_schema()
    schema["title"] = "mini-infra benchmark suite"
    schema["required"] = ["name"]
    schema["properties"]["extends"] = {
        "type": "string",
        "description": "相对当前文件的基底 YAML 路径（映射深合并，列表整体替换）",
    }
    schema["properties"]["workload"] = {
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": {"type": "string", "description": "workload_id，可含 ${var}"},
            "vars": {"type": "object", "description": "可被 sweep.workload 覆盖的变量"},
            "groups": {
                "type": "array",
                "items": {"type": "object", "description": "字段含义见 $defs.GroupSpec（值可用 ${var} 占位）"},
                "description": "请求组，按顺序拼接",
            },
        },
        "description": "单组可省略 groups，把 count / prompt_len / output_len / arrival_ms 平铺在此处；值中可用 ${var}",
    }
    group_schema = GroupSpec.model_json_schema()
    schema["$defs"].update(group_schema.pop("$defs", {}))
    schema["$defs"]["GroupSpec"] = group_schema
    schema["$defs"].pop("WorkloadTemplate", None)
    return schema


__all__ = [
    "CheckSpec",
    "ComparisonSpec",
    "MetricSpec",
    "MinMaxRunningRequestsCheck",
    "PagedCacheSpec",
    "SchedulerEngineSpec",
    "SuiteEngine",
    "SuiteSpec",
    "SweepSpec",
    "SystemSpec",
    "SystemSweepSpec",
    "WorkloadTemplate",
    "suite_json_schema",
]
