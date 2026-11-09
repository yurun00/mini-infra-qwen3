"""Suite YAML 声明式配置 schema（Pydantic v2，纯 CPU）。

YAML 是外部输入边界：这里一次性完成全部校验，下游（expand / suites / run_suite）只信任已校验的类型。

结构（与 docs 中的 YAML 设计一一对应）::

    name / engine / description
    execution:  warmup / repeats / interleave
    system:     max_running_requests / batching_method[] / caching_method[] / paged
    workload:   id / vars / (groups | 单组简写字段)
    sweep:      workload（dict-of-lists 笛卡尔积，或 list-of-dicts 显式场景）/ system
    comparison: metrics / group_by / baseline
    checks:     具名语义守卫
"""

import re
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from mini_infra.bench.cases import (
    CaseDimension,
    MetricKind,
)
from mini_infra.bench.spec.common.schema import (
    CheckSpec,
    ExecutionSpec,
    GroupSpec,
    MinMaxRunningRequestsCheck,
    MinOutputLenCheck,
    PagedSpec,
    PositiveInt,
    SuiteEngine,
    SystemSpec,
    _Strict,
)
from mini_infra.engine import CachingMethod

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


# --------------------------------------------------------------------------- 负载模板


class WorkloadTemplate(_Strict):
    """含 ${var} 占位符的负载模板；每个 sweep 点插值后才构成具体 WorkloadSpec。"""

    id: Annotated[str, Field(min_length=1)]
    vars: dict[str, Any] = Field(default_factory=dict)
    body: dict[str, Any]


# --------------------------------------------------------------------------- 扫描


SystemSweepKey = Literal["max_running_requests", "block_size"]


class SweepSpec(_Strict):
    """dict-of-lists = 笛卡尔积；list-of-dicts（仅 workload）= 显式场景列表。"""

    workload: dict[str, list[Any]] | list[dict[str, Any]] = Field(default_factory=dict)
    system: dict[SystemSweepKey, Annotated[list[PositiveInt], Field(min_length=1)]] = Field(default_factory=dict)

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
    engine: SuiteEngine
    description: str = ""
    execution: ExecutionSpec = ExecutionSpec()
    system: SystemSpec
    workload: WorkloadTemplate
    sweep: SweepSpec = SweepSpec()
    comparison: ComparisonSpec | None = None
    checks: list[CheckSpec] = Field(default_factory=list)

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
            "id": raw.get("id"),
            "vars": raw.get("vars", {}),
            "body": {k: v for k, v in raw.items() if k not in ("id", "vars")},
        }

    @model_validator(mode="after")
    def _check_cross_fields(self) -> "SuiteSpec":
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
        if "block_size" in self.sweep.system and CachingMethod.PAGED not in self.system.caching_method:
            raise ValueError("sweep.system.block_size 要求 system.caching_method 包含 paged-kv-cache")

        if self.engine == "scheduler":
            if self.system.max_running_requests is None and "max_running_requests" not in self.sweep.system:
                raise ValueError("scheduler suite 必须提供 system.max_running_requests（或在 sweep.system 中扫描）")
        else:
            self._check_lockstep()
        return self

    def _check_lockstep(self) -> None:
        if self.system.max_running_requests is not None or "max_running_requests" in self.sweep.system:
            raise ValueError("lockstep suite 不支持 max_running_requests")
        if CachingMethod.PAGED in self.system.caching_method:
            raise ValueError("lockstep suite 不支持 paged-kv-cache")
        if "block_size" in self.sweep.system:
            raise ValueError("lockstep suite 不支持 sweep.system.block_size")
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
    "MinOutputLenCheck",
    "PagedSpec",
    "SuiteEngine",
    "SuiteSpec",
    "SweepSpec",
    "SystemSpec",
    "SystemSweepKey",
    "WorkloadTemplate",
    "suite_json_schema",
]
