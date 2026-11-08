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

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from mini_infra.bench.cases import (
    CaseDimension,
    CaseDimensionValue,
    ComparisonMetric,
    ComparisonSpec,
    MetricKind,
)
from mini_infra.engine import BatchingMethod, CachingMethod

SuiteEngine = Literal["scheduler", "lockstep"]

# 严格正整数：拒绝 YAML 中的 true/false 与 8.0 之类的隐式转换。
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_LADDER_REQUESTS = 4


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- 长度分布


class ConstantLength(_Strict):
    """所有请求相同长度；lockstep 中保持标量（广播）语义。"""

    kind: Literal["constant"] = "constant"
    value: PositiveInt


class ExplicitLength(_Strict):
    """逐请求显式长度列表；长度必须等于组内 count。"""

    kind: Literal["explicit"] = "explicit"
    values: Annotated[list[PositiveInt], Field(min_length=1)]


class AlternatingLength(_Strict):
    """第 0、2、4… 个请求取 values[0]，第 1、3、5… 个请求取 values[1]。"""

    kind: Literal["alternating"] = "alternating"
    values: Annotated[list[PositiveInt], Field(min_length=2, max_length=2)]


class LadderLength(_Strict):
    """阶梯稀疏分布 [max(floor, m//8), max(floor, m//4), max(floor, m//2), m]；要求组内 count == 4。"""

    kind: Literal["ladder"] = "ladder"
    max: PositiveInt
    floor: PositiveInt = 1


def _normalize_length(raw: Any) -> Any:
    if isinstance(raw, bool):
        raise ValueError("长度不能是布尔值")
    if isinstance(raw, int):
        return {"kind": "constant", "value": raw}
    if isinstance(raw, list):
        return {"kind": "explicit", "values": raw}
    if isinstance(raw, dict) and len(raw) == 1 and "kind" not in raw:
        ((key, value),) = raw.items()
        if key == "alternating":
            return {"kind": "alternating", "values": value}
        if key == "ladder":
            if not isinstance(value, dict):
                raise ValueError("ladder 需要 {max: <int>, floor: <int>} 形式的映射")
            return {"kind": "ladder", **value}
    return raw


LengthSpec = Annotated[
    ConstantLength | ExplicitLength | AlternatingLength | LadderLength,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_length),
]


# --------------------------------------------------------------------------- 到达过程


class ConstantArrival(_Strict):
    """组内所有请求在同一逻辑时刻到达（默认 0 ms，即同时到达）。"""

    kind: Literal["constant"] = "constant"
    value: NonNegativeFloat = 0.0


class ExplicitArrival(_Strict):
    kind: Literal["explicit"] = "explicit"
    values: Annotated[list[NonNegativeFloat], Field(min_length=1)]


class IntervalArrival(_Strict):
    """等间隔到达：第 i 个请求在 i * interval_ms。"""

    kind: Literal["interval"] = "interval"
    interval_ms: NonNegativeFloat


def _normalize_arrival(raw: Any) -> Any:
    if isinstance(raw, bool):
        raise ValueError("到达时间不能是布尔值")
    if isinstance(raw, int | float):
        return {"kind": "constant", "value": raw}
    if isinstance(raw, list):
        return {"kind": "explicit", "values": raw}
    if isinstance(raw, dict) and len(raw) == 1 and "kind" not in raw and "interval" in raw:
        return {"kind": "interval", "interval_ms": raw["interval"]}
    return raw


ArrivalSpec = Annotated[
    ConstantArrival | ExplicitArrival | IntervalArrival,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_arrival),
]


# --------------------------------------------------------------------------- 请求负载


class GroupSpec(_Strict):
    """一组同构请求；多个组按声明顺序拼接成整体请求序列（request_idx 连续编号）。"""

    count: PositiveInt
    prompt_len: LengthSpec
    output_len: LengthSpec
    arrival_ms: ArrivalSpec = ConstantArrival()

    @model_validator(mode="after")
    def _check_shapes(self) -> "GroupSpec":
        for field_name in ("prompt_len", "output_len"):
            length = getattr(self, field_name)
            if isinstance(length, ExplicitLength) and len(length.values) != self.count:
                raise ValueError(f"{field_name} 显式列表长度 {len(length.values)} 必须等于 count={self.count}")
            if isinstance(length, LadderLength) and self.count != _LADDER_REQUESTS:
                raise ValueError(f"{field_name} 的 ladder 要求 count == {_LADDER_REQUESTS}")
        if isinstance(self.arrival_ms, ExplicitArrival) and len(self.arrival_ms.values) != self.count:
            raise ValueError(f"arrival_ms 显式列表长度 {len(self.arrival_ms.values)} 必须等于 count={self.count}")
        return self


class WorkloadBody(_Strict):
    """插值之后的具体请求负载。"""

    groups: Annotated[list[GroupSpec], Field(min_length=1)]


_GROUP_FIELDS = frozenset(GroupSpec.model_fields)


def normalize_workload_body(body: dict[str, Any]) -> dict[str, Any]:
    """单组简写（count / prompt_len / output_len / arrival_ms 平铺）归一化为 groups。

    供 expand 阶段在插值完成后、模型校验前调用。
    """
    if "groups" in body:
        if _GROUP_FIELDS & body.keys():
            raise ValueError("groups 与单组简写字段（count / prompt_len / output_len / arrival_ms）不能同时出现")
        return body
    return {"groups": [dict(body)]}


class WorkloadTemplate(_Strict):
    """含 ${var} 占位符的负载模板；每个 sweep 点插值后才构成具体 WorkloadBody。"""

    id: Annotated[str, Field(min_length=1)]
    vars: dict[str, Any] = Field(default_factory=dict)
    body: dict[str, Any]


# --------------------------------------------------------------------------- 被测系统


class PagedSpec(_Strict):
    block_size: PositiveInt = 16
    num_blocks: Annotated[int, Field(strict=True, ge=2)] | None = None


class SystemSpec(_Strict):
    max_running_requests: PositiveInt | None = None
    batching_method: Annotated[list[BatchingMethod], Field(min_length=1)]
    caching_method: Annotated[list[CachingMethod], Field(min_length=1)]
    paged: PagedSpec = PagedSpec()


class ExecutionSpec(_Strict):
    warmup: Annotated[int, Field(strict=True, ge=0)] = 1
    repeats: PositiveInt = 3
    interleave: list[CaseDimension] = Field(default_factory=list)


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


class ComparisonConfig(_Strict):
    metrics: Annotated[list[MetricSpec], Field(min_length=1)]
    group_by: Annotated[list[CaseDimension], Field(min_length=1)]
    baseline: Annotated[dict[CaseDimension, str | int], Field(min_length=1)]

    def to_runtime(self) -> ComparisonSpec:
        """转换为运行期 ComparisonSpec（baseline 取值按维度还原为强类型）。"""
        baseline: dict[CaseDimension, CaseDimensionValue] = {}
        for dim, raw in self.baseline.items():
            if dim is CaseDimension.BATCHING_METHOD:
                baseline[dim] = BatchingMethod.from_str(str(raw))
            elif dim is CaseDimension.CACHING_METHOD:
                baseline[dim] = CachingMethod.from_str(str(raw))
            elif dim is CaseDimension.WORKLOAD_ID:
                baseline[dim] = str(raw)
            elif dim is CaseDimension.MAX_RUNNING_REQUESTS:
                if not isinstance(raw, int):
                    raise ValueError("baseline.max_running_requests 必须是整数")
                baseline[dim] = raw
            else:
                raise ValueError(f"baseline 不支持维度 {dim.value}")
        return ComparisonSpec(
            metrics=tuple(ComparisonMetric(m.path, m.kind, m.name) for m in self.metrics),
            group_by=tuple(self.group_by),
            baseline=baseline,
        )


class MinMaxRunningRequestsCheck(_Strict):
    min_max_running_requests: PositiveInt


class MinOutputLenCheck(_Strict):
    min_output_len: PositiveInt


CheckSpec = Literal["free_batch_slot_at_t0"] | MinMaxRunningRequestsCheck | MinOutputLenCheck


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
    comparison: ComparisonConfig | None = None
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
    "AlternatingLength",
    "ArrivalSpec",
    "CheckSpec",
    "ComparisonConfig",
    "ConstantArrival",
    "ConstantLength",
    "ExecutionSpec",
    "ExplicitArrival",
    "ExplicitLength",
    "GroupSpec",
    "IntervalArrival",
    "LadderLength",
    "LengthSpec",
    "MetricSpec",
    "MinMaxRunningRequestsCheck",
    "MinOutputLenCheck",
    "PagedSpec",
    "PositiveInt",
    "SuiteEngine",
    "SuiteSpec",
    "SweepSpec",
    "SystemSpec",
    "WorkloadBody",
    "WorkloadTemplate",
    "normalize_workload_body",
    "suite_json_schema",
]
