"""跨 spec 共享的基础 schema（Pydantic v2，纯 CPU）。

定义长度/到达时间分布、单组/多组负载规范、执行参数与基础校验标量，供 suite 与 release 共同继承与引用。
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from mini_infra.bench.cases import CaseDimension
from mini_infra.engine import BatchingMethod, CachingMethod

SuiteEngine = Literal["scheduler", "lockstep"]

# 严格正整数：拒绝 YAML 中的 true/false 与 8.0 之类的隐式转换。
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]

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


class LogUniformLength(_Strict):
    """分层对数均匀分布（Stratified Log-Uniform）+ 随机打乱：短/中请求居多、长尾并存的偏态长度；需要 workload.seed。"""

    kind: Literal["log_uniform"] = "log_uniform"
    min: PositiveInt
    max: PositiveInt

    @model_validator(mode="after")
    def _check_range(self) -> "LogUniformLength":
        if self.min > self.max:
            raise ValueError(f"log_uniform 要求 min <= max，实际为 {self.min} > {self.max}")
        return self


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
        if key == "log_uniform":
            if not isinstance(value, dict):
                raise ValueError("log_uniform 需要 {min: <int>, max: <int>} 形式的映射")
            return {"kind": "log_uniform", **value}
    return raw


LengthSpec = Annotated[
    ConstantLength | ExplicitLength | AlternatingLength | LadderLength | LogUniformLength,
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


class PoissonArrival(_Strict):
    """带初始突发的泊松过程：前 initial_burst 个请求在 0 ms，其后为指数间隔（夹在 [5, 3×均值] ms）；需要 seed。"""

    kind: Literal["poisson"] = "poisson"
    initial_burst: PositiveInt
    mean_interval_ms: Annotated[float, Field(gt=0.0)]


def _normalize_arrival(raw: Any) -> Any:
    if isinstance(raw, bool):
        raise ValueError("到达时间不能是布尔值")
    if isinstance(raw, int | float):
        return {"kind": "constant", "value": raw}
    if isinstance(raw, list):
        return {"kind": "explicit", "values": raw}
    if isinstance(raw, dict) and len(raw) == 1 and "kind" not in raw and "interval" in raw:
        return {"kind": "interval", "interval_ms": raw["interval"]}
    if isinstance(raw, dict) and len(raw) == 1 and "kind" not in raw and "poisson" in raw:
        if not isinstance(raw["poisson"], dict):
            raise ValueError("poisson 需要 {initial_burst: <int>, mean_interval_ms: <float>} 形式的映射")
        return {"kind": "poisson", **raw["poisson"]}
    return raw


ArrivalSpec = Annotated[
    ConstantArrival | ExplicitArrival | IntervalArrival | PoissonArrival,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_arrival),
]


# --------------------------------------------------------------------------- 请求负载与执行


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


class WorkloadSpec(_Strict):
    """插值之后的具体请求负载。

    seed / seed_salt：随机分布（log_uniform / poisson）共用一个 `random.Random(seed * 1000 + seed_salt)`，
    按「组顺序 → 每组 prompt_len → output_len → arrival_ms」依次抽样，因此结果由种子严格确定。
    """

    groups: Annotated[list[GroupSpec], Field(min_length=1)]
    seed: Annotated[int, Field(strict=True)] | None = None
    seed_salt: Annotated[int, Field(strict=True)] = 0

    @model_validator(mode="after")
    def _stochastic_needs_seed(self) -> "WorkloadSpec":
        stochastic = any(
            isinstance(g.prompt_len, LogUniformLength)
            or isinstance(g.output_len, LogUniformLength)
            or isinstance(g.arrival_ms, PoissonArrival)
            for g in self.groups
        )
        if stochastic and self.seed is None:
            raise ValueError("使用 log_uniform / poisson 时必须提供 workload.seed")
        return self


_GROUP_FIELDS = frozenset(GroupSpec.model_fields)


def normalize_workload_body(body: dict[str, Any]) -> dict[str, Any]:
    """单组简写（count / prompt_len / output_len / arrival_ms 平铺）归一化为 groups。

    供 expand 阶段在插值完成后、模型校验前调用。
    """
    meta = {key: body[key] for key in ("seed", "seed_salt") if key in body}
    rest = {key: value for key, value in body.items() if key not in meta}
    if "groups" in rest:
        if _GROUP_FIELDS & rest.keys():
            raise ValueError("groups 与单组简写字段（count / prompt_len / output_len / arrival_ms）不能同时出现")
        return {**rest, **meta}
    return {"groups": [rest], **meta}


class ExecutionSpec(_Strict):
    warmup: Annotated[int, Field(strict=True, ge=0)] = 1
    repeats: PositiveInt = 3
    interleave: list[CaseDimension] = Field(default_factory=list)


# --------------------------------------------------------------------------- 被测系统


class PagedSpec(_Strict):
    block_size: PositiveInt = 16
    num_blocks: Annotated[int, Field(strict=True, ge=2)] | None = None


class SystemSpec(_Strict):
    max_running_requests: PositiveInt | None = None
    batching_method: BatchingMethod
    caching_method: CachingMethod
    paged: PagedSpec = PagedSpec()


# --------------------------------------------------------------------------- 语义检查守卫


class MinMaxRunningRequestsCheck(_Strict):
    min_max_running_requests: PositiveInt


class MinOutputLenCheck(_Strict):
    min_output_len: PositiveInt


CheckSpec = Literal["free_batch_slot_at_t0"] | MinMaxRunningRequestsCheck | MinOutputLenCheck


__all__ = [
    "AlternatingLength",
    "ArrivalSpec",
    "CheckSpec",
    "ConstantArrival",
    "ConstantLength",
    "ExecutionSpec",
    "ExplicitArrival",
    "ExplicitLength",
    "GroupSpec",
    "IntervalArrival",
    "LadderLength",
    "LengthSpec",
    "LogUniformLength",
    "MinMaxRunningRequestsCheck",
    "MinOutputLenCheck",
    "NonNegativeFloat",
    "PagedSpec",
    "PoissonArrival",
    "PositiveInt",
    "SuiteEngine",
    "SystemSpec",
    "WorkloadSpec",
    "_Strict",
    "normalize_workload_body",
]
