"""跨 spec 共享的基础 schema（Pydantic v2，纯 CPU）。

定义长度/到达时间分布、单组/多组负载规范、执行参数与基础校验标量，供 suite 与 release 共同继承与引用。
"""

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from mini_infra.bench.cases import CaseDimension
from mini_infra.core import (
    ChunkedPrefillConfig,
    DenseCacheConfig,
    NoCacheConfig,
    PagedCacheConfig,
    PrefillFirstConfig,
    SchedulerConfig,
    TokenBudgetConfig,
)
from mini_infra.engine import BatchingMethod, EngineConfig

SuiteEngine = Literal["scheduler", "lockstep"]

# 严格正整数：拒绝 YAML 中的 true/false 与 8.0 之类的隐式转换。
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
GroupName = Annotated[str, Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]*$")]

_LADDER_REQUESTS = 4


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def inherit_from(self, base: "_Strict") -> Any:
        """若与 base 同类型，用 base 递归补齐当前实例未显式指定的字段；否则原样返回 self。"""
        if type(self) is not type(base):
            return self
        merged: dict[str, Any] = {}
        for name in type(self).model_fields:
            self_val = getattr(self, name)
            base_val = getattr(base, name)
            if name not in self.model_fields_set or self_val is None:
                merged[name] = base_val
            elif isinstance(self_val, _Strict) and isinstance(base_val, _Strict):
                merged[name] = self_val.inherit_from(base_val)
            else:
                merged[name] = self_val
        return type(self)(**merged)


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
            raise ValueError(
                f"log_uniform 要求 min <= max，实际为 {self.min} > {self.max}"
            )
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
    if (
        isinstance(raw, dict)
        and len(raw) == 1
        and "kind" not in raw
        and "interval" in raw
    ):
        return {"kind": "interval", "interval_ms": raw["interval"]}
    if isinstance(raw, dict) and len(raw) == 1 and "kind" not in raw and "poisson" in raw:
        if not isinstance(raw["poisson"], dict):
            raise ValueError(
                "poisson 需要 {initial_burst: <int>, mean_interval_ms: <float>} 形式的映射"
            )
        return {"kind": "poisson", **raw["poisson"]}
    return raw


ArrivalSpec = Annotated[
    ConstantArrival | ExplicitArrival | IntervalArrival | PoissonArrival,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_arrival),
]


# --------------------------------------------------------------------------- 请求负载与执行


class GroupSpec(_Strict):
    """一组同构请求；多个组按声明顺序拼接成整体请求序列（request_idx 连续编号）。

    可选 name 将该组对应的 [start, end) 请求切片注册为具名子集，产出 group_metrics.<name>.* 统计。
    """

    name: GroupName | None = None
    count: PositiveInt
    prompt_len: LengthSpec
    output_len: LengthSpec
    arrival_ms: ArrivalSpec = ConstantArrival()

    @model_validator(mode="after")
    def _check_shapes(self) -> "GroupSpec":
        for field_name in ("prompt_len", "output_len"):
            length = getattr(self, field_name)
            if isinstance(length, ExplicitLength) and len(length.values) != self.count:
                raise ValueError(
                    f"{field_name} 显式列表长度 {len(length.values)} 必须等于 count={self.count}"
                )
            if isinstance(length, LadderLength) and self.count != _LADDER_REQUESTS:
                raise ValueError(
                    f"{field_name} 的 ladder 要求 count == {_LADDER_REQUESTS}"
                )
        if (
            isinstance(self.arrival_ms, ExplicitArrival)
            and len(self.arrival_ms.values) != self.count
        ):
            raise ValueError(
                f"arrival_ms 显式列表长度 {len(self.arrival_ms.values)} 必须等于 count={self.count}"
            )
        return self


class WorkloadSpec(_Strict):
    """插值之后的具体请求负载。

    seed / seed_salt：随机分布（log_uniform / poisson）共用一个 `random.Random(seed * 1000 + seed_salt)`，
    按「组顺序 → 每组 prompt_len → output_len → arrival_ms」依次抽样，因此结果由种子严格确定。
    shared_prefix_len：全体请求 Prompt 共享的公共前缀 Token 数（prompt_len 含该前缀），用于 prefix caching 评测；默认 0 即各请求独立。
    """

    groups: Annotated[list[GroupSpec], Field(min_length=1)]
    seed: Annotated[int, Field(strict=True)] | None = None
    seed_salt: Annotated[int, Field(strict=True)] = 0
    shared_prefix_len: Annotated[int, Field(strict=True, ge=0)] = 0

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
        names = [g.name for g in self.groups if g.name is not None]
        if len(names) != len(set(names)):
            raise ValueError(f"workload.groups 中存在重复的组名: {names}")
        return self


_GROUP_FIELDS = frozenset(GroupSpec.model_fields)


def normalize_workload_body(body: dict[str, Any]) -> dict[str, Any]:
    """单组简写（count / prompt_len / output_len / arrival_ms 平铺）归一化为 groups。

    供 expand 阶段在插值完成后、模型校验前调用。
    """
    meta = {
        key: body[key]
        for key in ("seed", "seed_salt", "shared_prefix_len")
        if key in body
    }
    rest = {key: value for key, value in body.items() if key not in meta}
    if "groups" in rest:
        if _GROUP_FIELDS & rest.keys():
            raise ValueError(
                "groups 与单组简写字段（count / prompt_len / output_len / arrival_ms）不能同时出现"
            )
        return {**rest, **meta}
    return {"groups": [rest], **meta}


class ExecutionSpec(_Strict):
    warmup: Annotated[int, Field(strict=True, ge=0)] = 1
    repeats: PositiveInt = 3
    interleave: list[CaseDimension] = Field(default_factory=list)


# --------------------------------------------------------------------------- 被测系统


class LockstepEngineSpec(_Strict):
    """Lockstep 离线执行引擎（全批次同步推进，无调度队列与动态到达）。"""

    kind: Literal["lockstep"] = "lockstep"


class SchedulerEngineSpec(_Strict):
    """Scheduler 离散事件调度引擎（支持到达队列、Running 并发容量管理）。"""

    kind: Literal["scheduler"] = "scheduler"


def _normalize_engine(raw: Any) -> Any:
    """归一化 engine_mode 输入：将标量字符串提升为标准判别字典。"""
    if isinstance(raw, str):
        return {"kind": raw}
    return raw


EngineModeSpec = Annotated[
    LockstepEngineSpec | SchedulerEngineSpec,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_engine),
]


class NoCacheSpec(_Strict):
    """不启用 KV Cache。"""

    kind: Literal["none"] = "none"

    @property
    def kv_cache_config(self) -> NoCacheConfig:
        return NoCacheConfig()


class DenseCacheSpec(_Strict):
    """Dense KV Cache（预分配连续 batch slot）。"""

    kind: Literal["dense-kv-cache"] = "dense-kv-cache"

    @property
    def kv_cache_config(self) -> DenseCacheConfig:
        return DenseCacheConfig()


class PagedCacheSpec(_Strict):
    """Paged KV Cache（physical block pool，按需动态分配；prefix_caching 开启跨请求复用 cached block）。"""

    kind: Literal["paged-kv-cache"] = "paged-kv-cache"
    block_size: PositiveInt = PagedCacheConfig.block_size
    num_blocks: Annotated[int, Field(strict=True, ge=2)] | None = None
    prefix_caching: Annotated[bool, Field(strict=True)] = False

    @property
    def kv_cache_config(self) -> PagedCacheConfig:
        return PagedCacheConfig(
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            is_prefix_caching_enabled=self.prefix_caching,
        )


def _normalize_caching(raw: Any) -> Any:
    """归一化 caching_spec 输入：将标量字符串提升为标准判别字典。"""
    if isinstance(raw, str):
        return {"kind": raw}
    return raw


CachingSpec = Annotated[
    NoCacheSpec | DenseCacheSpec | PagedCacheSpec,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_caching),
]


class PrefillFirstSpec(_Strict):
    """prefill-first 单步组批策略（默认）：WaitingQueue 有可入场请求即整批 Prefill。"""

    kind: Literal["prefill-first"] = PrefillFirstConfig.kind

    @property
    def batching_policy(self) -> PrefillFirstConfig:
        return PrefillFirstConfig()


class TokenBudgetSpec(_Strict):
    """token-budget 单步组批策略：padded prompt token 预算限制单次 Prefill，并启用 decode protection。"""

    kind: Literal["token-budget"] = TokenBudgetConfig.kind
    token_budget: PositiveInt

    @property
    def batching_policy(self) -> TokenBudgetConfig:
        return TokenBudgetConfig(self.token_budget)


class ChunkedPrefillSpec(_Strict):
    """chunked-prefill 单步组批策略：padded chunk token 硬预算切分长 Prompt，并启用 decode protection。"""

    kind: Literal["chunked-prefill"] = ChunkedPrefillConfig.kind
    token_budget: PositiveInt

    @property
    def batching_policy(self) -> ChunkedPrefillConfig:
        return ChunkedPrefillConfig(self.token_budget)


BudgetedBatchingPolicySpec = TokenBudgetSpec | ChunkedPrefillSpec


def _normalize_batching_policy(raw: Any) -> Any:
    """归一化 batching_policy 输入：将标量字符串提升为标准判别字典。"""
    if isinstance(raw, str):
        return {"kind": raw}
    return raw


BatchingPolicySpec = Annotated[
    PrefillFirstSpec | TokenBudgetSpec | ChunkedPrefillSpec,
    Field(discriminator="kind"),
    BeforeValidator(_normalize_batching_policy),
]


class SchedulerSpec(_Strict):
    """迭代级调度器配置声明（与 core.SchedulerConfig 对齐）。"""

    max_running_requests: PositiveInt | None = None
    batching_policy: BatchingPolicySpec = PrefillFirstSpec()

    @property
    def scheduler_config(self) -> SchedulerConfig:
        return SchedulerConfig(
            max_running_requests=self.max_running_requests,
            batching_policy=self.batching_policy.batching_policy,
        )


class SystemSpec(_Strict):
    engine_mode: EngineModeSpec
    batching_method: BatchingMethod
    caching_spec: CachingSpec = DenseCacheSpec()
    scheduler_spec: SchedulerSpec = SchedulerSpec()

    @property
    def default_engine_config(self) -> EngineConfig:
        """系统配置声明的默认 EngineConfig。"""
        return EngineConfig(
            self.caching_spec.kv_cache_config,
            batching_method=self.batching_method,
            scheduler_config=self.scheduler_spec.scheduler_config,
        )


# --------------------------------------------------------------------------- 语义检查守卫


class MinMaxRunningRequestsCheck(_Strict):
    min_max_running_requests: PositiveInt


class MinOutputLenCheck(_Strict):
    min_output_len: PositiveInt


CheckSpec = (
    Literal["free_batch_slot_at_t0"] | MinMaxRunningRequestsCheck | MinOutputLenCheck
)


__all__ = [
    "AlternatingLength",
    "ArrivalSpec",
    "BatchingPolicySpec",
    "BudgetedBatchingPolicySpec",
    "CachingSpec",
    "CheckSpec",
    "ChunkedPrefillSpec",
    "ConstantArrival",
    "ConstantLength",
    "DenseCacheSpec",
    "EngineModeSpec",
    "ExecutionSpec",
    "ExplicitArrival",
    "ExplicitLength",
    "GroupSpec",
    "IntervalArrival",
    "LadderLength",
    "LengthSpec",
    "LockstepEngineSpec",
    "LogUniformLength",
    "MinMaxRunningRequestsCheck",
    "MinOutputLenCheck",
    "NoCacheSpec",
    "NonNegativeFloat",
    "PagedCacheSpec",
    "PoissonArrival",
    "PositiveInt",
    "PrefillFirstSpec",
    "SchedulerEngineSpec",
    "SchedulerSpec",
    "SuiteEngine",
    "SystemSpec",
    "TokenBudgetSpec",
    "WorkloadSpec",
    "_Strict",
    "normalize_workload_body",
]
