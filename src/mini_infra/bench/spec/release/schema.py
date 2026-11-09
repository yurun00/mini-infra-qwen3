"""Release 标准负载的 YAML 声明（`benchmarks/specs/release/<engine>.yaml`）：三档确定性伪随机负载。

与 suite 不同，release 负载不固定 Mode（由 `bench release` 命令行在发布时指定），只声明：
    default_seed / default_mode / execution（warmup、repeats）/ scales（每档的 workload 与 seed_salt）

每档的随机分布共用 `random.Random(seed * 1000 + seed_salt)`（见 WorkloadSpec），`--seed` 只覆盖 seed。
"""

from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from mini_infra.bench.spec.common.schema import ExecutionSpec, GroupSpec, PositiveInt, SuiteEngine, _Strict
from mini_infra.engine import BatchingMethod, CachingMethod

ReleaseScale = Literal["small", "medium", "large"]
RELEASE_SCALES: tuple[ReleaseScale, ...] = ("small", "medium", "large")


class DefaultModeSpec(_Strict):
    caching_method: CachingMethod
    batching_method: BatchingMethod


class ScaleSpec(_Strict):
    seed_salt: Annotated[int, Field(strict=True)]
    max_running_requests: PositiveInt | None = None
    workload: dict[str, Any]


class ReleaseSpec(_Strict):
    engine: SuiteEngine
    description: str = ""
    default_seed: Annotated[int, Field(strict=True)]
    default_mode: DefaultModeSpec
    execution: ExecutionSpec = ExecutionSpec()
    scales: dict[ReleaseScale, ScaleSpec]

    @field_validator("scales")
    @classmethod
    def _all_scales_present(cls, value: dict[ReleaseScale, ScaleSpec]) -> dict[ReleaseScale, ScaleSpec]:
        if set(value) != set(RELEASE_SCALES):
            raise ValueError(f"scales 必须恰好包含 {list(RELEASE_SCALES)}，实际为 {list(value)}")
        return value

    @model_validator(mode="after")
    def _check_engine(self) -> "ReleaseSpec":
        if self.execution.interleave:
            raise ValueError("release 负载不支持 execution.interleave")
        for scale, spec in self.scales.items():
            if self.engine == "scheduler" and spec.max_running_requests is None:
                raise ValueError(f"scheduler release 的 scales.{scale} 必须提供 max_running_requests")
            if self.engine == "lockstep" and spec.max_running_requests is not None:
                raise ValueError(f"lockstep release 的 scales.{scale} 不支持 max_running_requests")
        return self


def release_json_schema() -> dict[str, Any]:
    """导出供 IDE 补全用的 release 声明 JSON Schema（描述 YAML 文件形状）。

    与内部模型的差异：允许 `extends`；`required` 置空（可由 extends 继承）；`scales` 展开为 small / medium / large
    三个显式属性；每档 `workload` 含 ${var} 之外的随机分布等，只约束外形（单组字段含义见 `$defs.GroupSpec`，仅作参考）。
    """
    schema = ReleaseSpec.model_json_schema()
    schema["title"] = "mini-infra release workload"
    schema.pop("required", None)
    schema["properties"]["extends"] = {
        "type": "string",
        "description": "相对当前文件的基底 YAML 路径（映射深合并，列表整体替换）",
    }
    schema["properties"]["scales"] = {
        "type": "object",
        "properties": {scale: {"$ref": "#/$defs/ScaleSpec"} for scale in RELEASE_SCALES},
        "additionalProperties": False,
        "description": "三档标准负载，必须恰好包含 small / medium / large",
    }
    scale_schema = schema["$defs"]["ScaleSpec"]
    scale_schema["properties"]["workload"] = {
        "type": "object",
        "description": "单组简写（count / prompt_len / output_len / arrival_ms）或 groups；字段含义见 $defs.GroupSpec",
        "properties": {"groups": {"type": "array", "items": {"type": "object"}}},
    }
    group_schema = GroupSpec.model_json_schema()
    schema["$defs"].update(group_schema.pop("$defs", {}))
    schema["$defs"]["GroupSpec"] = group_schema
    return schema


__all__ = [
    "DefaultModeSpec",
    "RELEASE_SCALES",
    "ReleaseScale",
    "ReleaseSpec",
    "ScaleSpec",
    "release_json_schema",
]
