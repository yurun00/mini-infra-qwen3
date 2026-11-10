"""评测用例层（case）类型：用例身份键、对比规范与 Scheduler 最小单用例配置（纯 CPU，无 PyTorch 依赖）。

YAML suite 声明（bench.spec）与 Scheduler 执行层共同依赖本模块，故置于 bench 包顶层以避免包级循环 import。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.core.request import RequestSpec, make_request_specs
from mini_infra.core.types import collapse_uniform
from mini_infra.engine import BatchingMethod, CachingMethod, FeatureFlag, Mode

if TYPE_CHECKING:
    from mini_infra.bench.spec.suite.schema import ComparisonSpec

CaseDimensionValue = str | CachingMethod | BatchingMethod | frozenset[FeatureFlag] | int | PagedCacheConfig | None


class CaseDimension(str, Enum):
    """用例身份键（CaseKey）的正交特征维度枚举。"""

    WORKLOAD_ID = "workload_id"
    CACHING_METHOD = "caching_method"
    BATCHING_METHOD = "batching_method"
    FLAGS = "flags"
    MAX_RUNNING_REQUESTS = "max_running_requests"
    PAGED_CACHE_CONFIG = "paged_cache_config"


@dataclass(frozen=True)
class CaseKey:
    """用例全局唯一身份键：由工作负载与系统配置正交维度完全确定。"""

    workload_id: str
    mode: Mode
    max_running_requests: int
    paged_cache_config: PagedCacheConfig | None = None

    def value(self, dim: CaseDimension | str) -> CaseDimensionValue:
        d = dim if isinstance(dim, CaseDimension) else CaseDimension(dim)
        if d is CaseDimension.WORKLOAD_ID:
            return self.workload_id
        if d is CaseDimension.CACHING_METHOD:
            return self.mode.caching_method
        if d is CaseDimension.BATCHING_METHOD:
            return self.mode.batching_method
        if d is CaseDimension.FLAGS:
            return self.mode.flags
        if d is CaseDimension.MAX_RUNNING_REQUESTS:
            return self.max_running_requests
        if d is CaseDimension.PAGED_CACHE_CONFIG:
            return self.paged_cache_config
        raise ValueError(f"未知维度: {dim}")

    def project(self, dims: Iterable[CaseDimension | str]) -> tuple[CaseDimensionValue, ...]:
        return tuple(self.value(d) for d in dims)

    @property
    def display_name(self) -> str:
        parts = [f"[{self.workload_id}]", str(self.mode), f"mrr={self.max_running_requests}"]
        if self.paged_cache_config is not None:
            parts.append(f"block={self.paged_cache_config.block_size}")
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.display_name

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "workload_id": self.workload_id,
            "mode": str(self.mode),
            "max_running_requests": self.max_running_requests,
        }
        if self.paged_cache_config is not None:
            data["paged_cache_config"] = {
                "block_size": self.paged_cache_config.block_size,
                "num_blocks": self.paged_cache_config.num_blocks,
            }
        return data


class MetricKind(str, Enum):
    """指标对比相对统计模式。"""

    SPEEDUP = "speedup"  # baseline / candidate（如耗时/makespan 加速比，数值越小越好）
    GAIN = "gain"  # (candidate - baseline) / baseline（如吞吐量增益，数值越大越好）
    REDUCTION = "reduction"  # (baseline - candidate) / baseline（如显存/token 削减，数值越小越好）


@dataclass(frozen=True)
class ComparisonMetric:
    """指定参与对比的指标点分路径与期望统计模式。"""

    path: str
    kind: MetricKind
    name: str | None = None

    @property
    def display_name(self) -> str:
        return self.name or self.path.rsplit(".", 1)[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind.value,
            "name": self.display_name,
        }


@dataclass(frozen=True)
class Comparison:
    """Suite 级泛型对比配置（已解析为强类型，纯 CPU，零 PyTorch 依赖）。"""

    metrics: tuple[ComparisonMetric, ...]
    group_by: tuple[CaseDimension, ...]
    baseline: Mapping[CaseDimension, CaseDimensionValue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": [m.to_dict() for m in self.metrics],
            "group_by": [d.value for d in self.group_by],
            "baseline": {
                d.value if isinstance(d, CaseDimension) else str(d): v.value if hasattr(v, "value") else str(v)
                for d, v in self.baseline.items()
            },
        }

    @classmethod
    def from_spec(cls, spec: ComparisonSpec) -> Comparison:
        """从 ComparisonSpec 声明模型装配强类型的运行时 Comparison 对比配置。"""
        baseline: dict[CaseDimension, CaseDimensionValue] = {}
        for dim, raw in spec.baseline.items():
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
        return cls(
            metrics=tuple(ComparisonMetric(m.path, m.kind, m.name) for m in spec.metrics),
            group_by=tuple(spec.group_by),
            baseline=baseline,
        )


@dataclass(frozen=True)
class WorkloadProfile:
    """被测工作负载的轻量级请求数值画像（纯数值元数据，无 Token，纯 CPU）。

    作为用例身份、执行结果 (CaseResult)、统计报表与绘图分析的共享负载描述符。
    """

    workload_id: str
    num_total_requests: int
    prompt_lens: tuple[int, ...]
    output_budgets: tuple[int, ...]
    arrival_ms: tuple[float, ...]
    prompt_len: int | tuple[int, ...] | None = None
    max_new_tokens: int | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_total_requests <= 0:
            raise ValueError(f"num_total_requests 必须大于 0，实际为 {self.num_total_requests}")
        if len(self.prompt_lens) != self.num_total_requests:
            raise ValueError(
                f"prompt_lens 长度 ({len(self.prompt_lens)}) 与 num_total_requests ({self.num_total_requests}) 不一致"
            )
        if len(self.output_budgets) != self.num_total_requests:
            raise ValueError(
                f"output_budgets 长度 ({len(self.output_budgets)}) 与 "
                f"num_total_requests ({self.num_total_requests}) 不一致"
            )
        if len(self.arrival_ms) != self.num_total_requests:
            raise ValueError(
                f"arrival_ms 长度 ({len(self.arrival_ms)}) 与 num_total_requests ({self.num_total_requests}) 不一致"
            )
        if self.prompt_len is None:
            object.__setattr__(self, "prompt_len", collapse_uniform(self.prompt_lens))
        if self.max_new_tokens is None:
            object.__setattr__(self, "max_new_tokens", collapse_uniform(self.output_budgets))


@dataclass(frozen=True)
class RequestPayload:
    """已绑定具体 Token 序列的确定性执行载荷（Traffic Payload，纯 CPU，零 PyTorch 依赖）。"""

    workload_id: str
    specs: tuple[RequestSpec, ...]
    profile: WorkloadProfile = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.workload_id:
            raise ValueError("workload_id 不能为空")
        if not self.specs:
            raise ValueError("specs 不能为空")
        if self.profile is None:
            object.__setattr__(
                self,
                "profile",
                WorkloadProfile(
                    workload_id=self.workload_id,
                    num_total_requests=self.request_count,
                    prompt_lens=self.prompt_lengths,
                    output_budgets=self.output_budgets,
                    arrival_ms=self.arrival_times_ms,
                ),
            )

    @property
    def request_count(self) -> int:
        return len(self.specs)

    @property
    def prompt_lengths(self) -> tuple[int, ...]:
        return tuple(len(s.token_ids) for s in self.specs)

    @property
    def output_budgets(self) -> tuple[int, ...]:
        return tuple(s.max_new_tokens for s in self.specs)

    @property
    def arrival_times_ms(self) -> tuple[float, ...]:
        return tuple(s.arrival_ms for s in self.specs)

    @classmethod
    def from_profile(cls, profile: WorkloadProfile, vocab_size: int) -> RequestPayload:
        """把 WorkloadProfile 数值画像装配为具体合成 Token 序列的 RequestPayload。"""
        from mini_infra.bench.workloads import make_sequences

        specs = tuple(
            make_request_specs(
                make_sequences(profile.prompt_lens, vocab_size),
                profile.output_budgets,
                profile.arrival_ms,
            )
        )
        return cls(workload_id=profile.workload_id, specs=specs, profile=profile)


@dataclass(frozen=True)
class SchedulerCase:
    """Scheduler 评测最小单用例配置：包含输入负载与 System 维度。"""

    payload: RequestPayload
    mode: Mode
    max_running_requests: int
    paged_cache_config: PagedCacheConfig | None = None
    suite_name: str | None = None

    def __post_init__(self) -> None:
        if self.mode.caching_method is CachingMethod.PAGED and self.paged_cache_config is None:
            raise ValueError(f"Paged 模式 (mode={self.mode}) 必须提供 paged_cache_config")
        if self.mode.caching_method is not CachingMethod.PAGED and self.paged_cache_config is not None:
            raise ValueError(f"非 Paged 模式 (mode={self.mode}) 不得提供 paged_cache_config")

    @property
    def workload_id(self) -> str:
        return self.payload.workload_id

    @property
    def profile(self) -> WorkloadProfile:
        return self.payload.profile

    @property
    def key(self) -> CaseKey:
        return CaseKey(
            workload_id=self.workload_id,
            mode=self.mode,
            max_running_requests=self.max_running_requests,
            paged_cache_config=self.paged_cache_config,
        )

    def to_spec_dict(self) -> dict[str, Any]:
        """导出用例负载规格字典。"""
        data: dict[str, Any] = {
            "workload_id": self.workload_id,
            "mode": str(self.mode),
            "max_running_requests": self.max_running_requests,
            "request_count": self.payload.request_count,
            "prompt_lengths": list(self.payload.prompt_lengths),
            "output_budgets": list(self.payload.output_budgets),
            "arrival_times_ms": list(self.payload.arrival_times_ms),
        }
        if self.suite_name is not None:
            data["suite_name"] = self.suite_name
        if self.paged_cache_config is not None:
            data["paged_cache_config"] = {
                "block_size": self.paged_cache_config.block_size,
                "num_blocks": self.paged_cache_config.num_blocks,
            }
        return data


@dataclass(frozen=True)
class LockstepCase:
    """Lockstep 评测最小单用例配置：包含输入负载与静态批处理模式。"""

    payload: RequestPayload
    mode: Mode
    suite_name: str | None = None

    @property
    def workload_id(self) -> str:
        return self.payload.workload_id

    @property
    def profile(self) -> WorkloadProfile:
        return self.payload.profile

    @property
    def key(self) -> CaseKey:
        return CaseKey(
            workload_id=self.workload_id,
            mode=self.mode,
            max_running_requests=self.payload.request_count,
            paged_cache_config=None,
        )


__all__ = [
    "CaseDimension",
    "CaseDimensionValue",
    "CaseKey",
    "Comparison",
    "ComparisonMetric",
    "LockstepCase",
    "MetricKind",
    "RequestPayload",
    "SchedulerCase",
    "WorkloadProfile",
]
