"""Scheduler 用例层（case）类型：用例身份键、对比规范与最小单用例配置（纯 CPU，无 PyTorch 依赖）。"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.core.request import RequestSpec
from mini_infra.engine import BatchingMethod, CachingMethod, FeatureFlag, Mode

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
class ComparisonSpec:
    """Suite 级泛型对比规范配置。"""

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


FIXED_VS_CONTINUOUS_COMPARISON_METRICS: tuple[ComparisonMetric, ...] = (
    ComparisonMetric("schedule_metrics.makespan_ms", MetricKind.SPEEDUP, "makespan"),
)

DENSE_VS_PAGED_COMPARISON_METRICS: tuple[ComparisonMetric, ...] = (
    ComparisonMetric("schedule_metrics.makespan_ms", MetricKind.SPEEDUP, "makespan"),
    ComparisonMetric("metrics.throughput_tok_s", MetricKind.GAIN, "throughput"),
    ComparisonMetric("memory.kv_cache.accessed_kv_cache_tokens", MetricKind.REDUCTION, "KV-read"),
    ComparisonMetric("memory.kv_cache.peak_kv_cache_pool_mb", MetricKind.REDUCTION, "pool"),
)


@dataclass(frozen=True)
class SchedulerSuiteCase:
    """Scheduler 评测最小单用例配置：包含用例身份键（CaseKey）对应的全量维度与请求规格。"""

    suite_name: str
    mode: Mode
    max_running_requests: int
    specs: tuple[RequestSpec, ...]
    workload_id: str
    paged_cache_config: PagedCacheConfig | None = None

    def __post_init__(self) -> None:
        if not self.workload_id:
            raise ValueError("workload_id 不能为空")
        if self.mode.caching_method is CachingMethod.PAGED and self.paged_cache_config is None:
            raise ValueError(f"Paged 模式 (mode={self.mode}) 必须提供 paged_cache_config")
        if self.mode.caching_method is not CachingMethod.PAGED and self.paged_cache_config is not None:
            raise ValueError(f"非 Paged 模式 (mode={self.mode}) 不得提供 paged_cache_config")

    @property
    def key(self) -> CaseKey:
        return CaseKey(
            workload_id=self.workload_id,
            mode=self.mode,
            max_running_requests=self.max_running_requests,
            paged_cache_config=self.paged_cache_config,
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

    def to_spec_dict(self) -> dict[str, Any]:
        """导出用例负载规格字典。"""
        data: dict[str, Any] = {
            "suite_name": self.suite_name,
            "workload_id": self.workload_id,
            "mode": str(self.mode),
            "max_running_requests": self.max_running_requests,
            "request_count": self.request_count,
            "prompt_lengths": list(self.prompt_lengths),
            "output_budgets": list(self.output_budgets),
            "arrival_times_ms": list(self.arrival_times_ms),
        }
        if self.paged_cache_config is not None:
            data["paged_cache_config"] = {
                "block_size": self.paged_cache_config.block_size,
                "num_blocks": self.paged_cache_config.num_blocks,
            }
        return data


__all__ = [
    "DENSE_VS_PAGED_COMPARISON_METRICS",
    "FIXED_VS_CONTINUOUS_COMPARISON_METRICS",
    "CaseDimension",
    "CaseDimensionValue",
    "CaseKey",
    "ComparisonMetric",
    "ComparisonSpec",
    "MetricKind",
    "SchedulerSuiteCase",
]
