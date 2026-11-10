"""评测用例层（case）类型：用例身份键、对比规范与 Scheduler 最小单用例配置（纯 CPU，无 PyTorch 依赖）。

YAML suite 声明（bench.spec）与 Scheduler 执行层共同依赖本模块，故置于 bench 包顶层以避免包级循环 import。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.core.request import RequestSpec, make_request_specs
from mini_infra.engine import BatchingMethod, CachingMethod, FeatureFlag, Mode

if TYPE_CHECKING:
    from mini_infra.bench.spec.suite.schema import ComparisonSpec

CaseDimensionValue = str | CachingMethod | BatchingMethod | frozenset[FeatureFlag] | int | None


class CaseDimension(str, Enum):
    """用例身份键（CaseKey）的正交特征维度枚举。

    对齐工业级基准测试（如 MLPerf / Prometheus 维度标签）规范：
    所有维度完全展平为一阶标量自变量，支持全局多维投影切片。
    """

    WORKLOAD_ID = "workload_id"
    CACHING_METHOD = "caching_method"
    BATCHING_METHOD = "batching_method"
    FLAGS = "flags"
    MAX_RUNNING_REQUESTS = "max_running_requests"
    BLOCK_SIZE = "block_size"
    NUM_BLOCKS = "num_blocks"


@dataclass(frozen=True, slots=True)
class CaseKey:
    """用例全局唯一身份键：所有正交维度完全展平为一阶标量。

    符合现代推理评测规范：
    1. 不可变哈希（frozen=True）且纯标量正交存储
    2. 字段与 CaseDimension 1:1 严格同构反射
    3. 规范化机器标识（canonical_id）与人类展示（display_name）分离
    4. 动态维度投影（project）与字典映射（as_dimension_map）
    """

    workload_id: str
    caching_method: CachingMethod
    batching_method: BatchingMethod
    max_running_requests: int
    flags: frozenset[FeatureFlag] = frozenset()
    block_size: int | None = None
    num_blocks: int | None = None

    @property
    def mode(self) -> Mode:
        """派生引擎运行所需的复合 Mode 对象。"""
        return Mode(
            caching_method=self.caching_method,
            batching_method=self.batching_method,
            flags=self.flags,
        )

    @property
    def paged_cache_config(self) -> PagedCacheConfig | None:
        """派生 Paged Cache 物理显存块配置对象。"""
        if self.caching_method is not CachingMethod.PAGED or self.block_size is None:
            return None
        return PagedCacheConfig(block_size=self.block_size, num_blocks=self.num_blocks)

    @classmethod
    def from_mode(
        cls,
        workload_id: str,
        mode: Mode,
        max_running_requests: int,
        paged_cache_config: PagedCacheConfig | None = None,
    ) -> CaseKey:
        """便利构造器：从 Mode 与 PagedCacheConfig 构造完全展平的 CaseKey。"""
        return cls(
            workload_id=workload_id,
            caching_method=mode.caching_method,
            batching_method=mode.batching_method,
            max_running_requests=max_running_requests,
            flags=mode.flags,
            block_size=paged_cache_config.block_size if paged_cache_config is not None else None,
            num_blocks=paged_cache_config.num_blocks if paged_cache_config is not None else None,
        )

    def value(self, dim: CaseDimension | str) -> CaseDimensionValue:
        """纯反射无分支提取正交维度标量值。"""
        d = dim if isinstance(dim, CaseDimension) else CaseDimension(dim)
        return getattr(self, d.value)

    def project(self, dims: Iterable[CaseDimension | str]) -> tuple[CaseDimensionValue, ...]:
        return tuple(self.value(d) for d in dims)

    def as_dimension_map(self) -> dict[CaseDimension, CaseDimensionValue]:
        """导出完整的正交特征标签映射（类似于指标系统的 Label-Set）。"""
        return {dim: self.value(dim) for dim in CaseDimension}

    @property
    def canonical_id(self) -> str:
        """机器友好的规范化唯一标识符（符合 MLPerf Slug 规范，无空格和特殊符号）。"""
        parts = [
            f"workload={self.workload_id}",
            f"cache={self.caching_method.value}",
            f"batch={self.batching_method.value}",
            f"mrr={self.max_running_requests}",
        ]
        if self.flags:
            flags_str = ",".join(sorted(f.value for f in self.flags))
            parts.append(f"flags={flags_str}")
        if self.block_size is not None:
            parts.append(f"block={self.block_size}")
        if self.num_blocks is not None:
            parts.append(f"num_blocks={self.num_blocks}")
        return "/".join(parts)

    @property
    def display_name(self) -> str:
        parts = [f"[{self.workload_id}]", str(self.mode), f"mrr={self.max_running_requests}"]
        if self.block_size is not None:
            blk_str = f"block={self.block_size}"
            if self.num_blocks is not None:
                blk_str += f"(n={self.num_blocks})"
            parts.append(blk_str)
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.display_name

    def to_dict(self) -> dict[str, Any]:
        """导出全展平自变量字段字典。"""
        data: dict[str, Any] = {
            "workload_id": self.workload_id,
            "caching_method": self.caching_method.value,
            "batching_method": self.batching_method.value,
            "max_running_requests": self.max_running_requests,
            "canonical_id": self.canonical_id,
        }
        if self.flags:
            data["flags"] = [f.value for f in self.flags]
        if self.block_size is not None:
            data["block_size"] = self.block_size
        if self.num_blocks is not None:
            data["num_blocks"] = self.num_blocks
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CaseKey:
        """从字典反向重构完全展平的 CaseKey。"""
        raw_flags = data.get("flags", ())
        return cls(
            workload_id=str(data["workload_id"]),
            caching_method=CachingMethod.from_str(str(data["caching_method"])),
            batching_method=BatchingMethod.from_str(str(data["batching_method"])),
            max_running_requests=int(data["max_running_requests"]),
            flags=frozenset(FeatureFlag(f) for f in raw_flags) if raw_flags else frozenset(),
            block_size=int(data["block_size"]) if data.get("block_size") is not None else None,
            num_blocks=int(data["num_blocks"]) if data.get("num_blocks") is not None else None,
        )


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
            elif dim is CaseDimension.BLOCK_SIZE:
                if not isinstance(raw, int):
                    raise ValueError("baseline.block_size 必须是整数")
                baseline[dim] = raw
            elif dim is CaseDimension.NUM_BLOCKS:
                if not isinstance(raw, int) and raw is not None:
                    raise ValueError("baseline.num_blocks 必须是整数或 null")
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

    def __post_init__(self) -> None:
        if not self.workload_id:
            raise ValueError("workload_id 不能为空")
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

    def to_payload(self, vocab_size: int) -> tuple[RequestSpec, ...]:
        """把数值画像按需装配为具体合成 Token 序列的 RequestSpec 载荷元组（即用即弃，函数局部变量）。"""
        from mini_infra.bench.synthetic import make_sequences

        return tuple(
            make_request_specs(
                make_sequences(self.prompt_lens, vocab_size),
                self.output_budgets,
                self.arrival_ms,
            )
        )

    @classmethod
    def from_specs(cls, workload_id: str, specs: Sequence[RequestSpec]) -> WorkloadProfile:
        """从已有的 RequestSpec 序列反向提取 WorkloadProfile 数值画像。"""
        return cls(
            workload_id=workload_id,
            num_total_requests=len(specs),
            prompt_lens=tuple(s.prompt_len for s in specs),
            output_budgets=tuple(s.max_new_tokens for s in specs),
            arrival_ms=tuple(s.arrival_ms for s in specs),
        )


@dataclass(frozen=True)
class SchedulerCase:
    """Scheduler 评测最小单用例配置：包含输入负载与 System 维度。"""

    profile: WorkloadProfile
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
        return self.profile.workload_id

    @property
    def key(self) -> CaseKey:
        return CaseKey.from_mode(
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
            "request_count": self.profile.num_total_requests,
            "prompt_lengths": list(self.profile.prompt_lens),
            "output_budgets": list(self.profile.output_budgets),
            "arrival_times_ms": list(self.profile.arrival_ms),
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

    profile: WorkloadProfile
    mode: Mode
    suite_name: str | None = None

    @property
    def workload_id(self) -> str:
        return self.profile.workload_id

    @property
    def key(self) -> CaseKey:
        return CaseKey.from_mode(
            workload_id=self.workload_id,
            mode=self.mode,
            max_running_requests=self.profile.num_total_requests,
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
    "SchedulerCase",
    "WorkloadProfile",
]
