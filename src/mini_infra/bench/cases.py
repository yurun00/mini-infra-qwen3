"""评测用例层（case）类型：用例身份键、对比规范与 Scheduler 最小单用例配置（纯 CPU，无 PyTorch 依赖）。

YAML suite 声明（bench.spec）与 Scheduler 执行层共同依赖本模块，故置于 bench 包顶层以避免包级循环 import。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, cast

from mini_infra.bench.format import format_lengths
from mini_infra.core import (
    DenseCacheConfig,
    KVCacheConfig,
    KVCacheKind,
    PagedCacheConfig,
    Request,
    kv_cache_config_from_str,
    make_requests,
)
from mini_infra.engine.config import BatchingMethod, EngineConfig, FeatureFlag

DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10


class ProfilingMode(str, Enum):
    """Profiler 性能剖析执行模式。"""

    DEDICATED_PASS = "dedicated-pass"
    ALL_REPEATS = "all-repeats"

    def __str__(self) -> str:
        return self.value


CaseDimensionValue = str | BatchingMethod | frozenset[FeatureFlag] | int | None


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

    @classmethod
    def from_str(cls, name: str) -> CaseDimension:
        """将字符串解析为强类型 CaseDimension。"""
        cleaned = name.strip()
        try:
            return cls(cleaned)
        except ValueError as err:
            raise ValueError(f"未知的 interleave 特征维度: {name}，可选: {[d.value for d in cls]}") from err

    @classmethod
    def from_interleave_args(
        cls,
        raw: Sequence[str] | None,
        default: tuple[CaseDimension, ...],
    ) -> tuple[CaseDimension, ...]:
        """将 CLI 字符串参数解析为强类型 CaseDimension 元组。"""
        if raw is None:
            return default
        if not raw or "none" in raw:
            return ()
        return tuple(cls.from_str(f) for f in raw)


@dataclass(frozen=True, slots=True)
class CaseKey:
    """用例全局唯一身份键：聚合 workload_id、caching_method、batching_method、max_running_requests 等全展平维度。

    符合现代推理评测规范：
    1. 不可变哈希（frozen=True）且纯标量正交存储
    2. 字段与 CaseDimension 1:1 严格同构反射（`getattr(self, d.value)`）
    3. 规范化机器标识（canonical_id）与人类展示（display_name）分离
    4. 动态维度投影（project）与字典映射（as_dimension_map）
    """

    workload_id: str
    caching_method: KVCacheKind = "dense-kv-cache"
    batching_method: BatchingMethod = BatchingMethod.NONE
    max_running_requests: int | None = None
    flags: frozenset[FeatureFlag] = frozenset()
    block_size: int | None = None
    num_blocks: int | None = None

    @property
    def engine_config(self) -> EngineConfig:
        """派生引擎运行所需的复合 EngineConfig 对象。"""
        return EngineConfig(
            kv_cache_config=kv_cache_config_from_str(
                self.caching_method,
                block_size=self.block_size,
                num_blocks=self.num_blocks,
            ),
            batching_method=self.batching_method,
            max_running_requests=self.max_running_requests,
            flags=self.flags,
        )

    def value(self, dim: CaseDimension | str) -> CaseDimensionValue:
        """基于正交维度枚举的零分支反射提取。"""
        d = dim if isinstance(dim, CaseDimension) else CaseDimension(dim)
        return getattr(self, d.value)

    def project(self, dims: Iterable[CaseDimension | str]) -> tuple[CaseDimensionValue, ...]:
        return tuple(self.value(d) for d in dims)

    def as_dimension_map(self) -> dict[CaseDimension, CaseDimensionValue]:
        """导出完整的正交特征标签映射（类似于指标系统的 Label-Set）。专供测试使用（Test-only）。"""
        return {dim: self.value(dim) for dim in CaseDimension}

    @property
    def canonical_id(self) -> str:
        """机器友好的规范化唯一标识符（符合 MLPerf Slug 规范，无空格和特殊符号）。"""
        parts = [
            f"workload={self.workload_id}",
            f"cache={self.caching_method}",
            f"batch={self.batching_method.value}",
        ]
        if self.max_running_requests is not None:
            parts.append(f"mrr={self.max_running_requests}")
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
        return f"[{self.workload_id}] | {self.engine_config}"

    @property
    def abbreviated_workload_id(self) -> str:
        """提取 workload_id 各词前 3 个字符缩写，保留 '_' 分隔符（如 long_short_stress -> lon_sho_str）。"""
        parts = [p for p in self.workload_id.replace("-", "_").split("_") if p]
        return "_".join(p if p.isdigit() else p[:3] for p in parts)

    @property
    def abbreviated_engine_config(self) -> str:
        """提取策略名各维度各词首字母缩写，保留 '+' 分隔符（如 dense-kv-cache+continuous-batching -> dkc+cb）。"""
        cache_label = "no-cache" if self.caching_method == "none" else self.caching_method
        parts = [*sorted(f.value for f in self.flags), cache_label]
        if self.batching_method != BatchingMethod.NONE:
            parts.append(self.batching_method.value)
        return "+".join("".join(w[0] for w in part.replace("_", "-").split("-") if w) for part in parts)

    @property
    def abbreviated_display_name(self) -> str:
        """提取用例键的缩写显示名（如 [vol] | dkc+cb | mrr=16 或 [lss] | pkc+cb | mrr=16 | blk-sz=16）。"""
        parts = [
            f"[{self.abbreviated_workload_id}]",
            self.abbreviated_engine_config,
        ]
        if self.max_running_requests is not None:
            parts.append(f"mrr={self.max_running_requests}")
        if self.block_size is not None:
            parts.append(f"blk-sz={self.block_size}")
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.display_name

    def to_dict(self) -> dict[str, Any]:
        """导出全展平自变量字段字典。"""
        data: dict[str, Any] = {
            "workload_id": self.workload_id,
            "caching_method": self.caching_method,
            "batching_method": self.batching_method.value,
            "canonical_id": self.canonical_id,
        }
        if self.max_running_requests is not None:
            data["max_running_requests"] = self.max_running_requests
        if self.flags:
            data["flags"] = [f.value for f in self.flags]
        if self.block_size is not None:
            data["block_size"] = self.block_size
        if self.num_blocks is not None:
            data["num_blocks"] = self.num_blocks
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CaseKey:
        """从字典反向重构 CaseKey。专供测试使用（Test-only）。"""
        raw_flags = data.get("flags", ())
        return cls(
            workload_id=str(data["workload_id"]),
            caching_method=cast(KVCacheKind, str(data["caching_method"])),
            batching_method=BatchingMethod.from_str(str(data["batching_method"])),
            max_running_requests=(
                int(data["max_running_requests"]) if data.get("max_running_requests") is not None else None
            ),
            flags=frozenset(FeatureFlag(f) for f in raw_flags) if raw_flags else frozenset(),
            block_size=int(data["block_size"]) if data.get("block_size") is not None else None,
            num_blocks=int(data["num_blocks"]) if data.get("num_blocks") is not None else None,
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

    def to_payload(self, vocab_size: int) -> tuple[Request, ...]:
        """把数值画像按需装配为具体合成 Token 序列的 Request 载荷元组（即用即弃，函数局部变量）。"""
        from mini_infra.bench.synthetic import make_sequences

        return tuple(
            make_requests(
                make_sequences(self.prompt_lens, vocab_size),
                self.output_budgets,
                self.arrival_ms,
            )
        )

    @classmethod
    def from_requests(cls, workload_id: str, requests: Sequence[Request]) -> WorkloadProfile:
        """从已有的 Request 序列反向提取 WorkloadProfile 数值画像。专供测试使用（Test-only）。"""
        return cls(
            workload_id=workload_id,
            num_total_requests=len(requests),
            prompt_lens=tuple(r.prompt_len for r in requests),
            output_budgets=tuple(r.max_new_tokens for r in requests),
            arrival_ms=tuple(r.arrival_ms for r in requests),
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """全仓统一的评测最小单用例执行实体：包含输入负载与系统执行维度（纯 CPU，解耦具体词表与 Tensor）。"""

    profile: WorkloadProfile
    batching_method: BatchingMethod = BatchingMethod.NONE
    kv_cache_config: KVCacheConfig = DenseCacheConfig()
    max_running_requests: int | None = None
    flags: frozenset[FeatureFlag] = frozenset()
    suite_name: str | None = None
    engine: Literal["scheduler", "lockstep"] = "scheduler"

    def __post_init__(self) -> None:
        if self.max_running_requests is None and self.engine == "scheduler":
            object.__setattr__(self, "max_running_requests", self.profile.num_total_requests)
        elif self.max_running_requests is not None and self.max_running_requests <= 0:
            raise ValueError(f"max_running_requests 必须大于 0，实际为 {self.max_running_requests}")

    @property
    def engine_config(self) -> EngineConfig:
        return EngineConfig(
            kv_cache_config=self.kv_cache_config,
            batching_method=self.batching_method,
            max_running_requests=self.max_running_requests,
            flags=self.flags,
        )

    @property
    def workload_id(self) -> str:
        return self.profile.workload_id

    @property
    def key(self) -> CaseKey:
        paged = self.kv_cache_config if isinstance(self.kv_cache_config, PagedCacheConfig) else None
        return CaseKey(
            workload_id=self.workload_id,
            caching_method=self.kv_cache_config.kind,
            batching_method=self.batching_method,
            max_running_requests=self.max_running_requests,
            flags=self.flags,
            block_size=paged.block_size if paged is not None else None,
            num_blocks=paged.num_blocks if paged is not None else None,
        )

    def format_summary(self) -> str:
        """格式化单用例配置摘要（供 `bench suites show` 预览）。"""
        if self.engine == "scheduler":
            return str(self.key)
        return (
            f"[{self.workload_id}] | {self.engine_config} | batch={self.profile.num_total_requests} "
            f"prompt={format_lengths(self.profile.prompt_lens)} "
            f"max_new_tokens={format_lengths(self.profile.output_budgets)}"
        )


__all__ = [
    "BenchmarkCase",
    "CaseDimension",
    "CaseDimensionValue",
    "CaseKey",
    "DEFAULT_REPEATS",
    "DEFAULT_WARMUP",
    "ProfilingMode",
    "WorkloadProfile",
]
