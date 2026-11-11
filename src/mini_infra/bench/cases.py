"""评测用例层（case）类型：用例身份键、对比规范与 Scheduler 最小单用例配置（纯 CPU，无 PyTorch 依赖）。

YAML suite 声明（bench.spec）与 Scheduler 执行层共同依赖本模块，故置于 bench 包顶层以避免包级循环 import。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from mini_infra.bench.format import format_lengths
from mini_infra.core.block_manager import PagedCacheConfig
from mini_infra.core.request import Request, make_requests
from mini_infra.engine.mode import BatchingMethod, CachingMethod, FeatureFlag, Mode

DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10


class ProfilingMode(str, Enum):
    """Profiler 性能剖析执行模式。"""

    DEDICATED_PASS = "dedicated-pass"
    ALL_REPEATS = "all-repeats"

    def __str__(self) -> str:
        return self.value


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

    @property
    def abbreviated_workload_id(self) -> str:
        """提取 workload_id 各词前 3 个字符缩写，保留 '_' 分隔符（如 long_short_stress -> lon_sho_str）。"""
        parts = [p for p in self.workload_id.replace("-", "_").split("_") if p]
        return "_".join(p if p.isdigit() else p[:3] for p in parts)

    @property
    def abbreviated_mode(self) -> str:
        """提取 mode 各维度各词首字母缩写，保留 '+' 分隔符（如 dense-kv-cache+continuous-batching -> dkc+cb）。"""
        return "+".join(
            "".join(w[0] for w in part.replace("_", "-").split("-") if w) for part in str(self.mode).split("+")
        )

    @property
    def abbreviated_display_name(self) -> str:
        """提取用例键的缩写显示名（如 [vol] | dkc+cb | mrr=16 或 [lss] | pkc+cb | mrr=16 | blk-sz=16）。"""
        parts = [
            f"[{self.abbreviated_workload_id}]",
            self.abbreviated_mode,
            f"mrr={self.max_running_requests}",
        ]
        if self.block_size is not None:
            parts.append(f"blk-sz={self.block_size}")
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
        """从已有的 Request 序列反向提取 WorkloadProfile 数值画像。"""
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
    mode: Mode
    max_running_requests: int | None = None
    paged_cache_config: PagedCacheConfig | None = None
    suite_name: str | None = None
    engine: Literal["scheduler", "lockstep"] = "scheduler"

    def __post_init__(self) -> None:
        if self.max_running_requests is None:
            object.__setattr__(self, "max_running_requests", self.profile.num_total_requests)
        elif self.max_running_requests <= 0:
            raise ValueError(f"max_running_requests 必须大于 0，实际为 {self.max_running_requests}")

        if self.mode.caching_method is CachingMethod.PAGED and self.paged_cache_config is None:
            raise ValueError(f"Paged 模式 (mode={self.mode}) 必须提供 paged_cache_config")
        if self.mode.caching_method is not CachingMethod.PAGED and self.paged_cache_config is not None:
            raise ValueError(f"非 Paged 模式 (mode={self.mode}) 不得提供 paged_cache_config")

    @property
    def workload_id(self) -> str:
        return self.profile.workload_id

    @property
    def key(self) -> CaseKey:
        assert self.max_running_requests is not None
        return CaseKey.from_mode(
            workload_id=self.workload_id,
            mode=self.mode,
            max_running_requests=self.max_running_requests,
            paged_cache_config=self.paged_cache_config,
        )

    def format_summary(self) -> str:
        """格式化单用例配置摘要（供 `bench suites show` 预览）。"""
        if self.engine == "scheduler":
            return str(self.key)
        return (
            f"[{self.workload_id}] | {self.mode} | batch={self.profile.num_total_requests} "
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
