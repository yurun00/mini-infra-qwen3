"""Suite 级跨用例基准指标对比规约（Comparison）与相对统计计算（ComparisonResult）（纯 CPU，无 PyTorch 依赖）。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from mini_infra.bench.cases import CaseDimension, CaseDimensionValue, CaseKey
from mini_infra.core import parse_kv_cache_kind
from mini_infra.engine import BatchingMethod

if TYPE_CHECKING:
    from mini_infra.bench.spec.suite.schema import ComparisonSpec


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
    def _display_name(self) -> str:
        return self.name or self.path.rsplit(".", 1)[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind.value,
            "name": self._display_name,
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
            "baseline": {d.value: v.value if hasattr(v, "value") else str(v) for d, v in self.baseline.items()},
        }

    @classmethod
    def from_spec(cls, spec: ComparisonSpec) -> Comparison:
        """从 ComparisonSpec 声明模型装配强类型的运行时 Comparison 对比配置。"""
        baseline: dict[CaseDimension, CaseDimensionValue] = {}
        for dim, raw in spec.baseline.items():
            if dim is CaseDimension.BATCHING_METHOD:
                baseline[dim] = BatchingMethod.from_str(str(raw))
            elif dim is CaseDimension.CACHING_METHOD:
                baseline[dim] = parse_kv_cache_kind(str(raw))
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
class ComparisonResult:
    """通用基准指标相对对比结果项。"""

    baseline_key: CaseKey
    candidate_key: CaseKey
    metric_path: str
    metric_name: str
    kind: MetricKind
    baseline_value: float
    candidate_value: float
    ratio: float
    relative_delta: float

    @property
    def _speedup(self) -> float:
        """baseline / candidate（针对越小越好的指标计算加速比，如 makespan）。"""
        return self.baseline_value / self.candidate_value if self.candidate_value > 0 else 1.0

    @property
    def _gain(self) -> float:
        """(candidate - baseline) / baseline（针对越大越好的指标计算相对增益，如吞吐量）。"""
        return (self.candidate_value - self.baseline_value) / self.baseline_value if self.baseline_value > 0 else 0.0

    @property
    def _reduction(self) -> float:
        """(baseline - candidate) / baseline（针对越小越好的指标计算相对削减比，如显存/Token）。"""
        return (self.baseline_value - self.candidate_value) / self.baseline_value if self.baseline_value > 0 else 0.0

    def _formatted_value(self) -> str:
        if self.kind == MetricKind.SPEEDUP:
            return f"speedup {self._speedup:.2f}x"
        if self.kind == MetricKind.GAIN:
            sign = "+" if self._gain >= 0 else ""
            return f"{self.metric_name} {sign}{self._gain * 100:.1f}%"
        if self.kind == MetricKind.REDUCTION:
            sign = "-" if self._reduction >= 0 else "+"
            return f"{self.metric_name} {sign}{abs(self._reduction) * 100:.1f}%"
        return f"{self.ratio:.4f}"

    def format_summary(self) -> str:
        """格式化单条基准对比摘要行。"""
        wid_abbr = self.candidate_key.abbreviated_workload_id
        cand_info = self.candidate_key.abbreviated_engine_config
        if self.candidate_key.block_size is not None:
            cand_info += f" (block={self.candidate_key.block_size})"
        return f"  [{wid_abbr}] {self.metric_name} ({cand_info} vs baseline): {self._formatted_value()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_key": self.baseline_key.to_dict(),
            "candidate_key": self.candidate_key.to_dict(),
            "metric_path": self.metric_path,
            "metric_name": self.metric_name,
            "kind": self.kind.value,
            "baseline_value": round(self.baseline_value, 4),
            "candidate_value": round(self.candidate_value, 4),
            "ratio": round(self.ratio, 4),
            "relative_delta": round(self.relative_delta, 4),
            "speedup": round(self._speedup, 4),
            "gain": round(self._gain, 4),
            "reduction": round(self._reduction, 4),
            "formatted": self._formatted_value(),
        }


def _extract_metric_by_path(obj: Any, path: str) -> float:
    """沿点分路径从对象中提取目标标量浮点值。"""
    curr = obj
    for segment in path.split("."):
        if not hasattr(curr, segment):
            raise AttributeError(f"{type(curr).__name__} 不包含属性 '{segment}'（完整路径: '{path}'）")
        curr = getattr(curr, segment)
    if hasattr(curr, "mean"):
        val = curr.mean
    elif isinstance(curr, (int, float)):
        val = curr
    else:
        raise TypeError(f"指标 '{path}' 提取得到不可转换为数值的对象: {curr!r} ({type(curr)})")
    return float(val)


def compute_comparisons(
    case_results: Sequence[Any],
    comparison: Comparison,
) -> list[ComparisonResult]:
    """根据 Comparison 将 case_results 分组，在每组中按 baseline 规则选出基线并计算指标对比。"""
    groups: dict[tuple[CaseDimensionValue, ...], list[Any]] = {}
    for r in case_results:
        g_key = r.key.project(comparison.group_by)
        groups.setdefault(g_key, []).append(r)

    comparisons: list[ComparisonResult] = []
    for g_key, group in groups.items():
        matching_baselines = [
            r for r in group if all(r.key.value(dim) == expected for dim, expected in comparison.baseline.items())
        ]
        if len(matching_baselines) == 0:
            raise ValueError(
                f"分组 {g_key} 内未找到符合 baseline 约束 {comparison.baseline} 的基线用例。"
                f"组内用例 keys: {[r.key.display_name for r in group]}"
            )
        if len(matching_baselines) > 1:
            raise ValueError(
                f"分组 {g_key} 内找到多个符合 baseline 约束 {comparison.baseline} 的候选用例: "
                f"{[r.key.display_name for r in matching_baselines]}"
            )
        baseline_res = matching_baselines[0]

        for candidate_res in group:
            if candidate_res.key == baseline_res.key:
                continue
            for metric in comparison.metrics:
                b_val = _extract_metric_by_path(baseline_res, metric.path)
                c_val = _extract_metric_by_path(candidate_res, metric.path)
                ratio = c_val / b_val if b_val != 0 else 1.0
                rel_delta = (c_val - b_val) / b_val if b_val != 0 else 0.0

                comparisons.append(
                    ComparisonResult(
                        baseline_key=baseline_res.key,
                        candidate_key=candidate_res.key,
                        metric_path=metric.path,
                        metric_name=metric._display_name,
                        kind=metric.kind,
                        baseline_value=b_val,
                        candidate_value=c_val,
                        ratio=ratio,
                        relative_delta=rel_delta,
                    )
                )
    return comparisons


__all__ = [
    "Comparison",
    "ComparisonMetric",
    "ComparisonResult",
    "MetricKind",
    "compute_comparisons",
]
