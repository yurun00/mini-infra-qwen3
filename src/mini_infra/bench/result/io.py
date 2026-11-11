"""Benchmark 评测结果信封构建、校验与 JSON 持久化读写。"""

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import BaseModel, TypeAdapter

from mini_infra.bench.correctness import CorrectnessReport
from mini_infra.bench.efficiency import HardwarePeakSpecs, ModelCostProfile
from mini_infra.bench.result.schema import (
    SCHEMA_VERSION,
    AnyBenchmarkRecord,
    LockstepBenchmarkRecord,
    SchedulerBenchmarkRecord,
)

if TYPE_CHECKING:
    from mini_infra.bench.comparison import ComparisonResult
    from mini_infra.bench.lockstep.types import LockstepCaseResult
    from mini_infra.bench.scheduler.types import SchedulerCaseResult

_SCALE_ORDER = {"small": 0, "medium": 1, "large": 2}

_BENCHMARK_RECORD_ADAPTER: TypeAdapter[LockstepBenchmarkRecord | SchedulerBenchmarkRecord] = TypeAdapter(
    AnyBenchmarkRecord
)


def build_benchmark_record(
    args: Any,
    model: torch.nn.Module,
    mode_label: str,
    cases: Sequence["LockstepCaseResult | SchedulerCaseResult"],
    correctness: CorrectnessReport | dict[str, CorrectnessReport],
    env: dict[str, Any],
    peaks: HardwarePeakSpecs,
    cost: ModelCostProfile | None,
    git: dict[str, Any],
    warmup: int,
    repeats: int,
    suite_info: dict[str, Any] | None = None,
    engine: Literal["lockstep", "scheduler"] | None = None,
    comparisons: Sequence["ComparisonResult"] | None = None,
) -> LockstepBenchmarkRecord | SchedulerBenchmarkRecord:
    """统一构建 Lockstep 或 Scheduler 标准评测结果信封（由 Pydantic 校验并返回模型）。"""
    if engine is None:
        if cases and hasattr(cases[0], "schedule_metrics"):
            engine = "scheduler"
        else:
            engine = "lockstep"

    record_cases = [c.to_record_dict() for c in cases]
    peaks_dict = peaks.to_dict()
    cost_dict = cost.to_dict() if cost is not None else None

    if isinstance(correctness, dict):
        correctness_dict = {k: v.to_dict() for k, v in correctness.items()}
    else:
        correctness_dict = correctness.to_dict()

    try:
        dtype_str = str(next(model.parameters()).dtype)
    except StopIteration:
        dtype_str = "unknown"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "tag": args.tag,
        "release_feature": getattr(args, "release_feature", None),
        "mode": mode_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git,
        "env": env,
        "peak_specs": peaks_dict,
        "model_cost": cost_dict,
        "config": {
            "warmup": warmup,
            "repeats": repeats,
            "model": getattr(args, "model", None),
            "revision": getattr(args, "revision", None),
            "dtype": dtype_str,
            "tokenizer_included": False,
            "profiling_mode": str(args.profiling_mode) if getattr(args, "profiling_mode", None) else None,
        },
        "correctness": correctness_dict,
        "cases": record_cases,
    }
    if suite_info is not None:
        payload["suite"] = suite_info

    if engine == "scheduler":
        if comparisons is not None:
            comp_list = [c.to_dict() for c in comparisons]
        else:
            comp_list = []
        payload["comparisons"] = comp_list

    return _BENCHMARK_RECORD_ADAPTER.validate_python(payload)


def _check_single_case_constraint(
    rec: LockstepBenchmarkRecord | SchedulerBenchmarkRecord,
    source: str | Path,
) -> None:
    """校验 results/history 约束：单 Case，或同版本同模式下按 small/medium/large 各存 1 个 Case。"""
    if not rec.cases:
        raise ValueError(f"基准记录 {source} 不包含任何 case (single-case 约束)。")
    if len(rec.cases) == 1:
        return
    scales = [c.scale for c in rec.cases]
    modes = {c.mode for c in rec.cases}
    if (
        rec.suite is None
        and len(modes) == 1
        and all(s in ("small", "medium", "large") for s in scales)
        and len(set(scales)) == len(scales)
    ):
        return
    raise ValueError(
        f"基准记录 {source} 必须只包含 1 个 case（或同模式下按 small/medium/large 各 1 个 case，single-case 约束），"
        f"实际包含 {len(rec.cases)} 个 cases。"
    )


def validate_benchmark_record(
    data: Any,
    *,
    require_single_case: bool = False,
    source: str | Path = "<memory>",
) -> LockstepBenchmarkRecord | SchedulerBenchmarkRecord:
    """将 dict 或已解析对象校验为强类型 BenchmarkRecord，可选强制检查单 Case 约束。"""
    if isinstance(data, LockstepBenchmarkRecord | SchedulerBenchmarkRecord):
        rec = data
    else:
        rec = _BENCHMARK_RECORD_ADAPTER.validate_python(data)

    if require_single_case:
        _check_single_case_constraint(rec, source)
    return rec


def load_benchmark_record(
    path: Path,
    *,
    require_single_case: bool = False,
) -> LockstepBenchmarkRecord | SchedulerBenchmarkRecord:
    """从 JSON 文件加载并校验 BenchmarkRecord，可选强制检查单 Case 约束。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    raw_text = path.read_text(encoding="utf-8")
    rec = _BENCHMARK_RECORD_ADAPTER.validate_json(raw_text)
    if require_single_case:
        _check_single_case_constraint(rec, path)
    return rec


def save_benchmark_record(output: Path, record: BaseModel, is_dirty: bool = False) -> None:
    """将评测结果保存至指定 JSON 文件路径（若同 tag 下按 scale 分步运行，自动按 small/medium/large 归并）。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump(mode="json", exclude_none=True)
    new_cases = payload.get("cases", [])

    if output.exists() and len(new_cases) == 1 and new_cases[0].get("scale") in _SCALE_ORDER:
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except Exception:  # infra-legacy: 历史破损 JSON 降级为全新覆写
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("tag") == payload.get("tag")
            and existing.get("engine") == payload.get("engine")
            and existing.get("mode") == payload.get("mode")
        ):
            new_scale = new_cases[0]["scale"]
            merged_cases = [
                c for c in existing.get("cases", []) if c.get("scale") in _SCALE_ORDER and c.get("scale") != new_scale
            ]
            merged_cases.append(new_cases[0])
            merged_cases.sort(key=lambda c: _SCALE_ORDER.get(c.get("scale", ""), 99))
            payload["cases"] = merged_cases

    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"结果已写入 {output}" + ("（工作区有未提交改动）" if is_dirty else ""))


__all__ = [
    "build_benchmark_record",
    "load_benchmark_record",
    "save_benchmark_record",
    "validate_benchmark_record",
]
