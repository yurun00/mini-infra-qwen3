"""Lockstep 评测套件：由 `benchmarks/specs/lockstep/*.yaml` 声明，这里只负责把声明绑定到 LockstepCase。

套件的全部配置（被测系统、请求负载、sweep、检查）都在 YAML 中；新增 suite 只需新增一个 YAML 文件：

    benchmarks/specs/lockstep/<name>.yaml         # name 必须与文件名一致

run_suite.py 不关心 suite 的内部细节，只按 `suite.cases()` 逐个执行并汇总结果：

    mini-infra bench sweep --engine lockstep --tag dev --suite serial-vs-fixed-batch-throughput

当前内置：
    serial-vs-fixed-batch-throughput   对每个 batch size，分别用 serial（逐请求串行）与 fixed-batching
                                       两种路径，在固定长度 prompt、固定输出长度、忽略 EOS 的条件下测量吞吐与显存。
    uniform-vs-variable-prompt-padding 固定 batch=4 与输出长度，对比等长 prompt 与阶梯不等长 prompt，
                                       测量 Padding 有效率、Prefill 耗时与有效 Prompt 吞吐。
    uniform-vs-variable-budget-slots   固定 batch=4 与 Prompt 长度，对比等长预算与阶梯不等长预算，
                                       测量 batch slot 利用率、Decode 耗时与端到端有效吞吐。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_infra.bench.cases import LockstepCase
from mini_infra.bench.spec import (
    CaseBlueprint,
    LoadedSuiteSpec,
    discover_suite_specs,
    get_suite_spec,
    load_suite_spec,
)
from mini_infra.bench.workloads import Workload
from mini_infra.engine import Mode


@dataclass(frozen=True)
class LockstepSuite:
    """一个已加载的 Lockstep 评测套件（YAML 声明 + 展开后的用例蓝图）。"""

    loaded: LoadedSuiteSpec
    blueprints: tuple[CaseBlueprint, ...]

    @property
    def name(self) -> str:
        return self.loaded.spec.name

    @property
    def warmup(self) -> int:
        return self.loaded.spec.execution.warmup

    @property
    def repeats(self) -> int:
        return self.loaded.spec.execution.repeats

    def modes(self) -> list[Mode]:
        """按展开顺序去重后的运行模式列表。"""
        return list(dict.fromkeys(bp.mode for bp in self.blueprints))

    def cases(self) -> list[LockstepCase]:
        return [
            LockstepCase(
                workload=Workload(
                    num_total_requests=bp.workload.num_requests,
                    prompt_len=bp.workload.prompt_len,
                    max_new_tokens=bp.workload.max_new_tokens,
                    workload_id=bp.workload.workload_id,
                ),
                mode=bp.mode,
                suite_name=self.name,
            )
            for bp in self.blueprints
        ]

    def to_dict(self) -> dict[str, Any]:
        """写入结果 JSON `suite` 块：合并后的完整 YAML 声明及其内容哈希。"""
        return self.loaded.info()


def suite_from_loaded(loaded: LoadedSuiteSpec) -> LockstepSuite:
    return LockstepSuite(loaded=loaded, blueprints=loaded.blueprints)


def load_lockstep_suite(path: Path, overrides: Mapping[str, Any] | None = None) -> LockstepSuite:
    """从任意 YAML 路径加载 Lockstep 套件。"""
    loaded = load_suite_spec(path, overrides)
    if loaded.spec.engine != "lockstep":
        raise ValueError(f"{path}: engine 必须是 lockstep，实际为 {loaded.spec.engine}")
    return suite_from_loaded(loaded)


def list_lockstep_suite_names() -> list[str]:
    return sorted(discover_suite_specs("lockstep"))


def get_lockstep_suite(name: str, overrides: Mapping[str, Any] | None = None) -> LockstepSuite:
    return suite_from_loaded(get_suite_spec("lockstep", name, overrides=overrides))


def get_all_lockstep_suites() -> Sequence[LockstepSuite]:
    return [get_lockstep_suite(name) for name in list_lockstep_suite_names()]


__all__ = [
    "LockstepCase",
    "LockstepSuite",
    "get_all_lockstep_suites",
    "get_lockstep_suite",
    "list_lockstep_suite_names",
    "load_lockstep_suite",
    "suite_from_loaded",
]
