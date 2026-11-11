"""基准评测目录与产物路径常量（零第三方与设备依赖）。"""

import os
from pathlib import Path

# 项目物理仓库根目录
REPO_ROOT = Path(__file__).resolve().parents[3]

# 评测产物与数据根目录（默认锚定至仓库根目录 benchmarks/，支持环境变量覆盖）
_DEFAULT_BENCH_DIR = REPO_ROOT / "benchmarks"
BENCH_ARTIFACTS_DIR = Path(os.environ.get("MINI_INFRA_BENCH_DIR", str(_DEFAULT_BENCH_DIR))).resolve()
RESULTS_DIR = BENCH_ARTIFACTS_DIR / "results"
RESULT_HISTORY_DIR = RESULTS_DIR / "history"
LOCKSTEP_RESULT_HISTORY_DIR = RESULT_HISTORY_DIR / "lockstep"
SCHEDULER_RESULT_HISTORY_DIR = RESULT_HISTORY_DIR / "scheduler"
CASE_DIR = BENCH_ARTIFACTS_DIR / "case"
SUITES_DIR = RESULTS_DIR / "suites"
SPECS_DIR = BENCH_ARTIFACTS_DIR / "specs"
REPORTS_DIR = BENCH_ARTIFACTS_DIR / "reports"
LOCKSTEP_REPORTS_DIR = REPORTS_DIR / "lockstep"
SCHEDULER_REPORTS_DIR = REPORTS_DIR / "scheduler"
GOLDEN_OUTPUT_PATH = BENCH_ARTIFACTS_DIR / "golden" / "golden_output_tokens.json"

__all__ = [
    "BENCH_ARTIFACTS_DIR",
    "CASE_DIR",
    "GOLDEN_OUTPUT_PATH",
    "LOCKSTEP_REPORTS_DIR",
    "LOCKSTEP_RESULT_HISTORY_DIR",
    "REPORTS_DIR",
    "REPO_ROOT",
    "RESULTS_DIR",
    "RESULT_HISTORY_DIR",
    "SCHEDULER_REPORTS_DIR",
    "SCHEDULER_RESULT_HISTORY_DIR",
    "SPECS_DIR",
    "SUITES_DIR",
]
