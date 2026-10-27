"""性能回归分析与比较内核（Regression Analysis & Verification）。

提供轻量基准触发、历史记录对齐索引与中位数劣化判断算法，供质量门禁与 CI/CLI 调用。
"""

import subprocess
import sys
from typing import Any

from mini_infra.bench.common import BENCH_ARTIFACTS_DIR, REPO_ROOT

DEFAULT_REFERENCE = BENCH_ARTIFACTS_DIR / "data" / "reference.json"
# *.local.json 已被 .gitignore 忽略
DEFAULT_CANDIDATE = BENCH_ARTIFACTS_DIR / "candidate.local.json"

# 轻量 case：一个短、一个中等长度，约半分钟内完成
LIGHT_ARGS = ["--prompt-lens", "32", "512", "--max-new-tokens", "32", "--warmup", "2", "--repeats", "7"]
METRICS = [("ttft_ms", "TTFT"), ("tpot_ms", "TPOT")]


def run_light_benchmark(output: Any, tag: str) -> int:
    """运行轻量 benchmark 评测并生成临时候选结果 JSON。"""
    cmd = [
        sys.executable,
        "-m",
        "mini_infra.cli",
        "bench",
        "batch",
        "case",
        "--tag",
        tag,
        "--output",
        str(output),
        *LIGHT_ARGS,
    ]
    print("运行轻量 benchmark:", " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def index(record: dict[str, Any]) -> dict[tuple[int, Any, Any], dict[str, Any]]:
    """以 (batch_size, prompt_len, max_new_tokens) 为三元组 key 索引用例结果。"""
    return {
        (
            w["batch_size"],
            tuple(w["prompt_len"]) if isinstance(w["prompt_len"], list) else w["prompt_len"],
            w.get("max_new_tokens", w.get("output_len")),
        ): w
        for w in record["cases"]
    }


def compare(candidate: dict[str, Any], reference: dict[str, Any], threshold: float) -> tuple[bool | None, list[str]]:
    """比对当前与参考结果，返回 (是否回归, 报告行)。延迟类指标：越大越差。"""
    cand, ref = index(candidate), index(reference)
    common = sorted(set(cand) & set(ref), key=lambda k: (k[0], str(k[1]), k[2]))
    if not common:
        return None, ["没有可比较的共同 case"]

    regressed, lines = False, []
    lines.append(f"{'case (bs/prompt/out)':<26}{'指标':<6}{'参考(ms)':>10}{'当前(ms)':>10}{'变化':>9}  结果")
    for key in common:
        for metric, name in METRICS:
            r, c = ref[key][metric]["median"], cand[key][metric]["median"]
            change = (c - r) / r if r else 0.0
            # 噪声保护：变化需同时超过阈值和参考 std 的 3 倍，避免噪声误报
            noise = 3 * max(ref[key][metric]["std"], cand[key][metric]["std"])
            bad = change > threshold and (c - r) > noise
            regressed |= bad
            status = "❌ 回归" if bad else ("✅ 提升" if change < -threshold else "ok")
            lines.append(f"{'/'.join(map(str, key)):<26}{name:<6}{r:>10.2f}{c:>10.2f}{change:>+9.1%}  {status}")
    return regressed, lines


__all__ = [
    "DEFAULT_CANDIDATE",
    "DEFAULT_REFERENCE",
    "LIGHT_ARGS",
    "METRICS",
    "compare",
    "index",
    "run_light_benchmark",
]
