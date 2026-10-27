"""基准测试共享基础设施模块：环境探测、算力利用率计算、正确性门禁、CLI公共参数与结果落盘。"""

import argparse
import functools
import json
import os
import platform
import re
import subprocess
import time
import warnings
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from mini_infra.bench.batch.types import CaseBenchmarkResult
from mini_infra.engine import generate
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)

DEFAULT_WARMUP = 3
DEFAULT_REPEATS = 10


class ProfileMode(str, Enum):
    """Profiler 性能剖析执行模式。"""

    EXTRA_PROFILE = "extra-profile"
    FULL_PROFILE = "full-profile"

    def __str__(self) -> str:
        return self.value


# 评测产物与数据根目录（纯数据/结果/报告资产，不包含任何 Python 源码）
BENCH_ARTIFACTS_DIR = Path(os.environ.get("MINI_INFRA_BENCH_DIR", "benchmarks")).resolve()
RESULT_HISTORY_DIR = BENCH_ARTIFACTS_DIR / "result-history"
SUITES_DIR = BENCH_ARTIFACTS_DIR / "suites"
REPORTS_DIR = BENCH_ARTIFACTS_DIR / "reports"
GOLDEN_OUTPUT_PATH = BENCH_ARTIFACTS_DIR / "data" / "golden_output_tokens.json"
GOLDEN_PROMPT = "请用一句话介绍 KV Cache。"
GOLDEN_MAX_OUTPUT_TOKENS = 32
SCHEMA_VERSION = 1


GPU_PEAK_SPECS = {
    "L4": (300.0, 121.0),
    "T4": (320.0, 65.0),
    "A10": (600.0, 125.0),
    "A100": (1555.0, 312.0),
    "H100": (3350.0, 989.0),
    "L40S": (864.0, 362.0),
    "4090": (1008.0, 165.0),
    "3090": (936.0, 71.0),
}


def _match_gpu_spec(gpu_name: str) -> tuple[float | None, float | None]:
    """按词边界匹配 GPU 型号，优先最长 key：避免 `A10` 命中 `A100`、`L4` 命中 `L40S`。"""
    best_key: str | None = None
    for key in GPU_PEAK_SPECS:
        if re.search(rf"(?<![0-9A-Za-z]){re.escape(key)}(?![0-9])", gpu_name):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    return GPU_PEAK_SPECS[best_key] if best_key else (None, None)


def resolve_peak_specs(
    gpu_name: str,
    bandwidth_gbs: float | None = None,
    tflops: float | None = None,
) -> dict[str, float | None]:
    """推断或覆盖 GPU 峰值显存带宽 (GB/s) 与 FP16 峰值算力 (TFLOPS)。"""
    spec = _match_gpu_spec(gpu_name)
    return {"bandwidth_gbs": bandwidth_gbs or spec[0], "tflops": tflops or spec[1]}


def require_cuda() -> torch.device:
    """要求存在可用 NVIDIA GPU，返回 CUDA 设备；各基准入口共用。"""
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 NVIDIA GPU")
    return torch.device("cuda")


def format_lengths(value: int | Sequence[int]) -> str:
    """将标量或逐请求长度序列格式化为 `16/32/64` 形式的展示串。"""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return "/".join(map(str, value))
    return str(value)


def model_cost(model: torch.nn.Module) -> dict[str, Any]:
    """理想工作量模型（与引擎实现无关）：用于计算 MBU/MFU。

    weight_bytes：每个 decode step 至少读取一遍全部权重（tied embedding 只算一份，lm_head 会读它）。
    flops_per_token：2 * 参数量（线性层）；注意力部分另按上下文长度计算。
    """
    cfg = model.config
    params = list(model.parameters())
    elem = params[0].element_size()
    n_params = sum(p.numel() for p in params)
    return {
        "n_params": n_params,
        "weight_bytes": n_params * elem,
        "kv_bytes_per_token": 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * elem,
        "attn_flop_coeff": cfg.num_hidden_layers * cfg.num_attention_heads * cfg.head_dim,
    }


def efficiency(
    cost: dict[str, Any],
    peaks: dict[str, float | None],
    batch: int,
    prompt_len: int | Sequence[int],
    max_new_tokens: int | Sequence[int],
    ttft_ms: float,
    tpot_ms: float,
) -> dict[str, float | None]:
    """基于理想工作量与实测延迟，计算 prefill/decode 的 MFU 与 decode 的 MBU（带宽利用率）。"""
    out: dict[str, float | None] = {"prefill_mfu": None, "decode_mfu": None, "decode_mbu": None}
    avg_out_len = (
        sum(max_new_tokens) / len(max_new_tokens)
        if isinstance(max_new_tokens, Sequence) and not isinstance(max_new_tokens, str)
        else max_new_tokens
    )

    if isinstance(prompt_len, Sequence) and not isinstance(prompt_len, str):
        avg_prompt_len = sum(prompt_len) / len(prompt_len)
        ctx = avg_prompt_len + avg_out_len / 2
        sum_prefill_flops = sum(2 * cost["n_params"] * pl + 2 * cost["attn_flop_coeff"] * pl**2 for pl in prompt_len)
    else:
        ctx = prompt_len + avg_out_len / 2
        sum_prefill_flops = batch * (2 * cost["n_params"] * prompt_len + 2 * cost["attn_flop_coeff"] * prompt_len**2)

    if peaks["tflops"]:
        peak_flops = peaks["tflops"] * 1e12
        out["prefill_mfu"] = sum_prefill_flops / (ttft_ms / 1000) / peak_flops if ttft_ms else None
        dec = batch * (2 * cost["n_params"] + 4 * cost["attn_flop_coeff"] * ctx)
        out["decode_mfu"] = dec / (tpot_ms / 1000) / peak_flops if tpot_ms else None
    if peaks["bandwidth_gbs"] and tpot_ms:
        moved = cost["weight_bytes"] + batch * ctx * cost["kv_bytes_per_token"]
        out["decode_mbu"] = moved / (tpot_ms / 1000) / (peaks["bandwidth_gbs"] * 1e9)
    return out


@functools.cache
def git_info() -> dict[str, Any]:
    """获取当前工作区 Git 提交摘要（进程内缓存，并带超时保护）。"""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=BENCH_ARTIFACTS_DIR.parent, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""

    return {
        "commit": run("rev-parse", "--short", "HEAD") or "unknown",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(run("status", "--porcelain")),
        "message": run("log", "-1", "--pretty=%s"),
    }


def gpu_info(device: torch.device | int = 0) -> dict[str, Any]:
    """获取测试机 GPU 硬件环境参数。"""
    props = torch.cuda.get_device_properties(device)
    info: dict[str, Any] = {
        "gpu": props.name,
        "total_memory_mb": props.total_memory / 1024**2,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,clocks.sm,clocks.max.sm,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            .stdout.strip()
            .split(",")
        )
        info.update(
            driver=out[0].strip(),
            sm_clock_mhz=out[1].strip(),
            max_sm_clock_mhz=out[2].strip(),
            temperature_c=out[3].strip(),
        )
    except (OSError, subprocess.SubprocessError, IndexError) as err:
        warnings.warn(f"nvidia-smi 查询失败，跳过驱动/频率/温度信息: {err}", stacklevel=2)
    return info


def check_correctness(
    model: torch.nn.Module,
    tokenizer: Any,
    mode: Any,
    device: torch.device,
    update_golden: bool = False,
) -> dict[str, Any]:
    """正确性门禁：固定 prompt 的 greedy 输出必须与 golden 一致。"""
    ids = torch.tensor([tokenizer.encode_chat_prompt(GOLDEN_PROMPT)], device=device)
    res = generate(mode, model, ids, GOLDEN_MAX_OUTPUT_TOKENS, device=device)
    tokens = res.padded_output_tokens(0).tolist()

    if update_golden:
        GOLDEN_OUTPUT_PATH.write_text(
            json.dumps({"prompt": GOLDEN_PROMPT, "tokens": tokens, "commit": git_info()["commit"]}, indent=2),
            encoding="utf-8",
        )
        print(f"[golden] 已写入 {GOLDEN_OUTPUT_PATH.name}（{len(tokens)} tokens）")
        return {"checked": True, "passed": True, "match_ratio": 1.0, "updated_golden": True}

    if not GOLDEN_OUTPUT_PATH.exists():
        print("[golden] 未找到 golden 文件，跳过正确性检查（先用 --update-golden 生成）")
        return {"checked": False, "passed": None, "match_ratio": None}

    golden = json.loads(GOLDEN_OUTPUT_PATH.read_text(encoding="utf-8"))["tokens"]
    n = min(len(golden), len(tokens))
    matched = sum(a == b for a, b in zip(golden[:n], tokens[:n]))
    ratio = matched / n if n else 0.0
    passed = tokens == golden
    print(f"[golden] 匹配 {matched}/{n}，{'通过' if passed else '失败'}")
    return {"checked": True, "passed": passed, "match_ratio": ratio}


def _format_case_row(w: CaseBenchmarkResult, mode: str | None, has_packing: bool) -> list[str]:
    """渲染单个用例的一行 CSV 字段（列顺序与 print_table 表头一致）。"""
    row = [
        str(w.workload.mode or mode or ""),
        format_lengths(w.workload.prompt_len),
        format_lengths(w.workload.max_new_tokens),
        f"{w.latency.ttft_ms.mean:.3f}",
        f"{w.latency.tpot_ms.mean:.3f}",
        f"{w.latency.tpot_ms.median:.3f}",
        f"{w.latency.tpot_ms.p95:.3f}",
        f"{w.latency.e2e_wall_ms.mean:.3f}",
        f"{w.latency.e2e_model_ms.mean:.3f}",
        f"{w.memory.prefill_allocated_mb:.2f}",
        f"{w.memory.peak_allocated_mb:.2f}",
        f"{w.memory.kv_cache_mb:.2f}",
    ]
    pack = w.packing
    if has_packing:
        if pack:
            eff_p = f"{pack.effective_prompt_throughput.mean:.1f}" if pack.effective_prompt_throughput else ""
            row.extend([f"{pack.padding_efficiency * 100:.2f}%", f"{pack.slot_utilization * 100:.2f}%", eff_p])
        else:
            row.extend(["", "", ""])
    return row


def print_table(case_results: Sequence[CaseBenchmarkResult], mode: str | None = None) -> None:
    """控制台标准 CSV 格式输出。"""
    header = [
        "mode",
        "prompt_tokens",
        "generated_tokens",
        "prefill_ms_mean",
        "decode_ms_mean",
        "decode_ms_p50",
        "decode_ms_p95",
        "end_to_end_wall_ms_mean",
        "end_to_end_model_ms_mean",
        "memory_after_prefill_mib_max",
        "peak_memory_mib_max",
        "kv_cache_mb",
    ]

    has_packing = any(w.packing is not None for w in case_results)
    if has_packing:
        header.extend(["padding_efficiency", "slot_utilization", "effective_prompt_tok_s"])

    print(",".join(header))
    for w in case_results:
        print(",".join(_format_case_row(w, mode, has_packing)))


def print_case_summary(result: CaseBenchmarkResult) -> None:
    """打印单用例的延迟/吞吐/显存摘要与 MBU/MFU 行（run_case 与 run_suite 共用）。"""
    print(
        "  TTFT %.2f ms | TPOT %.2f±%.2f ms | %.1f tok/s | %.2f req/s | %.0f MiB"
        % (
            result.latency.ttft_ms.median,
            result.latency.tpot_ms.median,
            result.latency.tpot_ms.std,
            result.throughput.throughput_tok_s.median,
            result.throughput.throughput_req_s.median,
            result.memory.peak_allocated_mb,
        )
    )
    if result.efficiency and result.efficiency.decode_mbu is not None:
        gpu_busy_str = "%.0f%%" % (100 * result.gpu_time_ratio.median) if result.gpu_time_ratio is not None else "N/A"
        print(
            "  MBU %.1f%% | decode MFU %.2f%% | prefill MFU %.1f%% | GPU busy %s"
            % (
                100 * result.efficiency.decode_mbu,
                100 * (result.efficiency.decode_mfu or 0),
                100 * (result.efficiency.prefill_mfu or 0),
                gpu_busy_str,
            )
        )


def add_common_args(p: argparse.ArgumentParser) -> None:
    """向 ArgumentParser 添加 run_case 与 run_suite 共享的基础参数。"""
    p.add_argument("--tag", required=True, help="版本标签，如 baseline、v0.1.0、v0.2.0")
    p.add_argument(
        "--release-feature",
        default=None,
        help="当前版本对应的核心特性名称（如 'no cache'、'KV cache'、'FlashAttention'，可选）",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"预热次数（默认: {DEFAULT_WARMUP}）")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help=f"正式测量重复次数（默认: {DEFAULT_REPEATS}）")
    p.add_argument("--allow-mismatch", action="store_true", help="正确性不通过时仍记录结果（会标记）")
    p.add_argument("--peak-bandwidth-gbs", type=float, help="覆盖 GPU 峰值显存带宽（GB/s），用于 MBU")
    p.add_argument("--peak-tflops", type=float, help="覆盖 GPU 峰值 FP16/BF16 算力（TFLOPS），用于 MFU")
    p.add_argument(
        "--profile-mode",
        type=ProfileMode,
        choices=list(ProfileMode),
        default=None,
        help="Profiler 性能剖析模式: extra-profile 或 full-profile",
    )
    p.add_argument("--tensorboard-dir", type=Path, default=None, help="导出 TensorBoard 格式性能剖析日志的目录（可选）")
    p.add_argument("--output", type=Path, help="自定义输出路径")


def build_benchmark_record(
    args: argparse.Namespace,
    model: torch.nn.Module,
    mode_label: str,
    cases: Sequence[CaseBenchmarkResult],
    correctness: dict[str, Any],
    env: dict[str, Any],
    peaks: dict[str, float | None],
    cost: dict[str, Any] | None,
    git: dict[str, Any],
    warmup: int,
    repeats: int,
    suite_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构建标准评测结果 JSON 字典。"""
    record_cases = [c.to_record_dict() for c in cases]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "tag": args.tag,
        "release_feature": args.release_feature,
        "mode": mode_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git,
        "env": env,
        "peak_specs": peaks,
        "model_cost": cost,
        "config": {
            "warmup": warmup,
            "repeats": repeats,
            "model": args.model,
            "revision": args.revision,
            "dtype": str(next(model.parameters()).dtype),
            "tokenizer_included": False,
            "profile_mode": str(args.profile_mode) if args.profile_mode else None,
        },
        "correctness": correctness,
        "cases": record_cases,
    }
    if suite_info is not None:
        record["suite"] = suite_info
    return record


def save_benchmark_record(output: Path, record: dict[str, Any], dirty: bool = False) -> None:
    """将评测结果保存至指定 JSON 文件路径。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"结果已写入 {output}" + ("（工作区有未提交改动）" if dirty else ""))
