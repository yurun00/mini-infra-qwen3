"""基准测试共享基础设施模块：环境探测、算力利用率计算、正确性门禁、CLI公共参数与结果落盘。"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from mini_infra.generation import (
    DEFAULT_GOLDEN_MAX_OUTPUT_TOKENS,
    DEFAULT_GOLDEN_PROMPT,
    generate,
    percentile,
)
from mini_infra.qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
)


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
GOLDEN_PROMPT = DEFAULT_GOLDEN_PROMPT
GOLDEN_MAX_OUTPUT_TOKENS = DEFAULT_GOLDEN_MAX_OUTPUT_TOKENS
SCHEMA_VERSION = 1


def summarize(values: Sequence[float]) -> dict[str, float]:
    """计算延迟分位数与均值方差指标。"""
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


GPU_PEAK_SPECS = {
    "L4": (300.0, 121.0),
    "T4": (320.0, 65.0),
    "A10": (600.0, 125.0),
    "A100": (1555.0, 312.0),
    "H100": (3350.0, 989.0),
    "4090": (1008.0, 165.0),
    "3090": (936.0, 71.0),
}


def resolve_peak_specs(
    gpu_name: str,
    bandwidth_gbs: float | None = None,
    tflops: float | None = None,
) -> dict[str, float | None]:
    """推断或覆盖 GPU 峰值显存带宽 (GB/s) 与 FP16 峰值算力 (TFLOPS)。"""
    spec = next((v for k, v in GPU_PEAK_SPECS.items() if k in gpu_name), (None, None))
    return {"bandwidth_gbs": bandwidth_gbs or spec[0], "tflops": tflops or spec[1]}


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


def git_info() -> dict[str, Any]:
    """获取当前工作区 Git 提交摘要。"""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=BENCH_ARTIFACTS_DIR.parent, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "--short", "HEAD") or "unknown",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(run("status", "--porcelain")),
        "message": run("log", "-1", "--pretty=%s"),
    }


def gpu_info() -> dict[str, Any]:
    """获取测试机 GPU 硬件环境参数。"""
    props = torch.cuda.get_device_properties(0)
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
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass
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
    tokens = res.padded_output_tokens[0].tolist()

    if update_golden:
        GOLDEN_OUTPUT_PATH.write_text(
            json.dumps({"prompt": GOLDEN_PROMPT, "tokens": tokens, "commit": git_info()["commit"]}, indent=2)
        )
        print(f"[golden] 已写入 {GOLDEN_OUTPUT_PATH.name}（{len(tokens)} tokens）")
        return {"checked": True, "passed": True, "match_ratio": 1.0, "updated_golden": True}

    if not GOLDEN_OUTPUT_PATH.exists():
        print("[golden] 未找到 golden 文件，跳过正确性检查（先用 --update-golden 生成）")
        return {"checked": False, "passed": None, "match_ratio": None}

    golden = json.loads(GOLDEN_OUTPUT_PATH.read_text())["tokens"]
    n = min(len(golden), len(tokens))
    matched = sum(a == b for a, b in zip(golden[:n], tokens[:n]))
    ratio = matched / n if n else 0.0
    passed = tokens == golden
    print(f"[golden] 匹配 {matched}/{n}，{'通过' if passed else '失败'}")
    return {"checked": True, "passed": passed, "match_ratio": ratio}


def print_table(case_results: Sequence[dict[str, Any]], mode: str | None = None) -> None:
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
    print(",".join(header))
    for w in case_results:
        if isinstance(w, list | tuple):
            print(",".join(str(x) for x in w))
            continue
        if "prompt_tokens" in w:
            print(",".join(str(w.get(k, "")) for k in header))
            continue

        mode_str = w.get("mode") or mode or ""
        prompt_tokens = w.get("prompt_len", 0)
        generated_tokens = w.get("max_new_tokens", w.get("output_len", 0))
        prefill_ms_mean = w["ttft_ms"]["mean"]
        decode_ms_mean = w["tpot_ms"]["mean"]
        decode_ms_p50 = w["tpot_ms"].get("p50", w["tpot_ms"]["median"])
        decode_ms_p95 = w["tpot_ms"].get("p95", percentile(w["raw"]["tpot_ms"], 95) if "raw" in w else 0.0)
        end_to_end_wall_ms_mean = w["e2e_wall_ms"]["mean"]
        end_to_end_model_ms_mean = w["e2e_model_ms"]["mean"]
        mem = w.get("memory", {})
        prefill_mem = mem.get("prefill_allocated_mb", 0.0)
        peak_mem = mem.get("peak_allocated_mb", 0.0)
        kv_cache_mb = mem.get("kv_cache_mb", 0.0)

        prompt_str = (
            "/".join(map(str, prompt_tokens)) if isinstance(prompt_tokens, list | tuple) else str(prompt_tokens)
        )
        gen_str = (
            "/".join(map(str, generated_tokens))
            if isinstance(generated_tokens, list | tuple)
            else str(generated_tokens)
        )

        row = [
            str(mode_str),
            prompt_str,
            gen_str,
            f"{prefill_ms_mean:.3f}",
            f"{decode_ms_mean:.3f}",
            f"{decode_ms_p50:.3f}",
            f"{decode_ms_p95:.3f}",
            f"{end_to_end_wall_ms_mean:.3f}",
            f"{end_to_end_model_ms_mean:.3f}",
            f"{prefill_mem:.2f}",
            f"{peak_mem:.2f}",
            f"{kv_cache_mb:.2f}",
        ]
        print(",".join(row))


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
    cases: list[dict[str, Any]],
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
        "cases": cases,
    }
    if suite_info is not None:
        record["suite"] = suite_info
    return record


def save_benchmark_record(output: Path, record: dict[str, Any], dirty: bool = False) -> None:
    """将评测结果保存至指定 JSON 文件路径。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"结果已写入 {output}" + ("（工作区有未提交改动）" if dirty else ""))
