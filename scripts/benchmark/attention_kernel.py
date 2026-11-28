#!/usr/bin/env python3
r"""eager 与 flash attention backend 的独立进程长序列 Attention 微基准（对齐 https://github.com/SVDX-lab/miniinfra/tree/main/inference/ chapter_09 benchmark_attention.py）。

每个 (长度, backend) 组合在独立子进程中运行，避免前一个 OOM 或 CUDA Caching Allocator 状态污染后续样本。
输入创建与 Triton Kernel 首次 JIT 编译不计入延迟（warmup 吸收）。增量峰值显存为 Q/K/V、Mask 与一个预热输出
已存在后，再执行一次 Attention 所增加的 max_memory_allocated。

默认形状与 Qwen3-0.6B 单层 Attention 一致：num_heads=16、num_kv_heads=8（GQA）、head_dim=128、bfloat16、
因果 Mask；eager 走真实 baseline 路径（含 GQA repeat_interleave 展开与加性 attn_bias 物化）。

典型运行命令：

1. Full Prefill（Q = K = L）：
    python scripts/benchmark/attention_kernel.py \
      --lengths 256 1024 4096 8192 --warmup 5 --repeats 20 \
      --output tmp/attention_kernel_prefill.local.json

2. Decode（Q = 1，K = L，batch=8）：
    python scripts/benchmark/attention_kernel.py \
      --mode decode --batch-size 8 --lengths 1024 4096 16384 \
      --output tmp/attention_kernel_decode.local.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from mini_infra.model import CausalPaddingMask, get_attn_backend
from mini_infra.utils import percentile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="eager 与 flash attention backend 长序列微基准"
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 4096, 8192])
    parser.add_argument(
        "--mode",
        choices=("prefill", "decode"),
        default="prefill",
        help="prefill: Q = K = L 因果 Full Prefill；decode: Q = 1, K = L（默认: %(default)s）",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output", help="可选 JSON 输出路径（建议写入 tmp/ 或 *.local.json）"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("eager", "flash"), help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    """在当前进程执行单个 (backend, length) 组合的测量。"""
    torch.manual_seed(9)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    kv_len = args.length
    query_len = kv_len if args.mode == "prefill" else 1
    query = (
        torch.randn(
            (args.batch_size, args.num_heads, query_len, args.head_dim),
            device=device,
            dtype=dtype,
        )
        * 0.25
    )
    key, value = (
        torch.randn(
            (args.batch_size, args.num_kv_heads, kv_len, args.head_dim),
            device=device,
            dtype=dtype,
        )
        * 0.25
        for _ in range(2)
    )
    key_valid = torch.ones((args.batch_size, kv_len), dtype=torch.bool, device=device)
    mask = CausalPaddingMask(
        key_valid, key_valid[:, kv_len - query_len :], kv_len - query_len
    )
    backend = get_attn_backend(args.backend)
    scaling = args.head_dim**-0.5
    row: dict[str, Any] = {"backend": args.backend, "length": kv_len}
    try:
        with torch.inference_mode():
            # prepare_inputs 计入单次 Attention 成本：eager 的 attn_bias 物化本身就是其显存开销的一部分
            for _ in range(args.warmup):
                output = backend.forward(
                    query, key, value, backend.prepare_inputs(mask, dtype), scaling
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_bytes = torch.cuda.memory_allocated(device)
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(args.repeats)
            ]
            for start, end in events:
                start.record()
                output = backend.forward(
                    query, key, value, backend.prepare_inputs(mask, dtype), scaling
                )
                end.record()
            torch.cuda.synchronize()
            samples_ms = [start.elapsed_time(end) for start, end in events]
            peak_bytes = torch.cuda.max_memory_allocated(device)
            del output
    except torch.OutOfMemoryError as error:
        row.update(status="oom", error=str(error).splitlines()[0])
        return row
    p50_ms = percentile(samples_ms, 50)
    row.update(
        status="ok",
        latency_ms_p50=p50_ms,
        latency_ms_p95=percentile(samples_ms, 95),
        query_tokens_per_s=args.batch_size * query_len / (p50_ms / 1000),
        incremental_peak_memory_bytes=max(0, peak_bytes - baseline_bytes),
    )
    return row


def system_environment() -> dict[str, Any]:
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_gib": properties.total_memory / 1024**3,
        "compute_capability": "%d.%d" % (properties.major, properties.minor),
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "triton": __import__("triton").__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _forwarded_args(args: argparse.Namespace) -> list[str]:
    return [
        "--mode", args.mode,
        "--batch-size", str(args.batch_size),
        "--num-heads", str(args.num_heads),
        "--num-kv-heads", str(args.num_kv_heads),
        "--head-dim", str(args.head_dim),
        "--dtype", args.dtype,
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
    ]  # fmt: skip


def run_suite(args: argparse.Namespace) -> None:
    """父进程：逐 (长度, backend) 拉起独立子进程测量并汇总。"""
    rows = []
    script = str(Path(__file__).resolve())
    for length in args.lengths:
        for backend in ("eager", "flash"):
            command = [
                sys.executable, script, "--worker", "--backend", backend,
                "--length", str(length), *_forwarded_args(args),
            ]  # fmt: skip
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                row = {
                    "backend": backend,
                    "length": length,
                    "status": "error",
                    "error": completed.stderr.strip().splitlines()[-1],
                }
            else:
                row = json.loads(completed.stdout.strip().splitlines()[-1])
            rows.append(row)
            if row["status"] == "ok":
                print(
                    "%-5s L=%6d: p50 %.3f ms / p95 %.3f ms, 增量峰值显存 %.1f MiB"
                    % (
                        backend,
                        length,
                        row["latency_ms_p50"],
                        row["latency_ms_p95"],
                        row["incremental_peak_memory_bytes"] / 1024**2,
                    )
                )
            else:
                print("%-5s L=%6d: %s" % (backend, length, row["status"]))

    comparisons = []
    for length in args.lengths:
        eager, flash = (
            next(r for r in rows if r["length"] == length and r["backend"] == name)
            for name in ("eager", "flash")
        )
        comparison: dict[str, Any] = {"length": length}
        if eager["status"] == flash["status"] == "ok":
            comparison["speedup_p50"] = eager["latency_ms_p50"] / flash["latency_ms_p50"]
            print("L=%6d p50 加速比: %.2fx" % (length, comparison["speedup_p50"]))
        comparisons.append(comparison)

    if args.output:
        payload = {
            "environment": system_environment(),
            "configuration": {
                key: getattr(args, key)
                for key in (
                    "mode",
                    "batch_size",
                    "num_heads",
                    "num_kv_heads",
                    "head_dim",
                    "dtype",
                    "warmup",
                    "repeats",
                )
            },  # fmt: skip
            "results": rows,
            "comparisons": comparisons,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        print("JSON 结果已写入:", output_path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("微基准需要 NVIDIA GPU")
    if args.worker:
        print(json.dumps(run_case(args)))
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
