#!/usr/bin/env python3
r"""固定显存容量实验脚本（对齐 Chapter 05 §5.3.7 表 43）。

在固定显存 GPU 环境中，通过连续调度运行混合长短 Prompt 请求负载，
扫描 Dense 与 Paged KV Cache 在多级并发档位（4/8/12/16）下的容量边界与吞吐表现。
遇到 CUDA OOM 时优雅捕获，熔断该配置后续重复并记录状态，保证全量配置扫描顺利完成。

典型运行命令：

1. 快速检查拓扑参数与执行计划（Dry Run，无需 GPU）：
    python scripts/benchmark/paged_kv_cache_capacity.py --dry-run

2. 正式执行 GPU 容量实验（对齐教材 §5.3.7 默认参数）：
    python scripts/benchmark/paged_kv_cache_capacity.py \
      --concurrencies 4 12 20 24\
      --total-requests 32 \
      --short-prompt-length 128 \
      --long-prompt-length 4096 \
      --short-output 32 \
      --followup-output 32 \
      --long-output 8 \
      --block-size 16 \
      --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="扫描 Dense 与 Paged KV Cache 在不同并发档位下的显存容量与吞吐（表 43 对照实验）"
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[4, 8, 12, 16],
        help="并发档位列表（默认: [4, 8, 12, 16]）",
    )
    parser.add_argument("--total-requests", type=int, default=24, help="总请求数（默认: 24）")
    parser.add_argument("--short-prompt-length", type=int, default=128, help="短 Prompt 长度（默认: 128）")
    parser.add_argument("--long-prompt-length", type=int, default=4096, help="长 Prompt 长度（默认: 4096）")
    parser.add_argument("--short-output", type=int, default=32, help="初始短请求输出 Token 预算（默认: 32）")
    parser.add_argument("--followup-output", type=int, default=32, help="后续短请求输出 Token 预算（默认: 32）")
    parser.add_argument("--long-output", type=int, default=8, help="长请求输出 Token 预算（默认: 8）")
    parser.add_argument("--long-arrival-ms", type=float, default=200.0, help="长请求逻辑到达时间 ms（默认: 200.0）")
    parser.add_argument(
        "--followup-arrival-ms",
        type=float,
        default=201.0,
        help="后续短请求逻辑到达时间 ms（默认: 201.0）",
    )
    parser.add_argument("--block-size", type=int, default=16, help="Paged KV Cache 物理块大小（默认: 16）")
    parser.add_argument("--repeats", type=int, default=3, help="成功配置重复测量次数（默认: 3）")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="模型 ID 或路径（默认: Qwen/Qwen3-0.6B）")
    parser.add_argument("--revision", default=None, help="模型 Revision")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/benchmark-capacity-results.local.json"),
        help="结果 JSON 落盘路径（默认: /tmp/benchmark-capacity-results.local.json）",
    )
    parser.add_argument("--tag", default="capacity-scan", help="用例记录 Tag（默认: capacity-scan）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印执行计划与拓扑参数，不启动实际推理",
    )
    return parser.parse_args()


def format_table_cell(value: float | None, fmt: str = ".2f") -> str:
    return f"{value:{fmt}}" if value is not None else "–"


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "路径",
        "并发",
        "状态",
        "Makespan/ms",
        "输出token/s",
        "ITL p95/ms",
        "Pool/MiB",
        "峰值/MiB",
    ]
    col_widths = [8, 6, 8, 14, 14, 12, 12, 12]

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=True))
    separator_line = "  ".join("-" * w for w in col_widths)

    print()
    print("=" * len(header_line))
    print("表 43: 固定显存容量实验 — 离散并发容量扫描结果")
    print("=" * len(header_line))
    print(header_line)
    print(separator_line)

    for r in rows:
        cells = [
            str(r["path"]).ljust(col_widths[0]),
            str(r["concurrency"]).rjust(col_widths[1]),
            str(r["status"]).center(col_widths[2]),
            format_table_cell(r["makespan_ms"], ".2f").rjust(col_widths[3]),
            format_table_cell(r["throughput_tok_s"], ".2f").rjust(col_widths[4]),
            format_table_cell(r["itl_p95_ms"], ".2f").rjust(col_widths[5]),
            format_table_cell(r["pool_mb"], ".2f").rjust(col_widths[6]),
            format_table_cell(r["peak_mb"], ".2f").rjust(col_widths[7]),
        ]
        print("  ".join(cells))

    print(separator_line)
    print()


def main() -> int:
    args = parse_args()

    for c in args.concurrencies:
        if c < 1:
            raise ValueError(f"并发数必须为正整数: {c}")
        if c > args.total_requests:
            raise ValueError(f"并发数 ({c}) 不能超过总请求数 ({args.total_requests})")

    rows: list[dict[str, Any]] = []

    print(
        f"==> 启动固定显存容量扫描: 并发档位={args.concurrencies}, "
        f"总请求数={args.total_requests}, repeats={args.repeats}"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for concurrency in args.concurrencies:
            initial_shorts = concurrency - 1
            followup_shorts = args.total_requests - concurrency

            prompt_lens = (
                [args.short_prompt_length] * initial_shorts
                + [args.long_prompt_length]
                + [args.short_prompt_length] * followup_shorts
            )
            output_budgets = (
                [args.short_output] * initial_shorts + [args.long_output] + [args.followup_output] * followup_shorts
            )
            arrivals = [0.0] * initial_shorts + [args.long_arrival_ms] + [args.followup_arrival_ms] * followup_shorts

            print(
                f"\n--- [Concurrency = {concurrency:2d}] "
                f"初始短请求={initial_shorts}, 长请求=1 (len={args.long_prompt_length}), "
                f"后续短请求={followup_shorts} ---"
            )

            for mode_name in ("dense", "paged"):
                path_label = "Dense" if mode_name == "dense" else "Paged"
                caching_method = "dense-kv-cache" if mode_name == "dense" else "paged-kv-cache"

                tmp_case_output = tmp_dir_path / f"case_{mode_name}_{concurrency}.json"

                cmd = [
                    sys.executable,
                    "-m",
                    "mini_infra.cli.main",
                    "bench",
                    "scheduler",
                    "case",
                    "--tag",
                    args.tag,
                    "--caching-method",
                    caching_method,
                    "--batching-method",
                    "continuous-batching",
                    "--max-running-requests",
                    str(concurrency),
                    "--repeats",
                    str(args.repeats),
                    "--warmup",
                    "0",
                    "--prompt-lens",
                    *map(str, prompt_lens),
                    "--output-budgets",
                    *map(str, output_budgets),
                    "--arrival-times-ms",
                    *map(str, arrivals),
                    "--output",
                    str(tmp_case_output),
                ]

                if mode_name == "paged":
                    cmd.extend(["--block-size", str(args.block_size)])
                if args.model:
                    cmd.extend(["--model", args.model])
                if args.revision:
                    cmd.extend(["--revision", args.revision])

                if args.dry_run:
                    print(f"[{path_label:5s}] (Dry-Run) CMD: {' '.join(cmd[:12])} ...")
                    rows.append(
                        {
                            "path": path_label,
                            "concurrency": concurrency,
                            "status": "dry-run",
                            "makespan_ms": None,
                            "throughput_tok_s": None,
                            "itl_p95_ms": None,
                            "pool_mb": None,
                            "peak_mb": None,
                        }
                    )
                    continue

                print(f"[{path_label:5s}] 运行中 (并发={concurrency:2d}) ...", flush=True)

                env = os.environ.copy()
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

                proc = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                )

                is_oom = proc.returncode != 0 and any(
                    err_kw in (proc.stderr + proc.stdout)
                    for err_kw in ("CUDA out of memory", "OutOfMemoryError", "CUDA OOM")
                )

                if is_oom:
                    print(f"[{path_label:5s}] concurrency={concurrency:2d} -> 触发 OOM（已熔断跳过后续重复）")
                    rows.append(
                        {
                            "path": path_label,
                            "concurrency": concurrency,
                            "status": "OOM",
                            "makespan_ms": None,
                            "throughput_tok_s": None,
                            "itl_p95_ms": None,
                            "pool_mb": None,
                            "peak_mb": None,
                        }
                    )
                elif proc.returncode == 0 and tmp_case_output.exists():
                    try:
                        with tmp_case_output.open("r", encoding="utf-8") as f:
                            data = json.load(f)

                        case_info = data["cases"][0]
                        sched_metrics = case_info.get("schedule", {})
                        mem_metrics = case_info.get("memory", {})
                        kv_metrics = mem_metrics.get("kv_cache", {})

                        makespan = sched_metrics.get("makespan_ms", {}).get("mean")
                        throughput = case_info.get("throughput_tok_s", {}).get("mean")
                        itl_p95 = sched_metrics.get("itl_ms_p95", {}).get("mean")
                        pool_mb = kv_metrics.get("peak_kv_cache_pool_mb")
                        peak_mb = mem_metrics.get("peak_allocated_mb")

                        print(
                            f"[{path_label:5s}] concurrency={concurrency:2d} -> 成功 | "
                            f"吞吐: {throughput:.2f} tok/s, 峰值显存: {peak_mb:.1f} MiB"
                        )
                        rows.append(
                            {
                                "path": path_label,
                                "concurrency": concurrency,
                                "status": "成功",
                                "makespan_ms": makespan,
                                "throughput_tok_s": throughput,
                                "itl_p95_ms": itl_p95,
                                "pool_mb": pool_mb,
                                "peak_mb": peak_mb,
                            }
                        )
                    except Exception as err:
                        print(f"[{path_label:5s}] 解析结果 JSON 出错: {err}")
                        rows.append(
                            {
                                "path": path_label,
                                "concurrency": concurrency,
                                "status": f"ParseErr: {err}",
                                "makespan_ms": None,
                                "throughput_tok_s": None,
                                "itl_p95_ms": None,
                                "pool_mb": None,
                                "peak_mb": None,
                            }
                        )
                else:
                    err_msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "未知错误"
                    print(f"[{path_label:5s}] 异常退出 (code={proc.returncode}): {err_msg}")
                    rows.append(
                        {
                            "path": path_label,
                            "concurrency": concurrency,
                            "status": f"Fail({proc.returncode})",
                            "makespan_ms": None,
                            "throughput_tok_s": None,
                            "itl_p95_ms": None,
                            "pool_mb": None,
                            "peak_mb": None,
                        }
                    )

    # 打印格式化表格
    print_table(rows)

    # 导出落盘结果 JSON
    if not args.dry_run and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "experiment": "chapter_05_5_3_7_capacity_benchmark",
            "table": "Table 43",
            "parameters": {
                "concurrencies": args.concurrencies,
                "total_requests": args.total_requests,
                "short_prompt_length": args.short_prompt_length,
                "long_prompt_length": args.long_prompt_length,
                "short_output": args.short_output,
                "followup_output": args.followup_output,
                "long_output": args.long_output,
                "block_size": args.block_size,
                "repeats": args.repeats,
                "model": args.model,
                "revision": args.revision,
            },
            "results": rows,
        }
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"==> 完整结果已保存至: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
