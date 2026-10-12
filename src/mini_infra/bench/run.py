"""统一 benchmark：多 workload、多次重复、统计量、正确性门禁，结果以 JSON 落盘。

数据源与绘图约定：
    本命令是项目中【唯一】产生全量基准数据并归档到 benchmarks/result-history/ 的工具。
    只有本命令写入 result-history/ 的 JSON 结果才会被 mini-infra-bench-plot 读取并绘制图表与报表。
    （mini-infra-bench-regression 仅用于 0/1 门禁判定，其临时结果不写入该目录，亦不参与画图）。
    数据目录默认为当前目录下的 benchmarks/，可通过环境变量 MINI_INFRA_BENCH_DIR 修改。
    

用法示例（需 pip install -e .，在项目根目录执行）：
    # 首次在 baseline 上生成 golden 输出（之后所有优化都与它比对）
    mini-infra-bench --tag baseline --update-golden

    # 每次优化后运行，结果写入 benchmarks/result-history/<commit>_<tag>.json
    mini-infra-bench --tag kv-cache

    # 快速试跑
    mini-infra-bench --tag dev --prompt-lens 32 --output-lens 16 --repeats 3

新增优化（如 KV Cache）时，只需在 ENGINES 中注册一个新的 generate 适配函数。
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from mini_infra.qwen3_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_handwritten_model,
    resolve_model_directory,
)
from mini_infra.qwen3_tokenizer import Qwen3Tokenizer

BENCH_DIR = Path(os.environ.get("MINI_INFRA_BENCH_DIR", "benchmarks")).resolve()
RESULTS_DIR = BENCH_DIR / "result-history"
GOLDEN_OUTPUT_PATH = BENCH_DIR / "data" / "golden_output_tokens.json"
GOLDEN_PROMPT = "请用一句话介绍 KV Cache。"
GOLDEN_MAX_OUTPUT_TOKENS = 32
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# 引擎适配层：返回 (生成的 token 张量, 每步的 CUDA Event 起止对)。
# 第 0 步 = Prefill，其余 = Decode。后续优化只需新增一个适配函数。
# --------------------------------------------------------------------------
@torch.inference_mode()
def generate_full_recompute(model, input_ids, new_tokens):
    """baseline：每步重算完整上下文（无 KV Cache）。固定长度，不因 EOS 提前停止。"""

    generated = input_ids
    events = []
    for _ in range(new_tokens):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        logits = model(generated)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        end.record()
        events.append((start, end))
        generated = torch.cat((generated, next_token), dim=1)
    return generated[:, input_ids.shape[1]:], events


ENGINES = {"full-recompute": generate_full_recompute}


# --------------------------------------------------------------------------
def percentile(values, q):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[index]


def summarize(values):
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p90": percentile(values, 90),
        "max": max(values),
    }


def git_info():
    def run(*args):
        return subprocess.run(
            ["git", *args], cwd=BENCH_DIR.parent, capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "--short", "HEAD") or "unknown",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(run("status", "--porcelain")),
        "message": run("log", "-1", "--pretty=%s"),
    }


def gpu_info():
    props = torch.cuda.get_device_properties(0)
    info = {
        "gpu": props.name,
        "total_memory_mb": props.total_memory / 1024**2,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,clocks.sm,clocks.max.sm,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().split(",")
        info.update(driver=out[0].strip(), sm_clock_mhz=out[1].strip(),
                    max_sm_clock_mhz=out[2].strip(), temperature_c=out[3].strip())
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass
    return info


def make_prompt(vocab_size, length, batch_size, device):
    """固定 seed 的合成 prompt，保证每次运行输入完全一致。"""

    generator = torch.Generator(device="cpu").manual_seed(1234 + length)
    ids = torch.randint(1000, min(vocab_size, 100000), (batch_size, length), generator=generator)
    return ids.to(device)


def check_correctness(model, tokenizer, engine, device, update_golden):
    """正确性门禁：固定 prompt 的 greedy 输出必须与 golden 一致。"""

    ids = torch.tensor([tokenizer.encode_chat_prompt(GOLDEN_PROMPT)], device=device)
    out, _ = engine(model, ids, GOLDEN_MAX_OUTPUT_TOKENS)
    tokens = out[0].tolist()

    if update_golden:
        GOLDEN_OUTPUT_PATH.write_text(json.dumps(
            {"prompt": GOLDEN_PROMPT, "tokens": tokens, "commit": git_info()["commit"]},
            indent=2))
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


def run_workload(model, engine, prompt_len, output_len, batch_size, warmup, repeats, vocab, device):
    ids = make_prompt(vocab, prompt_len, batch_size, device)

    for _ in range(warmup):
        engine(model, ids, output_len)
    torch.cuda.synchronize()

    ttft, tpot, total, throughput, peak = [], [], [], [], []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        _, events = engine(model, ids, output_len)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000

        steps = [s.elapsed_time(e) for s, e in events]
        decode = steps[1:]
        ttft.append(steps[0])
        tpot.append(sum(decode) / len(decode) if decode else 0.0)
        total.append(wall_ms)
        throughput.append(batch_size * output_len / (wall_ms / 1000))
        peak.append(torch.cuda.max_memory_allocated() / 1024**2)

    return {
        "prompt_len": prompt_len, "output_len": output_len, "batch_size": batch_size,
        "ttft_ms": summarize(ttft), "tpot_ms": summarize(tpot),
        "e2e_ms": summarize(total), "throughput_tok_s": summarize(throughput),
        "peak_memory_mb": max(peak),
        "raw": {"ttft_ms": ttft, "tpot_ms": tpot, "e2e_ms": total},
    }


def parse_args():
    p = argparse.ArgumentParser(description="mini-infra-qwen3 benchmark")
    p.add_argument("--tag", required=True, help="版本/特性标签，如 baseline、kv-cache")
    p.add_argument("--engine", default="full-recompute", choices=ENGINES)
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[32, 128, 512, 1024])
    p.add_argument("--output-lens", type=int, nargs="+", default=[64])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--update-golden", action="store_true", help="用当前输出覆盖 golden（仅在 baseline 上使用）")
    p.add_argument("--allow-mismatch", action="store_true", help="正确性不通过时仍记录结果（会标记）")
    p.add_argument("--output", type=Path, help="自定义输出路径")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    engine = ENGINES[args.engine]

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_handwritten_model(model_dir, device)
    vocab = model.config.vocab_size if hasattr(model, "config") else 150000

    correctness = check_correctness(model, tokenizer, engine, device, args.update_golden)
    if correctness["passed"] is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    workloads = []
    for bs in args.batch_sizes:
        for pl in args.prompt_lens:
            for ol in args.output_lens:
                print(f"[bench] batch={bs} prompt={pl} output={ol} ...", flush=True)
                try:
                    result = run_workload(model, engine, pl, ol, bs, args.warmup,
                                          args.repeats, vocab, device)
                except torch.cuda.OutOfMemoryError:
                    print("  OOM，跳过")
                    torch.cuda.empty_cache()
                    continue
                workloads.append(result)
                print("  TTFT %.2f ms | TPOT %.2f±%.2f ms | %.1f tok/s | %.0f MiB" % (
                    result["ttft_ms"]["median"], result["tpot_ms"]["median"],
                    result["tpot_ms"]["std"], result["throughput_tok_s"]["median"],
                    result["peak_memory_mb"]))

    git = git_info()
    record = {
        "schema_version": SCHEMA_VERSION,
        "tag": args.tag,
        "engine": args.engine,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git,
        "env": gpu_info(),
        "config": {"warmup": args.warmup, "repeats": args.repeats, "model": args.model,
                   "revision": args.revision, "dtype": str(next(model.parameters()).dtype)},
        "correctness": correctness,
        "workloads": workloads,
    }

    output = args.output or RESULTS_DIR / f"{git['commit']}_{args.tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"结果已写入 {output}" + ("（工作区有未提交改动）" if git["dirty"] else ""))


if __name__ == "__main__":
    main()
