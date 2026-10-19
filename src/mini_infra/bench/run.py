"""统一 benchmark：多 workload、多次重复、统计量、正确性门禁，结果以 JSON 落盘。

数据源与绘图约定：
    本命令是项目中【唯一】产生全量基准数据并归档到 benchmarks/result-history/ 的工具。
    只有本命令写入 result-history/ 的 JSON 结果才会被 mini-infra-bench-plot-history 读取并绘制图表与报表。
    （mini-infra-bench-regression 仅用于 0/1 门禁判定，其临时结果不写入该目录，亦不参与画图）。
    数据目录默认为当前目录下的 benchmarks/，可通过环境变量 MINI_INFRA_BENCH_DIR 修改。


用法示例（需 pip install -e .，在项目根目录执行）：
    # 首次在 baseline 上生成 golden 输出（之后所有优化都与它比对）
    mini-infra-bench --tag baseline --update-golden

    # 每次优化后运行，结果写入 benchmarks/result-history/<commit>_<tag>.json
    mini-infra-bench --tag kv-cache

    # 快速试跑
    mini-infra-bench --tag dev --prompt-lens 32 --output-lens 16 --repeats 3

新增优化时，在 mini_infra.generation 的 Mode 结构体中加 feature 字段即可。
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

from mini_infra.generation import (
    Feature,
    Mode,
    generate,
    percentile,
    synchronize,
)
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
# 模式适配层：Mode 结构体包含所有 feature 字段，直接从 mini_infra.generation 导入。
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
def summarize(values):
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


# 峰值规格表（dense、无稀疏）：(显存带宽 GB/s, FP16/BF16 Tensor 算力 TFLOPS)。
# 未命中时请用 --peak-bandwidth-gbs / --peak-tflops 手动指定，否则 MBU/MFU 记为 None。
GPU_PEAK_SPECS = {
    "L4": (300.0, 121.0),
    "T4": (320.0, 65.0),
    "A10": (600.0, 125.0),
    "A100": (1555.0, 312.0),
    "H100": (3350.0, 989.0),
    "4090": (1008.0, 165.0),
    "3090": (936.0, 71.0),
}


def resolve_peak_specs(gpu_name, bandwidth_gbs=None, tflops=None):
    spec = next((v for k, v in GPU_PEAK_SPECS.items() if k in gpu_name), (None, None))
    return {"bandwidth_gbs": bandwidth_gbs or spec[0], "tflops": tflops or spec[1]}


def model_cost(model):
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


def efficiency(cost, peaks, batch, prompt_len, output_len, ttft_ms, tpot_ms):
    """基于理想工作量与实测延迟，计算 prefill/decode 的 MFU 与 decode 的 MBU（带宽利用率）。"""

    out = {"prefill_mfu": None, "decode_mfu": None, "decode_mbu": None}
    ctx = prompt_len + output_len / 2  # decode 阶段平均上下文长度
    if peaks["tflops"]:
        peak_flops = peaks["tflops"] * 1e12
        # prefill：线性层 2*N*P；因果注意力 QK^T 与 PV 各 2*P^2*d，取一半 -> 2*coeff*P^2
        pre = batch * (2 * cost["n_params"] * prompt_len + 2 * cost["attn_flop_coeff"] * prompt_len**2)
        out["prefill_mfu"] = pre / (ttft_ms / 1000) / peak_flops if ttft_ms else None
        dec = batch * (2 * cost["n_params"] + 4 * cost["attn_flop_coeff"] * ctx)
        out["decode_mfu"] = dec / (tpot_ms / 1000) / peak_flops if tpot_ms else None
    if peaks["bandwidth_gbs"] and tpot_ms:
        # 理想 decode step 至少读取：全部权重 + batch 个序列的 KV Cache
        moved = cost["weight_bytes"] + batch * ctx * cost["kv_bytes_per_token"]
        out["decode_mbu"] = moved / (tpot_ms / 1000) / (peaks["bandwidth_gbs"] * 1e9)
    return out


# 算子类别关键字（按顺序匹配，小写）。kernel 名无法区分 attention 的 matmul 与线性层 GEMM，
# 这部分归因交给下面的 module 级 scope 统计。
KERNEL_CATEGORIES = [
    ("attention", ("flash", "fmha", "mem_efficient", "sdpa", "attention")),
    ("softmax", ("softmax",)),
    ("gemm", ("gemm", "gemv", "cutlass", "cublas", "nvjet", "sgemm", "matmul", "bmm", "addmm", "mm")),
    ("norm", ("rms", "layer_norm", "layernorm")),
    ("reduce", ("reduce_kernel", "reduce")),
    ("memory", ("memcpy", "memset", "copy", "catarray", "cat_", "index", "gather", "embedding")),
    ("elementwise", ("elementwise", "vectorized", "mul", "add", "silu", "sigmoid", "rsqrt", "pow", "neg")),
]

# module 级 scope：类名 -> 类别。最外层的 scope 吸收子模块（如 attention 内的 q/k norm 与投影）。
MODULE_SCOPES = {
    "Qwen3Attention": "attention_block",
    "Qwen3MLP": "mlp_block",
    "RMSNorm": "norm",
    "RotaryEmbedding": "rope",
    "Embedding": "embedding",
}


def kernel_category(name):
    lowered = name.lower()
    for category, keywords in KERNEL_CATEGORIES:
        if any(k in lowered for k in keywords):
            return category
    return "other"


def _install_scope_hooks(model):
    """给关键模块挂 record_function 范围，使 profiler 能把 GPU 时间归到 attention/mlp/norm/lm_head 等模块。"""

    from torch.profiler import record_function

    handles, stack = [], []

    def pre(label):
        def hook(_module, _inputs):
            if stack:  # 已在某个 scope 内，不重复计数
                stack.append(None)
                return
            ctx = record_function(f"mi_scope::{label}")
            ctx.__enter__()
            stack.append(ctx)

        return hook

    def post(_module, _inputs, _output):
        ctx = stack.pop()
        if ctx is not None:
            ctx.__exit__(None, None, None)

    for name, module in model.named_modules():
        label = MODULE_SCOPES.get(type(module).__name__) or ("lm_head" if name == "lm_head" else None)
        if label:
            handles.append(module.register_forward_pre_hook(pre(label)))
            handles.append(module.register_forward_hook(post))
    return handles


def profile_ops(model, mode, ids, output_len, top=10):
    """用 torch.profiler 跑一次，返回三组归因信息（定位下一步优化目标）：

    - top_kernels：按 CUDA 自身耗时排序的 top-N kernel，每个带 category 标签；
    - by_category：按 kernel 类别（gemm/softmax/elementwise/...）汇总的 GPU 耗时占比；
    - by_module：按模块（attention_block/mlp_block/norm/lm_head/...）汇总的 GPU 耗时占比。
    """

    from torch.profiler import ProfilerActivity, profile

    handles = _install_scope_hooks(model)
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            generate(
                mode,
                model,
                ids,
                output_len,
                eos_token_id=-1,
                device=ids.device,
                stop_on_eos=False,
            )
            synchronize(ids.device)
    finally:
        for h in handles:
            h.remove()

    averages = list(prof.key_averages())
    kernels = [
        e for e in averages if getattr(e, "self_device_time_total", 0) > 0 and not e.key.startswith("mi_scope::")
    ]
    total = sum(e.self_device_time_total for e in kernels) or 1
    kernels.sort(key=lambda e: e.self_device_time_total, reverse=True)

    by_category = {}
    for e in kernels:
        c = kernel_category(e.key)
        by_category[c] = by_category.get(c, 0.0) + e.self_device_time_total / total

    scopes = [e for e in averages if e.key.startswith("mi_scope::")]
    scope_total = sum(e.device_time_total for e in scopes) or 1
    by_module = {e.key.split("::", 1)[1]: e.device_time_total / scope_total for e in scopes}

    return {
        "top_kernels": [
            {
                "name": e.key[:80],
                "category": kernel_category(e.key),
                "cuda_ms": e.self_device_time_total / 1000,
                "share": e.self_device_time_total / total,
                "calls": e.count,
            }
            for e in kernels[:top]
        ],
        "by_category": dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
        "by_module": dict(sorted(by_module.items(), key=lambda kv: -kv[1])),
    }


def git_info():
    def run(*args):
        return subprocess.run(["git", *args], cwd=BENCH_DIR.parent, capture_output=True, text=True).stdout.strip()

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


def make_prompt(vocab_size, length, batch_size, device):
    """固定 seed 的合成 prompt，保证每次运行输入完全一致。"""

    generator = torch.Generator(device="cpu").manual_seed(1234 + length)
    ids = torch.randint(1000, min(vocab_size, 100000), (batch_size, length), generator=generator)
    return ids.to(device)


def check_correctness(model, tokenizer, mode, device, update_golden):
    """正确性门禁：固定 prompt 的 greedy 输出必须与 golden 一致。"""

    ids = torch.tensor([tokenizer.encode_chat_prompt(GOLDEN_PROMPT)], device=device)
    res = generate(mode, model, ids, GOLDEN_MAX_OUTPUT_TOKENS, device=device)
    tokens = res.new_tokens[0].tolist()

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


def run_workload(
    model,
    mode,
    prompt_len,
    max_new_tokens,
    batch_size,
    warmup,
    repeats,
    vocab,
    device,
    cost=None,
    peaks=None,
    with_profile=False,
):
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2，才能测量 Decode")
    input_ids = make_prompt(vocab, prompt_len, batch_size, device)

    for _ in range(warmup):
        generate(
            mode,
            model,
            input_ids,
            max_new_tokens,
            eos_token_id=-1,
            device=device,
            stop_on_eos=False,
        )

    # 1. 执行多次迭代压测并采集单次运行结果
    runs = [
        generate(
            mode,
            model,
            input_ids,
            max_new_tokens,
            eos_token_id=-1,
            device=device,
            stop_on_eos=False,
        )
        for _ in range(repeats)
    ]
    raw = {
        "ttft_ms": [r.ttft_ms for r in runs],
        "tpot_ms": [r.tpot_ms for r in runs],
        "e2e_ms": [r.e2e_ms for r in runs],
        "throughput_tok_s": [r.throughput_tok_s for r in runs],
        "gpu_time_ratio": [r.gpu_time_ratio for r in runs],
        "step_latencies_ms": [r.step_latencies_ms for r in runs],
    }

    # 2. 显存结构化拆解：常驻(weights+输入) / 动态峰值(激活+KV Cache+临时张量) / allocator 保留（含碎片）
    mib = 1024**2
    memory = {
        "weights_mb": cost["weight_bytes"] / mib if cost else None,
        "baseline_allocated_mb": max(r.memory["baseline_allocated_mb"] for r in runs),
        "prefill_allocated_mb": max(r.memory["prefill_allocated_mb"] for r in runs),
        "peak_allocated_mb": max(r.memory["peak_allocated_mb"] for r in runs),
        "dynamic_peak_mb": max(r.memory["dynamic_peak_mb"] for r in runs),
        "peak_reserved_mb": max(r.memory["peak_reserved_mb"] for r in runs),
        "fragmentation_mb": max(r.memory["fragmentation_mb"] for r in runs),
        "cache_mb": max(r.memory.get("cache_bytes", 0) for r in runs) / mib,
    }

    # 3. 结构化汇总报告（按 Metadata / Latency / Throughput / Memory / Raw 组织）
    result = {
        # 负载元数据
        "mode": str(mode),
        "prompt_len": prompt_len,
        "output_len": max_new_tokens,
        "batch_size": batch_size,
        # 核心时延指标（分布聚合）
        "ttft_ms": summarize(raw["ttft_ms"]),
        "tpot_ms": summarize(raw["tpot_ms"]),
        "e2e_ms": summarize(raw["e2e_ms"]),
        # 吞吐与利用率指标
        "throughput_tok_s": summarize(raw["throughput_tok_s"]),
        "gpu_time_ratio": (
            summarize(raw["gpu_time_ratio"]) if all(v is not None for v in raw["gpu_time_ratio"]) else None
        ),
        # 显存指标（统一收敛于 memory 中，细粒度拆解）
        "memory": memory,
        # 原始采样列表
        "raw": raw,
    }
    if cost and peaks:
        result["efficiency"] = efficiency(
            cost,
            peaks,
            batch_size,
            prompt_len,
            max_new_tokens,
            result["ttft_ms"]["median"],
            result["tpot_ms"]["median"],
        )
    if with_profile:
        result["op_breakdown"] = profile_ops(model, mode, input_ids, max_new_tokens)
    return result


def print_table(workloads, mode=None):
    """控制台标准 CSV 格式输出，方便直接复制进 Excel/文档或重定向保存为 .csv。"""

    header = [
        "mode",
        "prompt_tokens",
        "generated_tokens",
        "prefill_ms_mean",
        "decode_ms_mean",
        "decode_ms_p50",
        "decode_ms_p95",
        "end_to_end_ms_mean",
        "memory_after_prefill_mib_max",
        "peak_memory_mib_max",
        "cache_mib",
    ]
    print(",".join(header))
    for w in workloads:
        if isinstance(w, list | tuple):
            print(",".join(str(x) for x in w))
            continue
        if "prompt_tokens" in w:
            print(",".join(str(w.get(k, "")) for k in header))
            continue

        mode_str = w.get("mode") or mode or ""
        prompt_tokens = w.get("prompt_len", 0)
        generated_tokens = w.get("output_len", 0)
        prefill_ms_mean = w["ttft_ms"]["mean"]
        decode_ms_mean = w["tpot_ms"]["mean"]
        decode_ms_p50 = w["tpot_ms"].get("p50", w["tpot_ms"]["median"])
        decode_ms_p95 = w["tpot_ms"].get("p95", percentile(w["raw"]["tpot_ms"], 95) if "raw" in w else 0.0)
        end_to_end_ms_mean = w["e2e_ms"]["mean"]
        mem = w.get("memory", {})
        prefill_mem = mem.get("prefill_allocated_mb", 0.0)
        peak_mem = mem.get("peak_allocated_mb", 0.0)
        cache_mib = mem.get("cache_mb", 0.0)

        row = [
            str(mode_str),
            str(prompt_tokens),
            str(generated_tokens),
            f"{prefill_ms_mean:.2f}",
            f"{decode_ms_mean:.2f}",
            f"{decode_ms_p50:.2f}",
            f"{decode_ms_p95:.2f}",
            f"{end_to_end_ms_mean:.2f}",
            f"{prefill_mem:.2f}",
            f"{peak_mem:.2f}",
            f"{cache_mib:.2f}",
        ]
        print(",".join(row))


def parse_args():
    p = argparse.ArgumentParser(description="mini-infra-qwen3 benchmark")
    p.add_argument("--tag", required=True, help="版本标签，如 baseline、v0.1.0、v0.2.0")
    p.add_argument(
        "--release-feature",
        default=None,
        help="当前版本对应的核心特性名称（如 'no cache'、'KV cache'、'FlashAttention'，可选）",
    )
    p.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 KV Cache 特性（默认: True）",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[32, 128, 512, 1024])
    p.add_argument("--output-lens", type=int, nargs="+", default=[64])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--update-golden", action="store_true", help="用当前输出覆盖 golden（仅在 baseline 上使用）")
    p.add_argument("--allow-mismatch", action="store_true", help="正确性不通过时仍记录结果（会标记）")
    p.add_argument("--peak-bandwidth-gbs", type=float, help="覆盖 GPU 峰值显存带宽（GB/s），用于 MBU")
    p.add_argument("--peak-tflops", type=float, help="覆盖 GPU 峰值 FP16/BF16 算力（TFLOPS），用于 MFU")
    p.add_argument(
        "--profile-ops", action="store_true", help="每个 workload 额外跑一次 profiler，记录 top kernel 耗时占比"
    )
    p.add_argument("--output", type=Path, help="自定义输出路径")
    args = p.parse_args()
    if any(ol < 2 for ol in args.output_lens):
        raise ValueError("--output-lens 至少为 2，才能测量 Decode")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 NVIDIA GPU")
    device = torch.device("cuda")
    features = {Feature.KV_CACHE} if args.kv_cache else set()
    mode = Mode(features)

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_handwritten_model(model_dir, device)
    vocab = model.config.vocab_size if hasattr(model, "config") else 150000

    correctness = check_correctness(model, tokenizer, mode, device, args.update_golden)
    if correctness["passed"] is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info()
    peaks = resolve_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = model_cost(model)
    if None in peaks.values():
        print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
    workloads = []
    for bs in args.batch_sizes:
        for pl in args.prompt_lens:
            for ol in args.output_lens:
                print(f"[bench] batch={bs} prompt={pl} output={ol} ...", flush=True)
                try:
                    result = run_workload(
                        model,
                        mode,
                        pl,
                        ol,
                        bs,
                        args.warmup,
                        args.repeats,
                        vocab,
                        device,
                        cost=cost,
                        peaks=peaks,
                        with_profile=args.profile_ops,
                    )
                except torch.cuda.OutOfMemoryError:
                    print("  OOM，跳过")
                    torch.cuda.empty_cache()
                    continue
                workloads.append(result)
                print(
                    "  TTFT %.2f ms | TPOT %.2f±%.2f ms | %.1f tok/s | %.0f MiB"
                    % (
                        result["ttft_ms"]["median"],
                        result["tpot_ms"]["median"],
                        result["tpot_ms"]["std"],
                        result["throughput_tok_s"]["median"],
                        result["memory"]["peak_allocated_mb"],
                    )
                )
                eff = result.get("efficiency") or {}
                if eff.get("decode_mbu") is not None:
                    gpu_busy_str = (
                        "%.0f%%" % (100 * result["gpu_time_ratio"]["median"]) if result.get("gpu_time_ratio") else "N/A"
                    )
                    print(
                        "  MBU %.1f%% | decode MFU %.2f%% | prefill MFU %.1f%% | GPU busy %s"
                        % (
                            100 * eff["decode_mbu"],
                            100 * (eff["decode_mfu"] or 0),
                            100 * (eff["prefill_mfu"] or 0),
                            gpu_busy_str,
                        )
                    )

    git = git_info()
    release_feature = args.release_feature
    record = {
        "schema_version": SCHEMA_VERSION,
        "tag": args.tag,
        "release_feature": release_feature,
        "mode": str(mode),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git,
        "env": env,
        "peak_specs": peaks,
        "model_cost": cost,
        "config": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "model": args.model,
            "revision": args.revision,
            "dtype": str(next(model.parameters()).dtype),
            "tokenizer_included": False,
        },
        "correctness": correctness,
        "workloads": workloads,
    }

    output = args.output or RESULTS_DIR / f"{git['commit']}_{args.tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"结果已写入 {output}" + ("（工作区有未提交改动）" if git["dirty"] else ""))
    if workloads:
        print()
        print_table(workloads, mode=mode)


if __name__ == "__main__":
    main()
