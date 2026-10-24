"""单用例基准测试（run_case）：一次调用只执行正好一个 Case。

三个核心参数遵循 numpy 式广播规则：
    --batch-size      标量 int（可选）。若未指定，自动由变长 prompt_lens / max_new_tokens
                      长度推断；若均为单值则默认为 1。
    --prompt-lens     int 列表。1 项表示全 Batch 等长；多项表示逐请求长度。
    --max-new-tokens  int 列表。1 项表示全 Batch 等长；多项表示逐请求预算。

用法示例：
    # 单用例快速测试（batch=1, prompt=32, max_new_tokens=16）
    mini-infra-bench-case --tag dev --prompt-lens 32 --max-new-tokens 16

    # 批处理等长测试（batch=4, prompt=128, max_new_tokens=32）
    mini-infra-bench-case --tag dev --batch-size 4 --prompt-lens 128 \
        --max-new-tokens 32 --batching-method fixed-batching

    # 不等长变长测试（batch 自动由 prompt_lens 推断为 4）
    mini-infra-bench-case --tag dev --prompt-lens 16 32 64 128 --max-new-tokens 32 --batching-method fixed-batching
"""

import argparse
import sys
from collections.abc import Sequence

import torch

from mini_infra.bench.common import (
    RESULT_HISTORY_DIR,
    ProfileMode,
    add_common_args,
    build_benchmark_record,
    check_correctness,
    efficiency,
    git_info,
    gpu_info,
    model_cost,
    print_table,
    resolve_peak_specs,
    save_benchmark_record,
    summarize,
)
from mini_infra.bench.suites import SuiteCase
from mini_infra.generation import (
    BatchingMethod,
    FeatureFlag,
    Mode,
    generate,
    make_prompt,
)
from mini_infra.profiling import TensorBoardOptions, merge_op_latency_breakdowns
from mini_infra.qwen3_model import (
    load_handwritten_model,
    resolve_model_directory,
)
from mini_infra.qwen3_tokenizer import Qwen3Tokenizer


def resolve_case_specs(
    prompt_lens: Sequence[int],
    max_new_tokens: Sequence[int],
    batch_size: int | None = None,
) -> tuple[int, int | tuple[int, ...], int | tuple[int, ...]]:
    """根据 numpy 式广播规则，从输入参数解析出 (effective_batch_size, prompt_len, max_new_tokens)。"""
    if not prompt_lens or any(p < 1 for p in prompt_lens):
        raise ValueError("prompt_lens 必须非空且每项大于 0")
    if not max_new_tokens or any(o < 2 for o in max_new_tokens):
        raise ValueError("max_new_tokens 必须非空且每项至少为 2，才能测量 Decode")

    len_p = len(prompt_lens)
    len_o = len(max_new_tokens)

    if batch_size is None:
        if len_p > 1 and len_o > 1:
            if len_p != len_o:
                raise ValueError(f"prompt_lens 长度 ({len_p}) 与 max_new_tokens 长度 ({len_o}) 不匹配")
            target_bs = len_p
        elif len_p > 1:
            target_bs = len_p
        elif len_o > 1:
            target_bs = len_o
        else:
            target_bs = 1
    else:
        if batch_size < 1:
            raise ValueError(f"batch_size 必须为正整数，得到 {batch_size}")
        if len_p != 1 and len_p != batch_size:
            raise ValueError(f"prompt_lens 数量 ({len_p}) 既不为 1，也不等于指定的 batch_size ({batch_size})")
        if len_o != 1 and len_o != batch_size:
            raise ValueError(f"max_new_tokens 数量 ({len_o}) 既不为 1，也不等于指定的 batch_size ({batch_size})")
        target_bs = batch_size

    resolved_prompt_len: int | tuple[int, ...] = prompt_lens[0] if len_p == 1 else tuple(prompt_lens)
    resolved_max_new_tokens: int | tuple[int, ...] = max_new_tokens[0] if len_o == 1 else tuple(max_new_tokens)
    return target_bs, resolved_prompt_len, resolved_max_new_tokens


def run_case(
    model: torch.nn.Module,
    mode: Mode,
    prompt_len: int | Sequence[int],
    max_new_tokens: int | Sequence[int],
    batch_size: int,
    warmup: int,
    repeats: int,
    vocab: int,
    device: torch.device,
    cost: dict | None = None,
    peaks: dict | None = None,
    profile_mode: ProfileMode | None = None,
    tensorboard_options: TensorBoardOptions | None = None,
) -> dict:
    """运行单一用例基准测试并返回详细指标字典。"""
    min_tokens = min(max_new_tokens) if isinstance(max_new_tokens, Sequence) else max_new_tokens
    if min_tokens < 2:
        raise ValueError("max_new_tokens 至少为 2，才能测量 Decode")

    is_variable_len = isinstance(prompt_len, Sequence) and not isinstance(prompt_len, str)
    effective_batch_size = len(prompt_len) if is_variable_len else batch_size
    input_ids = make_prompt(vocab, prompt_len, effective_batch_size, device)

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

    is_full_profile = profile_mode == ProfileMode.FULL_PROFILE
    runs = [
        generate(
            mode,
            model,
            input_ids,
            max_new_tokens,
            eos_token_id=-1,
            device=device,
            stop_on_eos=False,
            profile=is_full_profile,
            tensorboard_options=tensorboard_options if is_full_profile else None,
        )
        for _ in range(repeats)
    ]

    raw = {
        "ttft_ms": [r.ttft_ms for r in runs],
        "tpot_ms": [r.tpot_ms for r in runs],
        "e2e_wall_ms": [r.e2e_wall_ms for r in runs],
        "e2e_model_ms": [r.e2e_model_ms for r in runs],
        "throughput_tok_s": [r.throughput_tok_s for r in runs],
        "throughput_req_s": [r.throughput_req_s for r in runs],
        "gpu_time_ratio": [r.gpu_time_ratio for r in runs],
        "step_latencies_ms": [r.step_latencies_ms for r in runs],
    }

    mib = 1024**2
    memory = {
        "weights_mb": cost["weight_bytes"] / mib if cost else None,
        "baseline_allocated_mb": max(r.memory["baseline_allocated_mb"] for r in runs),
        "prefill_allocated_mb": max(r.memory["prefill_allocated_mb"] for r in runs),
        "peak_allocated_mb": max(r.memory["peak_allocated_mb"] for r in runs),
        "dynamic_peak_mb": max(r.memory["dynamic_peak_mb"] for r in runs),
        "peak_reserved_mb": max(r.memory["peak_reserved_mb"] for r in runs),
        "fragmentation_mb": max(r.memory["fragmentation_mb"] for r in runs),
        "kv_cache_mb": max(r.memory["kv_cache_mb"] for r in runs),
    }

    extra = dict(runs[0].extra) if runs and getattr(runs[0], "extra", None) is not None else None
    if extra is not None:
        extra_metric_names = (
            "effective_prompt_token_throughput_per_second",
            "padded_prompt_token_throughput_per_second",
        )
        for extra_metric_name in extra_metric_names:
            if extra_metric_name in runs[0].extra:
                samples = [
                    r.extra[extra_metric_name]
                    for r in runs
                    if getattr(r, "extra", None) and extra_metric_name in r.extra
                ]
                if samples:
                    extra[extra_metric_name] = summarize(samples)
                    raw[extra_metric_name] = samples

    result = {
        "mode": str(mode),
        "prompt_len": list(prompt_len) if is_variable_len else prompt_len,
        "max_new_tokens": list(max_new_tokens) if isinstance(max_new_tokens, Sequence) else max_new_tokens,
        "batch_size": effective_batch_size,
        "ttft_ms": summarize(raw["ttft_ms"]),
        "tpot_ms": summarize(raw["tpot_ms"]),
        "e2e_wall_ms": summarize(raw["e2e_wall_ms"]),
        "e2e_model_ms": summarize(raw["e2e_model_ms"]),
        "throughput_tok_s": summarize(raw["throughput_tok_s"]),
        "throughput_req_s": summarize(raw["throughput_req_s"]),
        "gpu_time_ratio": (
            summarize(raw["gpu_time_ratio"]) if all(v is not None for v in raw["gpu_time_ratio"]) else None
        ),
        "memory": memory,
        "extra": extra,
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
    if is_full_profile and runs:
        breakdowns = [r.op_latency_breakdown for r in runs if r.op_latency_breakdown is not None]
        result["op_latency_breakdown"] = merge_op_latency_breakdowns(breakdowns)
    elif profile_mode == ProfileMode.EXTRA_PROFILE:
        prof_res = generate(
            mode=mode,
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=-1,
            stop_on_eos=False,
            profile=True,
            tensorboard_options=tensorboard_options,
        )
        result["op_latency_breakdown"] = prof_res.op_latency_breakdown
    return result


def parse_args():
    p = argparse.ArgumentParser(description="mini-infra-qwen3 benchmark single case runner")
    add_common_args(p)
    p.add_argument(
        "--kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否启用 KV Cache 特性（默认: True）",
    )
    p.add_argument(
        "--batching-method",
        default=BatchingMethod.NONE.value,
        choices=[m.value for m in BatchingMethod],
        help="指定批处理策略（默认: none）",
    )
    p.add_argument(
        "--batch-size", type=int, default=None, help="批处理大小；未指定时自动由 prompt_lens/max_new_tokens 推断"
    )
    p.add_argument(
        "--prompt-lens",
        type=int,
        nargs="+",
        default=[128],
        help="Prompt 长度（1 项表示等长，多项表示逐请求长度，默认: 128）",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[64],
        help="最大新生成 Token 预算（1 项表示等长，多项表示逐请求长度，默认: 64）",
    )
    p.add_argument("--warmup", type=int, default=3, help="预热次数（默认: 3）")
    p.add_argument("--repeats", type=int, default=10, help="正式测量重复次数（默认: 10）")
    p.add_argument("--update-golden", action="store_true", help="用当前输出覆盖 golden（仅在 baseline 上使用）")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 NVIDIA GPU")
    device = torch.device("cuda")

    target_bs, resolved_prompt_len, resolved_max_new_tokens = resolve_case_specs(
        args.prompt_lens, args.max_new_tokens, args.batch_size
    )

    flags = {FeatureFlag.KV_CACHE} if args.kv_cache else set()
    mode = Mode(flags, batching_method=args.batching_method)
    case = SuiteCase(mode, target_bs, resolved_prompt_len, resolved_max_new_tokens)

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_handwritten_model(model_dir, device)
    vocab = model.config.vocab_size

    correctness = check_correctness(model, tokenizer, case.mode, device, args.update_golden)
    if correctness["passed"] is False and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info()
    peaks = resolve_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = model_cost(model)
    if None in peaks.values():
        print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
    tb_options = TensorBoardOptions(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None

    pl_disp = "/".join(map(str, case.prompt_len)) if isinstance(case.prompt_len, tuple) else case.prompt_len
    ol_disp = "/".join(map(str, case.max_new_tokens)) if isinstance(case.max_new_tokens, tuple) else case.max_new_tokens
    print(
        f"[bench-case] mode={case.mode} batch={case.batch_size} prompt={pl_disp} max_new_tokens={ol_disp} ...",
        flush=True,
    )

    result = run_case(
        model,
        case.mode,
        list(case.prompt_len) if isinstance(case.prompt_len, tuple) else case.prompt_len,
        list(case.max_new_tokens) if isinstance(case.max_new_tokens, tuple) else case.max_new_tokens,
        case.batch_size,
        args.warmup,
        args.repeats,
        vocab,
        device,
        cost=cost,
        peaks=peaks,
        profile_mode=args.profile_mode,
        tensorboard_options=tb_options,
    )

    print(
        "  TTFT %.2f ms | TPOT %.2f±%.2f ms | %.1f tok/s | %.2f req/s | %.0f MiB"
        % (
            result["ttft_ms"]["median"],
            result["tpot_ms"]["median"],
            result["tpot_ms"]["std"],
            result["throughput_tok_s"]["median"],
            result["throughput_req_s"]["median"],
            result["memory"]["peak_allocated_mb"],
        )
    )
    eff = result.get("efficiency") or {}
    if eff.get("decode_mbu") is not None:
        gpu_busy_str = "%.0f%%" % (100 * result["gpu_time_ratio"]["median"]) if result.get("gpu_time_ratio") else "N/A"
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
    record = build_benchmark_record(
        args=args,
        model=model,
        mode_label=str(case.mode),
        cases=[result],
        correctness=correctness,
        env=env,
        peaks=peaks,
        cost=cost,
        git=git,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    output = args.output or RESULT_HISTORY_DIR / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, dirty=git["dirty"])
    print()
    print_table([result], mode=str(case.mode))


if __name__ == "__main__":
    main()
