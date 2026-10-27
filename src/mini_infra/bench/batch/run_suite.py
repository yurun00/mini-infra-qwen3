"""套件基准测试（run_suite）：一次调用执行整套 BenchmarkSuite 的全部用例。

配置全部来自 suites.py（包括 batch_sizes, prompt_length, max_new_tokens, warmup, repeats, modes 等）。
结果默认落盘至 benchmarks/suites/<suite_name>/<commit>_<tag>.json。

用法示例：
    mini-infra-bench-suite --tag dev --suite serial-vs-fixed-batch-throughput
"""

import argparse
import sys

import torch

from mini_infra.bench.batch.run_case import run_case
from mini_infra.bench.batch.suites import SUITES, get_suite
from mini_infra.bench.common import (
    SUITES_DIR,
    add_common_args,
    build_benchmark_record,
    check_correctness,
    format_lengths,
    git_info,
    gpu_info,
    model_cost,
    print_case_summary,
    print_table,
    require_cuda,
    resolve_peak_specs,
    save_benchmark_record,
)
from mini_infra.model import Qwen3Tokenizer, load_model, resolve_model_directory
from mini_infra.profiling import TensorBoardOptions


def parse_args():
    p = argparse.ArgumentParser(description="mini-infra-qwen3 benchmark suite runner")
    add_common_args(p)
    p.add_argument(
        "--suite",
        choices=sorted(SUITES),
        required=True,
        help="运行 suites.py 中注册的命名 suite（配置来自该文件，结果写入 benchmarks/suites/<name>/）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = require_cuda()

    suite = get_suite(args.suite)
    modes = suite.modes()
    cases = suite.cases()
    warmup = args.warmup if args.warmup is not None else suite.warmup
    repeats = args.repeats if args.repeats is not None else suite.repeats
    mode_label = ",".join(str(m) for m in modes)

    model_dir = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_dir)
    model = load_model(model_dir, device)
    vocab = model.config.vocab_size

    correctness = {str(m): check_correctness(model, tokenizer, m, device, update_golden=False) for m in modes}
    gate_failed = any(c["passed"] is False for c in correctness.values())
    if gate_failed and not args.allow_mismatch:
        sys.exit("正确性检查失败：输出与 golden 不一致，拒绝记录性能结果（--allow-mismatch 可忽略）")

    env = gpu_info()
    peaks = resolve_peak_specs(env["gpu"], args.peak_bandwidth_gbs, args.peak_tflops)
    cost = model_cost(model)
    if None in peaks.values():
        print(f"[warn] 未识别 GPU 峰值规格 {env['gpu']}，对应 MBU/MFU 将为空（可用 --peak-* 指定）")
    tb_options = TensorBoardOptions(log_dir=args.tensorboard_dir) if args.tensorboard_dir is not None else None

    case_results = []
    for case in cases:
        print(
            f"[bench-suite] mode={case.mode} batch={case.batch_size} prompt={format_lengths(case.prompt_len)} "
            f"max_new_tokens={format_lengths(case.max_new_tokens)} ...",
            flush=True,
        )
        try:
            case_result = run_case(
                model,
                case.mode,
                list(case.prompt_len) if isinstance(case.prompt_len, tuple) else case.prompt_len,
                list(case.max_new_tokens) if isinstance(case.max_new_tokens, tuple) else case.max_new_tokens,
                case.batch_size,
                warmup,
                repeats,
                vocab,
                device,
                cost=cost,
                peaks=peaks,
                profile_mode=args.profile_mode,
                tensorboard_options=tb_options,
            )
        except torch.cuda.OutOfMemoryError:
            print("  OOM，跳过")
            torch.cuda.empty_cache()
            continue
        case_results.append(case_result)
        print_case_summary(case_result)

    git = git_info()
    record = build_benchmark_record(
        args=args,
        model=model,
        mode_label=mode_label,
        cases=case_results,
        correctness=correctness,
        env=env,
        peaks=peaks,
        cost=cost,
        git=git,
        warmup=warmup,
        repeats=repeats,
        suite_info=suite.to_dict(),
    )

    default_dir = SUITES_DIR / suite.name
    output = args.output or default_dir / f"{git['commit']}_{args.tag}.json"
    save_benchmark_record(output, record, dirty=git["dirty"])
    if case_results:
        print()
        print_table(case_results, mode=mode_label)


if __name__ == "__main__":
    main()
