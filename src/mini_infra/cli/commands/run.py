"""CLI 子命令：mini-infra run

手写模型单请求/批处理推理与特性对照。
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from mini_infra.core import (
    KV_CACHE_KINDS,
    SCHEDULING_POLICY_KINDS,
    DenseCacheConfig,
    NoCacheConfig,
    PagedCacheConfig,
    PrefillFirstConfig,
    kv_cache_config_from_str,
    make_requests,
    scheduler_config_from_str,
)
from mini_infra.device import synchronize
from mini_infra.engine import (
    BatchingMethod,
    EngineConfig,
    FeatureFlag,
    GenerationResult,
    Qwen3Tokenizer,
    ScheduleResult,
    run_scheduler,
)
from mini_infra.entrypoints import GenerationOptions, generate
from mini_infra.model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    load_model,
    resolve_model_directory,
)
from mini_infra.profiling import TensorBoardOptions


def _parse_feature_flags(value: str) -> frozenset[FeatureFlag]:
    """解析逗号分隔的 FeatureFlag 列表；空串或 none 表示不启用任何特性。"""
    if value in ("", "none"):
        return frozenset()
    return frozenset(FeatureFlag.from_str(item) for item in value.split(","))


# 合成 EngineConfig 的全部 CLI 参数（flag -> argparse kwargs），也是 --compare-feature 可选的对比维度。
# 新增 EngineConfig 字段时只需在此登记并在 build_engine_config 中消费，对比能力自动获得。
_ENGINE_CONFIG_ARGS: dict[str, dict[str, Any]] = {
    "--caching-method": {
        "default": DenseCacheConfig.kind,
        "choices": list(KV_CACHE_KINDS),
        "help": "指定基准 KV Cache 策略（默认: %(default)s）",
    },
    "--kv-block-size": {
        "type": int,
        "default": PagedCacheConfig.block_size,
        "help": "Paged KV Cache 每个物理块容纳的 Token 数，仅在 paged-kv-cache 策略下生效（默认: %(default)s）",
    },
    "--num-kv-blocks": {
        "type": int,
        "default": None,
        "help": "Paged KV Cache physical block pool 大小（含 null block），默认按工作负载自动推导为不会耗尽的最小值",
    },
    "--batching-method": {
        "default": BatchingMethod.NONE.value,
        "choices": [m.value for m in BatchingMethod],
        "help": "指定基准批处理策略（默认: %(default)s）",
    },
    "--max-running-requests": {
        "type": int,
        "default": None,
        "help": "调度器最大并发请求数，仅对 fixed/continuous 调度路径生效（默认: 不限，即 prompt 数量）",
    },
    "--scheduling-policy": {
        "default": PrefillFirstConfig.kind,
        "choices": list(SCHEDULING_POLICY_KINDS),
        "help": "迭代级调度策略，token-budget 仅支持 continuous-batching（默认: %(default)s）",
    },
    "--token-budget": {
        "type": int,
        "default": None,
        "help": "token-budget 策略下单次 Prefill 的 padded prompt token 上限，须不小于并发上限",
    },
    "--feature-flags": {
        "type": _parse_feature_flags,
        "default": frozenset(),
        "help": f"逗号分隔的启用特性标志，可选: {[f.value for f in FeatureFlag]}（默认: 不启用）",
    },
}


def add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """向指定 parser 添加 run 命令的所有参数。"""
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="模型 ID 或本地目录")
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["用一句话解释 Prefill。", "用一句话解释 Decode。"],
        help="输入提示词列表（默认包含 Prefill 与 Decode 两条提示词）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        nargs="+",
        default=[32],
        metavar="N",
        help="输出 Token 预算：提供 1 个值（所有请求共用）或与 prompts 数量一致的逐请求预算（默认: 32）",
    )
    parser.add_argument("--warmup", type=int, default=1)
    for flag, kwargs in _ENGINE_CONFIG_ARGS.items():
        parser.add_argument(flag, **kwargs)
    parser.add_argument(
        "--compare-feature",
        nargs=3,
        default=None,
        metavar=("ARG", "VALUE1", "VALUE2"),
        help=(
            "对比单个 EngineConfig 参数的两个取值：其余参数保持不变，分别合成两个 EngineConfig 后对照。"
            f"ARG 可选: {[flag.removeprefix('--') for flag in _ENGINE_CONFIG_ARGS]}；"
            "默认值为不限的整型参数可传 none（如 max-running-requests none 2）"
        ),
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否启用底层 PyTorch Profiler 采集算子与 Kernel 耗时明细（默认: False）",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=10,
        help="展示耗时前 K 个的 CUDA Kernel（默认: 10）",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="导出 TensorBoard 格式性能剖析日志的目录（需配合 --profile 使用）",
    )
    return parser


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在根/上级 subparsers 中注册 run 子命令。"""
    parser = subparsers.add_parser("run", help="运行手写模型单请求/批处理推理与特性对照")
    add_run_args(parser)
    parser.set_defaults(func=_handle_run)
    return parser


def build_engine_config(args: argparse.Namespace) -> EngineConfig:
    """由扁平的 CLI 参数合成嵌套的 EngineConfig（CLI 参数到配置结构的唯一映射点）。"""
    return EngineConfig(
        kv_cache_config_from_str(args.caching_method, block_size=args.kv_block_size, num_blocks=args.num_kv_blocks),
        batching_method=BatchingMethod.from_str(args.batching_method),
        max_running_requests=args.max_running_requests,
        flags=args.feature_flags,
        scheduler_config=scheduler_config_from_str(args.scheduling_policy, token_budget=args.token_budget),
    )


def _parse_engine_arg_value(flag: str, raw: str) -> Any:
    """按参数登记的 type / choices 解析对比取值；默认值为 None 的参数接受 none 表示不设置。"""
    kwargs = _ENGINE_CONFIG_ARGS[flag]
    if raw == "none" and kwargs["default"] is None:
        return None
    value = kwargs.get("type", str)(raw)
    choices = kwargs.get("choices")
    if choices is not None and value not in choices:
        raise ValueError(f"--compare-feature {flag.removeprefix('--')} 的取值 '{raw}' 非法，可选: {choices}")
    return value


def resolve_compare_feature(
    args: argparse.Namespace,
) -> tuple[tuple[EngineConfig, EngineConfig], str]:
    """解析 --compare-feature ARG VALUE1 VALUE2：ARG 分别取两个值、其余参数不变，合成两个 EngineConfig。

    返回 ((config1, config2), 对比描述)。若两个取值在当前其余参数下合成出相同配置（如 dense-kv-cache 下
    对比 kv-block-size），说明该参数不生效，直接报错而非输出无意义的对照。
    """
    arg_name, raw1, raw2 = args.compare_feature
    flag = f"--{arg_name}"
    if flag not in _ENGINE_CONFIG_ARGS:
        valid = [f.removeprefix("--") for f in _ENGINE_CONFIG_ARGS]
        raise ValueError(f"--compare-feature 参数 '{arg_name}' 不是 EngineConfig 参数，可选: {valid}")
    dest = arg_name.replace("-", "_")
    config1, config2 = (
        build_engine_config(argparse.Namespace(**{**vars(args), dest: _parse_engine_arg_value(flag, raw)}))
        for raw in (raw1, raw2)
    )
    if config1 == config2:
        raise ValueError(
            f"--compare-feature {arg_name} 的两个取值 '{raw1}' / '{raw2}' "
            f"在当前其余参数下合成了相同的 EngineConfig（{config1}），该参数不生效"
        )
    return (config1, config2), f"{arg_name}: {raw1} vs {raw2}"


def resolve_budgets(max_new_tokens: Sequence[int], num_prompts: int) -> int | list[int]:
    """将 --max-new-tokens 归一化：1 个值广播为 int，与 prompts 等长则逐请求预算，否则报错。"""
    budgets = list(max_new_tokens)
    if any(b < 1 for b in budgets):
        raise ValueError("--max-new-tokens 必须大于 0")
    if len(budgets) == 1:
        return budgets[0]
    if len(budgets) == num_prompts:
        return budgets
    raise ValueError(
        f"--max-new-tokens 应提供 1 个值或与 prompts 数量一致（{num_prompts}），当前传入 {len(budgets)} 个"
    )


def shrink_budgets(budgets: int | list[int], limit: int) -> int | list[int]:
    """将预算截断到不超过 limit（用于 warmup）。"""
    if isinstance(budgets, int):
        return min(budgets, limit)
    return [min(b, limit) for b in budgets]


def uses_scheduler(configs: Sequence[EngineConfig]) -> bool:
    """只要任一配置使用 continuous batching 或 paged-kv-cache，所有结果都统一由 run_scheduler 产出（指标可比）。"""
    return any(
        c.batching_method is BatchingMethod.CONTINUOUS or isinstance(c.kv_cache_config, PagedCacheConfig)
        for c in configs
    )


def run_config(
    config: EngineConfig,
    model: torch.nn.Module,
    input_ids,
    sequences: Sequence[Sequence[int]],
    budgets: int | list[int],
    options: GenerationOptions,
    via_scheduler: bool,
) -> GenerationResult:
    """执行单个 EngineConfig：via_scheduler 为 True 时统一由 run_scheduler 产出；否则沿用 generate。"""
    if via_scheduler:
        if isinstance(config.kv_cache_config, NoCacheConfig):
            raise ValueError("调度路径依赖 KV Cache，请将 --caching-method 设为 dense-kv-cache 或 paged-kv-cache")
        if options.capture_logits or options.profile:
            raise ValueError("调度路径暂不支持 capture_logits/profile，请去掉 --profile 或 continuous 对比")
        requests = make_requests(sequences, budgets)
        return run_scheduler(
            config,
            model,
            requests,
            options.eos_token_id,
            pad_token_id=options.pad_token_id,
            stop_on_eos=options.stop_on_eos,
        )
    return generate(config, model, input_ids, budgets, options=options)


def _print_schedule_metrics(result: ScheduleResult) -> None:
    sm = result.schedule_metrics
    print("Makespan: %.2f ms" % sm.makespan_ms)
    print("排队延迟 p50/p95: %.2f / %.2f ms" % (sm.queue_ms_p50, sm.queue_ms_p95))
    print("TTFT p50/p95: %.2f / %.2f ms" % (sm.ttft_ms_p50, sm.ttft_ms_p95))
    print("ITL p50/p95/max: %.2f / %.2f / %.2f ms" % (sm.itl_ms_p50, sm.itl_ms_p95, sm.itl_ms_max))
    print("Cache 管理耗时占比: %.2f%%" % (sm.cache_management_fraction * 100))
    info = result.batch_info
    if info is not None and info.running_capacity_utilization is not None:
        print("并发容量利用率: %.2f%%" % (info.running_capacity_utilization * 100))


def print_result(config: EngineConfig, result, tokenizer: Qwen3Tokenizer, tensorboard_dir: Path | None = None) -> None:
    print("\n[%s]" % config)
    for seq in result.sequences:
        print("请求 %d Token IDs: %s" % (seq.request_idx, seq.token_ids))
        print("请求 %d 输出: %s" % (seq.request_idx, tokenizer.decode(seq.token_ids, skip_special_tokens=True)))

    print("端到端 Wall-clock 延迟: %.2f ms" % result.metrics.e2e_wall_ms)
    print("端到端模型计算延迟: %.2f ms" % result.metrics.e2e_model_ms)
    print("请求吞吐: %.3f requests/s" % result.metrics.throughput_req_s)
    print("输出吞吐: %.3f tokens/s" % result.metrics.throughput_tok_s)

    # 细粒度执行与显存指标（包含 batch 及超集分位数分析）
    if result.metrics.ttft_ms is not None:
        print("Prefill 延迟: %.2f ms" % result.metrics.ttft_ms)
    if result.metrics.mean_tpot_ms is not None:
        print("平均 Decode 迭代延迟: %.2f ms" % result.metrics.mean_tpot_ms)
    if result.metrics.p50_tpot_ms is not None:
        print("P50 Decode 延迟: %.2f ms" % result.metrics.p50_tpot_ms)
    if result.metrics.p95_tpot_ms is not None:
        print("P95 Decode 延迟: %.2f ms" % result.metrics.p95_tpot_ms)
    if config.is_batch and result.batch_info is not None:
        print("Padding 有效率: %.2f%%" % (result.batch_info.padding_efficiency * 100))
        print("batch slot 利用率: %.2f%%" % (result.batch_info.slot_utilization * 100))
    if isinstance(result, ScheduleResult):
        _print_schedule_metrics(result)
    kv_profile = result.memory.kv_cache
    print("峰值显存: %.2f MiB" % result.memory.peak_allocated_mb)
    print("最终 KV Cache: %.2f MiB" % kv_profile.peak_live_kv_cache_mb)
    if kv_profile.peak_kv_cache_pool_mb > 0:
        print("峰值 KV Cache Pool: %.2f MiB" % kv_profile.peak_kv_cache_pool_mb)
    if kv_profile.paged_cache is not None:
        pc = kv_profile.paged_cache
        print(
            "Paged physical block pool: block_size=%d, 峰值使用块=%d/%d"
            % (pc.block_size, pc.peak_used_blocks, pc.num_blocks)
        )
    decode_times = result.trace.step_latencies_ms[1:]
    if decode_times:
        print("逐步 Decode 延迟(ms):", [round(value, 2) for value in decode_times])

    if result.op_breakdown:
        breakdown = result.op_breakdown
        print("\n--- [%s] 算子与模块耗时分析 (Profiler) ---" % config)
        if breakdown.time_share_by_module:
            print("各模块耗时占比 (Module Scope):")
            for mod, share in breakdown.time_share_by_module.items():
                print("  %-20s: %6.2f%%" % (mod, share * 100))
        if breakdown.time_share_by_kernel_category:
            print("算子类型耗时占比 (Kernel Category):")
            for cat, share in breakdown.time_share_by_kernel_category.items():
                print("  %-20s: %6.2f%%" % (cat, share * 100))
        top_k = breakdown.top_kernels
        if top_k:
            print("Top-%d CUDA Kernels:" % len(top_k))
            for rank, k in enumerate(top_k, 1):
                print(
                    "  %2d. %-45s | %8.3f ms | %6.2f%% | %4d 次"
                    % (rank, k.name[:45], k.cuda_ms, k.share * 100, k.calls)
                )
    if tensorboard_dir is not None:
        print("TensorBoard Trace 目录: %s" % tensorboard_dir)


def _handle_run(args: argparse.Namespace) -> int:
    """执行 run 命令。"""
    if args.max_running_requests is not None and args.max_running_requests < 1:
        raise ValueError("--max-running-requests 必须大于 0")
    if args.warmup < 0:
        raise ValueError("--warmup 不能小于 0")
    # 先合成配置再加载模型：非法参数组合在加载权重前即报错
    compare_feature_name = None
    if args.compare_feature:
        configs, compare_feature_name = resolve_compare_feature(args)
    else:
        configs = (build_engine_config(args),)
    if not torch.cuda.is_available():
        raise RuntimeError("本实验需要可用的 NVIDIA GPU")

    device = torch.device("cuda")
    model_directory = resolve_model_directory(args.model, args.revision)
    tokenizer = Qwen3Tokenizer(model_directory)

    prompts = args.prompts
    sequences = [tokenizer.encode_chat_prompt(prompt) for prompt in prompts]
    if len(sequences) == 1:
        input_ids = torch.tensor(sequences, dtype=torch.long, device=device)
    else:
        input_ids = sequences

    budgets = resolve_budgets(args.max_new_tokens, len(sequences))
    model = load_model(model_directory, device)

    via_scheduler = uses_scheduler(configs)
    if args.max_running_requests is not None and not via_scheduler:
        print("[warn] --max-running-requests 仅对调度路径（含 continuous-batching）生效，当前对比未启用，将被忽略。")

    for cfg in configs:
        warmup_opts = GenerationOptions(
            eos_token_id=tokenizer.eos_token_id,
        )
        for _ in range(args.warmup):
            run_config(cfg, model, input_ids, sequences, shrink_budgets(budgets, 2), warmup_opts, via_scheduler)
        synchronize(device)

    if isinstance(input_ids, torch.Tensor):
        print(f"Prompt 数量: {input_ids.shape[0]}，Token 数: {input_ids.shape[1]}")
    else:
        prompt_lens = [len(s) for s in sequences]
        print(f"Prompt 数量: {len(sequences)}，各请求 Token 数: {prompt_lens}")

    if args.tensorboard_dir is not None and not args.profile:
        print("[warn] 指定了 --tensorboard-dir 但未开启 --profile，将不会采集并导出 TensorBoard 日志。")

    do_profile = args.profile
    results = {}
    for cfg in configs:
        if do_profile and args.tensorboard_dir is not None:
            tb_dir = args.tensorboard_dir / str(cfg) if args.compare_feature else args.tensorboard_dir
            tb_options = TensorBoardOptions(log_dir=tb_dir)
        else:
            tb_dir = None
            tb_options = None

        run_opts = GenerationOptions(
            eos_token_id=tokenizer.eos_token_id,
            profile=do_profile,
            top_kernels=args.top_kernels,
            tensorboard_options=tb_options,
        )
        results[cfg] = run_config(cfg, model, input_ids, sequences, budgets, run_opts, via_scheduler)
        print_result(cfg, results[cfg], tokenizer, tensorboard_dir=tb_dir)

    if len(configs) == 2:
        baseline = results[configs[0]]
        optimized = results[configs[1]]
        print(f"\n[特性 '{compare_feature_name}' 对照结论]")
        same_tokens = [s.token_ids for s in baseline.sequences] == [s.token_ids for s in optimized.sequences]
        print("逐请求 Token 一致:", same_tokens)
        baseline_decode = baseline.metrics.mean_tpot_ms
        optimized_decode = optimized.metrics.mean_tpot_ms
        if baseline_decode is not None and optimized_decode is not None and optimized_decode > 0:
            print("平均 Decode 加速比: %.2fx" % (baseline_decode / optimized_decode))
        if baseline.metrics.e2e_wall_ms > 0 and optimized.metrics.e2e_wall_ms > 0:
            print(
                "端到端 Wall-clock 耗时: %.2f ms -> %.2f ms"
                % (baseline.metrics.e2e_wall_ms, optimized.metrics.e2e_wall_ms)
            )
    return 0


__all__ = [
    "add_run_args",
    "build_engine_config",
    "print_result",
    "register_parser",
    "resolve_budgets",
    "resolve_compare_feature",
    "run_config",
    "shrink_budgets",
    "uses_scheduler",
]
