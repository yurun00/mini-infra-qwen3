"""算子性能剖析与耗时归因模块。

提供模块级 Scope 打点、算子分类与 PyTorch Profiler 的封装，用于定位推理瓶颈。
"""

import os
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
from typing_extensions import NotRequired


@dataclass(frozen=True)
class TensorBoardOptions:
    """TensorBoard 性能分析日志导出配置选项。

    参数:
        log_dir: 日志输出目录（如 "./log/profiler_trace"，默认 "./log"）
        record_shapes: 是否记录每个算子的输入 Tensor Shape（默认 True）
        profile_memory: 是否记录 GPU 显存分配与释放轨迹曲线（默认 True）
        with_stack: 是否记录 Python 代码调用栈回溯（默认 True）
    """

    log_dir: str | Path = "./log"
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True


class KernelInfo(TypedDict):
    name: str
    category: str
    cuda_ms: float
    share: float
    calls: int


class OpLatencyBreakdown(TypedDict):
    total_ms: float
    top_kernels: list[KernelInfo]
    time_share_by_kernel_category: dict[str, float]
    time_share_by_module: dict[str, float]
    all_kernels: NotRequired[list[KernelInfo]]


# 算子类别关键字（按顺序匹配，小写）。kernel 名无法区分 attention 的 matmul 与线性层 GEMM，
# 这部分归因交给下面的 module 级 scope 统计。
KERNEL_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("attention", ("flash", "fmha", "mem_efficient", "sdpa", "attention")),
    ("softmax", ("softmax",)),
    ("gemm", ("gemm", "gemv", "cutlass", "cublas", "nvjet", "sgemm", "matmul", "bmm", "addmm", "mm")),
    ("norm", ("rms", "layer_norm", "layernorm")),
    ("reduce", ("reduce_kernel", "reduce")),
    ("memory", ("memcpy", "memset", "copy", "catarray", "cat_", "index", "gather", "embedding")),
    ("elementwise", ("elementwise", "vectorized", "mul", "add", "silu", "sigmoid", "rsqrt", "pow", "neg")),
]

# module 级 scope：类名 -> 类别。最外层的 scope 吸收子模块（如 attention 内的 q/k norm 与投影）。
MODULE_SCOPES: dict[str, str] = {
    "Qwen3Attention": "attention_block",
    "Qwen3MLP": "mlp_block",
    "RMSNorm": "norm",
    "RotaryEmbedding": "rope",
    "Embedding": "embedding",
}


def kernel_category(name: str) -> str:
    """按算子名称关键字匹配归类；未命中则返回 'other'。"""
    lowered = name.lower()
    for category, keywords in KERNEL_CATEGORIES:
        if any(k in lowered for k in keywords):
            return category
    return "other"


def install_scope_hooks(model: torch.nn.Module) -> list[Any]:
    """给关键模块挂 record_function 范围，使 profiler 能把耗时归到 attention/mlp/norm/lm_head 等模块。"""
    from torch.profiler import record_function

    handles = []
    stack: list[Any] = []

    def pre(label: str):
        def hook(_module: torch.nn.Module, _inputs: Any) -> None:
            if stack:  # 已在某个 scope 内，不重复计数
                stack.append(None)
                return
            ctx = record_function(f"mi_scope::{label}")
            ctx.__enter__()
            stack.append(ctx)

        return hook

    def post(_module: torch.nn.Module, _inputs: Any, _output: Any) -> None:
        ctx = stack.pop()
        if ctx is not None:
            ctx.__exit__(None, None, None)

    for name, module in model.named_modules():
        label = MODULE_SCOPES.get(type(module).__name__) or ("lm_head" if name == "lm_head" else None)
        if label:
            handles.append(module.register_forward_pre_hook(pre(label)))
            handles.append(module.register_forward_hook(post))
    return handles


_install_scope_hooks = install_scope_hooks


def parse_op_latency_breakdown(prof: Any, top: int | None = 10) -> OpLatencyBreakdown:
    """从 PyTorch profiler 收集到的事件中解析算子分类与耗时归因。"""
    averages = list(prof.key_averages())
    has_cuda_time = any(getattr(e, "self_device_time_total", 0) > 0 for e in averages)

    if has_cuda_time:
        kernels = [
            e for e in averages if getattr(e, "self_device_time_total", 0) > 0 and not e.key.startswith("mi_scope::")
        ]
        total_time = sum(e.self_device_time_total for e in kernels)
        total_ms = total_time / 1000
        total_div = total_time or 1
        kernels.sort(key=lambda e: e.self_device_time_total, reverse=True)

        time_share_by_kernel_category: dict[str, float] = {}
        for e in kernels:
            c = kernel_category(e.key)
            time_share_by_kernel_category[c] = (
                time_share_by_kernel_category.get(c, 0.0) + e.self_device_time_total / total_div
            )

        scopes = [e for e in averages if e.key.startswith("mi_scope::")]
        scope_total = sum(e.device_time_total for e in scopes) or 1
        time_share_by_module = {e.key.split("::", 1)[1]: e.device_time_total / scope_total for e in scopes}

        all_kernels: list[KernelInfo] = [
            {
                "name": e.key[:80],
                "category": kernel_category(e.key),
                "cuda_ms": e.self_device_time_total / 1000,
                "share": e.self_device_time_total / total_div,
                "calls": e.count,
            }
            for e in kernels
        ]
        top_kernels = all_kernels[:top] if top is not None else all_kernels
    else:
        # Fallback to CPU time when no device time is recorded (e.g., CPU-only inference or unit tests)
        kernels = [
            e for e in averages if getattr(e, "self_cpu_time_total", 0) > 0 and not e.key.startswith("mi_scope::")
        ]
        total_time = sum(e.self_cpu_time_total for e in kernels)
        total_ms = total_time / 1000
        total_div = total_time or 1
        kernels.sort(key=lambda e: e.self_cpu_time_total, reverse=True)

        time_share_by_kernel_category = {}
        for e in kernels:
            c = kernel_category(e.key)
            time_share_by_kernel_category[c] = (
                time_share_by_kernel_category.get(c, 0.0) + e.self_cpu_time_total / total_div
            )

        scopes = [e for e in averages if e.key.startswith("mi_scope::")]
        scope_total = sum(e.cpu_time_total for e in scopes) or 1
        time_share_by_module = {e.key.split("::", 1)[1]: e.cpu_time_total / scope_total for e in scopes}

        all_kernels = [
            {
                "name": e.key[:80],
                "category": kernel_category(e.key),
                "cuda_ms": e.self_cpu_time_total / 1000,
                "share": e.self_cpu_time_total / total_div,
                "calls": e.count,
            }
            for e in kernels
        ]
        top_kernels = all_kernels[:top] if top is not None else all_kernels

    return {
        "total_ms": total_ms,
        "top_kernels": top_kernels,
        "time_share_by_kernel_category": dict(sorted(time_share_by_kernel_category.items(), key=lambda kv: -kv[1])),
        "time_share_by_module": dict(sorted(time_share_by_module.items(), key=lambda kv: -kv[1])),
        "all_kernels": all_kernels,
    }


def merge_op_latency_breakdowns(
    op_latency_breakdowns: Sequence[OpLatencyBreakdown],
    top: int | None = 10,
) -> OpLatencyBreakdown | None:
    """聚合多个 OpLatencyBreakdown（如多次重复压测或多请求批处理结果）。

    - total_ms: 算子总耗时的累加值
    - top_kernels: 按 (name, category) 聚合所有 breakdown 的全量 kernel 后全局排序，取前 top 个全局热点算子
    - all_kernels: 全量聚合后的所有算子列表（按耗时倒序）
    - time_share_by_kernel_category: 按每次 run 的 total_ms 加权归一化
    - time_share_by_module: 按每次 run 的 total_ms 加权归一化
    """
    valid = [b for b in op_latency_breakdowns if b]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    count = len(valid)
    sum_total_ms = sum(b.get("total_ms", 0.0) for b in valid)

    # 1. 聚合 kernel 耗时与调用次数（优先取全量 all_kernels，若无则降级取 top_kernels）
    sum_kernel_stats: dict[tuple[str, str], dict[str, float]] = {}
    for b in valid:
        kernel_list = b.get("all_kernels") or b.get("top_kernels", [])
        for k in kernel_list:
            key = (k["name"], k["category"])
            if key not in sum_kernel_stats:
                sum_kernel_stats[key] = {"sum_cuda_ms": 0.0, "sum_calls": 0.0}
            sum_kernel_stats[key]["sum_cuda_ms"] += k.get("cuda_ms", 0.0)
            sum_kernel_stats[key]["sum_calls"] += k.get("calls", 0)

    safe_sum_total_ms = sum_total_ms if sum_total_ms > 0 else 1.0
    sorted_sum_kernel_stats = sorted(sum_kernel_stats.items(), key=lambda kv: -kv[1]["sum_cuda_ms"])
    all_kernels: list[KernelInfo] = [
        {
            "name": name,
            "category": cat,
            "cuda_ms": s["sum_cuda_ms"],
            "share": s["sum_cuda_ms"] / safe_sum_total_ms,
            "calls": int(s["sum_calls"]),
        }
        for (name, cat), s in sorted_sum_kernel_stats
    ]
    top_kernels = all_kernels[:top] if top is not None else all_kernels

    # 2. 聚合 category 耗时占比 (按 total_ms 加权)
    safe_sum_weight = sum_total_ms if sum_total_ms > 0 else float(count)
    sum_category_ms: dict[str, float] = {}
    for b in valid:
        w = b.get("total_ms", 0.0) if sum_total_ms > 0 else 1.0
        for cat, share in b.get("time_share_by_kernel_category", {}).items():
            sum_category_ms[cat] = sum_category_ms.get(cat, 0.0) + share * w

    time_share_by_kernel_category = {
        cat: ms / safe_sum_weight for cat, ms in sorted(sum_category_ms.items(), key=lambda kv: -kv[1])
    }

    # 3. 聚合 module 耗时占比 (按 total_ms 加权)
    sum_module_ms: dict[str, float] = {}
    for b in valid:
        w = b.get("total_ms", 0.0) if sum_total_ms > 0 else 1.0
        for mod, share in b.get("time_share_by_module", {}).items():
            sum_module_ms[mod] = sum_module_ms.get(mod, 0.0) + share * w

    time_share_by_module = {
        mod: ms / safe_sum_weight for mod, ms in sorted(sum_module_ms.items(), key=lambda kv: -kv[1])
    }

    return {
        "total_ms": sum_total_ms,
        "top_kernels": top_kernels,
        "time_share_by_kernel_category": time_share_by_kernel_category,
        "time_share_by_module": time_share_by_module,
        "all_kernels": all_kernels,
    }


class ProfileCollector:
    """包装已完成的 Profiler 实例，提供指标解析方法。"""

    def __init__(self, prof: Any):
        self.prof = prof

    def get_latency_breakdown(self, top: int | None = 10) -> OpLatencyBreakdown:
        return parse_op_latency_breakdown(self.prof, top=top)


@contextmanager
def profile_scope(
    model: torch.nn.Module,
    device: torch.device | None = None,
    tensorboard_options: TensorBoardOptions | None = None,
):
    """上下文管理器：为 model 安装 scope hooks，并在进入和退出时控制 torch.profiler。

    参数:
        model: 需要被监控的 PyTorch 模型
        device: 计算设备
        tensorboard_options: 可选。传入 TensorBoardOptions 结构体对象；
                             若指定则在 trace 就绪时自动保存为 TensorBoard 格式文件；
                             若为 None（默认）则不向磁盘写入任何 log 文件。
    """
    from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

    is_cuda = (device is not None and device.type == "cuda") or (device is None and torch.cuda.is_available())
    activities = [ProfilerActivity.CPU]
    if is_cuda:
        activities.append(ProfilerActivity.CUDA)

    on_trace_ready = None
    if tensorboard_options is not None:
        os.makedirs(str(tensorboard_options.log_dir), exist_ok=True)
        on_trace_ready = tensorboard_trace_handler(str(tensorboard_options.log_dir))

    handles = install_scope_hooks(model)
    try:
        with profile(
            activities=activities,
            on_trace_ready=on_trace_ready,
            record_shapes=tensorboard_options.record_shapes if tensorboard_options is not None else False,
            profile_memory=tensorboard_options.profile_memory if tensorboard_options is not None else False,
            with_stack=tensorboard_options.with_stack if tensorboard_options is not None else False,
        ) as prof:
            yield ProfileCollector(prof)
    finally:
        for h in handles:
            h.remove()
