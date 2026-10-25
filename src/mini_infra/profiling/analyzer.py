"""算子分类归因与耗时聚合分析器。"""

from collections.abc import Sequence
from typing import Any

from mini_infra.profiling.types import KernelInfo, OpLatencyBreakdown

# 算子类别关键字（按顺序匹配，小写）。kernel 名无法区分 attention 的 matmul 与线性层 GEMM，
# 这部分归因交给 module 级 scope 统计。
KERNEL_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("attention", ("flash", "fmha", "mem_efficient", "sdpa", "attention")),
    ("softmax", ("softmax",)),
    ("gemm", ("gemm", "gemv", "cutlass", "cublas", "nvjet", "sgemm", "matmul", "bmm", "addmm", "mm")),
    ("norm", ("rms", "layer_norm", "layernorm")),
    ("reduce", ("reduce_kernel", "reduce")),
    ("memory", ("memcpy", "memset", "copy", "catarray", "cat_", "index", "gather", "embedding")),
    ("elementwise", ("elementwise", "vectorized", "mul", "add", "silu", "sigmoid", "rsqrt", "pow", "neg")),
]


def kernel_category(name: str) -> str:
    """按算子名称关键字匹配归类；未命中则返回 'other'。"""
    lowered = name.lower()
    for category, keywords in KERNEL_CATEGORIES:
        if any(k in lowered for k in keywords):
            return category
    return "other"


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
            KernelInfo(
                name=e.key[:80],
                category=kernel_category(e.key),
                cuda_ms=e.self_device_time_total / 1000,
                share=e.self_device_time_total / total_div,
                calls=e.count,
            )
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
            KernelInfo(
                name=e.key[:80],
                category=kernel_category(e.key),
                cuda_ms=e.self_cpu_time_total / 1000,
                share=e.self_cpu_time_total / total_div,
                calls=e.count,
            )
            for e in kernels
        ]
        top_kernels = all_kernels[:top] if top is not None else all_kernels

    return OpLatencyBreakdown(
        total_ms=total_ms,
        top_kernels=top_kernels,
        time_share_by_kernel_category=dict(sorted(time_share_by_kernel_category.items(), key=lambda kv: -kv[1])),
        time_share_by_module=dict(sorted(time_share_by_module.items(), key=lambda kv: -kv[1])),
        all_kernels=all_kernels,
    )


def merge_op_latency_breakdowns(
    op_latency_breakdowns: Sequence[OpLatencyBreakdown],
    top: int | None = 10,
) -> OpLatencyBreakdown | None:
    """聚合多个 OpLatencyBreakdown（如多次重复压测或多请求批处理结果）。"""
    valid = [b for b in op_latency_breakdowns if b is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]

    count = len(valid)
    sum_total_ms = sum(b.total_ms for b in valid)

    # 1. 聚合 kernel 耗时与调用次数（优先取全量 all_kernels，若无则降级取 top_kernels）
    sum_kernel_stats: dict[tuple[str, str], dict[str, float]] = {}
    for b in valid:
        kernel_list = b.all_kernels if b.all_kernels else b.top_kernels
        for k in kernel_list:
            key = (k.name, k.category)
            if key not in sum_kernel_stats:
                sum_kernel_stats[key] = {"sum_cuda_ms": 0.0, "sum_calls": 0.0}
            sum_kernel_stats[key]["sum_cuda_ms"] += k.cuda_ms
            sum_kernel_stats[key]["sum_calls"] += k.calls

    safe_sum_total_ms = sum_total_ms if sum_total_ms > 0 else 1.0
    sorted_sum_kernel_stats = sorted(sum_kernel_stats.items(), key=lambda kv: -kv[1]["sum_cuda_ms"])
    all_kernels: list[KernelInfo] = [
        KernelInfo(
            name=name,
            category=cat,
            cuda_ms=s["sum_cuda_ms"],
            share=s["sum_cuda_ms"] / safe_sum_total_ms,
            calls=int(s["sum_calls"]),
        )
        for (name, cat), s in sorted_sum_kernel_stats
    ]
    top_kernels = all_kernels[:top] if top is not None else all_kernels

    # 2. 聚合 category 耗时占比 (按 total_ms 加权)
    safe_sum_weight = sum_total_ms if sum_total_ms > 0 else float(count)
    sum_category_ms: dict[str, float] = {}
    for b in valid:
        w = b.total_ms if sum_total_ms > 0 else 1.0
        for cat, share in b.time_share_by_kernel_category.items():
            sum_category_ms[cat] = sum_category_ms.get(cat, 0.0) + share * w

    time_share_by_kernel_category = {
        cat: ms / safe_sum_weight for cat, ms in sorted(sum_category_ms.items(), key=lambda kv: -kv[1])
    }

    # 3. 聚合 module 耗时占比 (按 total_ms 加权)
    sum_module_ms: dict[str, float] = {}
    for b in valid:
        w = b.total_ms if sum_total_ms > 0 else 1.0
        for mod, share in b.time_share_by_module.items():
            sum_module_ms[mod] = sum_module_ms.get(mod, 0.0) + share * w

    time_share_by_module = {
        mod: ms / safe_sum_weight for mod, ms in sorted(sum_module_ms.items(), key=lambda kv: -kv[1])
    }

    return OpLatencyBreakdown(
        total_ms=sum_total_ms,
        top_kernels=top_kernels,
        time_share_by_kernel_category=time_share_by_kernel_category,
        time_share_by_module=time_share_by_module,
        all_kernels=all_kernels,
    )
