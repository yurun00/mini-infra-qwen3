"""PyTorch Profiler 探针与 Module Scope 追踪器。"""

import os
from contextlib import contextmanager
from typing import Any

import torch

from mini_infra.profiling.analyzer import parse_op_latency_breakdown
from mini_infra.profiling.types import OpLatencyBreakdown, TensorBoardOptions

# module 级 scope：类名 -> 类别。最外层的 scope 吸收子模块（如 attention 内的 q/k norm 与投影）。
MODULE_SCOPES: dict[str, str] = {
    "Qwen3Attention": "attention_block",
    "Qwen3MLP": "mlp_block",
    "RMSNorm": "norm",
    "RotaryEmbedding": "rope",
    "Embedding": "embedding",
}


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
