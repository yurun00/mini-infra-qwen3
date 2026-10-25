"""性能剖析领域数据类型定义。"""

from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass(frozen=True)
class KernelInfo:
    name: str
    category: str
    cuda_ms: float
    share: float
    calls: int


@dataclass(frozen=True)
class OpLatencyBreakdown:
    total_ms: float
    top_kernels: list[KernelInfo]
    time_share_by_kernel_category: dict[str, float]
    time_share_by_module: dict[str, float]
    all_kernels: list[KernelInfo] = field(default_factory=list)
