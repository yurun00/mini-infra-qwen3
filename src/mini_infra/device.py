"""通用硬件设备同步、单步计时与显存统计工具。"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from types import TracebackType

import torch

__all__ = [
    "PendingStepLatencies",
    "StepTimer",
    "allocated_memory_mb",
    "peak_memory_mb",
    "reset_peak_memory",
    "synchronize",
]


def synchronize(device: torch.device) -> None:
    """CUDA 设备上等待所有已提交 kernel 完成；其他设备为空操作。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StepTimer:
    """单步即时计时器：with 块退出后立即结算 elapsed_ms。

    用于调度主循环中各步（Prefill/Decode）纯模型前向耗时的独立测量。
    CUDA 下进入前与退出后各做一次硬件同步再结算（保证测得纯算子耗时）；非 CUDA 下使用 CPU 墙钟。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.elapsed_ms: float = 0.0
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self._wall_t0_ms: float = 0.0

    def __enter__(self) -> "StepTimer":
        if self.device.type == "cuda":
            synchronize(self.device)
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._wall_t0_ms = time.perf_counter() * 1000.0
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.device.type == "cuda":
            assert self._end_event is not None and self._start_event is not None
            self._end_event.record()
            synchronize(self.device)
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self.elapsed_ms = time.perf_counter() * 1000.0 - self._wall_t0_ms


class PendingStepLatencies:
    """跨步延迟异步收集器：在循环中通过 measure_step() 连续打点，循环结束后调用 settle() 一次性结算。

    用于齐步自回归（Lockstep）等追求纯吞吐的执行循环。
    CUDA 下记录 Event 对并在 settle() 一次性同步结算，避免在每步 Decode 中间同步破坏 kernel 发射流水线；
    非 CUDA 下内部记录 CPU 墙钟。对外彻底屏蔽硬件差异，严禁底层 Event 跨层泄漏。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._is_cuda = device.type == "cuda"
        self._cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_times_ms: list[float] = []

    @contextmanager
    def measure_step(self) -> Generator[None, None, None]:
        """单步打点上下文管理器。CUDA 下记录 Event，CPU 下记录墙钟，绝不当场同步。"""
        if self._is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._cuda_events.append((start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._cpu_times_ms.append((time.perf_counter() - t0) * 1000.0)

    def settle(self) -> list[float]:
        """循环结束后调用：等待硬件排空，批量转换为每步耗时（ms）。"""
        if self._is_cuda:
            if not self._cuda_events:
                return []
            synchronize(self.device)
            latencies = [start.elapsed_time(end) for start, end in self._cuda_events]
            self._cuda_events.clear()
            return latencies
        else:
            latencies = list(self._cpu_times_ms)
            self._cpu_times_ms.clear()
            return latencies


def reset_peak_memory(device: torch.device) -> None:
    """重置 CUDA 峰值显存统计；非 CUDA 设备为空操作。"""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def allocated_memory_mb(device: torch.device) -> float:
    """返回当前已分配显存（MB）；非 CUDA 设备返回 0.0。"""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024**2


def peak_memory_mb(device: torch.device) -> tuple[float, float]:
    """返回 (峰值已分配, 峰值已预留) 显存（MB）；非 CUDA 设备返回 (0.0, 0.0)。"""
    if device.type != "cuda":
        return 0.0, 0.0
    mib = 1024**2
    return torch.cuda.max_memory_allocated(device) / mib, torch.cuda.max_memory_reserved(device) / mib
