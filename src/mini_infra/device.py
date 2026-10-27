"""通用硬件设备同步与显存统计工具。"""

import torch


def synchronize(device: torch.device) -> None:
    """CUDA 设备上等待所有已提交 kernel 完成；其他设备为空操作。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    """重置 CUDA 峰值显存统计；非 CUDA 设备为空操作。"""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> tuple[float, float]:
    """返回 (峰值已分配, 峰值已预留) 显存（MB）；非 CUDA 设备返回 (0.0, 0.0)。"""
    if device.type != "cuda":
        return 0.0, 0.0
    mib = 1024**2
    return torch.cuda.max_memory_allocated(device) / mib, torch.cuda.max_memory_reserved(device) / mib
