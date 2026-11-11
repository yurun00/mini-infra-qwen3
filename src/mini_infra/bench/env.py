"""宿主机、CUDA 设备与 Git 工作区环境探测。"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
import warnings
from pathlib import Path

import torch

from mini_infra.bench.paths import REPO_ROOT


def require_cuda() -> torch.device:
    """要求存在可用 NVIDIA GPU，返回 CUDA 设备；各基准入口共用。"""
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 NVIDIA GPU")
    return torch.device("cuda")


@functools.cache
def git_info() -> dict[str, str | bool]:
    """获取当前工作区 Git 提交摘要（进程内缓存，并带超时保护）。"""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""

    return {
        "commit": run("rev-parse", "--short", "HEAD") or "unknown",
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(run("status", "--porcelain")),
        "message": run("log", "-1", "--pretty=%s"),
    }


def _cpu_model_name() -> str:
    """获取宿主机 CPU 型号名称。"""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                _, _, value = line.partition(":")
                if value.strip():
                    return value.strip()
    return platform.processor() or platform.machine() or "unknown"


def _system_memory_mb() -> float | None:
    """获取宿主机物理内存总量（MiB）。"""
    if hasattr(os, "sysconf"):
        names = os.sysconf_names
        if "SC_PAGE_SIZE" in names and "SC_PHYS_PAGES" in names:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            if page_size > 0 and phys_pages > 0:
                return round(page_size * phys_pages / 1024**2, 1)
    return None


def gpu_info(device: torch.device | int = 0) -> dict[str, str | float | None]:
    """获取测试机 GPU 与宿主机环境参数。"""
    props = torch.cuda.get_device_properties(device)
    info: dict[str, str | float | None] = {
        "gpu": props.name,
        "total_memory_mb": props.total_memory / 1024**2,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "cpu": _cpu_model_name(),
        "system_memory_mb": _system_memory_mb(),
        "platform": platform.platform(),
        "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
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
                timeout=10,
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
    except (OSError, subprocess.SubprocessError, IndexError) as err:
        warnings.warn(f"nvidia-smi 查询失败，跳过驱动/频率/温度信息: {err}", stacklevel=2)
    return info


__all__ = [
    "git_info",
    "gpu_info",
    "require_cuda",
]
