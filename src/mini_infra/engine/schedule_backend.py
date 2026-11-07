"""批处理调度后端契约：抽象基类与单步产出数据结构。

KVCache 物理显存管理与模型执行被下放到 worker/，本模块定义 engine 主循环依赖的后端接口，
使控制流（engine）与执行细节（worker）保持单向依赖。Dense / Paged 两个具体实现分别位于
dense_schedule_backend.py 与 paged_schedule_backend.py。
"""

import abc
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mini_infra.core.request import RequestState
from mini_infra.core.scheduler import RunningQueue, WaitingQueue
from mini_infra.device import allocated_memory_mb, reset_peak_memory, synchronize
from mini_infra.engine.metrics import PagedCacheProfile
from mini_infra.engine.mode import BatchingMethod, CachingMethod

__all__ = ["ScheduleBackend", "ScheduleStepOutput"]


@dataclass(frozen=True)
class ScheduleStepOutput:
    """后端执行一次 Prefill 或 Decode 调度步后的度量与结果数据契约。"""

    next_tokens: list[int]
    model_ms: float
    cache_ms: float
    physical_batch_size: int
    live_kv_cache_bytes: int
    kv_cache_pool_bytes: int = 0
    cache_realign_moved_bytes: int = 0
    accessed_kv_cache_tokens: int = 0


class ScheduleBackend(abc.ABC):
    """抽象调度单步执行后端基类（承载通用硬件初始化与单步调度执行契约）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_running_requests: int,
        batching_method: BatchingMethod,
    ) -> None:
        self.model = model
        self.device = device
        self.max_running_requests = max_running_requests
        self.batching_method = batching_method
        synchronize(device)
        self.baseline_mb = allocated_memory_mb(device)
        reset_peak_memory(device)

    @property
    @abc.abstractmethod
    def caching_method(self) -> CachingMethod:
        """该后端采用的 KV Cache 策略（DENSE 或 PAGED）。"""
        ...

    @abc.abstractmethod
    def determine_admission(self, waiting_queue: WaitingQueue, running_queue: RunningQueue) -> int:
        """评估当前时刻允许从 waiting_queue 准入到 running_queue 的请求数量。"""
        ...

    @abc.abstractmethod
    def execute_prefill(self, admitted: list[RequestState]) -> ScheduleStepOutput:
        """执行一批新请求的 Prefill 前向，将 KV 写入缓存并返回首个 Token 与耗时。"""
        ...

    @abc.abstractmethod
    def execute_decode(self, running_queue: RunningQueue) -> ScheduleStepOutput:
        """为当前运行中请求组装输入，执行一步增量 Decode 并写回缓存。"""
        ...

    @abc.abstractmethod
    def retire(self, finished: Sequence[RequestState], running_queue: RunningQueue) -> None:
        """回收已结束请求的槽位或物理块资源，并在空闲时执行重置。"""
        ...

    def collect_paged_cache_profile(self) -> PagedCacheProfile | None:
        """运行结束后采集 Paged 物理块池画像（Dense 后端无块池，返回 None）。"""
        return None
