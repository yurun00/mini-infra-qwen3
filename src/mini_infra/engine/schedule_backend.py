"""批处理调度后端契约：抽象基类与单步产出数据结构。

KVCache 物理显存管理与模型执行被下放到 worker/，本模块定义 engine 主循环依赖的后端接口，
使控制流（engine）与执行细节（worker）保持单向依赖。Dense / Paged 两个具体实现分别位于
dense_schedule_backend.py 与 paged_schedule_backend.py。
"""

import abc
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mini_infra.core.block_manager import (
    NULL_KV_CACHE_MANAGER,
    BaseKVCacheManager,
)
from mini_infra.core.request import RequestState
from mini_infra.core.scheduler import PrefillChunk
from mini_infra.core.types import BatchingMethod, KVCacheConfig
from mini_infra.device import (
    allocated_memory_mb,
    reset_peak_memory,
    synchronize,
)
from mini_infra.engine.metrics import PagedCacheProfile
from mini_infra.profiling import MemorySnapshotTracker

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
    first_token_logits: list[torch.Tensor] | None = None


class ScheduleBackend(abc.ABC):
    """抽象调度单步执行后端基类（承载通用硬件初始化与单步调度执行契约）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        max_running_requests: int,
        batching_method: BatchingMethod,
    ) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self.max_running_requests = max_running_requests
        self.batching_method = batching_method
        # 交由 core.IterationScheduler 统管块分配与释放；Dense 后端沿用默认空对象，Paged 后端覆写为 KVCacheManager
        self.kv_cache_manager: BaseKVCacheManager = NULL_KV_CACHE_MANAGER
        synchronize(self.device)
        self.baseline_mb = allocated_memory_mb(self.device)
        reset_peak_memory(self.device)
        self.snapshot_tracker = MemorySnapshotTracker()
        self.snapshot_tracker.start()

    @property
    @abc.abstractmethod
    def kv_cache_config(self) -> KVCacheConfig:
        """该后端采用的运行时 KV Cache 配置（DenseCacheConfig 或 PagedCacheConfig）。"""
        ...

    @abc.abstractmethod
    def execute_prefill(
        self,
        chunks: Sequence[PrefillChunk],
        *,
        should_capture_first_token_logits: bool = False,
    ) -> ScheduleStepOutput:
        """执行一批 Prompt 区间（整段 Prompt 或 chunk）的 Prefill 前向，将 KV 写入缓存并返回耗时。

        ScheduleStepOutput.next_tokens（及启用采集时的 first_token_logits）仅覆盖本步写满 Prompt 的请求
        （PrefillChunk.is_last），按 chunks 次序。
        """
        ...

    @abc.abstractmethod
    def execute_decode(
        self, scheduled_requests: Sequence[RequestState]
    ) -> ScheduleStepOutput:
        """为当前运行中请求组装输入，执行一步增量 Decode 并写回缓存。"""
        ...

    def retire(self, finished: Sequence[RequestState], *, is_idle: bool) -> None:
        """回收已结束请求的物理缓存资源，并在无活跃请求（is_idle）时执行重置（默认无操作）。"""
        return None

    def collect_paged_cache_profile(self) -> PagedCacheProfile | None:
        """运行结束后采集 Paged physical block pool 画像（Dense 后端无 physical block pool，返回 None）。"""
        return None

    def __del__(self) -> None:
        if hasattr(self, "snapshot_tracker"):
            self.snapshot_tracker.stop()
