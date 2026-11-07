"""连续批处理的分页 KV Cache 调度后端（基于 PagedRunner、PagedKVCachePool 与 KVCacheManager）。"""

import time
from collections.abc import Sequence

import torch

from mini_infra.core.block_manager import (
    KVCacheManager,
    PagedCacheConfig,
    num_blocks_for_tokens,
    required_pool_blocks,
)
from mini_infra.core.request import RequestSpec, RequestState
from mini_infra.core.scheduler import RunningQueue, WaitingQueue
from mini_infra.device import synchronize
from mini_infra.engine.metrics import PagedCacheProfile
from mini_infra.engine.mode import BatchingMethod, CachingMethod
from mini_infra.engine.schedule_backend import ScheduleBackend, ScheduleStepOutput
from mini_infra.worker import PagedKVCachePool, PagedRunner

__all__ = ["PagedScheduleBackend"]


class PagedScheduleBackend(ScheduleBackend):
    """连续批处理与固定批次的分页 KV Cache 调度后端（基于 PagedRunner、PagedKVCachePool 与 KVCacheManager）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        request_specs: Sequence[RequestSpec],
        max_running_requests: int,
        paged_cache_config: PagedCacheConfig,
        batching_method: BatchingMethod = BatchingMethod.CONTINUOUS,
        pad_token_id: int = 0,
    ) -> None:
        self.block_size = paged_cache_config.block_size
        num_blocks = paged_cache_config.num_blocks or required_pool_blocks(
            request_specs, max_running_requests, self.block_size
        )
        largest_request_blocks = max(
            num_blocks_for_tokens(spec.prompt_len + spec.max_new_tokens - 1, self.block_size) for spec in request_specs
        )
        if largest_request_blocks > num_blocks - 1:
            raise ValueError(
                f"num_blocks={num_blocks} 无法容纳单个请求所需的 {largest_request_blocks} "
                "个物理块（另需 1 个 null block）"
            )
        super().__init__(model, device, max_running_requests, batching_method)
        self.is_continuous = batching_method is BatchingMethod.CONTINUOUS
        config = model.config  # type: ignore[attr-defined]
        self.runner = PagedRunner(model, device, pad_token_id)
        self.pool = PagedKVCachePool(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=self.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=next(model.parameters()).dtype,
            device=device,
        )
        self.kv_cache_manager = KVCacheManager(self.block_size, num_blocks)

    @property
    def caching_method(self) -> CachingMethod:
        return CachingMethod.PAGED

    def determine_admission(self, waiting_queue: WaitingQueue, running_queue: RunningQueue) -> int:
        can_admit = running_queue.can_admit() if self.is_continuous else not running_queue
        if can_admit and waiting_queue:
            candidates = waiting_queue.requests[: running_queue.num_free_batch_slots]
            return self.kv_cache_manager.num_admissible([s.spec.prompt_len for s in candidates])
        return 0

    def execute_prefill(self, admitted: list[RequestState]) -> ScheduleStepOutput:
        prefill_out = self.runner.run_prefill_step([s.spec.token_ids for s in admitted])
        cache_lengths = [s.cache_length for s in admitted]
        synchronize(self.device)
        cache_start = time.perf_counter()
        for state in admitted:
            self.kv_cache_manager.allocate_slots(state.spec.request_idx, state.cache_length)
        self.runner.store_prefill(
            self.pool,
            prefill_out.prefill_cache,
            [self.kv_cache_manager.block_table(s.spec.request_idx) for s in admitted],
            cache_lengths,
        )
        synchronize(self.device)
        cache_ms = (time.perf_counter() - cache_start) * 1000
        next_tokens = prefill_out.next_tokens
        model_ms = prefill_out.model_ms
        del prefill_out
        return ScheduleStepOutput(
            next_tokens=next_tokens,
            model_ms=model_ms,
            cache_ms=cache_ms,
            physical_batch_size=len(admitted),
            live_kv_cache_bytes=self.kv_cache_manager.num_used_blocks * self.pool.bytes_per_block,
            kv_cache_pool_bytes=self.pool.total_bytes,
            cache_realign_moved_bytes=0,
            accessed_kv_cache_tokens=0,
        )

    def execute_decode(self, running_queue: RunningQueue) -> ScheduleStepOutput:
        active_before = len(running_queue)
        synchronize(self.device)
        prepare_start = time.perf_counter()
        for state in running_queue:
            if not self.kv_cache_manager.allocate_slots(state.spec.request_idx, 1):
                raise RuntimeError("Paged KV Cache 物理块已耗尽（当前未实现抢占），请增大 num_blocks")
        input_ids, position_ids, metadata = self.runner.prepare_decode_inputs(
            [self.kv_cache_manager.block_table(s.spec.request_idx) for s in running_queue],
            last_tokens=[s.generated[-1] for s in running_queue],
            positions=[s.cache_length for s in running_queue],
            block_size=self.block_size,
        )
        synchronize(self.device)
        cache_ms = (time.perf_counter() - prepare_start) * 1000
        decode_out = self.runner.run_decode_step(input_ids, position_ids, metadata, self.pool)
        return ScheduleStepOutput(
            next_tokens=decode_out.next_tokens,
            model_ms=decode_out.model_ms,
            cache_ms=cache_ms,
            physical_batch_size=active_before,
            live_kv_cache_bytes=self.kv_cache_manager.num_used_blocks * self.pool.bytes_per_block,
            kv_cache_pool_bytes=self.pool.total_bytes,
            cache_realign_moved_bytes=0,
            accessed_kv_cache_tokens=sum(s.cache_length + 1 for s in running_queue),
        )

    def retire(self, finished: Sequence[RequestState], running_queue: RunningQueue) -> None:
        for state in finished:
            self.kv_cache_manager.free(state.spec.request_idx)
        running_queue.retire(finished)

    def collect_paged_cache_profile(self) -> PagedCacheProfile:
        return PagedCacheProfile(
            block_size=self.block_size,
            bytes_per_block=self.pool.bytes_per_block,
            num_blocks=self.pool.num_blocks,
            peak_used_blocks=self.kv_cache_manager.peak_used_blocks,
            block_cold_allocation_count=self.kv_cache_manager.block_cold_allocation_count,
            block_reuse_count=self.kv_cache_manager.block_reuse_count,
            block_release_count=self.kv_cache_manager.block_release_count,
        )
