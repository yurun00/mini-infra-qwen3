"""连续批处理的分页 KV Cache 调度后端（基于 PagedRunner、PagedBlockCache 与 KVCacheManager）。"""

import time
from collections.abc import Sequence

import torch

from mini_infra.core.block_manager import (
    KVCacheManager,
    PagedCacheConfig,
    infer_num_blocks_in_pool,
    num_blocks_for_tokens,
)
from mini_infra.core.request import Request, RequestState
from mini_infra.core.scheduler import PrefillChunk
from mini_infra.core.types import BatchingMethod, KVCacheConfig
from mini_infra.device import synchronize
from mini_infra.engine.metrics import PagedCacheProfile
from mini_infra.engine.schedule_backend import (
    ScheduleBackend,
    ScheduleStepOutput,
)
from mini_infra.model.attention import AttentionBackendKind
from mini_infra.worker import PagedBlockCache, PagedRunner

__all__ = ["PagedScheduleBackend"]


class PagedScheduleBackend(ScheduleBackend):
    """连续批处理与固定批次的分页 KV Cache 调度后端（基于 PagedRunner、PagedBlockCache 与 KVCacheManager）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        requests: Sequence[Request],
        max_running_requests: int,
        paged_cache_config: PagedCacheConfig,
        batching_method: BatchingMethod = BatchingMethod.CONTINUOUS,
        pad_token_id: int = 0,
        attention_backend: AttentionBackendKind = "eager",
    ) -> None:
        self.paged_cache_config = paged_cache_config
        self.block_size = paged_cache_config.block_size
        num_blocks = paged_cache_config.num_blocks or infer_num_blocks_in_pool(
            requests, max_running_requests, self.block_size
        )
        largest_request_blocks = max(
            num_blocks_for_tokens(
                req.prompt_len + req.max_new_tokens - 1, self.block_size
            )
            for req in requests
        )
        if largest_request_blocks > num_blocks - 1:
            raise ValueError(
                f"num_blocks={num_blocks} 无法容纳单个请求所需的 {largest_request_blocks} "
                "个物理块（另需 1 个 null block）"
            )
        super().__init__(model, max_running_requests, batching_method)
        config = model.config  # type: ignore[attr-defined]
        self.runner = PagedRunner(model, pad_token_id, attention_backend)
        self.snapshot_tracker.record_stage("00_paged_before_pool_init")
        self.paged_block_cache = PagedBlockCache(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=self.block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=next(model.parameters()).dtype,
            device=self.device,
        )
        self.snapshot_tracker.record_stage("01_paged_pool_allocated")
        self.kv_cache_manager = KVCacheManager(
            self.block_size,
            num_blocks,
            is_prefix_caching_enabled=paged_cache_config.is_prefix_caching_enabled,
        )

    @property
    def kv_cache_config(self) -> KVCacheConfig:
        return self.paged_cache_config

    def execute_prefill(
        self,
        chunks: Sequence[PrefillChunk],
        *,
        should_capture_first_token_logits: bool = False,
    ) -> ScheduleStepOutput:
        block_tables = [
            self.kv_cache_manager.block_table(c.state.request.request_idx) for c in chunks
        ]
        start_positions = [c.start for c in chunks]
        prefill_out = self.runner.run_prefill_chunk_step(
            [c.state.request.token_ids[c.start : c.end] for c in chunks],
            start_positions,
            block_tables,
            self.paged_block_cache,
            should_capture_logits=should_capture_first_token_logits,
        )
        self.snapshot_tracker.record_stage("02_paged_prefill_forward_done")
        synchronize(self.device)
        cache_start = time.perf_counter()
        self.runner.store_prefill(
            self.paged_block_cache,
            prefill_out.prefill_cache,
            block_tables,
            [c.num_tokens for c in chunks],
            start_positions,
        )
        synchronize(self.device)
        cache_ms = (time.perf_counter() - cache_start) * 1000
        self.snapshot_tracker.record_stage("03_paged_prefill_stored_dual_kv")
        # 仅 Prompt 写满的 chunk 产出首个 Token；中间 chunk 的末位 logits 无意义
        next_tokens = [
            token for token, c in zip(prefill_out.next_tokens, chunks) if c.is_last
        ]
        first_token_logits = (
            [
                prefill_out.last_logits[row_idx]
                for row_idx, c in enumerate(chunks)
                if c.is_last
            ]
            if prefill_out.last_logits is not None
            else None
        )
        model_ms = prefill_out.model_ms
        del prefill_out
        self.snapshot_tracker.record_stage("04_paged_prefill_cleaned")
        return ScheduleStepOutput(
            next_tokens=next_tokens,
            model_ms=model_ms,
            cache_ms=cache_ms,
            physical_batch_size=len(chunks),
            live_kv_cache_bytes=self.kv_cache_manager.num_used_blocks
            * self.paged_block_cache.bytes_per_block,
            kv_cache_pool_bytes=self.paged_block_cache.total_bytes,
            cache_realign_moved_bytes=0,
            accessed_kv_cache_tokens=sum(c.start for c in chunks),
            first_token_logits=first_token_logits,
        )

    def execute_decode(
        self, scheduled_requests: Sequence[RequestState]
    ) -> ScheduleStepOutput:
        active_before = len(scheduled_requests)
        synchronize(self.device)
        prepare_start = time.perf_counter()
        input_ids, position_ids, metadata = self.runner.prepare_decode_inputs(
            [
                self.kv_cache_manager.block_table(s.request.request_idx)
                for s in scheduled_requests
            ],
            last_tokens=[s.generated[-1] for s in scheduled_requests],
            positions=[s.cache_length for s in scheduled_requests],
            block_size=self.block_size,
        )
        synchronize(self.device)
        cache_ms = (time.perf_counter() - prepare_start) * 1000
        decode_out = self.runner.run_decode_step(
            input_ids, position_ids, metadata, self.paged_block_cache
        )
        self.snapshot_tracker.record_stage("05_paged_decode_step", dump=False)
        return ScheduleStepOutput(
            next_tokens=decode_out.next_tokens,
            model_ms=decode_out.model_ms,
            cache_ms=cache_ms,
            physical_batch_size=active_before,
            live_kv_cache_bytes=self.kv_cache_manager.num_used_blocks
            * self.paged_block_cache.bytes_per_block,
            kv_cache_pool_bytes=self.paged_block_cache.total_bytes,
            cache_realign_moved_bytes=0,
            accessed_kv_cache_tokens=sum(s.cache_length + 1 for s in scheduled_requests),
        )

    def retire(self, finished: Sequence[RequestState], *, is_idle: bool) -> None:
        if is_idle:
            assert self.kv_cache_manager.num_used_blocks == 0, (
                "RunningQueue 已排空，但仍有未释放引用的 physical block: "
                f"num_used_blocks={self.kv_cache_manager.num_used_blocks}"
            )

    def collect_paged_cache_profile(self) -> PagedCacheProfile:
        stats = self.kv_cache_manager.stats
        return PagedCacheProfile(
            block_size=self.block_size,
            bytes_per_block=self.paged_block_cache.bytes_per_block,
            num_blocks=self.paged_block_cache.num_blocks,
            peak_used_blocks=stats.peak_used_blocks,
            block_cold_allocation_count=stats.block_cold_allocation_count,
            block_reuse_count=stats.block_reuse_count,
            block_release_count=stats.block_release_count,
            prefix_cache_block_eviction_count=stats.prefix_cache_block_eviction_count,
            prefix_cache_prompt_tokens=stats.prefix_cache_prompt_tokens,
            prefix_cache_hit_tokens=stats.prefix_cache_hit_tokens,
        )
