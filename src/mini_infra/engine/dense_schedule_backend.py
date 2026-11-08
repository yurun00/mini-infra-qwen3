"""连续批处理与固定批次的稠密 batch slot Cache 调度后端（基于 DenseSlotCache 与 SlotRunner）。"""

import time
from collections.abc import Sequence

import torch

from mini_infra.core.request import RequestState
from mini_infra.core.scheduler import RunningQueue, WaitingQueue
from mini_infra.device import synchronize
from mini_infra.engine.mode import BatchingMethod, CachingMethod
from mini_infra.engine.schedule_backend import ScheduleBackend, ScheduleStepOutput
from mini_infra.profiling import MemorySnapshotTracker
from mini_infra.worker import DenseSlotCache, SlotRunner

__all__ = ["DenseScheduleBackend"]


class DenseScheduleBackend(ScheduleBackend):
    """连续批处理与固定批次的稠密 batch slot Cache 调度后端（基于 DenseSlotCache 与 SlotRunner）。"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_running_requests: int,
        batching_method: BatchingMethod,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__(model, device, max_running_requests, batching_method)
        self.is_continuous = batching_method is BatchingMethod.CONTINUOUS
        self.snapshot_tracker = MemorySnapshotTracker()
        self.snapshot_tracker.start()
        self.runner = SlotRunner(model, device, pad_token_id)
        self.dense_slot_cache = DenseSlotCache(max_running_requests, device)
        self.snapshot_tracker.record_stage("01_dense_init")

    @property
    def caching_method(self) -> CachingMethod:
        return CachingMethod.DENSE

    def determine_admission(self, waiting_queue: WaitingQueue, running_queue: RunningQueue) -> int:
        can_admit = running_queue.can_admit() if self.is_continuous else not running_queue
        if can_admit and waiting_queue:
            return min(len(waiting_queue), running_queue.num_free_batch_slots)
        return 0

    def execute_prefill(self, admitted: list[RequestState]) -> ScheduleStepOutput:
        prefill_out = self.runner.run_prefill_step([s.spec.token_ids for s in admitted])
        self.snapshot_tracker.record_stage("02_dense_prefill_forward_done")
        batch_slot_indices = [s.batch_slot_idx for s in admitted if s.batch_slot_idx is not None]
        cache_lengths = [s.cache_length for s in admitted]
        if self.is_continuous:
            synchronize(self.device)
            cache_start = time.perf_counter()
            moved_bytes = self.dense_slot_cache.absorb_prefill(
                batch_slot_indices, cache_lengths, prefill_out.prefill_cache
            )
            synchronize(self.device)
            cache_ms = (time.perf_counter() - cache_start) * 1000
        else:
            self.dense_slot_cache.adopt_prefill(batch_slot_indices, cache_lengths, prefill_out.prefill_cache)
            cache_ms = 0.0
            moved_bytes = 0
        self.snapshot_tracker.record_stage("03_dense_prefill_absorbed")
        next_tokens = prefill_out.next_tokens
        model_ms = prefill_out.model_ms
        del prefill_out
        self.snapshot_tracker.record_stage("04_dense_prefill_cleaned")
        return ScheduleStepOutput(
            next_tokens=next_tokens,
            model_ms=model_ms,
            cache_ms=cache_ms,
            physical_batch_size=len(admitted),
            live_kv_cache_bytes=self.dense_slot_cache.total_bytes,
            kv_cache_pool_bytes=self.dense_slot_cache.total_bytes,
            cache_realign_moved_bytes=moved_bytes,
            accessed_kv_cache_tokens=0,
        )

    def execute_decode(self, running_queue: RunningQueue) -> ScheduleStepOutput:
        batch_slot_indices = [s.batch_slot_idx for s in running_queue if s.batch_slot_idx is not None]
        last_tokens = [s.generated[-1] for s in running_queue]
        positions = [s.cache_length for s in running_queue]
        synchronize(self.device)
        prepare_start = time.perf_counter()
        input_ids, position_ids, attention_mask = self.runner.prepare_decode_inputs(
            self.dense_slot_cache,
            running_batch_slot_indices=batch_slot_indices,
            last_tokens=last_tokens,
            positions=positions,
        )
        synchronize(self.device)
        prepare_ms = (time.perf_counter() - prepare_start) * 1000
        decode_out = self.runner.run_decode_step(input_ids, attention_mask, position_ids, self.dense_slot_cache)
        self.snapshot_tracker.record_stage("05_dense_decode_step", dump=False)
        cache_ms = prepare_ms if self.is_continuous else 0.0
        next_tokens = [decode_out.batch_slot_tokens[s.batch_slot_idx] for s in running_queue]
        return ScheduleStepOutput(
            next_tokens=next_tokens,
            model_ms=decode_out.model_ms,
            cache_ms=cache_ms,
            physical_batch_size=self.dense_slot_cache.capacity,
            live_kv_cache_bytes=self.dense_slot_cache.total_bytes,
            kv_cache_pool_bytes=self.dense_slot_cache.total_bytes,
            cache_realign_moved_bytes=0,
            accessed_kv_cache_tokens=self.dense_slot_cache.capacity * self.dense_slot_cache.length,
        )

    def retire(self, finished: Sequence[RequestState], running_queue: RunningQueue) -> None:
        for state in finished:
            if state.batch_slot_idx is not None:
                self.dense_slot_cache.release(state.batch_slot_idx)
        running_queue.retire(finished)
        if not running_queue:
            self.dense_slot_cache.reset()
            self.snapshot_tracker.stop()

    def __del__(self) -> None:
        if hasattr(self, "snapshot_tracker"):
            self.snapshot_tracker.stop()
