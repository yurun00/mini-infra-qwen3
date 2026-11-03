"""模型执行原语（ModelRunner 层）：只负责“喂入一步输入、跑一次前向、产出下一个 Token 与耗时”。

不做任何调度决策（谁入场、谁退场见 core/scheduler.py），也不维护请求生命周期、EOS/预算判定或逻辑时钟
（见 engine/lockstep_loop.py 与 engine/scheduler_loop.py）。两类计时口径并存，调用方按场景选用：

- step_forward：底层单步张量前向与贪心采样，CUDA 下默认用 Event 异步打点（should_sync=False），
  供齐步（Lockstep）循环使用；
- prefill_requests / decode_slots：面向请求槽位的输入组装与同步前向（should_sync=True），供调度循环推进逻辑时钟。
"""

import time
from collections.abc import Sequence
from typing import Any, NamedTuple

import torch

from mini_infra.core.request import RequestState
from mini_infra.device import synchronize
from mini_infra.engine.preprocessing import left_pad_sequences
from mini_infra.worker.dense_slot_cache import DenseSlotCache
from mini_infra.worker.kv_cache import KVCache


class StepForwardResult(NamedTuple):
    """单步前向计时输出对象（五字段 NamedTuple）：

    - 属性访问：.next_token, .present_key_values, .elapsed_ms, .event_pair, .last_logits
    - present_key_values 是模型返回的逐层 (key, value) 纯张量列表，由调用方决定丢弃还是写回容器；
    - last_logits 仅在 capture_logits=True 时填充；CUDA 下 should_sync=False 时 event_pair 存放计时 Event，
      elapsed_ms 为 0。
    """

    next_token: torch.Tensor
    present_key_values: list[tuple[torch.Tensor, torch.Tensor]]
    elapsed_ms: float = 0.0
    event_pair: tuple[Any, Any] | None = None
    last_logits: torch.Tensor | None = None


class SlotDecodeResult(NamedTuple):
    """一步槽位 Decode 的产出：按槽位索引的下一个 Token，以及模型前向与输入准备耗时（ms）。"""

    slot_tokens: list[int]
    model_ms: float
    prepare_ms: float


def prepare_decode_inputs(
    slot_cache: DenseSlotCache,
    running_slot_indices: Sequence[int],
    last_tokens: Sequence[int],
    positions: Sequence[int],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """构造 capacity 行的单步 Decode 输入，返回 (input_ids, position_ids, attention_mask, 准备耗时 ms)。

    空闲/已结束槽位填 Padding 并整行 mask；先在 Host 侧组装列表再一次性传输，避免逐槽位的小 kernel。
    """
    if slot_cache.history_mask is None:
        raise RuntimeError("Cache 尚未分配，无法构造 Decode 输入")
    synchronize(slot_cache.device)
    start = time.perf_counter()
    token_ids = [pad_token_id] * slot_cache.capacity
    pos_list = [0] * slot_cache.capacity
    active = [False] * slot_cache.capacity
    for slot_idx, token_id, pos in zip(running_slot_indices, last_tokens, positions):
        token_ids[slot_idx] = token_id
        pos_list[slot_idx] = pos
        active[slot_idx] = True
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=slot_cache.device).unsqueeze(1)
    position_ids = torch.tensor(pos_list, dtype=torch.long, device=slot_cache.device).unsqueeze(1)
    active_mask = torch.tensor(active, dtype=torch.bool, device=slot_cache.device).unsqueeze(1)
    attention_mask = torch.cat((slot_cache.history_mask, active_mask), dim=1)
    synchronize(slot_cache.device)
    return input_ids, position_ids, attention_mask, (time.perf_counter() - start) * 1000


class ModelRunner:
    """模型执行器（ModelRunner 数据面）：负责管理模型前向与单步贪心采样，隐藏硬件计时与输入组装细节。

    对请求生命周期（max_new_tokens、eos_token_id、finish_reason）完全无感知：
    - step_forward：单步张量前向与采样原语（支持 should_sync=False 异步 Event 打点与 should_sync=True 同步测时）；
    - prefill_requests / decode_slots：接收活跃请求或 DenseSlotCache，组装张量并调用 step_forward(should_sync=True)。
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, pad_token_id: int = 0):
        self.model = model
        self.device = device
        self.pad_token_id = pad_token_id

    def step_forward(
        self,
        model_input: torch.Tensor,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        capture_logits: bool = False,
        should_sync: bool = False,
    ) -> StepForwardResult:
        """单步张量前向推理与贪心采样。

        - past_key_values 为 None 时做全量上下文计算（Prefill 或 No-Cache baseline），否则做增量 Decode；
        - should_sync=False（默认）：CUDA 下通过 Event 异步打点，供静态自回归循环避免每步 CPU-GPU 同步；
        - should_sync=True：执行前后硬件同步，直接计算耗时 ms，供调度器推进逻辑时钟。
        """
        if should_sync:
            synchronize(self.device)

        event_pair = None
        elapsed_ms = 0.0

        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            t0 = time.perf_counter()

        logits, present_key_values = self.model(
            model_input, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values
        )
        last_logits = logits[:, -1, :]
        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        if self.device.type == "cuda":
            end.record()
            if should_sync:
                synchronize(self.device)
                elapsed_ms = start.elapsed_time(end)
            else:
                event_pair = (start, end)
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000

        result_logits = last_logits if capture_logits else None
        return StepForwardResult(
            next_token=next_token,
            present_key_values=present_key_values,
            elapsed_ms=elapsed_ms,
            event_pair=event_pair,
            last_logits=result_logits,
        )

    def prefill_requests(self, states: Sequence[RequestState]) -> tuple[list[int], KVCache, float]:
        """对一批新入场请求做左 Padding Prefill，返回 (各请求首个 Token, Prefill Cache, 模型耗时 ms)。"""
        input_ids, attention_mask, position_ids, _ = left_pad_sequences(
            [state.spec.token_ids for state in states], self.pad_token_id, self.device
        )
        step_out = self.step_forward(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, should_sync=True
        )
        next_tokens = step_out.next_token.squeeze(-1).tolist()
        return next_tokens, KVCache(step_out.present_key_values), step_out.elapsed_ms

    def decode_slots(self, dense_slot_cache: DenseSlotCache, running: Sequence[RequestState]) -> SlotDecodeResult:
        """对 DenseSlotCache 的全部槽位做一步 Decode，并把返回的新 KV 与本步 attention_mask 一起提交回槽位。"""
        running_slot_indices = [state.slot_idx for state in running if state.slot_idx is not None]
        last_tokens = [state.generated[-1] for state in running]
        positions = [state.cache_length for state in running]
        input_ids, position_ids, attention_mask, prepare_ms = prepare_decode_inputs(
            slot_cache=dense_slot_cache,
            running_slot_indices=running_slot_indices,
            last_tokens=last_tokens,
            positions=positions,
            pad_token_id=self.pad_token_id,
        )
        step_out = self.step_forward(
            input_ids,
            past_key_values=dense_slot_cache.past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            should_sync=True,
        )
        dense_slot_cache.commit_decode_step(attention_mask, step_out.present_key_values)
        next_tokens = step_out.next_token.squeeze(-1).tolist()
        return SlotDecodeResult(next_tokens, step_out.elapsed_ms, prepare_ms)


__all__ = [
    "ModelRunner",
    "SlotDecodeResult",
    "StepForwardResult",
    "prepare_decode_inputs",
]
