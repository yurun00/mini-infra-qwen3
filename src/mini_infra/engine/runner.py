"""模型执行原语（ModelRunner 层）：只负责“喂入一步输入、跑一次前向、产出下一个 Token 与耗时”。

不做任何调度决策（谁入场、谁退场见 core/scheduler.py），也不维护请求生命周期或逻辑时钟
（见 engine/serving.py）。两类计时口径并存，调用方按场景选用：

- step_forward：CUDA 下用 Event 硬件计时（不强制同步），供 generate 的自回归循环使用；
- prefill_requests / decode_slots / step_batch：同步后用 CPU 墙钟计时，供 Serving 调度器推进逻辑时钟。
"""

import time
from collections.abc import Sequence
from typing import Any, NamedTuple

import torch

from mini_infra.core.request import RequestState
from mini_infra.device import synchronize
from mini_infra.engine.batch_state import BatchGenerationState
from mini_infra.engine.preprocessing import left_pad_sequences
from mini_infra.memory.kv_cache import KVCache
from mini_infra.memory.slot_cache import DenseSlotCache


class StepForwardResult(NamedTuple):
    """单步前向计时输出对象（四字段 NamedTuple）：

    - 属性访问：.next_token, .elapsed_ms, .event_pair, .last_logits
    - last_logits 仅在 return_logits=True 时填充；CUDA 下 event_pair 存放计时 Event，elapsed_ms 为 0。
    """

    next_token: torch.Tensor
    elapsed_ms: float = 0.0
    event_pair: tuple[Any, Any] | None = None
    last_logits: torch.Tensor | None = None


class SlotDecodeResult(NamedTuple):
    """一步槽位 Decode 的产出：按槽位索引的下一个 Token，以及模型前向与输入准备耗时（ms）。"""

    slot_tokens: list[int]
    model_ms: float
    prepare_ms: float


class ModelRunner:
    """模型执行器（ModelRunner 层）：负责管理模型前向与单步采样，隐藏硬件设备与输入组装细节。

    不做任何调度决策（谁入场、谁退场见 core/scheduler.py），也不维护请求生命周期或逻辑时钟
    （见 engine/scheduler_loop.py）。支持两类执行模式：
    - step_forward：单步张量前向与贪心采样（CUDA 下用 Event 硬件计时），供自回归生成循环使用；
    - prefill_requests / decode_slots / step_batch：接收业务状态对象（RequestState / Slot / Batch），
      组装输入并执行前后同步的墙钟测时前向。
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, pad_token_id: int = 0):
        self.model = model
        self.device = device
        self.pad_token_id = pad_token_id

    def step_forward(
        self,
        model_input: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_logits: bool = False,
        sync: bool = False,
    ) -> StepForwardResult:
        """单步张量前向推理与贪心采样。

        - sync=False（默认）：CUDA 下通过 Event 异步打点，供自回归循环避免频繁同步；
        - sync=True：执行前后硬件同步，直接计算耗时 ms，供调度器推进逻辑时钟。
        """
        if sync:
            synchronize(self.device)

        event_pair = None
        elapsed_ms = 0.0

        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            t0 = time.perf_counter()

        logits = self.model(model_input, kv_cache=kv_cache, attention_mask=attention_mask, position_ids=position_ids)
        last_logits = logits[:, -1, :]
        next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

        if self.device.type == "cuda":
            end.record()
            if sync:
                synchronize(self.device)
                elapsed_ms = start.elapsed_time(end)
            else:
                event_pair = (start, end)
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000

        result_logits = last_logits if return_logits else None
        return StepForwardResult(
            next_token=next_token, elapsed_ms=elapsed_ms, event_pair=event_pair, last_logits=result_logits
        )

    def prefill_requests(self, states: Sequence[RequestState]) -> tuple[list[int], KVCache, float]:
        """对一批新入场请求做左 Padding Prefill，返回 (各请求首个 Token, Prefill Cache, 模型耗时 ms)。"""
        input_ids, attention_mask, position_ids, _ = left_pad_sequences(
            [state.spec.token_ids for state in states], self.pad_token_id, self.device
        )
        cache = KVCache()
        step_out = self.step_forward(
            input_ids, kv_cache=cache, attention_mask=attention_mask, position_ids=position_ids, sync=True
        )
        next_tokens = step_out.next_token.squeeze(-1).tolist()
        return next_tokens, cache, step_out.elapsed_ms

    def decode_slots(self, slot_cache: DenseSlotCache, running: Sequence[RequestState]) -> SlotDecodeResult:
        """对 DenseSlotCache 的全部槽位做一步 Decode，并把本步 attention_mask 提交为新的历史 Mask。"""
        input_ids, position_ids, attention_mask, prepare_ms = slot_cache.prepare_decode_inputs(
            running, self.pad_token_id
        )
        step_out = self.step_forward(
            input_ids, kv_cache=slot_cache.cache, attention_mask=attention_mask, position_ids=position_ids, sync=True
        )
        slot_cache.commit_decode_step(attention_mask)
        next_tokens = step_out.next_token.squeeze(-1).tolist()
        return SlotDecodeResult(next_tokens, step_out.elapsed_ms, prepare_ms)

    def step_batch(self, batch: BatchGenerationState, cache: KVCache) -> tuple[torch.Tensor, float]:
        """对批状态机（BatchGenerationState）执行单步前向，返回 (各行下一个 Token, 模型耗时 ms)。"""
        model_input, attention_mask, position_ids = batch.get_model_inputs(use_cache=True)
        step_out = self.step_forward(
            model_input, kv_cache=cache, attention_mask=attention_mask, position_ids=position_ids, sync=True
        )
        return step_out.next_token.squeeze(-1), step_out.elapsed_ms
