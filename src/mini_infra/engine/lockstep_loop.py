"""齐步自回归生成主循环（Lockstep Generation Loop）。

不经过请求队列调度器（Scheduler），所有请求在 t=0 同时入场、打包进同一张量齐步执行（1 步 Prefill + N 步 Decode）。
面向算子基准测试与数值等价性校验（支持 use_cache=False 全量重算对照、capture_logits=True 逐步 Logits 采集
以及 CUDA Event 全异步计时）。内部持有向量化状态机 LockstepState，每步直接调用
ModelRunner.step_forward(sync=False) 执行前向推理，避免每步 CPU-GPU 同步开销。
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mini_infra.core.types import BatchScheduleInfo, SequenceOutput
from mini_infra.device import synchronize
from mini_infra.engine.preprocessing import prepare_prefill_inputs
from mini_infra.engine.runner import ModelRunner
from mini_infra.engine.types import LockstepResult
from mini_infra.memory.kv_cache import KVCache, cache_size_bytes

__all__ = ["LockstepState", "run_lockstep"]


@dataclass
class LockstepState:
    """管理一次齐步（Lockstep）生成中所有槽位的生命周期与张量更新。"""

    batch_size: int
    padded_prompt_length: int
    prompt_lengths: torch.Tensor
    budgets: torch.Tensor
    max_steps: int
    device: torch.device
    pad_token_id: int

    # 动态槽位生命周期与跟踪
    generated_counts: torch.Tensor
    curr_active: torch.Tensor
    active_counts_per_step: list[int]
    outputs: list[list[int]]
    completion_steps: list[int | None]
    finish_reasons: list[str]

    # 模型输入张量
    history_mask: torch.Tensor
    curr_input_ids: torch.Tensor
    curr_position_ids: torch.Tensor
    full_input_ids: torch.Tensor | None
    full_position_ids: torch.Tensor | None

    @classmethod
    def create(
        cls,
        input_ids_tensor: torch.Tensor,
        history_mask: torch.Tensor,
        position_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
        budgets: torch.Tensor,
        device: torch.device,
        pad_token_id: int,
        use_cache: bool,
    ) -> "LockstepState":
        batch_size, padded_prompt_length = input_ids_tensor.shape
        max_steps = int(budgets.max().item())

        return cls(
            batch_size=batch_size,
            padded_prompt_length=padded_prompt_length,
            prompt_lengths=prompt_lengths,
            budgets=budgets,
            max_steps=max_steps,
            device=device,
            pad_token_id=pad_token_id,
            generated_counts=torch.zeros(batch_size, dtype=torch.long, device=device),
            curr_active=torch.ones(batch_size, dtype=torch.bool, device=device),
            active_counts_per_step=[],
            outputs=[[] for _ in range(batch_size)],
            completion_steps=[None] * batch_size,
            finish_reasons=["max_tokens"] * batch_size,
            history_mask=history_mask,
            curr_input_ids=input_ids_tensor,
            curr_position_ids=position_ids,
            full_input_ids=input_ids_tensor if not use_cache else None,
            full_position_ids=position_ids if not use_cache else None,
        )

    def get_model_inputs(self, use_cache: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取当前前向所需的 (input_ids, attention_mask, position_ids)。"""
        model_input = self.curr_input_ids if use_cache else self.full_input_ids
        position_ids = self.curr_position_ids if use_cache else self.full_position_ids
        return model_input, self.history_mask, position_ids

    def update_step(
        self,
        next_tokens: torch.Tensor,
        step: int,
        eos_token_id: int | None,
        stop_on_eos: bool,
    ) -> bool:
        """更新槽位输出 Token、预算/EOS 判定并返回是否所有槽位均已完成。

        完成判定在设备侧向量化完成，再用一次堆叠传输取回 Host，避免逐请求 `.item()` 带来的 O(B) 次同步。
        """
        active_before = self.curr_active
        self.generated_counts += active_before.long()
        if stop_on_eos and eos_token_id is not None:
            stopped = active_before & (next_tokens == eos_token_id)
        else:
            stopped = torch.zeros_like(active_before)
        reached_budget = self.generated_counts >= self.budgets
        finished = active_before & (stopped | reached_budget)
        self.curr_active = active_before & ~finished

        tokens, was_active, now_finished, by_eos, still_active = torch.stack(
            (next_tokens.long(), active_before.long(), finished.long(), stopped.long(), self.curr_active.long())
        ).tolist()
        self.active_counts_per_step.append(sum(was_active))
        for req_idx in range(self.batch_size):
            if not was_active[req_idx]:
                continue
            self.outputs[req_idx].append(tokens[req_idx])
            if now_finished[req_idx]:
                self.completion_steps[req_idx] = step + 1
                self.finish_reasons[req_idx] = "stop" if by_eos[req_idx] else "max_tokens"

        return not any(still_active)

    def prepare_next_step(self, next_tokens: torch.Tensor, use_cache: bool) -> None:
        """准备下一步前向输入：活跃槽位传真实 Token，已完成槽位填 Padding 并 mask 掉。"""
        step_token = torch.where(
            self.curr_active, next_tokens, torch.full_like(next_tokens, self.pad_token_id)
        ).unsqueeze(1)
        self.history_mask = torch.cat((self.history_mask, self.curr_active.unsqueeze(1)), dim=1)
        next_positions = self.prompt_lengths + self.generated_counts - 1
        curr_pos = torch.where(self.curr_active, next_positions, torch.zeros_like(next_positions)).unsqueeze(1)

        if use_cache:
            self.curr_input_ids = step_token
            self.curr_position_ids = curr_pos
        else:
            self.full_input_ids = torch.cat((self.full_input_ids, step_token), dim=1)
            self.full_position_ids = torch.cat((self.full_position_ids, curr_pos), dim=1)

    def compute_batch_metrics(
        self,
        step_latencies_ms: list[float],
        kv_cache: Any = None,
    ) -> BatchScheduleInfo:
        """统计批处理生成过程特有明细指标（利用率、Padding 效率、各请求完成时延等）。"""
        step_count = len(step_latencies_ms)
        valid_generated_tokens_count = sum(len(output) for output in self.outputs)

        num_prompt_tokens = int(self.prompt_lengths.sum().item())
        num_padded_prompt_tokens = self.batch_size * self.padded_prompt_length
        capacity_tokens = self.batch_size * step_count

        completion_steps_per_req: list[int] = [s if s is not None else step_count for s in self.completion_steps]
        ttft_ms = step_latencies_ms[0] if step_latencies_ms else 0.0
        return BatchScheduleInfo(
            padding_efficiency=num_prompt_tokens / num_padded_prompt_tokens if num_padded_prompt_tokens else 0.0,
            slot_utilization=valid_generated_tokens_count / capacity_tokens if capacity_tokens else 0.0,
            num_prompt_tokens=num_prompt_tokens,
            num_padded_prompt_tokens=num_padded_prompt_tokens,
            prompt_lengths_per_req=[int(x) for x in self.prompt_lengths.cpu().tolist()],
            completion_steps_per_req=completion_steps_per_req,
            ttft_ms=ttft_ms,
        )


def _resolve_step_latencies(
    step_event_pairs: list[tuple[Any, Any]], step_times_ms: list[float], is_cuda: bool
) -> list[float]:
    """CUDA 下使用 Event 硬件测时，非 CUDA 下使用 CPU 墙钟。"""
    if is_cuda and step_event_pairs:
        return [s.elapsed_time(e) for s, e in step_event_pairs]
    return step_times_ms


def run_lockstep(
    model: torch.nn.Module,
    input_ids: torch.Tensor | Sequence[Any],
    token_sequences: list[torch.Tensor],
    has_uniform_len: bool,
    budgets: torch.Tensor,
    use_cache: bool,
    device: torch.device,
    *,
    pad_token_id: int = 0,
    eos_token_id: int | None = None,
    stop_on_eos: bool = True,
    capture_logits: bool = False,
) -> LockstepResult:
    """执行齐步（Lockstep）自回归生成迭代循环（所有请求同时开始，含 step 0 Prefill 与后续 Decode 步）。

    内部完成 Prefill 输入组装与 LockstepState 状态机驱动，每步调用 ModelRunner.step_forward，
    返回封装好的 LockstepResult。
    """
    is_cuda = device.type == "cuda"
    input_tensor, history_mask, pos_ids, prompt_lens = prepare_prefill_inputs(
        input_ids, token_sequences, has_uniform_len, pad_token_id, device
    )
    kv_cache = KVCache() if use_cache else None
    runner = ModelRunner(model, device, pad_token_id)
    state = LockstepState.create(
        input_ids_tensor=input_tensor,
        history_mask=history_mask,
        position_ids=pos_ids,
        prompt_lengths=prompt_lens,
        budgets=budgets,
        device=device,
        pad_token_id=pad_token_id,
        use_cache=use_cache,
    )

    step_event_pairs: list[tuple[Any, Any]] = []
    step_times_ms: list[float] = []
    logits_trace = [] if capture_logits else None
    memory_after_prefill_bytes = 0

    wall_start = time.perf_counter()

    for step in range(state.max_steps):
        model_input, attention_mask, position_ids = state.get_model_inputs(use_cache)
        step_out = runner.step_forward(
            model_input=model_input,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_logits=capture_logits,
            sync=False,
        )
        if step_out.event_pair is not None:
            step_event_pairs.append(step_out.event_pair)
        else:
            step_times_ms.append(step_out.elapsed_ms)

        if step == 0 and is_cuda:
            memory_after_prefill_bytes = torch.cuda.memory_allocated(device)

        if capture_logits and step_out.last_logits is not None:
            logits_trace.append(step_out.last_logits.float().cpu())

        next_tokens = step_out.next_token.squeeze(1)
        is_all_finished = state.update_step(next_tokens, step, eos_token_id, stop_on_eos)
        if is_all_finished:
            break
        state.prepare_next_step(next_tokens, use_cache)

    synchronize(device)
    e2e_wall_ms = (time.perf_counter() - wall_start) * 1000
    step_latencies_ms = _resolve_step_latencies(step_event_pairs, step_times_ms, is_cuda)

    valid_tokens = sum(len(o) for o in state.outputs)
    e2e_model_ms = sum(step_latencies_ms)
    batch_info = state.compute_batch_metrics(step_latencies_ms, kv_cache)

    sequences = [
        SequenceOutput(
            request_index=idx,
            token_ids=state.outputs[idx],
            prompt_len=int(state.prompt_lengths[idx].item()),
            output_len=len(state.outputs[idx]),
            finish_reason=state.finish_reasons[idx],
        )
        for idx in range(state.batch_size)
    ]
    cache_bytes = cache_size_bytes(kv_cache) if kv_cache is not None else 0

    return LockstepResult(
        sequences=sequences,
        batch_info=batch_info,
        step_latencies_ms=step_latencies_ms,
        num_active_tokens_per_step=state.active_counts_per_step,
        logits_trace=logits_trace,
        e2e_wall_ms=e2e_wall_ms,
        e2e_model_ms=e2e_model_ms,
        total_tokens=valid_tokens,
        memory_after_prefill_bytes=memory_after_prefill_bytes,
        cache_bytes=cache_bytes,
    )
