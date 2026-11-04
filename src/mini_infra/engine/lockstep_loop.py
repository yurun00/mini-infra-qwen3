"""齐步自回归生成主循环（Lockstep Generation Loop）。

不做请求队列调度（Scheduler），所有请求在 t=0 同时入场、打包进同一批次齐步执行（1 步 Prefill + N 步 Decode）。
面向算子基准测试与数值等价性校验（支持 use_cache=False 全量重算对照、capture_logits=True 逐步 Logits 采集
以及 CUDA Event 全异步计时）。
作为协调层编排者，内部通过纯 CPU 状态机 LockstepState 驱动生命周期与完成判定，
GPU 张量组装、物理 KV Cache 更新与采样由 Worker 层的 LockstepRunner 内部自治闭环。
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mini_infra.core.types import SequenceOutput
from mini_infra.device import synchronize
from mini_infra.engine.metrics import (
    BatchScheduleInfo,
    GenerationTrace,
    build_memory_profile,
    compute_step_aggregated_metrics,
)
from mini_infra.engine.results import GenerationResult
from mini_infra.worker import LockstepRunner

__all__ = ["run_lockstep"]


@dataclass
class LockstepState:
    """管理一次齐步（Lockstep）生成中所有槽位的生命周期与完成判定（纯 CPU 列表状态机）。"""

    batch_size: int
    padded_prompt_length: int
    prompt_lengths: list[int]
    budgets: list[int]
    max_steps: int

    # 动态槽位生命周期与跟踪
    generated_counts: list[int]
    curr_active: list[bool]
    active_counts_per_step: list[int]
    outputs: list[list[int]]
    completion_steps: list[int | None]
    finish_reasons: list[str]

    @classmethod
    def _create(
        cls,
        batch_size: int,
        padded_prompt_length: int,
        prompt_lengths: list[int],
        budgets: list[int],
    ) -> "LockstepState":
        max_steps = max(budgets) if budgets else 0
        return cls(
            batch_size=batch_size,
            padded_prompt_length=padded_prompt_length,
            prompt_lengths=prompt_lengths,
            budgets=budgets,
            max_steps=max_steps,
            generated_counts=[0] * batch_size,
            curr_active=[True] * batch_size,
            active_counts_per_step=[],
            outputs=[[] for _ in range(batch_size)],
            completion_steps=[None] * batch_size,
            finish_reasons=["max_tokens"] * batch_size,
        )

    def _commit_step(
        self,
        next_tokens: Sequence[int],
        step: int,
        eos_token_id: int | None,
        stop_on_eos: bool,
    ) -> bool:
        """更新槽位输出 Token、预算/EOS 判定并返回是否所有槽位均已完成（纯 CPU 列表操作）。"""
        num_active = sum(1 for a in self.curr_active if a)
        self.active_counts_per_step.append(num_active)

        for i in range(self.batch_size):
            if not self.curr_active[i]:
                continue
            token = next_tokens[i]
            self.generated_counts[i] += 1
            self.outputs[i].append(token)

            stopped_by_eos = stop_on_eos and (eos_token_id is not None) and (token == eos_token_id)
            reached_budget = self.generated_counts[i] >= self.budgets[i]

            if stopped_by_eos or reached_budget:
                self.curr_active[i] = False
                self.completion_steps[i] = step + 1
                self.finish_reasons[i] = "stop" if stopped_by_eos else "max_tokens"

        return not any(self.curr_active)

    def _next_positions(self) -> list[int]:
        """计算当前活跃槽位在 Decode 阶段的 position_id（即已生成的总长度 - 1）。"""
        return [self.prompt_lengths[i] + self.generated_counts[i] - 1 for i in range(self.batch_size)]

    def _compute_batch_metrics(self, step_latencies_ms: list[float]) -> BatchScheduleInfo[float]:
        """统计批处理生成过程特有明细指标（利用率、Padding 效率等）。"""
        step_count = len(step_latencies_ms)
        valid_generated_tokens_count = sum(len(output) for output in self.outputs)

        num_prompt_tokens = sum(self.prompt_lengths)
        num_padded_prompt_tokens = self.batch_size * self.padded_prompt_length
        capacity_tokens = self.batch_size * step_count

        return BatchScheduleInfo(
            padding_efficiency=num_prompt_tokens / num_padded_prompt_tokens if num_padded_prompt_tokens else 0.0,
            slot_utilization=valid_generated_tokens_count / capacity_tokens if capacity_tokens else 0.0,
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
    has_uniform_prompt_len: bool,
    budgets: torch.Tensor,
    use_cache: bool,
    device: torch.device,
    *,
    pad_token_id: int = 0,
    eos_token_id: int | None = None,
    stop_on_eos: bool = True,
    capture_logits: bool = False,
    baseline_allocated_bytes: int = 0,
) -> GenerationResult:
    """执行齐步（Lockstep）自回归生成迭代循环（所有请求同时开始，含 step 0 Prefill 与后续 Decode 步）。

    内部驱动纯 CPU LockstepState 状态机，委托 LockstepRunner 在数据面自治流转 GPU 输入与 KV Cache，
    直接构建并返回由 5 大核心共享值对象组成的 GenerationResult。
    """
    is_cuda = device.type == "cuda"
    runner = LockstepRunner(model, device, pad_token_id)
    batch = runner.init_input_batch(
        input_ids=input_ids,
        token_sequences=token_sequences,
        has_uniform_prompt_len=has_uniform_prompt_len,
        use_cache=use_cache,
    )
    budgets_list = budgets.tolist() if isinstance(budgets, torch.Tensor) else list(budgets)
    state = LockstepState._create(
        batch_size=len(batch.prompt_lengths),
        padded_prompt_length=batch.padded_prompt_length,
        prompt_lengths=batch.prompt_lengths,
        budgets=budgets_list,
    )

    step_event_pairs: list[tuple[Any, Any]] = []
    step_times_ms: list[float] = []
    logits_trace = [] if capture_logits else None
    memory_after_prefill_bytes = 0

    wall_start = time.perf_counter()

    for step in range(state.max_steps):
        step_out = runner.step(batch, capture_logits=capture_logits, use_cache=use_cache)
        if step_out.event_pair is not None:
            step_event_pairs.append(step_out.event_pair)
        else:
            step_times_ms.append(step_out.elapsed_ms)

        if step == 0 and is_cuda:
            memory_after_prefill_bytes = torch.cuda.memory_allocated(device)

        if capture_logits and step_out.last_logits is not None:
            logits_trace.append(step_out.last_logits.float().cpu())

        is_all_finished = state._commit_step(
            next_tokens=step_out.next_tokens,
            step=step,
            eos_token_id=eos_token_id,
            stop_on_eos=stop_on_eos,
        )
        if is_all_finished:
            break

        runner.prepare_next_inputs(
            batch,
            curr_active=state.curr_active,
            next_positions=state._next_positions(),
            next_tokens=step_out.next_tokens,
            use_cache=use_cache,
        )

    synchronize(device)
    e2e_wall_ms = (time.perf_counter() - wall_start) * 1000
    step_latencies_ms = _resolve_step_latencies(step_event_pairs, step_times_ms, is_cuda)

    valid_tokens = sum(len(o) for o in state.outputs)
    e2e_model_ms = sum(step_latencies_ms)
    batch_info = state._compute_batch_metrics(step_latencies_ms)

    step_count = len(step_latencies_ms)
    sequences = [
        SequenceOutput(
            request_idx=request_idx,
            token_ids=state.outputs[request_idx],
            prompt_len=state.prompt_lengths[request_idx],
            output_len=len(state.outputs[request_idx]),
            finish_reason=state.finish_reasons[request_idx],
            completion_step=state.completion_steps[request_idx]
            if state.completion_steps[request_idx] is not None
            else step_count,
        )
        for request_idx in range(state.batch_size)
    ]
    cache_bytes = batch.cache_bytes if use_cache else 0

    metrics = compute_step_aggregated_metrics(
        step_latencies_ms=step_latencies_ms,
        e2e_wall_ms=e2e_wall_ms,
        e2e_model_ms=e2e_model_ms,
        total_tokens=valid_tokens,
        batch_size=state.batch_size,
        is_cuda=is_cuda,
        num_prompt_tokens=sum(state.prompt_lengths),
        num_padded_prompt_tokens=state.batch_size * state.padded_prompt_length,
    )
    memory = build_memory_profile(
        baseline_allocated_bytes=baseline_allocated_bytes,
        memory_after_prefill_bytes=memory_after_prefill_bytes,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if is_cuda else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if is_cuda else 0,
        cache_bytes=cache_bytes,
    )
    trace = GenerationTrace(
        step_latencies_ms=step_latencies_ms,
        num_active_tokens_per_step=state.active_counts_per_step,
        logits_trace=logits_trace,
    )

    return GenerationResult(
        sequences=sequences,
        metrics=metrics,
        memory=memory,
        trace=trace,
        batch_size=state.batch_size,
        batch_info=batch_info,
    )
