"""批生成执行状态机，管理槽位激活状态、输入张量更新与完成判断。"""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class BatchGenerationState:
    """管理一次批处理生成中所有槽位的生命周期与张量更新。"""

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
    ) -> "BatchGenerationState":
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
        """更新槽位输出 Token、预算/EOS 判定并返回是否所有槽位均已完成。"""
        curr_active_before = self.curr_active.clone()
        self.active_counts_per_step.append(int(curr_active_before.sum().item()))

        for req_idx in range(self.batch_size):
            if not bool(curr_active_before[req_idx].item()):
                continue
            token_id = int(next_tokens[req_idx].item())
            self.outputs[req_idx].append(token_id)
            self.generated_counts[req_idx] += 1
            reached_eos = eos_token_id is not None and token_id == eos_token_id
            reached_budget = bool((self.generated_counts[req_idx] >= self.budgets[req_idx]).item())
            if (stop_on_eos and reached_eos) or reached_budget:
                self.curr_active[req_idx] = False
                self.completion_steps[req_idx] = step + 1

        return not bool(self.curr_active.any().item())

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

    def build_padded_output_tokens(self, step_count: int) -> torch.Tensor:
        """构建 [B, step_count] 的 padded_output_tokens 矩形张量。"""
        padded_output_tokens = torch.full((self.batch_size, step_count), self.pad_token_id, dtype=torch.long)
        for req_idx, output in enumerate(self.outputs):
            padded_output_tokens[req_idx, : len(output)] = torch.tensor(output, dtype=torch.long)
        return padded_output_tokens

    def compute_batch_metrics(
        self,
        step_latencies_ms: list[float],
        kv_cache: Any,
    ) -> dict[str, Any]:
        """统计批处理生成过程特有明细指标（利用率、Padding 效率、各请求完成时延等）。"""
        step_count = len(step_latencies_ms)
        valid_generated_tokens_count = sum(len(output) for output in self.outputs)

        cumulative_ms: list[float] = []
        running_ms = 0.0
        for value in step_latencies_ms:
            running_ms += value
            cumulative_ms.append(running_ms)

        prompt_tokens = int(self.prompt_lengths.sum().item())
        padded_prompt_tokens = self.batch_size * self.padded_prompt_length
        capacity_tokens = self.batch_size * step_count
        ttft_ms = step_latencies_ms[0] if step_latencies_ms else 0.0

        return {
            "prompt_lengths_per_req": self.prompt_lengths.cpu().tolist(),
            "prompt_tokens": prompt_tokens,
            "padded_prompt_tokens": padded_prompt_tokens,
            "padding_efficiency": prompt_tokens / padded_prompt_tokens,
            "effective_prompt_token_throughput_per_second": prompt_tokens * 1000 / ttft_ms if ttft_ms else 0.0,
            "padded_prompt_token_throughput_per_second": padded_prompt_tokens * 1000 / ttft_ms if ttft_ms else 0.0,
            "active_counts_per_step": self.active_counts_per_step,
            "slot_utilization": valid_generated_tokens_count / capacity_tokens if capacity_tokens else 0.0,
            "completion_steps_per_req": self.completion_steps,
            "completion_times_ms_per_req": [
                cumulative_ms[s - 1] if s is not None else None for s in self.completion_steps
            ],
            "cache_length": kv_cache.get_seq_length() if kv_cache is not None else 0,
        }
