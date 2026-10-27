"""请求规格与运行态：Serving 调度所需的最小数据模型。"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class RequestStatus(str, Enum):
    """请求生命周期状态。"""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RequestSpec:
    """单个请求的不可变规格（逻辑时间线上的到达时间，不涉及真实 sleep）。"""

    request_index: int
    token_ids: tuple[int, ...]
    max_new_tokens: int
    arrival_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.request_index < 0:
            raise ValueError("request_index 不能为负数")
        if not self.token_ids:
            raise ValueError("Prompt Token 不能为空")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens 必须大于 0")
        if self.arrival_ms < 0:
            raise ValueError("arrival_ms 不能小于 0")

    @property
    def prompt_len(self) -> int:
        return len(self.token_ids)


@dataclass
class RequestState:
    """请求在调度过程中的可变运行态（槽位、已生成 Token 与关键时间戳）。"""

    spec: RequestSpec
    status: RequestStatus = RequestStatus.WAITING
    generated: list[int] = field(default_factory=list)
    slot_index: int | None = None
    cache_length: int = 0
    admitted_ms: float | None = None
    first_token_ms: float | None = None
    completion_ms: float | None = None
    token_times_ms: list[float] = field(default_factory=list)
    finish_reason: str = "max_tokens"  # "stop"（命中 EOS）或 "max_tokens"（达到预算上限）

    def append_token(
        self,
        token_id: int,
        timestamp_ms: float,
        eos_token_id: int | None = None,
        stop_on_eos: bool = True,
    ) -> bool:
        """追加一个生成的 Token 并记录时间戳；若触发结束条件（EOS 或预算耗尽）则记录完成元数据并返回 True。"""
        self.generated.append(token_id)
        self.token_times_ms.append(timestamp_ms)
        if self.first_token_ms is None:
            self.first_token_ms = timestamp_ms

        reached_eos = stop_on_eos and eos_token_id is not None and token_id == eos_token_id
        if reached_eos or len(self.generated) >= self.spec.max_new_tokens:
            self.completion_ms = timestamp_ms
            self.finish_reason = "stop" if reached_eos else "max_tokens"
            return True
        return False


def make_request_specs(
    token_sequences: Sequence[Any],
    max_new_tokens: int | Sequence[int],
    arrival_times_ms: Sequence[float] | None = None,
) -> list[RequestSpec]:
    """把 Prompt 序列、输出预算与到达时间组装成 RequestSpec 列表（request_index 按提交顺序编号）。"""
    sequences = [tuple(int(t) for t in torch.as_tensor(seq).tolist()) for seq in token_sequences]
    budgets = [max_new_tokens] * len(sequences) if isinstance(max_new_tokens, int) else list(max_new_tokens)
    arrivals = [0.0] * len(sequences) if arrival_times_ms is None else list(arrival_times_ms)
    if len(budgets) != len(sequences) or len(arrivals) != len(sequences):
        raise ValueError("Prompt、输出预算和到达时间的数量必须一致")
    return [
        RequestSpec(index, sequence, int(budget), float(arrival))
        for index, (sequence, budget, arrival) in enumerate(zip(sequences, budgets, arrivals))
    ]


def validate_request_specs(request_specs: Sequence[RequestSpec]) -> None:
    """校验请求列表非空且 request_index 唯一（单个规格的合法性已由 RequestSpec 自身保证）。"""
    if not request_specs:
        raise ValueError("request_specs 不能为空")
    if len({spec.request_index for spec in request_specs}) != len(request_specs):
        raise ValueError("request_index 必须唯一")
