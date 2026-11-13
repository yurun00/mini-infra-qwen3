"""单元测试共享 fixture。"""

from collections.abc import Callable

import pytest
import torch

from mini_infra.core import SchedulePhase
from mini_infra.engine import ScheduleResult
from mini_infra.verification.smoke import build_tiny_model


@pytest.fixture(scope="session")
def diverse_model():
    """输出有区分度的 tiny 模型（CPU）。

    build_tiny_model 的 lm_head 与 embedding 绑定，随机初始化下贪心输出会退化为“不断重复最后一个输入 Token”，
    此时位置编码、Cache 错位等调度 bug 都不会改变 Token，parity 测试形同虚设。
    这里解绑 lm_head 并使用独立随机种子，使输出依赖注意力与历史上下文。
    （注意：种子不能与 build_tiny_model 的 manual_seed(0) 相同，否则 lm_head 会与 embedding 完全相同。）
    """
    model = build_tiny_model()
    generator = torch.Generator().manual_seed(1234)
    model.lm_head.weight = torch.nn.Parameter(torch.randn(model.lm_head.weight.shape, generator=generator))
    return model


def _assert_token_budget_invariants(result: ScheduleResult, token_budget: int) -> None:
    prompt_len_by_request_idx = {s.request_idx: s.prompt_len for s in result.sequences}
    for event in result.events:
        if event.phase is SchedulePhase.PREFILL:
            prompt_lens = [prompt_len_by_request_idx[i] for i in event.admitted_request_ids]
            assert event.num_scheduled_tokens == len(prompt_lens) * max(prompt_lens)
            # ① 非 oversize prefill 不超预算；② oversize prefill 只含 1 个请求
            assert event.num_scheduled_tokens <= token_budget or len(prompt_lens) == 1
        else:
            assert event.num_scheduled_tokens == event.num_active_requests
    # ③ RunningQueue 非空时不出现连续两步 PREFILL（前一步入场后仍有请求在跑即违反 decode protection）
    for prev, curr in zip(result.events, result.events[1:]):
        if prev.phase is SchedulePhase.PREFILL and curr.phase is SchedulePhase.PREFILL:
            assert prev.num_active_requests == len(prev.finished_request_ids)


@pytest.fixture
def assert_token_budget_invariants() -> Callable[[ScheduleResult, int], None]:
    """token-budget 调度的事件不变量断言（Dense 与 Paged 路径共用）。"""
    return _assert_token_budget_invariants
