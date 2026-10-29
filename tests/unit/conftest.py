"""单元测试共享 fixture。"""

import pytest
import torch

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
