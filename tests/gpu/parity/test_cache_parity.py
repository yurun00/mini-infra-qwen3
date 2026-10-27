"""GPU 测试：验证同一手写模型在开启 KV Cache 与关闭 Cache 时的逐 Token / Logits 完全对齐。

需要 GPU 和已下载的权重；不满足时自动跳过。
运行：pytest -m gpu
"""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_cache():
    from mini_infra.cli import main

    assert main(["verify", "cache", "--prompt", "请用一句话介绍 KV Cache。", "--max-new-tokens", "16"]) == 0
