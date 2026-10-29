"""GPU 测试：与 Hugging Face Transformers 的逐 Token / Logits 对齐。

需要 GPU、已下载的权重和 `pip install -e ".[compare]"`；不满足时自动跳过。
运行：pytest -m gpu
"""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_parity_with_transformers():
    pytest.importorskip("transformers")
    from mini_infra.cli import main

    assert main(["verify", "transformers", "--prompt", "请用一句话介绍 KV Cache。", "--max-new-tokens", "16"]) == 0
