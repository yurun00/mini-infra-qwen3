"""GPU 测试：验证同一手写模型在开启 KV Cache 与关闭 Cache 时的逐 Token / Logits 完全对齐。

需要 GPU 和已下载的权重；不满足时自动跳过。
运行：pytest -m gpu
"""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_cache():
    from mini_infra.testing.parity.compare_cache import main

    # 不一致时 main 会 raise SystemExit(非零)，pytest 会将其视为失败
    main(["--prompt", "请用一句话介绍 KV Cache。", "--max-new-tokens", "16"])
