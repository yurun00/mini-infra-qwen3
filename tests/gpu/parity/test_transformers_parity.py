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
    from mini_infra.verification.parity.compare_with_transformers import main

    # 不一致时 main 会 raise SystemExit(非零)，pytest 会将其视为失败
    main(["--prompt", "请用一句话介绍 KV Cache。", "--max-new-tokens", "16"])
