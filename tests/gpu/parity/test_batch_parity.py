"""GPU 测试：验证固定批处理在不同 Batch Size、排列、Padding 及共存请求下的请求隔离与数值对齐。

需要 GPU 和已下载的权重；不满足时自动跳过。
运行：pytest -m gpu
"""

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA GPU")
def test_compare_batch():
    from mini_infra.verification.parity.compare_batch import main

    # 隔离或数值检查未通过时 main 会 raise SystemExit(非零)，pytest 会将其视为失败
    main(["--prompt", "用一句话解释固定批处理。", "--max-new-tokens", "8"])
