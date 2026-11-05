"""设备端贪心采样算子（Worker 层 Sampler）。

只吃裸 logits 张量、吐 token 张量，对请求状态、调度策略与计时完全无感知。
"""

import torch


def greedy_sample(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对最后位置做贪心采样（取 argmax）。

    ```text
    logits:      [batch, seq_len, vocab_size], 继承模型 dtype, contiguous
    next_token:  [batch, 1],                    int64
    last_logits: [batch, vocab_size],           继承模型 dtype
    ```

    Returns:
        (next_token, last_logits)；last_logits 供 capture_logits 路径留存，不参与采样以外的计算。
    """
    last_logits = logits[:, -1, :]
    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
    return next_token, last_logits


__all__ = [
    "greedy_sample",
]
