"""数值对齐与等价性验证工具集（Parity Verification）。

包含：
- compare_cache：KV Cache 前后逐步 Logits 对齐与数值等价性验证
- compare_transformers：手写模型实现与 Hugging Face 参考实现的黄金基准对齐
- compare_lockstep_batch：Lockstep 齐步批处理不同 Batch Size / 排列 / Padding 变种的请求隔离与数值等价性验证
- compare_scheduler_batch：Scheduler 批处理（Fixed 与 Continuous）动态请求归属与输出一致性验证
"""
