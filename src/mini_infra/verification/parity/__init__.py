"""数值对齐与等价性验证工具集（Parity Verification）。

包含：
- compare_cache：KV Cache 前后逐步 Logits 对齐与数值等价性验证
- compare_paged_cache_logits：Dense 与 Paged KV Cache 逐 Token Logits 对齐与数值等价性验证
- compare_paged_cache_scheduler：Scheduler 调度环境下 Dense 与多 Block Size Paged KV Cache 输出一致性验证
- compare_transformers：手写模型实现与 Hugging Face 参考实现的黄金基准对齐
- compare_batch_lockstep：Lockstep 齐步批处理不同 Batch Size / 排列 / Padding 变种的请求隔离与数值等价性验证
- compare_batch_scheduler：Scheduler 批处理（Fixed 与 Continuous）动态请求归属与输出一致性验证
- compare_schedulers_token_budget：prefill-first 与 token-budget 两种迭代级调度策略的输出一致性与调度轨迹验证
"""
