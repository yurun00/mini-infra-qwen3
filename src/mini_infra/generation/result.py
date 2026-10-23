"""生成结果结构体与指标聚合。"""

from typing import Any

import torch

from mini_infra.profiling import merge_op_latency_breakdowns


def percentile(values: list[float], percent: int | float) -> float:
    """使用 nearest-rank 定义计算小样本延迟分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, (len(ordered) * percent + 99) // 100)
    return ordered[min(rank, len(ordered)) - 1]


class GenerationResult:
    """单次生成结果对象，采用分层结构组织：

    1. 核心结果 (Core Outputs)：
       - padded_output_tokens: [B, output_len] 输出 Token 矩形张量（不含 Prompt，短请求在右侧填充 pad_token_id）
       - unpadded_output_token_ids: 各请求真实有效输出 Token ID 列表（不含 Prompt，变长无 padding）
       - logits_trace: 调试用 Logits 轨迹（若启用 capture_logits）

    2. 核心标量指标 (Core Scalar Metrics，一等公民属性)：
       - 负载元数据：prompt_len, output_len, batch_size
       - 核心时延：ttft_ms, tpot_ms, p50_decode_step_ms, p95_decode_step_ms, e2e_wall_ms, e2e_model_ms
       - 吞吐与利用率：throughput_tok_s, request_throughput_per_second, gpu_time_ratio
       - 显存结构字典：memory（包含 baseline/prefill/peak/reserved/fragmentation/kv_cache_mb 等）

    3. 跟踪明细与策略扩展 (Details & Extra Metrics)：
       - step_latencies_ms: 各 decode 步耗时明细列表（Trace）
       - extra: 批处理或具体策略特有的指标字典（如 slot_utilization, padding_efficiency, 串行聚合模式的 runs 等）
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "GenerationResult 不允许直接实例化，请使用工厂方法 GenerationResult.build(...) "
            "或 GenerationResult._merge_generation_results(...)。"
        )

    @classmethod
    def _create(
        cls,
        # 1. 核心结果 (Core Outputs)
        padded_output_tokens: torch.Tensor,
        unpadded_output_token_ids: list[list[int]] | None = None,
        logits_trace: list[torch.Tensor] | None = None,
        *,
        # 2. 核心标量指标 (Core Scalar Metrics)
        # 2.1 负载元数据
        prompt_len: int,
        output_len: int,
        batch_size: int,
        # 2.2 核心时延
        ttft_ms: float,
        tpot_ms: float,
        p50_decode_step_ms: float,
        p95_decode_step_ms: float,
        e2e_wall_ms: float,
        e2e_model_ms: float,
        # 2.3 吞吐与利用率
        throughput_tok_s: float,
        request_throughput_per_second: float,
        gpu_time_ratio: float | None,
        # 2.4 显存结构字典 (MB)
        memory: dict[str, Any],
        # 3. 跟踪明细与策略扩展 (Details & Extra Metrics)
        step_latencies_ms: list[float],
        op_latency_breakdown: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "GenerationResult":
        """内部专用的分层实例构造方法，参数顺序严格对应类文档的 3 层架构。"""
        self = object.__new__(cls)

        # 1. 核心结果 (Core Outputs)
        self.padded_output_tokens = padded_output_tokens
        self.unpadded_output_token_ids = unpadded_output_token_ids or []
        self.logits_trace = logits_trace

        # 2. 核心标量属性 (Core Scalar Metrics)
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.batch_size = batch_size
        self.ttft_ms = ttft_ms
        self.tpot_ms = tpot_ms
        self.p50_decode_step_ms = p50_decode_step_ms
        self.p95_decode_step_ms = p95_decode_step_ms
        self.e2e_wall_ms = e2e_wall_ms
        self.e2e_model_ms = e2e_model_ms
        self.throughput_tok_s = throughput_tok_s
        self.request_throughput_per_second = request_throughput_per_second
        self.gpu_time_ratio = gpu_time_ratio
        self.memory = memory

        # 3. 原始明细与批处理策略扩展 (Details & Extra Metrics)
        self.step_latencies_ms = step_latencies_ms
        self.op_latency_breakdown = op_latency_breakdown
        self.extra = extra or {}

        return self

    def __getattr__(self, name: str) -> Any:
        if name in self.extra:
            return self.extra[name]
        if name in self.memory:
            return self.memory[name]
        raise AttributeError(f"'GenerationResult' object has no attribute '{name}'")

    def __getitem__(self, item: Any) -> Any:
        if not isinstance(item, str):
            raise KeyError(item)
        if hasattr(self, item) and item not in ("get", "__getitem__", "__getattr__"):
            return getattr(self, item)
        if item in self.extra:
            return self.extra[item]
        if item in self.memory:
            return self.memory[item]
        raise KeyError(item)

    def __contains__(self, item: Any) -> bool:
        if not isinstance(item, str):
            return False
        if hasattr(self, item) and item not in ("get", "__getitem__", "__getattr__"):
            return True
        return item in self.extra or item in self.memory

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    @classmethod
    def build(
        cls,
        # 1. 核心结果 (Core Outputs)
        padded_output_tokens: torch.Tensor,
        unpadded_output_token_ids: list[list[int]],
        logits_trace: list[torch.Tensor] | None = None,
        *,
        # 2. 核心标量指标源数据 (Core Scalar Metrics Sources)
        # 2.1 负载规格
        prompt_len: int,
        output_len: int,
        batch_size: int,
        # 2.2 耗时与时延源数据（用于推导 ttft, tpot, throughput, gpu_ratio）
        e2e_wall_ms: float,
        step_latencies_ms: list[float],
        e2e_model_ms: float | None = None,
        # 2.3 硬件显存监控（原始字节数，内部统一折算为 MB）
        baseline_allocated_bytes: int,
        memory_after_prefill_bytes: int,
        peak_allocated_bytes: int,
        peak_reserved_bytes: int,
        cache_bytes: int,
        # 3. 硬件环境与策略扩展 (Flags & Extra Metrics)
        is_cuda: bool = False,
        valid_generated_tokens_count: int | None = None,
        op_latency_breakdown: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "GenerationResult":
        """工厂方法：严格按照 (1) 核心结果 -> (2) 标量指标源数据 -> (3) 环境与扩展 分层构建 GenerationResult。"""

        mib = 1024**2
        decode_latencies = step_latencies_ms[1:]
        ttft_ms = step_latencies_ms[0] if step_latencies_ms else 0.0
        tpot_ms = (sum(decode_latencies) / len(decode_latencies)) if decode_latencies else 0.0
        p50_decode_step_ms = percentile(decode_latencies, 50)
        p95_decode_step_ms = percentile(decode_latencies, 95)
        if e2e_model_ms is None:
            e2e_model_ms = sum(step_latencies_ms)
        # 批处理时各请求输出长度不同，用 valid_generated_tokens_count 统计真正有效的输出 Token 数
        total_tokens = (
            valid_generated_tokens_count if valid_generated_tokens_count is not None else batch_size * output_len
        )
        throughput = total_tokens / (e2e_model_ms / 1000) if e2e_model_ms > 0 else 0.0
        request_throughput_per_second = batch_size * 1000 / e2e_model_ms if e2e_model_ms > 0 else 0.0
        # gpu_time_ratio 专属于 GPU（GPU 忙碌时间 / 墙钟时间）：非 CUDA 设备记为 None
        gpu_time_ratio = (min(1.0, e2e_model_ms / e2e_wall_ms) if e2e_wall_ms > 0 else 0.0) if is_cuda else None

        # 显存结构化指标：统一为 MB 维度
        baseline_mb = baseline_allocated_bytes / mib
        prefill_mb = memory_after_prefill_bytes / mib
        peak_mb = peak_allocated_bytes / mib
        reserved_mb = peak_reserved_bytes / mib
        memory = {
            "baseline_allocated_mb": baseline_mb,
            "prefill_allocated_mb": prefill_mb,
            "peak_allocated_mb": peak_mb,
            "dynamic_peak_mb": max(0.0, peak_mb - baseline_mb),
            "peak_reserved_mb": reserved_mb,
            "fragmentation_mb": max(0.0, reserved_mb - peak_mb),
            "kv_cache_mb": cache_bytes / mib,
        }

        return cls._create(
            # 1. 核心结果
            padded_output_tokens=padded_output_tokens,
            unpadded_output_token_ids=unpadded_output_token_ids,
            logits_trace=logits_trace,
            # 2. 核心标量指标
            prompt_len=prompt_len,
            output_len=output_len,
            batch_size=batch_size,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            p50_decode_step_ms=p50_decode_step_ms,
            p95_decode_step_ms=p95_decode_step_ms,
            e2e_wall_ms=e2e_wall_ms,
            e2e_model_ms=e2e_model_ms,
            throughput_tok_s=throughput,
            request_throughput_per_second=request_throughput_per_second,
            gpu_time_ratio=gpu_time_ratio,
            memory=memory,
            # 3. 跟踪明细与策略扩展
            step_latencies_ms=step_latencies_ms,
            op_latency_breakdown=op_latency_breakdown,
            extra=extra,
        )

    @classmethod
    def _merge_generation_results(cls, runs: list["GenerationResult"], pad_token_id: int) -> "GenerationResult":
        """把各运行结果聚合为一个 GenerationResult（时间相加，显存取峰值，请求展开）。"""
        output_ids: list[list[int]] = []
        for run in runs:
            output_ids.extend(run.unpadded_output_token_ids)

        total_requests = len(output_ids)
        valid_generated_tokens_count = sum(len(o) for o in output_ids)
        max_output_len = max((len(o) for o in output_ids), default=0)
        padded_output_tokens = torch.full((total_requests, max_output_len), pad_token_id, dtype=torch.long)
        for req_idx, out in enumerate(output_ids):
            padded_output_tokens[req_idx, : len(out)] = torch.tensor(out, dtype=torch.long)

        e2e_wall_ms = sum(run.e2e_wall_ms for run in runs)
        e2e_model_ms = sum(run.e2e_model_ms for run in runs)
        step_latencies = [x for run in runs for x in run.step_latencies_ms]
        decode = [x for run in runs for x in run.step_latencies_ms[1:]]
        gpu_time_ratios = [run.gpu_time_ratio for run in runs if run.gpu_time_ratio is not None]
        mems = [run.memory for run in runs]
        memory = {
            "baseline_allocated_mb": mems[0]["baseline_allocated_mb"] if mems else 0.0,
            "prefill_allocated_mb": max((m["prefill_allocated_mb"] for m in mems), default=0.0),
            "peak_allocated_mb": max((m["peak_allocated_mb"] for m in mems), default=0.0),
            "dynamic_peak_mb": max((m["dynamic_peak_mb"] for m in mems), default=0.0),
            "peak_reserved_mb": max((m["peak_reserved_mb"] for m in mems), default=0.0),
            "fragmentation_mb": max((m["fragmentation_mb"] for m in mems), default=0.0),
            "kv_cache_mb": max((m["kv_cache_mb"] for m in mems), default=0.0),
        }
        request_throughput_per_second = total_requests * 1000 / e2e_model_ms if e2e_model_ms else 0.0
        op_latency_breakdowns = [
            r.op_latency_breakdown for r in runs if getattr(r, "op_latency_breakdown", None) is not None
        ]
        op_latency_breakdown = merge_op_latency_breakdowns(op_latency_breakdowns)
        extra = {
            "runs": runs,
        }

        return cls._create(
            # 1. 核心结果
            padded_output_tokens=padded_output_tokens,
            unpadded_output_token_ids=output_ids,
            logits_trace=None,
            # 2. 核心标量指标
            prompt_len=max((run.prompt_len for run in runs), default=0),
            output_len=max_output_len,
            batch_size=total_requests,
            ttft_ms=(sum(run.ttft_ms * run.batch_size for run in runs) / total_requests) if total_requests else 0.0,
            tpot_ms=sum(decode) / len(decode) if decode else 0.0,
            p50_decode_step_ms=percentile(decode, 50),
            p95_decode_step_ms=percentile(decode, 95),
            e2e_wall_ms=e2e_wall_ms,
            e2e_model_ms=e2e_model_ms,
            throughput_tok_s=valid_generated_tokens_count * 1000 / e2e_model_ms if e2e_model_ms else 0.0,
            request_throughput_per_second=request_throughput_per_second,
            gpu_time_ratio=sum(gpu_time_ratios) / len(gpu_time_ratios) if gpu_time_ratios else None,
            memory=memory,
            # 3. 跟踪明细与策略扩展
            step_latencies_ms=step_latencies,
            op_latency_breakdown=op_latency_breakdown,
            extra=extra,
        )
