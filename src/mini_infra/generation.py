"""两条单请求生成路径与计时工具。"""

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import torch

from mini_infra.qwen3_model import KVCache, cache_size_bytes


class FeatureFlag(str, Enum):
    """支持的所有布尔型特性标志（Feature Flag）；每个成员对应一个可独立开启/关闭（enable/disable）的二值特性。"""

    KV_CACHE = "kv-cache"

    @classmethod
    def from_str(cls, value: "str | FeatureFlag") -> "FeatureFlag":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"未知特性标志: {value}，可选特性: {[m.value for m in cls]}") from err

    def __str__(self) -> str:
        return self.value


class BatchingMethod(str, Enum):
    """互斥的批处理策略。"""

    NONE = "none"
    FIXED = "fixed-batching"
    CONTINUOUS = "continuous-batching"

    @classmethod
    def from_str(cls, value: "str | BatchingMethod | None") -> "BatchingMethod":
        if value is None:
            return cls.NONE
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"未知批处理策略: {value}，可选策略: {[m.value for m in cls]}") from err

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Mode:
    """运行模式结构体：内部存储已启用的 FeatureFlag 集合及批处理策略（BatchingMethod）。

    - flags: 正交的功能开关集合（如 FeatureFlag.KV_CACHE）。
    - batching: 互斥的批处理策略（BatchingMethod.NONE / FIXED / CONTINUOUS）。
    """

    flags: frozenset[FeatureFlag] = frozenset()
    batching: BatchingMethod = BatchingMethod.NONE

    def __init__(
        self,
        flags: "Iterable[FeatureFlag | str] | None" = None,
        batching: BatchingMethod | str = BatchingMethod.NONE,
    ):
        if flags is None:
            flag_set = frozenset()
        elif isinstance(flags, frozenset):
            flag_set = flags
        else:
            flag_set = frozenset(FeatureFlag.from_str(f) for f in flags)

        batching_method = BatchingMethod.from_str(batching)

        object.__setattr__(self, "flags", flag_set)
        object.__setattr__(self, "batching", batching_method)

    def is_enabled(self, flag: FeatureFlag | str) -> bool:
        """检查指定的 FeatureFlag 是否启用。"""
        if isinstance(flag, str):
            flag = FeatureFlag.from_str(flag)
        return flag in self.flags

    def with_flag(self, flag: FeatureFlag, enabled: bool = True) -> "Mode":
        """返回指定 FeatureFlag 状态修改后的新 Mode 实例。"""
        new_flags = (self.flags | {flag}) if enabled else (self.flags - {flag})
        return Mode(flags=new_flags, batching=self.batching)

    def with_batching(self, batching: BatchingMethod | str) -> "Mode":
        """返回指定 batching 策略修改后的新 Mode 实例。"""
        return Mode(flags=self.flags, batching=batching)

    @property
    def kv_cache(self) -> bool:
        return self.is_enabled(FeatureFlag.KV_CACHE)

    @property
    def fixed_batching(self) -> bool:
        return self.batching == BatchingMethod.FIXED

    @property
    def continuous_batching(self) -> bool:
        return self.batching == BatchingMethod.CONTINUOUS

    @property
    def is_batch(self) -> bool:
        return self.batching != BatchingMethod.NONE

    def to_dict(self) -> dict[str, Any]:
        """将 mode 中的所有 feature flag 状态及 batching 转换为字典。"""
        result = {flag.value: self.is_enabled(flag) for flag in FeatureFlag}
        result["batching"] = self.batching.value
        return result

    def __contains__(self, item: FeatureFlag | BatchingMethod | str) -> bool:
        if isinstance(item, BatchingMethod):
            return self.batching == item
        if isinstance(item, str):
            if item in [b.value for b in BatchingMethod]:
                return self.batching == BatchingMethod(item)
            return FeatureFlag.from_str(item) in self.flags
        return item in self.flags

    def __str__(self) -> str:
        parts = sorted(flag.value for flag in self.flags)
        if self.batching != BatchingMethod.NONE:
            parts.append(self.batching.value)
        if not parts:
            return "no-cache"
        return "+".join(parts)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], percent: int | float) -> float:
    """使用 nearest-rank 定义计算小样本延迟分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, (len(ordered) * percent + 99) // 100)
    return ordered[min(rank, len(ordered)) - 1]


class StepForwardResult(NamedTuple):
    """单步前向计时输出对象：

    - 属性访问：.next_token, .elapsed_ms, .event_pair, .last_logits
    - 兼容元组解包：(next_token, elapsed_ms, event_pair)（last_logits 仅在 return_logits=True 时填充）
    """

    next_token: torch.Tensor
    elapsed_ms: float = 0.0
    event_pair: tuple[Any, Any] | None = None
    last_logits: torch.Tensor | None = None


def _timed_forward(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    device: torch.device | None = None,
    kv_cache: KVCache | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    return_logits: bool = False,
) -> StepForwardResult:
    """单步前向推理并计时。

    attention_mask / position_ids 用于批处理（左 Padding）；return_logits=True 时额外返回最后位置的 logits。
    """
    if device is None:
        device = model_input.device

    event_pair = None
    elapsed_ms = 0.0

    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        t0 = time.perf_counter()

    logits = model(model_input, kv_cache=kv_cache, attention_mask=attention_mask, position_ids=position_ids)
    last_logits = logits[:, -1, :]
    next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

    if device.type == "cuda":
        end.record()
        event_pair = (start, end)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000

    result_logits = last_logits if return_logits else None
    del logits
    return StepForwardResult(
        next_token=next_token, elapsed_ms=elapsed_ms, event_pair=event_pair, last_logits=result_logits
    )


def _resolve_step_latencies(
    step_event_pairs: list[tuple[Any, Any]], step_times_ms: list[float], is_cuda: bool
) -> list[float]:
    """CUDA 下使用 Event 硬件测时，非 CUDA 下使用 CPU 墙钟。"""
    if is_cuda and step_event_pairs:
        return [s.elapsed_time(e) for s, e in step_event_pairs]
    return step_times_ms


class GenerationResult:
    """单次生成结果对象，采用分层结构组织：

    1. 核心结果 (Core Outputs)：
       - new_tokens: [B, steps] 输出 Token 张量
       - new_token_ids: 各请求真实有效输出 Token ID 列表
       - logits_trace: 调试用 Logits 轨迹（若启用 capture_logits）

    2. 核心标量指标 (Core Scalar Metrics，一等公民属性)：
       - 负载元数据：prompt_len, output_len, batch_size
       - 核心时延：ttft_ms, tpot_ms, p50_decode_step_ms, p95_decode_step_ms, e2e_ms
       - 吞吐与利用率：throughput_tok_s, gpu_time_ratio
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
        new_tokens: torch.Tensor,
        new_token_ids: list[list[int]] | None = None,
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
        e2e_ms: float,
        # 2.3 吞吐与利用率
        throughput_tok_s: float,
        gpu_time_ratio: float | None,
        # 2.4 显存结构字典 (MB)
        memory: dict[str, Any],
        # 3. 跟踪明细与策略扩展 (Details & Extra Metrics)
        step_latencies_ms: list[float],
        extra: dict[str, Any] | None = None,
    ) -> "GenerationResult":
        """内部专用的分层实例构造方法，参数顺序严格对应类文档的 3 层架构。"""
        self = object.__new__(cls)

        # 1. 核心结果 (Core Outputs)
        self.new_tokens = new_tokens
        self.new_token_ids = new_token_ids
        self.logits_trace = logits_trace

        # 2. 核心标量属性 (Core Scalar Metrics)
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.batch_size = batch_size
        self.ttft_ms = ttft_ms
        self.tpot_ms = tpot_ms
        self.p50_decode_step_ms = p50_decode_step_ms
        self.p95_decode_step_ms = p95_decode_step_ms
        self.e2e_ms = e2e_ms
        self.throughput_tok_s = throughput_tok_s
        self.gpu_time_ratio = gpu_time_ratio
        self.memory = memory

        # 3. 原始明细与批处理策略扩展 (Details & Extra Metrics)
        self.step_latencies_ms = step_latencies_ms
        self.extra = extra or {}

        return self

    @property
    def metrics(self) -> dict[str, Any]:
        """合并核心标量指标与扩展指标后的完整字典视图（保持与原有代码及 JSON 落盘兼容）。"""
        res = {
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "batch_size": self.batch_size,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "p50_decode_step_ms": self.p50_decode_step_ms,
            "p95_decode_step_ms": self.p95_decode_step_ms,
            "e2e_ms": self.e2e_ms,
            "step_latencies_ms": self.step_latencies_ms,
            "throughput_tok_s": self.throughput_tok_s,
            "gpu_time_ratio": self.gpu_time_ratio,
            "memory": self.memory,
        }
        res.update(self.extra)
        return res

    def to_dict(self) -> dict[str, Any]:
        """全量指标转换为字典格式。"""
        return self.metrics

    def __getattr__(self, name: str) -> Any:
        if name in self.extra:
            return self.extra[name]
        if name in self.memory:
            return self.memory[name]
        raise AttributeError(f"'GenerationResult' object has no attribute '{name}'")

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item) and item not in ("get", "__getitem__", "__getattr__"):
            return getattr(self, item)
        if item in self.extra:
            return self.extra[item]
        if item in self.memory:
            return self.memory[item]
        raise KeyError(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    @classmethod
    def build(
        cls,
        # 1. 核心结果 (Core Outputs)
        new_tokens: torch.Tensor,
        new_token_ids: list[list[int]] | None = None,
        logits_trace: list[torch.Tensor] | None = None,
        *,
        # 2. 核心标量指标源数据 (Core Scalar Metrics Sources)
        # 2.1 负载规格
        prompt_len: int,
        output_len: int,
        batch_size: int,
        # 2.2 耗时与时延源数据（用于推导 ttft, tpot, throughput, gpu_ratio）
        e2e_ms: float,
        step_latencies_ms: list[float],
        # 2.3 硬件显存监控（原始字节数，内部统一折算为 MB）
        baseline_allocated_bytes: int,
        memory_after_prefill_bytes: int,
        peak_allocated_bytes: int,
        peak_reserved_bytes: int,
        cache_bytes: int,
        # 3. 硬件环境与策略扩展 (Flags & Extra Metrics)
        is_cuda: bool = False,
        valid_generated_tokens_count: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "GenerationResult":
        """工厂方法：严格按照 (1) 核心结果 -> (2) 标量指标源数据 -> (3) 环境与扩展 分层构建 GenerationResult。"""
        mib = 1024**2
        decode_latencies = step_latencies_ms[1:]
        ttft_ms = step_latencies_ms[0] if step_latencies_ms else 0.0
        tpot_ms = (sum(decode_latencies) / len(decode_latencies)) if decode_latencies else 0.0
        p50_decode_step_ms = percentile(decode_latencies, 50)
        p95_decode_step_ms = percentile(decode_latencies, 95)
        # 批处理时各请求输出长度不同，用 valid_generated_tokens_count 统计真正有效的输出 Token 数
        total_tokens = (
            valid_generated_tokens_count if valid_generated_tokens_count is not None else batch_size * output_len
        )
        throughput = total_tokens / (e2e_ms / 1000) if e2e_ms > 0 else 0.0
        # gpu_time_ratio 专属于 GPU（GPU 忙碌时间 / 墙钟时间）：非 CUDA 设备记为 None
        gpu_ratio = (min(1.0, sum(step_latencies_ms) / e2e_ms) if e2e_ms > 0 else 0.0) if is_cuda else None

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
            new_tokens=new_tokens,
            new_token_ids=new_token_ids,
            logits_trace=logits_trace,
            # 2. 核心标量指标
            prompt_len=prompt_len,
            output_len=output_len,
            batch_size=batch_size,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            p50_decode_step_ms=p50_decode_step_ms,
            p95_decode_step_ms=p95_decode_step_ms,
            e2e_ms=e2e_ms,
            throughput_tok_s=throughput,
            gpu_time_ratio=gpu_ratio,
            memory=memory,
            # 3. 跟踪明细与策略扩展
            step_latencies_ms=step_latencies_ms,
            extra=extra,
        )

    @classmethod
    def _merge_generation_results(cls, runs: list["GenerationResult"], pad_token_id: int) -> "GenerationResult":
        """把各运行结果聚合为一个 GenerationResult（时间相加，显存取峰值，请求展开）。"""
        output_ids: list[list[int]] = []
        for run in runs:
            output_ids.extend(run.new_token_ids)

        total_requests = len(output_ids)
        valid_generated_tokens_count = sum(len(o) for o in output_ids)
        max_len = max((len(o) for o in output_ids), default=0)
        new_tokens = torch.full((total_requests, max_len), pad_token_id, dtype=torch.long)
        for req_idx, out in enumerate(output_ids):
            new_tokens[req_idx, : len(out)] = torch.tensor(out, dtype=torch.long)

        e2e_ms = sum(run.e2e_ms for run in runs)
        step_latencies = [x for run in runs for x in run.step_latencies_ms]
        decode = [x for run in runs for x in run.step_latencies_ms[1:]]
        gpu_ratios = [run.gpu_time_ratio for run in runs if run.gpu_time_ratio is not None]
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
        extra = {
            "request_throughput_per_second": total_requests * 1000 / e2e_ms if e2e_ms else 0.0,
            "runs": runs,
        }

        return cls._create(
            # 1. 核心结果
            new_tokens=new_tokens,
            new_token_ids=output_ids,
            logits_trace=None,
            # 2. 核心标量指标
            prompt_len=max((run.prompt_len for run in runs), default=0),
            output_len=max_len,
            batch_size=total_requests,
            ttft_ms=(sum(run.ttft_ms * run.batch_size for run in runs) / total_requests) if total_requests else 0.0,
            tpot_ms=sum(decode) / len(decode) if decode else 0.0,
            p50_decode_step_ms=percentile(decode, 50),
            p95_decode_step_ms=percentile(decode, 95),
            e2e_ms=e2e_ms,
            throughput_tok_s=valid_generated_tokens_count * 1000 / e2e_ms if e2e_ms else 0.0,
            gpu_time_ratio=sum(gpu_ratios) / len(gpu_ratios) if gpu_ratios else None,
            memory=memory,
            # 3. 跟踪明细与策略扩展
            step_latencies_ms=step_latencies,
            extra=extra,
        )


def left_pad_sequences(token_sequences, pad_token_id, device):
    """把一组非空的一维 Token 序列左 Padding 为矩形 Tensor。

    Returns:
        input_ids [B, L], attention_mask [B, L] (bool), position_ids [B, L], lengths [B]
    """
    if not token_sequences:
        raise ValueError("token_sequences 不能为空")
    lengths = []
    normalized = []
    for sequence in token_sequences:
        tensor = torch.as_tensor(sequence, dtype=torch.long, device=device)
        if tensor.ndim != 1 or tensor.numel() < 1:
            raise ValueError("每个 Prompt 必须是一维非空 Token 序列")
        normalized.append(tensor)
        lengths.append(tensor.numel())

    max_length = max(lengths)
    batch_size = len(normalized)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    for row, sequence in enumerate(normalized):
        length = sequence.numel()
        input_ids[row, max_length - length :] = sequence
        attention_mask[row, max_length - length :] = True

    # 真实 Token 的逻辑位置从 0 开始；Padding 位置记为 0（会被 mask 掉）。
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.clamp_(min=0)
    return input_ids, attention_mask, position_ids, torch.tensor(lengths, dtype=torch.long, device=device)


def _normalize_budgets(max_new_tokens, batch_size, device):
    if isinstance(max_new_tokens, int):
        budgets = [max_new_tokens] * batch_size
    else:
        budgets = list(max_new_tokens)
    if len(budgets) != batch_size or any(value < 1 for value in budgets):
        raise ValueError("每个请求必须提供一个大于 0 的输出预算")
    return torch.tensor(budgets, dtype=torch.long, device=device)


def _normalize_token_sequences(input_ids: Sequence[Any]) -> list[torch.Tensor]:
    """统一将非 Tensor 的输入归一化为一维 Token 序列 Tensor 列表。"""
    if len(input_ids) > 0 and isinstance(input_ids[0], int):
        return [torch.as_tensor(input_ids, dtype=torch.long)]
    return [torch.as_tensor(s, dtype=torch.long) for s in input_ids]


@torch.inference_mode()
def generate(
    mode: Mode,
    model: torch.nn.Module,
    input_ids: "torch.Tensor | Sequence[Any]",
    max_new_tokens: "int | Sequence[int]",
    eos_token_id: int | None = None,
    device: torch.device | None = None,
    stop_on_eos: bool = True,
    pad_token_id: int = 0,
    capture_logits: bool = False,
) -> GenerationResult:
    """统一生成入口；mode 为 Mode 结构体对象。

    不同 mode 对应不同的 feature 开关，但生成的 output token 完全一致。

    batching=NONE 且请求数大于 1 时，请求按顺序逐个执行（串行 baseline）：
    input_ids 可为 [B, L] 张量或 B 条（可变长）一维 Token 序列，max_new_tokens 可为 int 或逐请求预算列表。
    此时返回的 GenerationResult 额外带有 new_token_ids（各请求输出）与 runs（各请求单独结果）。

    batching=FIXED 时运行成员固定的 Batch（左 Padding，已完成槽位不会被替换）；
    是否启用 KV Cache 严格由 FeatureFlag.KV_CACHE 特性决定（固定批处理不默认附带）。
    返回的 GenerationResult 带有性能与显存指标、new_token_ids，capture_logits=True 时带 logits_trace。
    """
    # 1. 规约输入：判断输入各请求的 Prompt 长度是否天然一致（Tensor 输入天然等长）
    has_uniform_prompt_len = isinstance(input_ids, torch.Tensor)
    if has_uniform_prompt_len:
        tensor = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
        token_sequences = list(tensor)
        if device is None:
            device = tensor.device
    else:
        token_sequences = _normalize_token_sequences(input_ids)
        if device is None:
            device = token_sequences[0].device if token_sequences[0].device.type != "cpu" else torch.device("cpu")

    num_requests = len(token_sequences)

    # 2. 仅当 batching=NONE 且请求数 > 1 时，按顺序逐个执行请求（串行 baseline）
    if mode.batching == BatchingMethod.NONE and num_requests > 1:
        if isinstance(max_new_tokens, int):
            budgets = [max_new_tokens] * num_requests
        else:
            budgets = list(max_new_tokens)
        if len(budgets) != num_requests:
            raise ValueError("输出预算数量必须与请求数量一致")

        runs = [
            generate(
                mode,
                model,
                seq.to(device=device, dtype=torch.long).unsqueeze(0),
                budget,
                eos_token_id=eos_token_id,
                device=device,
                stop_on_eos=stop_on_eos,
            )
            for seq, budget in zip(token_sequences, budgets)
        ]
        return GenerationResult._merge_generation_results(runs, pad_token_id)

    # 3. 统一生成内核
    # 若各 prompt 长度天然一致，无需执行复杂的 left padding
    if has_uniform_prompt_len:
        input_ids_tensor = (input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)).to(
            device=device, dtype=torch.long
        )
        batch_size, padded_prompt_length = input_ids_tensor.shape
        history_mask = torch.ones((batch_size, padded_prompt_length), dtype=torch.bool, device=device)
        position_ids = (
            torch.arange(padded_prompt_length, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        )
        prompt_lengths = torch.full((batch_size,), padded_prompt_length, dtype=torch.long, device=device)
    else:
        input_ids_tensor, history_mask, position_ids, prompt_lengths = left_pad_sequences(
            token_sequences, pad_token_id, device
        )
        batch_size, padded_prompt_length = input_ids_tensor.shape

    budgets = _normalize_budgets(max_new_tokens, batch_size, device)
    max_steps = int(budgets.max().item())

    # 是否启用 KV Cache 严格由 FeatureFlag.KV_CACHE 特性决定（固定批处理不默认附带）
    use_cache = mode.is_enabled(FeatureFlag.KV_CACHE)
    kv_cache = KVCache() if use_cache else None

    generated_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    curr_active = torch.ones(batch_size, dtype=torch.bool, device=device)
    active_counts_per_step = []
    outputs = [[] for _ in range(batch_size)]
    completion_steps = [None] * batch_size
    step_event_pairs = []
    step_times_ms = []
    logits_trace = [] if capture_logits else None
    memory_after_prefill_bytes = 0

    is_cuda = device.type == "cuda"
    synchronize(device)
    baseline_allocated_bytes = torch.cuda.memory_allocated(device) if is_cuda else 0
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()

    curr_input_ids = input_ids_tensor
    curr_position_ids = position_ids
    full_input_ids = input_ids_tensor if not use_cache else None
    full_position_ids = position_ids if not use_cache else None

    for step in range(max_steps):
        curr_active_before = curr_active.clone()
        step_out = _timed_forward(
            model=model,
            model_input=curr_input_ids if use_cache else full_input_ids,
            device=device,
            kv_cache=kv_cache,
            attention_mask=history_mask,
            position_ids=curr_position_ids if use_cache else full_position_ids,
            return_logits=capture_logits,
        )
        next_tokens = step_out.next_token.squeeze(1)
        if step_out.event_pair is not None:
            step_event_pairs.append(step_out.event_pair)
        else:
            step_times_ms.append(step_out.elapsed_ms)
        if step == 0 and is_cuda:
            memory_after_prefill_bytes = torch.cuda.memory_allocated(device)
        if capture_logits:
            logits_trace.append(step_out.last_logits.float().cpu())

        active_counts_per_step.append(int(curr_active_before.sum().item()))
        for req_idx in range(batch_size):
            if not bool(curr_active_before[req_idx].item()):
                continue
            token_id = int(next_tokens[req_idx].item())
            outputs[req_idx].append(token_id)
            generated_counts[req_idx] += 1
            reached_eos = eos_token_id is not None and token_id == eos_token_id
            reached_budget = bool((generated_counts[req_idx] >= budgets[req_idx]).item())
            if (stop_on_eos and reached_eos) or reached_budget:
                curr_active[req_idx] = False
                completion_steps[req_idx] = step + 1

        if not bool(curr_active.any().item()):
            break

        # 准备下一步输入：仅活跃槽位传入真实 Token，已完成槽位填 Padding 并 mask 掉
        step_token = torch.where(curr_active, next_tokens, torch.full_like(next_tokens, pad_token_id)).unsqueeze(1)
        history_mask = torch.cat((history_mask, curr_active.unsqueeze(1)), dim=1)
        next_positions = prompt_lengths + generated_counts - 1
        curr_pos = torch.where(curr_active, next_positions, torch.zeros_like(next_positions)).unsqueeze(1)

        if use_cache:
            curr_input_ids = step_token
            curr_position_ids = curr_pos
        else:
            full_input_ids = torch.cat((full_input_ids, step_token), dim=1)
            full_position_ids = torch.cat((full_position_ids, curr_pos), dim=1)

    synchronize(device)
    e2e_ms = (time.perf_counter() - wall_start) * 1000
    step_latencies_ms = _resolve_step_latencies(step_event_pairs, step_times_ms, is_cuda)

    step_count = len(step_latencies_ms)
    valid_generated_tokens_count = sum(len(output) for output in outputs)
    new_tokens = torch.full((batch_size, step_count), pad_token_id, dtype=torch.long)
    for req_idx, output in enumerate(outputs):
        new_tokens[req_idx, : len(output)] = torch.tensor(output, dtype=torch.long)

    cumulative_ms = []
    running_ms = 0.0
    for value in step_latencies_ms:
        running_ms += value
        cumulative_ms.append(running_ms)
    prompt_tokens = int(prompt_lengths.sum().item())
    padded_prompt_tokens = batch_size * padded_prompt_length
    capacity_tokens = batch_size * step_count
    ttft_ms = step_latencies_ms[0] if step_latencies_ms else 0.0

    extra = {
        "prompt_lengths_per_req": prompt_lengths.cpu().tolist(),
        "prompt_tokens": prompt_tokens,
        "padded_prompt_tokens": padded_prompt_tokens,
        "padding_efficiency": prompt_tokens / padded_prompt_tokens,
        "effective_prompt_token_throughput_per_second": prompt_tokens * 1000 / ttft_ms if ttft_ms else 0.0,
        "padded_prompt_token_throughput_per_second": padded_prompt_tokens * 1000 / ttft_ms if ttft_ms else 0.0,
        "request_throughput_per_second": batch_size * 1000 / e2e_ms if e2e_ms else 0.0,
        "active_counts_per_step": active_counts_per_step,
        "slot_utilization": valid_generated_tokens_count / capacity_tokens if capacity_tokens else 0.0,
        "completion_steps_per_req": completion_steps,
        "completion_times_ms_per_req": [cumulative_ms[s - 1] if s is not None else None for s in completion_steps],
        "cache_length": kv_cache.get_seq_length() if kv_cache is not None else 0,
    }

    return GenerationResult.build(
        # 1. 核心结果 (Core Outputs)
        new_tokens=new_tokens,
        new_token_ids=outputs,
        logits_trace=logits_trace,
        # 2. 核心标量指标源数据 (Core Scalar Metrics Sources)
        prompt_len=padded_prompt_length,
        output_len=step_count,
        batch_size=batch_size,
        e2e_ms=e2e_ms,
        step_latencies_ms=step_latencies_ms,
        baseline_allocated_bytes=baseline_allocated_bytes,
        memory_after_prefill_bytes=memory_after_prefill_bytes,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if is_cuda else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if is_cuda else 0,
        cache_bytes=cache_size_bytes(kv_cache) if kv_cache is not None else 0,
        # 3. 硬件环境与策略扩展 (Flags & Extra Metrics)
        is_cuda=is_cuda,
        valid_generated_tokens_count=valid_generated_tokens_count,
        extra=extra,
    )
