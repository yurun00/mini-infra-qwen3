"""两条单请求生成路径与计时工具。"""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

import torch

from mini_infra.qwen3_model import KVCache, cache_size_bytes


class Feature(str, Enum):
    """支持的所有 feature 枚举；每个成员对应一个可独立开关的特性。"""

    KV_CACHE = "kv-cache"

    @classmethod
    def from_str(cls, value: "str | Feature") -> "Feature":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"未知特性: {value}，可选特性: {[m.value for m in cls]}") from err

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Mode:
    """运行模式结构体：内部存储已启用的 Feature 集合。

    不同 mode 是不同特性的开关组合，但底层执行与输出结构（GenerationResult）保持一致。
    """

    features: frozenset[Feature] = frozenset()

    def __init__(self, features: "Iterable[Feature] | None" = None):
        if features is None:
            object.__setattr__(self, "features", frozenset())
        elif isinstance(features, frozenset):
            object.__setattr__(self, "features", features)
        else:
            object.__setattr__(self, "features", frozenset(features))

    def is_enabled(self, feature: Feature) -> bool:
        """检查指定的 feature enum 是否启用。"""
        return feature in self.features

    def with_feature(self, feature: Feature, enabled: bool = True) -> "Mode":
        """返回指定 feature enum 状态修改后的新 Mode 实例。"""
        if enabled:
            return Mode(self.features | {feature})
        return Mode(self.features - {feature})

    @property
    def kv_cache(self) -> bool:
        return self.is_enabled(Feature.KV_CACHE)

    def to_dict(self) -> dict[str, Any]:
        """将 mode 中的所有 feature 状态转换为字典。"""
        return {feat.value: self.is_enabled(feat) for feat in Feature}

    def __contains__(self, feature: Feature) -> bool:
        return self.is_enabled(feature)

    def __str__(self) -> str:
        if not self.features:
            return "no-cache"
        return "+".join(sorted(feat.value for feat in self.features))


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

    - 属性访问：.next_token, .elapsed_ms, .event_pair
    - 兼容元组解包：(next_token, elapsed_ms, event_pair)
    """

    next_token: torch.Tensor
    elapsed_ms: float = 0.0
    event_pair: tuple[Any, Any] | None = None


def _timed_forward(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    device: torch.device | None = None,
    kv_cache: KVCache | None = None,
) -> StepForwardResult:
    """单步前向推理并计时。"""
    if device is None:
        device = model_input.device

    event_pair = None
    elapsed_ms = 0.0

    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        t0 = time.perf_counter()

    logits = model(model_input, kv_cache=kv_cache)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    if device.type == "cuda":
        end.record()
        event_pair = (start, end)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000

    del logits
    return StepForwardResult(next_token=next_token, elapsed_ms=elapsed_ms, event_pair=event_pair)


class GenerationResult:
    """单次生成结果对象，结构对齐 run.py 组织方式。

    包含：
    - 输出 Token：new_tokens
    - 元数据：prompt_len, output_len, batch_size
    - 核心指标：ttft_ms, tpot_ms, e2e_ms, throughput_tok_s, gpu_time_ratio
    - 结构化显存：memory (包含 baseline_allocated_mb, prefill_allocated_mb,
      peak_allocated_mb, dynamic_peak_mb, peak_reserved_mb, fragmentation_mb, cache_bytes)
    """

    def __init__(
        self,
        new_tokens: torch.Tensor,
        metrics: dict[str, Any],
    ):
        self.new_tokens = new_tokens
        self.metrics = metrics

        # 顶层一等公民属性（对齐 run.py 组织结构）
        self.prompt_len: int = metrics.get("prompt_len", 0)
        self.output_len: int = metrics.get("output_len", 0)
        self.batch_size: int = metrics.get("batch_size", 1)
        self.ttft_ms: float = metrics.get("ttft_ms", 0.0)
        self.tpot_ms: float = metrics.get("tpot_ms", 0.0)
        self.e2e_ms: float = metrics.get("e2e_ms", 0.0)
        self.throughput_tok_s: float = metrics.get("throughput_tok_s", 0.0)
        self.gpu_time_ratio: float | None = metrics.get("gpu_time_ratio")
        self.step_latencies_ms: list[float] = metrics.get("step_latencies_ms", [])
        self.memory: dict[str, Any] = metrics.get("memory", {})
        self.cache_bytes: int = self.memory.get("cache_bytes", 0)

    def __getattr__(self, name: str) -> Any:
        if name in self.metrics:
            return self.metrics[name]
        if isinstance(self.memory, dict) and name in self.memory:
            return self.memory[name]
        if name == "cache_bytes":
            return self.cache_bytes
        raise AttributeError(f"'GenerationResult' object has no attribute '{name}'")

    def __getitem__(self, item: str) -> Any:
        if item in self.metrics:
            return self.metrics[item]
        if hasattr(self, item) and item not in ("get", "__getitem__", "__getattr__"):
            return getattr(self, item)
        if isinstance(self.memory, dict) and item in self.memory:
            return self.memory[item]
        if item == "cache_bytes":
            return self.cache_bytes
        raise KeyError(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default


def _build_generation_result(
    new_tokens: torch.Tensor,
    prompt_len: int,
    output_len: int,
    batch_size: int,
    step_latencies_ms: list[float],
    wall_ms: float,
    baseline_allocated_bytes: int,
    memory_after_prefill_bytes: int,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    cache_bytes: int,
    is_cuda: bool = False,
) -> GenerationResult:
    mib = 1024**2
    decode_latencies = step_latencies_ms[1:]
    ttft_ms = step_latencies_ms[0] if step_latencies_ms else 0.0
    tpot_ms = (sum(decode_latencies) / len(decode_latencies)) if decode_latencies else 0.0
    throughput = (batch_size * output_len) / (wall_ms / 1000) if wall_ms > 0 else 0.0
    # gpu_time_ratio 专属于 GPU（GPU 忙碌时间 / 墙钟时间）：非 CUDA 设备记为 None
    gpu_ratio = (min(1.0, sum(step_latencies_ms) / wall_ms) if wall_ms > 0 else 0.0) if is_cuda else None

    # 1. 显存结构化指标（对齐 run.py 中的 memory 分组）
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
        "cache_bytes": cache_bytes,
    }

    # 2. 单次生成指标字典（对齐 run.py result 的结构：Metadata / Latency / Throughput / Memory）
    metrics = {
        # 负载元数据
        "prompt_len": prompt_len,
        "output_len": output_len,
        "batch_size": batch_size,
        # 核心时延指标
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": wall_ms,
        "step_latencies_ms": step_latencies_ms,
        # 吞吐与利用率指标
        "throughput_tok_s": throughput,
        "gpu_time_ratio": gpu_ratio,
        # 显存指标（统一收敛于 memory 中，细粒度拆解）
        "memory": memory,
    }

    return GenerationResult(
        new_tokens=new_tokens,
        metrics=metrics,
    )


@torch.inference_mode()
def generate(
    mode: Mode,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    device: torch.device | None = None,
    stop_on_eos: bool = True,
) -> GenerationResult:
    """统一单请求生成入口；mode 为 Mode 结构体对象。

    不同 mode 对应不同的 feature 开关，但生成的 output token 完全一致。
    """
    use_cache = mode.is_enabled(Feature.KV_CACHE)
    if device is None:
        device = input_ids.device

    batch_size = input_ids.shape[0]

    # 1. 运行前常驻显存基线与重置峰值
    synchronize(device)
    baseline_allocated_bytes = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 2. 墙钟开始时间
    wall_start = time.perf_counter()

    cache = KVCache() if use_cache else None
    model_input = input_ids
    generated = input_ids
    step_event_pairs = []
    step_times_ms = []
    memory_after_prefill_bytes = 0

    for step in range(max_new_tokens):
        step_out = _timed_forward(
            model=model,
            model_input=model_input,
            device=device,
            kv_cache=cache,
        )
        next_token = step_out.next_token

        if step_out.event_pair is not None:
            step_event_pairs.append(step_out.event_pair)
        else:
            step_times_ms.append(step_out.elapsed_ms)

        generated = torch.cat((generated, next_token), dim=1)
        # 有 Cache 时下一步只喂最新 Token；无 Cache 时每步重算完整上下文
        model_input = next_token if use_cache else generated

        if step == 0 and device.type == "cuda":
            memory_after_prefill_bytes = torch.cuda.memory_allocated(device)

        if stop_on_eos and eos_token_id is not None and (next_token == eos_token_id).all():
            break

    # 3. 墙钟结束时间
    synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000

    is_cuda = device.type == "cuda"

    # 4. 统计各 step 推理时延（CUDA 下使用 Event 硬件测时，非 CUDA 下使用 CPU 墙钟）
    if is_cuda and step_event_pairs:
        step_latencies_ms = [s.elapsed_time(e) for s, e in step_event_pairs]
    else:
        step_latencies_ms = step_times_ms

    # 5. 显存峰值
    peak_allocated_bytes = torch.cuda.max_memory_allocated(device) if is_cuda else 0
    peak_reserved_bytes = torch.cuda.max_memory_reserved(device) if is_cuda else 0

    prompt_len = input_ids.shape[1]
    new_tokens = generated[:, prompt_len:]
    output_len = new_tokens.shape[1]

    return _build_generation_result(
        new_tokens=new_tokens,
        prompt_len=prompt_len,
        output_len=output_len,
        batch_size=batch_size,
        step_latencies_ms=step_latencies_ms,
        wall_ms=wall_ms,
        baseline_allocated_bytes=baseline_allocated_bytes,
        memory_after_prefill_bytes=memory_after_prefill_bytes,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        cache_bytes=cache_size_bytes(cache) if cache is not None else 0,
        is_cuda=is_cuda,
    )
