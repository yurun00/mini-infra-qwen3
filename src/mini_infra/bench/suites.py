"""Benchmark Suite 配置注册表：描述“跑什么套件”，由 run_suite.py 负责“怎么跑、怎么落盘”。

每个 suite 是一份不可变配置，包含受控实验所需的完整参数：
    1. 基础配置（batch_sizes、prompt_length、max_new_tokens、warmup、repeats、运行模式……）；
    2. 这些参数展开成哪些具体的执行用例（SuiteCase 列表）。

run_suite.py 不关心 suite 的内部细节，只按 `suite.cases()` 逐个执行并汇总结果：

    mini-infra-bench-suite --tag dev --suite serial-vs-fixed-batch-throughput

新增 suite：定义一个提供 name / warmup / repeats / modes() / cases() / to_dict() 的配置类，
再注册到 SUITES 即可。

当前已实现：
    serial-vs-fixed-batch-throughput  对每个 batch size，分别用 serial（逐请求串行）与
                                     fixed-batch（固定批处理）两种路径，在固定长度 prompt、
                                     固定输出长度、忽略 EOS 的条件下测量吞吐与显存。
"""

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from mini_infra.generation import BatchingMethod, FeatureFlag, Mode


@dataclass(frozen=True)
class SuiteCase:
    """一次具体执行用例：在 mode 下，用 batch_size 条 prompt 生成 max_new_tokens 个 Token。"""

    mode: Mode
    batch_size: int
    prompt_len: int | tuple[int, ...]
    max_new_tokens: int | tuple[int, ...]


@runtime_checkable
class Suite(Protocol):
    """run_suite.py 依赖的最小 suite 接口规范。"""

    name: str
    warmup: int
    repeats: int

    def modes(self) -> list[Mode]: ...

    def cases(self) -> list[SuiteCase]: ...

    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class SerialVsFixedBatchThroughputSuite:
    """串行（Serial）与固定批处理（Fixed Batching）吞吐对比实验套件。

    对每个 batch size，分别用 serial（逐请求串行）与 fixed-batch（固定批处理）
    两种执行路径，在固定长度 prompt、固定输出长度、忽略 EOS 的条件下测量吞吐与显存。

    Attributes:
        batch_sizes: 要扫描的 batch size 列表。
        prompt_length: 每条 prompt 的 Token 数（所有请求等长）。
        max_new_tokens: 每条请求生成的 Token 数（忽略 EOS，必须 >= 2 才能测量 Decode）。
        warmup / repeats: 每个用例的预热与正式测量次数。
        batching_methods: 要对比的执行路径；NONE 对应 serial，FIXED 对应 fixed-batching。
        kv_cache: 是否启用 KV Cache（两条路径均默认启用 KV Cache）。
    """

    name: str = "serial-vs-fixed-batch-throughput"
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)
    prompt_length: int = 128
    max_new_tokens: int = 32
    batching_methods: tuple[BatchingMethod, ...] = (BatchingMethod.NONE, BatchingMethod.FIXED)
    kv_cache: bool = True
    warmup: int = 1
    repeats: int = 3

    def __post_init__(self) -> None:
        if not self.batch_sizes or any(b < 1 for b in self.batch_sizes):
            raise ValueError("batch_sizes 必须非空且每项为正")
        if self.prompt_length < 1 or self.repeats < 1 or self.warmup < 0:
            raise ValueError("prompt_length、repeats 必须为正，warmup 不能小于 0")
        if self.max_new_tokens < 2:
            raise ValueError("max_new_tokens 至少为 2，才能测量 Decode")
        if not self.batching_methods:
            raise ValueError("batching_methods 不能为空")

    def modes(self) -> list[Mode]:
        flags = {FeatureFlag.KV_CACHE} if self.kv_cache else set()
        return [Mode(flags, batching_method=bm) for bm in self.batching_methods]

    def cases(self) -> list[SuiteCase]:
        # 嵌套顺序：外层 batch size，内层执行路径。
        return [
            SuiteCase(mode, batch_size, self.prompt_length, self.max_new_tokens)
            for batch_size in self.batch_sizes
            for mode in self.modes()
        ]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["batching_methods"] = [str(m) for m in self.batching_methods]
        return data


SUITES: dict[str, Suite] = {
    "serial-vs-fixed-batch-throughput": SerialVsFixedBatchThroughputSuite(),
}


def get_suite(name: str) -> Suite:
    try:
        return SUITES[name]
    except KeyError as err:
        raise ValueError(f"未知 suite: {name}，可选: {sorted(SUITES)}") from err
