"""Benchmark Suite 配置注册表：描述“跑什么套件”，由 run_suite.py 负责“怎么跑、怎么落盘”。

每个 suite 是一份不可变配置，包含受控实验所需的完整参数：
    1. 基础配置（batch_sizes、prompt_length、max_new_tokens、warmup、repeats、运行模式……）；
    2. 这些参数展开成哪些具体的执行用例（SuiteCase 列表）。

run_suite.py 不关心 suite 的内部细节，只按 `suite.cases()` 逐个执行并汇总结果：

    mini-infra bench lockstep suite --tag dev --suite serial-vs-fixed-batch-throughput

新增 suite：定义一个提供 name / warmup / repeats / modes() / cases() / to_dict() 的配置类，
再注册到 SUITES 即可。

当前已实现：
    serial-vs-fixed-batch-throughput  对每个 batch size，分别用 serial（逐请求串行）与
                                     fixed-batch（固定批处理）两种路径，在固定长度 prompt、
                                     固定输出长度、忽略 EOS 的条件下测量吞吐与显存。
    uniform-vs-variable-prompt-padding 固定 Batch Size 4 与输出长度，对比等长 prompt
                                     与阶梯不等长 prompt（uniform vs variable），
                                     测量 Padding 有效率、Prefill 耗时与有效 Prompt 吞吐。
    uniform-vs-variable-budget-slots 固定 Batch Size 4 与 Prompt 输入长度，对比等长预算
                                     与阶梯不等长预算（uniform vs variable），
                                     测量槽位利用率、Decode 耗时与端到端有效吞吐。
"""

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from mini_infra.bench.workloads import (
    make_variable_output_budgets,
    make_variable_prompt_lengths,
)
from mini_infra.engine import (
    BatchingMethod,
    FeatureFlag,
    Mode,
)


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


@dataclass(frozen=True)
class UniformVsVariablePromptPaddingSuite:
    """固定批处理下等长 Prompt（Uniform）与变长 Prompt（Variable）的 Padding 对照实验套件。

    固定 batch_size=4 与输出长度，对比两种 Prompt 长度分布：
    1. uniform: 全 Batch 等长（例如 (128, 128, 128, 128)），Padding 有效率 100%；
    2. variable: 阶梯稀疏不等长（例如 (16, 32, 64, 128)），Padding 有效率 46.88%。

    通过对比两者的 Prefill 耗时、峰值显存与有效 Prompt 吞吐，评估 Padding 带来的无效计算与显存开销。

    Attributes:
        batch_size: 批大小，固定为 4（对应 1/8, 1/4, 1/2, 1 阶梯）。
        max_prompt_length: 基准最长 Prompt Token 数，默认 128。
        max_new_tokens: 生成 Token 数，默认 32。
        kv_cache: 是否启用 KV Cache（默认 True）。
        batching_method: 批处理策略（默认 FIXED）。
        warmup / repeats: 预热与正式测量次数（默认 1 / 3）。
    """

    name: str = "uniform-vs-variable-prompt-padding"
    batch_size: int = 4
    max_prompt_length: int = 128
    max_new_tokens: int = 32
    kv_cache: bool = True
    batching_method: BatchingMethod = BatchingMethod.FIXED
    warmup: int = 1
    repeats: int = 3

    def __post_init__(self) -> None:
        if self.batch_size != 4:
            raise ValueError(
                "uniform-vs-variable-prompt-padding 套件要求 batch_size 为 4（对应 1/8, 1/4, 1/2, 1 阶梯）"
            )
        if self.max_prompt_length < 1 or self.repeats < 1 or self.warmup < 0:
            raise ValueError("max_prompt_length、repeats 必须为正，warmup 不能小于 0")
        if self.max_new_tokens < 2:
            raise ValueError("max_new_tokens 至少为 2，才能测量 Decode")
        if self.batching_method != BatchingMethod.FIXED:
            raise ValueError("uniform-vs-variable-prompt-padding 套件仅支持 fixed-batching")

    def modes(self) -> list[Mode]:
        flags = {FeatureFlag.KV_CACHE} if self.kv_cache else set()
        return [Mode(flags, batching_method=self.batching_method)]

    def cases(self) -> list[SuiteCase]:
        mode = self.modes()[0]
        homo_lengths = tuple([self.max_prompt_length] * self.batch_size)
        variable_lengths = tuple(make_variable_prompt_lengths(self.max_prompt_length))
        return [
            SuiteCase(mode, self.batch_size, homo_lengths, self.max_new_tokens),
            SuiteCase(mode, self.batch_size, variable_lengths, self.max_new_tokens),
        ]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class UniformVsVariableBudgetSlotsSuite:
    """固定批处理下等长预算（Uniform Budget）与变长预算（Variable Length Budget）的槽位利用率对照实验套件。

    固定 batch_size=4 与 Prompt 输入长度，对比两种生成 Token 预算（Budget）分布：
    1. uniform: 全 Batch 请求生成预算相同（例如 (64, 64, 64, 64)），槽位利用率 100%；
    2. variable: 阶梯稀疏预算分布（例如 (8, 16, 32, 64)），槽位利用率 46.88%。

    通过对比两者的 Decode 耗时、端到端吞吐与槽位利用率，评估短请求提早结束后空占 Batch 槽位所带来的算力浪费。

    Attributes:
        batch_size: 批大小，固定为 4（对应 1/8, 1/4, 1/2, 1 预算阶梯）。
        prompt_length: 每条 Prompt 的 Token 数（等长，默认 128）。
        max_new_tokens: 最大生成 Token 数（默认 64）。
        kv_cache: 是否启用 KV Cache（默认 True）。
        batching_method: 批处理策略（默认 FIXED）。
        warmup / repeats: 预热与正式测量次数（默认 1 / 3）。
    """

    name: str = "uniform-vs-variable-budget-slots"
    batch_size: int = 4
    prompt_length: int = 128
    max_new_tokens: int = 64
    kv_cache: bool = True
    batching_method: BatchingMethod = BatchingMethod.FIXED
    warmup: int = 1
    repeats: int = 3

    def __post_init__(self) -> None:
        if self.batch_size != 4:
            raise ValueError("uniform-vs-variable-budget-slots 套件要求 batch_size 为 4（对应 1/8, 1/4, 1/2, 1 阶梯）")
        if self.prompt_length < 1 or self.repeats < 1 or self.warmup < 0:
            raise ValueError("prompt_length、repeats 必须为正，warmup 不能小于 0")
        if self.max_new_tokens < 2:
            raise ValueError("max_new_tokens 至少为 2，才能测量 Decode")
        if self.batching_method != BatchingMethod.FIXED:
            raise ValueError("uniform-vs-variable-budget-slots 套件仅支持 fixed-batching")

    def modes(self) -> list[Mode]:
        flags = {FeatureFlag.KV_CACHE} if self.kv_cache else set()
        return [Mode(flags, batching_method=self.batching_method)]

    def cases(self) -> list[SuiteCase]:
        mode = self.modes()[0]
        prompt_lens = tuple([self.prompt_length] * self.batch_size)
        uniform_budgets = tuple([self.max_new_tokens] * self.batch_size)
        variable_budgets = tuple(make_variable_output_budgets(self.max_new_tokens))
        return [
            SuiteCase(mode, self.batch_size, prompt_lens, uniform_budgets),
            SuiteCase(mode, self.batch_size, prompt_lens, variable_budgets),
        ]

    def to_dict(self) -> dict:
        return asdict(self)


SUITES: dict[str, Suite] = {
    "serial-vs-fixed-batch-throughput": SerialVsFixedBatchThroughputSuite(),
    "uniform-vs-variable-prompt-padding": UniformVsVariablePromptPaddingSuite(),
    "uniform-vs-variable-budget-slots": UniformVsVariableBudgetSlotsSuite(),
}


def get_suite(name: str) -> Suite:
    try:
        return SUITES[name]
    except KeyError as err:
        raise ValueError(f"未知 suite: {name}，可选: {sorted(SUITES)}") from err
