"""PyTorch CUDA 内存快照（Memory Snapshot）追踪器与生命周期分阶段深度诊断教程。

================================================================================
一、设计目的与背景
================================================================================
在大模型推理系统（如 Paged KV Cache）中，显存指标往往呈现“常驻池显存”、“增量活跃 KV Cache”
与“单步前向激活值（Activation）”多重交织的现象。常见疑难问题包括：
1. 为什么使用了 Paged KV Cache 消除碎片后，全局峰值显存（peak allocated memory）反而变高了？
2. Prefill 阶段与常驻物理块池是否存在时空叠加？
3. 请求退场后是否存在物理块泄漏或未释放的中间张量？

PyTorch Memory Snapshot 是定位 GPU 显存物理分配与生命周期最精确的武器。本模块提供针对
推理引擎调度器关键生命周期节点的打点与快照导出工具，记录每一个活着的张量（形状、大小、生命周期）
及其分配处的完整 Python 代码调用栈。

================================================================================
二、核心生命周期阶段点位（6 大关键节点）
================================================================================
在 PagedScheduleBackend 等调度后端中，内置了以下细粒度阶段快照：
1. `00_paged_before_pool_init`：模型加载完毕，尚未分配物理块池（基准模型权重显存）；
2. `01_paged_pool_allocated`：PagedKVCachePool 物理块池分配完成（记录常驻显存定格点）；
3. `02_paged_prefill_forward_done`：Prefill 前向计算结束，prefill_cache（稠密 KV 张量）生成，
   单步前向激活值（Attention Logits、MLP 中间层）达到瞬时全局最高峰；
4. `03_paged_prefill_stored_dual_kv`：物理块表散写（store_prefill）完成。此时 self.pool 与
   prefill_cache 均存活，处于双份 KV Cache 在显存中并存的关键时刻；
5. `04_paged_prefill_cleaned`：显式执行 del prefill_out 后，临时稠密 KV 显存被回收，显存回落；
6. `05_paged_decode_step`：单步增量 Decode（观察 Gather 拷贝与块表扩展）；
7. `snapshot_full.pickle`：全生命周期连续时间线快照。

================================================================================
三、使用方式
================================================================================
【方式 1：环境变量零侵入激活（推荐排查基准测试）】
无需修改任何调用代码，直接在任意 CLI 命令前声明 MINI_INFRA_SNAPSHOT_DIR：
    $ MINI_INFRA_SNAPSHOT_DIR=./paged_snapshots \\
        mini-infra bench scheduler case \\
        --tag debug \\
        --caching-method paged-kv-cache \\
        --batching-method continuous-batching \\
        --scale large \\
        --repeats 1 \\
        --warmup 0

运行后控制台将输出各阶段实时显存，并在指定目录下落盘一系列分阶段快照及 snapshot_full.pickle。

【方式 2：Python 脚本显式上下文管理器调用】
    from mini_infra.profiling import MemorySnapshotTracker

    with MemorySnapshotTracker("./my_snapshots") as tracker:
        # 执行需要排查的模型推理或调度循环
        tracker.record_stage("custom_stage_1")
        ...

【零性能损耗原则】：
未设置 MINI_INFRA_SNAPSHOT_DIR 或未显式启用时，self.enabled 恒为 False，仅执行单个
布尔判断后立即返回，生产推理与基准评测开销为 0。

================================================================================
四、可视化分析诊断指南（memory_viz）
================================================================================
1. 打开浏览器访问官方可视化分析器：https://pytorch.org/memory_viz
2. 将导出的 snapshot_full.pickle 或某阶段独立的 .pickle 文件拖拽至网页中；
3. 关键排查技巧：
   - 查看时间线（Timeline）：横轴为时间，纵轴为物理显存占用（MiB）；
   - 点击最高柱子（Peak）：左侧将展开该瞬时所有存活的张量块；
     * 查看是否存在 PagedKVCachePool 张量（如 1744.75 MiB）；
     * 查看是否存在未释放的 prefill_cache 稠密张量（如 1526.00 MiB）；
     * 查看是否存在高达数 GB 的 Attention 矩阵与 MLP 激活值；
   - 点击任意张量，右侧即可查看分配该张量的精确代码文件与行号调用栈。
"""

import os
from pathlib import Path
from types import TracebackType

import torch


class MemorySnapshotTracker:
    """分阶段 PyTorch CUDA 显存快照记录器。"""

    def __init__(self, output_dir: str | Path | None = None, enabled: bool | None = None):
        env_dir = os.environ.get("MINI_INFRA_SNAPSHOT_DIR")
        target_dir = output_dir or env_dir
        self.output_dir = Path(target_dir).expanduser().resolve() if target_dir else None

        if enabled is not None:
            self.enabled = enabled and torch.cuda.is_available()
        else:
            self.enabled = bool(self.output_dir) and torch.cuda.is_available()

        self._step_counter = 0
        self._stage_counts: dict[str, int] = {}
        self._is_recording = False

    def start(self, max_entries: int = 100000) -> None:
        """开始捕获 CUDA 内存分配历史（含 Python 调用栈）。"""
        if not self.enabled or self._is_recording:
            return
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
        self._is_recording = True
        self._step_counter = 0
        self._stage_counts.clear()

    def record_stage(
        self,
        stage_name: str,
        dump: bool = True,
        sampling_rate: float | None = None,
    ) -> tuple[float, float]:
        """记录指定阶段的瞬时显存，并可选导出该阶段独立的内存快照 pickle。

        Args:
            stage_name: 阶段语义标识（如 '01_paged_pool_allocated', '05_paged_decode_step'）。
            dump: 是否立即导出当前时间点的快照 pickle 文件。
            sampling_rate: 同一阶段重复调用时的采样率。
                - 为 None 时，每次调用均执行记录。
                - 为 float（0.0 < sampling_rate < 1.0）时，按比例（1 / sampling_rate）进行间隔采样。
                - 为 int / float (>= 1.0) 时，表示每隔 N 次调用采样一次（如 10 表示每 10 次记录一次）。
                - <= 0 时不进行记录。

        Returns:
            (allocated_mb, peak_mb): 当前已分配显存与历史峰值已分配显存（MiB）。未触发采样时返回 (0.0, 0.0)。
        """
        if not self.enabled:
            return 0.0, 0.0

        stage_count = self._stage_counts.get(stage_name, 0)
        self._stage_counts[stage_name] = stage_count + 1

        if sampling_rate is not None:
            if sampling_rate <= 0:
                return 0.0, 0.0
            interval = max(1, round(1.0 / sampling_rate)) if sampling_rate < 1.0 else max(1, int(sampling_rate))
            if stage_count % interval != 0:
                return 0.0, 0.0

        torch.cuda.synchronize()
        alloc_mb = torch.cuda.memory_allocated() / (1024**2)
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        msg = (
            f"[Memory Snapshot] [{self._step_counter:03d}_{stage_name}] "
            f"已分配: {alloc_mb:.2f} MiB | 峰值: {peak_mb:.2f} MiB"
        )
        print(msg)

        if dump and self.output_dir:
            snapshot_path = self.output_dir / f"snapshot_{self._step_counter:03d}_{stage_name}.pickle"
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
        self._step_counter += 1
        return alloc_mb, peak_mb

    def stop(self, final_dump_name: str = "snapshot_full.pickle") -> None:
        """停止捕获显存历史，并可选导出包含全生命周期的完整快照。"""
        if not self.enabled or not self._is_recording:
            return
        if self.output_dir and final_dump_name:
            full_path = self.output_dir / final_dump_name
            torch.cuda.memory._dump_snapshot(str(full_path))
            print(f"[Memory Snapshot] 全生命周期快照已导出至: {full_path}")
        torch.cuda.memory._record_memory_history(enabled=None)
        self._is_recording = False

    def __enter__(self) -> "MemorySnapshotTracker":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()


__all__ = ["MemorySnapshotTracker"]
