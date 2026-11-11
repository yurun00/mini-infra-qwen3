"""评测 CLI 子命令共享的基础参数定义。"""

import argparse
from pathlib import Path

from mini_infra.bench.cases import DEFAULT_REPEATS, DEFAULT_WARMUP, ProfilingMode
from mini_infra.model import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION


def add_common_args(p: argparse.ArgumentParser) -> None:
    """向 ArgumentParser 添加单用例与套件运行共享的基础参数。"""
    p.add_argument("--tag", required=True, help="版本标签，如 baseline、v0.1.0、v0.2.0")
    p.add_argument(
        "--release-feature",
        default=None,
        help="当前版本对应的核心特性名称（如 'no cache'、'KV cache'、'FlashAttention'，可选）",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help=f"预热次数（默认: {DEFAULT_WARMUP}）")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help=f"正式测量重复次数（默认: {DEFAULT_REPEATS}）")
    p.add_argument("--allow-mismatch", action="store_true", help="正确性不通过时仍记录结果（会标记）")
    p.add_argument("--peak-bandwidth-gbs", type=float, help="覆盖 GPU 峰值显存带宽（GB/s），用于 MBU")
    p.add_argument("--peak-tflops", type=float, help="覆盖 GPU 峰值 FP16/BF16 算力（TFLOPS），用于 MFU")
    p.add_argument(
        "--profiling-mode",
        type=ProfilingMode,
        choices=list(ProfilingMode),
        default=None,
        help="Profiler 性能剖析模式: dedicated-pass 或 all-repeats",
    )
    p.add_argument("--tensorboard-dir", type=Path, default=None, help="导出 TensorBoard 格式性能剖析日志的目录（可选）")
    p.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        default=None,
        help="工作负载规模标签（small / medium / large，可选；用于同版本多规模归档）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="伪随机工作负载种子记录标签（可选）",
    )
    p.add_argument("--output", type=Path, help="自定义输出路径")


__all__ = ["add_common_args"]
