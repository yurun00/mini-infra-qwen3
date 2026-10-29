"""CLI 子命令：mini-infra bench check-regression

性能回归与质量门禁检查入口。
"""

import argparse
import json
from pathlib import Path

from mini_infra.bench.check_regression import (
    DEFAULT_CANDIDATE,
    DEFAULT_REFERENCE,
    compare,
    run_light_benchmark,
)


def add_regression_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="参考 baseline 结果 JSON 路径")
    p.add_argument("--candidate", type=Path, help="已有候选结果；缺省则现场运行轻量 benchmark")
    p.add_argument("--threshold", type=float, default=0.05, help="中位数劣化阈值，默认 5%%")
    p.add_argument("--update-reference", action="store_true", help="运行并覆盖参考结果")
    p.add_argument("--strict-hardware", action="store_true", help="GPU 型号不一致时失败而非跳过")
    return p


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 check-regression。"""
    p = subparsers.add_parser(
        "check-regression",
        help="运行轻量 benchmark 检查性能回归与质量门禁（退出码: 0通过/跳过, 1回归, 2错误）",
    )
    add_regression_args(p)
    p.set_defaults(func=handle_regression)
    return p


def handle_regression(args: argparse.Namespace) -> int:
    if args.update_reference:
        code = run_light_benchmark(args.reference, "reference")
        if code:
            return 2
        print(f"参考结果已更新：{args.reference}")
        return 0

    if not args.reference.exists():
        print(f"未找到参考结果 {args.reference}，先运行 --update-reference，本次跳过")
        return 0

    candidate_path = args.candidate
    if candidate_path is None:
        candidate_path = DEFAULT_CANDIDATE
        if run_light_benchmark(candidate_path, "ci-check"):
            print("benchmark 运行失败或正确性检查未通过")
            return 2

    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))

    if candidate["env"]["gpu"] != reference["env"]["gpu"]:
        msg = f"GPU 不一致（参考 {reference['env']['gpu']} / 当前 {candidate['env']['gpu']}），性能不可比"
        if args.strict_hardware:
            print("失败：" + msg)
            return 1
        print("跳过：" + msg)
        return 0

    regressed, lines = compare(candidate, reference, args.threshold)
    print(f"\n参考版本 {reference['git']['commit']} → 当前 {candidate['git']['commit']}，阈值 {args.threshold:.0%}")
    print("\n".join(lines))
    if regressed:
        print(f"\n性能回归：存在中位数劣化超过 {args.threshold:.0%} 的指标")
        return 1
    print("\n性能检查通过")
    return 0
