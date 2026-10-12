"""性能回归检查：运行轻量版 benchmark，与参考结果比较，中位数劣化超阈值则失败。

定位与数据流说明：
    本模块是纯粹的【质量门禁与防回退守门人（Quality Gate）】，只输出退出码（0/1/2）与控制台文本。
    其默认生成的 candidate.local.json 为临时比对文件（已被 .gitignore 忽略），不会写入
    benchmarks/result-history/，因此【不参与】mini-infra-bench-plot 的绘图与时间线分析。
    绘图所需的全量数据源唯一定义来自 mini-infra-bench（run.py）。

用法：
    # 首次/有意接受新性能后，刷新参考结果（需提交 benchmarks/data/reference.json）
    mini-infra-bench-regression --update-reference

    # CI / pre-push：运行轻量 benchmark 并对比参考结果，劣化 >5% 返回非零退出码
    mini-infra-bench-regression

    # 只比较两份已有结果，不重新运行
    mini-infra-bench-regression --candidate a.json --reference b.json

退出码：0 通过 / 跳过，1 性能回归，2 正确性失败或运行错误。
参考结果与硬件绑定：GPU 型号不一致时默认跳过（--strict-hardware 则视为失败）。
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(os.environ.get("MINI_INFRA_BENCH_DIR", "benchmarks")).resolve()
DEFAULT_REFERENCE = BENCH_DIR / "data" / "reference.json"
# *.local.json 已被 .gitignore 忽略
DEFAULT_CANDIDATE = BENCH_DIR / "candidate.local.json"

# 轻量 workload：一个短、一个中等长度，约半分钟内完成
LIGHT_ARGS = ["--prompt-lens", "32", "512", "--output-lens", "32",
              "--warmup", "2", "--repeats", "7"]
METRICS = [("ttft_ms", "TTFT"), ("tpot_ms", "TPOT")]


def run_light_benchmark(output, tag):
    cmd = [sys.executable, "-m", "mini_infra.bench.run", "--tag", tag,
           "--output", str(output), *LIGHT_ARGS]
    print("运行轻量 benchmark:", " ".join(cmd[1:]), flush=True)
    return subprocess.run(cmd, cwd=BENCH_DIR.parent).returncode


def index(record):
    return {(w["batch_size"], w["prompt_len"], w["output_len"]): w
            for w in record["workloads"]}


def compare(candidate, reference, threshold):
    """返回 (是否回归, 报告行)。延迟类指标：越大越差。"""

    cand, ref = index(candidate), index(reference)
    common = sorted(set(cand) & set(ref))
    if not common:
        return None, ["没有可比较的共同 workload"]

    regressed, lines = False, []
    lines.append(f"{'workload (bs/prompt/out)':<26}{'指标':<6}{'参考(ms)':>10}"
                 f"{'当前(ms)':>10}{'变化':>9}  结果")
    for key in common:
        for metric, name in METRICS:
            r, c = ref[key][metric]["median"], cand[key][metric]["median"]
            change = (c - r) / r if r else 0.0
            # 噪声保护：变化需同时超过阈值和参考 std 的 3 倍，避免噪声误报
            noise = 3 * max(ref[key][metric]["std"], cand[key][metric]["std"])
            bad = change > threshold and (c - r) > noise
            regressed |= bad
            status = "❌ 回归" if bad else ("✅ 提升" if change < -threshold else "ok")
            lines.append(f"{'/'.join(map(str, key)):<26}{name:<6}{r:>10.2f}{c:>10.2f}"
                         f"{change:>+9.1%}  {status}")
    return regressed, lines


def main():
    p = argparse.ArgumentParser(description="性能回归检查")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--candidate", type=Path, help="已有结果；缺省则现场运行轻量 benchmark")
    p.add_argument("--threshold", type=float, default=0.05, help="中位数劣化阈值，默认 5%%")
    p.add_argument("--update-reference", action="store_true", help="运行并覆盖参考结果")
    p.add_argument("--strict-hardware", action="store_true", help="GPU 型号不一致时失败而非跳过")
    args = p.parse_args()

    if args.update_reference:
        code = run_light_benchmark(args.reference, "reference")
        if code:
            sys.exit(2)
        print(f"参考结果已更新：{args.reference}")
        return

    if not args.reference.exists():
        print(f"未找到参考结果 {args.reference}，先运行 --update-reference，本次跳过")
        return

    candidate_path = args.candidate
    if candidate_path is None:
        candidate_path = DEFAULT_CANDIDATE
        if run_light_benchmark(candidate_path, "ci-check"):
            print("benchmark 运行失败或正确性检查未通过")
            sys.exit(2)

    candidate = json.loads(Path(candidate_path).read_text())
    reference = json.loads(args.reference.read_text())

    if candidate["env"]["gpu"] != reference["env"]["gpu"]:
        msg = (f"GPU 不一致（参考 {reference['env']['gpu']} / 当前 {candidate['env']['gpu']}），"
               "性能不可比")
        if args.strict_hardware:
            print("失败：" + msg)
            sys.exit(1)
        print("跳过：" + msg)
        return

    regressed, lines = compare(candidate, reference, args.threshold)
    print(f"\n参考版本 {reference['git']['commit']} → 当前 {candidate['git']['commit']}，"
          f"阈值 {args.threshold:.0%}")
    print("\n".join(lines))
    if regressed:
        print(f"\n性能回归：存在中位数劣化超过 {args.threshold:.0%} 的指标")
        sys.exit(1)
    print("\n性能检查通过")


if __name__ == "__main__":
    main()
