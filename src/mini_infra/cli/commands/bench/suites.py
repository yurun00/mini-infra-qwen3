"""CLI 子命令：mini-infra bench suites (list / show / schema)

浏览与检视声明式 suite（benchmarks/specs/<engine>/*.yaml），无需加载模型、无需 GPU。
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from mini_infra.bench.cases import CaseKey
from mini_infra.bench.common import format_lengths
from mini_infra.bench.spec import (
    CaseSpec,
    LoadedSuiteSpec,
    SuiteEngine,
    discover_suite_specs,
    load_suite_spec,
    release_json_schema,
    suite_json_schema,
)

_ENGINES: tuple[SuiteEngine, ...] = ("lockstep", "scheduler")


def register_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """在 bench 下注册 suites 组及 list/show/schema 子命令。"""
    suites_parser = subparsers.add_parser("suites", help="浏览与检视声明式 suite（YAML）")
    subs = suites_parser.add_subparsers(dest="suites_subcommand", required=True)

    list_parser = subs.add_parser("list", help="列出内置 suite")
    list_parser.add_argument("--engine", choices=_ENGINES, default=None, help="仅列出指定引擎的 suite")
    list_parser.set_defaults(func=handle_list)

    show_parser = subs.add_parser("show", help="打印 suite 合并后的 YAML 与展开后的用例列表")
    show_parser.add_argument("target", help="内置 suite 名称，或 YAML 文件路径")
    show_parser.add_argument("--engine", choices=_ENGINES, default=None, help="名称重名时指定引擎")
    show_parser.set_defaults(func=handle_show)

    schema_parser = subs.add_parser("schema", help="导出 suite / release YAML 的 JSON Schema（供 IDE 补全）")
    schema_parser.add_argument(
        "--kind", choices=["suite", "release"], default="suite", help="导出哪类声明的 Schema（默认: suite）"
    )
    schema_parser.add_argument("--output", type=Path, default=None, help="写入文件（默认打印到标准输出）")
    schema_parser.set_defaults(func=handle_schema)
    return suites_parser


def _describe_case(engine: SuiteEngine, case_spec: CaseSpec) -> str:
    if engine == "scheduler":
        assert case_spec.max_running_requests is not None
        return str(
            CaseKey.from_mode(
                case_spec.workload.workload_id,
                case_spec.mode,
                case_spec.max_running_requests,
                case_spec.paged_cache_config,
            )
        )
    return (
        f"[{case_spec.workload.workload_id}] | {case_spec.mode} | batch={case_spec.workload.num_total_requests} "
        f"prompt={format_lengths(case_spec.workload.prompt_lens)} "
        f"max_new_tokens={format_lengths(case_spec.workload.output_budgets)}"
    )


def _resolve_target(target: str, engine: SuiteEngine | None) -> LoadedSuiteSpec:
    path = Path(target)
    if path.suffix in (".yaml", ".yml") or path.exists():
        return load_suite_spec(path)
    matches = [
        path
        for eng in ([engine] if engine else _ENGINES)
        for name, path in discover_suite_specs(eng).items()
        if name == target
    ]
    if not matches:
        raise ValueError(f"未知 suite: {target}（可用 `mini-infra bench suites list` 查看）")
    if len(matches) > 1:
        raise ValueError(f"suite 名称 {target} 在多个引擎下重名，请用 --engine 指定")
    return load_suite_spec(matches[0])


def handle_list(args: argparse.Namespace) -> int:
    rows: list[tuple[str, str, int, str]] = []
    for engine in [args.engine] if args.engine else _ENGINES:
        for name, path in discover_suite_specs(engine).items():
            loaded = load_suite_spec(path)
            summary = loaded.spec.description.strip().splitlines()[0] if loaded.spec.description.strip() else ""
            rows.append((engine, name, len(loaded.cases), summary))
    if not rows:
        print("未发现任何 suite（benchmarks/specs/<engine>/*.yaml）")
        return 0
    name_width = max(len(r[1]) for r in rows)
    print(f"{'engine':<10} | {'suite':<{name_width}} | {'cases':>5} | description")
    for engine, name, n_cases, summary in rows:
        print(f"{engine:<10} | {name:<{name_width}} | {n_cases:>5} | {summary}")
    return 0


def handle_show(args: argparse.Namespace) -> int:
    try:
        loaded = _resolve_target(args.target, args.engine)
    except ValueError as err:
        sys.exit(str(err))
    engine = loaded.spec.engine
    print(f"# {loaded.path}  (sha256={loaded.sha256[:12]})")
    print(yaml.safe_dump(loaded.raw, sort_keys=False, allow_unicode=True).rstrip())
    cases = loaded.cases
    print(f"\n# 展开后的用例（共 {len(cases)} 个，顺序即执行顺序）")
    for i, case_spec in enumerate(cases, 1):
        print(f"  [{i:02d}] {_describe_case(engine, case_spec)}")
    return 0


def handle_schema(args: argparse.Namespace) -> int:
    schema = release_json_schema() if args.kind == "release" else suite_json_schema()
    text = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0
