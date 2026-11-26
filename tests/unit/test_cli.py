"""CLI 单元测试：统一根命令与分级子命令树。"""

import argparse
import functools
import inspect
from dataclasses import replace
from unittest.mock import patch

import pytest

from mini_infra.cli import build_parser, main
from mini_infra.cli.commands.verify import parity


def test_cli_help(capsys):
    """测试根帮助输出。"""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr().out
    assert "mini-infra-qwen3" in captured
    assert "run" in captured
    assert "bench" in captured
    assert "verify" in captured


def test_cli_no_args_shows_help(capsys):
    """未传任何参数时展示帮助并以 0 退出。"""
    with patch("sys.argv", ["mini-infra"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
    captured = capsys.readouterr().out
    assert "mini-infra-qwen3" in captured


def test_cli_subcommand_helps():
    """测试各级子命令的 --help 参数解析不报错。"""
    parser = build_parser()
    subcommand_help_paths = [
        ["run", "--help"],
        ["bench", "--help"],
        ["verify", "--help"],
        ["verify", "smoke", "--help"],
        ["verify", "transformers", "--help"],
        ["verify", "cache", "--help"],
        ["verify", "batch-lockstep", "--help"],
        ["verify", "batch-scheduler", "--help"],
        ["verify", "schedulers-token-budget", "--help"],
        ["verify", "chunked-prefill", "--help"],
        ["verify", "prefix-caching", "--help"],
        ["verify", "paged-cache-logits", "--help"],
        ["verify", "paged-cache-scheduler", "--help"],
        ["bench", "lockstep", "case", "--help"],
        ["bench", "scheduler", "case", "--help"],
        ["bench", "sweep", "--help"],
        ["bench", "suites", "--help"],
        ["bench", "suites", "list", "--help"],
        ["bench", "suites", "show", "--help"],
        ["bench", "suites", "schema", "--help"],
        ["bench", "plot", "history", "--help"],
        ["bench", "plot", "compare", "--help"],
        ["bench", "check-regression", "--help"],
        ["bench", "release", "--help"],
    ]

    for path in subcommand_help_paths:
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(path)
        assert exc_info.value.code == 0


def test_cli_verify_smoke(capsys):
    """测试 mini-infra verify smoke 端到端执行。"""
    ret = main(["verify", "smoke"])
    assert ret == 0
    captured = capsys.readouterr().out
    assert "smoke test passed" in captured


@pytest.mark.parametrize("spec", parity._COMMANDS, ids=lambda s: s.name)
def test_cli_verify_parity_defaults_match_runner_signature(monkeypatch, spec):
    """验证每个 verify parity 子命令默认参数与 execute_compare_* 签名真源 1:1 对齐透传。"""
    captured_kwargs: dict[str, object] = {}

    @functools.wraps(spec.runner)
    def fake_runner(**kwargs: object) -> int:
        captured_kwargs.update(kwargs)
        return 0

    patched_commands = tuple(
        replace(item, runner=fake_runner) if item.name == spec.name else item
        for item in parity._COMMANDS
    )
    monkeypatch.setattr(parity, "_COMMANDS", patched_commands)

    assert main(["verify", spec.name]) == 0
    expected_kwargs = {
        name: (None if param.default is inspect.Parameter.empty else param.default)
        for name, param in inspect.signature(spec.runner).parameters.items()
    }
    assert captured_kwargs == expected_kwargs


def test_cli_verify_parity_custom_args_and_unknown_param_guard(monkeypatch):
    """验证自定义 CLI 参数覆盖透传，以及未登记形参在构建解析器时快速失败。"""
    captured_kwargs: dict[str, object] = {}
    target_spec = next(s for s in parity._COMMANDS if s.name == "chunked-prefill")

    @functools.wraps(target_spec.runner)
    def fake_runner(**kwargs: object) -> int:
        captured_kwargs.update(kwargs)
        return 0

    monkeypatch.setattr(
        parity,
        "_COMMANDS",
        tuple(
            replace(item, runner=fake_runner) if item.name == "chunked-prefill" else item
            for item in parity._COMMANDS
        ),
    )
    ret = main(
        [
            "verify",
            "chunked-prefill",
            "--model",
            "/tmp/local-qwen3",
            "--prompts",
            "p1",
            "p2",
            "--dtype",
            "bfloat16",
            "--max-new-tokens",
            "12",
            "--token-budget",
            "48",
            "--block-size",
            "8",
            "--max-running-requests",
            "2",
            "--logits-atol",
            "0.5",
            "--output",
            "tmp/parity.local.json",
        ]
    )
    assert ret == 0
    assert captured_kwargs["model"] == "/tmp/local-qwen3"
    assert captured_kwargs["prompts"] == ["p1", "p2"]
    assert captured_kwargs["dtype"] == "bfloat16"
    assert captured_kwargs["max_new_tokens"] == 12
    assert captured_kwargs["token_budget"] == 48
    assert captured_kwargs["block_size"] == 8
    assert captured_kwargs["max_running_requests"] == 2
    assert captured_kwargs["logits_atol"] == 0.5
    assert captured_kwargs["output"] == "tmp/parity.local.json"

    def bad_runner(unregistered_param: int = 1) -> int:
        return unregistered_param

    bad_parser = argparse.ArgumentParser()
    bad_subs = bad_parser.add_subparsers()
    with pytest.raises(KeyError, match="unregistered_param"):
        parity._register_command(
            bad_subs,
            parity._ParityCommandSpec(name="bad", help="bad", runner=bad_runner),
        )


def test_cli_scheduler_suite_dry_run(capsys):
    """测试 mini-infra bench sweep --engine scheduler --dry-run 端到端执行。"""
    ret = main(
        [
            "bench",
            "sweep",
            "--engine",
            "scheduler",
            "--tag",
            "dev",
            "--dry-run",
            "--suite",
            "fixed-vs-continuous-variable-output-length",
            "--repeats",
            "2",
        ]
    )
    assert ret == 0
    captured = capsys.readouterr().out
    assert (
        "=== [Dry Run] Suite: fixed-vs-continuous-variable-output-length ===" in captured
    )
    assert "Scheduler 基准评测执行计划" in captured


def test_cli_scheduler_paged_suite_dry_run(capsys):
    """测试 mini-infra bench sweep --engine scheduler 对 dense-vs-paged 套件的 --dry-run 执行。"""
    ret = main(
        [
            "bench",
            "sweep",
            "--engine",
            "scheduler",
            "--tag",
            "dev",
            "--dry-run",
            "--suite",
            "dense-vs-paged-block-size",
            "--repeats",
            "2",
            "--interleave",
            "caching_method",
            "block_size",
        ]
    )
    assert ret == 0
    captured = capsys.readouterr().out
    assert "=== [Dry Run] Suite: dense-vs-paged-block-size ===" in captured
    assert "Scheduler 基准评测执行计划" in captured
    assert "block_size=8" in captured
    assert "block_size=64" in captured


def test_cli_plot_history_no_records(tmp_path, monkeypatch, capsys):
    """测试 mini-infra bench plot history 在无历史记录时的优雅提示退出。"""
    from mini_infra.cli.commands.bench import plot

    monkeypatch.setattr(plot, "RESULT_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(plot, "load_results", lambda _dir=None: [])

    with pytest.raises(SystemExit) as exc_info:
        main(["bench", "plot", "history"])
    assert exc_info.value.code == 1
    captured = capsys.readouterr().out
    assert "没有结果" in captured


def test_cli_unknown_command():
    """测试未知子命令报错。"""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["unknown"])
