"""CLI 单元测试：统一根命令与分级子命令树。"""

from unittest.mock import patch

import pytest

from mini_infra.cli import build_parser, main


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
        ["verify", "batch", "--help"],
        ["verify", "continuous", "--help"],
        ["verify", "lockstep-batch", "--help"],
        ["verify", "scheduler-batch", "--help"],
        ["verify", "parity", "transformers", "--help"],
        ["verify", "parity", "cache", "--help"],
        ["verify", "parity", "batch", "--help"],
        ["verify", "parity", "continuous", "--help"],
        ["verify", "parity", "lockstep-batch", "--help"],
        ["verify", "parity", "scheduler-batch", "--help"],
        ["bench", "lockstep", "case", "--help"],
        ["bench", "lockstep", "suite", "--help"],
        ["bench", "scheduler", "case", "--help"],
        ["bench", "scheduler", "suite", "--help"],
        ["bench", "plot", "history", "--help"],
        ["bench", "plot", "compare", "--help"],
        ["bench", "check-regression", "--help"],
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


def test_cli_scheduler_suite_dry_run(capsys):
    """测试 mini-infra bench scheduler suite --dry-run 端到端执行。"""
    ret = main(
        [
            "bench",
            "scheduler",
            "suite",
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
    assert "=== [Dry Run] Suite: fixed-vs-continuous-variable-output-length ===" in captured
    assert "Scheduler 基准评测执行计划" in captured


def test_cli_plot_history_no_records(tmp_path, monkeypatch, capsys):
    """测试 mini-infra bench plot history 在无历史记录时的优雅提示退出。"""
    from mini_infra.cli.commands.bench import plot

    monkeypatch.setattr(plot, "RESULT_HISTORY_DIR", tmp_path)
    monkeypatch.setattr(plot, "load_results", lambda: [])

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
