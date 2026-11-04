from __future__ import annotations

import io
import os
import re
import tomllib
from pathlib import Path

import pytest

from launcher import cli
from launcher.ui import Style

COMMANDS = ["setup", "pull", "run", "doctor", "help"]
ROOT = Path(__file__).resolve().parents[2]
ENV_VARS = ["HF_TOKEN", "HF_HOME", "HF_HUB_CACHE", "MOJO_DLLM_GPU_ARCH"]


def test_help_lists_every_command_and_the_environment() -> None:
    text = cli.help_text(Style(False))
    for name in COMMANDS:
        assert f"  {name} " in text, name
    for var in ENV_VARS:
        assert var in text, var
    assert "Quick start" in text and "Examples" in text
    assert "./mojo-dllm run" in text and "./mojo-dllm doctor" in text and "./mojo-dllm setup" in text
    assert "inspect" in text and "tokenize" in text
    assert "--version" in text


def test_help_has_no_colour_when_disabled_and_some_when_enabled() -> None:
    assert "\x1b[" not in cli.help_text(Style(False))
    assert "\x1b[" in cli.help_text(Style(True))


def test_style_follows_tty_and_no_color() -> None:
    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    assert Style.for_stream(Tty(), {}).enabled
    assert not Style.for_stream(Tty(), {"NO_COLOR": "1"}).enabled
    assert not Style.for_stream(io.StringIO(), {}).enabled


@pytest.mark.parametrize("argv", [[], ["help"], ["-h"], ["--help"]])
def test_help_forms_exit_zero(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 0
    assert "Quick start" in capsys.readouterr().out


def test_launcher_commands_answer_help_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--help"]) == 0
    assert "Quick start" in capsys.readouterr().out


def test_run_without_a_prompt_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and './mojo-dllm run "' in err


def test_pull_nonsense_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["pull", "nonsense"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "owner/repo[:QUANT]" in err


def test_setup_usage_error_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["setup", "--bogus"]) == 2
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.parametrize(
    "argv",
    [
        ["run"],
        ["run", "hi", "--steps"],
        ["run", "hi", "-m"],
        ["run", "Explain", "diffusion"],
        ["run", "hi", "--device", "tpu"],
        ["run", "--prompt", "hi"],
        ["pull", "--bogus"],
        ["pull", "a/b", "c/d"],
    ],
)
def test_command_line_mistakes_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_a_mistyped_command_builds_nothing_and_names_the_closest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a mistyped command must not build or exec")

    monkeypatch.setattr(cli, "ensure_built", forbidden)
    monkeypatch.setattr(os, "execv", forbidden)
    assert cli.main(["doctr"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "did you mean doctor?" in err and "./mojo-dllm help" in err


def test_version_prints_the_pyproject_version_and_builds_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("--version must not build, probe, or exec")

    monkeypatch.setattr(cli, "ensure_built", forbidden)
    monkeypatch.setattr(cli, "real_probes", forbidden)
    monkeypatch.setattr(os, "execv", forbidden)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == f"mojo-dllm {version}\n"


def test_version_without_pyproject_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["--version"]) == 1
    assert capsys.readouterr().err.startswith("error: cannot read pyproject.toml")


def test_a_word_close_to_nothing_still_points_at_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "ensure_built", lambda: pytest.fail("built"))
    assert cli.main(["zzzzzzzz"]) == 2
    err = capsys.readouterr().err
    assert "did you mean" not in err and "./mojo-dllm help" in err


@pytest.mark.parametrize("command", ["inspect", "tokenize", "logits"])
def test_each_binary_command_is_forwarded(command: str, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "ensure_built", lambda: False)
    monkeypatch.setattr(os, "execv", lambda path, argv: seen.append(argv))
    cli.main([command, "--x"])
    assert seen == [[str(cli.BINARY), command, "--x"]]


def test_binary_commands_match_src_main_mojo() -> None:
    src = (ROOT / "src/main.mojo").read_text()
    start = src.find("def command_spec(")
    assert start != -1, "src/main.mojo has no command_spec(); update this test"
    body = src[start:].split("\ndef ", 1)[0]
    in_binary = set(re.findall(r'command\s*==\s*"([a-z][a-z-]*)"', body))
    assert in_binary, "found no command names in command_spec(); update this test"
    assert set(cli.BINARY_COMMANDS) == in_binary - {"run"}
    assert set(cli.LAUNCHER_COMMANDS) >= {"run"}


def test_a_missing_binary_after_the_build_names_doctor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "ensure_built", lambda: False)
    monkeypatch.setattr(cli, "BINARY", tmp_path / "absent")
    assert cli.main(["inspect", "m.gguf"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "./mojo-dllm doctor" in err and "absent" in err
