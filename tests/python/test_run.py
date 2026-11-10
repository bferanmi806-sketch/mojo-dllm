from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from launcher import runcmd
from launcher.errors import LauncherError, UsageError
from launcher.gpu import Gpu
from launcher.runcmd import BOOLEAN_FLAGS, RunArgs, parse_run_args, run_argv

ROOT = Path(__file__).resolve().parents[2]


def test_prompt_model_and_device_are_extracted() -> None:
    a = parse_run_args(["Say hi", "-m", "a/b:Q8_0", "--device", "cpu"])
    assert a == RunArgs(prompt="Say hi", model="a/b:Q8_0", device="cpu", offline=False, passthrough=[])


def test_defaults() -> None:
    a = parse_run_args(["hi"])
    assert (a.model, a.device, a.offline) == ("llada-8b", "auto", False)


def test_long_model_flag_and_offline() -> None:
    a = parse_run_args(["--model", "x/y", "--offline", "hi"])
    assert (a.model, a.offline, a.prompt) == ("x/y", True, "hi")


def test_other_flags_pass_through_in_order() -> None:
    a = parse_run_args(
        ["--max-tokens", "64", "--verbose", "Explain", "--steps", "32", "--seed", "7", "--json"]
    )
    assert a.prompt == "Explain"
    assert a.passthrough == ["--max-tokens", "64", "--verbose", "--steps", "32", "--seed", "7", "--json"]


def test_a_flag_value_is_never_the_prompt() -> None:
    with pytest.raises(LauncherError) as e:
        parse_run_args(["--max-tokens", "64"])
    assert './mojo-dllm run "Explain speculative decoding in two sentences."' in str(e.value)


def test_unknown_flags_take_a_value() -> None:
    a = parse_run_args(["--future-flag", "v", "hi"])
    assert a.prompt == "hi" and a.passthrough == ["--future-flag", "v"]


def test_a_second_bare_word_asks_for_quotes() -> None:
    with pytest.raises(LauncherError, match="quote the prompt"):
        parse_run_args(["Explain", "diffusion"])


def test_a_flag_without_its_value_is_an_error() -> None:
    with pytest.raises(LauncherError, match="--steps needs a value"):
        parse_run_args(["hi", "--steps"])
    with pytest.raises(LauncherError, match="-m needs a value"):
        parse_run_args(["hi", "-m"])


def test_bad_device_is_an_error() -> None:
    with pytest.raises(LauncherError, match="auto, cpu or gpu"):
        parse_run_args(["hi", "--device", "tpu"])


def test_argv_for_the_binary() -> None:
    a = parse_run_args(["Hi there", "--max-tokens", "8", "--verbose"])
    assert run_argv(Path("/r/build/mojo-dllm"), Path("/m/x.gguf"), a, "gpu") == [
        "/r/build/mojo-dllm", "run", "--model", "/m/x.gguf", "--prompt", "Hi there",
        "--device", "gpu", "--max-tokens", "8", "--verbose",
    ]


def test_sampler_flags_pass_through() -> None:
    a = parse_run_args(["Hi", "--threshold", "0.9", "--dual-cache"])
    assert run_argv(Path("/r/build/mojo-dllm"), Path("/m/x.gguf"), a, "gpu")[-3:] == [
        "--threshold", "0.9", "--dual-cache",
    ]


def test_boolean_flags_match_the_binary() -> None:
    src = (ROOT / "src/main.mojo").read_text()
    body = src.split("def boolean_flags()", 1)[1].split("return [", 1)[1].split("]", 1)[0]
    in_binary = {w.strip('"') for w in body.replace(",", " ").split() if w.startswith('"')}
    assert {f.removeprefix("--") for f in BOOLEAN_FLAGS} == in_binary


def test_double_dash_ends_the_options_so_a_prompt_may_start_with_a_dash() -> None:
    a = parse_run_args(["--max-tokens", "8", "--", "-5 plus 3?"])
    assert a.prompt == "-5 plus 3?" and a.passthrough == ["--max-tokens", "8"]


def test_everything_after_double_dash_is_positional() -> None:
    with pytest.raises(UsageError, match="quote the prompt"):
        parse_run_args(["--", "a", "--verbose"])


def test_prompt_as_a_flag_is_refused() -> None:
    with pytest.raises(UsageError, match="first word"):
        parse_run_args(["--prompt", "hi"])


def test_run_argument_mistakes_are_usage_errors() -> None:
    for argv in (["hi", "--steps"], ["a", "b"], ["hi", "--device", "tpu"], []):
        with pytest.raises(UsageError):
            parse_run_args(argv)


def _smi(monkeypatch: pytest.MonkeyPatch, run: object) -> None:
    monkeypatch.setattr(runcmd.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(runcmd.subprocess, "run", run)


def test_no_nvidia_smi_is_no_gpu_and_no_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runcmd.shutil, "which", lambda name: None)
    assert runcmd.detect_gpus() == ([], None)


def test_nvidia_smi_output_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, lambda *a, **k: subprocess.CompletedProcess(a, 0, "NVIDIA RTX, 8.6, 5120\n", ""))
    assert runcmd.detect_gpus() == ([Gpu("NVIDIA RTX", 8.6, 5120)], None)


@pytest.mark.parametrize(
    ("failure", "why"),
    [
        (subprocess.TimeoutExpired("nvidia-smi", 60), "timed out after 60 s"),
        (OSError("exec format error"), "exec format error"),
    ],
)
def test_nvidia_smi_that_cannot_run_is_reported(monkeypatch: pytest.MonkeyPatch, failure: Exception, why: str) -> None:
    def run(*args: object, **kwargs: object) -> None:
        raise failure

    _smi(monkeypatch, run)
    assert runcmd.detect_gpus() == ([], why)


def test_nvidia_smi_non_zero_exit_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi(monkeypatch, lambda *a, **k: subprocess.CompletedProcess(a, 9, "", "driver mismatch\n"))
    assert runcmd.detect_gpus() == ([], "exit 9: driver mismatch")


def test_run_says_why_there_is_no_gpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = tmp_path / "m.gguf"
    monkeypatch.setattr(runcmd, "ensure_built", lambda: False)
    monkeypatch.setattr(runcmd, "ensure_model", lambda *a, **k: model)
    monkeypatch.setattr(runcmd, "detect_gpus", lambda: ([], "timed out after 60 s"))
    monkeypatch.setattr(runcmd, "BINARY", tmp_path / "absent")
    with pytest.raises(LauncherError):
        runcmd.cmd_run(["hi"])
    assert "device: cpu (nvidia-smi failed: timed out after 60 s)" in capsys.readouterr().err


def test_a_missing_binary_in_run_names_doctor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runcmd, "ensure_built", lambda: False)
    monkeypatch.setattr(runcmd, "ensure_model", lambda *a, **k: tmp_path / "m.gguf")
    monkeypatch.setattr(runcmd, "detect_gpus", lambda: ([], None))
    monkeypatch.setattr(runcmd, "BINARY", tmp_path / "absent")
    with pytest.raises(LauncherError, match="./mojo-dllm doctor"):
        runcmd.cmd_run(["hi"])


def _cache_file(hub, tmp_path: Path, content: bytes) -> Path:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(content)
    hub.cached[("mradermacher/LLaDA-8B-Instruct-GGUF", "LLaDA-8B-Instruct.Q4_K_M.gguf")] = cached
    return cached


def _pull_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, hub, store, *, argv: list[str]) -> tuple[int, str]:
    import io

    out = io.StringIO()
    monkeypatch.setattr(runcmd, "default_store", lambda: store)
    monkeypatch.setattr(runcmd, "Hub", lambda offline=False: hub)
    monkeypatch.setattr(runcmd.sys, "stdout", out)
    hub.offline = True
    return runcmd.cmd_pull(argv), out.getvalue()


def test_pull_offline_of_an_unverified_cached_file_says_so_and_exits_3(monkeypatch, tmp_path: Path, hub, store) -> None:
    hub.files = {"LLaDA-8B-Instruct.Q4_K_M.gguf": 9}
    cached = _cache_file(hub, tmp_path, b"truncated")
    code, text = _pull_with(monkeypatch, tmp_path, hub, store, argv=["--offline"])
    assert code == 3
    assert f"already downloaded (not verified)\n{cached}" in text
    assert "pull" in text.lower() and "without --offline" in text


def test_pull_of_a_verified_cached_file_keeps_its_output_and_exits_0(monkeypatch, tmp_path: Path, hub, store) -> None:
    hub.files = {"LLaDA-8B-Instruct.Q4_K_M.gguf": 9}
    cached = _cache_file(hub, tmp_path, hub.content)
    store.record(cached, hub.sha256)
    code, text = _pull_with(monkeypatch, tmp_path, hub, store, argv=["--offline"])
    assert code == 0 and text == f"already downloaded\n{cached}\n"


@pytest.mark.parametrize(
    ("argv", "two_arg_form"),
    [
        (["--device=cpu", "hi"], "--device cpu"),
        (["hi", "--max-tokens=64"], "--max-tokens 64"),
        (["--model=foo.gguf", "hi"], "--model foo.gguf"),
        (["-m=a/b", "hi"], "-m a/b"),
        (["hi", "--steps=8"], "--steps 8"),
        (["hi", "--verbose=1"], "--verbose"),
    ],
)
def test_a_flag_with_an_equals_sign_is_refused_with_the_two_argument_form(argv: list[str], two_arg_form: str) -> None:
    with pytest.raises(UsageError) as e:
        parse_run_args(argv)
    msg = str(e.value)
    assert f"write {two_arg_form} (the flag" in msg and "needs a value" not in msg and "needs a prompt" not in msg


def test_a_prompt_that_contains_an_equals_sign_is_not_a_flag() -> None:
    assert parse_run_args(["a=b"]).prompt == "a=b"
    assert parse_run_args(["--", "--x=1"]).prompt == "--x=1"
    assert parse_run_args(["--seed", "7", "hi"]).passthrough == ["--seed", "7"]


def test_a_negative_value_is_the_flags_value_not_a_flag() -> None:
    a = parse_run_args(["hi", "--temperature", "-1"])
    assert a.prompt == "hi" and a.passthrough == ["--temperature", "-1"]


def test_a_repeated_device_flag_keeps_the_last_one() -> None:
    assert parse_run_args(["hi", "--device", "cpu", "--device", "gpu"]).device == "gpu"


def test_an_empty_prompt_is_a_prompt() -> None:
    assert parse_run_args([""]).prompt == ""
