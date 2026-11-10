from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from launcher import runcmd
from launcher.errors import LauncherError, UsageError
from launcher.gpu import Gpu
from launcher.models import ModelResult
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


class _Wiring:
    """Monkeypatches the seams of cmd_run and records what reaches each of them."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, gpus: object = ([], None)) -> None:
        self.model = tmp_path / "models" / ".." / "m.gguf"
        (tmp_path / "models").mkdir()
        self.tmp = tmp_path
        self.execv: list[tuple[str, list[str]]] = []
        self.hubs: list[bool] = []
        self.ensure: list[tuple[str, bool]] = []
        self.detect_calls = 0
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setattr(runcmd, "BINARY", tmp_path / "mojo-dllm")
        monkeypatch.setattr(runcmd, "ensure_built", lambda: False)

        def hub(offline: bool = False) -> object:
            self.hubs.append(offline)
            return object()

        def ensure_model(text: str, hub: object, out: object, *, announce_cached: bool, store: object) -> Path:
            self.ensure.append((text, announce_cached))
            assert out is runcmd.sys.stderr, "progress must go to stderr so stdout stays the model's text"
            return self.model

        def detect() -> object:
            self.detect_calls += 1
            return gpus

        monkeypatch.setattr(runcmd, "Hub", hub)
        monkeypatch.setattr(runcmd, "ensure_model", ensure_model)
        monkeypatch.setattr(runcmd, "detect_gpus", detect)
        monkeypatch.setattr(runcmd, "default_store", lambda: None)
        monkeypatch.setattr(runcmd.os, "environ", {"PATH": "/usr/bin"})
        monkeypatch.setattr(runcmd.os, "execv", lambda path, argv: self.execv.append((path, argv)))


def test_run_hands_the_binary_the_resolved_model_the_chosen_device_and_the_other_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = _Wiring(monkeypatch, tmp_path, gpus=([Gpu("NVIDIA RTX", 8.6, 5120)], None))
    runcmd.cmd_run(["Say hi", "--max-tokens", "64", "--verbose", "-m", "a/b:Q4_K_M", "--steps", "8"])
    binary = str(tmp_path / "mojo-dllm")
    assert w.execv == [(binary, [
        binary, "run", "--model", str((tmp_path / "m.gguf").resolve()), "--prompt", "Say hi",
        "--device", "gpu", "--max-tokens", "64", "--verbose", "--steps", "8",
    ])]
    assert w.ensure == [("a/b:Q4_K_M", False)]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("device: gpu (NVIDIA RTX")


def test_run_on_cpu_never_asks_nvidia_smi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _Wiring(monkeypatch, tmp_path, gpus=([Gpu("NVIDIA RTX", 8.6, 5120)], None))
    runcmd.cmd_run(["hi", "--device", "cpu"])
    assert w.detect_calls == 0 and w.execv[0][1][w.execv[0][1].index("--device") + 1] == "cpu"


def test_run_auto_asks_nvidia_smi_once_and_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _Wiring(monkeypatch, tmp_path)
    runcmd.cmd_run(["hi"])
    assert w.detect_calls == 1 and w.execv[0][1][w.execv[0][1].index("--device") + 1] == "cpu"


@pytest.mark.parametrize(
    ("argv", "env", "offline"),
    [
        (["hi"], {}, False),
        (["hi", "--offline"], {}, True),
        (["hi"], {"HF_HUB_OFFLINE": "1"}, True),
        (["hi"], {"HF_HUB_OFFLINE": "0"}, False),
    ],
    ids=["online", "flag", "variable", "variable-off"],
)
def test_run_builds_an_offline_hub_for_the_flag_and_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str], env: dict[str, str], offline: bool
) -> None:
    w = _Wiring(monkeypatch, tmp_path)
    runcmd.os.environ.update(env)
    runcmd.cmd_run([*argv, "--device", "cpu"])
    assert w.hubs == [offline]


def test_pull_with_no_arguments_pulls_the_default_model_and_announces_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    w = _Wiring(monkeypatch, tmp_path)
    seen: list[tuple[str, bool, object]] = []

    def ensure_status(text: str, hub: object, out: object, *, announce_cached: bool, store: object) -> object:
        seen.append((text, announce_cached, out))
        return ModelResult(w.model, True)

    monkeypatch.setattr(runcmd, "ensure_model_status", ensure_status)
    assert runcmd.cmd_pull([]) == 0
    assert seen == [("llada-8b", True, runcmd.sys.stdout)] and w.hubs == [False]


def test_pull_offline_by_flag_or_variable_builds_an_offline_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _Wiring(monkeypatch, tmp_path)
    monkeypatch.setattr(runcmd, "ensure_model_status", lambda *a, **k: ModelResult(w.model, True))
    runcmd.cmd_pull(["--offline"])
    runcmd.os.environ["HF_HUB_OFFLINE"] = "true"
    runcmd.cmd_pull([])
    assert w.hubs == [True, True]


def test_pull_of_a_named_model_passes_it_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    w = _Wiring(monkeypatch, tmp_path)
    seen: list[str] = []
    monkeypatch.setattr(
        runcmd, "ensure_model_status", lambda text, *a, **k: seen.append(text) or ModelResult(w.model, True)
    )
    runcmd.cmd_pull(["a/b:Q6_K"])
    assert seen == ["a/b:Q6_K"]


def _smi_result(monkeypatch: pytest.MonkeyPatch, code: int, out: str, err: str) -> None:
    monkeypatch.setattr(runcmd.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(runcmd.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, code, out, err))


def test_nvidia_smi_failing_without_stderr_still_says_the_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    _smi_result(monkeypatch, 9, "", "")
    assert runcmd.detect_gpus() == ([], "exit 9")


@pytest.mark.parametrize("out", ["", "\n", "  \n"])
def test_nvidia_smi_succeeding_with_no_output_is_no_gpu_and_no_failure(monkeypatch: pytest.MonkeyPatch, out: str) -> None:
    _smi_result(monkeypatch, 0, out, "")
    assert runcmd.detect_gpus() == ([], None)


def test_nvidia_smi_succeeding_with_garbage_aborts_run_and_names_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # Intended today: an unreadable listing stops `run --device auto` (the message says to pass --device cpu).
    # Issue #81 covers whether it should fall back to the CPU instead.
    _smi_result(monkeypatch, 0, "garbage\n", "")
    with pytest.raises(LauncherError, match="--device cpu"):
        runcmd.detect_gpus()
