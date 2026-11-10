from __future__ import annotations

import dataclasses
import io
import subprocess
from pathlib import Path

import pytest

from conftest import FakeHub, make_probes
from launcher import wizard
from launcher.errors import LauncherError, UsageError
from launcher.ui import Style
from launcher.verified import VerifiedStore
from launcher.wizard import SetupArgs, SetupEnv, parse_setup_args, run_setup

REPO = "mradermacher/LLaDA-8B-Instruct-GGUF"
FILE = "LLaDA-8B-Instruct.Q4_K_M.gguf"


class Env:
    """Builds a SetupEnv with recording fakes."""

    def __init__(self, hub: FakeHub, store: VerifiedStore, *, tty: bool = False, answer: str = "",
                 binary: bool = True, **probe_overrides: object) -> None:
        self.calls: list[str] = []
        self.questions: list[str] = []
        self.out, self.err = io.StringIO(), io.StringIO()

        def ask(question: str) -> str:
            self.questions.append(question)
            return answer

        self.env = SetupEnv(
            probes=make_probes(**probe_overrides),
            hub=hub,
            store=store,
            is_tty=tty,
            ask=ask,
            sync=lambda: self.calls.append("sync"),
            build=self._build,
            binary_ready=lambda: binary,
            smoke=self._smoke,
            style=Style(False),
            out=self.out,
            err=self.err,
        )

    def _build(self) -> bool:
        self.calls.append("build")
        return True

    def _smoke(self, model: Path, device: str, extra_env: object = None) -> float:
        self.calls.append(f"smoke:{device}")
        self.smoke_env = extra_env
        return 3.25

    def run(self, argv: list[str]) -> int:
        return run_setup(parse_setup_args(argv), self.env)


def test_without_a_tty_or_yes_the_model_is_an_outstanding_step(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    assert e.run([]) == 3
    text = e.out.getvalue()
    assert hub.downloads == [] and e.questions == []
    assert e.calls == ["sync", "build"]
    assert "Setup complete." not in text and "Try it:" not in text
    assert "  ./mojo-dllm pull" in text
    assert "not downloaded (no terminal to ask on; pass --yes to download)" in text
    for step in ("[1/4] Toolchain", "[2/4] Build", "[3/4] Model", "[4/4] Check"):
        assert step in text


def test_outstanding_steps_are_printed_before_the_end_summary(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    e.run([])
    text = e.out.getvalue()
    assert text.rstrip().splitlines()[-1] == "  ./mojo-dllm pull"
    assert "1 step outstanding" in text


def test_yes_downloads_and_finishes(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    assert e.run(["--yes"]) == 0
    text = e.out.getvalue()
    assert hub.downloads == [(REPO, FILE)] and e.questions == []
    assert "sha256 verified" in text and "Setup complete." in text
    assert text.index("Setup complete.") < text.index("Try it:")
    assert './mojo-dllm run "Explain speculative decoding in two sentences."' in text
    assert "./mojo-dllm doctor" in text
    assert e.calls == ["sync", "build", "smoke:gpu"]
    assert "3.2 s" in text or "3.3 s" in text


def test_smoke_test_uses_cpu_without_a_gpu(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, gpu_text="")
    assert e.run(["--yes"]) == 0 and e.calls[-1] == "smoke:cpu"
    assert "device: cpu (no NVIDIA GPU found)" in e.out.getvalue()


def test_the_smoke_test_leaves_a_cuda_visible_devices_the_user_set_and_says_so(
    hub: FakeHub, store: VerifiedStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_gpu_select import TWO_GPUS

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    e = Env(hub, store, gpu_text=TWO_GPUS)
    assert e.run(["--yes"]) == 0 and e.calls[-1] == "smoke:gpu"
    assert e.smoke_env == {}
    assert e.out.getvalue().count("did not pin the card") == 1


def test_the_smoke_test_runs_on_the_card_that_qualifies(
    hub: FakeHub, store: VerifiedStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_gpu_select import NEW_UUID, TWO_GPUS

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    e = Env(hub, store, gpu_text=TWO_GPUS)
    assert e.run(["--yes"]) == 0 and e.calls[-1] == "smoke:gpu"
    assert "RTX 3090" in e.out.getvalue()
    assert e.smoke_env == {"CUDA_VISIBLE_DEVICES": NEW_UUID, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def test_an_unreadable_nvidia_smi_is_not_reported_as_no_gpu(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, gpu_text=None)
    assert e.run(["--yes"]) == 0 and e.calls[-1] == "smoke:cpu"
    text = e.out.getvalue()
    assert "device: cpu (nvidia-smi missing or could not be read; GPU not checked)" in text
    assert "no NVIDIA GPU found" not in text


def test_tty_asks_and_empty_answer_means_yes(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, tty=True, answer="")
    assert e.run([]) == 0
    assert len(e.questions) == 1
    assert e.questions[0].startswith(f"Download {REPO}/{FILE} (4.9 GB) to ")
    assert e.questions[0].endswith("? [Y/n]")
    assert str(hub.cache_dir) in e.questions[0]
    assert hub.downloads == [(REPO, FILE)]


def test_tty_no_leaves_the_step_outstanding(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, tty=True, answer="n")
    assert e.run([]) == 3 and hub.downloads == []


def test_no_model_skips_silently_and_completes(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, tty=True)
    assert e.run(["--no-model"]) == 0
    assert e.questions == [] and hub.downloads == []
    text = e.out.getvalue()
    assert "Setup complete (no smoke test)" in text and "Setup complete." not in text
    assert e.calls == ["sync", "build"]


def test_a_cached_model_is_not_asked_about_and_setup_is_idempotent(
    hub: FakeHub, store: VerifiedStore, tmp_path: Path
) -> None:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(hub.content)
    hub.cached[(REPO, FILE)] = cached
    e = Env(hub, store, tty=True)
    assert e.run([]) == 0
    assert e.questions == [] and hub.downloads == []
    assert "already downloaded" in e.out.getvalue()


def test_setup_checks_an_unverified_cached_model_once(hub: FakeHub, store: VerifiedStore, tmp_path: Path) -> None:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(hub.content)
    hub.cached[(REPO, FILE)] = cached
    assert Env(hub, store).run([]) == 0
    assert Env(hub, store).run([]) == 0
    assert hub.sha_requests == 1 and store.is_verified(cached)


def test_setup_offline_with_an_unverified_cached_model_leaves_a_step_outstanding(
    hub: FakeHub, store: VerifiedStore, tmp_path: Path
) -> None:
    hub.offline = True
    hub.files = {FILE: 10}
    cached = tmp_path / "c.gguf"
    cached.write_bytes(hub.content)
    hub.cached[(REPO, FILE)] = cached
    e = Env(hub, store)
    assert e.run([]) == 3
    text = e.out.getvalue()
    assert f"already downloaded (not verified)\n{cached}" in text
    assert "Setup complete." not in text and text.rstrip().splitlines()[-1] == "  ./mojo-dllm pull"


def test_setup_with_a_verified_cached_model_completes(hub: FakeHub, store: VerifiedStore, tmp_path: Path) -> None:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(hub.content)
    hub.cached[(REPO, FILE)] = cached
    store.record(cached, hub.sha256)
    e = Env(hub, store)
    assert e.run([]) == 0 and f"already downloaded\n{cached}" in e.out.getvalue()


def test_setup_refuses_a_cached_model_that_fails_its_check(hub: FakeHub, store: VerifiedStore, tmp_path: Path) -> None:
    cached = tmp_path / "c.gguf"
    cached.write_bytes(b"truncated")
    hub.cached[(REPO, FILE)] = cached
    e = Env(hub, store)
    assert e.run([]) == 1
    assert "does not match" in e.err.getvalue() and not cached.exists()
    assert "smoke" not in " ".join(e.calls)


def test_a_failing_preflight_stops_before_anything_runs(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, glibc="2.31")
    assert e.run(["--yes"]) == 1
    assert e.calls == [] and hub.downloads == []
    assert "✗" in e.out.getvalue() + e.err.getvalue()
    assert "error:" in e.err.getvalue()


def test_model_flag_selects_the_model(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    e.run(["--yes", "--model", "a/b:Q8_0"])
    assert hub.downloads == [("a/b", "LLaDA-8B-Instruct.Q8_0.gguf")]


def test_outstanding_pull_names_a_non_default_model(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)
    e.run(["--model", "a/b:Q8_0"])
    assert "  ./mojo-dllm pull a/b:Q8_0" in e.out.getvalue()


def test_a_build_that_leaves_no_binary_stops_setup(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store, binary=False)
    assert e.run(["--yes"]) == 1
    assert "build/mojo-dllm does not exist" in e.err.getvalue()
    assert "Setup complete" not in e.out.getvalue() and "smoke" not in " ".join(e.calls)


class _Input:
    def __init__(self, answer: str | None) -> None:
        self.answer = answer

    def __call__(self, question: str) -> str:
        if self.answer is None:
            raise EOFError
        return self.answer


@pytest.mark.parametrize(
    ("answer", "downloads"),
    [("y", 1), ("Y", 1), ("yes", 1), ("", 1), ("n", 0), ("maybe", 0), (None, 0)],
    ids=["y", "Y", "yes", "enter", "n", "maybe", "ctrl-d"],
)
def test_the_download_question_accepts_yes_and_treats_anything_else_as_no(
    hub: FakeHub, store: VerifiedStore, answer: str | None, downloads: int
) -> None:
    e = Env(hub, store, tty=True)
    e.env = dataclasses.replace(e.env, ask=_Input(answer))
    code = e.run([])
    assert len(hub.downloads) == downloads
    assert code == (0 if downloads else 3)
    assert ("not downloaded (declined)" in e.out.getvalue()) == (not downloads)


def test_a_local_model_path_never_touches_the_hub(hub: FakeHub, store: VerifiedStore, tmp_path: Path) -> None:
    model = tmp_path / "x.gguf"
    model.write_bytes(b"gguf")
    hub.list_files = lambda repo: pytest.fail("the hub was asked about a local file")  # type: ignore[method-assign]
    e = Env(hub, store)
    assert e.run(["--model", str(model)]) == 0
    assert f"using {model}" in e.out.getvalue()
    assert e.calls[-1] == "smoke:gpu" and hub.downloads == []


def test_setup_accepts_offline_and_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parse_setup_args(["--offline"]).offline is True
    assert parse_setup_args([]).offline is False
    monkeypatch.setattr("launcher.probes.real_probes", lambda: make_probes())
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert wizard.real_env(offline=False).hub.offline is False
    assert wizard.real_env(offline=True).hub.offline is True
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert wizard.real_env(offline=False).hub.offline is True


def test_offline_setup_with_a_cached_model_does_not_need_the_network(
    hub: FakeHub, store: VerifiedStore, tmp_path: Path
) -> None:
    hub.offline = True
    hub.files = {FILE: 10}
    cached = tmp_path / "c.gguf"
    cached.write_bytes(hub.content)
    hub.cached[(REPO, FILE)] = cached
    store.record(cached, hub.sha256)
    e = Env(hub, store)
    assert e.run([]) == 0 and hub.downloads == [] and hub.sha_requests == 0


def test_usage_errors() -> None:
    for argv in (["--bogus"], ["--model"], ["extra"]):
        with pytest.raises(UsageError):
            parse_setup_args(argv)
    assert parse_setup_args(["--yes", "--no-model", "--model", "x/y", "--offline"]) == SetupArgs(
        model="x/y", yes=True, no_model=True, offline=True
    )


def _fake_run(monkeypatch: pytest.MonkeyPatch, code: int) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, code)

    monkeypatch.setattr(wizard.subprocess, "run", run)
    return calls


def test_a_failed_uv_sync_stops_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_run(monkeypatch, 1)
    with pytest.raises(LauncherError, match="uv sync --frozen failed"):
        wizard.real_sync()
    assert calls == [["uv", "sync", "--frozen"]]


def test_a_good_uv_sync_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_run(monkeypatch, 0)
    wizard.real_sync()


def _smoke_run(monkeypatch: pytest.MonkeyPatch, *, code: int = 0, stdout: str = "hello\n") -> list[dict[str, object]]:
    seen: list[dict[str, object]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append({"cmd": cmd, **kwargs})
        return subprocess.CompletedProcess(cmd, code, stdout)

    monkeypatch.setattr(wizard.subprocess, "run", run)
    return seen


def test_the_smoke_test_binary_gets_the_card_pinning_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _smoke_run(monkeypatch)
    monkeypatch.setenv("KEEP", "1")
    wizard.real_smoke(tmp_path / "m.gguf", "gpu", {"CUDA_VISIBLE_DEVICES": "GPU-x"})
    env = seen[0]["env"]
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-x" and env["KEEP"] == "1"  # type: ignore[index]


def test_a_failed_smoke_test_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _smoke_run(monkeypatch, code=3)
    with pytest.raises(LauncherError, match="smoke test failed"):
        wizard.real_smoke(tmp_path / "m.gguf", "cpu")
    assert seen[0]["cmd"][1:3] == ["run", "--model"]  # type: ignore[index]


@pytest.mark.parametrize("stdout", ["", "  \n\n"], ids=["empty", "blank"])
def test_a_smoke_test_that_prints_nothing_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: str) -> None:
    _smoke_run(monkeypatch, stdout=stdout)
    with pytest.raises(LauncherError, match="printed nothing"):
        wizard.real_smoke(tmp_path / "m.gguf", "cpu")


def test_a_hung_smoke_test_is_a_launcher_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def run(cmd: list[str], **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])  # type: ignore[arg-type]

    monkeypatch.setattr(wizard.subprocess, "run", run)
    with pytest.raises(LauncherError, match="did not finish within 300 s"):
        wizard.real_smoke(tmp_path / "m.gguf", "cpu")


def test_the_smoke_test_has_a_timeout_a_full_argv_and_returns_the_measured_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seen = _smoke_run(monkeypatch, stdout="Hello there\n")
    clock = iter([100.0, 102.5])
    monkeypatch.setattr(wizard.time, "monotonic", lambda: next(clock))
    model = tmp_path / "m.gguf"
    assert wizard.real_smoke(model, "cpu") == 2.5
    assert seen[0]["cmd"] == [
        str(wizard.build.BINARY), "run", "--model", str(model.resolve()), "--prompt", "Say hello.",
        "--max-tokens", "16", "--steps", "8", "--device", "cpu",
    ]
    assert seen[0]["timeout"] == 300
    assert capsys.readouterr().out == "Hello there\n"


def test_an_os_error_during_setup_is_an_error_line_and_status_1(hub: FakeHub, store: VerifiedStore) -> None:
    e = Env(hub, store)

    def sync() -> None:
        raise PermissionError(13, "Permission denied", "/repo/.venv")

    e.env = dataclasses.replace(e.env, sync=sync)
    assert e.run(["--yes"]) == 1
    err = e.err.getvalue()
    assert err.startswith("error: ") and "/repo/.venv" in err and "Permission denied" in err


def test_garbage_from_nvidia_smi_means_cpu_after_the_download_not_a_failure(hub: FakeHub, store: VerifiedStore) -> None:
    # doctor only warns about an unreadable listing; setup must not fail after a multi-gigabyte download for it.
    e = Env(hub, store, gpu_text="garbage\n")
    assert e.run(["--yes"]) == 0
    text = e.out.getvalue()
    assert hub.downloads == [(REPO, FILE)] and e.calls[-1] == "smoke:cpu"
    assert "device: cpu (nvidia-smi output is unreadable: nvidia-smi printed a line the launcher cannot read" in text
    assert e.smoke_env == {}
