from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from launcher import build, probes
from launcher.errors import LauncherError
from launcher.probes import read_mojo_pin, read_project_version
from launcher.verified import VerifiedStore


def test_the_pin_is_read_from_pyproject() -> None:
    assert read_mojo_pin('dependencies = [\n    "mojo==1.1.0",\n    "max-core==26.6.0",\n]') == "1.1.0"


def test_a_missing_pin_is_a_launcher_error_naming_the_file() -> None:
    with pytest.raises(LauncherError, match=r"pyproject\.toml has no mojo==X pin"):
        read_mojo_pin('dependencies = ["max-core==26.6.0"]')


def test_the_version_is_read_from_pyproject() -> None:
    assert read_project_version('[project]\nname = "x"\nversion = "0.1.0"\n') == "0.1.0"


def test_another_table_cannot_override_the_version() -> None:
    text = '[tool.x]\nversion = "9"\n[project]\nname = "x"\nversion = "0.1.0"\n'
    assert read_project_version(text) == "0.1.0"


@pytest.mark.parametrize("text", ['[project]\nname = "x"\n', '', 'not = [toml'])
def test_a_missing_version_is_a_launcher_error_naming_the_file(text: str) -> None:
    with pytest.raises(LauncherError, match=r"pyproject\.toml has no \[project\] version"):
        read_project_version(text)


def _machine_without_tools(monkeypatch: pytest.MonkeyPatch, cached: Path | None, store: VerifiedStore) -> None:
    monkeypatch.setattr(probes.shutil, "which", lambda name: None)
    monkeypatch.setattr(probes, "cached_default_model", lambda hub: (cached, None))
    monkeypatch.setattr(probes, "default_store", lambda: store)


def test_real_probes_says_whether_the_cached_model_was_verified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"gguf")
    store = VerifiedStore(tmp_path / "verified.json")
    _machine_without_tools(monkeypatch, model, store)
    unchecked = probes.real_probes()
    assert unchecked.cached_model == model and unchecked.verified is False
    store.record(model, "0" * 64)
    assert probes.real_probes().verified is True


def test_real_probes_with_no_cached_model_is_not_verified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _machine_without_tools(monkeypatch, None, VerifiedStore(tmp_path / "verified.json"))
    got = probes.real_probes()
    assert got.cached_model is None and got.verified is False


def _run_raises(error: Exception):
    def run(*args: object, **kwargs: object) -> None:
        raise error

    return run


def test_capture_keeps_the_output_and_no_reason_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probes.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "out\n", ""))
    assert probes._capture(["x"]) == ("out\n", None)


@pytest.mark.parametrize(
    ("run", "reason"),
    [
        (_run_raises(subprocess.TimeoutExpired("x", 60)), "timed out after 60 s"),
        (_run_raises(FileNotFoundError(2, "No such file or directory")), "x: No such file or directory"),
        (lambda cmd, **k: subprocess.CompletedProcess(cmd, 9, "", "driver mismatch\nmore\n"), "exit 9: driver mismatch"),
        (lambda cmd, **k: subprocess.CompletedProcess(cmd, 3, "", ""), "exit 3"),
    ],
    ids=["timeout", "oserror", "exit-with-stderr", "exit-silent"],
)
def test_capture_says_why_it_failed(monkeypatch: pytest.MonkeyPatch, run: object, reason: str) -> None:
    monkeypatch.setattr(probes.subprocess, "run", run)
    assert probes._capture(["x"]) == (None, reason)


def test_read_returns_the_text_or_the_reason(tmp_path: Path) -> None:
    f = tmp_path / "cpuinfo"
    f.write_text("flags : avx2\n")
    assert probes._read(str(f)) == ("flags : avx2\n", None)
    text, why = probes._read(str(tmp_path / "missing"))
    assert text is None and why == "No such file or directory"


def test_the_default_model_lookup_names_a_launcher_error_and_an_os_error() -> None:
    class Failing:
        def __init__(self, error: Exception) -> None:
            self.error = error
            self.cache_dir = Path("/c")
            self.offline = True

        def list_files(self, repo: str) -> dict[str, int]:
            raise self.error

    assert probes.cached_default_model(Failing(LauncherError("the repo has no .gguf file"))) == (
        None, "the repo has no .gguf file")
    assert probes.cached_default_model(Failing(PermissionError(13, "Permission denied"))) == (
        None, "Permission denied")


def _fake_machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, libc: str = "glibc",
                  smi_out: str = "NVIDIA RTX 3050, 8.6, 5120, 580.65.06\n") -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        out = "Mojo 1.1.0 (abc)\n" if cmd[0] == "uv" else smi_out
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(probes.subprocess, "run", run)
    monkeypatch.setattr(probes.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(probes.platform, "libc_ver", lambda: (libc, "2.39"))
    monkeypatch.setattr(probes, "cached_default_model", lambda hub: (None, None))
    monkeypatch.setattr(probes, "default_store", lambda: VerifiedStore(tmp_path / "v.json"))
    return calls


def test_real_probes_never_syncs_and_reads_the_driver_in_the_same_nvidia_smi_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _fake_machine(monkeypatch, tmp_path)
    got = probes.real_probes()
    uv = next(c for c in calls if c[0] == "uv")
    assert "--no-sync" in uv and "--frozen" in uv
    smi = [c for c in calls if c[0] == "nvidia-smi"]
    assert len(smi) == 1 and "driver_version" in smi[0][1]
    assert got.gpu_text == "NVIDIA RTX 3050, 8.6, 5120\n" and got.driver_text == "580.65.06\n"
    assert got.mojo_version_output == "Mojo 1.1.0 (abc)\n" and got.mojo_failure is None and got.gpu_failure is None


def test_real_probes_glibc_is_empty_for_another_libc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_machine(monkeypatch, tmp_path, libc="musl")
    assert probes.real_probes().glibc == ""
    _fake_machine(monkeypatch, tmp_path, libc="glibc")
    assert probes.real_probes().glibc == "2.39"


def test_real_probes_carries_the_reasons(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_machine(monkeypatch, tmp_path)

    def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "uv":
            raise subprocess.TimeoutExpired(cmd, 60)
        return subprocess.CompletedProcess(cmd, 9, "", "driver/library version mismatch\n")

    monkeypatch.setattr(probes.subprocess, "run", run)
    monkeypatch.setattr(probes, "_read", lambda path: (None, "Permission denied"))
    got = probes.real_probes()
    assert got.mojo_version_output is None and got.mojo_failure == "timed out after 60 s"
    assert got.gpu_text is None and got.gpu_failure == "exit 9: driver/library version mismatch"
    assert got.cpuinfo is None and got.cpuinfo_error == "Permission denied"
    assert got.meminfo is None and got.meminfo_error == "Permission denied"


def test_real_probes_says_when_nvidia_smi_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_machine(monkeypatch, tmp_path)
    monkeypatch.setattr(probes.shutil, "which", lambda name: None if name == "nvidia-smi" else f"/usr/bin/{name}")
    got = probes.real_probes()
    assert got.gpu_text is None and got.gpu_failure == "nvidia-smi not found on PATH"


def test_real_probes_reports_a_stale_stamp_and_an_empty_src(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_machine(monkeypatch, tmp_path)
    (tmp_path / "pyproject.toml").write_text('dependencies = ["mojo==1.1.0"]\n')
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "mojo-dllm").write_text("binary")
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "BINARY", tmp_path / "build" / "mojo-dllm")
    monkeypatch.setenv("MOJO_DLLM_GPU_ARCH", "sm_86")
    got = probes.real_probes()
    assert got.source_mtimes == () and "no stamp" in str(got.stamp_problem)
