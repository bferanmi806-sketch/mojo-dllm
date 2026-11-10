from __future__ import annotations

import os
from pathlib import Path

import pytest

from launcher import build
from launcher.build import is_stale
from launcher.errors import LauncherError


def test_missing_binary_is_stale() -> None:
    assert is_stale(None, [1.0])


def test_newer_source_makes_it_stale() -> None:
    assert is_stale(100.0, [50.0, 100.5])


def test_equal_or_older_sources_are_fresh() -> None:
    assert not is_stale(100.0, [50.0, 100.0])


def test_no_sources_is_an_error_not_a_fresh_binary() -> None:
    with pytest.raises(LauncherError, match="no sources"):
        is_stale(100.0, [])


PYPROJECT = 'dependencies = [\n    "mojo==1.1.0",\n]\n'


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project tree with a fresh, stamped binary, and a GPU architecture chosen by the environment."""
    for rel, text in (
        ("src/main.mojo", "fn main(): pass\n"), ("pyproject.toml", PYPROJECT),
        ("uv.lock", "lock\n"), ("scripts/build.sh", "#!/bin/bash\n"),
    ):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        os.utime(f, (1000, 1000))
    monkeypatch.setattr(build, "ROOT", tmp_path)
    monkeypatch.setattr(build, "BINARY", tmp_path / "build" / "mojo-dllm")
    monkeypatch.setenv("MOJO_DLLM_GPU_ARCH", "sm_86")
    monkeypatch.setattr(build, "_gpu_compute_caps", lambda: pytest.fail("nvidia-smi asked although the arch is set"))
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "mojo-dllm").write_text("binary")
    os.utime(tmp_path / "build" / "mojo-dllm", (2000, 2000))
    build.write_stamp()
    return tmp_path


def test_a_stamped_binary_newer_than_its_inputs_is_fresh(tree: Path) -> None:
    assert build.needs_build() is False and build.stamp_problem() is None


@pytest.mark.parametrize("rel", ["src/main.mojo", "pyproject.toml", "uv.lock", "scripts/build.sh"])
def test_touching_an_input_makes_the_binary_stale(tree: Path, rel: str) -> None:
    os.utime(tree / rel, (3000, 3000))
    assert build.needs_build() is True


def test_a_changed_gpu_arch_makes_the_binary_stale(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOJO_DLLM_GPU_ARCH", "sm_90")
    assert build.needs_build() is True
    assert "sm_86" in str(build.stamp_problem()) and "sm_90" in str(build.stamp_problem())


def test_a_changed_mojo_pin_makes_the_binary_stale_even_with_old_mtimes(tree: Path) -> None:
    (tree / "pyproject.toml").write_text(PYPROJECT.replace("1.1.0", "1.2.0"))
    os.utime(tree / "pyproject.toml", (1000, 1000))
    assert build.needs_build() is True and "1.2.0" in str(build.stamp_problem())


def test_a_binary_without_a_stamp_is_rebuilt(tree: Path) -> None:
    build.stamp_path().unlink()
    assert build.needs_build() is True and "no stamp" in str(build.stamp_problem())


def test_an_empty_src_raises(tree: Path) -> None:
    (tree / "src" / "main.mojo").unlink()
    with pytest.raises(LauncherError, match="no sources"):
        build.needs_build()


def test_the_arch_falls_back_to_the_local_gpu_then_sm_80(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOJO_DLLM_GPU_ARCH", raising=False)
    monkeypatch.setattr(build, "_gpu_compute_caps", lambda: "8.6\n7.5\n")
    assert build.target_arch() == "sm_86"
    monkeypatch.setattr(build, "_gpu_compute_caps", lambda: "")
    assert build.target_arch() == "sm_80"
    monkeypatch.setattr(build, "_gpu_compute_caps", lambda: "[N/A]\n")
    assert build.target_arch() == "sm_80"


def test_project_env_drops_virtual_env_and_keeps_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/.venv")
    monkeypatch.setenv("KEEP_ME", "1")
    env = build.project_env()
    assert "VIRTUAL_ENV" not in env and env["KEEP_ME"] == "1"


class FakeBuild:
    """Stands in for the Popen of scripts/build.sh."""

    def __init__(self, lines: list[str], code: int) -> None:
        self.stdout = iter(f"{line}\n" for line in lines)
        self._code = code

    def wait(self) -> int:
        return self._code


KWARGS: list[dict[str, object]] = []


def _fake_popen(monkeypatch: pytest.MonkeyPatch, lines: list[str], code: int) -> list[list[str]]:
    calls: list[list[str]] = []
    KWARGS.clear()

    def popen(cmd: list[str], **kwargs: object) -> FakeBuild:
        calls.append(cmd)
        KWARGS.append(kwargs)
        return FakeBuild(lines, code)

    monkeypatch.setattr(build, "needs_build", lambda: True)
    monkeypatch.setattr(build.subprocess, "Popen", popen)
    return calls


def test_a_failed_build_raises_with_the_last_20_lines(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_popen(monkeypatch, [f"line {i}" for i in range(50)], 2)
    with pytest.raises(LauncherError) as e:
        build.ensure_built()
    msg = str(e.value)
    assert "the build failed" in msg and "./mojo-dllm doctor" in msg
    assert "line 30" in msg.splitlines()[1] and "line 49" in msg and "line 29\n" not in msg + "\n"


def test_a_good_build_reports_that_it_built(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_popen(monkeypatch, ["ok"], 0)
    assert build.ensure_built() is True
    assert calls == [["bash", "scripts/build.sh"]]


def test_the_build_runs_in_the_project_without_virtual_env_and_writes_to_stderr(
    tree: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/.venv")
    _fake_popen(monkeypatch, ["compiling x"], 0)
    build.ensure_built()
    kwargs = KWARGS[0]
    assert kwargs["cwd"] == tree and "VIRTUAL_ENV" not in kwargs["env"]  # type: ignore[operator]
    captured = capsys.readouterr()
    assert "compiling x" in captured.err and captured.out == ""


def test_a_good_build_writes_the_stamp_and_a_failed_one_does_not(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build.stamp_path().unlink()
    _fake_popen(monkeypatch, [], 2)
    with pytest.raises(LauncherError):
        build.ensure_built()
    assert not build.stamp_path().exists()
    _fake_popen(monkeypatch, [], 0)
    build.ensure_built()
    assert build.stamp_path().read_text() == "mojo 1.1.0\narch sm_86\n"


def test_a_current_binary_is_not_rebuilt(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_popen(monkeypatch, [], 0)
    monkeypatch.setattr(build, "needs_build", lambda: False)
    assert build.ensure_built() is False and calls == []


def test_a_missing_bash_is_a_launcher_error(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def popen(cmd: list[str], **kwargs: object) -> None:
        raise FileNotFoundError(2, "No such file or directory", "bash")

    monkeypatch.setattr(build, "needs_build", lambda: True)
    monkeypatch.setattr(build.subprocess, "Popen", popen)
    with pytest.raises(LauncherError, match="cannot start bash"):
        build.ensure_built()
