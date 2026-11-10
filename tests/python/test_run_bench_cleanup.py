"""run_bench puts the machine back on every way out, and reports the error that stopped it (#235).

powerprofilesctl is the conftest FakeMachine and os.kill is a stub: nothing here changes the real power profile
or signals a real process.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import isolation  # noqa: E402
import run_bench  # noqa: E402
from conftest import FakeMachine  # noqa: E402
from test_bench_isolation_wiring import FAKE_MOJO_RUN, FAKE_SMI, _script  # noqa: E402


@pytest.fixture
def live(fake_machine: FakeMachine) -> FakeMachine:
    fake_machine.auto = True
    fake_machine.profile = fake_machine.platform_profile = "balanced"
    return fake_machine


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live: FakeMachine) -> Iterator[Path]:
    """A config, fake binaries and an output path; yields the output path. argv is set for main()."""
    _script(tmp_path / "bin/nvidia-smi", FAKE_SMI)
    # The real powerprofilesctl is never reachable: a code path that bypasses the FakeMachine hits this stub,
    # leaves a mark and exits 97, and the fixture fails the test at teardown even if the code ignored the exit code.
    _script(tmp_path / "bin/powerprofilesctl", '#!/usr/bin/env bash\necho "$@" >> "$0.called"\nexit 97\n')
    _script(tmp_path / "build/mojo-dllm", FAKE_MOJO_RUN)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    cfg = {"models": {"gguf": str(model)}, "llama_cpp": str(tmp_path), "diffuse_cpp": str(tmp_path),
           "runtimes": ["mojo-dllm"], "prompts": ["hello"], "gen_length": 16, "steps": 4, "block_length": 4,
           "threads": 2, "busy_threshold": 0.1, "max_quiet_wait_s": 5, "kill_pytest": False}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(run_bench, "ROOT", tmp_path)
    monkeypatch.setattr(run_bench, "mojo_version", lambda: "Mojo 1.1.0 (fake)")
    out = tmp_path / "results" / "x.json"
    monkeypatch.setattr(sys, "argv", ["run_bench.py", "--config", str(tmp_path / "config.json"), "--reps", "1",
                                     "--out", str(out), "--allow-battery"])
    yield out
    called = tmp_path / "bin/powerprofilesctl.called"
    assert not called.exists(), f"the real powerprofilesctl was called: {called.read_text()}"


def env_file(out: Path) -> Path:
    return out.parent / "env" / "x.env.json"


def fail_after_the_arm_started(monkeypatch: pytest.MonkeyPatch, out: Path, damage: str) -> None:
    """Tokenizing runs after the preflight wrote the env file: damage the file, then fail the way a real run would."""
    def boom(text: str) -> tuple[str, int]:
        path = env_file(out)
        if damage == "corrupt":
            path.write_text("{not json")
        else:
            path.unlink()
        raise RuntimeError("the original error")

    monkeypatch.setattr(run_bench, "parse_token_ids", boom)


@pytest.mark.parametrize("damage", ["corrupt", "deleted"])
def test_a_damaged_env_file_neither_hides_the_error_nor_keeps_the_profile(
    bench: Path, live: FakeMachine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    fail_after_the_arm_started(monkeypatch, bench, damage)
    with pytest.raises(RuntimeError) as e:
        run_bench.main()
    assert str(e.value) == "the original error"
    assert live.profile == "balanced", "the profile must be back after the run, not left on performance"
    assert "!! could not write the closing snapshot" in capsys.readouterr().out


def test_the_profile_is_set_and_restored_with_a_read_back(bench: Path, live: FakeMachine) -> None:
    assert run_bench.main() == 0
    sets = [c for c in live.calls if c[:2] == ["powerprofilesctl", "set"]]
    assert [c[2] for c in sets] == ["performance", "balanced"]
    assert live.profile == "balanced"


def test_a_profile_that_does_not_take_stops_the_run_before_any_measurement(
    bench: Path, live: FakeMachine
) -> None:
    live.profile_set_works = False
    with pytest.raises(SystemExit) as e:
        run_bench.main()
    assert "did not take" in str(e.value)
    assert not bench.exists(), "nothing was measured, so no result file"


def test_an_unreadable_profile_is_not_replaced_by_a_guess(bench: Path, live: FakeMachine) -> None:
    live.profile = ""
    with pytest.raises(SystemExit) as e:
        run_bench.main()
    assert "profile to restore is unknown" in str(e.value)
    assert not [c for c in live.calls if c[:2] == ["powerprofilesctl", "set"]], "no `set` from a guessed profile"


def test_a_restore_that_does_not_take_is_logged_with_the_command_to_run(
    bench: Path, live: FakeMachine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(text: str) -> tuple[str, int]:
        live.profile_set_works = False
        raise RuntimeError("the original error")

    monkeypatch.setattr(run_bench, "parse_token_ids", boom)
    with pytest.raises(RuntimeError, match="the original error"):
        run_bench.main()
    assert "could not restore the power profile to balanced" in capsys.readouterr().out


def test_the_battery_pause_hooks_are_read_back(live: FakeMachine) -> None:
    logs: list[str] = []
    on_pause, on_resume = run_bench.profile_hooks(live, "balanced", logs.append)
    live.profile = "performance"
    on_pause()
    assert live.profile == "balanced" and logs == []
    on_resume()
    assert live.profile == "performance"
    live.profile_set_works = False
    on_pause()
    assert logs and logs[0].startswith("!!") and "powerprofilesctl set balanced" in logs[0]
    live.profile = "balanced"
    with pytest.raises(isolation.IsolationError):
        on_resume()


def fake_proc(root: Path, entries: dict[int, str]) -> Path:
    for pid, comm in entries.items():
        (root / str(pid)).mkdir(parents=True)
        (root / str(pid) / "comm").write_text(comm + "\n")
    return root


def test_a_refused_kill_is_logged_and_the_sweep_goes_on(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    proc = fake_proc(tmp_path, {10: "pytest", 11: "pytest", 12: "bash"})
    killed: list[int] = []

    def kill(pid: int, sig: int) -> None:
        assert sig == signal.SIGKILL
        if pid == 10:
            raise PermissionError("operation not permitted")
        killed.append(pid)

    log: list[dict] = []
    wd = run_bench.Watchdog("pytest", log, proc=proc, kill=kill)
    wd.sweep()
    assert killed == [11] and [k["pid"] for k in log] == [11], "only the successful kill is counted"
    assert len(wd.errors) == 1 and "pid 10" in wd.errors[0]
    assert "!! could not kill pytest pid 10" in capsys.readouterr().out


def test_another_users_pytest_is_never_signalled(tmp_path: Path) -> None:
    proc = fake_proc(tmp_path, {10: "pytest"})

    def kill(pid: int, sig: int) -> None:
        raise AssertionError("signalled a process of another user")

    wd = run_bench.Watchdog("pytest", [], proc=proc, kill=kill, uid=os.getuid() + 1)
    wd.sweep()
    assert wd.log == [] and wd.errors == []


def test_a_vanished_process_is_skipped_silently(tmp_path: Path) -> None:
    proc = fake_proc(tmp_path, {10: "pytest"})

    def kill(pid: int, sig: int) -> None:
        raise ProcessLookupError

    wd = run_bench.Watchdog("pytest", [], proc=proc, kill=kill)
    wd.sweep()
    assert wd.log == [] and wd.errors == []


class Args:
    tune = False
    allow_noisy = True
    allow_battery = True
    isolation_retries = 0


@pytest.mark.parametrize("damage", ["corrupt", "deleted"])
def test_arm_session_reports_the_original_error_when_the_env_file_is_damaged(
    tmp_path: Path, live: FakeMachine, damage: str
) -> None:
    logs: list[str] = []
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    with pytest.raises(RuntimeError, match="the original error"):
        with isolation.arm_session(Args(), arm="a", device="cpu", model=model, runs_dir=tmp_path, machine=live,
                                   log=logs.append):
            path = tmp_path / "a.env.json"
            path.write_text("{not json") if damage == "corrupt" else path.unlink()
            raise RuntimeError("the original error")
    assert any(l.startswith("!! could not write the closing snapshot") for l in logs)


def test_a_closing_snapshot_that_raises_does_not_replace_the_error_or_keep_the_profile(
    bench: Path, live: FakeMachine, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_finish(self: isolation.Arm) -> None:
        raise RuntimeError("the snapshot broke")

    monkeypatch.setattr(isolation.Arm, "finish", broken_finish)
    monkeypatch.setattr(run_bench, "parse_token_ids", lambda text: (_ for _ in ()).throw(RuntimeError("the original error")))
    with pytest.raises(RuntimeError, match="the original error"):
        run_bench.main()
    assert live.profile == "balanced"


def test_main_hands_the_pause_and_resume_hooks_to_every_run(
    bench: Path, live: FakeMachine, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}
    original = isolation.Arm.run

    def spy(self: isolation.Arm, fn, **kw):  # type: ignore[no-untyped-def]
        seen.update(kw)
        return original(self, fn, **kw)

    monkeypatch.setattr(isolation.Arm, "run", spy)
    assert run_bench.main() == 0
    live.profile = "performance"
    seen["on_pause"]()
    assert live.profile == "balanced", "on_pause puts the profile main found back"
    seen["on_resume"]()
    assert live.profile == "performance"
