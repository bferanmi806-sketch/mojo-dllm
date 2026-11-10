"""scripts/bench-quiet.sh against fake `systemctl`, `nvidia-smi` and `runuser` shims on a temporary PATH.

The script refuses to run unless it is root; tests set BENCH_QUIET_ASSUME_ROOT=1 (a documented seam) so the
shims, which keep their state in files under tmp_path, do the work. Nothing here touches a real unit or GPU.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench-quiet.sh"

FAKE_SYSTEMCTL = r"""#!/usr/bin/env bash
# fake systemctl: a timer is active when $FAKE_DIR/active/NAME exists. Every call is logged.
echo "systemctl $*" >> "$FAKE_DIR/calls"
verb="$1"; shift
case "$verb" in
  is-active) [ "$1" = --quiet ] && shift; [ -e "$FAKE_DIR/active/$1" ] ;;
  stop)  grep -qx "timer ${1%.timer}" "$BENCH_QUIET_STATE" 2>/dev/null && echo "$1" >> "$FAKE_DIR/recorded_before_stop"
         [ "${FAKE_FAIL_STOP:-}" = "$1" ] && exit 1; rm -f "$FAKE_DIR/active/$1" ;;
  start) [ "${FAKE_FAIL_START:-}" = "$1" ] && exit 1; touch "$FAKE_DIR/active/$1" ;;
  *) exit 2 ;;
esac
"""
FAKE_NVIDIA = r"""#!/usr/bin/env bash
# fake nvidia-smi with $FAKE_GPUS cards (default 1). Card 0 keeps its state in $FAKE_DIR/persistence and
# $FAKE_DIR/locked, card N in persistence.N and locked.N. A call without `-i N` acts on every card, which the
# script under test must never do for a write: the call is logged as `ALL` and fails.
echo "nvidia-smi $*" >> "$FAKE_DIR/calls"
n="${FAKE_GPUS:-1}"
idx=""
if [ "$1" = -i ]; then idx="$2"; shift 2; fi
suffix=""; [ -n "$idx" ] && [ "$idx" != 0 ] && suffix=".$idx"
case "$1" in
  --query-gpu=index) seq 0 $((n - 1)) ;;
  --query-gpu=persistence_mode) [ "$FAKE_FAIL_QUERY" = "$idx" ] && exit 1; cat "$FAKE_DIR/persistence$suffix" ;;
  --query-gpu=clocks.max.sm) echo $((2100 + ${idx:-0} * 100)) ;;
  -pm)
    [ -n "$idx" ] || { echo "ALL" >> "$FAKE_DIR/calls"; exit 5; }
    [ "$2" = 1 ] && grep -qx "gpu-persistence $idx Disabled" "$BENCH_QUIET_STATE" 2>/dev/null && echo "pm $idx" >> "$FAKE_DIR/recorded_before_write"
    [ "$FAKE_FAIL_PM" = "$idx" ] && exit 1
    if [ "$2" = 1 ]; then echo Enabled > "$FAKE_DIR/persistence$suffix"; else echo Disabled > "$FAKE_DIR/persistence$suffix"; fi ;;
  -lgc)
    [ -n "$idx" ] || { echo "ALL" >> "$FAKE_DIR/calls"; exit 5; }
    grep -qx "gpu-clocks $idx locked" "$BENCH_QUIET_STATE" 2>/dev/null && echo "lgc $idx" >> "$FAKE_DIR/recorded_before_write"
    { [ -n "${FAKE_REFUSE_LGC:-}" ] || [ "${FAKE_REFUSE_LGC_INDEX:-}" = "$idx" ]; } && { echo "not supported" >&2; exit 4; }
    echo "$2" > "$FAKE_DIR/locked$suffix" ;;
  -rgc)
    [ -n "$idx" ] || { echo "ALL" >> "$FAKE_DIR/calls"; exit 5; }
    rm -f "$FAKE_DIR/locked$suffix" ;;
  *) exit 3 ;;
esac
"""
FAKE_RUNUSER = r"""#!/usr/bin/env bash
echo "runuser $*" >> "$FAKE_DIR/calls"
shift 2     # -u USER
shift       # --
exec "$@"
"""
FAKE_CRONTAB = "#!/usr/bin/env bash\necho \"crontab $*\" >> \"$FAKE_DIR/calls\"\n"
# The only real programs the script may reach. PATH is this directory alone, so a real systemctl or nvidia-smi
# can never be found: a test that forgets a shim fails with "command not found" instead of touching the machine.
TOOLS = ("bash", "cat", "grep", "head", "id", "kill", "rm", "sed", "seq", "sleep", "tr", "touch", "true", "false", "env",
         "dirname")
TIMERS = ["sysstat-collect", "apt-daily", "apt-daily-upgrade", "fwupd-refresh", "anacron", "man-db", "motd-news"]


def _default_signals() -> None:
    """A job started from a background shell or under nohup inherits SIGINT or SIGHUP ignored; the child under
    test must start with the default dispositions."""
    for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_DFL)


def _script(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0


class Rig:
    def __init__(self, tmp: Path) -> None:
        self.dir, self.bin = tmp / "fake", tmp / "bin"
        (self.dir / "active").mkdir(parents=True)
        for tool in TOOLS:
            found = shutil.which(tool)
            assert found, tool
            (self.bin).mkdir(parents=True, exist_ok=True)
            (self.bin / tool).symlink_to(found)
        for name, body in (("systemctl", FAKE_SYSTEMCTL), ("nvidia-smi", FAKE_NVIDIA), ("runuser", FAKE_RUNUSER),
                           ("crontab", FAKE_CRONTAB)):
            _script(self.bin / name, body)
        for i in range(4):
            self.set_persistence(i, "Disabled")
        self.state = tmp / "state"
        self.env = {**os.environ, "PATH": str(self.bin), "FAKE_DIR": str(self.dir),
                    "BENCH_QUIET_ASSUME_ROOT": "1", "BENCH_QUIET_TESTING": "1", "BENCH_QUIET_STATE": str(self.state)}
        self.env.pop("SUDO_USER", None)

    def active(self, *names: str) -> None:
        for n in names:
            (self.dir / "active" / f"{n}.timer").touch()

    def is_active(self, name: str) -> bool:
        return (self.dir / "active" / f"{name}.timer").exists()

    def calls(self) -> list[str]:
        f = self.dir / "calls"
        return f.read_text().splitlines() if f.exists() else []

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["bash", str(SCRIPT), *args], env={**self.env, **env}, capture_output=True, text=True,
                              timeout=60)

    @property
    def persistence(self) -> str:
        return self.persistence_of(0)

    @property
    def locked(self) -> bool:
        return self.locked_of(0)

    def persistence_of(self, i: int) -> str:
        return (self.dir / (f"persistence.{i}" if i else "persistence")).read_text().strip()

    def locked_of(self, i: int) -> bool:
        return (self.dir / (f"locked.{i}" if i else "locked")).exists()

    def set_persistence(self, i: int, value: str) -> None:
        (self.dir / (f"persistence.{i}" if i else "persistence")).write_text(value + "\n")


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def test_the_script_is_valid_shell_and_passes_shellcheck() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    assert os.access(SCRIPT, os.X_OK)
    if shutil.which("shellcheck"):
        p = subprocess.run(["shellcheck", "--severity=warning", str(SCRIPT)], capture_output=True, text=True)
        assert p.returncode == 0, p.stdout
    assert "\\x" not in SCRIPT.read_text()


def test_fake_shims_pass_bash_n(tmp_path: Path) -> None:
    for i, body in enumerate((FAKE_SYSTEMCTL, FAKE_NVIDIA, FAKE_RUNUSER, FAKE_CRONTAB)):
        _script(tmp_path / f"s{i}", body)


@pytest.mark.skipif(os.geteuid() == 0, reason="the refusal is for non-root users")
def test_refuses_to_run_without_root_and_changes_nothing(rig: Rig) -> None:
    rig.active(*TIMERS)
    env = {k: v for k, v in rig.env.items() if k not in ("BENCH_QUIET_ASSUME_ROOT", "BENCH_QUIET_TESTING")}
    p = subprocess.run(["bash", str(SCRIPT), "--", "true"], env=env, capture_output=True, text=True)
    assert p.returncode == 2 and "must run as root" in p.stderr
    assert rig.calls() == [] and all(rig.is_active(t) for t in TIMERS)


def test_no_command_is_a_usage_error(rig: Rig) -> None:
    assert rig.run("--").returncode == 2 and rig.run().returncode == 2
    assert rig.calls() == []


def test_stops_only_active_timers_runs_the_command_and_restores_everything(rig: Rig) -> None:
    rig.active("sysstat-collect", "apt-daily", "man-db")            # the other four are not active
    marker = rig.dir / "seen"
    cmd = f'for t in sysstat-collect apt-daily man-db; do test ! -e "{rig.dir}/active/$t.timer" || exit 9; done; ' \
          f'cat "{rig.dir}/persistence" > "{marker}"; test -e "{rig.dir}/locked" || exit 8; exit 7'
    p = rig.run("--", "bash", "-c", cmd)
    assert p.returncode == 7, p.stderr                                  # the command's own status
    assert marker.read_text().strip() == "Enabled"                       # quieted while it ran
    assert all(rig.is_active(t) for t in ("sysstat-collect", "apt-daily", "man-db"))
    assert not rig.is_active("anacron") and not rig.is_active("fwupd-refresh")          # never started what it did not stop
    assert rig.persistence == "Disabled" and not rig.locked and not rig.state.exists()
    assert "changed:  stopped sysstat-collect.timer stopped apt-daily.timer stopped man-db.timer" in p.stdout
    assert "persistence mode on" in p.stdout and "GPU 0 clocks locked at 2100 MHz" in p.stdout
    assert "restored: " in p.stdout and "started man-db.timer" in p.stdout and "GPU 0 clocks reset" in p.stdout
    assert not any(c.startswith("systemctl start anacron") for c in rig.calls())


def test_persistence_already_on_is_left_on(rig: Rig) -> None:
    (rig.dir / "persistence").write_text("Enabled\n")
    p = rig.run("--", "true")
    assert p.returncode == 0 and rig.persistence == "Enabled"
    assert "nvidia-smi -i 0 -pm 0" not in rig.calls() and "nvidia-smi -i 0 -pm 1" not in rig.calls()


def test_a_driver_that_refuses_the_clock_lock_is_reported_and_not_reset(rig: Rig) -> None:
    p = rig.run("--", "true", FAKE_REFUSE_LGC="1")
    assert p.returncode == 0 and "refused to lock the clocks of GPU 0" in p.stdout
    assert "nvidia-smi -i 0 -rgc" not in rig.calls() and rig.persistence == "Disabled"


def test_no_nvidia_smi_leaves_the_gpu_alone(rig: Rig) -> None:
    (rig.bin / "nvidia-smi").unlink()
    rig.active("man-db")
    p = rig.run("--", "true")
    assert p.returncode == 0 and "no nvidia-smi" in p.stdout and rig.is_active("man-db")


def test_never_touches_cron(rig: Rig) -> None:
    rig.active(*TIMERS)
    assert rig.run("--", "true").returncode == 0
    assert not any(c.startswith("crontab") for c in rig.calls())
    text = SCRIPT.read_text()
    assert "crontab -e" not in text and "/etc/cron" not in text


@pytest.mark.parametrize("sig,code", [(signal.SIGINT, 130), (signal.SIGTERM, 143)])
def test_restores_on_a_signal(rig: Rig, sig: int, code: int) -> None:
    rig.active("sysstat-collect", "anacron")
    p = subprocess.Popen(["bash", str(SCRIPT), "--", "sleep", "30"], env=rig.env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, preexec_fn=_default_signals)
    deadline = time.time() + 10
    while not rig.state.exists() or "gpu-clocks" not in rig.state.read_text():
        assert time.time() < deadline and p.poll() is None
        time.sleep(0.05)
    assert os.path.basename(os.readlink(f"/proc/{p.pid}/exe")) == "bash"          # the pid is the script's shell
    assert not rig.is_active("sysstat-collect")
    time.sleep(0.3)                                                               # the child has started
    os.kill(p.pid, sig)
    out, err = p.communicate(timeout=30)
    assert p.returncode == code, (out, err)
    assert rig.is_active("sysstat-collect") and rig.is_active("anacron")
    assert rig.persistence == "Disabled" and not rig.locked and not rig.state.exists()
    assert "restored: " in out and "started anacron.timer" in out
    assert not Path(f"/proc/{p.pid}").exists()                                    # the shell is gone


def test_a_failed_restore_is_reported_kept_in_the_state_file_and_exits_nonzero(rig: Rig) -> None:
    rig.active("man-db", "anacron")
    p = rig.run("--", "true", FAKE_FAIL_START="man-db.timer")
    assert p.returncode == 70 and "NOT restored: start man-db.timer" in p.stderr and "--restore" in p.stderr
    assert rig.state.read_text().splitlines() == ["timer man-db"]            # the rest went back
    assert rig.is_active("anacron") and not rig.is_active("man-db")
    p2 = rig.run("--restore")
    assert p2.returncode == 0 and rig.is_active("man-db") and not rig.state.exists()


def test_a_stale_state_file_blocks_a_new_run_until_restored(rig: Rig) -> None:
    rig.state.write_text("timer apt-daily\ngpu-persistence 0 Disabled\ngpu-clocks 0 locked\n")
    (rig.dir / "persistence").write_text("Enabled\n")
    (rig.dir / "locked").write_text("2100,2100\n")
    p = rig.run("--", "true")
    assert p.returncode == 1 and "--restore" in p.stderr and rig.calls() == []
    p = rig.run("--restore")
    assert p.returncode == 0 and rig.is_active("apt-daily") and rig.persistence == "Disabled" and not rig.locked
    assert not rig.state.exists()
    assert "nothing to restore" in rig.run("--restore").stdout


def test_a_timer_that_will_not_stop_is_left_alone_and_not_recorded(rig: Rig) -> None:
    rig.active("man-db", "anacron")
    p = rig.run("--", "true", FAKE_FAIL_STOP="man-db.timer")
    assert p.returncode == 0 and "could not stop man-db.timer" in p.stderr
    assert "started man-db.timer" not in p.stdout and rig.is_active("man-db") and rig.is_active("anacron")


def test_the_command_runs_as_the_invoking_user(rig: Rig) -> None:
    p = rig.run("--", "true", SUDO_USER="vlad")
    assert p.returncode == 0 and "runuser -u vlad -- true" in rig.calls()
    rig.calls().clear()
    q = rig.run("--", "true", SUDO_USER="root")
    assert q.returncode == 0 and sum(c.startswith("runuser") for c in rig.calls()) == 1


def test_the_command_sees_that_the_machine_was_quieted(rig: Rig) -> None:
    out = rig.dir / "env"
    rig.active("man-db", "anacron")
    p = rig.run("--", "bash", "-c", f'printf "%s" "${{BENCH_QUIET_ACTIVE:-unset}}" > "{out}"')
    assert p.returncode == 0 and out.read_text() == "timers:2,pm:1,clocks:1,refused:0"


@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGQUIT, signal.SIGUSR1])
def test_restores_on_other_signals_and_leaves_no_child(rig: Rig, sig: int, tmp_path: Path) -> None:
    """A dropped ssh session (HUP), QUIT, or a stray USR1 that no trap names ends the shell: the EXIT trap restores."""
    rig.active("sysstat-collect")
    pidfile = tmp_path / "child.pid"
    p = subprocess.Popen(["bash", str(SCRIPT), "--", "bash", "-c", f'echo $$ > "{pidfile}"; exec sleep 30'],
                         env=rig.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         preexec_fn=_default_signals)
    deadline = time.time() + 10
    while not pidfile.exists() or not pidfile.read_text().strip():
        assert time.time() < deadline and p.poll() is None
        time.sleep(0.05)
    child = int(pidfile.read_text())
    assert os.path.basename(os.readlink(f"/proc/{p.pid}/exe")) == "bash" and os.readlink(f"/proc/{child}/exe").endswith("sleep")
    os.kill(p.pid, sig)
    p.communicate(timeout=30)
    assert rig.is_active("sysstat-collect") and rig.persistence == "Disabled" and not rig.locked
    assert not rig.state.exists()
    for _ in range(50):
        if not Path(f"/proc/{child}").exists():
            break
        time.sleep(0.1)
    assert not Path(f"/proc/{child}").exists(), "the benchmark command was left running"


@pytest.mark.skipif(os.geteuid() == 0, reason="the refusal is for non-root users")
def test_the_root_seam_alone_does_not_skip_the_root_check(rig: Rig) -> None:
    """A stray BENCH_QUIET_ASSUME_ROOT=1 in someone's environment must not turn the guard off."""
    rig.active(*TIMERS)
    env = {k: v for k, v in rig.env.items() if k != "BENCH_QUIET_TESTING"}
    p = subprocess.run(["bash", str(SCRIPT), "--", "true"], env=env, capture_output=True, text=True)
    assert p.returncode == 2 and "must run as root" in p.stderr
    assert rig.calls() == [] and all(rig.is_active(t) for t in TIMERS)


def test_help_prints_the_whole_header_including_the_state_variable(rig: Rig) -> None:
    p = rig.run("--help")
    assert p.returncode == 0
    assert "BENCH_QUIET_STATE=PATH" in p.stdout and "default /run/bench-quiet.state" in p.stdout
    assert "BENCH_QUIET_TESTING=1" in p.stdout and "SIGKILL" in p.stdout
    assert "set -uo pipefail" not in p.stdout                  # the header only, not the code


# What was quieted is recorded, and every card is handled on its own (#247) ------------------------------------

def exported(rig: Rig, **env: str) -> str:
    out = rig.dir / "exported"
    p = rig.run("--", "bash", "-c", f'printf "%s" "${{BENCH_QUIET_ACTIVE:-unset}}" > "{out}"', **env)
    assert p.returncode == 0, p.stderr
    return out.read_text()


def test_a_refused_timer_stop_shows_in_the_exported_value(rig: Rig) -> None:
    rig.active("man-db", "anacron", "apt-daily")
    assert exported(rig, FAKE_FAIL_STOP="man-db.timer") == "timers:2,pm:1,clocks:1,refused:1"


def test_a_refused_clock_lock_shows_in_the_exported_value(rig: Rig) -> None:
    assert exported(rig, FAKE_REFUSE_LGC="1") == "timers:0,pm:1,clocks:0,refused:1"


def test_a_failed_persistence_switch_shows_in_the_exported_value_and_is_not_restored(rig: Rig) -> None:
    out = exported(rig, FAKE_FAIL_PM="0")
    assert out == "timers:0,pm:0,clocks:1,refused:1"
    assert "nvidia-smi -i 0 -pm 0" not in rig.calls()                 # it never switched it on, so it never switches it off


def test_an_unreadable_persistence_state_is_counted_and_the_card_left_alone(rig: Rig) -> None:
    assert exported(rig, FAKE_FAIL_QUERY="0") == "timers:0,pm:0,clocks:1,refused:1"
    assert "nvidia-smi -i 0 -pm 1" not in rig.calls()


def test_no_gpu_tool_exports_zero_changes(rig: Rig) -> None:
    (rig.bin / "nvidia-smi").unlink()
    rig.active("anacron")
    assert exported(rig) == "timers:1,pm:0,clocks:0,refused:0"


def test_every_timer_is_recorded_in_the_state_file_before_it_is_stopped(rig: Rig) -> None:
    rig.active("man-db", "anacron")
    assert exported(rig, FAKE_DEBUG="") == "timers:2,pm:1,clocks:1,refused:0"
    assert sorted((rig.dir / "recorded_before_stop").read_text().split()) == ["anacron.timer", "man-db.timer"]


def test_a_timer_that_will_not_stop_leaves_no_state_line(rig: Rig) -> None:
    rig.active("man-db")
    seen = rig.dir / "state-during"
    p = rig.run("--", "bash", "-c", f'cat "$BENCH_QUIET_STATE" > "{seen}"', FAKE_FAIL_STOP="man-db.timer")
    assert p.returncode == 0 and "timer man-db" not in seen.read_text().splitlines()
    assert rig.is_active("man-db")


def test_two_cards_are_each_restored_to_their_own_state(rig: Rig) -> None:
    rig.set_persistence(0, "Enabled")                                     # an administrator turned this one on
    rig.set_persistence(1, "Disabled")
    during = rig.dir / "during"
    cmd = f'echo "$(cat "{rig.dir}/persistence") $(cat "{rig.dir}/persistence.1")" > "{during}"'
    p = rig.run("--", "bash", "-c", cmd, FAKE_GPUS="2")
    assert p.returncode == 0, p.stderr
    assert during.read_text().split() == ["Enabled", "Enabled"]          # both quieted while it ran
    assert rig.persistence_of(0) == "Enabled" and rig.persistence_of(1) == "Disabled"
    calls = rig.calls()
    assert "nvidia-smi -i 0 -pm 0" not in calls and "nvidia-smi -i 1 -pm 0" in calls
    assert "nvidia-smi -i 0 -pm 1" not in calls and "nvidia-smi -i 1 -pm 1" in calls
    assert not rig.locked_of(0) and not rig.locked_of(1) and not rig.state.exists()
    assert "nvidia-smi -i 0 -lgc 2100,2100" in calls and "nvidia-smi -i 1 -lgc 2200,2200" in calls
    assert "GPU 1 clocks locked at 2200 MHz" in p.stdout and "GPU 1 persistence mode off" in p.stdout
    assert "GPU 0 persistence mode was already on" in p.stdout


def test_a_card_that_refuses_the_lock_is_not_reset_while_the_other_is(rig: Rig) -> None:
    p = rig.run("--", "true", FAKE_GPUS="2", FAKE_REFUSE_LGC_INDEX="1")
    assert p.returncode == 0 and "lock the clocks of GPU 1" in p.stdout
    calls = rig.calls()
    assert "nvidia-smi -i 0 -rgc" in calls and "nvidia-smi -i 1 -rgc" not in calls


def test_the_exported_value_counts_every_card(rig: Rig) -> None:
    rig.set_persistence(1, "Enabled")
    assert exported(rig, FAKE_GPUS="3", FAKE_REFUSE_LGC_INDEX="2") == "timers:0,pm:2,clocks:2,refused:1"


def test_the_script_never_writes_to_every_card_at_once(rig: Rig) -> None:
    rig.run("--", "true", FAKE_GPUS="2")
    assert "ALL" not in rig.calls()
    writes = [c for c in rig.calls() if any(f" {flag}" in c for flag in ("-pm", "-lgc", "-rgc"))]
    assert writes and all(c.startswith(("nvidia-smi -i 0 ", "nvidia-smi -i 1 ")) for c in writes)


def test_a_stale_state_file_of_two_cards_restores_each(rig: Rig) -> None:
    rig.state.write_text("gpu-persistence 1 Disabled\ngpu-clocks 1 locked\ngpu-persistence 0 Enabled\n")
    rig.set_persistence(0, "Enabled")
    rig.set_persistence(1, "Enabled")
    (rig.dir / "locked.1").write_text("2200,2200\n")
    p = rig.run("--restore")
    assert p.returncode == 0 and rig.persistence_of(1) == "Disabled" and rig.persistence_of(0) == "Enabled"
    assert not rig.locked_of(1) and not rig.state.exists()
    assert "nvidia-smi -i 0 -pm 0" not in rig.calls()


def test_a_failed_listing_of_the_cards_leaves_the_gpus_alone(rig: Rig) -> None:
    (rig.bin / "nvidia-smi").write_text("#!/usr/bin/env bash\necho \"nvidia-smi $*\" >> \"$FAKE_DIR/calls\"\nexit 9\n")
    out = rig.dir / "exported2"
    p = rig.run("--", "bash", "-c", f'printf "%s" "$BENCH_QUIET_ACTIVE" > "{out}"')
    assert p.returncode == 0 and "could not list the GPUs" in p.stderr
    assert out.read_text() == "timers:0,pm:0,clocks:0,refused:1"
    assert [c for c in rig.calls() if c.startswith("nvidia-smi")] == ["nvidia-smi --query-gpu=index --format=csv,noheader"]


def test_each_gpu_change_is_recorded_in_the_state_file_before_it_is_made(rig: Rig) -> None:
    p = rig.run("--", "true", FAKE_GPUS="2")
    assert p.returncode == 0, p.stderr
    assert sorted((rig.dir / "recorded_before_write").read_text().splitlines()) == ["lgc 0", "lgc 1", "pm 0", "pm 1"]
