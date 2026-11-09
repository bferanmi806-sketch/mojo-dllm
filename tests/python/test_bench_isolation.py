"""bench/isolation.py: a timed run proves it had the machine to itself, and a problem that did not is run again.

Every system reader goes through isolation.Machine. These tests give it a fake /proc and /sys tree
(conftest.FakeMachine), fake `nvidia-smi`, `systemctl` and `powerprofilesctl` scripts on a temporary PATH, and
real processes only where the point is that a real burner is seen. No test touches the real GPU or changes the
real power profile.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import eval_gsm8k  # noqa: E402
import isolation  # noqa: E402
import power  # noqa: E402
from conftest import FakeMachine  # noqa: E402

HOT = isolation.Thresholds()


@pytest.fixture(autouse=True)
def by_hand(fake_machine: FakeMachine) -> None:
    """The unit tests move the clock themselves."""
    fake_machine.auto = False


def pre(fm: FakeMachine, device: str = "cpu", need: float = 4.0, **kw: object) -> isolation.Report:
    fm.flush()
    return isolation.preflight(device, need, machine=fm, sample_s=2.0, **kw)  # type: ignore[arg-type]


def status(rep: isolation.Report, name: str) -> str:
    got = [c.status for c in rep.checks if c.name == name]
    assert got, f"no check named {name}: {[c.name for c in rep.checks]}"
    return got[0]


def detail(rep: isolation.Report, name: str) -> str:
    return next(c.detail for c in rep.checks if c.name == name)


def without_thermal(fm: FakeMachine) -> None:
    """A host with no thermal_throttle files (an AMD CPU, a VM), whatever the fake rewrites."""
    original = fm.flush

    def flush() -> None:
        original()
        for f in fm.sys.glob("devices/system/cpu/cpu*/thermal_throttle"):
            subprocess.run(["rm", "-rf", str(f)], check=True)
    fm.flush = flush                                         # type: ignore[method-assign]
    flush()


def clock_only(fm: FakeMachine) -> None:
    """Time passes but no /proc counter moves."""
    def tick(seconds: float) -> None:
        fm.now += seconds
    fm.sleep = tick
    fm.advance = tick                                        # type: ignore[method-assign]


def window(fm: FakeMachine, steps: int = 10, *, device: str = "cpu", each=None, pid: int = 4242,
           present: set[str] | None = None, th: isolation.Thresholds | None = None) -> dict:
    w = isolation.Window(pid=pid, device=device, machine=fm, thread=False, thresholds=th, present=present)
    with w:
        for i in range(steps):
            fm.advance(0.5)
            if each:
                each(i)
            w.tick()
    return w.result


# a quiet machine passes


def test_quiet_machine_passes_every_check(fake_machine: FakeMachine) -> None:
    rep = pre(fake_machine)
    assert rep.ok and [c for c in rep.checks if c.status == "FAIL"] == []
    assert {c.name for c in rep.checks if c.status == "PASS"} >= {
        "power", "profile", "governor", "cpu-busy", "psi-cpu", "swap-movement", "memory", "thermal"}
    assert "gpu-processes" not in {c.name for c in rep.checks}      # a cpu arm does not check the GPU


def test_quiet_gpu_machine_passes_the_gpu_checks(fake_machine: FakeMachine) -> None:
    rep = pre(fake_machine, "gpu")
    assert rep.ok
    assert {"gpu-processes", "gpu-memory", "gpu-utilization", "gpu-clock-reasons"} <= {
        c.name for c in rep.checks if c.status == "PASS"}


# every FAIL condition fires on a fake that violates only that condition


def test_fail_on_battery(fake_machine: FakeMachine) -> None:
    fake_machine.set_power(plugged=False)
    rep = pre(fake_machine)
    assert status(rep, "power") == "FAIL" and [c.name for c in rep.failed] == ["power"]


def test_on_battery_is_a_warn_when_battery_is_allowed(fake_machine: FakeMachine) -> None:
    fake_machine.set_power(plugged=False)
    assert status(pre(fake_machine, allow_battery=True), "power") == "WARN"


def test_fail_on_the_balanced_profile(fake_machine: FakeMachine) -> None:
    fake_machine.profile = fake_machine.platform_profile = "balanced"
    rep = pre(fake_machine)
    assert [c.name for c in rep.failed] == ["profile"] and "performance" in detail(rep, "profile")


def test_fail_when_only_one_profile_source_is_not_performance(fake_machine: FakeMachine) -> None:
    fake_machine.profile = "balanced"
    assert [c.name for c in pre(fake_machine).failed] == ["profile"]


def test_fail_on_powersave_governor_with_a_balanced_preference(fake_machine: FakeMachine) -> None:
    fake_machine.governor, fake_machine.epp = "powersave", "balance_performance"
    rep = pre(fake_machine)
    assert [c.name for c in rep.failed] == ["governor"]


def test_powersave_with_preference_performance_passes(fake_machine: FakeMachine) -> None:
    fake_machine.governor, fake_machine.epp = "powersave", "performance"
    assert status(pre(fake_machine), "governor") == "PASS"


def test_fail_on_busy_cpu_and_name_the_offender(fake_machine: FakeMachine) -> None:
    fake_machine.add_proc(777, 1, "omp", rate=120)         # 1.2 cores of 4 = 30%
    rep = pre(fake_machine)
    assert [c.name for c in rep.failed] == ["cpu-busy"]
    assert "omp (777)" in detail(rep, "cpu-busy") and rep.offenders[0]["pid"] == 777


def test_the_preflight_does_not_blame_itself(fake_machine: FakeMachine) -> None:
    fake_machine.procs[4242]["rate"] = 100                 # this process
    assert all(o["pid"] != 4242 for o in pre(fake_machine).offenders)


def test_fail_on_psi_avg10(fake_machine: FakeMachine) -> None:
    fake_machine.psi_avg10 = 6.5
    rep = pre(fake_machine)
    assert [c.name for c in rep.failed] == ["psi-cpu"] and "6.50" in detail(rep, "psi-cpu")


def test_fail_when_swap_pages_move_during_the_sample(fake_machine: FakeMachine) -> None:
    fake_machine.swap_rate = 50
    assert [c.name for c in pre(fake_machine).failed] == ["swap-movement"]


def test_fail_when_available_memory_is_below_the_need(fake_machine: FakeMachine) -> None:
    fake_machine.mem_available_kb = 3 * 1048576
    rep = pre(fake_machine, need=4.0)
    assert [c.name for c in rep.failed] == ["memory"] and "3.0 GB" in detail(rep, "memory")


def test_swap_in_use_is_warned_about(fake_machine: FakeMachine) -> None:
    fake_machine.swap_free_kb = 304
    rep = pre(fake_machine)
    assert rep.ok and status(rep, "swap-in-use") == "WARN" and "2047 of 2047" in detail(rep, "swap-in-use")


def test_fail_when_a_thermal_counter_rises_during_the_sample(fake_machine: FakeMachine) -> None:
    rep = isolation.preflight("cpu", 4.0, machine=fake_machine, sample_s=2.0)
    assert rep.ok
    fake_machine.sleep = lambda s: (fake_machine.advance(s), fake_machine.thermal.update(core=9))[0]
    assert [c.name for c in pre(fake_machine).failed] == ["thermal"]


def test_fail_on_a_gpu_compute_process(fake_machine: FakeMachine) -> None:
    fake_machine.gpu_apps = [(31337, 50)]
    rep = pre(fake_machine, "gpu")
    assert [c.name for c in rep.failed] == ["gpu-processes"] and "31337" in detail(rep, "gpu-processes")


def test_fail_on_gpu_memory(fake_machine: FakeMachine) -> None:
    fake_machine.gpu["mem"] = 450
    assert [c.name for c in pre(fake_machine, "gpu").failed] == ["gpu-memory"]


def test_fail_on_gpu_utilization(fake_machine: FakeMachine) -> None:
    fake_machine.gpu["util"] = 30
    assert [c.name for c in pre(fake_machine, "gpu").failed] == ["gpu-utilization"]


def test_fail_on_a_gpu_event_reason_other_than_idle(fake_machine: FakeMachine) -> None:
    fake_machine.gpu["reasons"] = 0x20 | 0x1
    rep = pre(fake_machine, "gpu")
    assert [c.name for c in rep.failed] == ["gpu-clock-reasons"] and "sw_thermal" in detail(rep, "gpu-clock-reasons")


def test_a_software_power_cap_alone_is_recorded_not_a_failure(fake_machine: FakeMachine) -> None:
    fake_machine.gpu["reasons"] = 0x4 | 0x1                  # intrinsic on a 30 W laptop GPU
    assert pre(fake_machine, "gpu").ok


def test_a_display_clock_setting_alone_is_not_a_failure(fake_machine: FakeMachine) -> None:
    fake_machine.gpu["reasons"] = 0x1 | 0x100
    assert pre(fake_machine, "gpu").ok


def test_allow_turns_a_named_fail_into_a_warn(fake_machine: FakeMachine) -> None:
    fake_machine.profile = fake_machine.platform_profile = "balanced"
    rep = pre(fake_machine, allow=("profile",))
    assert rep.ok and status(rep, "profile") == "WARN" and "allowed" in detail(rep, "profile")


# fail closed: a missing or empty source is a WARN naming it, never a PASS


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_every_source_missing_yields_warns_and_no_pass(tmp_path: Path, device: str) -> None:
    (tmp_path / "nobin").mkdir()
    empty = isolation.Machine(proc=tmp_path / "noproc", sysfs=tmp_path / "nosys", power_root=tmp_path / "nopower",
                              env={"PATH": str(tmp_path / "nobin")}, sleep=lambda s: None)
    rep = isolation.preflight(device, 4.0, machine=empty, sample_s=0.0)
    assert [c.status for c in rep.checks if c.status == "PASS"] == [], rep.text()
    assert rep.checks and all(c.status in ("WARN", "FAIL") for c in rep.checks)
    assert "stat" in detail(rep, "cpu-busy") and "pressure/cpu" in detail(rep, "psi-cpu")


def test_empty_files_are_not_a_pass(fake_machine: FakeMachine) -> None:
    for name in ("stat", "pressure/cpu", "vmstat", "meminfo"):
        (fake_machine.proc / name).write_text("")
    fake_machine.flush = lambda: None                       # keep the empty files through the sample
    rep = pre(fake_machine)
    for name in ("cpu-busy", "psi-cpu", "swap-movement", "memory"):
        assert status(rep, name) == "WARN", (name, rep.text())


def test_a_sample_in_which_no_ticks_elapse_is_not_a_pass(fake_machine: FakeMachine) -> None:
    clock_only(fake_machine)
    rep = isolation.preflight("cpu", 4.0, machine=fake_machine, sample_s=2.0)
    assert status(rep, "cpu-busy") == "WARN" and "no CPU ticks" in detail(rep, "cpu-busy")


def test_nvidia_smi_printing_nothing_is_not_a_pass(fake_machine: FakeMachine) -> None:
    fake_machine.run = lambda cmd, timeout=15.0: (0, "")      # type: ignore[assignment]
    rep = pre(fake_machine, "gpu")
    assert status(rep, "gpu") == "WARN" and not any(c.name.startswith("gpu-") and c.status == "PASS"
                                                    for c in rep.checks)


def test_unreadable_power_supply_is_a_warn(fake_machine: FakeMachine) -> None:
    (fake_machine.root / "power/ADP1/online").unlink()
    (fake_machine.root / "power/ADP1/online").mkdir()          # a directory where a file should be
    assert status(pre(fake_machine), "power") == "WARN"


# snapshot: every row of the measured table is read and recorded


def test_snapshot_records_every_source(fake_machine: FakeMachine) -> None:
    fake_machine.add_proc(900, 1, "python3", rate=100)
    fake_machine.swap_free_kb = 304
    fake_machine.gpu_apps = [(900, 120)]
    fake_machine.flush()
    fake_machine.timers = [("sysstat-collect.timer", 600), ("man-db.timer", 7200), ("apt-daily.timer", 3000)]
    s = isolation.snapshot(fake_machine, sample_s=2.0)
    assert s["governors"] == ["performance"] and s["intel_pstate_status"] == "active" and s["no_turbo"] == "0"
    assert s["energy_performance_preference"] == ["performance"] and s["platform_profile"] == "performance"
    assert s["powerprofilesctl"] == "performance" and s["cpu_freq_mhz"] == {"min": 3400, "max": 3400}
    assert s["thermal_throttle"] == {"core": 3, "package": 5} and s["loadavg"] == [0.1, 0.2, 0.3]
    assert s["psi"]["cpu"]["avg10"] == 0.0 and s["psi"]["memory"]["total"] == 1.0
    assert s["swap"] == {"total_mb": 2047, "used_mb": 2047} and s["mem_available_mb"] == 16384
    assert s["swap_pages"] == {"pswpin": 100, "pswpout": 200}
    assert s["top_cpu_processes"][0]["name"] == "python3" and s["top_cpu_processes"][0]["percent"] == 100.0
    g = s["gpu"]
    assert g["reasons"] == 1 and g["reason_names"] == ["idle"] and g["memory_used_mib"] == 120
    assert g["sm_clock_mhz"] == 210 and g["max_sm_clock_mhz"] == 2100 and g["persistence_mode"] == "Disabled"
    assert g["compute_mode"] == "Default" and g["power_draw_w"] == 8.5 and g["utilization_percent"] == 0
    assert g["temperature_c"] == 45 and g["compute_apps"] == [{"pid": 900, "memory_mib": 120.0}]
    assert s["timers_due_within_1h"] == ["apt-daily.timer", "sysstat-collect.timer"]      # names only, 2 h away left out
    assert s["user_crontab_lines"] == 1 and s["unavailable"] == []


def test_snapshot_names_the_sources_it_could_not_read(tmp_path: Path) -> None:
    (tmp_path / "nobin").mkdir()
    bare = isolation.Machine(proc=tmp_path / "noproc", sysfs=tmp_path / "nosys", power_root=tmp_path / "nopower",
                             env={"PATH": str(tmp_path / "nobin")})
    s = isolation.snapshot(bare, sample_s=0)
    assert {"cpu-governor", "thermal", "meminfo", "vmstat", "psi-cpu", "loadavg", "power-profile"} <= set(s["unavailable"])
    assert s["gpu"] is None


# the real command runner against fake scripts on a temporary PATH


def _default_signals() -> None:
    """A job started from a background shell or under nohup inherits SIGINT or SIGHUP ignored; the child under
    test must start with the default dispositions."""
    for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_DFL)


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    return path


FAKE_NVIDIA = """#!/usr/bin/env bash
# fake nvidia-smi: an old driver that only knows clocks_throttle_reasons.active
case "$1" in
  --query-compute-apps=*) printf '4001, 300\\n4002, 20\\n' ;;
  --query-gpu=*)
    case "$1" in
      *clocks_event_reasons*) echo "Field \\"clocks_event_reasons.active\\" is not a valid field to query." >&2; exit 2 ;;
    esac
    printf '0x0000000000000044, 97, 5100, 71, 1200, 2100, Enabled, Default, 62.5\\n' ;;
  *) exit 3 ;;
esac
"""
FAKE_PPD = """#!/usr/bin/env bash
state="${FAKE_PPD_STATE:?}"
case "$1" in
  get) cat "$state" ;;
  set) printf '%s\\n' "$2" > "$state" ;;
  version) echo "0.30" ;;
  *) exit 1 ;;
esac
"""


def test_gpu_reader_falls_back_to_the_old_reasons_field(tmp_path: Path) -> None:
    _script(tmp_path / "bin/nvidia-smi", FAKE_NVIDIA)
    m = isolation.Machine(env={"PATH": f"{tmp_path / 'bin'}{os.pathsep}/usr/bin:/bin"})
    row = isolation.read_gpu(m)
    assert row is not None and row["reasons"] == 0x44 and row["utilization.gpu"] == "97"
    assert isolation.contended_reasons(row["reasons"]) == ["hw_thermal"]            # the software power cap is recorded elsewhere, not contended
    assert isolation.read_gpu_apps(m) == [(4001, 300.0), (4002, 20.0)]


def test_an_empty_file_is_no_reading(tmp_path: Path) -> None:
    m = isolation.Machine()
    f = tmp_path / "f"
    f.write_text("")
    assert m.read(f) is None
    f.write_text(" \n")
    assert m.read(f) is None
    f.write_text("0\n")
    assert m.read(f) == "0\n"                                  # a zero is a reading
    assert m.read(tmp_path / "missing") is None


def test_machine_run_reports_a_missing_program_and_a_timeout(tmp_path: Path) -> None:
    _script(tmp_path / "bin/slow", "#!/usr/bin/env bash\nexec sleep 30\n")
    m = isolation.Machine(env={"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin"})
    assert m.run(["definitely-not-installed-xyz"]) == (127, "")
    assert m.run(["slow"], timeout=0.3) == (124, "")
    assert not m.has("definitely-not-installed-xyz") and m.has("slow")


def test_nvidia_smi_that_errors_on_both_names_is_unavailable(tmp_path: Path) -> None:
    _script(tmp_path / "bin/nvidia-smi", "#!/usr/bin/env bash\necho broken >&2\nexit 9\n")
    m = isolation.Machine(env={"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin"})
    assert isolation.read_gpu(m) is None and isolation.read_gpu_apps(m) is None


# Window: the child's own CPU is not contention, other processes' CPU is


def test_the_childs_own_cpu_is_not_counted_as_other(fake_machine: FakeMachine) -> None:
    fake_machine.add_proc(5000, 4242, "llama-diffusion-cli", rate=100)          # one full core of four
    fake_machine.add_proc(5001, 5000, "worker", rate=100)                       # a grandchild, another core
    res = window(fake_machine)
    assert res["clean"] is True, res
    assert res["other_cpu_share_max"] == pytest.approx(0.0) and res["other_cpu_share_mean"] == pytest.approx(0.0)


def test_another_process_using_cpu_flags_the_window(fake_machine: FakeMachine) -> None:
    fake_machine.add_proc(5000, 4242, "child", rate=100)
    fake_machine.add_proc(6000, 1, "burner", rate=400)                           # every core, outside the tree
    res = window(fake_machine, steps=4)
    assert res["clean"] is False and res["other_cpu_share_mean"] > 0.05 and res["other_cpu_share_max"] > 0.25
    assert any(r.startswith("other-cpu-mean") for r in res["reasons"])
    assert res["other_top_processes"][0] == {"pid": 6000, "name": "burner", "percent": 400.0}      # named, not just counted
    assert all(o["pid"] not in (4242, 5000) for o in res["other_top_processes"])                  # the tree is not blamed
    assert 0.74 < res["other_cpu_share_mean"] < 0.76            # (400 / 500 busy ticks) x 4 cores x 25 ticks... exact: 3 of 4 cores


def test_a_sustained_small_load_trips_the_mean_but_not_the_peak(fake_machine: FakeMachine) -> None:
    fake_machine.extra_busy = 40                                                  # 10% of capacity in every sample
    res = window(fake_machine)
    assert res["clean"] is False
    assert [r.split()[0] for r in res["reasons"]] == ["other-cpu-mean"]
    assert res["other_cpu_share_max"] == pytest.approx(0.10)


def test_a_short_spike_trips_the_peak_but_not_the_mean(fake_machine: FakeMachine) -> None:
    def spike(i: int) -> None:
        fake_machine.extra_busy = 140 if i == 4 else 0
    fake_machine.extra_busy = 0
    # the spike has to be set before the step that advances, so set it ahead of each advance
    w = isolation.Window(pid=4242, machine=fake_machine, thread=False)
    with w:
        for i in range(10):
            fake_machine.extra_busy = 140 if i == 4 else 0
            fake_machine.advance(0.5)
            w.tick()
    res = w.result
    assert res["clean"] is False and [r.split()[0] for r in res["reasons"]] == ["other-cpu-peak"]
    assert res["other_cpu_share_max"] == pytest.approx(0.35) and res["other_cpu_share_mean"] < 0.05


def test_psi_stall_is_recorded_but_never_makes_the_window_noisy(fake_machine: FakeMachine) -> None:
    """A 12-thread run stalls 9% of the time by itself (measured), so PSI cannot tell contention from the run."""
    fake_machine.psi_rate = 0.30                              # 30% of the window: far above any old limit
    res = window(fake_machine)
    assert res["clean"] is True, res
    assert res["psi_cpu_stall_ms"] == pytest.approx(1500.0) and res["reasons"] == []


def test_psi_memory_stall_is_recorded_only(fake_machine: FakeMachine) -> None:
    fake_machine.psi_mem_rate = 0.2
    res = window(fake_machine)
    assert res["clean"] is True and res["psi_memory_stall_ms"] == pytest.approx(1000.0)


def test_the_quiet_gate_does_not_wait_on_pressure_averages(fake_machine: FakeMachine) -> None:
    fake_machine.psi_avg10 = 9.0                              # the avg10 decays for seconds after a 12-thread problem
    g = isolation.quiet_gate("cpu", machine=fake_machine, sample_s=1.0, max_wait_s=0)
    assert g["ok"] is True and g["reasons"] == []


def test_one_swapped_page_flags_the_window(fake_machine: FakeMachine) -> None:
    res = window(fake_machine, each=lambda i: setattr(fake_machine, "pswpin", fake_machine.pswpin + (1 if i == 3 else 0))
                 or fake_machine.flush())
    assert res["clean"] is False and res["swap_pages"] == 1 and res["reasons"] == ["swap 1 pages"]


def test_a_thermal_counter_step_flags_the_window(fake_machine: FakeMachine) -> None:
    res = window(fake_machine, each=lambda i: fake_machine.thermal.update(package=5 + (1 if i >= 5 else 0))
                 or fake_machine.flush())
    assert res["clean"] is False and res["thermal_events"] == 1 and res["reasons"] == ["thermal 1 throttle events"]


def test_gpu_reason_other_than_idle_flags_and_is_recorded(fake_machine: FakeMachine) -> None:
    def power_cap(i: int) -> None:
        fake_machine.gpu["reasons"] = 0x21 if i == 7 else 0x1       # reading 8 is a GPU sample (every 2 s)
        fake_machine.gpu["sm"] = 900 if i == 7 else 1900
    res = window(fake_machine, device="gpu", each=power_cap)
    assert res["clean"] is False and res["gpu_reasons_seen"] == ["sw_thermal"] and res["gpu_power_capped"] is False
    assert res["sm_clock_min_mhz"] == 210 or res["sm_clock_min_mhz"] == 900      # baseline reading is the idle clock
    assert res["sm_clock_max_mhz"] == 1900


def test_gpu_display_clock_is_recorded_but_does_not_flag(fake_machine: FakeMachine) -> None:
    res = window(fake_machine, device="gpu", each=lambda i: fake_machine.gpu.update(reasons=0x101))
    assert res["clean"] is True and res["gpu_reasons_seen"] == ["display_clock"]


def test_a_foreign_gpu_pid_flags_but_one_in_the_tree_does_not(fake_machine: FakeMachine) -> None:
    fake_machine.add_proc(5000, 4242, "child", rate=100)
    fake_machine.gpu_apps = [(5000, 2000)]
    assert window(fake_machine, device="gpu")["clean"] is True
    fake_machine.gpu_apps = [(5000, 2000), (31337, 300)]
    res = window(fake_machine, device="gpu")
    assert res["clean"] is False and res["gpu_foreign_pids"] == [31337] and "gpu-foreign-pids [31337]" in res["reasons"]


def test_result_has_every_documented_key(fake_machine: FakeMachine) -> None:
    res = window(fake_machine, device="gpu")
    assert {"clean", "reasons", "other_cpu_share_max", "other_cpu_share_mean", "psi_cpu_stall_ms", "swap_pages",
            "thermal_events", "gpu_reasons_seen", "gpu_foreign_pids", "sm_clock_min_mhz", "sm_clock_max_mhz",
            "window_s", "other_top_processes"} <= set(res)
    assert res["window_s"] == pytest.approx(5.0) and res["clean"] is True


# fail closed in the window


def test_a_dead_sampler_makes_the_window_not_clean(fake_machine: FakeMachine) -> None:
    fake_machine.auto = True
    calls = {"n": 0}
    original = fake_machine.read

    def boom(path: Path) -> str | None:
        calls["n"] += 1
        if calls["n"] > 40:
            raise RuntimeError("sampler fell over")
        return original(path)

    fake_machine.read = boom                                 # type: ignore[method-assign]
    before = threading.active_count()
    w = isolation.Window(pid=4242, machine=fake_machine, sample_s=0.01)
    with w:
        time.sleep(0.4)
    assert w.result["clean"] is False and "sampler-failed" in w.result["reasons"]
    assert threading.active_count() == before and not any(t.name == "isolation-sampler" for t in threading.enumerate())


def test_a_stuck_sampler_is_joined_with_a_timeout_and_reported(fake_machine: FakeMachine) -> None:
    fake_machine.auto = True
    gate = threading.Event()
    original = fake_machine.read
    state = {"n": 0}

    def stall(path: Path) -> str | None:
        state["n"] += 1
        if state["n"] > 12 and threading.current_thread().name == "isolation-sampler":
            gate.wait(10)
        return original(path)

    fake_machine.read = stall                                # type: ignore[method-assign]
    w = isolation.Window(pid=4242, machine=fake_machine, sample_s=0.01, join_timeout_s=0.3)
    started = time.monotonic()
    with w:
        time.sleep(0.1)
    took = time.monotonic() - started
    gate.set()
    assert took < 3 and w.result["clean"] is False and "sampler-failed" in w.result["reasons"]
    for t in threading.enumerate():
        if t.name == "isolation-sampler":
            t.join(2)
    assert not any(t.name == "isolation-sampler" for t in threading.enumerate())


def test_a_source_that_goes_unreadable_mid_window_cannot_claim_clean(fake_machine: FakeMachine) -> None:
    def lose(i: int) -> None:
        if i == 5:
            for f in fake_machine.sys.glob("devices/system/cpu/cpu*/thermal_throttle/*"):
                f.unlink()
    fake_machine.flush = lambda: None if fake_machine.now > 2.4 else type(fake_machine).flush(fake_machine)  # type: ignore
    res = window(fake_machine, each=lose)
    assert res["clean"] is False and "unmeasured:thermal" in res["reasons"] and "thermal" in res["unavailable"]


def test_a_source_absent_on_the_host_is_not_checked(tmp_path: Path) -> None:
    no_thermal = FakeMachine(tmp_path / "amd", auto=False)
    without_thermal(no_thermal)
    res = window(no_thermal)
    assert "thermal" not in isolation.present_sources(no_thermal, "cpu")
    assert res["clean"] is True and res["thermal_events"] is None and "thermal" in res["unavailable"]


def test_a_source_present_in_the_preflight_but_gone_now_is_unmeasured(tmp_path: Path) -> None:
    gone = FakeMachine(tmp_path / "g", auto=False)
    without_thermal(gone)
    res = window(gone, present={"thermal", "cpu-stat", "psi-cpu", "vmstat", "proc-tree"})
    assert res["clean"] is False and "unmeasured:thermal" in res["reasons"]


def test_no_cpu_ticks_elapsing_is_unmeasured_not_zero(fake_machine: FakeMachine) -> None:
    clock_only(fake_machine)
    res = window(fake_machine, steps=3)
    assert res["clean"] is False and "unmeasured:cpu-stat" in res["reasons"] and res["other_cpu_share_mean"] is None


def test_a_gpu_arm_without_a_readable_gpu_is_unmeasured(fake_machine: FakeMachine) -> None:
    fake_machine.gpu_ok = False
    res = window(fake_machine, device="gpu")
    assert res["clean"] is False and "unmeasured:gpu" in res["reasons"]


def test_a_missing_root_pid_is_unmeasured(fake_machine: FakeMachine) -> None:
    res = window(fake_machine, pid=999999)
    assert res["clean"] is False and "unmeasured:proc-tree" in res["reasons"] or "proc-tree" in res["unavailable"]


def test_the_thread_is_stopped_when_the_body_raises(fake_machine: FakeMachine) -> None:
    fake_machine.auto = True
    with pytest.raises(ValueError):
        with isolation.Window(pid=4242, machine=fake_machine, sample_s=0.01):
            raise ValueError("problem failed")
    assert not any(t.name == "isolation-sampler" for t in threading.enumerate())


# a real burner, seen through the real /proc


BURN = "import time\nend = time.monotonic() + 1.2\nwhile time.monotonic() < end:\n    pass\n"


def test_a_real_burner_on_every_cpu_is_flagged(tmp_path: Path) -> None:
    ncpu = os.cpu_count() or 1
    real = isolation.Machine(sysfs=tmp_path / "nosys", power_root=tmp_path / "nopower")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2.5)"])
    try:
        w = isolation.Window(pid=child.pid, machine=real, sample_s=0.25)
        with w:
            time.sleep(0.5)
            burners = [subprocess.Popen([sys.executable, "-c", BURN]) for _ in range(ncpu)]
            for b in burners:
                b.wait()
            time.sleep(0.3)
    finally:
        child.kill()
        child.wait()
    assert w.result["clean"] is False, w.result
    assert w.result["other_cpu_share_max"] > 0.05 and any(r.startswith("other-cpu") for r in w.result["reasons"])


# the quiet gate


def test_gate_passes_at_once_on_a_quiet_machine(fake_machine: FakeMachine) -> None:
    g = isolation.quiet_gate("cpu", machine=fake_machine, sample_s=1.0)
    assert g["ok"] is True and g["waited_s"] == 1.0 and g["cpu_busy_before"] == 0.0 and g["reasons"] == []   # one sample


def test_gate_waits_while_the_machine_is_busy_then_runs(fake_machine: FakeMachine) -> None:
    fake_machine.extra_busy = 200
    real_sleep = fake_machine.sleep
    state = {"n": 0}

    def sleeper(s: float) -> None:
        state["n"] += 1
        if state["n"] == 6:
            fake_machine.extra_busy = 0
        real_sleep(s)

    fake_machine.sleep = sleeper
    g = isolation.quiet_gate("cpu", machine=fake_machine, sample_s=1.0, poll_s=3.0, max_wait_s=60)
    assert g["ok"] is True and g["waited_s"] > 0 and g["cpu_busy_before"] == 0.0


def test_gate_gives_up_after_the_maximum_and_says_why(fake_machine: FakeMachine) -> None:
    fake_machine.extra_busy = 200
    g = isolation.quiet_gate("cpu", machine=fake_machine, sample_s=1.0, poll_s=3.0, max_wait_s=10)
    assert g["ok"] is False and g["waited_s"] >= 10 and g["cpu_busy_before"] == pytest.approx(0.5)
    assert g["reasons"][0].startswith("cpu-busy")


def test_gate_treats_unreadable_cpu_stat_as_busy(fake_machine: FakeMachine) -> None:
    clock_only(fake_machine)
    g = isolation.quiet_gate("cpu", machine=fake_machine, sample_s=1.0, max_wait_s=0)
    assert g["ok"] is False and g["cpu_busy_before"] == 1.0


def test_gate_sees_a_gpu_process_on_a_gpu_arm(fake_machine: FakeMachine) -> None:
    fake_machine.gpu_apps = [(4000, 10)]
    g = isolation.quiet_gate("gpu", machine=fake_machine, sample_s=1.0, max_wait_s=0)
    assert g["ok"] is False and any("gpu processes" in r for r in g["reasons"])


def test_legacy_wait_for_quiet_keeps_its_meaning(fake_machine: FakeMachine) -> None:
    busy, waited = isolation.wait_for_quiet(0.1, 300, machine=fake_machine)
    assert busy == 0.0 and waited == 2.0                       # one sample, as before: the sample counts as waiting
    fake_machine.extra_busy = 100                             # 25% busy
    busy, waited = isolation.wait_for_quiet(0.1, 20, machine=fake_machine)
    assert busy == pytest.approx(0.25) and waited > 20        # gave up after max_wait, as run_bench did
    assert isolation.cpu_busy(1.0, machine=fake_machine) == pytest.approx(0.25)


# run_isolated: a noisy problem is run again, the retry count is recorded, the row is kept after the retries


class Scripted:
    """A stand-in Window that reports scripted results, one per entry."""

    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.opened = 0

    def __call__(self, **kw: object) -> "Scripted":
        self.kw = kw
        return self

    def __enter__(self) -> "Scripted":
        self.result = self.results[min(self.opened, len(self.results) - 1)]
        self.opened += 1
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def win(clean: bool, *reasons: str) -> dict:
    return {"clean": clean, "reasons": list(reasons), "other_cpu_share_max": 0.0}


def test_a_noisy_problem_is_run_again_and_the_retry_is_recorded(fake_machine: FakeMachine) -> None:
    calls: list[int] = []
    script = Scripted([win(False, "other-cpu-mean 9%"), win(True)])
    out, info = isolation.run_isolated(lambda: calls.append(1) or len(calls), device="cpu", machine=fake_machine,
                                       window=script, allow_battery=True, log=lambda m: None, quiet_sample_s=0.1)
    assert out == 2 and calls == [1, 1]
    assert info["isolation"]["clean"] is True and info["isolation"]["retries"] == 1 and info["isolation"]["noisy"] is False


def test_after_the_retries_the_noisy_row_is_kept_with_its_reasons(fake_machine: FakeMachine) -> None:
    calls: list[int] = []
    script = Scripted([win(False, "swap 3 pages")])
    out, info = isolation.run_isolated(lambda: calls.append(1) or "answer", device="cpu", machine=fake_machine,
                                       window=script, retries=2, allow_battery=True, log=lambda m: None,
                                       quiet_sample_s=0.1)
    assert out == "answer" and len(calls) == 3                 # first run and two repeats
    assert info["isolation"]["clean"] is False and info["isolation"]["noisy"] is True
    assert info["isolation"]["retries"] == 2 and info["isolation"]["reasons"] == ["swap 3 pages"]


def test_zero_retries_runs_once(fake_machine: FakeMachine) -> None:
    calls: list[int] = []
    isolation.run_isolated(lambda: calls.append(1), device="cpu", machine=fake_machine, window=Scripted([win(False, "x")]),
                           retries=0, allow_battery=True, log=lambda m: None, quiet_sample_s=0.1)
    assert calls == [1]


def test_a_clean_problem_is_run_once_and_keeps_the_power_keys(fake_machine: FakeMachine) -> None:
    out, info = isolation.run_isolated(lambda: "x", device="cpu", machine=fake_machine, window=Scripted([win(True)]),
                                       allow_battery=True, log=lambda m: None, quiet_sample_s=0.1)
    assert out == "x" and info["isolation"]["retries"] == 0
    assert {"power_retries", "on_battery", "power_source", "isolation"} <= set(info)
    assert {"cpu_busy_before", "waited_s"} <= set(info["isolation"])


def test_a_failing_problem_propagates_and_leaves_no_thread(fake_machine: FakeMachine) -> None:
    fake_machine.auto = True

    def boom() -> None:
        raise RuntimeError("engine died")

    with pytest.raises(RuntimeError, match="engine died"):
        isolation.run_isolated(boom, device="cpu", machine=fake_machine, allow_battery=True, log=lambda m: None,
                               quiet_sample_s=0.1, window_sample_s=0.01)
    assert not any(t.name == "isolation-sampler" for t in threading.enumerate())


def test_run_isolated_with_the_real_window_on_the_quiet_fake_machine(fake_machine: FakeMachine) -> None:
    fake_machine.auto = True
    out, info = isolation.run_isolated(lambda: 7, device="cpu", machine=fake_machine, allow_battery=True,
                                       log=lambda m: None, quiet_sample_s=0.1)
    assert out == 7 and info["isolation"]["clean"] is True, info


def test_retries_must_not_be_negative(fake_machine: FakeMachine) -> None:
    with pytest.raises(ValueError):
        isolation.run_isolated(lambda: 1, device="cpu", retries=-1, machine=fake_machine)


# the arm: preflight once, ARM.env.json, FAIL stops unless --allow-noisy


def test_begin_arm_writes_the_env_file(fake_machine: FakeMachine, tmp_path: Path) -> None:
    logs: list[str] = []
    arm = isolation.begin_arm("a1", device="gpu", need_ram_gb=4.0, runs_dir=tmp_path / "runs", machine=fake_machine,
                              sample_s=2.0, versions={"mojo-dllm": "abc1234"}, log=logs.append)
    doc = json.loads((tmp_path / "runs/a1.env.json").read_text())
    s = doc["sessions"][0]
    assert doc["arm"] == "a1" and s["preflight"]["ok"] is True and s["snapshot_start"]["governors"] == ["performance"]
    assert s["thresholds"]["other_cpu_mean_max"] == 0.05 and s["tool_versions"]["mojo-dllm"] == "abc1234"
    assert s["tool_versions"]["nvidia_driver"] == "595.99.02" and s["preflight_failed_allowed"] is False
    assert arm.report.present >= {"cpu-stat", "psi-cpu", "gpu"} and "preflight (gpu" in logs[0]
    arm.finish()
    assert "snapshot_end" in json.loads((tmp_path / "runs/a1.env.json").read_text())["sessions"][0]


def test_the_env_file_records_whether_bench_quiet_ran(fake_machine: FakeMachine, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    for value, want in (("1", True), (None, False)):
        if value is None:
            monkeypatch.delenv("BENCH_QUIET_ACTIVE", raising=False)
        else:
            monkeypatch.setenv("BENCH_QUIET_ACTIVE", value)
        isolation.begin_arm(f"q{want}", device="cpu", need_ram_gb=1.0, runs_dir=tmp_path, machine=fake_machine,
                            sample_s=1.0, log=lambda m: None)
        assert json.loads((tmp_path / f"q{want}.env.json").read_text())["sessions"][0]["bench_quiet"] is want


def test_a_failed_preflight_stops_the_arm_but_records_why(fake_machine: FakeMachine, tmp_path: Path) -> None:
    fake_machine.profile = fake_machine.platform_profile = "balanced"
    with pytest.raises(isolation.IsolationError) as e:
        isolation.begin_arm("a2", device="cpu", need_ram_gb=1.0, runs_dir=tmp_path, machine=fake_machine,
                            sample_s=1.0, log=lambda m: None)
    assert "profile" in str(e.value) and "--allow-noisy" in str(e.value) and "--tune" in str(e.value)
    assert json.loads((tmp_path / "a2.env.json").read_text())["sessions"][0]["preflight"]["ok"] is False


def test_allow_noisy_continues_and_records_it(fake_machine: FakeMachine, tmp_path: Path) -> None:
    fake_machine.profile = fake_machine.platform_profile = "balanced"
    arm = isolation.begin_arm("a3", device="cpu", need_ram_gb=1.0, runs_dir=tmp_path, machine=fake_machine,
                              sample_s=1.0, allow_noisy=True, log=lambda m: None)
    s = json.loads((tmp_path / "a3.env.json").read_text())["sessions"][0]
    assert arm.report.failed and s["preflight_failed_allowed"] is True and s["allow_noisy"] is True


def test_explicit_thresholds_are_used_and_recorded(fake_machine: FakeMachine, tmp_path: Path) -> None:
    fake_machine.extra_busy = 60                              # 15% busy: over the default 10%, under 20%
    assert not pre(fake_machine).ok
    looser = isolation.Thresholds(cpu_busy_max=0.20)
    isolation.begin_arm("a4", device="cpu", need_ram_gb=1.0, runs_dir=tmp_path, machine=fake_machine, sample_s=1.0,
                        thresholds=looser, log=lambda m: None)
    assert json.loads((tmp_path / "a4.env.json").read_text())["sessions"][0]["thresholds"]["cpu_busy_max"] == 0.20


def test_a_resumed_arm_appends_a_session(fake_machine: FakeMachine, tmp_path: Path) -> None:
    for _ in range(2):
        isolation.begin_arm("a5", device="cpu", need_ram_gb=1.0, runs_dir=tmp_path, machine=fake_machine,
                            sample_s=1.0, log=lambda m: None)
    assert len(json.loads((tmp_path / "a5.env.json").read_text())["sessions"]) == 2


def test_thresholds_are_the_documented_constants() -> None:
    t = isolation.Thresholds()
    assert (t.cpu_busy_max, t.psi_avg10_max, t.gpu_mem_used_max_mib, t.gpu_util_max) == (0.10, 5.0, 300, 5.0)
    assert (t.other_cpu_mean_max, t.other_cpu_sample_max) == (0.05, 0.25)
    assert (t.swap_pages_max, t.thermal_events_max, t.foreign_gpu_pids_max) == (0, 0, 0)
    assert isolation.PREFLIGHT_SAMPLE_S == 5.0 and isolation.WINDOW_SAMPLE_S == 0.5


def test_the_docs_state_the_threshold_constants() -> None:
    doc = (ROOT / "docs" / "benchmarks.md").read_text()
    section = doc[doc.index("## Benchmark environment"):doc.index("## Steps and quality")]
    for phrase in (f"more than {isolation.PREFLIGHT_CPU_BUSY_MAX:.0%} of all CPUs", f"avg10 above {isolation.PREFLIGHT_PSI_AVG10_MAX:g}%",
                   f"more than {isolation.GPU_MEM_USED_MAX_MIB} MiB", f"above {isolation.GPU_UTIL_MAX:g}%",
                   f"more than {isolation.OTHER_CPU_MEAN_MAX:.0%} of the machine", f"exceeds {isolation.OTHER_CPU_SAMPLE_MAX:.0%}",
                   f"{isolation.PREFLIGHT_SAMPLE_S:g} second sample",
                   f"every {isolation.WINDOW_SAMPLE_S:g} s", f"{isolation.RAM_FACTOR} times the model file"):
        assert phrase in section, phrase
    for timer in isolation.TIMERS:
        assert f"`{timer}`" in section, timer
        assert timer in (ROOT / "scripts" / "bench-quiet.sh").read_text()


# --tune: the profile is restored on every way out


def _ppd(tmp_path: Path, start: str = "balanced") -> tuple[Path, dict[str, str]]:
    state = tmp_path / "profile"
    state.write_text(start + "\n")
    _script(tmp_path / "bin/powerprofilesctl", FAKE_PPD)
    return state, {"PATH": f"{tmp_path / 'bin'}{os.pathsep}/usr/bin:/bin", "FAKE_PPD_STATE": str(state)}


def test_tune_sets_performance_and_restores(tmp_path: Path) -> None:
    state, env = _ppd(tmp_path)
    m = isolation.Machine(env=env)
    with isolation.tuned(m, log=lambda s: None) as info:
        assert state.read_text().strip() == "performance" and info["previous"] == "balanced"
    assert state.read_text().strip() == "balanced"


def test_tune_restores_on_an_exception(tmp_path: Path) -> None:
    state, env = _ppd(tmp_path, "power-saver")
    with pytest.raises(RuntimeError):
        with isolation.tuned(isolation.Machine(env=env), log=lambda s: None):
            assert state.read_text().strip() == "performance"
            raise RuntimeError("the run failed")
    assert state.read_text().strip() == "power-saver"


def test_tune_does_not_restore_twice(tmp_path: Path) -> None:
    state, env = _ppd(tmp_path)
    m = isolation.Machine(env=env)
    log: list[str] = []
    with isolation.tuned(m, log=log.append):
        pass
    state.write_text("performance\n")                         # someone else changed it afterwards
    assert state.read_text().strip() == "performance" and sum("restored" in l for l in log) == 1


def test_tune_refuses_when_the_profile_does_not_take_and_puts_it_back(fake_machine: FakeMachine) -> None:
    fake_machine.profile, fake_machine.profile_set_works = "balanced", False
    with pytest.raises(isolation.IsolationError, match="did not take"):
        with isolation.tuned(fake_machine, log=lambda s: None):
            pytest.fail("must not run on an unverified profile")
    assert fake_machine.profile == "balanced"
    assert ["powerprofilesctl", "set", "balanced"] in fake_machine.calls


def test_tune_without_powerprofilesctl_is_an_error(fake_machine: FakeMachine) -> None:
    fake_machine.programs.discard("powerprofilesctl")
    with pytest.raises(isolation.IsolationError, match="powerprofilesctl"):
        with isolation.tuned(fake_machine, log=lambda s: None):
            pytest.fail("must not run")


def test_tune_fails_when_the_current_profile_is_unreadable(fake_machine: FakeMachine) -> None:
    orig = fake_machine.run
    fake_machine.run = lambda cmd, timeout=15.0: (1, "") if cmd[:2] == ["powerprofilesctl", "get"] else orig(cmd, timeout)  # type: ignore
    with pytest.raises(isolation.IsolationError, match="unknown"):
        with isolation.tuned(fake_machine, log=lambda s: None):
            pytest.fail("must not run")
    assert not any(c[:2] == ["powerprofilesctl", "set"] for c in fake_machine.calls)


TUNE_CHILD = textwrap.dedent("""
    import os, sys, time
    sys.path.insert(0, {bench!r})
    import isolation
    m = isolation.Machine(env=dict(os.environ))
    mode = sys.argv[1]
    cm = isolation.tuned(m, log=lambda s: None)
    cm.__enter__()
    print("ready", flush=True)
    if mode == "exit":
        sys.exit(0)                      # leaves the context open: only atexit can restore
    time.sleep(30)
""")


def _tune_child(tmp_path: Path, mode: str) -> tuple[subprocess.Popen, Path]:
    state, env = _ppd(tmp_path)
    script = tmp_path / "child.py"
    script.write_text(TUNE_CHILD.format(bench=str(ROOT / "bench")))
    p = subprocess.Popen([sys.executable, str(script), mode], env={**os.environ, **env}, stdout=subprocess.PIPE,
                         text=True, preexec_fn=_default_signals)
    assert p.stdout is not None and p.stdout.readline().strip() == "ready"
    assert state.read_text().strip() == "performance"
    return p, state


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_tune_restores_on_a_signal_and_the_signal_still_stops_the_run(tmp_path: Path, sig: int) -> None:
    p, state = _tune_child(tmp_path, "sleep")
    assert os.readlink(f"/proc/{p.pid}/exe") == os.path.realpath(sys.executable)      # the pid is the interpreter
    os.kill(p.pid, sig)
    rc = p.wait(timeout=10)
    assert state.read_text().strip() == "balanced"
    assert rc != 0                                                                   # still stopped by the signal


def test_tune_restores_at_exit_when_nothing_else_ran(tmp_path: Path) -> None:
    p, state = _tune_child(tmp_path, "exit")
    assert p.wait(timeout=10) == 0
    assert state.read_text().strip() == "balanced"


# compare: generate_s_mean_clean on a hand-computed example


def _rows(secs: list[float], clean: list[bool | None]) -> dict[int, dict]:
    rows = {}
    for i, (s, c) in enumerate(zip(secs, clean)):
        r = {"index": i, "correct": True, "forward_passes": 4, "generate_s": s, "flags": ["f"], "version": "v",
             "model": "m"}
        if c is not None:
            r["isolation"] = {"clean": c, "reasons": [] if c else ["x"], "retries": 0}
        rows[i] = r
    return rows


def test_clean_means_over_the_rows_clean_in_both_arms() -> None:
    # A: 10 (clean), 20 (clean), 90 (noisy).  B: 12 (clean), 100 (noisy), 30 (clean).
    # Clean in both: row 0 only. A mean 10, B mean 12; A has 1 noisy row, B has 1 noisy row.
    A = _rows([10, 20, 90], [True, True, False])
    B = _rows([12, 100, 30], [True, False, True])
    got = eval_gsm8k.clean_summary(A, B, [0, 1, 2])
    assert got["generate_s_mean_clean"] == {"a": pytest.approx(10.0), "b": pytest.approx(12.0)}
    assert got["clean_rows"] == 1 and got["noisy_rows"] == {"a": 1, "b": 1} and got["unchecked_rows"] == {"a": 0, "b": 0}
    # the plain mean is untouched: (10 + 20 + 90) / 3 = 40 and (12 + 100 + 30) / 3 = 47.33
    assert sum(A[i]["generate_s"] for i in A) / 3 == pytest.approx(40.0)


def test_rows_without_a_window_are_unchecked_not_clean() -> None:
    A = _rows([10, 20, 30], [True, None, True])
    B = _rows([11, 21, 31], [True, True, True])
    got = eval_gsm8k.clean_summary(A, B, [0, 1, 2])
    assert got["clean_rows"] == 2 and got["unchecked_rows"] == {"a": 1, "b": 0}
    assert got["generate_s_mean_clean"] == {"a": pytest.approx(20.0), "b": pytest.approx(21.0)}


def test_no_clean_pair_gives_none_not_zero() -> None:
    A = _rows([10, 20], [False, False])
    B = _rows([11, 21], [True, True])
    got = eval_gsm8k.clean_summary(A, B, [0, 1])
    assert got["generate_s_mean_clean"] == {"a": None, "b": None} and got["clean_rows"] == 0


def test_compare_writes_the_clean_fields_and_leaves_accuracy_over_all_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         capsys: pytest.CaptureFixture) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (tmp_path / "bench" / "eval").mkdir(parents=True)
    monkeypatch.setattr(eval_gsm8k, "RUNS", runs)
    monkeypatch.setattr(eval_gsm8k, "ROOT", tmp_path)
    A = _rows([10, 20, 90], [True, True, False])
    B = _rows([12, 100, 30], [True, False, True])
    A[2]["correct"] = False
    for name, rows in (("a", A), ("b", B)):
        (runs / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    assert eval_gsm8k.cmd_compare("a", "b") == 0
    doc = json.loads(next((tmp_path / "bench/eval").glob("*-a-vs-b.json")).read_text())
    assert doc["generate_s_mean_clean"] == {"a": pytest.approx(10.0), "b": pytest.approx(12.0)}
    assert doc["noisy_rows"] == {"a": 1, "b": 1} and doc["clean_rows"] == 1
    assert doc["accuracy"]["a"] == pytest.approx(2 / 3) and doc["accuracy"]["b"] == 1.0          # all rows
    assert doc["generate_s_mean"]["a"] == pytest.approx(40.0)


# need_ram_gb, and the fake scripts are valid shell


def test_unreadable_model_size_is_none_not_zero(tmp_path: Path) -> None:
    assert isolation.need_ram_gb(tmp_path / "missing") is None
    assert isolation.need_ram_gb(tmp_path) is None            # a directory is not a model file


def test_the_memory_check_fails_naming_the_source_when_the_need_is_unknown(fake_machine: FakeMachine) -> None:
    rep = isolation.preflight("cpu", None, machine=fake_machine, sample_s=1.0)
    assert [c.name for c in rep.failed] == ["memory"] and "model file" in detail(rep, "memory")


def test_need_ram_is_a_multiple_of_the_model_file(tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * 4_000_000)
    assert isolation.need_ram_gb(f) == pytest.approx(0.01)


def test_fake_scripts_pass_bash_n(tmp_path: Path) -> None:
    for name, body in (("nvidia", FAKE_NVIDIA), ("ppd", FAKE_PPD)):
        path = tmp_path / name
        path.write_text(body)
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    assert "\\x" not in FAKE_NVIDIA + FAKE_PPD


# B1: the benchmark child's own last ticks are not "other" when it exits before the closing tick


BURN_13 = "import time\nend = time.monotonic() + 1.3\nwhile time.monotonic() < end:\n    pass\n"


def test_a_real_child_that_exits_inside_the_window_is_not_counted_as_other(tmp_path: Path) -> None:
    """Reviewer's repro: the child's ticks leave /proc when it exits, system busy keeps them."""
    ncpu = os.cpu_count() or 1
    real = isolation.Machine(sysfs=tmp_path / "nosys", power_root=tmp_path / "nopower")
    # This test reads the real /proc, so it needs a machine nobody else is using: a second of idle first, and a
    # skip (not a failure) when other processes already use more than a tenth of it.
    idle = isolation.Window(pid=os.getpid(), machine=real, sample_s=0.25)
    with idle:
        time.sleep(1.0)
    if idle.result["other_cpu_share_mean"] > 0.10:
        pytest.skip(f"the machine is busy ({idle.result['other_cpu_share_mean']:.0%} other CPU); this test needs it idle")
    w = isolation.Window(pid=os.getpid(), machine=real, sample_s=0.5)
    with w:
        kids = [subprocess.Popen([sys.executable, "-c", BURN_13]) for _ in range(ncpu)]
        for k in kids:
            k.wait()
    # The child's own ticks counted as other read 0.97 here before the cutime fix; ordinary neighbours read a few
    # percent. Half the machine separates the two with room on both sides.
    assert w.result["other_cpu_share_max"] < 0.5, w.result


def test_a_tree_member_that_vanishes_between_samples_is_not_other(fake_machine: FakeMachine) -> None:
    fake_machine.add_proc(5000, 4242, "child", rate=200)                         # two of four cores
    w = isolation.Window(pid=4242, machine=fake_machine, thread=False)
    with w:
        for i in range(6):
            fake_machine.advance(0.5)
            if i == 5:
                fake_machine.remove_proc(5000)                                    # exits, its parent waits for it
            w.tick()
    assert w.result["clean"] is True, w.result
    assert w.result["other_cpu_share_max"] == pytest.approx(0.0)


def test_the_childs_system_time_is_part_of_its_ticks(fake_machine: FakeMachine) -> None:
    fake_machine.add_proc(5000, 4242, "child", rate=200, sys_share=1.0)           # all kernel time
    assert window(fake_machine)["clean"] is True


def test_iowait_is_idle_not_busy(fake_machine: FakeMachine) -> None:
    fake_machine.iowait_rate = 200                                                # half of the capacity waits on disk
    res = window(fake_machine)
    assert res["clean"] is True and res["other_cpu_share_max"] == pytest.approx(0.0)
    assert status(pre(fake_machine), "cpu-busy") == "PASS"


@pytest.mark.parametrize("which", ["swap_rate", "swapin_rate"])
def test_swap_in_and_swap_out_each_flag_the_preflight(fake_machine: FakeMachine, which: str) -> None:
    setattr(fake_machine, which, 50)
    assert [c.name for c in pre(fake_machine).failed] == ["swap-movement"]


@pytest.mark.parametrize("which", ["swap_rate", "swapin_rate"])
def test_swap_in_and_swap_out_each_flag_the_window(fake_machine: FakeMachine, which: str) -> None:
    setattr(fake_machine, which, 50)
    res = window(fake_machine)
    assert res["clean"] is False and res["swap_pages"] > 0


# N4: the GPU is sampled every 2 s, not on every 0.5 s tick


def test_gpu_is_sampled_every_two_seconds(fake_machine: FakeMachine) -> None:
    window(fake_machine, steps=10, device="gpu")
    queries = [c for c in fake_machine.calls if c[0] == "nvidia-smi" and c[1].startswith("--query-gpu")]
    assert len(queries) == 4                                  # the opening reading, 2 s, 4 s and the closing one


def test_a_gpu_reason_between_gpu_samples_is_missed_but_one_at_a_sample_is_not(fake_machine: FakeMachine) -> None:
    res = window(fake_machine, steps=10, device="gpu",
                 each=lambda i: fake_machine.gpu.update(reasons=0x21 if i in (0, 3) else 0x1))
    assert res["clean"] is False and res["gpu_reasons_seen"] == ["sw_thermal"]


# N5: a software power cap is recorded, the other throttle reasons stay noisy


def test_a_software_power_cap_is_recorded_not_noisy(fake_machine: FakeMachine) -> None:
    res = window(fake_machine, device="gpu", each=lambda i: fake_machine.gpu.update(reasons=0x5))
    assert res["clean"] is True and res["gpu_power_capped"] is True
    assert "sw_power_cap" in res["gpu_reasons_seen"]


@pytest.mark.parametrize("bit,name", [(0x2, "applications_clocks"), (0x8, "hw_slowdown"), (0x20, "sw_thermal"),
                                       (0x40, "hw_thermal"), (0x80, "hw_power_brake")])
def test_these_gpu_reasons_stay_noisy(fake_machine: FakeMachine, bit: int, name: str) -> None:
    res = window(fake_machine, device="gpu", each=lambda i: fake_machine.gpu.update(reasons=0x1 | bit))
    assert res["clean"] is False and name in res["gpu_reasons_seen"] and res["gpu_power_capped"] is False


# N7: every row says whether the preflight was waved through, and which session wrote it


def test_a_row_carries_its_session_and_the_allow_noisy_flag(fake_machine: FakeMachine, tmp_path: Path) -> None:
    fake_machine.profile = fake_machine.platform_profile = "balanced"
    fake_machine.auto = True
    arm = isolation.begin_arm("n7", device="cpu", need_ram_gb=1.0, runs_dir=tmp_path, machine=fake_machine,
                              sample_s=1.0, allow_noisy=True, allow_battery=True, log=lambda m: None)
    _, info = arm.run(lambda: 1, quiet_sample_s=0.1)
    iso = info["isolation"]
    session = json.loads((tmp_path / "n7.env.json").read_text())["sessions"][-1]
    assert iso["allow_noisy"] is True and iso["preflight_failed_allowed"] is True
    assert iso["session"] == session["session_id"] and session["session_id"]
    again = isolation.begin_arm("n7", device="cpu", need_ram_gb=1.0, runs_dir=tmp_path, machine=fake_machine,
                                sample_s=1.0, allow_noisy=True, allow_battery=True, log=lambda m: None)
    assert again.session_id != arm.session_id


def test_a_clean_arm_rows_say_allow_noisy_false(fake_machine: FakeMachine, tmp_path: Path) -> None:
    fake_machine.auto = True
    arm = isolation.begin_arm("n7b", device="cpu", need_ram_gb=1.0, runs_dir=tmp_path, machine=fake_machine,
                              sample_s=1.0, allow_battery=True, log=lambda m: None)
    _, info = arm.run(lambda: 1, quiet_sample_s=0.1)
    assert info["isolation"]["allow_noisy"] is False and info["isolation"]["preflight_failed_allowed"] is False


def test_a_child_that_ran_before_the_window_edge_and_exits_is_counted_once(fake_machine: FakeMachine) -> None:
    """The parent's waited-for time holds the child's whole life, so a per-process difference counts the ticks the
    child had already used before the previous sample a second time and hides a real intruder. The sum over the
    tree does not."""
    fake_machine.add_proc(5000, 4242, "child", rate=200)
    w = isolation.Window(pid=4242, machine=fake_machine, thread=False)
    with w:
        for i in range(6):
            fake_machine.extra_busy = 80 if i == 5 else 0                      # an intruder in the last interval only
            fake_machine.advance(0.5)
            if i == 5:
                fake_machine.remove_proc(5000)
            w.tick()
    assert w.result["other_cpu_share_max"] == pytest.approx(0.2)               # 40 ticks of 200 in that interval
