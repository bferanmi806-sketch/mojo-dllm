"""A timed run says whether it had the machine to itself, and a problem that did not is run again.

`bench/power.py` answers one question about the machine: is it on mains power. This module answers the others.
Before an arm starts, `preflight` reads the power profile, the CPU governor, system CPU load, pressure stall
information (PSI), swap, free memory, thermal throttle counters and, for a GPU arm, the GPU's processes, memory,
utilization and clock event reasons. Around every problem, a `Window` samples the same sources every 0.5 s,
subtracts what the benchmark's own process tree used, and reports `clean` with the reasons when it was not.
`run_isolated` waits for a quiet moment, runs the problem inside a window and repeats it up to `retries` times
while the window is noisy. After that the row is kept with its reasons; accuracy never depends on a window.

The checks fail closed, as `power.py` does. A source that cannot be read is a WARN in the preflight and an
`unavailable` entry in the window, and a window with an unmeasured dimension is never clean. An empty reading
is not a pass: no ticks elapsed, a file with no content and a `nvidia-smi` that printed nothing all read as
unavailable. Sources that do not exist on this host are not checked: a machine without an NVIDIA GPU has no GPU
check on a cpu arm, and a source that was absent in the preflight cannot make a window unmeasured.

Every reader goes through a `Machine` (paths for /proc and /sys, a command runner, a clock), so tests run
against fake trees and fake `nvidia-smi`, `systemctl` and `powerprofilesctl` scripts, and no test touches the
real GPU or the real power profile. Standard library only.

Thresholds are the module constants below. An arm may override them only by passing a `Thresholds` explicitly,
and `begin_arm` records the ones it used in ARM.env.json.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import datetime as dt
import glob
import json
import os
import pathlib
import platform
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Callable, Iterator, TypeVar

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import power  # noqa: E402  (bench/power.py)

T = TypeVar("T")

# Preflight, before an arm: FAIL when exceeded over a 5 s sample.
PREFLIGHT_SAMPLE_S = 5.0
PREFLIGHT_CPU_BUSY_MAX = 0.10      # share of all CPUs busy (LLVM benchmarking tips: measure on an idle machine)
PREFLIGHT_PSI_AVG10_MAX = 5.0      # PSI cpu "some" avg10, in percent (kernel accounting/psi)
GPU_MEM_USED_MAX_MIB = 300         # a desktop session holds some; more means another job
GPU_UTIL_MAX = 5.0                 # percent, mean of three samples
# Window, around one problem.
WINDOW_SAMPLE_S = 0.5
OTHER_CPU_MEAN_MAX = 0.05          # share of total machine capacity used by processes outside the benchmark tree
OTHER_CPU_SAMPLE_MAX = 0.25        # the same, in any single 0.5 s sample
SWAP_PAGES_MAX = 0                 # pages swapped in or out during the window
THERMAL_EVENTS_MAX = 0             # core + package throttle counter steps during the window
FOREIGN_GPU_PIDS_MAX = 0           # GPU compute processes that are not in the benchmark tree
RAM_FACTOR = 1.25                  # free memory wanted, as a multiple of the model file

# NVML clocks event reasons (nvidia-smi --help-query-gpu, clocks_event_reasons.active).
GPU_REASONS = {0x1: "idle", 0x2: "applications_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
               0x10: "sync_boost", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake",
               0x100: "display_clock"}
GPU_REASON_IDLE = 0x1
GPU_REASON_DISPLAY = 0x100         # a display clock setting says nothing about contention: recorded only
GPU_REASON_POWER_CAP = 0x4         # a software power cap is intrinsic on a 30 W laptop GPU: recorded, not noisy
GPU_REASONS_IGNORED = GPU_REASON_IDLE | GPU_REASON_DISPLAY | GPU_REASON_POWER_CAP
GPU_SAMPLE_S = 2.0                 # the GPU is read this often inside a window (two nvidia-smi processes per read)
TIMERS = ("sysstat-collect", "apt-daily", "apt-daily-upgrade", "fwupd-refresh", "anacron", "man-db", "motd-news")


@dataclasses.dataclass(frozen=True)
class Thresholds:
    """The limits a preflight, a gate and a window apply. Defaults are the module constants."""

    cpu_busy_max: float = PREFLIGHT_CPU_BUSY_MAX
    psi_avg10_max: float = PREFLIGHT_PSI_AVG10_MAX
    gpu_mem_used_max_mib: float = GPU_MEM_USED_MAX_MIB
    gpu_util_max: float = GPU_UTIL_MAX
    other_cpu_mean_max: float = OTHER_CPU_MEAN_MAX
    other_cpu_sample_max: float = OTHER_CPU_SAMPLE_MAX
    swap_pages_max: int = SWAP_PAGES_MAX
    thermal_events_max: int = THERMAL_EVENTS_MAX
    foreign_gpu_pids_max: int = FOREIGN_GPU_PIDS_MAX


class IsolationError(RuntimeError):
    """The machine is not fit for a timed run, or tuning it failed."""


# the machine: every system reader goes through this, so tests substitute a fake


class Machine:
    """Where the readers look: /proc, /sys, the power supplies, commands, the clocks."""

    def __init__(self, proc: str | os.PathLike = "/proc", sysfs: str | os.PathLike = "/sys",
                 power_root: str | os.PathLike | None = None, env: dict[str, str] | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time, clk_tck: int | None = None,
                 self_pid: int | None = None) -> None:
        self.proc, self.sys = pathlib.Path(proc), pathlib.Path(sysfs)
        self.power_root = pathlib.Path(power_root) if power_root is not None else power.POWER_SUPPLY
        self.env, self.sleep, self.clock, self.wall = env, sleep, clock, wall
        self.clk_tck = clk_tck or os.sysconf("SC_CLK_TCK")
        self.self_pid = os.getpid() if self_pid is None else self_pid

    def read(self, path: pathlib.Path) -> str | None:
        """The file's text, or None when it is missing, unreadable or empty (an empty reading is no reading)."""
        try:
            text = path.read_text()
        except OSError:
            return None
        return text if text.strip() else None

    def exists(self, path: pathlib.Path) -> bool:
        return path.exists()

    def run(self, cmd: list[str], timeout: float = 15.0) -> tuple[int, str]:
        """(exit code, stdout); 127 when the program is missing, 124 on a timeout, 126 when it cannot start."""
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=self.env,
                               stdin=subprocess.DEVNULL)
        except FileNotFoundError:
            return 127, ""
        except subprocess.TimeoutExpired:
            return 124, ""
        except OSError:
            return 126, ""
        return p.returncode, p.stdout

    def has(self, program: str) -> bool:
        return shutil.which(program, path=(self.env or os.environ).get("PATH")) is not None


_MACHINE = Machine()


def get_machine() -> Machine:
    return _MACHINE


def use_machine(m: Machine) -> Machine:
    """Replace the machine the readers look at (tests); returns the previous one."""
    global _MACHINE
    previous, _MACHINE = _MACHINE, m
    return previous


# readers: each returns None when the source is missing, unreadable, empty or malformed


def read_cpu_stat(m: Machine) -> tuple[int, int, int] | None:
    """(total ticks, busy ticks, online CPUs) from /proc/stat. Guest time is inside user, so it is left out."""
    text = m.read(m.proc / "stat")
    if text is None:
        return None
    ncpu, first = 0, None
    for line in text.splitlines():
        name = line.split(None, 1)[0] if line.strip() else ""
        if name == "cpu":
            first = line.split()[1:]
        elif name.startswith("cpu") and name[3:].isdigit():
            ncpu += 1
    try:
        vals = [int(x) for x in (first or [])][:8]
    except ValueError:
        return None
    if len(vals) < 5 or not ncpu:
        return None
    vals += [0] * (8 - len(vals))
    total = sum(vals)
    return total, total - vals[3] - vals[4], ncpu


def read_psi(m: Machine, kind: str) -> dict[str, float] | None:
    """{"avg10": percent, "total": microseconds} of the `some` line of /proc/pressure/<kind>."""
    text = m.read(m.proc / "pressure" / kind)
    if text is None:
        return None
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "some":
            try:
                kv = dict(p.split("=", 1) for p in parts[1:])
                return {"avg10": float(kv["avg10"]), "total": float(kv["total"])}
            except (KeyError, ValueError):
                return None
    return None


def read_swap_pages(m: Machine) -> dict[str, int] | None:
    text = m.read(m.proc / "vmstat")
    if text is None:
        return None
    kv = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("pswpin", "pswpout"):
            try:
                kv[parts[0]] = int(parts[1])
            except ValueError:
                return None
    return kv if len(kv) == 2 else None


def read_meminfo(m: Machine) -> dict[str, int] | None:
    """MemAvailable, SwapTotal and SwapFree in kB."""
    text = m.read(m.proc / "meminfo")
    if text is None:
        return None
    kv = {}
    for line in text.splitlines():
        parts = line.replace(":", " ").split()
        if len(parts) >= 2 and parts[0] in ("MemAvailable", "SwapTotal", "SwapFree"):
            try:
                kv[parts[0]] = int(parts[1])
            except ValueError:
                return None
    return kv if len(kv) == 3 else None


def read_thermal(m: Machine) -> dict[str, int] | None:
    """Throttle counters summed over CPUs: {"core": n, "package": n}. Kernel intel_thermal_throttle."""
    out = {"core": 0, "package": 0}
    seen = 0
    for kind in out:
        for path in sorted(glob.glob(str(m.sys / "devices/system/cpu/cpu[0-9]*/thermal_throttle" / f"{kind}_throttle_count"))):
            text = m.read(pathlib.Path(path))
            if text is None:
                return None
            try:
                out[kind] += int(text.strip())
            except ValueError:
                return None
            seen += 1
    return out if seen else None


def _cpu_files(m: Machine, rel: str) -> list[pathlib.Path]:
    return [pathlib.Path(p) for p in sorted(glob.glob(str(m.sys / "devices/system/cpu/cpu[0-9]*" / rel)))]


def read_values(m: Machine, rel: str) -> list[str] | None:
    """The stripped text of a per-CPU file for every CPU; None when any CPU's file cannot be read."""
    files = _cpu_files(m, rel)
    vals = [m.read(f) for f in files]
    if not files or any(v is None for v in vals):
        return None
    return [v.strip() for v in vals if v is not None]


def read_text(m: Machine, path: pathlib.Path) -> str | None:
    text = m.read(path)
    return text.strip() if text else None


def process_table(m: Machine) -> dict[int, tuple[int, int, str, int]] | None:
    """{pid: (ppid, utime + stime, name, cutime + cstime)} in ticks for every process, or None when /proc cannot be listed.

    cutime and cstime are the time of children the process has waited for: when a child exits and is reaped, its
    ticks leave the table as a process and arrive here, so a tree's total does not lose them."""
    try:
        names = [n for n in os.listdir(m.proc) if n.isdigit()]
    except OSError:
        return None
    table: dict[int, tuple[int, int, str]] = {}
    for n in names:
        text = m.read(m.proc / n / "stat")
        if text is None:
            continue    # the process ended between the listing and the read
        head, _, tail = text.rpartition(")")
        parts = tail.split()
        if not head or len(parts) < 15:
            continue
        try:
            table[int(n)] = (int(parts[1]), int(parts[11]) + int(parts[12]), head.partition("(")[2],
                             int(parts[13]) + int(parts[14]))
        except ValueError:
            continue
    return table


def tree_of(table: dict[int, tuple[int, int, str, int]], root: int) -> set[int]:
    """`root` and its descendants, by parent pid. Empty when `root` is not in the table."""
    if root not in table:
        return set()
    children: dict[int, list[int]] = {}
    for pid, (ppid, *_) in table.items():
        children.setdefault(ppid, []).append(pid)
    out, todo = {root}, [root]
    while todo:
        for c in children.get(todo.pop(), []):
            if c not in out:
                out.add(c)
                todo.append(c)
    return out


def _na(x: str | None) -> str | None:
    x = (x or "").strip()
    return None if not x or x.startswith("[") or x.upper() == "N/A" else x


def _num(x: str | None) -> float | None:
    x = _na(x)
    try:
        return float(x) if x is not None else None
    except ValueError:
        return None


GPU_FIELDS = ("clocks_event_reasons.active", "utilization.gpu", "memory.used", "temperature.gpu", "clocks.sm",
              "clocks.max.sm", "persistence_mode", "compute_mode", "power.draw")
GPU_WINDOW_FIELDS = ("clocks_event_reasons.active", "clocks.sm")
_OLD_REASONS = "clocks_throttle_reasons.active"


def read_gpu(m: Machine, fields: tuple[str, ...] = GPU_FIELDS) -> dict[str, Any] | None:
    """One `nvidia-smi --query-gpu` row keyed by field name; None when nvidia-smi fails or prints nothing.

    Older drivers call the reasons field clocks_throttle_reasons.active; it is tried when the first name errors.
    The reasons come back as an int under `reasons`."""
    row = None
    for reasons_field in ("clocks_event_reasons.active", _OLD_REASONS):
        names = tuple(reasons_field if f == "clocks_event_reasons.active" else f for f in fields)
        rc, out = m.run(["nvidia-smi", f"--query-gpu={','.join(names)}", "--format=csv,noheader,nounits"])
        lines = [l for l in out.splitlines() if l.strip()]
        if rc == 0 and lines:
            row = [c.strip() for c in lines[0].split(",")]
            if len(row) == len(fields):
                break
            row = None
        if "clocks_event_reasons.active" not in fields:
            break
    if row is None:
        return None
    got: dict[str, Any] = dict(zip(fields, row))
    raw = _na(got.get("clocks_event_reasons.active"))
    try:
        got["reasons"] = int(raw, 16) if raw is not None else None
    except ValueError:
        got["reasons"] = None
    return got


def read_gpu_apps(m: Machine) -> list[tuple[int, float]] | None:
    """[(pid, MiB)] of GPU compute processes. An empty list is an answer; a failed query is None."""
    rc, out = m.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    if rc != 0:
        return None
    apps = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            apps.append((int(parts[0]), float(parts[1]) if len(parts) > 1 and _na(parts[1]) else 0.0))
        except ValueError:
            return None
    return apps


def reason_names(bits: int) -> list[str]:
    return [name for mask, name in GPU_REASONS.items() if bits & mask]


def contended_reasons(bits: int | None) -> list[str]:
    """The event reasons that count against a run: everything but idle, the display clock setting and a software
    power cap, which are recorded (`gpu_power_capped`) and nothing else."""
    return reason_names((bits or 0) & ~GPU_REASONS_IGNORED)


def gpu_present(m: Machine) -> bool:
    return m.has("nvidia-smi")


def top_processes(before: dict, after: dict, dt_s: float, tck: int, skip: set[int], n: int = 5) -> list[dict]:
    """The processes that used the most CPU between two process tables, as percent of one core."""
    if dt_s <= 0:
        return []
    rows = []
    for pid, (_, ticks, name, _c) in after.items():
        if pid in skip:
            continue
        used = ticks - before[pid][1] if pid in before else ticks
        pct = max(used, 0) / (tck * dt_s) * 100
        if pct >= 1.0:
            rows.append({"pid": pid, "name": name, "percent": round(pct, 1)})
    return sorted(rows, key=lambda r: -r["percent"])[:n]


def _fmt_procs(procs: list[dict]) -> str:
    return ", ".join(f"{p['name']} ({p['pid']}) {p['percent']:.0f}%" for p in procs[:3]) or "none above 1%"


# snapshot


def _count_crontab(m: Machine) -> int | None:
    rc, out = m.run(["crontab", "-l"])
    if rc == 127:
        return None
    if rc != 0:
        return 0 if not out.strip() else None     # "no crontab for user" exits 1 and prints on stderr
    return sum(1 for l in out.splitlines() if l.strip() and not l.lstrip().startswith("#"))


def _timers_due(m: Machine, horizon_s: float = 3600.0) -> list[str] | None:
    names: list[str] = []
    got_any = False
    for cmd in (["systemctl", "list-timers", "--all", "--output=json", "--no-pager"],
                ["systemctl", "--user", "list-timers", "--all", "--output=json", "--no-pager"]):
        rc, out = m.run(cmd)
        if rc != 0 or not out.strip():
            continue
        try:
            entries = json.loads(out)
        except ValueError:
            continue
        got_any = True
        now_us = m.wall() * 1e6
        names += [e["unit"] for e in entries if isinstance(e, dict) and e.get("unit") and e.get("next")
                  and 0 <= e["next"] - now_us <= horizon_s * 1e6]
    return sorted(set(names)) if got_any else None


def snapshot(m: Machine | None = None, *, sample_s: float = 1.0, top: list[dict] | None = None) -> dict:
    """Everything the checks look at, in one dict; `unavailable` names the sources that could not be read.

    The top CPU processes are measured over `sample_s` unless the caller passes `top` from its own sample."""
    m = m or get_machine()
    unavailable: list[str] = []

    def need(name: str, value: Any) -> Any:
        if value is None:
            unavailable.append(name)
        return value

    snap: dict[str, Any] = {"taken": dt.datetime.fromtimestamp(m.wall()).isoformat(timespec="seconds")}
    gov = read_values(m, "cpufreq/scaling_governor")
    epp = read_values(m, "cpufreq/energy_performance_preference")
    freq = read_values(m, "cpufreq/scaling_cur_freq")
    snap["governors"] = sorted(set(need("cpu-governor", gov) or []))
    snap["energy_performance_preference"] = sorted(set(epp or []))
    snap["intel_pstate_status"] = read_text(m, m.sys / "devices/system/cpu/intel_pstate/status")
    snap["no_turbo"] = read_text(m, m.sys / "devices/system/cpu/intel_pstate/no_turbo")
    snap["platform_profile"] = read_text(m, m.sys / "firmware/acpi/platform_profile")
    snap["powerprofilesctl"] = None
    if m.has("powerprofilesctl"):
        rc, out = m.run(["powerprofilesctl", "get"])
        snap["powerprofilesctl"] = out.strip() if rc == 0 and out.strip() else need("powerprofilesctl", None)
    if snap["platform_profile"] is None and snap["powerprofilesctl"] is None:
        unavailable.append("power-profile")
    khz = [int(f) for f in (freq or []) if f.isdigit()]
    snap["cpu_freq_mhz"] = {"min": min(khz) // 1000, "max": max(khz) // 1000} if khz else need("cpu-freq", None)
    snap["thermal_throttle"] = need("thermal", read_thermal(m))
    load = read_text(m, m.proc / "loadavg")
    snap["loadavg"] = [float(x) for x in load.split()[:3]] if load else need("loadavg", None)
    snap["psi"] = {k: need(f"psi-{k}", read_psi(m, k)) for k in ("cpu", "memory")}
    mem = need("meminfo", read_meminfo(m))
    snap["mem_available_mb"] = mem["MemAvailable"] // 1024 if mem else None
    snap["swap"] = ({"total_mb": mem["SwapTotal"] // 1024, "used_mb": (mem["SwapTotal"] - mem["SwapFree"]) // 1024}
                    if mem else None)
    snap["swap_pages"] = need("vmstat", read_swap_pages(m))
    if top is None:
        before = process_table(m)
        t0 = m.clock()
        if sample_s > 0:
            m.sleep(sample_s)
        after = process_table(m)
        top = (top_processes(before, after, m.clock() - t0, m.clk_tck, {m.self_pid})
               if before is not None and after is not None else need("processes", None))
    snap["top_cpu_processes"] = top
    if gpu_present(m):
        gpu = read_gpu(m)
        apps = read_gpu_apps(m)
        if gpu is None or apps is None:
            unavailable.append("gpu")
        snap["gpu"] = None if gpu is None else {
            "reasons": gpu["reasons"], "reason_names": reason_names(gpu["reasons"] or 0),
            "utilization_percent": _num(gpu["utilization.gpu"]), "memory_used_mib": _num(gpu["memory.used"]),
            "temperature_c": _num(gpu["temperature.gpu"]), "sm_clock_mhz": _num(gpu["clocks.sm"]),
            "max_sm_clock_mhz": _num(gpu["clocks.max.sm"]), "persistence_mode": _na(gpu["persistence_mode"]),
            "compute_mode": _na(gpu["compute_mode"]), "power_draw_w": _num(gpu["power.draw"]),
            "compute_apps": None if apps is None else [{"pid": p, "memory_mib": mib} for p, mib in apps]}
    else:
        snap["gpu"] = None
    snap["timers_due_within_1h"] = need("systemd-timers", _timers_due(m))
    snap["user_crontab_lines"] = need("crontab", _count_crontab(m))
    snap["unavailable"] = unavailable
    return snap


def present_sources(m: Machine, device: str) -> set[str]:
    """The sources that exist on this host (readable or not): a window must measure each of these."""
    out = set()
    if m.exists(m.proc / "stat"):
        out.add("cpu-stat")
    if m.exists(m.proc / "pressure" / "cpu"):
        out.add("psi-cpu")
    if m.exists(m.proc / "vmstat"):
        out.add("vmstat")
    if _cpu_files(m, "thermal_throttle/core_throttle_count") or _cpu_files(m, "thermal_throttle/package_throttle_count"):
        out.add("thermal")
    if m.exists(m.proc / str(m.self_pid) / "stat"):
        out.add("proc-tree")
    if device == "gpu":
        out.add("gpu")
    return out


# preflight


@dataclasses.dataclass
class Check:
    name: str
    status: str          # PASS, WARN or FAIL
    detail: str


@dataclasses.dataclass
class Report:
    device: str
    checks: list[Check]
    snapshot: dict
    present: set[str]
    offenders: list[dict]
    sample_s: float

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    @property
    def ok(self) -> bool:
        return not self.failed

    def text(self) -> str:
        lines = [f"preflight ({self.device}, {self.sample_s:g} s sample): "
                 f"{'FAIL' if self.failed else 'ok'}, {len(self.failed)} FAIL, "
                 f"{sum(c.status == 'WARN' for c in self.checks)} WARN"]
        lines += [f"  {c.status:4s} {c.name}: {c.detail}" for c in self.checks]
        if self.offenders:
            lines.append(f"  top CPU users during the sample: {_fmt_procs(self.offenders)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"device": self.device, "ok": self.ok, "sample_s": self.sample_s,
                "checks": [dataclasses.asdict(c) for c in self.checks], "offenders": self.offenders,
                "present": sorted(self.present)}


def _profile_check(m: Machine) -> Check:
    seen: dict[str, str] = {}
    if m.has("powerprofilesctl"):
        rc, out = m.run(["powerprofilesctl", "get"])
        if rc == 0 and out.strip():
            seen["powerprofilesctl"] = out.strip()
    pp = read_text(m, m.sys / "firmware/acpi/platform_profile")
    if pp:
        seen["platform_profile"] = pp
    if not seen:
        return Check("profile", "WARN", f"cannot read the power profile ({m.sys / 'firmware/acpi/platform_profile'} "
                     "and `powerprofilesctl get` both failed): unchecked")
    bad = {k: v for k, v in seen.items() if v != "performance"}
    shown = ", ".join(f"{k}={v}" for k, v in seen.items())
    return Check("profile", "FAIL" if bad else "PASS",
                 f"{shown}" + ("; the published rows use `performance` (run with --tune)" if bad else ""))


def _governor_check(m: Machine) -> Check:
    gov = read_values(m, "cpufreq/scaling_governor")
    if gov is None:
        return Check("governor", "WARN", f"cannot read {m.sys}/devices/system/cpu/cpu*/cpufreq/scaling_governor: unchecked")
    govs = set(gov)
    if govs == {"performance"}:
        return Check("governor", "PASS", "performance on every CPU")
    epp = set(read_values(m, "cpufreq/energy_performance_preference") or [])
    status = read_text(m, m.sys / "devices/system/cpu/intel_pstate/status")
    if govs == {"powersave"} and status == "active" and epp == {"performance"}:
        return Check("governor", "PASS", "powersave with intel_pstate active and preference performance")
    return Check("governor", "FAIL", f"governors {sorted(govs)}, preference {sorted(epp) or 'unreadable'}, "
                 f"intel_pstate {status or 'unreadable'}: not performance")


def preflight(device: str, need_ram_gb: float | None, allow: tuple[str, ...] = (), *, machine: Machine | None = None,
              thresholds: Thresholds | None = None, sample_s: float = PREFLIGHT_SAMPLE_S,
              allow_battery: bool = False) -> Report:
    """Check the machine before an arm. Never raises on a missing source: it is a WARN naming the source.

    `allow` names checks whose FAIL is reported as a WARN. FAIL lines name their offenders."""
    m, th = machine or get_machine(), thresholds or Thresholds()
    checks: list[Check] = []

    def warn(name: str, what: str) -> None:
        checks.append(Check(name, "WARN", f"cannot read {what}: unchecked"))

    # power
    try:
        battery = power.on_battery(m.power_root)
    except power.PowerError as e:
        checks.append(Check("power", "WARN", f"{e}: unchecked"))
    else:
        if battery is None:
            checks.append(Check("power", "WARN", f"no battery or charger under {m.power_root}: the power state is unchecked"))
        elif battery:
            checks.append(Check("power", "WARN" if allow_battery else "FAIL", "on battery"))
        else:
            checks.append(Check("power", "PASS", "mains"))
    checks += [_profile_check(m), _governor_check(m)]

    # one sample window for CPU, PSI, swap, thermal, processes and (three readings) the GPU
    use_gpu = device == "gpu"
    t0, cpu0, psi0, sw0, th0 = m.clock(), read_cpu_stat(m), read_psi(m, "cpu"), read_swap_pages(m), read_thermal(m)
    proc0 = process_table(m)
    gpu_rows: list[dict | None] = []
    if use_gpu:
        gpu_rows.append(read_gpu(m))
    half = sample_s / 2
    m.sleep(half)
    if use_gpu:
        gpu_rows.append(read_gpu(m))
    m.sleep(sample_s - half)
    cpu1, psi1, sw1, th1 = read_cpu_stat(m), read_psi(m, "cpu"), read_swap_pages(m), read_thermal(m)
    proc1 = process_table(m)
    elapsed = m.clock() - t0
    if use_gpu:
        gpu_rows.append(read_gpu(m))
    offenders = (top_processes(proc0, proc1, elapsed, m.clk_tck, {m.self_pid})
                 if proc0 is not None and proc1 is not None else [])

    # CPU busy
    if cpu0 is None or cpu1 is None:
        warn("cpu-busy", f"{m.proc}/stat")
    elif cpu1[0] - cpu0[0] <= 0:
        checks.append(Check("cpu-busy", "WARN", f"no CPU ticks elapsed in {m.proc}/stat during the sample: unchecked"))
    else:
        busy = (cpu1[1] - cpu0[1]) / (cpu1[0] - cpu0[0])
        over = busy > th.cpu_busy_max
        checks.append(Check("cpu-busy", "FAIL" if over else "PASS",
                            f"{busy:.1%} of all CPUs busy over {sample_s:g} s (limit {th.cpu_busy_max:.0%})"
                            + (f"; offenders: {_fmt_procs(offenders)}" if over else "")))
    # PSI
    if psi1 is None:
        warn("psi-cpu", f"{m.proc}/pressure/cpu")
    else:
        over = psi1["avg10"] > th.psi_avg10_max
        checks.append(Check("psi-cpu", "FAIL" if over else "PASS",
                            f"cpu some avg10 {psi1['avg10']:.2f}% (limit {th.psi_avg10_max:g}%)"
                            + (f"; offenders: {_fmt_procs(offenders)}" if over else "")))
    # swap
    if sw0 is None or sw1 is None:
        warn("swap-movement", f"{m.proc}/vmstat")
    else:
        pages = (sw1["pswpin"] - sw0["pswpin"]) + (sw1["pswpout"] - sw0["pswpout"])
        over = pages > th.swap_pages_max
        checks.append(Check("swap-movement", "FAIL" if over else "PASS", f"{pages} pages swapped during the sample"))
    # memory and swap in use
    mem = read_meminfo(m)
    if mem is None:
        warn("memory", f"{m.proc}/meminfo")
    else:
        avail_gb = mem["MemAvailable"] / 1048576
        if need_ram_gb is None:
            checks.append(Check("memory", "FAIL", f"cannot read the model file size, so the memory need is unknown "
                                f"(MemAvailable {avail_gb:.1f} GB)"))
        else:
            low = avail_gb < need_ram_gb
            checks.append(Check("memory", "FAIL" if low else "PASS",
                                f"MemAvailable {avail_gb:.1f} GB, need {need_ram_gb:.1f} GB"))
        used_mb = (mem["SwapTotal"] - mem["SwapFree"]) // 1024
        if used_mb > 0:
            checks.append(Check("swap-in-use", "WARN", f"{used_mb} of {mem['SwapTotal'] // 1024} MB swap used: "
                                "a run that touches swapped pages measures the disk"))
    # thermal
    if th0 is None or th1 is None:
        warn("thermal", f"{m.sys}/devices/system/cpu/cpu*/thermal_throttle/*_throttle_count")
    else:
        steps = sum(th1.values()) - sum(th0.values())
        over = steps > th.thermal_events_max
        checks.append(Check("thermal", "FAIL" if over else "PASS",
                            f"{steps} throttle counter steps during the sample (core {th1['core']}, "
                            f"package {th1['package']} since boot)"))
    # GPU
    if use_gpu:
        rows = [r for r in gpu_rows if r is not None]
        apps = read_gpu_apps(m)
        if len(rows) != len(gpu_rows) or apps is None:
            warn("gpu", "`nvidia-smi --query-gpu` / `--query-compute-apps`")
        else:
            checks.append(Check("gpu-processes", "FAIL" if apps else "PASS",
                                f"compute processes: {[p for p, _ in apps]}" if apps else "none"))
            mem_used = _num(rows[-1]["memory.used"])
            if mem_used is None:
                warn("gpu-memory", "memory.used")
            else:
                over = mem_used > th.gpu_mem_used_max_mib
                checks.append(Check("gpu-memory", "FAIL" if over else "PASS",
                                    f"{mem_used:.0f} MiB used (limit {th.gpu_mem_used_max_mib:g})"))
            utils = [_num(r["utilization.gpu"]) for r in rows]
            if any(u is None for u in utils):
                warn("gpu-utilization", "utilization.gpu")
            else:
                mean_u = sum(u for u in utils if u is not None) / len(utils)
                over = mean_u > th.gpu_util_max
                checks.append(Check("gpu-utilization", "FAIL" if over else "PASS",
                                    f"{mean_u:.1f}% mean of {len(utils)} samples (limit {th.gpu_util_max:g}%)"))
            bits = [r["reasons"] for r in rows]
            if any(b is None for b in bits):
                warn("gpu-clock-reasons", "clocks_event_reasons.active")
            else:
                seen = sorted({n for b in bits for n in contended_reasons(b)})
                checks.append(Check("gpu-clock-reasons", "FAIL" if seen else "PASS",
                                    f"active: {seen}" if seen else "idle only"))

    allowed = set(allow)
    for c in checks:
        if c.status == "FAIL" and c.name in allowed:
            c.status, c.detail = "WARN", f"allowed: {c.detail}"
    snap = snapshot(m, sample_s=0, top=offenders)
    present = present_sources(m, device)
    return Report(device, checks, snap, present, offenders, sample_s)


# the quiet gate before each problem


def cpu_busy(sample_s: float = 2.0, *, machine: Machine | None = None) -> float:
    """Fraction of all CPU time spent busy over a short sample (0..1); 1.0 when it cannot be measured.

    The load average is the wrong signal between runs: it is a 1-minute moving average, so it still carries the
    previous benchmark run. A sample that elapses no ticks, or a /proc/stat that cannot be read, reads as busy."""
    m = machine or get_machine()
    a = read_cpu_stat(m)
    m.sleep(sample_s)
    b = read_cpu_stat(m)
    if a is None or b is None or b[0] - a[0] <= 0:
        return 1.0
    return (b[1] - a[1]) / (b[0] - a[0])


def wait_for_quiet(threshold: float, max_wait: float, *, machine: Machine | None = None,
                   sample_s: float = 2.0, poll_s: float = 3.0) -> tuple[float, float]:
    """(busy fraction, seconds waited): start when a sample shows the CPU under `threshold`, or after `max_wait`."""
    m = machine or get_machine()
    t0 = m.clock()
    while True:
        busy = cpu_busy(sample_s, machine=m)
        if busy < threshold or m.clock() - t0 > max_wait:
            return busy, m.clock() - t0
        m.sleep(poll_s)


def quiet_gate(device: str, *, machine: Machine | None = None, thresholds: Thresholds | None = None,
               max_wait_s: float = 60.0, sample_s: float = 1.0, poll_s: float = 3.0) -> dict:
    """The preflight's dynamic checks (CPU busy, swap movement, GPU processes; not PSI, which a 12-thread run raises by itself) over a short sample, repeated until they pass or `max_wait_s` has gone.

    Returns {"ok", "cpu_busy_before", "waited_s", "reasons"}; a gate that times out lets the problem run, and the
    window then decides whether it was clean."""
    m, th = machine or get_machine(), thresholds or Thresholds()
    t0 = m.clock()
    while True:
        reasons: list[str] = []
        c0, s0 = read_cpu_stat(m), read_swap_pages(m)
        g0 = read_gpu_apps(m) if device == "gpu" else []
        m.sleep(sample_s)
        c1, s1 = read_cpu_stat(m), read_swap_pages(m)
        if c0 is None or c1 is None or c1[0] - c0[0] <= 0:
            busy = 1.0
            reasons.append("cpu-busy unmeasured")
        else:
            busy = (c1[1] - c0[1]) / (c1[0] - c0[0])
            if busy > th.cpu_busy_max:
                reasons.append(f"cpu-busy {busy:.1%}")
        if s0 is None or s1 is None:
            reasons.append("swap unmeasured")
        elif sum(s1.values()) - sum(s0.values()) > th.swap_pages_max:
            reasons.append("swap moving")
        if device == "gpu":
            g1 = read_gpu_apps(m)
            if g0 is None or g1 is None:
                reasons.append("gpu unmeasured")
            elif g1:
                reasons.append(f"gpu processes {[p for p, _ in g1]}")
        waited = m.clock() - t0
        if not reasons or waited > max_wait_s:
            return {"ok": not reasons, "cpu_busy_before": round(busy, 3), "waited_s": round(waited, 1),
                    "reasons": reasons}
        m.sleep(poll_s)


# the window around one problem


class Window:
    """Samples the machine while one problem runs and reports whether anything else was using it.

    `with Window(pid=...) as w: ...` then `w.result`. `pid` is the root of the benchmark process tree (default:
    this process, so every child the runner starts is in it). Per sample: system busy ticks minus the tree's
    ticks (own use is not contention), PSI cpu total, swap and thermal counters, and for a GPU arm the event
    reasons, the compute processes that are not in the tree, and the SM clock. A sampler thread that dies makes
    the window not clean with reason `sampler-failed`. `thread=False` takes no sampler thread: tests call
    `tick()` themselves."""

    def __init__(self, *, pid: int | None = None, device: str = "cpu", machine: Machine | None = None,
                 thresholds: Thresholds | None = None, present: set[str] | None = None,
                 sample_s: float = WINDOW_SAMPLE_S, thread: bool = True, join_timeout_s: float | None = None) -> None:
        self.m = machine or get_machine()
        self.pid = self.m.self_pid if pid is None else pid
        self.device, self.th, self.sample_s = device, thresholds or Thresholds(), sample_s
        self.present = set(present) if present is not None else present_sources(self.m, device)
        if device == "gpu":
            self.present.add("gpu")
        self.use_thread = thread
        self.join_timeout_s = join_timeout_s if join_timeout_s is not None else max(5.0, sample_s * 10)
        self.readings: list[dict] = []
        self.error: BaseException | None = None
        self.stuck = False
        self.result: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _read(self, gpu_read: bool) -> dict:
        m = self.m
        table = process_table(m)
        members = set() if table is None else tree_of(table, self.pid)
        # a tree member's ticks plus what its waited-for children left: an exiting child moves from one to the other
        tree = None if table is None else {p: table[p][1] + table[p][3] for p in members}
        r: dict[str, Any] = {"t": m.clock(), "cpu": read_cpu_stat(m), "psi": read_psi(m, "cpu"),
                             "psim": read_psi(m, "memory"),
                             "others": {} if table is None else {p: (v[1], v[2]) for p, v in table.items()
                                                                 if p not in members},
                             "swap": read_swap_pages(m), "thermal": read_thermal(m), "tree": tree,
                             "gpu": None, "apps": None}
        r["gpu_due"] = False
        if self.device == "gpu" and gpu_read:
            r["gpu_due"] = True
            r["gpu"] = read_gpu(m, GPU_WINDOW_FIELDS)
            apps = read_gpu_apps(m)
            r["apps"] = None if apps is None else [p for p, _ in apps]
        return r

    def tick(self, final: bool = False) -> None:
        """One reading. The GPU is read on the first, every `GPU_SAMPLE_S` seconds and on the last one."""
        every = max(1, round(GPU_SAMPLE_S / self.sample_s))
        self.readings.append(self._read(final or len(self.readings) % every == 0))

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self.sample_s):
                self.tick()
        except BaseException as e:  # noqa: BLE001 - the window reports it as sampler-failed
            self.error = e

    def __enter__(self) -> "Window":
        self._t0 = self.m.clock()
        self.tick()
        if self.use_thread:
            self._thread = threading.Thread(target=self._loop, name="isolation-sampler", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.join_timeout_s)
            self.stuck = self._thread.is_alive()
        if self.error is None and not self.stuck:
            try:
                self.tick(final=True)
            except BaseException as e:  # noqa: BLE001
                self.error = e
        self.result = self._evaluate()

    def _top_others(self, window_s: float) -> list[dict]:
        """The processes outside the tree that used the most CPU in the window, as percent of one core."""
        if window_s <= 0:
            return []
        used: dict[int, list] = {}
        for a, b in zip(self.readings, self.readings[1:]):
            for pid, (ticks, name) in b["others"].items():
                gained = ticks - a["others"][pid][0] if pid in a["others"] else ticks
                if gained > 0:
                    used.setdefault(pid, [name, 0])[1] += gained
        rows = [{"pid": pid, "name": n, "percent": round(t / (self.m.clk_tck * window_s) * 100, 1)}
                for pid, (n, t) in used.items()]
        return sorted((r for r in rows if r["percent"] >= 1.0), key=lambda r: -r["percent"])[:5]

    def _evaluate(self) -> dict:
        th, rs = self.th, self.readings
        window_s = (rs[-1]["t"] - rs[0]["t"]) if len(rs) >= 2 else 0.0
        reasons: list[str] = []
        unavailable: list[str] = []

        def lost(source: str) -> None:
            if source not in unavailable:
                unavailable.append(source)
            if source in self.present:
                reasons.append(f"unmeasured:{source}")

        if self.error is not None or self.stuck:
            reasons.append("sampler-failed")
        if len(rs) < 2 or window_s <= 0:
            reasons.append("no-samples")

        # CPU: system busy minus the benchmark tree, as a share of total machine capacity
        shares: list[float] = []
        other_ticks = total_ticks = 0
        root_seen = any(r["tree"] and self.pid in r["tree"] for r in rs)
        unread_pairs = 0
        for a, b in zip(rs, rs[1:]):
            if a["cpu"] is None or b["cpu"] is None or a["tree"] is None or b["tree"] is None:
                unread_pairs += 1
                continue
            d_total, d_busy = b["cpu"][0] - a["cpu"][0], b["cpu"][1] - a["cpu"][1]
            if d_total <= 0:
                continue    # no ticks elapsed: a zero would be a guess
            # Aggregate difference: a child that exited between the two readings left its ticks in its parent's
            # waited-for time, so the sum does not drop. Per-pid differences would lose them.
            d_tree = max(sum(b["tree"].values()) - sum(a["tree"].values()), 0)
            other = max(d_busy - d_tree, 0)
            shares.append(other / d_total)
            other_ticks += other
            total_ticks += d_total
        if not shares or unread_pairs:
            lost("cpu-stat")
        if not root_seen:
            lost("proc-tree")
        share_mean = other_ticks / total_ticks if total_ticks else None
        share_max = max(shares) if shares else None
        if share_mean is not None and share_mean > th.other_cpu_mean_max:
            reasons.append(f"other-cpu-mean {share_mean:.1%} > {th.other_cpu_mean_max:.0%}")
        if share_max is not None and share_max > th.other_cpu_sample_max:
            reasons.append(f"other-cpu-peak {share_max:.1%} > {th.other_cpu_sample_max:.0%}")

        def counter(key: str, sub: Callable[[Any], float]) -> float | None:
            """Last minus first reading of a monotonic counter; None unless both ends were read."""
            if len(rs) < 2 or rs[0][key] is None or rs[-1][key] is None:
                return None
            return sub(rs[-1][key]) - sub(rs[0][key])

        psi = counter("psi", lambda x: x["total"])
        psi_ms = psi / 1000 if psi is not None else None
        psim = counter("psim", lambda x: x["total"])
        psim_ms = psim / 1000 if psim is not None else None
        # PSI is recorded, not judged: a 12-thread run stalls about 9% of the time by itself, so no limit
        # separates the run from a neighbour. A missing PSI file is listed as unavailable and nothing more.
        for name, val in (("psi-cpu", psi_ms), ("psi-memory", psim_ms)):
            if val is None:
                unavailable.append(name)
        swap = counter("swap", lambda x: x["pswpin"] + x["pswpout"])
        if swap is None:
            lost("vmstat")
        elif swap > th.swap_pages_max:
            reasons.append(f"swap {swap:.0f} pages")
        thermal = counter("thermal", lambda x: x["core"] + x["package"])
        if thermal is None:
            lost("thermal")
        elif thermal > th.thermal_events_max:
            reasons.append(f"thermal {thermal:.0f} throttle events")

        reasons_seen: list[str] = []
        contended: list[str] = []
        foreign: list[int] = []
        clocks: list[float] = []
        capped = False
        if self.device == "gpu":
            due = [r for r in rs if r["gpu_due"]]        # the GPU is read every GPU_SAMPLE_S, not on every tick
            gpu_ok = bool(due) and all(r["gpu"] is not None and r["gpu"]["reasons"] is not None
                                       and r["apps"] is not None for r in due)
            if not gpu_ok:
                lost("gpu")
            for r in due:
                bits = None if r["gpu"] is None else r["gpu"]["reasons"]
                if bits is not None:
                    for n in reason_names(bits & ~GPU_REASON_IDLE):
                        if n not in reasons_seen:
                            reasons_seen.append(n)
                    capped = capped or bool(bits & GPU_REASON_POWER_CAP)
                    for n in contended_reasons(bits):
                        if n not in contended:
                            contended.append(n)
                clk = _num((r["gpu"] or {}).get("clocks.sm"))
                if clk is not None:
                    clocks.append(clk)
                if r["apps"] is not None and r["tree"] is not None:
                    foreign += [p for p in r["apps"] if p not in r["tree"] and p not in foreign]
            if contended:
                reasons.append(f"gpu-clock-reasons {contended}")
            if len(foreign) > th.foreign_gpu_pids_max:
                reasons.append(f"gpu-foreign-pids {foreign}")

        return {"clean": not reasons, "reasons": reasons, "other_cpu_share_max": share_max,
                "other_top_processes": self._top_others(window_s), "psi_memory_stall_ms": psim_ms,
                "other_cpu_share_mean": share_mean, "psi_cpu_stall_ms": psi_ms, "swap_pages": swap,
                "thermal_events": thermal, "gpu_reasons_seen": reasons_seen, "gpu_power_capped": capped, "gpu_foreign_pids": foreign,
                "sm_clock_min_mhz": min(clocks) if clocks else None,
                "sm_clock_max_mhz": max(clocks) if clocks else None, "window_s": round(window_s, 3),
                "samples": len(rs), "unavailable": unavailable}


# the arm: preflight once, then each problem inside a window


def _tool_versions(m: Machine) -> dict:
    out: dict[str, Any] = {"python": sys.version.split()[0], "kernel": platform.release()}
    if gpu_present(m):
        rc, text = m.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        out["nvidia_driver"] = text.strip().splitlines()[0] if rc == 0 and text.strip() else None
    if m.has("powerprofilesctl"):
        rc, text = m.run(["powerprofilesctl", "version"])
        out["powerprofilesctl"] = text.strip() if rc == 0 and text.strip() else None
    return out


@dataclasses.dataclass
class Arm:
    """What a started arm remembers: the thresholds, the sources the preflight found, and where to write."""

    name: str
    device: str
    report: Report
    thresholds: Thresholds
    env_path: pathlib.Path | None
    machine: Machine
    retries: int = 2
    allow_battery: bool = False
    allow_noisy: bool = False
    tuned: dict | None = None
    log: Callable[[str], None] = print
    session_id: str = ""

    def run(self, fn: Callable[[], T], **kw: Any) -> tuple[T, dict]:
        return run_isolated(fn, device=self.device, retries=self.retries, arm=self,
                            allow_battery=self.allow_battery, log=self.log, **kw)

    def finish(self) -> None:
        """Write the closing snapshot. A missing or corrupt env file is logged and left alone: this runs while
        another error may be propagating, and raising here would replace it and skip the cleanup after it."""
        if self.env_path is None:
            return
        try:
            doc = json.loads(self.env_path.read_text())
            doc["sessions"][-1]["snapshot_end"] = snapshot(self.machine, sample_s=0)
            doc["sessions"][-1]["ended"] = dt.datetime.fromtimestamp(self.machine.wall()).isoformat(timespec="seconds")
            self.env_path.write_text(json.dumps(doc, indent=2) + "\n")
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            self.log(f"!! could not write the closing snapshot to {self.env_path} ({type(e).__name__}: {e})")


def begin_arm(arm: str, *, device: str, need_ram_gb: float | None, runs_dir: pathlib.Path | None,
              machine: Machine | None = None, thresholds: Thresholds | None = None, retries: int = 2,
              allow_noisy: bool = False, allow_battery: bool = False, allow: tuple[str, ...] = (),
              tuned: dict | None = None, versions: dict | None = None, log: Callable[[str], None] = print,
              sample_s: float = PREFLIGHT_SAMPLE_S) -> Arm:
    """Preflight an arm once and write `runs_dir/ARM.env.json`.

    Raises IsolationError, with the report, when a check FAILs and `allow_noisy` is not set. With it, the run
    goes on and the file records `preflight_failed_allowed: true`. A resumed arm appends a session."""
    m, th = machine or get_machine(), thresholds or Thresholds()
    session_id = uuid.uuid4().hex[:12]
    report = preflight(device, need_ram_gb, allow, machine=m, thresholds=th, sample_s=sample_s,
                       allow_battery=allow_battery)
    log(report.text())
    env_path = None
    if runs_dir is not None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        env_path = runs_dir / f"{arm}.env.json"
        try:
            doc = json.loads(env_path.read_text()) if env_path.exists() else {"arm": arm, "sessions": []}
        except ValueError as e:
            raise IsolationError(f"{env_path} is not valid JSON ({e}); move it aside to start a new record") from e
        doc["sessions"].append({
            "session_id": session_id, "started": dt.datetime.fromtimestamp(m.wall()).isoformat(timespec="seconds"), "device": device,
            "preflight": report.to_dict(), "preflight_failed_allowed": bool(report.failed) and allow_noisy,
            "allow_noisy": allow_noisy, "allow_battery": allow_battery,
            "bench_quiet": os.environ.get("BENCH_QUIET_ACTIVE") == "1", "retries": retries,
            "need_ram_gb": need_ram_gb, "thresholds": dataclasses.asdict(th), "tuned": tuned,
            "snapshot_start": report.snapshot, "tool_versions": {**_tool_versions(m), **(versions or {})}})
        env_path.write_text(json.dumps(doc, indent=2) + "\n")
    if report.failed and not allow_noisy:
        raise IsolationError("the machine is not fit for a timed run:\n" + report.text() +
                             "\nfix the offenders, use --tune for the power profile, or continue with --allow-noisy "
                             "(every row then records its window)")
    return Arm(arm, device, report, th, env_path, m, retries, allow_battery, allow_noisy, tuned, log, session_id)


def run_isolated(fn: Callable[[], T], *, device: str, retries: int = 2, arm: Arm | None = None,
                 allow_battery: bool = False, log: Callable[[str], None] = print, machine: Machine | None = None,
                 thresholds: Thresholds | None = None, max_quiet_wait_s: float = 60.0, quiet_sample_s: float = 1.0,
                 poll_s: float = 3.0, window: Callable[..., Any] | None = None,
                 window_sample_s: float = WINDOW_SAMPLE_S, pid: int | None = None,
                 on_pause: Callable[[], None] | None = None,
                 on_resume: Callable[[], None] | None = None) -> tuple[T, dict]:
    """Run `fn` once the machine is quiet, inside a Window, and again while the window is noisy.

    Returns fn's result and `power.run_on_mains`' info plus `isolation`: the window dict, `retries` used,
    `noisy` (the kept window was not clean), and the quiet gate's `cpu_busy_before` and `waited_s`. After
    `retries` repeats the row is kept with its reasons; it is never dropped and never reported clean."""
    if retries < 0:
        raise ValueError("retries must be 0 or more")
    m = machine or (arm.machine if arm else get_machine())
    th = thresholds or (arm.thresholds if arm else Thresholds())
    present = arm.report.present if arm else None
    window = window or Window
    attempts = 0
    while True:
        gate = quiet_gate(device, machine=m, thresholds=th, max_wait_s=max_quiet_wait_s, sample_s=quiet_sample_s,
                          poll_s=poll_s)
        if not gate["ok"]:
            log(f"the machine did not go quiet in {gate['waited_s']:.0f} s ({'; '.join(gate['reasons'])}): running anyway")

        def timed() -> tuple[T, dict]:
            with window(pid=pid, device=device, machine=m, thresholds=th, present=present,
                        sample_s=window_sample_s) as w:
                value = fn()
            return value, w.result

        hooks = {k: v for k, v in (("on_pause", on_pause), ("on_resume", on_resume)) if v is not None}
        (result, win), pinfo = power.run_on_mains(timed, allow_battery=allow_battery, log=log, **hooks)
        if win["clean"] or attempts >= retries:
            break
        attempts += 1
        log(f"noisy window ({'; '.join(win['reasons'])}): repeating, {attempts} of {retries}")
    iso = {**win, "retries": attempts, "noisy": not win["clean"], "cpu_busy_before": gate["cpu_busy_before"],
           "waited_s": gate["waited_s"], "gate_ok": gate["ok"],
           # which start of the arm wrote this row, and whether that start went on past a failed preflight
           "session": arm.session_id if arm else None, "allow_noisy": bool(arm and arm.allow_noisy),
           "preflight_failed_allowed": bool(arm and arm.allow_noisy and arm.report.failed)}
    return result, {**pinfo, "isolation": iso}


def need_ram_gb(model: str | os.PathLike) -> float | None:
    """Free memory an arm wants: RAM_FACTOR times the model file (llama.cpp's resident set is 1.4 times the file).

    None when the size cannot be read (missing, unreadable, not a regular file): the preflight then FAILs
    `memory`, it never reads the need as zero."""
    try:
        st = os.stat(model)
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return round(st.st_size / 1e9 * RAM_FACTOR, 2)


# --tune: the power profile without root


class _Restore:
    """Puts the power profile back once, from try/finally, atexit or a signal, whichever comes first."""

    def __init__(self, m: Machine, previous: str, log: Callable[[str], None]) -> None:
        self.m, self.previous, self.log, self.done = m, previous, log, False

    def __call__(self) -> None:
        if self.done:
            return
        self.done = True
        rc, _ = self.m.run(["powerprofilesctl", "set", self.previous])
        rc2, now = self.m.run(["powerprofilesctl", "get"])
        if rc == 0 and rc2 == 0 and now.strip() == self.previous:
            self.log(f"power profile restored to {self.previous}")
        else:
            self.log(f"!! could not restore the power profile to {self.previous} (now {now.strip() or 'unreadable'}): "
                     f"run `powerprofilesctl set {self.previous}`")


@contextlib.contextmanager
def tuned(machine: Machine | None = None, log: Callable[[str], None] = print) -> Iterator[dict]:
    """`powerprofilesctl set performance`, verified by reading it back, and restored on every way out.

    The restore runs in `finally`, at exit and on SIGINT or SIGTERM (the signal is then delivered again with the
    previous handler, so Ctrl-C still stops the run). Yields {"previous": ..., "set": "performance"}."""
    m = machine or get_machine()
    if not m.has("powerprofilesctl"):
        raise IsolationError("--tune needs `powerprofilesctl` (power-profiles-daemon), which is not on PATH")
    rc, out = m.run(["powerprofilesctl", "get"])
    previous = out.strip()
    if rc != 0 or not previous:
        raise IsolationError("--tune: `powerprofilesctl get` failed, so the profile to restore is unknown")
    restore = _Restore(m, previous, log)
    atexit.register(restore)
    old: dict[int, Any] = {}

    def on_signal(signum: int, frame: object) -> None:
        restore()
        signal.signal(signum, old[signum])
        signal.raise_signal(signum)

    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGINT, signal.SIGTERM):
            old[s] = signal.signal(s, on_signal)
    try:
        rc, _ = m.run(["powerprofilesctl", "set", "performance"])
        rc2, now = m.run(["powerprofilesctl", "get"])
        if rc != 0 or rc2 != 0 or now.strip() != "performance":
            raise IsolationError(f"--tune: `powerprofilesctl set performance` did not take (profile reads "
                                 f"{now.strip() or 'nothing'}); the run stops instead of measuring on it")
        log(f"power profile: {previous} -> performance")
        yield {"previous": previous, "set": "performance"}
    finally:
        restore()
        for s, h in old.items():
            signal.signal(s, h)
        atexit.unregister(restore)


def add_arguments(parser: Any) -> None:
    parser.add_argument("--tune", action="store_true",
                        help="set the power profile to performance for the run and restore it after")
    parser.add_argument("--allow-noisy", action="store_true",
                        help="continue after a FAIL preflight (it is recorded in ARM.env.json)")
    parser.add_argument("--isolation-retries", type=int, default=2,
                        help="repeat a problem whose window was noisy up to this many times (default %(default)s)")


_FROM_FILE = object()


@contextlib.contextmanager
def arm_session(args: Any, *, arm: str, device: str, model: str | os.PathLike, runs_dir: pathlib.Path,
                versions: dict | None = None, log: Callable[[str], None] = lambda m: print(m, flush=True),
                machine: Machine | None = None, ram_gb: Any = _FROM_FILE) -> Iterator[Arm]:
    """The runners' wrapper: tune when asked, preflight, yield the Arm, write the closing snapshot.

    The memory the preflight wants is read from the model file unless `ram_gb` is given: an engine that has
    already loaded its weights (no file to size) passes what it still needs. `None` there means unknown, and
    the preflight FAILs `memory` as it does for a file it cannot read."""
    try:
        with contextlib.ExitStack() as stack:
            info = stack.enter_context(tuned(machine, log)) if getattr(args, "tune", False) else None
            started = begin_arm(arm, device=device, need_ram_gb=need_ram_gb(model) if ram_gb is _FROM_FILE else ram_gb, runs_dir=runs_dir,
                                machine=machine, retries=args.isolation_retries,
                                allow_noisy=args.allow_noisy, allow_battery=getattr(args, "allow_battery", False),
                                tuned=info, versions=versions, log=log)
            try:
                yield started
            finally:
                started.finish()
    except IsolationError as e:
        raise SystemExit(str(e)) from e
