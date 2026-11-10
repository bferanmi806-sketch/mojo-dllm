"""Shared fakes: nothing here touches the machine, the network or a GPU."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from launcher.probes import Probes
from launcher.verified import VerifiedStore

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import isolation  # noqa: E402  (bench/isolation.py)


@pytest.fixture(autouse=True)
def no_git_hook_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test as if git had not called it.

    git exports GIT_DIR, GIT_WORK_TREE and GIT_INDEX_FILE to its hooks, and the pre-push hook runs this suite.
    A test that then ran `git init` or `git commit` in a temporary directory acted on the repository being
    pushed: it added commits to the branch and set core.bare in the shared config
    (tests/python/test_git_env_isolation.py).
    """
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(name)

CPUINFO = "processor\t: 0\nflags\t\t: fpu sse4_2 avx avx2 fma\n"
MEMINFO = "MemTotal:       16000000 kB\nMemAvailable:   12000000 kB\n"
GPU_TEXT = "NVIDIA GeForce RTX 3050 6GB Laptop GPU, 8.6, 5120\n"


def make_probes(**overrides: object) -> Probes:
    """A machine where every doctor check passes; tests override one field."""
    base = Probes(
        system="Linux",
        machine="x86_64",
        glibc="2.39",
        cpuinfo=CPUINFO,
        meminfo=MEMINFO,
        uv_path="/usr/bin/uv",
        mojo_version_output="Mojo 1.1.0 (abc123)\n",
        mojo_pin="1.1.0",
        binary_mtime=200.0,
        source_mtimes=(100.0, 150.0),
        gpu_text=GPU_TEXT,
        driver_text="580.65.06\n",
        cached_model=Path("/cache/llada.gguf"),
        verified=True,
    )
    return dataclasses.replace(base, **overrides)


class FakeHub:
    """Stands in for launcher.hub.Hub. `cached` maps (repo, file) to a path."""

    def __init__(self, tmp_path: Path, files: dict[str, int] | None = None) -> None:
        self.cache_dir = tmp_path / "hub"
        self.offline = False
        self.files = files if files is not None else {
            "LLaDA-8B-Instruct.Q4_K_M.gguf": 4_925_226_272,
            "LLaDA-8B-Instruct.Q8_0.gguf": 8_522_513_696,
            "README.md": 3538,
        }
        self.cached: dict[tuple[str, str], Path] = {}
        self.content = b"gguf bytes"
        self.sha256 = hashlib.sha256(self.content).hexdigest()
        self.downloads: list[tuple[str, str]] = []
        self.sha_requests = 0
        self._tmp = tmp_path

    def list_files(self, repo: str) -> dict[str, int]:
        return self.files

    def cached_path(self, repo: str, filename: str) -> Path | None:
        return self.cached.get((repo, filename))

    def download(self, repo: str, filename: str) -> Path:
        self.downloads.append((repo, filename))
        path = self._tmp / filename
        path.write_bytes(self.content)
        self.cached[(repo, filename)] = path
        return path

    def expected_sha256(self, repo: str, filename: str) -> str | None:
        self.sha_requests += 1
        return self.sha256


@pytest.fixture
def hub(tmp_path: Path) -> FakeHub:
    return FakeHub(tmp_path)


@pytest.fixture
def store(tmp_path: Path) -> VerifiedStore:
    return VerifiedStore(tmp_path / "state" / "verified.json")


# A machine that is not the real one. `isolation` reads /proc, /sys, nvidia-smi and powerprofilesctl through a
# Machine, so every test below runs against this fake and none touches the GPU or the real power profile.


class FakeMachine(isolation.Machine):
    """Fake /proc and /sys trees under `root`, a fake clock, and scripted commands.

    `advance(s)` moves the clock and the counters: each process ticks at its `rate` (clock ticks per second,
    100 = one core), `extra_busy` is CPU use by processes the table does not list, and the counters for PSI,
    swap and thermal move by their rates. With `auto=True` every read of /proc/stat advances half a second, so
    a sampler thread sees time pass; tests that sample by hand set `auto=False`."""

    NCPU_TICKS = 100

    def __init__(self, root: Path, *, ncpu: int = 4, auto: bool = True, nvidia: bool = True) -> None:
        super().__init__(proc=root / "proc", sysfs=root / "sys", power_root=root / "power", sleep=self.advance,
                         clock=lambda: self.now, wall=lambda: 1794220545.0, clk_tck=100, self_pid=4242)
        self.root, self.ncpu, self.auto, self.now = root, ncpu, auto, 0.0
        self.total = self.busy = self.iowait = 0
        self.procs: dict[int, dict] = {}
        self.extra_busy = 0.0
        self.psi_avg10, self.psi_total, self.psi_rate = 0.0, 0.0, 0.0
        self.swap_rate, self.swapin_rate, self.pswpin, self.pswpout = 0.0, 0.0, 100, 200
        self.iowait_rate = 0.0
        self.psi_mem_rate, self.psi_mem_total = 0.0, 1.0
        self.thermal = {"core": 3, "package": 5}
        self.mem_available_kb, self.swap_total_kb, self.swap_free_kb = 16 * 1048576, 2097148, 2097148
        self.governor, self.epp, self.pstate = "performance", "performance", "active"
        self.platform_profile = "performance"
        self.profile = "performance"
        self.profile_set_works = True
        self.programs = {"powerprofilesctl", "systemctl", "crontab"} | ({"nvidia-smi"} if nvidia else set())
        self.gpu = {"reasons": 0x1, "util": 0, "mem": 120, "temp": 45, "sm": 210, "max_sm": 2100,
                    "persistence": "Disabled", "compute_mode": "Default", "power": 8.5}
        self.gpu_apps: list[tuple[int, int]] = []
        self.gpu_ok = True
        self.timers = [("sysstat-collect.timer", 600)]
        self.crontab_lines = ["* * * * * /bin/true"]
        self.calls: list[list[str]] = []
        self.add_proc(4242, 1, "python3", 0)
        self.add_proc(1, 0, "systemd", 0)
        self.set_power(plugged=True)
        self.flush()

    # state
    def add_proc(self, pid: int, ppid: int, name: str, rate: float = 0.0, ticks: int = 0,
                 sys_share: float = 0.0) -> None:
        """`sys_share` is the part of `rate` spent in the kernel (stime); `cticks` is what waited-for children left."""
        self.procs[pid] = {"ppid": ppid, "name": name, "rate": rate, "ticks": float(ticks), "cticks": 0.0,
                           "sys_share": sys_share, "stime": 0.0}
        if hasattr(self, "calls"):
            self.flush()

    def remove_proc(self, pid: int) -> None:
        """The process exits and its parent waits for it: its ticks move into the parent's cutime and cstime."""
        pr = self.procs.pop(pid)
        parent = self.procs.get(pr["ppid"])
        if parent is not None:
            parent["cticks"] += pr["ticks"] + pr["stime"] + pr["cticks"]
        subprocess.run(["rm", "-rf", str(self.proc / str(pid))], check=True)
        self.flush()

    def set_power(self, plugged: bool) -> None:
        for d in (self.root / "power" / "ADP1", self.root / "power" / "BAT1"):
            d.mkdir(parents=True, exist_ok=True)
        (self.root / "power/ADP1/type").write_text("Mains\n")
        (self.root / "power/ADP1/online").write_text("1\n" if plugged else "0\n")
        (self.root / "power/BAT1/type").write_text("Battery\n")
        (self.root / "power/BAT1/status").write_text("Full\n" if plugged else "Discharging\n")

    def advance(self, seconds: float) -> None:
        self.now += seconds
        ticks = self.ncpu * self.NCPU_TICKS * seconds
        used = self.extra_busy * seconds
        for pr in self.procs.values():
            pr["ticks"] += pr["rate"] * seconds * (1 - pr["sys_share"])
            pr["stime"] += pr["rate"] * seconds * pr["sys_share"]
            used += pr["rate"] * seconds
        used = min(used, ticks)
        self.total += int(ticks)
        self.busy += int(used)
        self.iowait += int(self.iowait_rate * seconds)
        self.psi_total += self.psi_rate * seconds * 1e6
        self.psi_mem_total += self.psi_mem_rate * seconds * 1e6
        self.pswpout += int(self.swap_rate * seconds)
        self.pswpin += int(self.swapin_rate * seconds)
        self.flush()

    def flush(self) -> None:
        p, s = self.proc, self.sys
        (p / "pressure").mkdir(parents=True, exist_ok=True)
        idle = self.total - self.busy - self.iowait
        lines = [f"cpu  {self.busy} 0 0 {idle} {self.iowait} 0 0 0 0 0"]
        lines += [f"cpu{i} {self.busy // self.ncpu} 0 0 {idle // self.ncpu} 0 0 0 0 0 0" for i in range(self.ncpu)]
        (p / "stat").write_text("\n".join(lines) + "\nintr 0\n")
        (p / "pressure/cpu").write_text(f"some avg10={self.psi_avg10:.2f} avg60=0.00 avg300=0.00 "
                                        f"total={int(self.psi_total)}\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
        (p / "pressure/memory").write_text(f"some avg10=0.00 avg60=0.00 avg300=0.00 total={int(self.psi_mem_total)}\n"
                                           "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
        (p / "vmstat").write_text(f"nr_free_pages 1\npswpin {self.pswpin}\npswpout {self.pswpout}\n")
        (p / "meminfo").write_text(f"MemTotal: 33000000 kB\nMemAvailable: {self.mem_available_kb} kB\n"
                                   f"SwapTotal: {self.swap_total_kb} kB\nSwapFree: {self.swap_free_kb} kB\n")
        (p / "loadavg").write_text("0.10 0.20 0.30 1/200 99\n")
        for pid, pr in self.procs.items():
            d = p / str(pid)
            d.mkdir(exist_ok=True)
            tail = (["S", str(pr["ppid"])] + ["0"] * 9 + [str(int(pr["ticks"])), str(int(pr["stime"]))]
                    + [str(int(pr["cticks"])), "0"] + ["0"] * 8)
            (d / "stat").write_text(f"{pid} ({pr['name']}) " + " ".join(tail) + "\n")
        for i in range(self.ncpu):
            c = s / f"devices/system/cpu/cpu{i}"
            (c / "cpufreq").mkdir(parents=True, exist_ok=True)
            (c / "thermal_throttle").mkdir(exist_ok=True)
            (c / "cpufreq/scaling_governor").write_text(self.governor + "\n")
            (c / "cpufreq/energy_performance_preference").write_text(self.epp + "\n")
            (c / "cpufreq/scaling_cur_freq").write_text("3400000\n")
            (c / "thermal_throttle/core_throttle_count").write_text(f"{self.thermal['core'] if i == 0 else 0}\n")
            (c / "thermal_throttle/package_throttle_count").write_text(f"{self.thermal['package'] if i == 0 else 0}\n")
        ps = s / "devices/system/cpu/intel_pstate"
        ps.mkdir(parents=True, exist_ok=True)
        (ps / "status").write_text(self.pstate + "\n")
        (ps / "no_turbo").write_text("0\n")
        (s / "firmware/acpi").mkdir(parents=True, exist_ok=True)
        (s / "firmware/acpi/platform_profile").write_text(self.platform_profile + "\n")

    # the isolation.Machine interface
    def read(self, path: Path) -> str | None:
        if self.auto and path == self.proc / "stat":
            self.advance(0.5)
        return super().read(path)

    def has(self, program: str) -> bool:
        return program in self.programs

    def run(self, cmd: list[str], timeout: float = 15.0) -> tuple[int, str]:
        self.calls.append(cmd)
        prog = cmd[0]
        if prog not in self.programs:
            return 127, ""
        if prog == "powerprofilesctl":
            if cmd[1:2] == ["get"]:
                return 0, self.profile + "\n"
            if cmd[1:2] == ["set"]:
                if self.profile_set_works:
                    self.profile = cmd[2]
                    self.platform_profile = cmd[2]
                    self.flush()
                return 0, ""
            return 0, "0.30\n"
        if prog == "nvidia-smi":
            if not self.gpu_ok:
                return 9, ""
            g = self.gpu
            if cmd[1].startswith("--query-compute-apps"):
                return 0, "".join(f"{pid}, {mib}\n" for pid, mib in self.gpu_apps)
            fields = cmd[1].split("=", 1)[1].split(",")
            if "clocks_event_reasons.active" not in fields and "clocks_throttle_reasons.active" not in fields \
                    and "driver_version" not in fields:
                return 2, ""
            vals = {"clocks_event_reasons.active": f"0x{g['reasons']:016x}", "clocks_throttle_reasons.active": f"0x{g['reasons']:016x}",
                    "utilization.gpu": g["util"], "memory.used": g["mem"], "temperature.gpu": g["temp"],
                    "clocks.sm": g["sm"], "clocks.max.sm": g["max_sm"], "persistence_mode": g["persistence"],
                    "compute_mode": g["compute_mode"], "power.draw": g["power"], "driver_version": "595.99.02"}
            return 0, ", ".join(str(vals[f]) for f in fields) + "\n"
        if prog == "systemctl":
            import json
            entries = [{"next": int((1794220545.0 + off) * 1e6), "unit": unit} for unit, off in self.timers]
            return 0, json.dumps(entries)
        if prog == "crontab":
            return 0, "\n".join(self.crontab_lines) + "\n"
        return 127, ""


@pytest.fixture
def fake_machine(tmp_path: Path) -> FakeMachine:
    return FakeMachine(tmp_path / "machine")


@pytest.fixture(autouse=True)
def quiet_fake_machine(fake_machine: FakeMachine) -> object:
    """Every test starts on a quiet fake machine, so no test reads the real /proc, the GPU or the power profile."""
    previous = isolation.use_machine(fake_machine)
    yield fake_machine
    isolation.use_machine(previous)
