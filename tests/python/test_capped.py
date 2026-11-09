"""scripts/capped.sh runs a command in its own memory-capped systemd scope, or refuses to run it.

The first tests never start systemd: they source the script's functions, or run it with `--dry-run`, or put a
stub `systemd-run` first on PATH. The tests marked `REAL` start a real scope and run only where the user manager
can enforce memory limits.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "capped.sh"
BASH = shutil.which("bash") or "/bin/bash"  # absolute: tests replace PATH
GiB = 1024**3


def sourced(call: str, tmp_path: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run one shell snippet after sourcing the script, which must not run main when sourced."""
    full = {**os.environ, **(env or {})}
    return subprocess.run(
        [BASH, "-c", f'source "{SCRIPT}"; {call}'], capture_output=True, text=True, env=full, cwd=tmp_path, check=False
    )


def meminfo(tmp_path: Path, available_kb: int | None, name: str = "meminfo") -> Path:
    lines = ["MemTotal:       31000000 kB\n", "MemFree:         1000000 kB\n"]
    if available_kb is not None:
        lines.append(f"MemAvailable:   {available_kb} kB\n")
    path = tmp_path / name
    path.write_text("".join(lines))
    return path


def controllers(tmp_path: Path, text: str | None) -> Path:
    path = tmp_path / "cgroup.controllers"
    if text is not None:
        path.write_text(text)
    return path


def run_script(
    args: list[str], tmp_path: Path, *, avail_gib: int = 40, ctl: str | None = "cpu memory pids", path: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "CAPPED_MEMINFO": str(meminfo(tmp_path, avail_gib * 1024 * 1024)),
        "CAPPED_CONTROLLERS_FILE": str(controllers(tmp_path, ctl)),
    }
    if path is not None:
        env["PATH"] = path
    return subprocess.run([BASH, str(SCRIPT), *args], capture_output=True, text=True, env=env, cwd=tmp_path, check=False)


def stub_dir(tmp_path: Path, rc: int = 0, tools: tuple[str, ...] = ("awk", "grep", "id")) -> Path:
    """A PATH directory with a recording `systemd-run` stub plus symlinks to the few tools the script uses."""
    d = tmp_path / "bin"
    d.mkdir()
    for name in tools:
        found = shutil.which(name)
        assert found, f"{name} is needed to run this test"
        (d / name).symlink_to(found)
    stub = d / "systemd-run"
    stub.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$@" > "{tmp_path}/stub-args"\n'
        f"exit {rc}\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return d


# parse_size ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("27G", 27 * GiB), ("512M", 512 * 1024**2), ("100K", 100 * 1024), ("7", 7), ("1G", GiB)],
)
def test_parse_size_reads_binary_units(tmp_path: Path, text: str, expected: int) -> None:
    out = sourced(f"parse_size {text}", tmp_path)
    assert out.returncode == 0
    assert out.stdout.strip() == str(expected)


@pytest.mark.parametrize("text", ["0", "0G", "", "1.5G", "27GB", "-1G", "G", "abc", "99999999999999999999G", "1 G"])
def test_parse_size_refuses_everything_else(tmp_path: Path, text: str) -> None:
    out = sourced(f'parse_size "{text}"', tmp_path)
    assert out.returncode != 0
    assert out.stdout == ""


# mem_available_bytes ------------------------------------------------------------------------------------------


def test_mem_available_reads_memavailable_not_memfree(tmp_path: Path) -> None:
    path = meminfo(tmp_path, 12_000_000)
    out = sourced(f'mem_available_bytes "{path}"', tmp_path)
    assert out.returncode == 0
    assert out.stdout.strip() == str(12_000_000 * 1024)


def test_mem_available_refuses_a_file_without_the_line(tmp_path: Path) -> None:
    path = meminfo(tmp_path, None)
    out = sourced(f'mem_available_bytes "{path}"', tmp_path)
    assert out.returncode != 0
    assert out.stdout == ""


def test_mem_available_refuses_a_missing_file(tmp_path: Path) -> None:
    out = sourced(f'mem_available_bytes "{tmp_path}/nope"', tmp_path)
    assert out.returncode != 0


# the command that would run -----------------------------------------------------------------------------------


def dry(args: list[str], tmp_path: Path, **kw: object) -> subprocess.CompletedProcess[str]:
    return run_script([*args, "--dry-run", "--", "echo", "hi there"], tmp_path, **kw)  # type: ignore[arg-type]


def test_dry_run_prints_a_user_scope_that_is_collected(tmp_path: Path) -> None:
    out = dry(["--mem", "27G"], tmp_path)
    assert out.returncode == 0, out.stderr
    words = out.stdout.split()
    assert words[:3] == ["systemd-run", "--user", "--scope"]
    assert "--collect" in words


def test_dry_run_sets_the_hard_memory_cap(tmp_path: Path) -> None:
    assert "MemoryMax=27G" in dry(["--mem", "27G"], tmp_path).stdout.split()


def test_dry_run_forbids_swap(tmp_path: Path) -> None:
    assert "MemorySwapMax=0" in dry(["--mem", "27G"], tmp_path).stdout.split()


def test_dry_run_kills_the_whole_scope_on_an_oom(tmp_path: Path) -> None:
    assert "OOMPolicy=kill" in dry(["--mem", "27G"], tmp_path).stdout.split()


def test_dry_run_never_sets_memoryhigh(tmp_path: Path) -> None:
    # MemoryHigh below the cap throttles a process with nothing to reclaim to a standstill (state D) instead of killing it.
    assert "MemoryHigh" not in dry(["--mem", "27G"], tmp_path).stdout


def test_dry_run_keeps_the_command_and_its_quoting(tmp_path: Path) -> None:
    out = dry(["--mem", "27G"], tmp_path).stdout
    assert out.strip().endswith("-- echo hi\\ there")


def test_cpus_becomes_a_cpu_quota(tmp_path: Path) -> None:
    assert "CPUQuota=400%" in dry(["--mem", "1G", "--cpus", "4"], tmp_path).stdout.split()


def test_no_cpus_means_no_cpu_quota(tmp_path: Path) -> None:
    assert "CPUQuota" not in dry(["--mem", "1G"], tmp_path).stdout


# the memory gate ----------------------------------------------------------------------------------------------


def test_refuses_when_less_memory_is_available_than_the_cap(tmp_path: Path) -> None:
    out = dry(["--mem", "27G"], tmp_path, avail_gib=12)
    assert out.returncode == 75
    assert "echo" not in out.stdout
    assert "12" in out.stderr and "27" in out.stderr


def test_need_replaces_the_cap_in_the_gate(tmp_path: Path) -> None:
    out = dry(["--mem", "27G", "--need", "8G"], tmp_path, avail_gib=12)
    assert out.returncode == 0, out.stderr


def test_gate_passes_at_exactly_the_needed_amount_and_refuses_one_kilobyte_below(tmp_path: Path) -> None:
    ok = run_script(["--mem", "4G", "--dry-run", "--", "true"], tmp_path, avail_gib=4)
    assert ok.returncode == 0, ok.stderr
    env = {**os.environ, "CAPPED_MEMINFO": str(meminfo(tmp_path, 4 * 1024 * 1024 - 1, "low")),
           "CAPPED_CONTROLLERS_FILE": str(controllers(tmp_path, "memory"))}
    low = subprocess.run([BASH, str(SCRIPT), "--mem", "4G", "--dry-run", "--", "true"], capture_output=True,
                         text=True, env=env, cwd=tmp_path, check=False)
    assert low.returncode == 75


def test_refuses_when_available_memory_cannot_be_read(tmp_path: Path) -> None:
    env = {**os.environ, "CAPPED_MEMINFO": str(meminfo(tmp_path, None)),
           "CAPPED_CONTROLLERS_FILE": str(controllers(tmp_path, "memory"))}
    out = subprocess.run([BASH, str(SCRIPT), "--mem", "1G", "--dry-run", "--", "true"], capture_output=True,
                         text=True, env=env, cwd=tmp_path, check=False)
    assert out.returncode == 75
    assert "MemAvailable" in out.stderr


# fail closed --------------------------------------------------------------------------------------------------


def test_refuses_when_the_memory_controller_is_not_delegated(tmp_path: Path) -> None:
    out = dry(["--mem", "1G"], tmp_path, ctl="cpu pids")
    assert out.returncode == 69
    assert "memory controller" in out.stderr


def test_refuses_when_the_controller_list_cannot_be_read(tmp_path: Path) -> None:
    out = dry(["--mem", "1G"], tmp_path, ctl=None)
    assert out.returncode == 69


def test_a_controller_with_memory_in_its_name_is_not_memory(tmp_path: Path) -> None:
    out = dry(["--mem", "1G"], tmp_path, ctl="cpu memory_extra pids")
    assert out.returncode == 69


def test_refuses_when_systemd_run_is_missing(tmp_path: Path) -> None:
    bindir = tmp_path / "nobin"
    bindir.mkdir()
    for name in ("awk", "grep", "id"):
        found = shutil.which(name)
        assert found
        (bindir / name).symlink_to(found)
    out = run_script(["--mem", "1G", "--", "echo", "ran"], tmp_path, path=str(bindir))
    assert out.returncode == 69
    assert "systemd-run" in out.stderr
    assert "ran" not in out.stdout


# usage --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--mem", "1G"],
        ["--", "true"],
        ["--mem", "lots", "--", "true"],
        ["--mem", "0", "--", "true"],
        ["--mem", "1G", "--need", "x", "--", "true"],
        ["--mem", "1G", "--cpus", "0", "--", "true"],
        ["--mem", "1G", "--cpus", "two", "--", "true"],
        ["--mem", "1G", "--bogus", "--", "true"],
        ["--mem"],
    ],
)
def test_bad_arguments_are_a_usage_error_and_run_nothing(tmp_path: Path, args: list[str]) -> None:
    out = run_script(args, tmp_path)
    assert out.returncode == 64
    assert "usage" in out.stderr.lower()


# running through systemd-run ----------------------------------------------------------------------------------


def stubbed(tmp_path: Path, rc: int, args: list[str]) -> subprocess.CompletedProcess[str]:
    return run_script(args, tmp_path, path=str(stub_dir(tmp_path, rc)))


def test_the_command_reaches_systemd_run_with_every_argument_intact(tmp_path: Path) -> None:
    out = stubbed(tmp_path, 0, ["--mem", "1G", "--", "prog", "a b", "--flag=1", ""])
    assert out.returncode == 0, out.stderr
    got = (tmp_path / "stub-args").read_text().split("\n")[:-1]
    assert got[:2] == ["--user", "--scope"]
    assert "MemoryMax=1G" in got
    assert got[got.index("--") :] == ["--", "prog", "a b", "--flag=1", ""]


def test_the_exit_status_of_the_command_is_returned(tmp_path: Path) -> None:
    out = stubbed(tmp_path, 3, ["--mem", "1G", "--", "true"])
    assert out.returncode == 3
    assert "memory cap" not in out.stderr


def test_a_sigkill_exit_says_the_cap_may_have_killed_it(tmp_path: Path) -> None:
    out = stubbed(tmp_path, 137, ["--mem", "1G", "--", "true"])
    assert out.returncode == 137
    assert "memory cap" in out.stderr


def test_the_summary_line_names_the_cap_and_what_was_available(tmp_path: Path) -> None:
    out = stubbed(tmp_path, 0, ["--mem", "2G", "--", "true"])
    assert "MemoryMax=2G" in out.stderr
    assert "40" in out.stderr


# a real scope -------------------------------------------------------------------------------------------------


def real_scope_available() -> bool:
    if sys.platform != "linux" or shutil.which("systemd-run") is None:
        return False
    uid = os.getuid()
    ctl = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/cgroup.controllers")
    try:
        if "memory" not in ctl.read_text().split():
            return False
        probe = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--collect", "true"],
                               capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


REAL = pytest.mark.skipif(
    not real_scope_available(), reason="no systemd user manager with the memory controller delegated to it"
)

HOG = "b=[]\nfor _ in range(64):\n    x=bytearray(16*1024*1024)\n    for i in range(0,len(x),4096): x[i]=1\n    b.append(x)\n"


# The kernel kills a process that reaches its cap, and the desktop (gsd-housekeeping) answers every such kill
# with an "Application Stopped" notification. Running that on every test run teaches nobody anything, so the
# default tests read the limits the kernel enforces and the kill itself runs only on request.
KILL = pytest.mark.skipif(
    os.environ.get("CAPPED_TEST_KILL") != "1",
    reason="set CAPPED_TEST_KILL=1 to kill a process in a real scope (raises a desktop 'Application Stopped' notification)",
)

# Prints memory.max, memory.swap.max and memory.oom.group of the scope this process runs in, one per line.
LIMITS = (
    "import pathlib\n"
    "rel = [l for l in pathlib.Path('/proc/self/cgroup').read_text().splitlines() if l.startswith('0::')][0][3:]\n"
    "d = pathlib.Path('/sys/fs/cgroup') / rel.lstrip('/')\n"
    "for n in ('memory.max', 'memory.swap.max', 'memory.oom.group'):\n"
    "    print(n, (d / n).read_text().strip())\n"
    "print('scope', d.name)\n"
)


@REAL
def test_the_scope_carries_the_cap_the_swap_limit_and_the_group_kill(tmp_path: Path) -> None:
    out = subprocess.run([BASH, str(SCRIPT), "--mem", "256M", "--", sys.executable, "-I", "-c", LIMITS],
                         capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    got = dict(line.split(" ", 1) for line in out.stdout.strip().splitlines())
    assert got["scope"].startswith("capped-") and got["scope"].endswith(".scope"), got
    assert got["memory.max"] == str(256 * 1024**2), got
    assert got["memory.swap.max"] == "0", got
    assert got["memory.oom.group"] == "1", got


@REAL
def test_a_cap_given_in_gigabytes_reaches_the_kernel(tmp_path: Path) -> None:
    out = subprocess.run([BASH, str(SCRIPT), "--mem", "3G", "--need", "64M", "--", sys.executable, "-I", "-c", LIMITS],
                         capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert dict(line.split(" ", 1) for line in out.stdout.strip().splitlines())["memory.max"] == str(3 * 1024**3)


@REAL
@KILL
def test_a_process_over_the_cap_is_killed_and_the_caller_survives(tmp_path: Path) -> None:
    out = subprocess.run([BASH, str(SCRIPT), "--mem", "256M", "--", sys.executable, "-I", "-c", HOG],
                         capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 137, out.stderr
    assert "memory cap" in out.stderr
    assert os.getpid() > 0  # this test process is the caller and is still running


@REAL
def test_a_process_under_the_cap_runs_to_the_end(tmp_path: Path) -> None:
    code = "b=bytearray(32*1024*1024)\nfor i in range(0,len(b),4096): b[i]=1\nprint('done')\n"
    out = subprocess.run([BASH, str(SCRIPT), "--mem", "256M", "--", sys.executable, "-I", "-c", code],
                         capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "done"
