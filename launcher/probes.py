"""Everything doctor and setup read from the machine, gathered in one object.

Checks take a `Probes` value instead of touching the machine, so tests can
hand them any machine they like.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from launcher import build
from launcher.build import read_mojo_pin
from launcher.errors import LauncherError
from launcher.gpu import QUERY
from launcher.hub import Hub
from launcher.models import DEFAULT_MODEL, parse_model_ref, plan_model
from launcher.verified import default_store


@dataclass(frozen=True)
class Probes:
    system: str
    machine: str
    glibc: str  # "" when the C library is not glibc
    cpuinfo: str | None  # None when /proc/cpuinfo could not be read (cpuinfo_error says why)
    meminfo: str | None
    uv_path: str | None
    mojo_version_output: str | None  # None when `mojo --version` could not run
    mojo_pin: str
    binary_mtime: float | None
    source_mtimes: tuple[float, ...]
    gpu_text: str | None  # None when nvidia-smi is missing or failed
    driver_text: str | None
    cached_model: Path | None
    verified: bool = True  # whether cached_model has passed its sha256 check
    stamp_problem: str | None = None  # why the binary's build stamp is out of date
    # Why a probe came back empty, so doctor can say that and not guess.
    mojo_failure: str | None = None
    gpu_failure: str | None = None
    cpuinfo_error: str | None = None
    meminfo_error: str | None = None
    model_error: str | None = None


def read_project_version(pyproject: str) -> str:
    try:
        project = tomllib.loads(pyproject).get("project")
        version = project.get("version") if isinstance(project, dict) else None
    except (tomllib.TOMLDecodeError, TypeError):
        version = None
    if not isinstance(version, str):
        raise LauncherError("pyproject.toml has no [project] version; restore it from git, then run ./mojo-dllm --version again.")
    return version


def _capture(cmd: list[str]) -> tuple[str | None, str | None]:
    """Run a probe command. Returns (stdout, None) on success and (None, why) on any failure."""
    try:
        done = subprocess.run(cmd, cwd=build.ROOT, env=build.project_env(), capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as e:
        return None, f"timed out after {e.timeout:g} s"
    except OSError as e:
        return None, f"{cmd[0]}: {e.strerror or e}"
    if done.returncode != 0:
        detail = done.stderr.strip().splitlines()
        return None, f"exit {done.returncode}: {detail[0]}" if detail else f"exit {done.returncode}"
    return done.stdout, None


def _read(path: str) -> tuple[str | None, str | None]:
    """The text of a file, or (None, why) when it cannot be read."""
    try:
        return Path(path).read_text(), None
    except OSError as e:
        return None, e.strerror or str(e)


def cached_default_model(hub: Hub) -> tuple[Path | None, str | None]:
    """The cached default model, and why the cache could not be read (None when it could)."""
    try:
        return plan_model(parse_model_ref(DEFAULT_MODEL), hub).cached, None
    except LauncherError as e:
        return None, str(e).splitlines()[0]
    except OSError as e:
        return None, e.strerror or str(e)


def _split_driver(text: str) -> tuple[str, str]:
    """Split the driver_version column, the last one, off nvidia-smi's rows.

    A row without a comma stays whole, so the parser reports it as unreadable.
    """
    rows: list[str] = []
    drivers: list[str] = []
    for line in text.splitlines():
        head, comma, driver = line.rpartition(",")
        if not comma or not head.strip():
            rows.append(line)
            continue
        rows.append(head)
        drivers.append(driver.strip())
    return "".join(r + "\n" for r in rows), "".join(d + "\n" for d in drivers[:1])


def _source_mtimes() -> tuple[float, ...]:
    try:
        return tuple(build.source_mtimes())
    except LauncherError:
        return ()  # check_binary reports the empty tree


def real_probes() -> Probes:
    libc, version = platform.libc_ver()
    uv = shutil.which("uv")
    smi = shutil.which("nvidia-smi") is not None
    cached, model_error = cached_default_model(Hub(offline=True))
    # --no-sync: doctor must not install the toolchain it is checking.
    mojo, mojo_failure = (
        _capture(["uv", "run", "--frozen", "--no-sync", "mojo", "--version"]) if uv else (None, "uv not found on PATH")
    )
    # One nvidia-smi call: the driver version is a column of the same query.
    smi_text, gpu_failure = (
        _capture(["nvidia-smi", f"--query-gpu={QUERY},driver_version", "--format=csv,noheader,nounits"])
        if smi
        else (None, "nvidia-smi not found on PATH")
    )
    gpu_text, driver_text = _split_driver(smi_text) if smi_text is not None else (None, None)
    cpuinfo, cpuinfo_error = _read("/proc/cpuinfo")
    meminfo, meminfo_error = _read("/proc/meminfo")
    return Probes(
        system=platform.system(),
        machine=platform.machine(),
        glibc=version if libc == "glibc" else "",
        cpuinfo=cpuinfo,
        meminfo=meminfo,
        uv_path=uv,
        mojo_version_output=mojo,
        mojo_pin=read_mojo_pin((build.ROOT / "pyproject.toml").read_text()),
        binary_mtime=build.binary_mtime(),
        source_mtimes=_source_mtimes(),
        stamp_problem=build.stamp_problem() if build.binary_mtime() is not None else None,
        gpu_text=gpu_text,
        driver_text=driver_text,
        cached_model=cached,
        verified=cached is not None and default_store().is_verified(cached),
        mojo_failure=mojo_failure,
        gpu_failure=gpu_failure,
        cpuinfo_error=cpuinfo_error,
        meminfo_error=meminfo_error,
        model_error=model_error,
    )
