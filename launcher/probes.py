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
    cpuinfo: str
    meminfo: str
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


def read_project_version(pyproject: str) -> str:
    try:
        project = tomllib.loads(pyproject).get("project")
        version = project.get("version") if isinstance(project, dict) else None
    except (tomllib.TOMLDecodeError, TypeError):
        version = None
    if not isinstance(version, str):
        raise LauncherError("pyproject.toml has no [project] version; restore it from git, then run ./mojo-dllm --version again.")
    return version


def _capture(cmd: list[str]) -> str | None:
    try:
        done = subprocess.run(cmd, cwd=build.ROOT, env=build.project_env(), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def cached_default_model(hub: Hub) -> Path | None:
    try:
        return plan_model(parse_model_ref(DEFAULT_MODEL), hub).cached
    except LauncherError:
        return None


def _source_mtimes() -> tuple[float, ...]:
    try:
        return tuple(build.source_mtimes())
    except LauncherError:
        return ()  # check_binary reports the empty tree


def real_probes() -> Probes:
    libc, version = platform.libc_ver()
    uv = shutil.which("uv")
    smi = shutil.which("nvidia-smi") is not None
    cached = cached_default_model(Hub(offline=True))
    return Probes(
        system=platform.system(),
        machine=platform.machine(),
        glibc=version if libc == "glibc" else "",
        cpuinfo=_read("/proc/cpuinfo"),
        meminfo=_read("/proc/meminfo"),
        uv_path=uv,
        # --no-sync: doctor must not install the toolchain it is checking.
        mojo_version_output=(
            _capture(["uv", "run", "--frozen", "--no-sync", "mojo", "--version"]) if uv else None
        ),
        mojo_pin=read_mojo_pin((build.ROOT / "pyproject.toml").read_text()),
        binary_mtime=build.binary_mtime(),
        source_mtimes=_source_mtimes(),
        stamp_problem=build.stamp_problem() if build.binary_mtime() is not None else None,
        gpu_text=_capture(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"])
        if smi
        else None,
        driver_text=_capture(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        if smi
        else None,
        cached_model=cached,
        verified=cached is not None and default_store().is_verified(cached),
    )
