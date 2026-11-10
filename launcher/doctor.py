"""`./mojo-dllm doctor`: read-only, offline checks of this machine."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

from launcher import build
from launcher.errors import LauncherError
from launcher.gpu import MIN_COMPUTE_CAP, parse_nvidia_smi
from launcher.probes import Probes
from launcher.ui import Style

UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
MIN_GLIBC = (2, 34)  # the Mojo wheels need it
MIN_MEM_GIB = 6


@dataclass(frozen=True)
class Check:
    status: str  # "ok", "warn" or "fail"
    label: str
    detail: str
    fix: str | None = None


def _ok(label: str, detail: str) -> Check:
    return Check("ok", label, detail)


def check_platform(p: Probes) -> Check:
    if (p.system, p.machine) == ("Linux", "x86_64"):
        return _ok("platform", "Linux x86_64")
    return Check(
        "fail", "platform", f"{p.system} {p.machine}; mojo-dllm needs Linux x86_64",
        "run it on a Linux x86_64 machine or in a container",
    )


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", text)[:2])


def check_glibc(p: Probes) -> Check:
    need = ".".join(map(str, MIN_GLIBC))
    if not p.glibc:
        return Check("fail", "glibc", f"not found; the Mojo wheels need glibc {need} or newer",
                     "use a glibc distribution such as Ubuntu 22.04 or newer")
    if _version_tuple(p.glibc) < MIN_GLIBC:
        return Check("fail", "glibc", f"{p.glibc}; the Mojo wheels need {need} or newer",
                     "upgrade the distribution (Ubuntu 22.04 or newer ships glibc 2.35)")
    return _ok("glibc", p.glibc)


def check_cpu(p: Probes) -> Check:
    for line in p.cpuinfo.splitlines():
        if line.startswith("flags") and "avx2" in line.split():
            return _ok("cpu", "AVX2")
    return Check("fail", "cpu", "no AVX2 in /proc/cpuinfo; the CPU kernels need it",
                 "run it on a CPU with AVX2 (Intel Haswell or AMD Excavator and newer)")


def check_memory(p: Probes) -> Check:
    match = re.search(r"^MemAvailable:\s+(\d+) kB", p.meminfo, re.M)
    if match is None:
        return Check("warn", "memory", "MemAvailable not readable from /proc/meminfo",
                     f"make sure about {MIN_MEM_GIB} GiB are free before you run a model")
    gib = int(match.group(1)) / 1024 / 1024
    if gib < MIN_MEM_GIB:
        return Check("warn", "memory", f"{gib:.1f} GiB available; a Q4_K_M 8B model needs about {MIN_MEM_GIB}",
                     "close other programs before you run a model")
    return _ok("memory", f"{gib:.1f} GiB available")


def check_toolchain(p: Probes) -> Check:
    if p.uv_path is None:
        return Check("fail", "toolchain", "uv not found", UV_INSTALL)
    if p.mojo_version_output is None:
        return Check("fail", "toolchain", f"mojo {p.mojo_pin} is not installed in .venv", "./mojo-dllm setup")
    match = re.search(r"\d+\.\d+\.\d+", p.mojo_version_output)
    found = match.group(0) if match else p.mojo_version_output.strip()
    if found != p.mojo_pin:
        return Check("fail", "toolchain", f"mojo {found} found, pyproject.toml pins {p.mojo_pin}", "uv sync --frozen")
    return _ok("toolchain", f"mojo {found} (uv)")


def check_binary(p: Probes) -> Check:
    if p.binary_mtime is None:
        return Check("fail", "binary", "build/mojo-dllm does not exist", "./mojo-dllm setup")
    if not p.source_mtimes:
        return Check("fail", "binary", "no sources found under src/, so the binary cannot be compared with them",
                     "restore src/ from git")
    if build.is_stale(p.binary_mtime, p.source_mtimes):
        return Check("warn", "binary", "build/mojo-dllm is older than its sources (src/, pyproject.toml, uv.lock, scripts/build.sh)",
                     "./mojo-dllm setup")
    if p.stamp_problem is not None:
        return Check("warn", "binary", p.stamp_problem, "./mojo-dllm setup")
    return _ok("binary", "build/mojo-dllm is up to date")


def check_gpu(p: Probes) -> Check:
    cpu_note = "device: cpu will be used"
    if p.gpu_text is None:
        return Check("warn", "gpu", "no NVIDIA GPU found (nvidia-smi missing or failed)", cpu_note)
    try:
        gpus = parse_nvidia_smi(p.gpu_text)
    except LauncherError as e:
        return Check("warn", "gpu", f"nvidia-smi output is unreadable: {str(e).splitlines()[0]}",
                     "run nvidia-smi and check the driver; " + cpu_note)
    if not gpus:
        return Check("warn", "gpu", "nvidia-smi lists no device", cpu_note)
    driver = (p.driver_text or "").strip().splitlines()
    driver_part = f", driver {driver[0]}" if driver else ""
    for g in gpus:
        if g.compute_cap >= MIN_COMPUTE_CAP:
            return _ok("gpu", f"{g.name}, compute {g.compute_cap:.1f}{driver_part}, {g.free_mib} MiB free")
    g = gpus[0]
    return Check("warn", "gpu", f"{g.name}, compute {g.compute_cap:.1f} is below {MIN_COMPUTE_CAP:.1f}{driver_part}",
                 "device: cpu will be used (the GPU kernels need compute 8.0 or newer)")


def check_model(p: Probes) -> Check:
    if p.cached_model is not None and not p.verified:
        return Check("warn", "model", f"{p.cached_model} (not verified against its sha256)",
                     "./mojo-dllm pull")
    if p.cached_model is not None:
        return _ok("model", str(p.cached_model))
    return Check("warn", "model", "llada-8b (Q4_K_M) is not in the Hugging Face cache", "./mojo-dllm pull")


CHECKS: tuple[Callable[[Probes], Check], ...] = (
    check_platform, check_glibc, check_cpu, check_memory,
    check_toolchain, check_binary, check_gpu, check_model,
)
# What must hold before setup installs anything.
PREFLIGHT: tuple[Callable[[Probes], Check], ...] = (check_platform, check_glibc, check_cpu)

_MARK = {"ok": ("✓", "green"), "warn": ("!", "yellow"), "fail": ("✗", "red")}


def preflight(p: Probes) -> list[Check]:
    return [check(p) for check in PREFLIGHT]


def render(check: Check, style: Style, out: TextIO) -> None:
    mark, colour = _MARK[check.status]
    print(f"{style.paint(colour, mark)} {check.label:<10} {check.detail}", file=out)
    if check.fix is not None:
        print(f"  → {check.fix}", file=out)


def run_doctor(p: Probes, out: TextIO, style: Style) -> int:
    results = [check(p) for check in CHECKS]
    for r in results:
        render(r, style, out)
    return 1 if any(r.status == "fail" for r in results) else 0
