"""The compiled binary: where it lives, when it is stale, how to build it."""

from __future__ import annotations

import collections
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from launcher.errors import LauncherError

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "build" / "mojo-dllm"


def project_env() -> dict[str, str]:
    """The environment for uv calls into the project's .venv.

    `uv run --script` exports VIRTUAL_ENV for its own cache environment, and uv
    warns when that differs from the project's .venv.
    """
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


# Files whose change can change the binary besides the Mojo sources: the compiler pin, the lock that
# resolves it and the build script.
EXTRA_INPUTS = ("pyproject.toml", "uv.lock", "scripts/build.sh")


def read_mojo_pin(pyproject: str) -> str:
    match = re.search(r'"mojo==([0-9][^"]*)"', pyproject)
    if match is None:
        raise LauncherError("pyproject.toml has no mojo==X pin; restore it from git, then run ./mojo-dllm doctor again.")
    return match.group(1)


def is_stale(binary_mtime: float | None, source_mtimes: Iterable[float]) -> bool:
    mtimes = list(source_mtimes)
    if not mtimes:
        # any([]) is False: a tree with nothing to compare would report every binary as fresh.
        raise LauncherError("no sources found under src/, so the binary cannot be compared with them; restore src/ from git.")
    if binary_mtime is None:
        return True
    return any(m > binary_mtime for m in mtimes)


def exec_binary(argv: list[str]) -> None:
    """Replace this process with the binary; a missing or unrunnable one is a launcher error."""
    try:
        os.execv(argv[0], argv)
    except OSError as e:
        raise LauncherError(f"cannot start {argv[0]}: {e.strerror}. Run ./mojo-dllm doctor to check the build.") from e


def source_mtimes() -> list[float]:
    """The mtimes of every input of the binary: src/**/*.mojo and the files in EXTRA_INPUTS."""
    mojo = [p.stat().st_mtime for p in (ROOT / "src").rglob("*.mojo")]
    if not mojo:
        raise LauncherError("no sources found under src/, so the binary cannot be compared with them; restore src/ from git.")
    extra = [(ROOT / rel).stat().st_mtime for rel in EXTRA_INPUTS if (ROOT / rel).is_file()]
    return mojo + extra


def _gpu_compute_caps() -> str:
    """What scripts/build.sh asks nvidia-smi for; empty when it is missing or fails."""
    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout if done.returncode == 0 else ""


def target_arch() -> str:
    """The GPU architecture scripts/build.sh compiles for: the override, the first GPU, else sm_80."""
    override = os.environ.get("MOJO_DLLM_GPU_ARCH", "")
    if override:
        return override
    first = _gpu_compute_caps().strip().splitlines()[:1]
    if first and re.fullmatch(r"\d+\.\d+", first[0].strip()):
        return "sm_" + first[0].strip().replace(".", "")
    return "sm_80"


def stamp_path() -> Path:
    return BINARY.with_name(BINARY.name + ".stamp")


def current_stamp() -> str:
    """What a binary built now would record: the pinned Mojo version and the GPU architecture."""
    pin = read_mojo_pin((ROOT / "pyproject.toml").read_text())
    return f"mojo {pin}\narch {target_arch()}\n"


def write_stamp(stamp: str | None = None) -> None:
    path = stamp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stamp if stamp is not None else current_stamp())


def stamp_problem() -> str | None:
    """Why the binary's stamp does not match this machine and tree, or None when it does."""
    try:
        recorded = stamp_path().read_text()
    except OSError:
        return "build/mojo-dllm has no stamp, so its Mojo version and GPU architecture are unknown"
    wanted = current_stamp()
    if recorded == wanted:
        return None
    have = dict(line.split(" ", 1) for line in recorded.splitlines() if " " in line)
    want = dict(line.split(" ", 1) for line in wanted.splitlines())
    changed = [f"{k} {have.get(k, '?')} to {v}" for k, v in want.items() if have.get(k) != v]
    return "build/mojo-dllm was built for a different setup (" + ", ".join(changed) + ")"


def binary_mtime() -> float | None:
    return BINARY.stat().st_mtime if BINARY.is_file() else None


def needs_build() -> bool:
    return is_stale(binary_mtime(), source_mtimes()) or stamp_problem() is not None


def ensure_built() -> bool:
    """Build when the binary is missing or stale. Returns whether it built."""
    if not needs_build():
        return False
    print("Building mojo-dllm (about a minute)...", file=sys.stderr, flush=True)
    stamp = current_stamp()  # taken before the build: it names what the build was asked for
    tail: collections.deque[str] = collections.deque(maxlen=20)
    proc = subprocess.Popen(
        ["bash", "scripts/build.sh"],
        cwd=ROOT,
        env=project_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stderr.write(line)
        tail.append(line.rstrip("\n"))
    if proc.wait() != 0:
        raise LauncherError(
            "the build failed (bash scripts/build.sh); last 20 lines:\n"
            + "\n".join(tail)
            + "\nRun ./mojo-dllm doctor to check the toolchain."
        )
    write_stamp(stamp)
    return True
