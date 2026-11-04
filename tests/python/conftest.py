"""Shared fakes: nothing here touches the machine, the network or a GPU."""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path

import pytest

from launcher.probes import Probes
from launcher.verified import VerifiedStore


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
