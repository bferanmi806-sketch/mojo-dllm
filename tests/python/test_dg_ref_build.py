"""scripts/build_dg_ref.sh pins a third-party commit and builds it offline.

Nothing here touches the network, the real llama.cpp or a model: the "remote"
is a throwaway local repository and `cmake` is a recording stub.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_dg_ref.sh"


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str]:
    """A local repository with one commit; returns it and the commit's full SHA."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "CMakeLists.txt").write_text("project(x)\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "one")
    return repo, git(repo, "rev-parse", "HEAD")


@pytest.fixture
def stub_cmake(tmp_path: Path) -> tuple[Path, Path]:
    """A `cmake` that records its arguments; returns the bin dir and the log."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "cmake.log"
    stub = bindir / "cmake"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$STUB_LOG"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return bindir, log


def run(
    tmp_path: Path, extra_env: dict[str, str], *args: str, bindir: Path | None = None, log: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": f"{bindir}:{os.environ['PATH']}" if bindir else os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "DG_LLAMA_SRC": str(tmp_path / "src"),
        "STUB_LOG": str(log or tmp_path / "unused.log"),
        **extra_env,
    }
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, cwd=ROOT)


def test_the_pin_is_a_full_commit_sha() -> None:
    text = SCRIPT.read_text()
    pins = re.findall(r'^PINNED_COMMIT="([0-9a-f]+)"$', text, flags=re.M)
    assert len(pins) == 1 and len(pins[0]) == 40, "a short SHA can resolve to a different commit later"


def test_an_unreachable_commit_is_refused_and_named(tmp_path: Path, upstream: tuple[Path, str]) -> None:
    repo, _ = upstream
    # The upstream has no commit with the pinned SHA, so the fetch must fail.
    done = run(tmp_path, {"DG_LLAMA_REPO": str(repo)}, "--llama-only")
    pin = re.search(r'^PINNED_COMMIT="([0-9a-f]{40})"$', SCRIPT.read_text(), flags=re.M)
    assert pin is not None
    assert done.returncode != 0
    assert "could not fetch" in done.stderr and pin.group(1) in done.stderr


def test_builds_cpu_only_into_its_own_directory(
    tmp_path: Path, upstream: tuple[Path, str], stub_cmake: tuple[Path, Path]
) -> None:
    repo, sha = upstream
    bindir, log = stub_cmake
    done = run(
        tmp_path,
        {"DG_LLAMA_REPO": str(repo), "DG_LLAMA_COMMIT": sha},
        "--llama-only",
        bindir=bindir,
        log=log,
    )
    assert done.returncode == 0, done.stderr
    src = tmp_path / "src"
    assert git(src, "rev-parse", "HEAD") == sha
    calls = log.read_text().splitlines()
    configure, build = calls[0], calls[1]
    assert f"-B {src}/build" in configure
    for flag in ("-DGGML_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF"):
        assert flag in configure
    for target in ("llama-tokenize", "llama-diffusion-gemma-eval", "llama"):
        assert target in build.split()
    assert sha in done.stdout


def test_a_modified_checkout_is_refused(tmp_path: Path, upstream: tuple[Path, str], stub_cmake: tuple[Path, Path]) -> None:
    repo, sha = upstream
    bindir, log = stub_cmake
    env = {"DG_LLAMA_REPO": str(repo), "DG_LLAMA_COMMIT": sha}
    assert run(tmp_path, env, "--llama-only", bindir=bindir, log=log).returncode == 0
    (tmp_path / "src" / "CMakeLists.txt").write_text("project(edited)\n")
    log.unlink()
    done = run(tmp_path, env, "--llama-only", bindir=bindir, log=log)
    assert done.returncode != 0
    assert "modified" in done.stderr
    assert not log.exists(), "cmake must not run on a tree that differs from the pin"


def test_an_abbreviated_commit_is_not_the_pin(tmp_path: Path, upstream: tuple[Path, str], stub_cmake: tuple[Path, Path]) -> None:
    repo, sha = upstream
    bindir, log = stub_cmake
    assert run(tmp_path, {"DG_LLAMA_REPO": str(repo), "DG_LLAMA_COMMIT": sha}, "--llama-only", bindir=bindir, log=log).returncode == 0
    log.unlink()
    # The checkout already holds the commit, so git resolves the short form; HEAD still must equal what was asked for.
    done = run(tmp_path, {"DG_LLAMA_REPO": str(repo), "DG_LLAMA_COMMIT": sha[:12]}, "--llama-only", bindir=bindir, log=log)
    assert done.returncode != 0 and "expected" in done.stderr
    assert not log.exists()


def test_git_hook_variables_do_not_reach_the_checkout(
    tmp_path: Path, upstream: tuple[Path, str], stub_cmake: tuple[Path, Path]
) -> None:
    # git exports GIT_DIR, GIT_WORK_TREE and GIT_INDEX_FILE to its hooks. With them set, `git init` and
    # `git -C` act on that repository instead of the one named, so the build would fetch into and check out
    # the repository being pushed. The decoy stands in for it.
    repo, sha = upstream
    bindir, log = stub_cmake
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    subprocess.run(["git", "init", "-q", str(decoy)], check=True)
    leaked = {
        "GIT_DIR": str(decoy / ".git"),
        "GIT_WORK_TREE": str(decoy),
        "GIT_INDEX_FILE": str(decoy / ".git" / "index"),
    }
    done = run(
        tmp_path, {"DG_LLAMA_REPO": str(repo), "DG_LLAMA_COMMIT": sha, **leaked}, "--llama-only", bindir=bindir, log=log
    )
    assert done.returncode == 0, done.stderr
    assert git(tmp_path / "src", "rev-parse", "HEAD") == sha
    assert git(decoy, "rev-list", "--all", "--count") == "0"


@pytest.mark.parametrize("bad", ["--upload-pack=x", "main", "2e715c0639", "G" * 40, "A" * 40])
def test_a_commit_that_is_not_40_lowercase_hex_is_refused_before_git_runs(tmp_path: Path, bad: str) -> None:
    done = run(tmp_path, {"DG_LLAMA_COMMIT": bad}, "--llama-only")
    assert done.returncode != 0 and "40 hex" in done.stderr
    assert not (tmp_path / "src").exists(), "nothing may be created before the commit is validated"
