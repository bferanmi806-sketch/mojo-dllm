"""Tests that run git must not act on the repository whose hook is running them.

git exports GIT_DIR, GIT_WORK_TREE and GIT_INDEX_FILE to its hooks, and the pre-push hook runs this suite.
A test that ran `git init` or `git commit` in a temporary directory then acted on the real repository:
it added commits to the branch being pushed and set core.bare in the shared config. The decoy below plays
the part of that repository.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Every test file that creates a repository or commits in one.
GIT_USING_TESTS = [
    "tests/python/test_dg_ref_build.py",
    "tests/python/test_gemma4_golden.py",
    "tests/python/test_parity_dgemma.py::test_the_checkout_must_be_the_pinned_commit",
]


def plain_git(*args: str) -> str:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, env=clean).stdout.strip()


def test_a_hook_environment_does_not_make_tests_touch_the_hooked_repository(tmp_path: Path) -> None:
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    plain_git("init", "-q", str(decoy))
    hook_env = {
        **os.environ,
        "GIT_DIR": str(decoy / ".git"),
        "GIT_WORK_TREE": str(decoy),
        "GIT_INDEX_FILE": str(decoy / ".git" / "index"),
    }
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *GIT_USING_TESTS],
        cwd=ROOT,
        env=hook_env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout[-2000:]
    assert plain_git("-C", str(decoy), "rev-list", "--all", "--count") == "0", "a test committed to the hooked repository"
    assert plain_git("-C", str(decoy), "config", "--get", "core.bare") == "false", "a test rewrote the repository's config"
