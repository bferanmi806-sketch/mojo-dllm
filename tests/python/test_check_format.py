"""scripts/check_format.sh sees untracked files and ignores nothing else (#248).

The formatter is a stub that rewrites the word UGLY to pretty in every file under src, tests and bench, so no
Mojo compiler runs. The repository is a throwaway `git init` in tmp_path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_format.sh"
FORMATTER = """#!/usr/bin/env bash
# fake `mojo format -q src tests bench`
[ "$1" = format ] || exit 3
[ -n "${FAKE_FORMAT_FAIL:-}" ] && exit 4
find src tests bench -type f -readable -print0 2>/dev/null | xargs -0 -r sed -i 's/UGLY/pretty/g'
"""


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts/check_format.sh")
    for d in ("src", "tests", "bench"):
        (tmp_path / d).mkdir()
    (tmp_path / "src/a.mojo").write_text("fn a():\n    pass\n")
    (tmp_path / "tests/test_a.mojo").write_text("fn t():\n    pass\n")
    fmt = tmp_path / "fake-mojo"
    fmt.write_text(FORMATTER)
    fmt.chmod(0o755)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def check(repo: Path, **env: str) -> subprocess.CompletedProcess[str]:
    import os
    return subprocess.run(["bash", str(repo / "scripts/check_format.sh")], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "MOJO": str(repo / "fake-mojo"), **env})


def test_a_formatted_tree_passes(repo: Path) -> None:
    assert check(repo).returncode == 0


def test_an_untracked_unformatted_file_fails_and_is_named(repo: Path) -> None:
    (repo / "src/new.mojo").write_text("fn UGLY():\n    pass\n")
    p = check(repo)
    assert p.returncode == 1
    assert "mojo format changed files" in p.stderr and "src/new.mojo" in p.stderr


def test_an_untracked_file_with_a_space_and_a_non_ascii_name_is_checked(repo: Path) -> None:
    (repo / "bench/café two.mojo").write_text("UGLY\n")
    p = check(repo)
    assert p.returncode == 1 and "café two.mojo" in p.stderr


def test_a_committed_unformatted_file_fails(repo: Path) -> None:
    (repo / "src/a.mojo").write_text("fn UGLY():\n    pass\n")
    git(repo, "commit", "-q", "-am", "ugly")
    assert check(repo).returncode == 1


def test_an_uncommitted_edit_that_is_already_formatted_passes(repo: Path) -> None:
    (repo / "src/a.mojo").write_text("fn a():\n    return\n")
    (repo / "src/fresh.mojo").write_text("fn fresh():\n    pass\n")
    assert check(repo).returncode == 0


def test_an_ignored_file_is_not_part_of_the_check(repo: Path) -> None:
    (repo / ".gitignore").write_text("bench/scratch.mojo\n")
    git(repo, "add", ".gitignore")
    git(repo, "commit", "-q", "-m", "ignore")
    (repo / "bench/scratch.mojo").write_text("UGLY\n")
    assert check(repo).returncode == 0


def test_a_tracked_file_deleted_from_the_tree_does_not_crash_the_check(repo: Path) -> None:
    (repo / "tests/test_a.mojo").unlink()
    assert check(repo).returncode == 0


def test_a_formatter_that_fails_fails_the_check(repo: Path) -> None:
    p = check(repo, FAKE_FORMAT_FAIL="1")
    assert p.returncode == 4 or p.returncode == 1


def test_a_missing_formatter_is_a_failure_and_not_a_pass(repo: Path) -> None:
    p = check(repo, MOJO=str(repo / "no-such-mojo"))
    assert p.returncode == 1 and "missing tool" in p.stderr


def test_an_unreadable_file_stops_the_check_and_does_not_read_as_unchanged(repo: Path) -> None:
    import os
    if os.geteuid() == 0:
        pytest.skip("root reads every file")
    (repo / "src/locked.mojo").write_text("x\n")
    (repo / "src/locked.mojo").chmod(0)
    try:
        p = check(repo)
    finally:
        (repo / "src/locked.mojo").chmod(0o644)
    assert p.returncode != 0
