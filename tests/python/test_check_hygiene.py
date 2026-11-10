"""scripts/check_hygiene.sh refuses each credential shape and the employer name, and reads what it says it reads (#251).

Every secret is assembled at run time from a prefix and a body, so this file holds none of them and passes the
very check it tests. The checks run in a throwaway `git init` repository (GIT_* is scrubbed by conftest.py).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_hygiene.sh"
EMPLOYER = "entr" + "opia"

# (label, prefix, body alphabet, minimum body length)
SHAPES = [
    ("OpenAI key", "sk-", "A", 40),
    ("OpenAI project key", "sk-proj-", "A", 40),
    ("Anthropic key", "sk-ant-", "A", 40),
    ("GitHub PAT", "ghp_", "A", 36),
    ("GitHub fine-grained PAT", "github_pat_", "A", 60),
    ("Hugging Face token", "hf_", "A", 34),
    ("Groq key", "gsk_", "A", 40),
    ("AWS access key", "AKIA", "A", 16),
    ("Google API key", "AIza", "A", 35),
    ("Slack token", "xoxb-", "A", 20),
]


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "work"          # conftest's autouse fake machine writes into tmp_path itself
    root.mkdir()
    git(root, "init", "-q")
    (root / "README.md").write_text("nothing to see\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def check(repo: Path, mode: str = "--tracked") -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), mode], cwd=repo, capture_output=True, text=True, timeout=60,
                          env={**os.environ})


def commit_file(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    git(repo, "add", "--", name)
    git(repo, "commit", "-q", "-m", f"add {name}")


def test_a_clean_tree_passes_and_counts_its_files(repo: Path) -> None:
    p = check(repo)
    assert p.returncode == 0 and "1 file(s) clean (0 binary skipped)" in p.stdout


@pytest.mark.parametrize("label,prefix,fill,minimum", SHAPES, ids=[s[0] for s in SHAPES])
def test_each_provider_pattern_fires_at_its_minimum_length_and_not_one_below(repo: Path, label: str, prefix: str, fill: str,
                                                                          minimum: int) -> None:
    commit_file(repo, "short.txt", f"token = {prefix}{fill * (minimum - 1)}\n")
    p = check(repo)
    assert p.returncode == 0, f"{label}: a body one character short must not match: {p.stderr}"
    commit_file(repo, "long.txt", f"token = {prefix}{fill * minimum}\n")
    p = check(repo)
    assert p.returncode == 1 and f"SECRET: {label} in long.txt" in p.stderr
    assert "short.txt" not in p.stderr and "refusing" in p.stderr


def test_the_message_never_prints_the_secret_beyond_its_prefix(repo: Path) -> None:
    body = "Zq" * 20
    commit_file(repo, "k.txt", f"gh = ghp_{body}\n")
    p = check(repo)
    assert p.returncode == 1 and body not in p.stderr and "starts ghp_" in p.stderr and "44 chars" in p.stderr


def test_the_employer_name_fails_in_text_in_any_case_and_in_a_file_name(repo: Path) -> None:
    commit_file(repo, "a.txt", f"written at {EMPLOYER.upper()} last year\n")
    p = check(repo)
    assert p.returncode == 1 and "EMPLOYER: a.txt" in p.stderr
    git(repo, "rm", "-q", "a.txt")
    git(repo, "commit", "-q", "-m", "rm")
    commit_file(repo, f"{EMPLOYER}-notes.txt", "fine\n")
    p = check(repo)
    assert p.returncode == 1 and f"EMPLOYER: path {EMPLOYER}-notes.txt" in p.stderr


def test_a_binary_file_is_skipped_and_counted(repo: Path) -> None:
    (repo / "blob.bin").write_bytes(b"\x00\x01ghp_" + b"A" * 40 + b"\x00")
    git(repo, "add", "blob.bin")
    git(repo, "commit", "-q", "-m", "bin")
    p = check(repo)
    assert p.returncode == 0 and "2 file(s) clean (1 binary skipped)" in p.stdout


def test_an_untracked_file_is_not_scanned_in_tracked_mode(repo: Path) -> None:
    (repo / "loose.txt").write_text("ghp_" + "A" * 40 + "\n")
    assert check(repo).returncode == 0


def test_staged_mode_reads_the_index_not_the_working_tree(repo: Path) -> None:
    (repo / "k.txt").write_text("ghp_" + "A" * 40 + "\n")
    git(repo, "add", "k.txt")
    (repo / "k.txt").write_text("clean now\n")               # the working copy is clean, the index is not
    p = check(repo, "--staged")
    assert p.returncode == 1 and "SECRET: GitHub PAT in k.txt" in p.stderr


def test_staged_mode_ignores_a_secret_that_is_only_in_the_working_tree(repo: Path) -> None:
    (repo / "k.txt").write_text("clean\n")
    git(repo, "add", "k.txt")
    (repo / "k.txt").write_text("ghp_" + "A" * 40 + "\n")
    assert check(repo, "--staged").returncode == 0


def test_staged_mode_with_nothing_staged_passes_with_zero_files(repo: Path) -> None:
    p = check(repo, "--staged")
    assert p.returncode == 0 and "0 file(s) clean" in p.stdout


@pytest.mark.parametrize("args", [[], ["--all"], ["tracked"]])
def test_a_missing_or_unknown_mode_is_a_usage_error(repo: Path, args: list[str]) -> None:
    p = subprocess.run(["bash", str(SCRIPT), *args], cwd=repo, capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and "usage" in p.stderr


def test_outside_a_git_repository_the_gate_fails_and_does_not_pass(tmp_path: Path) -> None:
    p = subprocess.run(["bash", str(SCRIPT), "--tracked"], cwd=tmp_path, capture_output=True, text=True, timeout=60,
                       env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path.parent)})
    assert not (tmp_path / ".git").exists()
    assert p.returncode == 1 and "could not list tracked files" in p.stderr


def test_every_pattern_of_the_script_has_a_case_here() -> None:
    block = SCRIPT.read_text().split("PATTERNS=(")[1].split("\n)")[0]
    entries = [line.strip().strip('"') for line in block.splitlines() if line.strip()]
    entries = [e for e in entries if not e.startswith("#")]
    assert [e.split(":", 1)[0] for e in entries][: len(SHAPES)] == [s[0] for s in SHAPES]


# file names that git prints quoted (#140) -------------------------------------------------------------------

ODD_NAMES = ["café.txt", "sp ace.txt", "tab\there.txt", 'quote"d.txt', "new\nline.txt", "日本語.txt", "-dash.txt",
             "back\\slash.txt", " leading.txt", "trailing.txt "]


@pytest.mark.parametrize("name", ODD_NAMES, ids=lambda n: n.encode("unicode_escape").decode())
def test_a_secret_in_a_file_whose_name_git_quotes_is_found_when_tracked(repo: Path, name: str) -> None:
    commit_file(repo, name, "token = ghp_" + "A" * 40 + "\n")
    p = check(repo)
    assert p.returncode == 1, p.stdout
    assert f"SECRET: GitHub PAT in {name}" in p.stderr


@pytest.mark.parametrize("name", ODD_NAMES, ids=lambda n: n.encode("unicode_escape").decode())
def test_a_secret_in_a_staged_file_whose_name_git_quotes_is_found(repo: Path, name: str) -> None:
    (repo / name).write_text("token = ghp_" + "A" * 40 + "\n")
    git(repo, "add", "--", name)
    p = check(repo, "--staged")
    assert p.returncode == 1, (p.stdout, p.stderr)
    assert f"SECRET: GitHub PAT in {name}" in p.stderr and "could not read staged bytes" not in p.stderr


def test_a_clean_staged_file_with_a_quoted_name_passes_and_is_counted(repo: Path) -> None:
    (repo / "café.txt").write_text("fine\n")
    git(repo, "add", "--", "café.txt")
    p = check(repo, "--staged")
    assert p.returncode == 0 and "1 file(s) clean (0 binary skipped)" in p.stdout


def test_every_tracked_file_with_an_odd_name_is_counted(repo: Path) -> None:
    for name in ODD_NAMES:
        commit_file(repo, name, "fine\n")
    p = check(repo)
    assert p.returncode == 0 and f"{1 + len(ODD_NAMES)} file(s) clean" in p.stdout


def test_the_employer_name_in_a_file_with_a_quoted_name_is_found(repo: Path) -> None:
    commit_file(repo, "café.txt", f"at {EMPLOYER}\n")
    p = check(repo)
    assert p.returncode == 1 and "EMPLOYER: café.txt" in p.stderr


def test_a_tracked_file_that_is_missing_from_the_tree_is_skipped_not_fatal(repo: Path) -> None:
    commit_file(repo, "gone.txt", "fine\n")
    (repo / "gone.txt").unlink()
    p = check(repo)
    assert p.returncode == 0 and "2 file(s) clean" in p.stdout


def test_outside_a_git_repository_staged_mode_fails_and_does_not_pass(tmp_path: Path) -> None:
    p = subprocess.run(["bash", str(SCRIPT), "--staged"], cwd=tmp_path, capture_output=True, text=True, timeout=60,
                       env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tmp_path.parent)})
    assert p.returncode == 1 and "could not list staged files" in p.stderr


def test_no_temporary_file_is_left_behind(repo: Path, tmp_path: Path) -> None:
    scratch = tmp_path / "tmpdir"
    scratch.mkdir()
    commit_file(repo, "café.txt", "fine\n")
    for mode in ("--tracked", "--staged"):
        p = subprocess.run(["bash", str(SCRIPT), mode], cwd=repo, capture_output=True, text=True, timeout=60,
                           env={**os.environ, "TMPDIR": str(scratch)})
        assert p.returncode == 0, p.stderr
    assert list(scratch.iterdir()) == []
