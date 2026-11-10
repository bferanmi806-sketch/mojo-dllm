"""tests/python/skip_policy.py: an unexpected skip fails the run, an allowed one does not."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import skip_policy

ROOT = Path(__file__).resolve().parents[2]
PYTHON_TESTS = ROOT / "tests" / "python"
ALLOWLIST = PYTHON_TESTS / "allowed-skips.txt"

SKIPPING = '''
import pytest

def test_ok():
    pass

def test_skips():
    pytest.skip("a made-up reason 42")

@pytest.mark.skipif(True, reason="marked reason")
def test_marked():
    pass
'''


def run_pytest(project: Path, allow: str | None, *extra: str) -> subprocess.CompletedProcess[str]:
    """pytest in a scratch project that loads the policy plugin. Its own config, so the repo's is not read."""
    project.mkdir(parents=True, exist_ok=True)
    (project / "pytest.ini").write_text("[pytest]\naddopts = -rs -p skip_policy -p no:cacheprovider\n")
    (project / "test_sample.py").write_text(SKIPPING)
    args = [sys.executable, "-m", "pytest", "-q", *extra]
    if allow is not None:
        (project / "allow.txt").write_text(allow)
        args += ["--skip-allowlist", str(project / "allow.txt")]
    env = {**os.environ, "PYTHONPATH": str(PYTHON_TESTS), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(args, cwd=project, capture_output=True, text=True, env=env, check=False, timeout=120)


def test_a_skip_the_list_does_not_name_fails_the_run(tmp_path: Path) -> None:
    result = run_pytest(tmp_path / "p", "# nothing is allowed\n")
    assert result.returncode == 1, result.stdout
    assert "unexpected skip: test_sample.py::test_skips: a made-up reason 42" in result.stdout
    assert "unexpected skip: test_sample.py::test_marked: marked reason" in result.stdout
    assert "1 passed, 2 skipped" in result.stdout


def test_skips_the_list_names_pass(tmp_path: Path) -> None:
    allow = "test_sample.py::test_skips | made-up reason\ntest_sample.py::test_m* | marked\n"
    result = run_pytest(tmp_path / "p", allow)
    assert result.returncode == 0, result.stdout
    assert "unexpected" not in result.stdout
    assert "SKIPPED" in result.stdout, "-rs must still list the allowed skips"


def test_the_right_test_with_another_reason_fails(tmp_path: Path) -> None:
    allow = "test_sample.py::test_skips | a different reason\ntest_sample.py::test_marked | marked\n"
    result = run_pytest(tmp_path / "p", allow)
    assert result.returncode == 1
    assert "unexpected skip: test_sample.py::test_skips" in result.stdout
    assert "unexpected skip: test_sample.py::test_marked" not in result.stdout


def test_the_right_reason_on_another_test_fails(tmp_path: Path) -> None:
    allow = "test_other.py::* | made-up reason\ntest_sample.py::test_marked | marked\n"
    result = run_pytest(tmp_path / "p", allow)
    assert result.returncode == 1
    assert "unexpected skip: test_sample.py::test_skips" in result.stdout


def test_a_run_without_skips_passes_with_an_empty_list(tmp_path: Path) -> None:
    result = run_pytest(tmp_path / "p", "", "-k", "test_ok")
    assert result.returncode == 0, result.stdout


def test_a_missing_or_malformed_list_stops_the_run(tmp_path: Path) -> None:
    missing = run_pytest(tmp_path / "a", None, "--skip-allowlist", str(tmp_path / "nope.txt"))
    assert missing.returncode != 0 and "does not exist" in missing.stdout + missing.stderr
    bad = run_pytest(tmp_path / "b", "test_sample.py::test_skips\n")
    assert bad.returncode != 0 and "expected '<test id glob> | <reason text>'" in bad.stdout + bad.stderr


def test_the_repository_runs_pytest_with_the_policy() -> None:
    options = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert "-rs" in options["addopts"].split() and "skip_policy" in options["addopts"]
    assert "tests/python" in options["pythonpath"]


def skip_sites() -> dict[str, list[str]]:
    """{file: [what skips]} for every test file that can skip: a skip call, a skipif or an importorskip."""
    found: dict[str, list[str]] = {}
    for path in sorted(PYTHON_TESTS.glob("test_*.py")):
        if path.name == "test_skip_policy.py":
            continue
        hits = re.findall(r"pytest\.(?:skip|importorskip|mark\.skipif)\(", path.read_text())
        if hits:
            found[f"tests/python/{path.name}"] = hits
    return found


def test_every_file_that_can_skip_is_in_the_list() -> None:
    """A skip that does not fire here (a root-only skip, a busy machine) is still judged, so another machine
    cannot be the first to find that it is unlisted."""
    listed = {glob.split("::", 1)[0] for glob, _ in skip_policy.load_allowlist(ALLOWLIST)}
    assert sorted(set(skip_sites()) - listed) == []
    assert listed - {name for name in listed if (ROOT / name).is_file()} == set(), "an entry names a missing file"


def test_the_shipped_list_parses_and_has_no_blank_reason() -> None:
    entries = skip_policy.load_allowlist(ALLOWLIST)
    assert entries and all(glob and reason for glob, reason in entries)


@pytest.mark.parametrize(
    ("nodeid", "reason", "allowed"),
    [
        ("tests/python/test_capped.py::test_kill", "set CAPPED_TEST_KILL=1 to kill", True),
        ("tests/python/test_capped.py::test_kill", "some other reason", False),
        ("tests/python/test_new.py::test_kill", "set CAPPED_TEST_KILL=1 to kill", False),
        ("tests/python/test_bench_quiet.py::test_x[a-b]", "the refusal is for non-root users", True),
    ],
)
def test_is_allowed_checks_both_halves(nodeid: str, reason: str, allowed: bool) -> None:
    assert skip_policy.is_allowed(nodeid, reason, skip_policy.load_allowlist(ALLOWLIST)) is allowed
