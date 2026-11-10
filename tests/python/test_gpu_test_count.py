"""scripts/check_gpu_test_count.sh: the GPU suite's test count against its pin, on a machine without a GPU."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_gpu_test_count.sh"


def run_script(directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env={"PATH": "/usr/bin:/bin", "TEST_DIR": str(directory)},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def suite(directory: Path, tests: dict[str, int], pin: str | None) -> Path:
    """A directory of test_*.mojo files with the given number of `def test_` each, and a pin file."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, n in tests.items():
        body = "from testing import TestSuite\n\n" + "".join(f"def test_{i}() raises:\n    pass\n\n" for i in range(n))
        (directory / name).write_text(body + "def main() raises:\n    TestSuite.discover_tests[__functions_in_module()]().run()\n")
    if pin is not None:
        (directory / "test-count.txt").write_text(pin)
    return directory


def test_the_shipped_suite_matches_its_pin() -> None:
    result = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, check=False, cwd=ROOT, timeout=30)
    assert result.returncode == 0, result.stderr
    pin = int((ROOT / "tests" / "gpu" / "test-count.txt").read_text())
    assert f"{pin} tests" in result.stdout


def test_a_matching_count_passes(tmp_path: Path) -> None:
    result = run_script(suite(tmp_path, {"test_a.mojo": 3, "test_b.mojo": 2}, "5\n"))
    assert result.returncode == 0, result.stderr
    assert "5 tests" in result.stdout


def test_an_added_test_without_a_new_pin_fails(tmp_path: Path) -> None:
    result = run_script(suite(tmp_path, {"test_a.mojo": 4}, "3\n"))
    assert result.returncode == 1
    assert "4 tests" in result.stderr and "pins 3" in result.stderr and "UPDATE_TEST_COUNT=1" in result.stderr


def test_a_removed_test_without_a_new_pin_fails(tmp_path: Path) -> None:
    assert run_script(suite(tmp_path, {"test_a.mojo": 2}, "3\n")).returncode == 1


def test_a_commented_out_or_nested_def_is_not_a_test(tmp_path: Path) -> None:
    directory = suite(tmp_path, {"test_a.mojo": 2}, "2\n")
    with (directory / "test_a.mojo").open("a") as handle:
        handle.write("# def test_commented() raises:\n    def test_nested() raises:\n        pass\n")
    assert run_script(directory).returncode == 0


def test_a_missing_or_unreadable_pin_fails(tmp_path: Path) -> None:
    missing = run_script(suite(tmp_path / "a", {"test_a.mojo": 1}, None))
    assert missing.returncode == 1 and "pin" in missing.stderr
    junk = run_script(suite(tmp_path / "b", {"test_a.mojo": 1}, "many\n"))
    assert junk.returncode == 1 and "does not hold a test count" in junk.stderr
    assert "does not hold a test count" in missing.stderr


def test_a_directory_with_no_test_files_fails(tmp_path: Path) -> None:
    result = run_script(suite(tmp_path, {}, "0\n"))
    assert result.returncode == 1 and "no test files" in result.stderr


def test_a_file_that_stops_defining_tests_fails_even_when_the_pin_follows(tmp_path: Path) -> None:
    """The zero-tests case run-tests.sh refuses: a file with no test is a file that dropped out."""
    result = run_script(suite(tmp_path, {"test_a.mojo": 0, "test_b.mojo": 2}, "2\n"))
    assert result.returncode == 1 and "test_a.mojo" in result.stderr


def test_a_file_grep_cannot_read_fails_instead_of_counting_zero(tmp_path: Path) -> None:
    directory = suite(tmp_path, {"test_a.mojo": 2}, "2\n")
    (directory / "test_unreadable.mojo").mkdir()  # grep exits 2 on a directory
    result = run_script(directory)
    assert result.returncode == 1 and "grep could not read" in result.stderr
