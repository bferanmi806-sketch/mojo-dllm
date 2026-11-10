"""scripts/run-tests.sh pins the total and each file's count (#248), and refuses what it cannot count.

The script and scripts/build.sh are copied into a throwaway tree whose `mojo` is a stub: the stub "compiles" a test
file into a program that prints the summary line the real TestSuite prints, with the count written in the file's
first line (`# tests: N`). Nothing builds Mojo and nothing touches the real build/ directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

FAKE_MOJO = r"""#!/usr/bin/env bash
# fake `mojo build ... FILE -o OUT`: OUT prints a TestSuite summary with the count FILE declares on its first line.
while [ "$#" -gt 0 ]; do
  case "$1" in -I) shift 2 ;; -o) out="$2"; shift 2 ;; --target-accelerator) shift 2 ;; build|-Werror) shift ;; *) file="$1"; shift ;; esac
done
declared="$(head -n 1 "$file")"
mkdir -p "$(dirname "$out")"
case "$declared" in
  "# build-fails") echo "error: boom" >&2; exit 1 ;;
  "# tests: "*) n="${declared#\# tests: }"; body="echo 'Summary [ 1 ms ] $n tests run: $n passed , 0 failed , 0 skipped'; exit 0" ;;
  "# fails") body="echo 'FAIL: one'; echo 'Summary [ 1 ms ] 3 tests run: 2 passed , 1 failed , 0 skipped'; exit 1" ;;
  "# no-summary") body="echo hello; exit 0" ;;
  "# exit-1-all-passed") body="echo 'Summary [ 1 ms ] 3 tests run: 3 passed , 0 failed , 0 skipped'; exit 1" ;;
  *) echo "bad declaration: $declared" >&2; exit 2 ;;
esac
printf '#!/usr/bin/env bash\n%s\n' "$body" > "$out"
chmod +x "$out"
"""


class Tree:
    def __init__(self, tmp: Path) -> None:
        self.root = tmp / "repo"
        (self.root / "scripts").mkdir(parents=True)
        for name in ("run-tests.sh", "build.sh"):
            shutil.copy(ROOT / "scripts" / name, self.root / "scripts" / name)
        (self.root / "tests").mkdir()
        self.mojo = tmp / "fake-mojo"
        self.mojo.write_text(FAKE_MOJO)
        self.mojo.chmod(0o755)

    def test_file(self, name: str, declaration: str) -> None:
        (self.root / "tests" / f"{name}.mojo").write_text(declaration + "\n")

    def tests(self, **counts: int) -> None:
        for name, n in counts.items():
            self.test_file(name, f"# tests: {n}")

    def pin(self, total: int | str | None, files: str | None) -> None:
        for name, text in (("test-count.txt", None if total is None else f"{total}\n"), ("test-counts.txt", files)):
            path = self.root / "tests" / name
            path.unlink(missing_ok=True)
            if text is not None:
                path.write_text(text)

    def run(self, **env: str) -> subprocess.CompletedProcess[str]:
        e = {**os.environ, "MOJO": str(self.mojo), "MOJO_DLLM_GPU_ARCH": "sm_80", **env}
        return subprocess.run(["bash", str(self.root / "scripts/run-tests.sh")], env=e, capture_output=True,
                              text=True, timeout=120)

    def pinned(self, name: str) -> str:
        return (self.root / "tests" / name).read_text()


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    t = Tree(tmp_path)
    t.tests(test_a=3, test_b=2)
    t.pin(5, "test_a 3\ntest_b 2\n")
    return t


def test_matching_pins_pass(tree: Tree) -> None:
    p = tree.run()
    assert p.returncode == 0, p.stderr
    assert "OK: 5 tests passed (pinned 5)" in p.stdout and "every file runs the number of tests pinned" in p.stdout


def test_update_writes_the_total_and_each_file(tree: Tree) -> None:
    tree.tests(test_a=4, test_c=1)
    p = tree.run(UPDATE_TEST_COUNT="1")
    assert p.returncode == 0 and "pinned test count: 7" in p.stdout
    assert tree.pinned("test-count.txt") == "7\n"
    assert tree.pinned("test-counts.txt") == "test_a 4\ntest_b 2\ntest_c 1\n"
    assert tree.run().returncode == 0


def test_tests_moved_between_files_fail_the_per_file_pin_though_the_total_is_unchanged(tree: Tree) -> None:
    tree.tests(test_a=1, test_b=4)                        # two tests moved from a to b: still 5 in all
    p = tree.run()
    assert p.returncode == 1
    assert "OK: 5 tests passed (pinned 5)" in p.stdout     # the total check alone is satisfied
    assert "differ from" in p.stderr
    assert "pinned: test_a 3" in p.stderr and "ran:    test_a 1" in p.stderr
    assert "pinned: test_b 2" in p.stderr and "ran:    test_b 4" in p.stderr
    assert "UPDATE_TEST_COUNT=1" in p.stderr


def test_a_new_file_that_is_not_pinned_fails(tree: Tree) -> None:
    tree.tests(test_c=0 + 1)
    tree.pin(6, "test_a 3\ntest_b 2\n")
    p = tree.run()
    assert p.returncode == 1 and "ran:    test_c 1" in p.stderr


def test_a_pinned_file_that_no_longer_runs_fails(tree: Tree) -> None:
    (tree.root / "tests/test_b.mojo").unlink()
    tree.pin(3, "test_a 3\ntest_b 2\n")
    p = tree.run()
    assert p.returncode == 1 and "pinned: test_b 2" in p.stderr


def test_a_missing_per_file_pin_is_a_failure_and_names_the_fix(tree: Tree) -> None:
    tree.pin(5, None)
    p = tree.run()
    assert p.returncode == 1 and "test-counts.txt is missing" in p.stderr and "UPDATE_TEST_COUNT=1" in p.stderr


def test_the_total_pin_still_fails_on_its_own(tree: Tree) -> None:
    tree.pin(9, "test_a 3\ntest_b 2\n")
    p = tree.run()
    assert p.returncode == 1 and "5 tests passed but" in p.stderr and "pins 9" in p.stderr


def test_the_pin_file_order_does_not_matter(tree: Tree) -> None:
    tree.pin(5, "test_b 2\ntest_a 3\n")
    assert tree.run().returncode == 0
