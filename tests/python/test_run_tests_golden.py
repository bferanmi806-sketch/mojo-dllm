"""scripts/run-tests.sh does not let an exported UPDATE_GOLDEN=1 rewrite the golden file behind a passing run.

tests/test_golden.mojo rewrites tests/fixtures/golden-generation.tsv and then compares against what it wrote, so
with the variable exported every run passes by construction. The script unsets it unless the caller asks with
--update-golden. The compiler is replaced by a stub build script whose "binary" records the variable it sees.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BUILD_STUB = """#!/usr/bin/env bash
# usage: build.sh SOURCE OUTPUT. Writes an executable that records UPDATE_GOLDEN and prints a passing summary.
out="$2"
cat > "$out" <<'BIN'
#!/usr/bin/env bash
echo "${UPDATE_GOLDEN-unset}" >> "$SEEN_LOG"
echo "Summary [ 0.001 ] 1 tests run: 1 passed , 0 failed , 0 skipped"
BIN
chmod +x "$out"
"""


def run(tmp_path: Path, *args: str, golden: str | None = "1") -> tuple[subprocess.CompletedProcess[str], list[str]]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "scripts" / "run-tests.sh").write_bytes((ROOT / "scripts" / "run-tests.sh").read_bytes())
    (repo / "scripts" / "build.sh").write_text(BUILD_STUB)
    (repo / "tests" / "test_a.mojo").write_text("")
    (repo / "tests" / "test-count.txt").write_text("1\n")
    (repo / "tests" / "test-counts.txt").write_text("test_a 1\n")
    log = tmp_path / "seen.log"
    log.touch()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "SEEN_LOG": str(log), "MOJO": "unused"}
    if golden is not None:
        env["UPDATE_GOLDEN"] = golden
    result = subprocess.run(
        ["bash", "scripts/run-tests.sh", *args], cwd=repo, env=env, capture_output=True, text=True, check=False, timeout=60
    )
    return result, log.read_text().splitlines()


def test_an_exported_update_golden_is_not_seen_by_the_tests(tmp_path: Path) -> None:
    result, seen = run(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert seen == ["unset"]


def test_the_script_says_it_ignored_the_variable(tmp_path: Path) -> None:
    result, _ = run(tmp_path)
    assert "ignoring UPDATE_GOLDEN=1" in result.stdout + result.stderr and "--update-golden" in result.stdout + result.stderr


def test_without_the_variable_nothing_is_said_and_nothing_is_set(tmp_path: Path) -> None:
    result, seen = run(tmp_path, golden=None)
    assert result.returncode == 0 and seen == ["unset"]
    assert "UPDATE_GOLDEN" not in result.stdout + result.stderr


def test_the_flag_asks_for_a_rewrite(tmp_path: Path) -> None:
    result, seen = run(tmp_path, "--update-golden", golden=None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert seen == ["1"]


def test_an_unknown_argument_is_a_usage_error_before_anything_runs(tmp_path: Path) -> None:
    result, seen = run(tmp_path, "--nope")
    assert result.returncode == 2 and "usage" in result.stderr and seen == []
