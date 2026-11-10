"""scripts/bench_table.py refuses an unknown or missing mode instead of exiting 0 with nothing checked (#142)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "bench_table.py"
sys.path.insert(0, str(ROOT / "bench"))


def load():
    spec = importlib.util.spec_from_file_location("bench_table_mode_t", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_table_mode_t"] = module
    spec.loader.exec_module(module)
    return module


bt = load()


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Three documents with the markers, and tables that are stubs: a refused mode must leave them as they were."""
    root = tmp_path / "site"
    (root / "docs").mkdir(parents=True)
    def doc(*keys: str) -> str:
        return "# t\n\n" + "".join(f"<!-- {k}:start -->\nold\n<!-- {k}:end -->\n\n" for k in keys)

    (root / "README.md").write_text(doc("bench"))
    (root / "docs/benchmarks.md").write_text(doc("bench", "quality", "eval"))
    (root / "docs/correctness.md").write_text(doc("parity", "genparity"))
    monkeypatch.setattr(bt, "ROOT", root)
    monkeypatch.setattr(bt, "TARGETS", [root / "README.md", root / "docs/benchmarks.md", root / "docs/correctness.md"])
    monkeypatch.setattr(bt, "render", lambda: "BENCH")
    monkeypatch.setattr(bt, "parity", lambda: ("PARITY", "GENPARITY"))
    monkeypatch.setattr(bt, "quality", lambda: "QUALITY")
    monkeypatch.setattr(bt, "eval_table", lambda allow_noisy=False: "EVAL")
    return root


def run(argv: list[str]) -> int:
    return bt.main(argv)


@pytest.mark.parametrize("argv", [["--chek"], [], ["check"], ["--check", "--write"], ["--allow-noisy"], ["--write", "x"], [""]])
def test_a_mode_that_is_not_exactly_one_of_write_and_check_exits_two_with_usage(
        site: Path, capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    before = {p: p.read_text() for p in bt.TARGETS}
    assert run(argv) == 2
    err = capsys.readouterr().err
    assert "usage: bench_table.py --write | --check" in err
    assert {p: p.read_text() for p in bt.TARGETS} == before


def test_the_script_itself_exits_two_on_a_typo_and_on_no_argument() -> None:
    for argv in (["--chek"], []):
        p = subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, timeout=120)
        assert p.returncode == 2 and "usage: bench_table.py" in p.stderr and p.stdout == ""




def test_a_mode_with_allow_noisy_runs_and_passes_the_flag_on(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(bt, "eval_table", lambda allow_noisy=False: seen.append(allow_noisy) or "EVAL")
    assert run(["--check", "--allow-noisy"]) == 1          # ran (the stub table differs from the document), not a usage error
    assert run(["--allow-noisy", "--write"]) == 0
    assert seen == [True, True]
    assert run(["--check"]) == 0 and seen == [True, True, False]
