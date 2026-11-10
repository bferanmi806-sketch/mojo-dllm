"""scripts/bench_table.py cannot pass by checking nothing (#141, #142).

`--check` compares each document with itself spliced with the tables, so a missing marker, a missing document or an
empty evidence directory used to leave the document unchanged and print "tables match the evidence". The tables are
stubs here and the documents live in a throwaway tree; the real evidence is covered by `bench_table.py --check` in CI.
"""

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
    spec = importlib.util.spec_from_file_location("bench_table_main_t", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_table_main_t"] = module
    spec.loader.exec_module(module)
    return module


bt = load()


def doc(*keys: str, body: str = "old") -> str:
    return "# t\n\n" + "".join(f"<!-- {k}:start -->\n{body}\n<!-- {k}:end -->\n\n" for k in keys)


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "site"
    (root / "docs").mkdir(parents=True)
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


def current(site: Path, rel: str) -> str:
    return (site / rel).read_text()


def run(argv: list[str]) -> int:
    return bt.main(argv)


# write and check still work ---------------------------------------------------------------------------------

def test_write_fills_every_table_and_check_then_passes(site: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--check"]) == 1                                  # the stubs differ from "old"
    assert "README.md" in capsys.readouterr().err
    assert run(["--write"]) == 0
    assert "BENCH" in current(site, "README.md")
    assert all(x in current(site, "docs/benchmarks.md") for x in ("BENCH", "QUALITY", "EVAL"))
    assert all(x in current(site, "docs/correctness.md") for x in ("PARITY", "GENPARITY"))
    capsys.readouterr()
    assert run(["--check"]) == 0
    assert "tables match the evidence" in capsys.readouterr().out


def test_check_names_each_stale_document(site: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(["--write"])
    (site / "docs/correctness.md").write_text(current(site, "docs/correctness.md").replace("GENPARITY", "edited by hand"))
    capsys.readouterr()
    assert run(["--check"]) == 1
    err = capsys.readouterr().err
    assert "docs/correctness.md" in err and "README.md" not in err


def test_check_does_not_write(site: Path) -> None:
    before = current(site, "README.md")
    run(["--check"])
    assert current(site, "README.md") == before


# missing marker, document and evidence (#141) ---------------------------------------------------------------

def test_a_renamed_start_marker_is_an_error_naming_the_file_and_the_marker(site: Path) -> None:
    (site / "README.md").write_text(current(site, "README.md").replace("<!-- bench:start -->", "<!-- bench:startX -->"))
    with pytest.raises(SystemExit) as e:
        run(["--check"])
    assert "README.md" in str(e.value) and "<!-- bench:start -->" in str(e.value) and "missing" in str(e.value)


def test_a_missing_end_marker_is_an_error(site: Path) -> None:
    (site / "docs/benchmarks.md").write_text(current(site, "docs/benchmarks.md").replace("<!-- quality:end -->", ""))
    with pytest.raises(SystemExit) as e:
        run(["--write"])
    assert "docs/benchmarks.md" in str(e.value) and "<!-- quality:end -->" in str(e.value)


def test_a_table_with_no_markers_in_a_document_that_requires_it_is_an_error(site: Path) -> None:
    (site / "docs/correctness.md").write_text(doc("parity"))        # genparity is required here
    with pytest.raises(SystemExit) as e:
        run(["--check"])
    assert "docs/correctness.md" in str(e.value) and "genparity:start" in str(e.value)


def test_a_marker_pair_in_the_wrong_order_or_twice_is_an_error(site: Path) -> None:
    (site / "README.md").write_text("<!-- bench:end -->\nx\n<!-- bench:start -->\n")
    with pytest.raises(SystemExit, match="comes before"):
        run(["--check"])
    (site / "README.md").write_text(doc("bench") + doc("bench"))
    with pytest.raises(SystemExit, match="2 times"):
        run(["--check"])


def test_a_missing_target_document_is_an_error(site: Path) -> None:
    (site / "docs/correctness.md").unlink()
    with pytest.raises(SystemExit) as e:
        run(["--check"])
    assert "docs/correctness.md does not exist" in str(e.value)


@pytest.mark.parametrize("key,rel", [("quality", "docs/benchmarks.md"), ("eval", "docs/benchmarks.md"),
                                     ("bench", "README.md")])
def test_an_empty_evidence_directory_is_an_error_when_the_document_holds_its_markers(
        site: Path, monkeypatch: pytest.MonkeyPatch, key: str, rel: str) -> None:
    stubs = {"quality": ("quality", lambda: ""), "eval": ("eval_table", lambda allow_noisy=False: ""),
             "bench": ("render", lambda: "")}
    name, stub = stubs[key]
    monkeypatch.setattr(bt, name, stub)
    with pytest.raises(SystemExit) as e:
        run(["--check"])
    literal = {"quality": "bench/quality/*.json", "eval": "bench/eval/*-vs-*.json", "bench": "bench/results/*.json"}[key]
    assert rel in str(e.value) and f"{key} table" in str(e.value) and literal in str(e.value)


def test_an_optional_table_that_is_not_in_a_document_is_not_spliced_into_it(site: Path) -> None:
    run(["--write"])
    assert "QUALITY" not in current(site, "README.md") and "PARITY" not in current(site, "README.md")


def test_the_required_tables_of_each_document_are_these() -> None:
    assert bt.REQUIRED == {"README.md": ("bench",), "docs/benchmarks.md": ("bench", "quality", "eval"),
                           "docs/correctness.md": ("parity", "genparity")}
    assert bt.EVIDENCE == {"bench": "bench/results/*.json", "parity": "bench/parity/*.json",
                           "genparity": "bench/parity/*.json", "quality": "bench/quality/*.json",
                           "eval": "bench/eval/*-vs-*.json"}


def test_an_end_marker_alone_in_a_document_that_does_not_require_it_is_an_error(site: Path) -> None:
    (site / "README.md").write_text(doc("bench") + "<!-- quality:end -->\n")
    with pytest.raises(SystemExit) as e:
        run(["--check"])
    assert "README.md" in str(e.value) and "<!-- quality:start -->" in str(e.value)


def test_allow_noisy_reaches_the_eval_table(site: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(bt, "eval_table", lambda allow_noisy=False: seen.append(allow_noisy) or "EVAL")
    run(["--check", "--allow-noisy"])
    run(["--check"])
    assert seen == [True, False]


def test_the_real_documents_hold_every_required_marker_once() -> None:
    for rel, keys in bt.REQUIRED.items():
        text = (ROOT / rel).read_text()
        for key in keys:
            start, end = bt.MARKERS[key]
            assert text.count(start) == 1 and text.count(end) == 1 and text.index(start) < text.index(end), (rel, key)


def test_the_real_check_passes_on_this_tree() -> None:
    p = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
