"""Noisy and unchecked runs are counted in the summary and named next to the table (#233)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_bench = load_module("run_bench_noisy", ROOT / "bench/run_bench.py")
tables = load_module("bench_table_noisy", ROOT / "scripts/bench_table.py")

ROWS = [("a", "**A**", "x")]
CLEAN = {"clean": True, "noisy": False, "reasons": []}
NOISY = {"clean": False, "noisy": True, "reasons": ["swap 9 pages"]}


def run(ms: float, iso: dict | None, busy: float = 0.01, rt: str = "a") -> dict:
    r = dict(runtime=rt, ok=True, ms_per_step=ms, startup_s=1.0, generate_s=2.0, peak_rss_bytes=1, cpu_busy_before=busy,
             rep=0)
    if iso is not None:
        r["isolation"] = iso
    return r


def test_the_summary_counts_clean_noisy_and_unchecked_runs() -> None:
    runs = [run(10, CLEAN), run(11, CLEAN), run(50, NOISY), run(12, None)]
    sm = run_bench.summarize_runs(runs, ["a"], 128)["a"]
    assert (sm["clean_runs"], sm["noisy_runs"], sm["unchecked_runs"]) == (2, 1, 1)
    assert sm["runs_ok"] == 4 and sm["ms_per_step_median"] == 11.5, "a noisy run stays in the median"


def test_a_window_that_says_noisy_is_never_clean_whatever_else_it_says() -> None:
    assert run_bench.window_state(run(1, {"clean": True, "noisy": True})) == "noisy"
    assert run_bench.window_state(run(1, {"clean": False})) == "noisy"
    assert run_bench.window_state(run(1, {})) == "noisy", "a window without a verdict is not a clean one"
    assert run_bench.window_state(run(1, None)) == "unchecked"


def test_the_table_names_noisy_runs_and_the_quiet_median() -> None:
    runs = [run(10, CLEAN), run(11, CLEAN), run(50, NOISY)]
    note = " ".join(tables.busy_note(runs, 0.1, ROWS))
    assert "1 of 3 runs ended in a noisy isolation window" in note
    assert "A 10 ms / step and 1.0 s startup (2 runs)" in note, "the median of the two clean runs, not of three"
    assert "quiet runs alone give" in note


def test_a_mixed_file_names_the_runs_without_a_window() -> None:
    runs = [run(10, CLEAN), run(11, None)]
    note = " ".join(tables.busy_note(runs, 0.1, ROWS))
    assert "1 of 2 runs carry no isolation window" in note


def test_a_file_from_before_the_windows_adds_no_sentence() -> None:
    assert tables.busy_note([run(10, None), run(11, None)], 0.1, ROWS) == []


def test_a_run_without_cpu_busy_before_is_refused_not_read_as_quiet() -> None:
    bad = run(10, CLEAN)
    del bad["cpu_busy_before"]
    with pytest.raises(SystemExit) as e:
        tables.busy_note([bad], 0.1, ROWS)
    assert "cpu_busy_before" in str(e.value)


def test_a_summary_that_disagrees_with_the_windows_is_refused() -> None:
    doc = dict(runs=[run(10, CLEAN), run(50, NOISY)],
               summary={"a": dict(runs_ok=2, noisy_runs=0, ms_per_step_median=30.0)})
    with pytest.raises(SystemExit) as e:
        tables.check_summary("f.json", doc)
    assert "noisy_runs" in str(e.value)
