"""bench_table re-derives every figure it renders from the runs, not only ms/step (#253)."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bt = load("bench_table_summary_t", ROOT / "scripts/bench_table.py")
rb = load("run_bench_summary_t", ROOT / "bench/run_bench.py")


def run(rt: str, i: int, seconds: float, startup: float, rss: int) -> dict:
    return dict(runtime=rt, ok=True, generate_s=seconds, ms_per_step=seconds * 1000 / 32, startup_s=startup,
                peak_rss_bytes=rss, forward_passes=32,
                isolation=dict(clean=True, noisy=False))


@pytest.fixture
def doc() -> dict:
    runs = [run("mojo-dllm", i, s, st, rss) for i, (s, st, rss) in enumerate(
        [(2.0, 0.5, 4_000_000_000), (4.0, 0.7, 5_000_000_000), (3.0, 0.9, 4_500_000_000)])]
    runs.append(dict(runtime="mojo-dllm", ok=False))
    return dict(config=dict(gen_length=128), runs=runs, summary=rb.summarize_runs(runs, ["mojo-dllm"], 128))


def test_a_consistent_summary_passes(doc: dict) -> None:
    bt.check_summary("f.json", doc)
    sm = doc["summary"]["mojo-dllm"]
    assert sm["tokens_per_s_median"] == pytest.approx(128 / 3.0) and sm["startup_s_median"] == 0.7
    assert sm["peak_rss_gb_max"] == 5.0


@pytest.mark.parametrize("field", ["tokens_per_s_median", "startup_s_median", "peak_rss_gb_max"])
def test_a_hand_edited_figure_is_refused_and_named(doc: dict, field: str) -> None:
    doc["summary"]["mojo-dllm"][field] *= 1.01
    with pytest.raises(SystemExit) as e:
        bt.check_summary("f.json", doc)
    assert field in str(e.value) and "f.json" in str(e.value) and "mojo-dllm" in str(e.value)


@pytest.mark.parametrize("field", ["tokens_per_s_median", "startup_s_median", "peak_rss_gb_max"])
def test_a_figure_deleted_or_nulled_is_refused(doc: dict, field: str) -> None:
    for edit in (lambda sm: sm.pop(field), lambda sm: sm.__setitem__(field, None)):
        broken = copy.deepcopy(doc)
        edit(broken["summary"]["mojo-dllm"])
        with pytest.raises(SystemExit, match=field):
            bt.check_summary("f.json", broken)


def test_a_figure_that_should_be_null_is_refused_when_it_holds_a_number(doc: dict) -> None:
    for r in doc["runs"]:
        r.pop("startup_s", None)
    doc["summary"]["mojo-dllm"]["startup_s_median"] = None
    bt.check_summary("f.json", doc)                         # no run measured it, and the summary says so
    doc["summary"]["mojo-dllm"]["startup_s_median"] = 0.7
    with pytest.raises(SystemExit, match="startup_s_median"):
        bt.check_summary("f.json", doc)


def test_a_figure_within_rounding_of_the_runs_passes(doc: dict) -> None:
    doc["summary"]["mojo-dllm"]["tokens_per_s_median"] *= 1 + 1e-12
    bt.check_summary("f.json", doc)


def test_the_gen_length_of_the_config_is_what_the_rate_uses(doc: dict) -> None:
    doc["config"]["gen_length"] = 256
    with pytest.raises(SystemExit, match="tokens_per_s_median"):
        bt.check_summary("f.json", doc)


def test_the_real_evidence_still_passes() -> None:
    import json
    files = sorted((ROOT / "bench/results").glob("*.json"))
    checked = 0
    for f in files:
        d = json.loads(f.read_text())
        if d.get("partial", True) or d["config"].get("kind") == "fairness":
            continue
        bt.check_summary(f.name, d)
        checked += 1
    assert checked >= 5
