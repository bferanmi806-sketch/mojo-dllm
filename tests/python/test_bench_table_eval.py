"""The GSM8K table of scripts/bench_table.py says only what its compare documents say (#232)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tables = load_module("bench_table_eval", ROOT / "scripts/bench_table.py")

BASE, ARM = "baseline", "other"


def compare_doc(limit: int = 0, **over: object) -> dict:
    cfg = dict(model="m.gguf", flags=["--steps", "256"], version="abc")
    doc: dict = dict(
        arms=[BASE, ARM], problems=200 if not limit else limit, limit=limit,
        config={BASE: cfg, ARM: dict(cfg)}, accuracy={BASE: 0.5, ARM: 0.5}, difference=0.0, ci95=[-0.1, 0.1],
        discordant={"a_only": 3, "b_only": 3}, mcnemar_p=1.0,
        forward_passes_mean={BASE: 256.0, ARM: 128.0}, generate_s_mean={BASE: 90.0, ARM: 45.0},
    )
    doc.update(over)
    return doc


def render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, full: dict, timed: dict | None, allow_noisy: bool = True) -> str:
    folder = tmp_path / "bench" / "eval"
    folder.mkdir(parents=True)
    (folder / f"d-{BASE}-vs-{ARM}.json").write_text(json.dumps(full))
    if timed is not None:
        (folder / f"d-{BASE}-vs-{ARM}-first50.json").write_text(json.dumps(timed))
    monkeypatch.setattr(tables, "ROOT", tmp_path)
    return tables.eval_table(allow_noisy)


def isolation(noisy: int, unchecked: int, clean_rows: int, mean: float | None = 40.0) -> dict:
    return dict(
        noisy_rows={BASE: noisy, ARM: 0}, unchecked_rows={BASE: unchecked, ARM: 0}, clean_rows=clean_rows,
        generate_s_mean_clean={BASE: mean, ARM: mean},
    )


def test_a_null_pass_count_prints_n_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = compare_doc(forward_passes_mean={BASE: 256.0, ARM: None})
    text = render(tmp_path, monkeypatch, full, compare_doc(50, forward_passes_mean={BASE: None, ARM: None}))
    assert f"| {ARM} |" in text and "| n/a |" in text
    base_row = next(l for l in text.splitlines() if l.startswith(f"| {BASE} |"))
    assert " n/a " not in base_row, "the baseline's count comes from the full comparison"


def test_a_null_baseline_count_prints_n_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full = compare_doc(forward_passes_mean={BASE: None, ARM: None})
    text = render(tmp_path, monkeypatch, full, None)
    base_row = next(l for l in text.splitlines() if l.startswith(f"| {BASE} |"))
    assert " n/a " in base_row


def test_a_document_without_isolation_data_does_not_claim_an_idle_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = render(tmp_path, monkeypatch, compare_doc(), compare_doc(50))
    assert "no isolation record" in text
    assert "nothing else on the machine" not in text and "came back clean" not in text


def test_every_window_clean_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = render(tmp_path, monkeypatch, compare_doc(), compare_doc(50, **isolation(0, 0, 50)))
    assert "came back clean" in text and "no isolation record" not in text


def test_noisy_rows_print_their_counts_and_the_clean_mean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timed = compare_doc(50, **isolation(noisy=4, unchecked=2, clean_rows=44, mean=41.2))
    text = render(tmp_path, monkeypatch, compare_doc(), timed)
    assert "4 noisy and 2 unchecked" in text and "came back clean" not in text
    assert "41.2 s" in text and "90.0 s" not in text, "the clean mean replaces the all-row mean"


def test_an_arm_with_no_clean_row_has_no_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timed = compare_doc(50, **isolation(noisy=50, unchecked=0, clean_rows=0, mean=None))
    text = render(tmp_path, monkeypatch, compare_doc(), timed)
    assert "no clean row" in text and "45.0 s" not in text


def test_different_weights_are_footnoted_with_both_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = dict(model="m.gguf", flags=["--steps", "256"], version="abc")
    full = compare_doc(
        weights_differ=True, models={BASE: "m.gguf", ARM: "hf-4bit"}, config={BASE: cfg, ARM: dict(cfg, model="hf-4bit")}
    )
    text = render(tmp_path, monkeypatch, full, None)
    assert f"| {ARM} ‡ |" in text
    assert "hf-4bit" in text.split("‡")[-1] and "includes the weights" in text


def test_same_weights_carry_no_footnote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = render(tmp_path, monkeypatch, compare_doc(weights_differ=False), None)
    assert "‡" not in text


def test_a_timed_round_with_noisy_rows_is_refused_unless_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timed = compare_doc(50, **isolation(noisy=4, unchecked=2, clean_rows=44))
    with pytest.raises(SystemExit) as e:
        render(tmp_path, monkeypatch, compare_doc(), timed, allow_noisy=False)
    assert "44 clean rows for 50 problems" in str(e.value) and "--allow-noisy" in str(e.value)


def test_a_fully_clean_timed_round_needs_no_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = render(tmp_path, monkeypatch, compare_doc(), compare_doc(50, **isolation(0, 0, 50)), allow_noisy=False)
    assert "came back clean" in text
