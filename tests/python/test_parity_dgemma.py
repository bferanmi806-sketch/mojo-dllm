"""The DiffusionGemma side of bench/parity.py: the bound, the file contract and the refusals.

The numeric comparison needs numpy, which the `test` group does not carry, so these tests
cover what surrounds it with plain data. The comparison on real logits is the reference
side's evidence in the PR, not a unit test.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
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


parity = load_module("parity_dgemma", ROOT / "bench/parity.py")

GOOD = dict(min_cosine=0.999, argmax_agree=64, rows=64, mean_top5_overlap=4.9, max_rel_rms=0.05, top1_max_rel_err=0.01)
BOUND = dict(
    min_cosine=0.9951, argmax_fraction=0.95, mean_top5_overlap=4.6, max_rel_rms=0.12, max_top1_rel_err=0.12
)


def test_splitmix64_matches_the_published_vector() -> None:
    # The reference implementation's first output for state 0 is 0xE220A8397B1DCDAF.
    assert parity.splitmix64_stream(0).__next__() == 0xE220A8397B1DCDAF


def test_canvas_is_fixed_and_inside_the_vocabulary() -> None:
    a = parity.dg_canvas(262144, 256, seed=66)
    assert a == parity.dg_canvas(262144, 256, seed=66)
    assert a != parity.dg_canvas(262144, 256, seed=67)
    assert len(a) == 256 and all(0 <= t < 262144 for t in a)
    assert len(set(a)) > 250, "a canvas of repeated ids would hide a position bug"


def test_token_ids_are_read_from_the_last_line_of_llama_tokenize() -> None:
    assert parity.parse_token_ids("noise\n[2, 1234, 5]\n") == [2, 1234, 5]
    for bad in ("", "[]", "no ids here", "[2, x]"):
        with pytest.raises(SystemExit):
            parity.parse_token_ids(bad)


def test_the_pin_is_read_from_the_build_script() -> None:
    pin = parity.dg_pinned_commit(ROOT)
    assert len(pin) == 40 and pin in (ROOT / "scripts/build_dg_ref.sh").read_text()


def test_the_checkout_must_be_the_pinned_commit(tmp_path: Path) -> None:
    repo = tmp_path / "llama"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["commit", "-q", "--allow-empty", "-m", "one"]):
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args], check=True
        )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    parity.dg_require_commit(repo, head)
    with pytest.raises(SystemExit) as e:
        parity.dg_require_commit(repo, "0" * 40)
    assert head in str(e.value) and "0" * 40 in str(e.value)


def write_llada(root: Path, entries: list[dict]) -> None:
    folder = root / "bench/parity"
    folder.mkdir(parents=True)
    (folder / "2026-10-03-llada.json").write_text(json.dumps({"logits": entries}))


def test_the_bound_is_the_worst_case_of_the_committed_llada_evidence(tmp_path: Path) -> None:
    write_llada(
        tmp_path,
        [
            dict(min_cosine=0.995, argmax_agree=62, rows=64, mean_top5_overlap=4.83, max_rel_rms=0.12),
            dict(min_cosine=0.999, argmax_agree=120, rows=126, mean_top5_overlap=4.67, max_rel_rms=0.05),
        ],
    )
    bound = parity.llada_bound(tmp_path)
    assert bound["min_cosine"] == 0.995
    assert bound["argmax_fraction"] == pytest.approx(120 / 126)
    assert bound["mean_top5_overlap"] == 4.67
    assert bound["max_rel_rms"] == 0.12
    assert bound["max_top1_rel_err"] == 0.12
    assert bound["source"] == "bench/parity/2026-10-03-llada.json"


def test_the_real_bound_follows_the_real_file() -> None:
    entries = json.loads((ROOT / "bench/parity/2026-10-03-llada.json").read_text())["logits"]
    bound = parity.llada_bound(ROOT)
    assert bound["min_cosine"] == min(e["min_cosine"] for e in entries)
    assert bound["max_rel_rms"] == max(e["max_rel_rms"] for e in entries)
    assert bound["argmax_fraction"] == min(e["argmax_agree"] / e["rows"] for e in entries)
    assert bound["max_top1_rel_err"] == bound["max_rel_rms"], "the top logit gets the relative error LLaDA's rows reach"


def test_a_missing_llada_file_is_an_error_not_a_free_pass(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as e:
        parity.llada_bound(tmp_path)
    assert "2026-10-03-llada.json" in str(e.value)


def test_a_result_equal_to_the_bound_passes_and_each_metric_can_fail_alone() -> None:
    at_bound = dict(
        min_cosine=0.9951, argmax_agree=95, rows=100, mean_top5_overlap=4.6, max_rel_rms=0.12, top1_max_rel_err=0.12
    )
    assert parity.dg_violations(at_bound, BOUND) == []
    assert parity.dg_violations(GOOD, BOUND) == []
    worse = {
        "min_cosine": dict(GOOD, min_cosine=0.9950),
        "argmax": dict(GOOD, argmax_agree=60),
        "top-5": dict(GOOD, mean_top5_overlap=4.5),
        "relative RMS": dict(GOOD, max_rel_rms=0.13),
        "top logit": dict(GOOD, top1_max_rel_err=0.13),
    }
    for name, metrics in worse.items():
        found = parity.dg_violations(metrics, BOUND)
        assert len(found) == 1 and name in found[0], (name, found)


def make_case(
    workdir: Path, variants: tuple[str, ...] = parity.DG_VARIANTS, mojo: tuple[str, ...] = (), spread: bool = True
) -> Path:
    case = workdir / "case0"
    case.mkdir(parents=True)
    meta = {"prompt": "p", "prompt_tokens": [2, 3], "canvas_tokens": [5, 6]}
    if spread:
        meta["ref_spread"] = {"cached vs unified": dict(min_cosine=1.0, argmax_agree=64, rows=64)}
    (case / "case.json").write_text(json.dumps(meta))
    for v in variants:
        (case / f"{v}.ref.f32").write_bytes(b"ref:" + v.encode())
    for v in mojo:
        (case / f"{v}.mojo.f32").write_bytes(b"mojo:" + v.encode())
    return case


def fake_compare(a: object, b: object, first_mask: int) -> dict:
    return dict(GOOD, rows=64)


def test_missing_mojo_output_stops_the_comparison_and_names_the_file(tmp_path: Path) -> None:
    case = make_case(tmp_path)
    calls: list[object] = []

    def compare(a: object, b: object, first_mask: int) -> dict:
        calls.append(a)
        return fake_compare(a, b, first_mask)

    with pytest.raises(SystemExit) as e:
        parity.dg_compare(tmp_path, BOUND, load=lambda p: p.read_bytes(), compare_fn=compare)
    message = str(e.value)
    assert str(case / "unified.mojo.f32") in message and "#64" in message
    assert calls == [], "nothing may be compared while a mojo-dllm output is missing"


def test_a_mojo_output_that_is_a_copy_of_the_reference_is_refused(tmp_path: Path) -> None:
    case = make_case(tmp_path, mojo=parity.DG_VARIANTS)
    (case / "cached.mojo.f32").write_bytes((case / "cached.ref.f32").read_bytes())
    with pytest.raises(SystemExit) as e:
        parity.dg_compare(tmp_path, BOUND, load=lambda p: p.read_bytes(), compare_fn=fake_compare)
    assert "identical" in str(e.value) and "cached.mojo.f32" in str(e.value)


def test_a_complete_set_is_compared_variant_by_variant(tmp_path: Path) -> None:
    make_case(tmp_path, mojo=parity.DG_VARIANTS)
    entries, violations = parity.dg_compare(
        tmp_path, BOUND, load=lambda p: p.read_bytes(), compare_fn=fake_compare
    )
    assert violations == []
    assert [e["variant"] for e in entries] == list(parity.DG_VARIANTS)
    assert all(e["prompt"] == "p" and e["rows"] == 64 for e in entries)


def test_a_variant_outside_the_bound_is_reported_with_its_name(tmp_path: Path) -> None:
    make_case(tmp_path, mojo=parity.DG_VARIANTS)

    def compare(a: bytes, b: bytes, first_mask: int) -> dict:
        return dict(GOOD, min_cosine=0.5) if a.endswith(b"cached-sc") else dict(GOOD)

    _, violations = parity.dg_compare(tmp_path, BOUND, load=lambda p: p.read_bytes(), compare_fn=compare)
    assert len(violations) == 1 and "cached-sc" in violations[0] and "min_cosine" in violations[0]


def test_no_cases_is_an_error_not_an_empty_pass(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as e:
        parity.dg_compare(tmp_path, BOUND, load=lambda p: p.read_bytes(), compare_fn=fake_compare)
    assert "--phase ref" in str(e.value)


def test_a_missing_reference_output_is_named_not_a_traceback(tmp_path: Path) -> None:
    case = make_case(tmp_path, mojo=parity.DG_VARIANTS)
    (case / "unified-sc.ref.f32").unlink()
    with pytest.raises(SystemExit) as e:
        parity.dg_compare(tmp_path, BOUND, load=lambda p: p.read_bytes(), compare_fn=fake_compare)
    assert str(case / "unified-sc.ref.f32") in str(e.value) and "--phase ref" in str(e.value)


def test_llada_evidence_with_no_comparisons_gives_no_bound(tmp_path: Path) -> None:
    write_llada(tmp_path, [])
    with pytest.raises(SystemExit) as e:
        parity.llada_bound(tmp_path)
    assert "no logit comparisons" in str(e.value)


def test_a_short_or_missing_pin_in_the_build_script_is_refused(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for body in ('PINNED_COMMIT="2e715c0639c6"\n', "no pin here\n", f'PINNED_COMMIT="{"a" * 40}"\nPINNED_COMMIT="{"b" * 40}"\n'):
        (scripts / "build_dg_ref.sh").write_text(body)
        with pytest.raises(SystemExit):
            parity.dg_pinned_commit(tmp_path)
    (scripts / "build_dg_ref.sh").write_text(f'PINNED_COMMIT="{"c" * 40}"\n')
    assert parity.dg_pinned_commit(tmp_path) == "c" * 40


def test_each_entry_carries_the_reference_spread_of_its_case(tmp_path: Path) -> None:
    case = make_case(tmp_path, mojo=parity.DG_VARIANTS)
    spread = {"cached vs unified": dict(min_cosine=0.92, argmax_agree=149, rows=256)}
    meta = json.loads((case / "case.json").read_text())
    (case / "case.json").write_text(json.dumps(dict(meta, ref_spread=spread)))
    entries, _ = parity.dg_compare(tmp_path, BOUND, load=lambda p: p.read_bytes(), compare_fn=fake_compare)
    assert all(e["reference_spread"] == spread for e in entries)


def test_a_case_without_a_recorded_spread_is_refused(tmp_path: Path) -> None:
    make_case(tmp_path, mojo=parity.DG_VARIANTS, spread=False)
    # Without it a failure could not be read against the reference's own sensitivity to rounding.
    with pytest.raises(SystemExit) as e:
        parity.dg_compare(tmp_path, BOUND, load=lambda p: p.read_bytes(), compare_fn=fake_compare)
    assert "ref_spread" in str(e.value) and "--phase ref" in str(e.value)


class Rows:
    """Stands in for a numpy matrix: max(axis=1) gives the row maxima, any other axis gives the column maxima."""

    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def max(self, axis: int) -> "Rows":
        out = [max(r) for r in self.rows] if axis == 1 else [max(c) for c in zip(*self.rows)]
        return Rows([out])

    def tolist(self) -> list[float]:
        return self.rows[0]


def test_the_top_logit_error_is_the_worst_row_relative_to_the_reference() -> None:
    # Row maxima 30 vs 30 (0), 27 vs 30 (0.1), 15 vs 30 (0.5): the worst is 0.5. Neither argument order nor
    # sign may matter to the magnitude, and the denominator is the reference (b).
    assert parity.dg_top1_error([30.0, 27.0, 15.0], [30.0, 30.0, 30.0]) == pytest.approx(0.5)
    assert parity.dg_top1_error([30.0, 33.0], [30.0, 30.0]) == pytest.approx(0.1)
    assert parity.dg_top1_error([60.0], [30.0]) == pytest.approx(1.0)
    assert parity.dg_top1_error([30.0], [60.0]) == pytest.approx(0.5), "relative to the second argument"
    for bad in ([], [1.0]):
        with pytest.raises(SystemExit):
            parity.dg_top1_error(bad, [1.0, 2.0])


def test_dg_metrics_reads_row_maxima_along_axis_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parity, "compare", lambda a, b, first: {"rows": 2})
    # Row maxima 10 and 20 against 10 and 10: the metric is 1.0. Column maxima (axis 0) would be 20 and 20
    # against 10 and 10, a different number.
    a, b = Rows([[10.0, 1.0], [2.0, 20.0]]), Rows([[10.0, 1.0], [2.0, 10.0]])
    m = parity.dg_metrics(a, b, 0)
    assert m["top1_max_rel_err"] == pytest.approx(1.0) and m["rows"] == 2
    # It differs from every other metric's view of the same data: the cap-removed case has equal argmax.
    same = parity.dg_metrics(Rows([[10.0, 1.0]]), Rows([[10.0, 1.0]]), 0)
    assert same["top1_max_rel_err"] == 0.0
