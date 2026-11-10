"""bench/parity.py's real `compare` and `dg_metrics` on known values (#250).

These run numpy, which the `test` group does not carry: the module is skipped, with that reason, unless numpy is
installed, and CI runs it in its own step under the `fixtures` group (`uv run --frozen --only-group test
--only-group fixtures pytest tests/python/test_parity_compare.py -rs`), where a skip is a failure of the step.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy is in the `fixtures` group, not the `test` group; CI runs this file there")

ROOT = Path(__file__).resolve().parents[2]


def load_parity():
    spec = importlib.util.spec_from_file_location("parity_compare_t", ROOT / "bench/parity.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["parity_compare_t"] = module
    spec.loader.exec_module(module)
    return module


parity = load_parity()
A = np.array([[1.0, 0.0], [0.0, 1.0]])
B = np.array([[1.0, 0.0], [1.0, 1.0]])


def test_compare_known_values() -> None:
    m = parity.compare(A, B, 1)
    assert m["rows"] == 2 and m["argmax_agree"] == 1
    assert m["min_cosine"] == pytest.approx(math.sqrt(0.5))
    assert m["median_cosine"] == pytest.approx((1.0 + math.sqrt(0.5)) / 2)
    assert m["masked_median_cosine"] == pytest.approx(math.sqrt(0.5))        # cos[1:] holds only row 1
    assert m["max_rel_rms"] == pytest.approx(math.sqrt(0.5)) and m["median_rel_rms"] == pytest.approx(math.sqrt(0.5) / 2)
    assert m["mean_top5_overlap"] == 2.0 and m["worst_rows"] == [1, 0]


def test_compare_of_identical_rows_is_exact() -> None:
    m = parity.compare(B, B, 0)
    assert m["min_cosine"] == pytest.approx(1.0) and m["max_rel_rms"] == 0.0 and m["argmax_agree"] == 2


def test_the_masked_median_starts_at_first_mask() -> None:
    a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]])
    assert parity.compare(a, b, 2)["masked_median_cosine"] == pytest.approx(1.0)
    assert parity.compare(a, b, 1)["masked_median_cosine"] == pytest.approx((1.0 + math.sqrt(0.5)) / 2)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_logit_is_refused_on_either_side(bad: float) -> None:
    for a, b in ((np.array([[1.0, bad], [0.0, 1.0]]), B), (A, np.array([[1.0, 0.0], [bad, 1.0]]))):
        with pytest.raises(SystemExit, match="non-finite"):
            parity.compare(a, b, 0)


def test_a_zero_row_is_refused_and_named_not_averaged_as_nan() -> None:
    with pytest.raises(SystemExit, match=r"row 1.*zero"):
        parity.compare(np.array([[1.0, 0.0], [0.0, 0.0]]), B, 0)
    with pytest.raises(SystemExit, match=r"row 0.*zero"):
        parity.compare(A, np.array([[0.0, 0.0], [1.0, 1.0]]), 0)


@pytest.mark.parametrize("first_mask", [2, 3, -1])
def test_a_first_mask_that_leaves_no_masked_row_is_refused(first_mask: int) -> None:
    with pytest.raises(SystemExit, match="first_mask"):
        parity.compare(A, B, first_mask)


def test_arrays_of_different_shape_or_no_rows_are_refused() -> None:
    with pytest.raises(SystemExit, match="shape"):
        parity.compare(A, np.array([[1.0, 0.0]]), 0)
    with pytest.raises(SystemExit, match="shape"):
        parity.compare(A, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0)
    with pytest.raises(SystemExit, match="shape"):
        parity.compare(np.zeros((0, 2)), np.zeros((0, 2)), 0)
    with pytest.raises(SystemExit, match="shape"):
        parity.compare(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0)


def test_dg_metrics_adds_the_largest_logit_error_of_real_arrays() -> None:
    a = np.array([[10.0, 1.0], [2.0, 20.0]])
    b = np.array([[10.0, 1.0], [2.0, 10.0]])
    m = parity.dg_metrics(a, b, 0)
    assert m["top1_max_rel_err"] == pytest.approx(1.0) and m["rows"] == 2 and m["argmax_agree"] == 2


def test_the_top_logit_error_refuses_a_zero_reference_row_maximum() -> None:
    with pytest.raises(SystemExit, match="zero"):
        parity.dg_top1_error([1.0, 2.0], [1.0, 0.0])
    assert parity.dg_top1_error([1.0, 3.0], [1.0, 2.0]) == pytest.approx(0.5)


# dg_compare's own shape and empty guard, on real files (#165) ----------------------------------------------------

BOUND = dict(min_cosine=0.5, argmax_fraction=0.0, mean_top5_overlap=0.0, max_rel_rms=10.0, max_top1_rel_err=10.0)
VOCAB = 4


def dg_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mojo_rows: int | None = 3, ref_rows: int = 3) -> Path:
    """Case folders with real f32 files; `mojo_rows` None writes an empty mojo file. The sidecar check is stubbed."""
    work = tmp_path / "work"
    cases = []
    for i, logit_case in enumerate(parity.LOGIT_CASES):
        d = work / f"case{i}"
        d.mkdir(parents=True)
        meta = dict(prompt=logit_case.prompt, prompt_tokens=[2, 3], n_vocab=VOCAB, ref_spread={"x": 1})
        (d / "case.json").write_text(json.dumps(meta))
        cases.append(d / "case.json")
        for v in parity.DG_VARIANTS:
            ref = np.arange(1, ref_rows * VOCAB + 1, dtype=np.float32).reshape(ref_rows, VOCAB)
            ref.tofile(d / f"{v}.ref.f32")
            if mojo_rows is None:
                (d / f"{v}.mojo.f32").write_bytes(b"")
            else:
                (ref[:mojo_rows] * 1.01 if mojo_rows <= ref_rows else np.ones((mojo_rows, VOCAB), np.float32)).astype(
                    np.float32).tofile(d / f"{v}.mojo.f32")
    monkeypatch.setattr(parity, "dg_case_files", lambda workdir: cases)
    monkeypatch.setattr(parity, "dg_check_sidecar", lambda name, mojo, meta: "abc1234")
    return work


def test_dg_compare_on_real_files_of_one_shape_compares_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries, violations = parity.dg_compare(dg_folder(tmp_path, monkeypatch), BOUND)
    assert len(entries) == 2 * len(parity.DG_VARIANTS) and all(e["rows"] == 3 for e in entries) and violations == []


def test_dg_compare_refuses_an_empty_mojo_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit, match="empty or differ in shape"):
        parity.dg_compare(dg_folder(tmp_path, monkeypatch, mojo_rows=None), BOUND)


@pytest.mark.parametrize("mojo_rows", [2, 4])
def test_dg_compare_refuses_files_of_different_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mojo_rows: int) -> None:
    with pytest.raises(SystemExit, match="empty or differ in shape"):
        parity.dg_compare(dg_folder(tmp_path, monkeypatch, mojo_rows=mojo_rows), BOUND)


def test_the_median_cosine_is_a_median_and_not_a_mean() -> None:
    """Cosines 1, 0.99 and 0.5: the median is 0.99, the mean 0.83."""
    a = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [math.cos(math.acos(0.99)), math.sin(math.acos(0.99))],
                  [math.cos(math.acos(0.5)), math.sin(math.acos(0.5))]])
    m = parity.compare(a, b, 0)
    assert m["median_cosine"] == pytest.approx(0.99) and m["min_cosine"] == pytest.approx(0.5)
    assert m["masked_median_cosine"] == pytest.approx(0.99)
