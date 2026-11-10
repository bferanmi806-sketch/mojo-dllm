"""bench/parity.py's real `compare` and `dg_metrics` on known values (#250).

These run numpy, which the `test` group does not carry: the module is skipped, with that reason, unless numpy is
installed, and CI runs it in its own step under the `fixtures` group (`uv run --frozen --only-group test
--only-group fixtures pytest tests/python/test_parity_compare.py -rs`), where a skip is a failure of the step.
"""

from __future__ import annotations

import importlib.util
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
