"""eval_gsm8k compare: the paired statistics are value-checked, few discordant pairs get an honest interval,
and a partial arm or an existing file is refused (#237)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import eval_gsm8k  # noqa: E402

T, F = True, False


@pytest.fixture
def evalroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(eval_gsm8k, "ROOT", tmp_path)
    monkeypatch.setattr(eval_gsm8k, "RUNS", tmp_path / "bench/eval/runs")
    (tmp_path / "bench/eval/runs").mkdir(parents=True)
    return tmp_path


def arm(root: Path, name: str, correct: list[bool]) -> None:
    rows = [dict(index=i, correct=c, forward_passes=4, generate_s=1.0 + i, flags=["--x"], version="v",
                 model="m.gguf") for i, c in enumerate(correct)]
    (root / "bench/eval/runs" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def compare(root: Path, a: str, b: str) -> dict:
    assert eval_gsm8k.cmd_compare(a, b) == 0
    (path,) = (root / "bench/eval").glob(f"*-{a}-vs-{b}.json")
    return json.loads(path.read_text())


def test_the_paired_statistics_have_the_right_values(evalroot: Path) -> None:
    arm(evalroot, "a", [T, T, F, F])
    arm(evalroot, "b", [T, F, T, T])
    doc = compare(evalroot, "a", "b")
    assert doc["discordant"] == {"a_only": 1, "b_only": 2}
    assert doc["difference"] == 0.25 and doc["accuracy"] == {"a": 0.5, "b": 0.75}
    lo, hi = doc["ci95"]
    assert lo < 0.25 < hi and -1.0 <= lo and hi <= 1.0
    assert doc["mcnemar_p"] == 1.0 and doc["ci95_method"] == "newcombe-hybrid"


def test_swapping_the_arms_swaps_the_discordant_counts_and_flips_the_difference(evalroot: Path) -> None:
    arm(evalroot, "a", [T, T, T, F, F, F])
    arm(evalroot, "b", [T, F, F, T, T, T])
    ab, ba = compare(evalroot, "a", "b"), compare(evalroot, "b", "a")
    assert ab["discordant"] == {"a_only": 2, "b_only": 3} and ba["discordant"] == {"a_only": 3, "b_only": 2}
    assert ab["difference"] == pytest.approx(1 / 6) and ba["difference"] == pytest.approx(-1 / 6)
    assert ab["ci95"][0] == pytest.approx(-ba["ci95"][1]) and ab["ci95"][1] == pytest.approx(-ba["ci95"][0])


def test_one_problem_arms_give_finite_keys_or_null(evalroot: Path) -> None:
    arm(evalroot, "a", [T])
    arm(evalroot, "b", [F])
    doc = compare(evalroot, "a", "b")
    assert doc["difference"] == -1.0 and doc["discordant"] == {"a_only": 1, "b_only": 0}
    assert doc["ci95"] is None or all(math.isfinite(x) for x in doc["ci95"])
    assert math.isfinite(doc["mcnemar_p"])


def test_arms_that_never_disagree_have_no_interval(evalroot: Path) -> None:
    arm(evalroot, "a", [T, F, T, F])
    arm(evalroot, "b", [T, F, T, F])
    doc = compare(evalroot, "a", "b")
    assert doc["ci95"] is None and "not estimable" in doc["ci95_note"]
    assert doc["difference"] == 0.0 and doc["mcnemar_p"] == 1.0


def test_arms_of_unequal_length_are_refused_by_name(evalroot: Path) -> None:
    arm(evalroot, "a", [T] * 37)
    arm(evalroot, "b", [T] * 200)
    with pytest.raises(SystemExit) as e:
        eval_gsm8k.cmd_compare("a", "b")
    assert "'a' has 37" in str(e.value) and "'b' has 200" in str(e.value)
    assert not list((evalroot / "bench/eval").glob("*-vs-*.json"))


def test_an_existing_output_file_is_not_overwritten(evalroot: Path) -> None:
    arm(evalroot, "a", [T, F, T])
    arm(evalroot, "b", [T, T, F])
    doc = compare(evalroot, "a", "b")
    (path,) = (evalroot / "bench/eval").glob("*-a-vs-b.json")
    before = path.read_bytes()
    arm(evalroot, "b", [F, F, F])
    with pytest.raises(SystemExit) as e:
        eval_gsm8k.cmd_compare("a", "b")
    assert "exists" in str(e.value) and path.read_bytes() == before and doc["problems"] == 3


# the interval itself


def tango(both: int, b: int, c: int, neither: int) -> tuple[float, float]:
    """Tango's score interval for a paired difference, found by bisection: an independent check on the hybrid."""
    n, r = both + b + c + neither, both + neither

    def score(d0: float) -> float:
        bq, cq = (b + c) - d0 * (b - c), d0 * (b - c) + r * d0 * d0
        q = max((bq + math.sqrt(bq * bq + 4 * n * cq)) / (2 * n), abs(d0) + 1e-12)
        u = c / (q + d0) - b / (q - d0)
        i_dd, i_dq = n * q / (q * q - d0 * d0), -n * d0 / (q * q - d0 * d0)
        return u * u / (i_dd - i_dq**2 / (i_dd + n / (1 - q)))

    z2 = eval_gsm8k.Z95**2

    def root(lo: float, hi: float) -> float:
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if (score(lo) > z2) == (score(mid) > z2) else (lo, mid)
        return (lo + hi) / 2

    dh = (c - b) / n
    return root(-0.999, dh), root(dh, 0.999)


@pytest.mark.parametrize("cells", [(150, 2, 6, 42), (100, 14, 16, 70), (100, 2, 1, 97), (100, 1, 1, 98), (80, 5, 5, 110)])
def test_the_hybrid_interval_agrees_with_the_score_interval(cells: tuple[int, int, int, int]) -> None:
    d, lo, hi = eval_gsm8k.paired_ci(*cells)
    t_lo, t_hi = tango(*cells)
    assert lo is not None and hi is not None and lo < d < hi
    assert abs(lo - t_lo) < 0.01 and abs(hi - t_hi) < 0.01, ((lo, hi), (t_lo, t_hi))


def test_three_discordant_pairs_are_not_a_narrow_interval() -> None:
    """The committed threshold09 row: 2 against 1 on 200 problems. The Wald interval was -2.2 to +1.2 points."""
    both, b, c = 161, 2, 1
    _, lo, hi = eval_gsm8k.paired_ci(both, b, c, 200 - both - b - c)
    assert lo is not None and hi is not None
    assert lo < -0.0220 and hi > 0.0120, "wider than the Wald interval it replaces"


def test_the_interval_stays_inside_the_possible_differences() -> None:
    _, lo, hi = eval_gsm8k.paired_ci(0, 5, 0, 0)
    assert lo is not None and hi is not None and -1.0 <= lo <= hi <= 1.0


def test_zero_problems_is_an_error_not_a_division() -> None:
    with pytest.raises(ValueError):
        eval_gsm8k.paired_ci(0, 0, 0, 0)


@pytest.mark.parametrize(
    "cells,want",
    [
        # correlation adjustment: more than n/2 (161 and 36 solved by both or neither), between 0 and n/2, and negative
        ((161, 2, 1, 36), (-0.027949906465478187, 0.017441128529654536)),
        ((1, 1, 1, 1), (-0.4949196078428281, 0.4949196078428281)),
        ((0, 5, 5, 0), (-0.5268138189748719, 0.5268138189748719)),
        ((3, 1, 2, 2), (-0.30281691156969776, 0.4962412092101879)),
    ],
)
def test_the_hybrid_interval_values_are_pinned_on_each_correlation_branch(
    cells: tuple[int, int, int, int], want: tuple[float, float]
) -> None:
    _, lo, hi = eval_gsm8k.paired_ci(*cells)
    assert (lo, hi) == pytest.approx(want, abs=1e-12)
