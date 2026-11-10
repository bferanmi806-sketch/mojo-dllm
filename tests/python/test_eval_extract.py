"""eval_gsm8k.extract and canon: wrapped answers, huge numbers and the committed rows (#236)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import eval_gsm8k  # noqa: E402

extract, canon = eval_gsm8k.extract, eval_gsm8k.canon


@pytest.mark.parametrize(
    "text,want",
    [
        ("The answer is \\(\\boxed{18}\\) after 2 days.", "18"),
        ("The answer is \\( 282 \\) minutes per week.", "282"),
        ("The answer is \\[ 40 \\] then 7 more", "40"),
        ("The answer is **18** apples, 3 left over", "18"),
        ("The answer is $\\boxed{1,250}$ dollars, 9 more", "1250"),
        ("The answer is: \\boxed{-4}.", "-4"),
        ("so 5 + 3 = 8. The answer is 8.", "8"),
        ("The answer is $1,250.", "1250"),
        ("first 3, then 7, finally 42", "42"),
    ],
)
def test_wrapped_answers_are_read_before_later_numbers(text: str, want: str) -> None:
    assert extract(text) == want


def test_a_huge_answer_does_not_abort_the_arm() -> None:
    assert extract("The answer is " + "9" * 400) == "9" * 400
    assert extract("The answer is " + "9" * 5000) == "9" * 5000, "past Python's int-to-string limit"


def test_a_twenty_digit_answer_keeps_every_digit() -> None:
    assert extract("The answer is 12345678901234567890.") == "12345678901234567890"
    assert canon("90071992547409931") == "90071992547409931", "above 2^53 a float would round this"


@pytest.mark.parametrize(
    "raw,want",
    [
        ("-0", "0"), ("-0.0", "0"), ("0", "0"), ("007", "7"), ("", None), ("abc", None), ("1.2.3", None),
        ("1,2,3", "123"), ("18.00", "18"), ("1,000", "1000"), ("12.50", "12.5"), ("-3.", "-3"), ("0.10", "0.1"),
        ("-12.5", "-12.5"), (".5", "0.5"), ("-", None), ("-.", None), (".", None), ("inf", None), ("nan", None), ("1e5", None),
    ],
)
def test_canon_is_total_and_canonical(raw: str, want: str | None) -> None:
    assert canon(raw) == want


def test_no_number_gives_none() -> None:
    assert extract("no number here") is None
    assert extract("") is None


def test_the_committed_rows_score_as_stored() -> None:
    """Every committed row, including the two that carry a wrapped answer (114 and 830), reads as it did."""
    seen = 0
    for path in sorted((ROOT / "bench/eval/runs").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            pred = extract(r["text"])
            assert pred == r["pred"], (path.name, r["index"], pred, r["pred"])
            assert (pred == canon(r["gold"])) == r["correct"], (path.name, r["index"])
            seen += 1
    assert seen >= 600
    base = {json.loads(l)["index"]: json.loads(l) for l in (ROOT / "bench/eval/runs/baseline.jsonl").read_text().splitlines()}
    assert extract(base[114]["text"]) == "5" and extract(base[830]["text"]) == "282"
