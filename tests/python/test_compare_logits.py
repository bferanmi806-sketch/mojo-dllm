"""compare_logits refuses non-finite logits and a zero norm, and reports a real median (#245)."""

from __future__ import annotations

import importlib.util
import math
import sys
from array import array
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
spec = importlib.util.spec_from_file_location("compare_logits_t", ROOT / "scripts/compare_logits.py")
assert spec is not None and spec.loader is not None
cl = importlib.util.module_from_spec(spec)
sys.modules["compare_logits_t"] = cl
spec.loader.exec_module(cl)


def row(*xs: float) -> array:
    return array("f", xs)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_logit_is_refused(bad: float) -> None:
    with pytest.raises(SystemExit, match="non-finite"):
        cl.require_finite([row(1.0, 2.0), row(1.0, bad)], "new gpu")


def test_the_refusal_names_the_row_and_the_run() -> None:
    with pytest.raises(SystemExit) as e:
        cl.require_finite([row(1.0), row(2.0), row(math.nan)], "base gpu")
    assert "base gpu" in str(e.value) and "row 2" in str(e.value)


def test_finite_rows_pass() -> None:
    cl.require_finite([row(1.0, -2.0), row(0.0, 3.5)], "x")


def test_a_zero_row_has_no_cosine() -> None:
    with pytest.raises(SystemExit, match="zero norm"):
        cl.cos(row(0.0, 0.0), row(1.0, 2.0))
    with pytest.raises(SystemExit, match="zero norm"):
        cl.cos(row(1.0, 2.0), row(0.0, 0.0))


def test_the_cosine_of_known_rows() -> None:
    assert cl.cos(row(1, 0), row(1, 1)) == pytest.approx(math.sqrt(0.5))
    assert cl.cos(row(2, 2), row(1, 1)) == pytest.approx(1.0)


def test_the_median_of_an_even_list_is_the_mean_of_the_middle_pair() -> None:
    assert cl.median([4.0, 1.0, 3.0, 2.0]) == 2.5
    assert cl.median([0.9, 0.1]) == 0.5
    assert cl.median([5.0, 1.0, 3.0]) == 3.0
    assert cl.median([7.0]) == 7.0


def test_the_median_of_nothing_is_refused() -> None:
    with pytest.raises(SystemExit, match="no values"):
        cl.median([])


def test_the_summary_reports_the_real_median_and_the_minima() -> None:
    ref = [row(1, 0), row(0, 1), row(1, 1), row(1, 2)]
    got = [row(1, 0), row(1, 1), row(1, 1), row(2, 1)]
    s = cl.summarize(got, ref, prompt_rows=2)
    c = [1.0, math.sqrt(0.5), 1.0, 4 / 5]
    assert s["min_cosine"] == pytest.approx(min(c))
    assert s["median_cosine"] == pytest.approx(sorted(c)[1] / 2 + sorted(c)[2] / 2)
    assert s["masked_rows_min_cosine"] == pytest.approx(min(c[2:]))
    assert s["argmax_agree_with_cpu"] == 2


def run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rows_for: dict[tuple[str, str], list[list[float]]]) -> dict:
    """main() against a fake `run`: the logits file of each (checkout, device) comes from rows_for."""
    import json

    def fake_run(cmd: list[str]) -> str:
        if cmd[1] == "tokenize":
            return "[5, 6]\n"
        root = Path(cmd[0]).parent.parent.name
        device = cmd[cmd.index("--device") + 1]
        Path(cmd[cmd.index("--out") + 1]).write_bytes(array("f", [v for r in rows_for[(root, device)] for v in r]).tobytes())
        return ""

    for name in ("base", "new"):
        (tmp_path / name / "build").mkdir(parents=True, exist_ok=True)
        (tmp_path / name / "build/mojo-dllm").write_bytes(f"binary of {name}".encode())
    monkeypatch.setattr(cl, "run", fake_run)
    out = tmp_path / "out" / "r.json"
    argv = ["compare_logits", "--model", "m.gguf", "--base", str(tmp_path / "base"), "--new", str(tmp_path / "new"),
            "--out", str(out), "--masked", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(cl, "version", lambda root: "head-of-" + root.name)
    assert cl.main() == 0
    return json.loads(out.read_text())


GOOD = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0]]


def test_main_writes_the_real_median_of_an_even_number_of_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    gpu = [[1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    new = [[1.0, 0.0], [1.0, 0.0], [1.0, 3.0], [1.0, 1.0]]
    doc = run_main(monkeypatch, tmp_path, {("base", "cpu"): GOOD, ("base", "gpu"): gpu, ("new", "gpu"): new})
    c = sorted([1.0, math.sqrt(0.5), 1.0, 0.8])
    assert doc["base"]["median_cosine"] == pytest.approx((c[1] + c[2]) / 2)
    # rows: (1,0).(1,0)=1; (1,1).(1,0)=0.7071; (1,1).(1,3)=4/(sqrt2 sqrt10)=0.8944; (2,1).(1,1)=3/(sqrt5 sqrt2)=0.9487
    ab = sorted([1.0, math.sqrt(0.5), 4 / math.sqrt(20), 3 / math.sqrt(10)])
    assert doc["base_vs_new"]["median_cosine"] == pytest.approx((ab[1] + ab[2]) / 2)
    assert doc["base_vs_new"]["min_cosine"] == pytest.approx(math.sqrt(0.5)) and doc["base_vs_new"]["argmax_agree"] == 3
    assert doc["rows"] == 4 and doc["prompt_rows"] == 2


def test_main_refuses_a_nan_row_from_either_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = [[1.0, 0.0], [1.0, 1.0], [math.nan, 1.0], [2.0, 1.0]]
    for which in ("base", "new"):
        rows = {("base", "cpu"): GOOD, ("base", "gpu"): GOOD, ("new", "gpu"): GOOD}
        rows[(which, "gpu")] = bad
        with pytest.raises(SystemExit, match="non-finite"):
            run_main(monkeypatch, tmp_path, rows)
        assert not (tmp_path / "out" / "r.json").exists()


def test_main_refuses_a_zero_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    zero = [[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]
    with pytest.raises(SystemExit, match="zero norm"):
        run_main(monkeypatch, tmp_path, {("base", "cpu"): GOOD, ("base", "gpu"): zero, ("new", "gpu"): zero})
