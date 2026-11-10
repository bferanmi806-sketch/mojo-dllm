"""One meaning for forward_passes, a validated engine result, and arm rows that say what they are (#241)."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import engines  # noqa: E402
import eval_engine  # noqa: E402
import eval_gsm8k  # noqa: E402
from engines import EngineError, GenResult  # noqa: E402
from test_bench_isolation_wiring import gsm8k, live  # noqa: E402,F401  (fixtures)


def ok(**over: object) -> GenResult:
    return GenResult(**{"text": "The answer is 18.", "generate_s": 1.5, **over})  # type: ignore[arg-type]


# the result record


def test_a_minimal_result_is_valid_and_readable_by_key() -> None:
    g = ok()
    assert g.forward_passes is None and g.steps is None and g["text"] == "The answer is 18."
    assert g.get("finish_reason") is None and g.get("nonsense", 7) == 7
    with pytest.raises(KeyError):
        g["nonsense"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.text = "x"  # type: ignore[misc]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0, 0.0, -1.0, True, "1.5", None])
def test_a_timing_that_is_not_a_positive_finite_number_is_refused(bad: object) -> None:
    with pytest.raises(EngineError) as e:
        ok(generate_s=bad)
    assert "generate_s" in str(e.value)


@pytest.mark.parametrize("bad", ["", "   \n", None, 3])
def test_blank_or_missing_text_is_refused(bad: object) -> None:
    with pytest.raises(EngineError):
        ok(text=bad)


@pytest.mark.parametrize("field,bad", [
    ("forward_passes", 0), ("forward_passes", -3), ("forward_passes", 2.5), ("forward_passes", True),
    ("steps", 0), ("steps", "7"), ("prompt_tokens", -1), ("completion_tokens", -1), ("completion_tokens", 1.0),
    ("finish_reason", 3), ("stopped_early", "yes"),
])
def test_each_count_and_flag_is_checked(field: str, bad: object) -> None:
    with pytest.raises(EngineError) as e:
        ok(**{field: bad})
    assert field in str(e.value)


def test_valid_counts_pass() -> None:
    g = ok(forward_passes=7, steps=7, prompt_tokens=41, completion_tokens=9, finish_reason="length", stopped_early=False)
    assert (g.forward_passes, g.completion_tokens, g.finish_reason) == (7, 9, "length")


# the engines

@pytest.mark.parametrize("name", ["llamacpp", "hf", "vllm", "sglang"])
def test_every_registered_engine_generates_a_genresult(name: str) -> None:
    src = (ROOT / f"bench/engines/{name}.py").read_text()
    assert "GenResult(" in src and "return dict(" not in src.split("def generate")[-1]


# the arm rows and the runner


class FakeEngine:
    name = "fake"
    result: object = None

    def __init__(self, args: object) -> None:
        pass

    @staticmethod
    def add_arguments(parser: object) -> None:
        pass

    def version(self) -> str:
        return "v1"

    def check(self, cfg: object) -> None:
        pass

    def flags(self, cfg: object) -> dict:
        return {"engine": "fake", "z": 1, "a": 2, "forward_passes_measured": False}

    def generate(self, prompt: str, cfg: object) -> object:
        return FakeEngine.result


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch, gsm8k: Path) -> type[FakeEngine]:
    monkeypatch.setitem(engines._REGISTRY, "fake", FakeEngine)
    return FakeEngine


def run_engine(root: Path, arm: str = "e1") -> int:
    return eval_engine.main(["run", "--engine", "fake", "--model", str(root / "model.gguf"), "--arm", arm,
                             "--device", "cpu", "--limit", "2", "--allow-battery"])


def test_engine_rows_say_what_they_are_and_keep_truncation_visible(gsm8k: Path, fake_engine: type[FakeEngine]) -> None:
    fake_engine.result = ok(forward_passes=None, steps=256, prompt_tokens=40, completion_tokens=256,
                            finish_reason="length")
    assert run_engine(gsm8k) == 0
    rows = [json.loads(l) for l in (gsm8k / "runs/e1.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["schema"] == "engine" and r["forward_passes"] is None and r["steps"] == 256 for r in rows)
    assert all(r["finish_reason"] == "length" and r["completion_tokens"] == 256 for r in rows)


def test_a_dict_from_an_engine_is_refused_by_name(gsm8k: Path, fake_engine: type[FakeEngine]) -> None:
    fake_engine.result = {"text": "The answer is 1.", "generate_s": 1.0, "forward_passes": 4}
    with pytest.raises(SystemExit) as e:
        run_engine(gsm8k)
    assert "dict" in str(e.value) and "GenResult" in str(e.value)
    assert not (gsm8k / "runs/e1.jsonl").exists()


def test_a_result_with_a_nan_timing_stops_the_run_without_a_row(gsm8k: Path, fake_engine: type[FakeEngine]) -> None:
    class Bad(FakeEngine):
        def generate(self, prompt: str, cfg: object) -> object:
            return ok(generate_s=math.nan)

    engines._REGISTRY["fake"] = Bad
    with pytest.raises(SystemExit) as e:
        run_engine(gsm8k)
    assert "generate_s" in str(e.value) and not (gsm8k / "runs/e1.jsonl").exists()


def test_a_resume_with_a_changed_flag_is_still_refused(gsm8k: Path, fake_engine: type[FakeEngine]) -> None:
    fake_engine.result = ok(steps=8)
    assert run_engine(gsm8k) == 0

    class Changed(FakeEngine):
        def flags(self, cfg: object) -> dict:
            return {**super().flags(cfg), "z": 2}

    engines._REGISTRY["fake"] = Changed
    with pytest.raises(SystemExit) as e:
        run_engine(gsm8k)
    assert "was started with" in str(e.value)


def test_mojo_rows_say_mojo_cli(gsm8k: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_bench_isolation_wiring import run_gsm8k

    assert run_gsm8k(monkeypatch, gsm8k) == 0
    rows = [json.loads(l) for l in (gsm8k / "runs/m1.jsonl").read_text().splitlines()]
    assert all(r["schema"] == "mojo-cli" for r in rows)


# compare


@pytest.fixture
def evalroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(eval_gsm8k, "ROOT", tmp_path)
    monkeypatch.setattr(eval_gsm8k, "RUNS", tmp_path / "bench/eval/runs")
    (tmp_path / "bench/eval/runs").mkdir(parents=True)
    return tmp_path


def arm(root: Path, name: str, flags: object, passes: list, key_order: bool = False) -> None:
    rows = []
    for i, p in enumerate(passes):
        f = dict(reversed(list(flags.items()))) if key_order and i % 2 and isinstance(flags, dict) else flags
        rows.append(dict(index=i, correct=i % 2 == 0, forward_passes=p, generate_s=1.0 + i, flags=f, version="v",
                         model="m.gguf"))
    (root / "bench/eval/runs" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def compare(root: Path, a: str, b: str) -> dict:
    assert eval_gsm8k.cmd_compare(a, b) == 0
    (path,) = (root / "bench/eval").glob(f"*-{a}-vs-{b}.json")
    return json.loads(path.read_text())


def test_an_arm_that_did_not_measure_passes_has_no_mean_even_if_its_rows_hold_a_number(evalroot: Path) -> None:
    arm(evalroot, "mojo", ["--steps", "4"], [4, 4, 4])
    arm(evalroot, "llama", {"engine": "llamacpp", "forward_passes_measured": False}, [4, 4, 4])   # an older file
    doc = compare(evalroot, "mojo", "llama")
    assert doc["forward_passes_mean"] == {"mojo": 4.0, "llama": None}
    assert doc["forward_passes_rows"] == {"mojo": 3, "llama": 0}
    assert doc["forward_passes_measured"] == {"mojo": True, "llama": False}


def test_a_measured_engine_arm_is_averaged(evalroot: Path) -> None:
    arm(evalroot, "mojo", ["--steps", "4"], [4, 4, 4])
    arm(evalroot, "vllm", {"engine": "vllm", "forward_passes_measured": True}, [6, None, 8])
    doc = compare(evalroot, "mojo", "vllm")
    assert doc["forward_passes_mean"] == {"mojo": 4.0, "vllm": 7.0} and doc["forward_passes_rows"]["vllm"] == 2


def test_two_unmeasured_arms_have_no_means(evalroot: Path) -> None:
    arm(evalroot, "a", {"forward_passes_measured": False}, [None, None])
    arm(evalroot, "b", {"forward_passes_measured": False}, [None, None])
    doc = compare(evalroot, "a", "b")
    assert doc["forward_passes_mean"] == {"a": None, "b": None}


def test_flag_key_order_does_not_make_an_arm_look_mixed(evalroot: Path) -> None:
    arm(evalroot, "a", {"x": 1, "y": 2}, [1, 2, 3, 4], key_order=True)
    arm(evalroot, "b", {"x": 1, "y": 2}, [1, 2, 3, 4])
    assert compare(evalroot, "a", "b")["problems"] == 4


def test_an_arm_with_two_configurations_is_still_refused(evalroot: Path) -> None:
    arm(evalroot, "a", {"x": 1}, [1, 2])
    rows = [json.loads(l) for l in (evalroot / "bench/eval/runs/a.jsonl").read_text().splitlines()]
    rows[1]["flags"] = {"x": 2}
    (evalroot / "bench/eval/runs/a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    arm(evalroot, "b", {"x": 1}, [1, 2])
    with pytest.raises(SystemExit) as e:
        eval_gsm8k.cmd_compare("a", "b")
    assert "mixes configurations" in str(e.value)
