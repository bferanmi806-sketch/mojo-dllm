"""bench/check_dg_bound.py: `check` rejects what it must and fails when the bound cannot separate good from bad, and
`make-mutant` changes exactly one layer's scale and refuses everything else (#251, #165).

Needs numpy and gguf, which are in the `fixtures` group: the module is skipped, with that reason shown, when either is
missing, and CI runs it in the step that installs the extra groups. The cases are tiny random logits, not a model.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy is in the `fixtures` group, not the `test` group; CI runs this file there")
gguf = pytest.importorskip("gguf", reason="gguf is in the `fixtures` group, not the `test` group; CI runs this file there")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))


def load():
    spec = importlib.util.spec_from_file_location("check_dg_bound_t", ROOT / "bench/check_dg_bound.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_dg_bound_t"] = module
    spec.loader.exec_module(module)
    return module


cdb = load()
parity = cdb.parity
VOCAB, ROWS, CAP = 64, 6, 30.0


def capped_logits(seed: int, scale: float) -> "np.ndarray":
    z = np.random.default_rng(seed).normal(0.0, scale, (ROWS, VOCAB))
    return (CAP * np.tanh(z / CAP)).astype(np.float32)


def make_cases(folder: Path, scale: float = 25.0, seed: int = 1) -> Path:
    folder.mkdir()
    for i, case in enumerate(parity.LOGIT_CASES):
        d = folder / f"case{i}"
        d.mkdir()
        (d / "case.json").write_text(json.dumps(dict(prompt=case.prompt, n_vocab=VOCAB, final_logit_softcapping=CAP,
                                                     model="m.gguf", model_sha256="0" * 64, llama_cpp_commit="c" * 40)))
        capped_logits(seed + i, scale).tofile(d / "unified.ref.f32")
    return folder


def run_check(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    cdb_argv = ["check_dg_bound", *argv]
    old = sys.argv
    sys.argv = cdb_argv
    try:
        try:
            rc = cdb.main()
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
            if isinstance(e.code, str):
                print(e.code, file=sys.stderr)
    finally:
        sys.argv = old
    out = capsys.readouterr()
    return rc, out.out, out.err


# check ------------------------------------------------------------------------------------------------------

def test_the_bound_accepts_the_control_and_rejects_a_removed_soft_cap(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = make_cases(tmp_path / "cases")
    out_file = tmp_path / "evidence.json"
    rc, out, err = run_check(["check", str(cases), "--out", str(out_file)], capsys)
    assert rc == 0, err
    doc = json.loads(out_file.read_text())
    assert doc["all_as_expected"] is True and doc["dropped_layer"] is None and doc["dropped_model"] is None
    by = {(r["case"], r["mutant"]): r for r in doc["results"]}
    assert len(by) == 4
    for case in ("case0", "case1"):
        assert by[(case, "control")]["rejected"] is False and by[(case, "control")]["violations"] == []
        assert by[(case, "soft-cap removed")]["rejected"] is True and by[(case, "soft-cap removed")]["violations"]
    assert "rejected=False" in out and "rejected=True" in out


def test_a_bound_that_lets_the_soft_cap_bug_through_fails_the_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = make_cases(tmp_path / "cases", scale=0.5)         # logits far below the cap: removing it changes nothing
    rc, out, err = run_check(["check", str(cases)], capsys)
    assert rc == 1 and "the bound does not separate good from bad" in err and "soft-cap removed: accepted" in err


def test_a_bound_that_rejects_perfect_agreement_fails_the_check(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    cases = make_cases(tmp_path / "cases")
    real = parity.llada_bound(ROOT)
    monkeypatch.setattr(parity, "llada_bound", lambda root: {**real, "min_cosine": 2.0})
    rc, out, err = run_check(["check", str(cases)], capsys)
    assert rc == 1 and "control: rejected, expected accepted" in err


def test_dropped_scale_logits_that_differ_are_rejected_as_expected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = make_cases(tmp_path / "cases")
    dropped = make_cases(tmp_path / "dropped", seed=99)       # other logits: far from the reference
    out_file = tmp_path / "e.json"
    rc, out, err = run_check(["check", str(cases), "--dropped", str(dropped), "--dropped-layer", "3", "--out", str(out_file)], capsys)
    assert rc == 0, err
    doc = json.loads(out_file.read_text())
    assert doc["dropped_layer"] == 3 and doc["dropped_model"] == {"model": "m.gguf", "model_sha256": "0" * 64}
    assert {r["mutant"] for r in doc["results"]} == {"control", "soft-cap removed", "layer 3 output scale dropped"}


def test_dropped_scale_logits_equal_to_the_reference_are_a_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = make_cases(tmp_path / "cases")
    dropped = make_cases(tmp_path / "dropped")                # the same logits: the "bug" changed nothing
    rc, out, err = run_check(["check", str(cases), "--dropped", str(dropped), "--dropped-layer", "3"], capsys)
    assert rc == 1 and "layer 3 output scale dropped: accepted" in err


def test_dropped_and_its_layer_go_together(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = make_cases(tmp_path / "cases")
    rc, out, err = run_check(["check", str(cases), "--dropped", str(cases)], capsys)
    assert rc == 2 and "go together" in err
    rc, out, err = run_check(["check", str(cases), "--dropped-layer", "3"], capsys)
    assert rc == 2


def test_a_dropped_folder_with_other_cases_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = make_cases(tmp_path / "cases")
    dropped = make_cases(tmp_path / "dropped")
    (dropped / "case1" / "case.json").unlink()
    rc, out, err = run_check(["check", str(cases), "--dropped", str(dropped), "--dropped-layer", "3"], capsys)
    assert rc != 0 and "refusing" in err


def test_a_missing_or_empty_reference_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = make_cases(tmp_path / "cases")
    (cases / "case0" / "unified.ref.f32").write_bytes(b"")
    rc, out, err = run_check(["check", str(cases)], capsys)
    assert rc != 0 and "is empty" in err
    (cases / "case0" / "unified.ref.f32").unlink()
    rc, out, err = run_check(["check", str(cases)], capsys)
    assert rc != 0 and "unified.ref.f32 is missing" in err


def test_a_folder_with_no_cases_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "empty").mkdir()
    rc, out, err = run_check(["check", str(tmp_path / "empty")], capsys)
    assert rc != 0 and "refusing" in err


# make-mutant ------------------------------------------------------------------------------------------------

def write_model(path: Path, scales: dict[str, "np.ndarray"]) -> None:
    w = gguf.GGUFWriter(str(path), "gemma4")
    for name, arr in scales.items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "model.gguf"
    write_model(path, {
        "blk.0.layer_output_scale.weight": np.array([2.5], dtype=np.float32),
        "blk.1.layer_output_scale.weight": np.array([1.0], dtype=np.float32),
        "blk.2.layer_output_scale.weight": np.array([1.5, 2.5], dtype=np.float32),
        "blk.3.layer_output_scale.weight": np.array([2.5], dtype=np.float16),
        "other.weight": np.arange(8, dtype=np.float32),
    })
    return path


def mutate(model: Path, layer: int, out: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    rc, o, e = run_check(["make-mutant", str(model), "--layer", str(layer), "--out", str(out)], capsys)
    return rc, o + e


def test_make_mutant_changes_exactly_the_four_bytes_of_the_chosen_scale(model: Path, tmp_path: Path,
                                                                     capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "mutant.gguf"
    rc, text = mutate(model, 0, out, capsys)
    assert rc == 0 and "blk.0.layer_output_scale.weight: 2.5 -> 1.0" in text
    before, after = model.read_bytes(), out.read_bytes()
    assert len(before) == len(after)
    diff = [i for i in range(len(before)) if before[i] != after[i]]
    offset = int(next(t for t in gguf.GGUFReader(str(model)).tensors if t.name == "blk.0.layer_output_scale.weight").data_offset)
    assert diff and offset <= diff[0] and diff[-1] < offset + 4
    assert struct.unpack("<f", before[offset:offset + 4])[0] == 2.5 and struct.unpack("<f", after[offset:offset + 4])[0] == 1.0
    assert after[:offset] == before[:offset] and after[offset + 4:] == before[offset + 4:]
    reader = {t.name: float(t.data[0]) for t in gguf.GGUFReader(str(out)).tensors if t.name.endswith("scale.weight")}
    assert reader["blk.0.layer_output_scale.weight"] == 1.0 and reader["blk.1.layer_output_scale.weight"] == 1.0


def test_make_mutant_never_writes_the_original(model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    original = model.read_bytes()
    mutate(model, 0, tmp_path / "m.gguf", capsys)
    assert model.read_bytes() == original


def test_make_mutant_refuses_an_existing_output(model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "taken.gguf"
    out.write_bytes(b"keep")
    rc, text = mutate(model, 0, out, capsys)
    assert rc != 0 and "exists" in text and out.read_bytes() == b"keep"


def test_make_mutant_refuses_a_layer_with_no_such_tensor(model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, text = mutate(model, 9, tmp_path / "m.gguf", capsys)
    assert rc != 0 and "has 0 tensors named blk.9.layer_output_scale.weight" in text and not (tmp_path / "m.gguf").exists()


def test_make_mutant_refuses_a_scale_that_is_not_a_single_f32(model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for layer, what in ((2, "2 x F32"), (3, "1 x F16")):
        rc, text = mutate(model, layer, tmp_path / f"m{layer}.gguf", capsys)
        assert rc != 0 and f"is not a single f32 ({what})" in text and not (tmp_path / f"m{layer}.gguf").exists()


def test_make_mutant_refuses_a_scale_that_is_already_one(model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc, text = mutate(model, 1, tmp_path / "m.gguf", capsys)
    assert rc != 0 and "already 1.0" in text and not (tmp_path / "m.gguf").exists()
