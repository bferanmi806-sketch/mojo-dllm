"""bench/parity/*.json: the writer and the reader share one schema, and its invariants are checked (#242)."""

from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import parity_schema as ps  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parity = load_module("parity_schema_writer", ROOT / "bench/parity.py")
tables = load_module("bench_table_schema", ROOT / "scripts/bench_table.py")
FILES = sorted((ROOT / "bench/parity").glob("*.json"))
FULL = ROOT / "bench/parity/2026-10-30-llada-LLaDA-1.5.Q4_K_M.json"


def doc() -> dict:
    return copy.deepcopy(json.loads(FULL.read_text()))


def problems(d: dict) -> str:
    return "; ".join(ps.parity_problems(d))


# the committed evidence


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_committed_file_validates(path: Path) -> None:
    assert ps.parity_problems(json.loads(path.read_text())) == []


def test_there_are_committed_files_to_check() -> None:
    assert len(FILES) >= 7


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_a_file_name_agrees_with_the_document(path: Path) -> None:
    """A name is the one `result_path` makes from the document's date, arch, device and model, or one of the two
    legacy forms from before names carried the model (DATE-ARCH.json and DATE-ARCH-gpu.json)."""
    d = json.loads(path.read_text())
    day = dt.datetime.fromisoformat(d["date"]).date()
    arch = d.get("arch", "llada")
    legacy = {f"{day}-{arch}.json", f"{day}-{arch}-gpu.json"}
    want = parity.result_path(d["model"], arch, d.get("device", "cpu"), day).name
    assert path.name in legacy | {want}, (path.name, want)
    if path.name in legacy:
        assert path.name.endswith("-gpu.json") == (d.get("device", "cpu") == "gpu")


def test_a_rerun_without_the_hand_written_fields_is_still_a_valid_document() -> None:
    d = doc()
    for k in ("machine", "measurement_scope", "metadata_audit", "model_source", "reference_build", "smoke"):
        del d[k]
    for g in d["generation"]:
        del g["mojo_generated_tokens"], g["reference_generated_tokens"]
    assert ps.parity_problems(d) == []


# each invariant, one by one


def test_matching_tokens_above_gen_length_is_refused() -> None:
    d = doc()
    g = d["generation"][0]
    g["matching_tokens"] = g["gen_length"] + 1
    assert "outside 0..gen_length" in problems(d)
    del g["mojo_generated_tokens"], g["reference_generated_tokens"]      # without the lists to contradict it
    assert "outside 0..gen_length" in problems(d)
    g["matching_tokens"] = -1
    assert "outside 0..gen_length" in problems(d)


def test_a_masked_median_below_the_minimum_is_refused() -> None:
    d = doc()
    d["logits"][0]["min_cosine"] = 0.9
    d["logits"][0]["median_cosine"] = 0.95
    d["logits"][0]["masked_median_cosine"] = 0.8
    assert "above a median" in problems(d)


def test_a_first_divergence_with_all_tokens_matching_is_refused() -> None:
    d = doc()
    g = d["generation"][0]
    g["matching_tokens"], g["first_divergence"] = g["gen_length"], 3
    assert "although all" in problems(d)


def test_no_first_divergence_with_a_mismatch_is_refused() -> None:
    d = doc()
    g = d["generation"][0]
    g["first_divergence"] = None
    g["matching_tokens"] = g["gen_length"] - 1
    assert "first_divergence" in problems(d)


def test_a_first_divergence_past_the_matching_count_is_refused() -> None:
    d = doc()
    g = d["generation"][0]
    g["matching_tokens"], g["first_divergence"] = 2, 9
    assert "first_divergence 9" in problems(d)


def test_token_lists_that_disagree_with_the_counts_are_refused() -> None:
    d = doc()
    g = d["generation"][0]
    g["matching_tokens"] -= 1
    assert "token lists agree" in problems(d)
    d = doc()
    d["generation"][0]["first_divergence"] += 1
    assert "first differ" in problems(d)


def test_only_one_token_list_is_refused() -> None:
    d = doc()
    del d["generation"][0]["reference_generated_tokens"]
    assert "only one of" in problems(d)


def test_argmax_agree_above_rows_is_refused() -> None:
    d = doc()
    d["logits"][0]["argmax_agree"] = d["logits"][0]["rows"] + 1
    assert "argmax_agree" in problems(d)


def test_a_minimum_cosine_above_a_median_is_refused() -> None:
    d = doc()
    d["logits"][0]["min_cosine"] = 1.0
    d["logits"][0]["median_cosine"] = 0.5
    assert "min_cosine" in problems(d)


def test_a_cosine_outside_minus_one_to_one_is_refused() -> None:
    d = doc()
    d["logits"][0]["median_cosine"] = 1.5
    assert "not a cosine" in problems(d)


def test_top5_overlap_outside_zero_to_five_is_refused() -> None:
    d = doc()
    d["logits"][0]["mean_top5_overlap"] = 5.5
    assert "mean_top5_overlap" in problems(d)


def test_a_max_below_the_median_is_refused() -> None:
    d = doc()
    d["logits"][0]["max_rel_rms"] = 0.0
    assert "max_rel_rms" in problems(d)


def test_worst_rows_outside_the_rows_are_refused() -> None:
    d = doc()
    d["logits"][0]["worst_rows"] = [d["logits"][0]["rows"]]
    assert "worst_rows" in problems(d)


def test_a_nan_metric_is_refused() -> None:
    d = doc()
    d["logits"][0]["median_cosine"] = float("nan")
    assert "finite" in problems(d)


def test_a_missing_key_is_named() -> None:
    d = doc()
    del d["versions"]
    assert "versions" in problems(d)
    d = doc()
    del d["logits"][0]["rows"]
    assert "rows" in problems(d)


def test_an_unknown_top_level_key_is_refused() -> None:
    d = doc()
    d["mojo_generated_token"] = 1
    assert "unknown keys" in problems(d)


def test_a_bad_date_arch_or_device_is_refused() -> None:
    for key, value, word in (("date", "yesterday", "ISO"), ("arch", "llama", "arch"), ("device", "tpu", "device")):
        d = doc()
        d[key] = value
        assert word in problems(d), key


def test_a_document_with_a_second_reference_needs_its_comparisons() -> None:
    d = json.loads((ROOT / "bench/parity/2026-10-09-dream-gpu.json").read_text())
    assert ps.parity_problems(d) == []
    del d["logits"][0]["vs_ref2"]
    assert "vs_ref2" in problems(d)


def test_empty_logits_are_refused() -> None:
    d = doc()
    d["logits"] = []
    assert "logits is empty" in problems(d)


def test_the_message_lists_every_problem_at_once() -> None:
    d = doc()
    d["logits"][0]["argmax_agree"] = -1
    d["generation"][0]["matching_tokens"] = 999
    with pytest.raises(ps.ParityError) as e:
        ps.validate_parity(d)
    assert "argmax_agree" in str(e.value) and "matching_tokens" in str(e.value)


# the bound and the metrics of the DiffusionGemma comparison


GOOD = dict(min_cosine=0.999, argmax_agree=64, rows=64, mean_top5_overlap=4.9, max_rel_rms=0.05, top1_max_rel_err=0.01)
BOUND = dict(min_cosine=0.99, argmax_fraction=0.9, mean_top5_overlap=4.5, max_rel_rms=0.1, max_top1_rel_err=0.1)


def test_a_misspelled_metric_key_is_named_not_a_keyerror() -> None:
    bad = {("top1_max_rel_error" if k == "top1_max_rel_err" else k): v for k, v in GOOD.items()}
    with pytest.raises(ps.ParityError) as e:
        parity.dg_violations(bad, BOUND)  # type: ignore[arg-type]
    assert "top1_max_rel_err" in str(e.value) and "the metrics" in str(e.value)


def test_a_misspelled_bound_key_is_named() -> None:
    bad = {("max_top1_rel_error" if k == "max_top1_rel_err" else k): v for k, v in BOUND.items()}
    with pytest.raises(ps.ParityError) as e:
        parity.dg_violations(GOOD, bad)  # type: ignore[arg-type]
    assert "max_top1_rel_err" in str(e.value) and "the bound" in str(e.value)


def test_a_good_pair_still_gives_no_violation() -> None:
    assert parity.dg_violations(GOOD, BOUND) == []  # type: ignore[arg-type]


# the writer and the reader


def test_the_writer_refuses_an_invalid_document_and_writes_nothing(tmp_path: Path) -> None:
    d = doc()
    d["generation"][0]["matching_tokens"] = d["generation"][0]["gen_length"] + 1
    out = tmp_path / "x.json"
    with pytest.raises(SystemExit) as e:
        parity.write_doc(out, d, force=False)
    assert "refusing to write" in str(e.value) and "matching_tokens" in str(e.value) and not out.exists()


def test_the_writer_writes_a_valid_document(tmp_path: Path) -> None:
    out = tmp_path / "x.json"
    parity.write_doc(out, doc(), force=False)
    assert json.loads(out.read_text()) == doc()


def test_the_table_refuses_an_invalid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "bench/parity"
    folder.mkdir(parents=True)
    d = doc()
    d["logits"][0]["argmax_agree"] = d["logits"][0]["rows"] + 1
    (folder / "2026-10-30-bad.json").write_text(json.dumps(d))
    monkeypatch.setattr(tables, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as e:
        tables.parity()
    assert "2026-10-30-bad.json" in str(e.value) and "argmax_agree" in str(e.value)


def test_the_machine_the_run_can_read_has_the_four_fields_of_the_hand_written_entry() -> None:
    m = parity.machine_info()
    assert set(m) == {"os", "cpu", "logical_cpus", "memory_bytes"}
    assert isinstance(m["logical_cpus"], int) and m["memory_bytes"] > 0 and isinstance(m["os"], str)


def test_a_finished_run_writes_the_fields_the_committed_files_carry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() with every runtime stubbed: the document it builds has threads, the machine and both token lists."""
    from types import ModuleType

    monkeypatch.setattr(parity, "ROOT", tmp_path)
    monkeypatch.setattr(parity, "LOGIT_CASES", [])
    monkeypatch.setattr(parity, "GEN_CASES", [parity.GenCase("p", 4, 4, 4)])
    monkeypatch.setattr(parity, "tokenize", lambda model, prompt: [1, 2])
    mine = {"text": "t", "tokens": [1, 2, 7, 8, 9, 10]}

    def fake_run(cmd: list[str]) -> str:
        if "run" in cmd:
            return json.dumps(mine) + "\n"
        if "generate" in cmd:
            return "1,2,7,8,9,11\n"
        return "abc1234"

    monkeypatch.setattr(parity, "run", fake_run)
    seen: list[dict] = []
    monkeypatch.setattr(parity, "validate_parity", lambda d: seen.append(d))
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    monkeypatch.setattr(sys, "argv", ["parity.py", "--model", "m.gguf", "--threads", "6"])
    assert parity.main() == 0
    (written,) = seen
    assert written["threads"] == 6 and set(written["machine"]) == {"os", "cpu", "logical_cpus", "memory_bytes"}
    g = written["generation"][0]
    assert g["mojo_generated_tokens"] == [7, 8, 9, 10] and g["reference_generated_tokens"] == [7, 8, 9, 11]
    assert g["matching_tokens"] == 3 and g["first_divergence"] == 3
    g.pop("mojo_text")
    assert ps.generation_problems(dict(g, mojo_text="t"), "g") == [], "the stubbed run's own record is consistent"


@pytest.mark.parametrize("arch,passes", [("dream", False), ("llada", True)])
def test_block_length_is_passed_to_mojo_only_for_blocked_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arch: str, passes: bool
) -> None:
    from types import ModuleType

    monkeypatch.setattr(parity, "ROOT", tmp_path)
    monkeypatch.setattr(parity, "LOGIT_CASES", [])
    monkeypatch.setattr(parity, "GEN_CASES", [parity.GenCase("p", 2, 2, 2)])
    monkeypatch.setattr(parity, "tokenize", lambda model, prompt: [1])
    cmds: list[list[str]] = []

    def fake_run(cmd: list[str]) -> str:
        cmds.append(cmd)
        if "run" in cmd:
            return json.dumps({"text": "t", "tokens": [1, 5, 6]}) + "\n"
        return "1,5,6\n"

    monkeypatch.setattr(parity, "run", fake_run)
    monkeypatch.setattr(parity, "validate_parity", lambda d: None)
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    monkeypatch.setattr(sys, "argv", ["parity.py", "--model", "m.gguf", "--arch", arch])
    assert parity.main() == 0
    (mojo,) = [c for c in cmds if "run" in c]
    assert ("--block-length" in mojo) is passes
