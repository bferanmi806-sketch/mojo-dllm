"""Result files are written atomically and never replace a dated file by accident (#239)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import atomic_write  # noqa: E402
import eval_gsm8k  # noqa: E402
import run_bench  # noqa: E402
from test_run_bench_cleanup import bench, live  # noqa: E402,F401  (fixtures)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parity = load_module("parity_writes", ROOT / "bench/parity.py")
quality = load_module("quality_writes", ROOT / "bench/quality.py")

OLD = '{"old": "complete result"}\n'


def existing(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OLD)
    return path.read_bytes()


def must_not_run(*a: object, **k: object) -> None:
    raise AssertionError("work started although the output file exists")


# the helper


def test_a_second_write_to_the_same_name_is_refused_and_leaves_the_first_byte_identical(tmp_path: Path) -> None:
    target = tmp_path / "2026-11-09-cpu.json"
    atomic_write.write_new(target, OLD)
    before = target.read_bytes()
    with pytest.raises(SystemExit) as e:
        atomic_write.write_new(target, '{"new": 1}\n')
    assert str(target) in str(e.value) and "--force" in str(e.value)
    assert target.read_bytes() == before


def test_force_replaces_the_file(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    atomic_write.write_new(target, OLD)
    atomic_write.write_new(target, '{"new": 1}\n', force=True)
    assert json.loads(target.read_text()) == {"new": 1}


def test_a_write_that_raises_leaves_neither_a_partial_file_nor_a_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    target = out / "x.json"
    atomic_write.write_new(target, OLD)
    before = target.read_bytes()

    def boom(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(atomic_write.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write.write_atomic(target, "x" * 10000)
    assert target.read_bytes() == before
    assert sorted(p.name for p in out.iterdir()) == ["x.json"], "no temporary file is left behind"


def test_a_new_file_that_fails_to_write_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atomic_write.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("io error")))
    out = tmp_path / "out"
    with pytest.raises(OSError):
        atomic_write.write_atomic(out / "new.json", "{}")
    assert list(out.iterdir()) == []


def test_the_target_is_only_ever_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Between the temporary write and the rename the target still holds the old, complete text."""
    target = tmp_path / "x.json"
    atomic_write.write_new(target, OLD)
    seen: list[str] = []
    real = os.replace

    def spy(src: str, dst: str) -> None:
        seen.append(Path(dst).read_text())
        real(src, dst)

    monkeypatch.setattr(atomic_write.os, "replace", spy)
    atomic_write.write_atomic(target, '{"new": 2}\n')
    assert seen == [OLD] and target.read_text() == '{"new": 2}\n'


# the five call sites


def test_eval_compare_force_replaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eval_gsm8k, "ROOT", tmp_path)
    monkeypatch.setattr(eval_gsm8k, "RUNS", tmp_path / "bench/eval/runs")
    (tmp_path / "bench/eval/runs").mkdir(parents=True)
    for name, correct in (("a", [True, False, True]), ("b", [True, True, False])):
        rows = [dict(index=i, correct=c, forward_passes=1, generate_s=1.0, flags=["--x"], version="v", model="m")
                for i, c in enumerate(correct)]
        (tmp_path / "bench/eval/runs" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert eval_gsm8k.cmd_compare("a", "b") == 0
    (path,) = (tmp_path / "bench/eval").glob("*-a-vs-b.json")
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        eval_gsm8k.cmd_compare("a", "b")
    assert path.read_bytes() == before
    assert eval_gsm8k.cmd_compare("a", "b", force=True) == 0


def test_quality_refuses_before_it_runs_anything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quality, "ROOT", tmp_path)
    out = tmp_path / "bench/quality" / f"{dt.date.today().isoformat()}-m.json"
    before = existing(out)
    monkeypatch.setattr(quality.subprocess, "run", must_not_run)
    monkeypatch.setattr(quality.power, "run_on_mains", must_not_run)
    monkeypatch.setattr(sys, "argv", ["quality.py", "--model", "m.gguf"])
    with pytest.raises(SystemExit) as e:
        quality.main()
    assert "exists" in str(e.value) and out.read_bytes() == before


def test_parity_refuses_before_it_runs_a_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parity, "ROOT", tmp_path)
    out = parity.result_path("m.gguf", "llada", "cpu", dt.date.today())
    before = existing(out)
    monkeypatch.setattr(parity, "run", must_not_run)
    monkeypatch.setattr(sys, "argv", ["parity.py", "--model", "m.gguf"])
    with pytest.raises(SystemExit) as e:
        parity.main()
    assert "exists" in str(e.value) and out.read_bytes() == before


def test_the_diffusion_gemma_compare_refuses_before_comparing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parity, "ROOT", tmp_path)
    out = parity.result_path("m.gguf", parity.DG_ARCH, "cpu", dt.date.today())
    before = existing(out)
    monkeypatch.setattr(parity, "dg_compare", must_not_run)
    monkeypatch.setattr(sys, "argv", ["parity.py", "--arch", "diffusion-gemma", "--phase", "compare", "--model", "m.gguf"])
    with pytest.raises(SystemExit) as e:
        parity.main()
    assert "exists" in str(e.value) and out.read_bytes() == before


def test_run_bench_refuses_before_it_starts_and_keeps_yesterdays_file(
    bench: Path, live: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = existing(bench)
    monkeypatch.setattr(run_bench.isolation, "begin_arm", must_not_run)
    with pytest.raises(SystemExit) as e:
        run_bench.main()
    assert "exists" in str(e.value) and bench.read_bytes() == before


def test_run_bench_force_replaces_and_writes_only_whole_files(
    bench: Path, live: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing(bench)
    monkeypatch.setattr(sys, "argv", sys.argv + ["--force"])
    seen: list[str] = []
    real = atomic_write.write_atomic

    def spy(path: Path, text: str) -> None:
        json.loads(text)          # every write the run makes is a whole document
        seen.append(text)
        real(path, text)

    monkeypatch.setattr(run_bench.atomic_write, "write_atomic", spy)
    assert run_bench.main() == 0
    assert seen and json.loads(bench.read_text())["partial"] is False


def test_a_finished_parity_run_writes_through_the_atomic_helper_and_a_second_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import ModuleType

    monkeypatch.setattr(parity, "ROOT", tmp_path)
    monkeypatch.setattr(parity, "LOGIT_CASES", [])
    monkeypatch.setattr(parity, "GEN_CASES", [])
    monkeypatch.setattr(parity, "run", lambda cmd: "head")
    monkeypatch.setattr(parity, "validate_parity", lambda doc: None)   # no inference runs, so no logits; see test_parity_schema
    monkeypatch.setitem(sys.modules, "numpy", ModuleType("numpy"))
    written: list[Path] = []
    real = atomic_write.write_atomic
    monkeypatch.setattr(parity.atomic_write, "write_atomic", lambda p, t: (written.append(p), real(p, t))[1])
    monkeypatch.setattr(sys, "argv", ["parity.py", "--arch", "dream", "--model", "m.gguf"])
    assert parity.main() == 0
    assert len(written) == 1 and written[0].exists()
    before = written[0].read_bytes()
    with pytest.raises(SystemExit):
        parity.main()
    assert written[0].read_bytes() == before


def test_a_diffusion_gemma_result_takes_the_build_from_the_dumps_and_is_written_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "ROOT", tmp_path)
    case = tmp_path / "build/dg-parity/m/case0/case.json"
    case.parent.mkdir(parents=True)
    case.write_text(json.dumps({"model": "m.gguf", "model_sha256": "s" * 64, "llama_cpp_commit": "l" * 40}))
    entry = dict(prompt="p", seq_len=300, reference_spread={}, mojo_dllm_commit="dump1234", variant="cached", rows=64,
                 min_cosine=0.999, median_cosine=0.9999, masked_median_cosine=0.9999, max_rel_rms=0.05,
                 median_rel_rms=0.01, argmax_agree=64, mean_top5_overlap=4.9, worst_rows=[1, 2, 3],
                 top1_max_rel_err=0.01)
    monkeypatch.setattr(parity, "dg_case_files", lambda workdir: [case])
    monkeypatch.setattr(parity, "llada_bound", lambda root: dict(
        min_cosine=0.99, argmax_fraction=0.9, mean_top5_overlap=4.5, max_rel_rms=0.1, max_top1_rel_err=0.1, source="x"))
    monkeypatch.setattr(parity, "dg_compare", lambda workdir, bound: ([entry], []))
    written: list[Path] = []
    real = atomic_write.write_atomic
    monkeypatch.setattr(parity.atomic_write, "write_atomic", lambda p, t: (written.append(p), real(p, t))[1])
    monkeypatch.setattr(sys, "argv", ["parity.py", "--arch", "diffusion-gemma", "--phase", "compare", "--model", "m.gguf"])
    assert parity.main() == 0
    (out,) = written
    doc = json.loads(out.read_text())
    assert doc["versions"]["mojo-dllm"] == "dump1234", "the build that wrote the dumps, not git HEAD at compare time"
    assert doc["logits"] == [entry]
