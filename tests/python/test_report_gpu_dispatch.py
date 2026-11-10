"""bench/report_gpu_dispatch.py derives what it prints and refuses what it cannot derive (#253)."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "bench/report_gpu_dispatch.py"


def load():
    spec = importlib.util.spec_from_file_location("report_gpu_dispatch_t", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["report_gpu_dispatch_t"] = module
    spec.loader.exec_module(module)
    return module


rg = load()
GPU = "NVIDIA RTX 6000D, GPU-{u}, 00000000:A1:00.0, 81920 MiB, 595.104.02, 600.00 W"


def timing(uuid: str, speedups: dict[int, list[float]], seq: dict[int, list[int]] | None = None) -> dict:
    runs, summary = [], []
    for length, ratios in speedups.items():
        for rep in (-1, 0, 1):
            for pi in range(len(ratios)):
                for variant in ("base", "new"):
                    runs.append(dict(ok=True, forward_passes=16, variant=variant, gen_length=length, rep=rep, prompt_index=pi,
                                     seq_len=(seq or {}).get(length, [length + 20] * len(ratios))[pi]))
        ratios_sorted = sorted(ratios)
        summary.append(dict(gen_length=length, median_ms_per_step=dict(base=20.0, new=20.0 / ratios_sorted[len(ratios) // 2]),
                            median_paired_speedup=ratios_sorted[len(ratios) // 2], paired_speedups=ratios))
    return dict(partial=False, method="run_bench.run_mojo; interleaved AB/BA, no early stop, no threshold or dual cache, "
                                      "no profiler synchronization; first pair per length is warmup",
                machine=dict(gpu=GPU.format(u=uuid)), steps=16, block_length=32, prompts=["a", "b"], runs=runs, summary=summary)


def exact(arch: str) -> dict:
    return dict(arch=arch, mask_id=7, logits=[dict(values=10, bytes_equal=True)] * 2,
                generation=[dict(tokens_equal=True)] * 3)


ENV = dict(sm_count=156, mojo="1.1.0 (abc)", max_core="26.6.0", compiled_for="sm_120", host_note="HOST NOTE.",
           source_note="SOURCE NOTE.")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    dest = root / "bench/results/6000d-dispatch"
    dest.mkdir(parents=True)
    (root / "bench/checks").mkdir()
    (dest / "llada.json").write_text(json.dumps(timing("aaa", {128: [1.18, 1.19, 1.2], 256: [1.0, 1.001, 0.999]},
                                                       {128: [148, 154, 154], 256: [276, 282, 282]})))
    (dest / "dream.json").write_text(json.dumps(timing("bbb", {128: [1.03, 1.2, 1.25], 256: [1.06, 1.06, 1.07]},
                                                       {128: [155, 160, 161], 256: [276, 282, 282]})))
    (dest / "environment.json").write_text(json.dumps(ENV))
    for arch in ("llada", "dream"):
        (root / f"bench/checks/2026-10-30-6000d-dispatch-{arch}-exact.json").write_text(json.dumps(exact(arch)))
    return root


def edit(path: Path, fn) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def dest(tree: Path) -> Path:
    return tree / "bench/results/6000d-dispatch"


def test_the_committed_readme_is_what_the_generator_prints_and_has_no_en_dash() -> None:
    text = rg.render(ROOT)
    assert (ROOT / "bench/results/6000d-dispatch/README.md").read_text(encoding="utf-8") == text
    assert "–" not in text and "—" not in text


def test_the_prose_is_computed_from_the_files(tree: Path) -> None:
    text = rg.render(tree)
    assert "Two RTX 6000D GPUs, 156 SMs each, 81,920 MiB, driver 595.104.02; Mojo 1.1.0 (abc), max-core 26.6.0, compiled for sm_120." in text
    assert "HOST NOTE." in text and "SOURCE NOTE." in text
    assert "Two prompts, two measured rounds per length" in text
    assert "16 forward passes, block length 32" in text
    assert "| llada | 128 | 20.000 | 16.807 | 1.190x | 1.180 to 1.200x |" in text
    assert "LLaDA at 256 generated tokens: effectively unchanged" in text
    assert "The widest pair range is Dream at 128 generated tokens, 1.030 to 1.250x. The prompts give canvases of 155, 160, 161 tokens" in text
    assert "- llada: 2 logit canvases, 20 finite float values checked byte for byte; 3 complete generated token lists equal. Mask ID 7." in text


def test_the_unchanged_sentence_follows_the_ratios(tree: Path) -> None:
    edit(dest(tree) / "llada.json", lambda d: d["summary"][1].update(median_paired_speedup=1.02))
    assert "LLaDA at" not in rg.render(tree).split("## Preservation")[0].split("Speedup is")[1]
    edit(dest(tree) / "dream.json", lambda d: d["summary"][1].update(median_paired_speedup=0.998))
    assert "Dream at 256 generated tokens: effectively unchanged" in rg.render(tree)


def test_a_wide_range_is_explained_only_by_prompt_lengths_that_differ(tree: Path) -> None:
    edit(dest(tree) / "dream.json", lambda d: [r.update(seq_len=150) for r in d["runs"]])
    text = rg.render(tree)
    assert "one canvas length there, so the spread is not from prompt length" in text and "canvases of" not in text


def test_the_card_count_is_the_number_of_distinct_uuids(tree: Path) -> None:
    edit(dest(tree) / "dream.json", lambda d: d["machine"].update(gpu=GPU.format(u="aaa")))
    assert "One RTX 6000D GPUs" in rg.render(tree)


def test_different_cards_or_drivers_are_refused(tree: Path) -> None:
    edit(dest(tree) / "dream.json", lambda d: d["machine"].update(gpu=GPU.format(u="bbb").replace("595.104.02", "600.1")))
    with pytest.raises(SystemExit, match="different cards or drivers"):
        rg.render(tree)


@pytest.mark.parametrize("gpu", ["", "NVIDIA RTX, notauuid, bus, 100 MiB, 1.0", "NVIDIA RTX, GPU-1"])
def test_a_missing_or_malformed_card_line_is_refused(tree: Path, gpu: str) -> None:
    edit(dest(tree) / "llada.json", lambda d: d["machine"].update(gpu=gpu))
    with pytest.raises(SystemExit, match="nvidia-smi line"):
        rg.render(tree)


@pytest.mark.parametrize("key", rg.ENVIRONMENT_KEYS)
def test_an_environment_value_that_is_recorded_nowhere_is_refused(tree: Path, key: str) -> None:
    edit(dest(tree) / "environment.json", lambda d: d.pop(key))
    with pytest.raises(SystemExit) as e:
        rg.render(tree)
    assert key in str(e.value) and "not recorded" in str(e.value)


def test_no_environment_file_at_all_is_refused(tree: Path) -> None:
    (dest(tree) / "environment.json").unlink()
    with pytest.raises(SystemExit, match="not recorded"):
        rg.render(tree)


def test_a_value_in_the_timing_file_wins_over_the_side_file(tree: Path) -> None:
    edit(dest(tree) / "llada.json", lambda d: d.update(environment=dict(sm_count=132)))
    assert "132 SMs each" in rg.render(tree)


def test_empty_runs_or_summary_are_refused_not_passed_vacuously(tree: Path) -> None:
    edit(dest(tree) / "llada.json", lambda d: d.update(runs=[]))
    with pytest.raises(SystemExit, match="no runs"):
        rg.render(tree)
    edit(dest(tree) / "llada.json", lambda d: d.update(runs=[dict(ok=True, forward_passes=32)], summary=[]))
    with pytest.raises(SystemExit, match="no runs or no summary"):
        rg.render(tree)


def test_the_empty_refusal_survives_python_dash_O(tree: Path) -> None:
    edit(dest(tree) / "llada.json", lambda d: d.update(runs=[]))
    code = ("import sys; sys.path.insert(0, %r); import report_gpu_dispatch as r; from pathlib import Path; r.main(Path(%r))"
            % (str(SCRIPT.parent), str(tree)))
    p = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode != 0 and "no runs" in p.stderr and not (dest(tree) / "README.md").exists()


def test_a_failed_run_a_wrong_pass_count_and_a_partial_file_are_refused(tree: Path) -> None:
    for patch, match in ((lambda d: d["runs"][0].update(ok=False), "failed or did not do"),
                         (lambda d: d["runs"][0].update(forward_passes=32), "failed or did not do"),
                         (lambda d: d.update(partial=True), "partial")):
        original = (dest(tree) / "llada.json").read_text()
        edit(dest(tree) / "llada.json", patch)
        with pytest.raises(SystemExit, match=match):
            rg.render(tree)
        (dest(tree) / "llada.json").write_text(original)


def test_a_method_that_no_longer_says_what_the_prose_says_is_refused(tree: Path) -> None:
    edit(dest(tree) / "dream.json", lambda d: d.update(method="run_bench.run_mojo; early stop on"))
    with pytest.raises(SystemExit, match="no early stop"):
        rg.render(tree)


def test_timing_files_with_different_settings_or_rounds_are_refused(tree: Path) -> None:
    edit(dest(tree) / "dream.json", lambda d: d.update(block_length=16))
    with pytest.raises(SystemExit, match="disagree on block length"):
        rg.render(tree)
    edit(dest(tree) / "dream.json", lambda d: d.update(block_length=32))
    edit(dest(tree) / "dream.json", lambda d: d.update(runs=[r for r in d["runs"] if r["rep"] != 1]))
    with pytest.raises(SystemExit, match="different numbers of measured rounds"):
        rg.render(tree)


def test_an_exactness_file_with_a_difference_or_nothing_in_it_is_refused(tree: Path) -> None:
    path = tree / "bench/checks/2026-10-30-6000d-dispatch-dream-exact.json"
    original = path.read_text()
    for patch, match in ((lambda d: d["logits"][0].update(bytes_equal=False), "records a difference"),
                         (lambda d: d["generation"][1].update(tokens_equal=False), "records a difference"),
                         (lambda d: d.update(logits=[]), "no logit canvases"),
                         (lambda d: d.update(generation=[]), "no logit canvases")):
        edit(path, patch)
        with pytest.raises(SystemExit, match=match):
            rg.render(tree)
        path.write_text(original)


def test_main_writes_the_readme_next_to_the_evidence(tree: Path) -> None:
    rg.main(tree)
    assert (dest(tree) / "README.md").read_text(encoding="utf-8") == rg.render(tree)
