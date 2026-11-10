"""run_bench publishes nothing that no run measured (#234): selection, step counts, equal work, token ids."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rb = load_module("run_bench_guards", ROOT / "bench/run_bench.py")

CONFIGURED = ["mojo-dllm", "llama.cpp"]
CFG = {"models": {"gguf": "m.gguf", "diffuse_gguf": "d.gguf"}, "gen_length": 32, "steps": 32, "block_length": 32,
       "diffuse_cpp": "/nonexistent-diffuse"}


def test_a_selection_of_zero_runtimes_is_refused() -> None:
    with pytest.raises(SystemExit) as e:
        rb.select_runtimes([], "", 3)
    assert "no runtime" in str(e.value)


def test_a_misspelled_runtime_is_refused_by_name() -> None:
    with pytest.raises(SystemExit) as e:
        rb.select_runtimes(CONFIGURED, "mojo-dll", 3)
    assert "mojo-dll" in str(e.value) and "mojo-dllm" in str(e.value)


def test_an_empty_only_list_is_refused() -> None:
    with pytest.raises(SystemExit):
        rb.select_runtimes(CONFIGURED, ",", 3)


@pytest.mark.parametrize("reps", [0, -1])
def test_fewer_than_one_rep_is_refused(reps: int) -> None:
    with pytest.raises(SystemExit) as e:
        rb.select_runtimes(CONFIGURED, "", reps)
    assert "--reps" in str(e.value)


def test_a_valid_selection_keeps_config_order() -> None:
    assert rb.select_runtimes(CONFIGURED, "llama.cpp,mojo-dllm", 1) == CONFIGURED
    assert rb.select_runtimes(CONFIGURED, "", 1) == CONFIGURED


def test_token_ids_must_be_digits_and_commas() -> None:
    assert rb.parse_token_ids("log line\n[2, 15, 7]\n") == ("2,15,7", 3)
    for bad in ("", "\n", "[]", "[2, x]", "loading model...", "[2,,3]"):
        with pytest.raises(SystemExit):
            rb.parse_token_ids(bad)


def mojo_stdout(**over: object) -> str:
    j = dict(generate_s=1.0, ms_per_step=31.0, forward_passes=32, prompt_tokens=5, gen_length=32, text="t",
             gemm_ms_per_forward=1.0, attention_ms_per_forward=1.0, lm_head_ms_per_forward=1.0, forward_ms=2.0)
    j.update(over)
    return json.dumps(j) + "\n"


def run_mojo_with(monkeypatch: pytest.MonkeyPatch, stdout: str, n_prompt: int | None = 5) -> dict:
    monkeypatch.setattr(rb, "timed", lambda cmd: (0, stdout, "", 2.0, 1000))
    return rb.run_mojo(CFG, "p", 4, [], n_prompt)


def test_mojo_with_the_expected_work_is_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    assert run_mojo_with(monkeypatch, mojo_stdout())["ok"] is True


def test_mojo_that_did_fewer_forward_passes_is_not_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    r = run_mojo_with(monkeypatch, mojo_stdout(forward_passes=20))
    assert r["ok"] is False and "20 forward passes, not 32" in r["error"]


def test_mojo_that_counted_another_prompt_length_is_not_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    r = run_mojo_with(monkeypatch, mojo_stdout(prompt_tokens=6))
    assert r["ok"] is False and "6 prompt tokens" in r["error"]


def test_llama_rows_say_their_pass_count_is_the_configured_one(monkeypatch: pytest.MonkeyPatch) -> None:
    out = "total time: 1000.0ms, time per step: 31.0ms\nhello\n"
    monkeypatch.setattr(rb, "timed", lambda cmd: (0, out, "", 2.0, 1000))
    r = rb.run_llama({**CFG, "llama_cpp": "/x"}, "p", 4, 37)
    assert r["ok"] and r["forward_passes"] == 32 and r["forward_passes_measured"] is False


REAL_POPEN = subprocess.Popen


class FakePopen:
    """Stands in for the diffuse-cli process: a real `python3` prints the given stderr, and /usr/bin/time's file is written."""

    stderr_text = ""

    def __init__(self, cmd: list[str], **kw: object) -> None:
        Path(cmd[4]).write_text("0.5 1000\n")
        script = "import sys,time; sys.stderr.write(sys.argv[1]); sys.stderr.flush()"
        self.proc = REAL_POPEN([sys.executable, "-c", script, self.stderr_text], **kw)
        self.stderr, self.stdout = self.proc.stderr, self.proc.stdout

    def wait(self) -> int:
        return self.proc.wait()


def run_diffuse_with(monkeypatch: pytest.MonkeyPatch, stderr: str) -> dict:
    FakePopen.stderr_text = stderr
    monkeypatch.setattr(rb.subprocess, "Popen", FakePopen)
    return rb.run_diffuse(CFG, "1,2,3", 4, 3, "equal")


def test_diffuse_with_step_lines_uses_the_last_step(monkeypatch: pytest.MonkeyPatch) -> None:
    r = run_diffuse_with(monkeypatch, "Generating 32\nstep 1/32\nstep 26/32\nGenerated token IDs: 1\n")
    assert r["ok"] is True and r["forward_passes"] == 26


def test_diffuse_without_step_lines_is_failed_not_divided_by_the_configured_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = run_diffuse_with(monkeypatch, "Generating 32\nGenerated token IDs: 1\n")
    assert r["ok"] is False and "step" in r["error"]
    assert "ms_per_step" not in r
