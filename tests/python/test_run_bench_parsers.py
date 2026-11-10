"""run_bench's parsers against real-looking output (#250): the CUDA layer check, llama.cpp's timing line, the median
helper, and what run_mojo and run_diffuse make of a process that failed or printed the wrong thing."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))


def load(name: str, path: Path):
    s = importlib.util.spec_from_file_location(name, path)
    assert s is not None and s.loader is not None
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


rb = load("run_bench_parsers", ROOT / "bench/run_bench.py")

LOADED = "".join(f"0.01.{i:03d}.000 D load_tensors: layer {i:3d} assigned to device {d}\n"
                 for i, d in enumerate(["CUDA0", "CUDA0", "CUDA0"]))


# cuda_layer_error ---------------------------------------------------------------------------------------

def test_every_layer_on_cuda_is_no_error() -> None:
    assert rb.cuda_layer_error(LOADED) is None
    assert rb.cuda_layer_error("layer 0 assigned to device CUDA0\nlayer   1 assigned to device CUDA1\n") is None


def test_a_mixed_cuda_and_cpu_layer_list_is_refused_and_names_both_devices() -> None:
    out = "layer 0 assigned to device CPU\nlayer 1 assigned to device CUDA0\nlayer 2 assigned to device CUDA0\n"
    err = rb.cuda_layer_error(out)
    assert err is not None and "['CPU', 'CUDA0']" in err


def test_a_run_with_no_layer_lines_is_refused() -> None:
    err = rb.cuda_layer_error("diffusion total time: 1.0ms, time per step: 1.0ms\n")
    assert err is not None and "[]" in err and "every layer" in err
    assert rb.cuda_layer_error("") is not None


def test_a_cpu_only_run_is_refused() -> None:
    err = rb.cuda_layer_error("layer 0 assigned to device CPU\nlayer 1 assigned to device CPU\n")
    assert err is not None and "['CPU']" in err


# parse_llama_output -------------------------------------------------------------------------------------

REAL = (
    "diffusion total time: 8912.00ms, time per step: 278.50ms, sampling time per step: 0.12ms\n"
    "The answer is 4.\n"
    "mainstream models differ.\n"
    "0.01.487.312 D load_tensors: offloaded 33/33 layers to GPU\n"
    "0.01.487.313 I something informational\n"
    "1.2.3.4 is an address, not a log line\n"
    "llama_perf_context_print: load time = 100 ms\n"
    "common_perf_print: sampling time = 3 ms\n"
    "main: done\n"
)


def test_the_timing_line_with_a_sampling_suffix_is_read_and_the_noise_lines_dropped() -> None:
    r = rb.parse_llama_output(REAL)
    assert r is not None
    assert r["total_s"] == pytest.approx(8.912) and r["ms_per_step"] == pytest.approx(278.5)
    assert r["text"].splitlines() == ["The answer is 4.", "mainstream models differ.", "1.2.3.4 is an address, not a log line"]


def test_the_timing_line_without_the_suffix_is_read() -> None:
    r = rb.parse_llama_output("total time: 1000.0ms, time per step: 31.0ms\nhello\nllama_perf: x\n")
    assert r == {"total_s": 1.0, "ms_per_step": 31.0, "text": "hello"}


def test_the_suffix_is_not_part_of_the_text() -> None:
    r = rb.parse_llama_output("total time: 10.0ms, time per step: 1.0ms, sampling time per step: 0.5ms\nhi\n")
    assert r is not None and r["text"] == "hi"


@pytest.mark.parametrize("output", ["", "no timing here\n", "total time: 10ms\n", "time per step: 2ms\n"])
def test_no_timing_line_is_none(output: str) -> None:
    assert rb.parse_llama_output(output) is None


# median_or_none -----------------------------------------------------------------------------------------

def test_median_or_none() -> None:
    assert rb.median_or_none([]) is None
    assert rb.median_or_none([3.0]) == 3.0
    assert rb.median_or_none([5.0, 1.0, 3.0]) == 3.0
    assert rb.median_or_none([1.0, 2.0, 3.0, 10.0]) == 2.5


# run_mojo -----------------------------------------------------------------------------------------------

CFG = {"models": {"gguf": "m.gguf"}, "gen_length": 32, "steps": 32, "block_length": 32}


def mojo_json(**over: object) -> str:
    j = dict(generate_s=2.0, ms_per_step=62.5, forward_passes=32, prompt_tokens=5, gen_length=32, text="out",
             gemm_ms_per_forward=1.5, attention_ms_per_forward=2.5, lm_head_ms_per_forward=3.5)
    j.update(over)
    return "loading...\n" + json.dumps(j) + "\n"


def test_run_mojo_maps_the_json_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "timed", lambda cmd: (0, mojo_json(), "", 3.0, 4096))
    r = rb.run_mojo(CFG, "p", 4, [], 5)
    assert r == dict(ok=True, wall_s=3.0, peak_rss_bytes=4096, generate_s=2.0, ms_per_step=62.5, forward_passes=32,
                     startup_s=1.0, seq_len=37, prompt_tokens=5, text="out", forward_ms=None,
                     gemm_ms_per_forward=1.5, attention_ms_per_forward=2.5, lm_head_ms_per_forward=3.5)


def test_run_mojo_reports_a_failed_process_with_the_tail_of_its_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "timed", lambda cmd: (3, "", "x" * 500 + "THE END", 1.0, 1))
    r = rb.run_mojo(CFG, "p", 4, [])
    assert r["ok"] is False and r["rc"] == 3 and r["error"].endswith("THE END") and len(r["error"]) == 400


def test_run_mojo_with_a_json_line_but_a_failed_exit_is_still_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "timed", lambda cmd: (1, mojo_json(), "crashed at exit", 1.0, 1))
    r = rb.run_mojo(CFG, "p", 4, [])
    assert r["ok"] is False and r["rc"] == 1 and r["error"] == "crashed at exit"


def test_run_mojo_without_a_json_line_is_a_failure_even_at_exit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rb, "timed", lambda cmd: (0, "plain text\n", "", 1.0, 1))
    r = rb.run_mojo(CFG, "p", 4, [])
    assert r["ok"] is False and r["error"] == "plain text\n"


def test_run_mojo_passes_extra_flags_last(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(rb, "timed", lambda cmd: (seen.append(cmd) or (0, mojo_json(), "", 1.0, 1)))
    rb.run_mojo(CFG, "p", 4, ["--device", "gpu"], 5)
    assert seen[0][-2:] == ["--device", "gpu"] and "--no-early-stop" in seen[0] and "--json" in seen[0]
    assert seen[0][seen[0].index("--threads") + 1] == "4"


# run_diffuse --------------------------------------------------------------------------------------------

REAL_POPEN = subprocess.Popen


class FakePopen:
    """The diffuse-cli process: a real python prints the given stderr and exits with the given code."""

    stderr_text = ""
    code = 0
    stdout_text = "tokens"

    def __init__(self, cmd: list[str], **kw: object) -> None:
        Path(cmd[4]).write_text("0.5 1000\n")
        script = "import sys; sys.stderr.write(sys.argv[1]); sys.stdout.write(sys.argv[3]); sys.exit(int(sys.argv[2]))"
        self.proc = REAL_POPEN([sys.executable, "-c", script, self.stderr_text, str(self.code), self.stdout_text], **kw)
        self.stderr, self.stdout = self.proc.stderr, self.proc.stdout

    def wait(self) -> int:
        return self.proc.wait()


def diffuse(monkeypatch: pytest.MonkeyPatch, stderr: str, code: int = 0) -> dict:
    FakePopen.stderr_text, FakePopen.code = stderr, code
    monkeypatch.setattr(rb.subprocess, "Popen", FakePopen)
    return rb.run_diffuse({**CFG, "diffuse_cpp": "/d", "models": {"gguf": "m", "diffuse_gguf": "d"}}, "1,2,3", 4, 3, "equal")


def test_run_diffuse_reads_the_last_step_and_the_generation_window(monkeypatch: pytest.MonkeyPatch) -> None:
    r = diffuse(monkeypatch, "Generating 32\nstep 1/32\nstep 32/32\nGenerated token IDs: 1\n")
    assert r["ok"] is True and r["forward_passes"] == 32 and r["seq_len"] == 3 + 32
    assert r["ms_per_step"] == pytest.approx(r["generate_s"] * 1000 / 32) and r["tokens"] == "tokens"


def test_a_failed_diffuse_process_is_a_failure_with_its_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    r = diffuse(monkeypatch, "Generating 32\nstep 1/32\nGenerated token IDs: 1\nboom\n", code=2)
    assert r["ok"] is False and r["rc"] == 2 and "boom" in r["error"]


@pytest.mark.parametrize("stderr", ["step 1/32\nGenerated token IDs: 1\n", "Generating 32\nstep 1/32\n", ""])
def test_missing_markers_make_a_failure(monkeypatch: pytest.MonkeyPatch, stderr: str) -> None:
    assert diffuse(monkeypatch, stderr)["ok"] is False
