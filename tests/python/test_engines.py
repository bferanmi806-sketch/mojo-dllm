"""bench/eval_engine.py and bench/engines: the GSM8K runner for engines other than mojo-dllm.

Every test runs against a fake `llama-diffusion-cli`, a shell script that prints llama.cpp's timing line and a
canned answer, and a fake `mojo-dllm tokenize`. Nothing here loads a model, touches the GPU or writes into
bench/eval/runs: the run directory is a temporary one.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import engines  # noqa: E402
import eval_engine  # noqa: E402
import eval_gsm8k  # noqa: E402
import isolation  # noqa: E402
import power  # noqa: E402
import run_bench  # noqa: E402
from engines import llamacpp  # noqa: E402

FAKE_LLAMA = r"""#!/usr/bin/env bash
# A stand-in for llama-diffusion-cli. FAKE_MODE picks what it does; FAKE_LOG_DIR gets one file per call
# holding the NUL-separated argv; FAKE_ANSWER is the text it "generates".
set -u
mode="${FAKE_MODE:-ok}"
if [ -n "${FAKE_LOG_DIR:-}" ]; then
    printf '%s\0' "$@" > "$FAKE_LOG_DIR/call-$(date +%s%N)-$$"
fi
gpu=0
for a in "$@"; do
    if [ "$a" = "-ngl" ]; then gpu=1; fi
done
if [ "$gpu" = 1 ]; then
    dev=CUDA0
    if [ "$mode" = cpu_fallback ]; then dev=CPU; fi
    for i in 0 1 2; do
        echo "load_tensors: layer   $i assigned to device $dev, is_swa = 0" >&2
    done
fi
echo "diffusion_params: - steps                     u32              = 4" >&2
if [ "$mode" = sleep ]; then exec sleep 30; fi
if [ "$mode" = notiming ]; then
    echo "The answer is 7." >&2
    exit 0
fi
echo "total time: 12345.6ms, time per step: 48.2ms, sampling time per step: 19.21ms" >&2
if [ "$mode" != empty ]; then
    printf '\n%s\n' "${FAKE_ANSWER:-The answer is 18.}" >&2
fi
echo "llama_perf_context_print: load time = 1.00 ms" >&2
if [ "$mode" = fail ]; then
    echo "ggml_cuda_error: out of memory" >&2
    exit 3
fi
exit 0
"""

FAKE_MOJO = r"""#!/usr/bin/env bash
# A stand-in for `mojo-dllm tokenize`: five token ids on the last line.
if [ -n "${FAKE_MOJO_SLEEP:-}" ]; then exec sleep "$FAKE_MOJO_SLEEP"; fi
printf 'loading\n[1,2,3,4,5]\n'
"""

N_PROMPT = 5  # the five ids FAKE_MOJO prints


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    return path


def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def _first_problems(n: int) -> list[dict]:
    return [json.loads(l) for l in eval_gsm8k.DATA.read_text().splitlines()][:n]


class Rig(SimpleNamespace):
    def run(self, *extra: str, arm: str = "arm", model: Path | None = None, limit: int = 2,
            device: str = "gpu", steps: str | None = "4", battery: bool = True) -> int:
        argv = ["run", "--engine", "llamacpp", "--model", str(model or self.model), "--arm", arm,
                "--device", device, "--gen-length", "16", "--limit", str(limit),
                "--llama-cpp", str(self.llama), "--mojo-dllm", str(self.mojo)]
        if steps is not None:
            argv += ["--steps", steps]
        if battery:
            argv.append("--allow-battery")
        return eval_engine.main(argv + list(extra))

    def rows(self, arm: str = "arm") -> list[dict]:
        path = self.runs / f"{arm}.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def calls(self) -> list[list[str]]:
        files = sorted(self.logs.iterdir())
        return [f.read_bytes().decode().split("\0")[:-1] for f in files]


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    llama = tmp_path / "llama.cpp"
    _git(tmp_path, "init", "-q", str(llama))
    _git(llama, "commit", "-q", "--allow-empty", "-m", "pin")
    for build in ("build", "build-cuda"):
        _script(llama / build / "bin" / "llama-diffusion-cli", FAKE_LLAMA)
    mojo = _script(tmp_path / "bin" / "mojo-dllm", FAKE_MOJO)
    model = tmp_path / "models" / "LLaDA-test.Q4_K_M.gguf"
    model.parent.mkdir()
    model.write_bytes(b"GGUF")
    runs = tmp_path / "runs"
    logs = tmp_path / "logs"
    logs.mkdir()
    (tmp_path / "bench" / "eval").mkdir(parents=True)
    monkeypatch.setattr(eval_gsm8k, "RUNS", runs)
    monkeypatch.setattr(eval_gsm8k, "PAUSE", runs / "PAUSE")
    monkeypatch.setattr(eval_gsm8k, "ROOT", tmp_path)
    monkeypatch.setenv("FAKE_LOG_DIR", str(logs))
    monkeypatch.delenv("FAKE_MODE", raising=False)
    monkeypatch.delenv("FAKE_ANSWER", raising=False)
    return Rig(llama=llama, mojo=mojo, model=model, runs=runs, logs=logs, root=tmp_path)


RECORD_KEYS = {"index", "gold", "pred", "correct", "forward_passes", "generate_s", "stopped_early", "text",
               "flags", "version", "model", "power_retries", "on_battery", "power_source", "isolation", "schema", "steps",
               "completion_tokens", "finish_reason"}


# registry


def test_registry_holds_llamacpp_and_it_satisfies_the_protocol() -> None:
    cls = engines.get("llamacpp")
    assert cls is llamacpp.LlamaCpp
    assert cls.name == "llamacpp"
    assert "llamacpp" in engines.names()
    assert isinstance(cls(SimpleNamespace(llama_cpp="x", mojo_dllm="y", threads=1)), engines.Engine)


def test_unknown_engine_names_the_known_ones() -> None:
    with pytest.raises(SystemExit) as e:
        engines.get("nosuchengine")
    assert "nosuchengine" in str(e.value) and "llamacpp" in str(e.value) and "vllm" in str(e.value)


# a run


def test_record_has_exactly_the_keys_of_the_mojo_arm(rig: Rig) -> None:
    assert rig.run() == 0
    rows = rig.rows()
    assert len(rows) == 2
    for r in rows:
        assert set(r) == RECORD_KEYS
    assert [r["index"] for r in rows] == [p["index"] for p in _first_problems(2)]


def test_record_values_for_llamacpp(rig: Rig) -> None:
    assert rig.run() == 0
    r = rig.rows()[0]
    assert r["model"] == "LLaDA-test.Q4_K_M.gguf"                      # the basename, as eval_gsm8k records it
    assert r["version"] == _git(rig.llama, "rev-parse", "--short", "HEAD")
    assert r["generate_s"] == pytest.approx(12.3456)                    # llama.cpp's own total time, in seconds
    assert r["forward_passes"] is None and r["steps"] == 4              # llama.cpp counts nothing: steps is the configured count
    assert r["schema"] == "engine" and r["completion_tokens"] is None and r["finish_reason"] is None
    assert r["stopped_early"] is None                                   # llama.cpp has no early stop to report
    assert r["flags"]["forward_passes_measured"] is False
    assert r["flags"]["schedule"] == "timestep"
    assert r["flags"]["engine"] == "llamacpp"
    assert r["text"] == "The answer is 18."
    assert isinstance(r["power_retries"], int) and r["power_source"] in {"mains", "battery", "none"}


def test_scores_with_the_shared_extractor(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    # The first two problems of gsm8k-200.jsonl have gold answers 70000 and 460 (the file is pinned at 200 rows).
    assert [p["answer"] for p in _first_problems(2)] == ["70000", "460"]
    monkeypatch.setenv("FAKE_ANSWER", "so it is cheap. The answer is 70000.")
    assert rig.run() == 0
    r0, r1 = rig.rows()
    assert r0["pred"] == "70000" and r0["correct"] is True and r0["gold"] == "70000"
    assert r1["pred"] == "70000" and r1["gold"] == "460" and r1["correct"] is False


def test_wrong_answer_is_scored_wrong(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_ANSWER", "The answer is 123456789.")
    assert rig.run() == 0
    assert [r["correct"] for r in rig.rows()] == [False, False]
    assert rig.rows()[0]["pred"] == "123456789"


def test_command_line_uses_the_real_flags_and_the_gpu_build(rig: Rig) -> None:
    assert rig.run() == 0
    argv = rig.calls()[0]
    assert argv[0] == "-m" and argv[1] == str(rig.model)
    prompt = argv[argv.index("-p") + 1]
    assert prompt.startswith(_first_problems(1)[0]["question"])
    assert prompt.endswith('"The answer is N." where N is the final number.')
    canvas = str(N_PROMPT + 16)
    for flag, value in [("-c", canvas), ("-b", canvas), ("-ub", canvas), ("--diffusion-steps", "4"),
                        ("--diffusion-eps", "0.001"), ("--temp", "0"), ("--seed", "42"), ("-ngl", "99")]:
        assert argv[argv.index(flag) + 1] == value, flag
    assert "-v" in argv
    assert "--diffusion-block-length" not in argv


def test_cpu_device_uses_the_cpu_build_without_offload(rig: Rig) -> None:
    (rig.llama / "build-cuda" / "bin" / "llama-diffusion-cli").unlink()
    assert rig.run(device="cpu") == 0
    argv = rig.calls()[0]
    assert "-ngl" not in argv and "-v" not in argv


def test_block_schedule_passes_the_block_length_and_no_eps(rig: Rig) -> None:
    assert rig.run("--schedule", "block", "--block-length", "8", arm="blocks") == 0
    argv = rig.calls()[0]
    assert argv[argv.index("--diffusion-block-length") + 1] == "8"
    assert "--diffusion-eps" not in argv
    assert rig.rows("blocks")[0]["flags"]["schedule"] == "block"


def test_extra_flags_after_double_dash_reach_the_binary_and_the_record(rig: Rig) -> None:
    assert rig.run("--", "--diffusion-algorithm", "1") == 0
    argv = rig.calls()[0]
    assert argv[argv.index("--diffusion-algorithm") + 1] == "1"
    assert rig.rows()[0]["flags"]["extra"] == ["--diffusion-algorithm", "1"]


def test_limit_bounds_the_problems(rig: Rig) -> None:
    assert rig.run(limit=1) == 0
    assert len(rig.rows()) == 1 and len(rig.calls()) == 1
    assert rig.run(limit=3) == 0          # resumes: only two more are answered
    assert len(rig.rows()) == 3 and len(rig.calls()) == 3


# refusals


def test_non_zero_exit_is_refused_and_nothing_is_scored(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "fail")        # the timing line and an answer are both there
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "exit" in str(e.value) and "3" in str(e.value)
    assert rig.rows() == []


def test_missing_timing_line_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "notiming")
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "time per step" in str(e.value)
    assert rig.rows() == []


def test_empty_text_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "empty")
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "no text" in str(e.value)
    assert rig.rows() == []


def test_cuda_run_that_left_a_layer_on_the_cpu_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "cpu_fallback")    # timing, text and exit 0 are all fine
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "every layer" in str(e.value) and "CPU" in str(e.value)
    assert rig.rows() == []


def test_missing_binary_is_refused_before_any_problem(rig: Rig) -> None:
    (rig.llama / "build-cuda" / "bin" / "llama-diffusion-cli").unlink()
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "build-cuda" in str(e.value) and "llama-diffusion-cli" in str(e.value)
    assert rig.rows() == [] and rig.calls() == []


def test_missing_model_is_refused_before_any_problem(rig: Rig) -> None:
    with pytest.raises(SystemExit) as e:
        rig.run(model=rig.root / "models" / "nope.gguf")
    assert "nope.gguf" in str(e.value)
    assert rig.calls() == []


# the shared CUDA check and parser (also what run_bench.run_llama uses)


def test_cuda_layer_error_cases() -> None:
    ok = "layer   0 assigned to device CUDA0, is_swa = 0\nlayer   1 assigned to device CUDA0, is_swa = 0\n"
    assert run_bench.cuda_layer_error(ok) is None
    mixed = ok + "layer   2 assigned to device CPU, is_swa = 0\n"
    assert "CPU" in run_bench.cuda_layer_error(mixed)
    assert "did not put every layer" in run_bench.cuda_layer_error("no device lines at all")


def test_parse_llama_output_drops_the_rest_of_the_timing_line() -> None:
    # the real line: llama.cpp appends a third field after "time per step"
    out = ("\ntotal time: 17338.78ms, time per step: 135.46ms, sampling time per step: 19.21ms\n"
           "The answer is 18.\nllama_perf_context_print: tail\n")
    got = run_bench.parse_llama_output(out)
    assert got == {"total_s": 17.33878, "ms_per_step": 135.46, "text": "The answer is 18."}


def test_parse_llama_output_keeps_the_whole_text() -> None:
    long = "x" * 900
    out = (f"0.01.487.312 D load_tensors: noise\ntotal time: 2000.0ms, time per step: 20.0ms\n"
           f"{long}\nllama_perf_context_print: tail\n")
    got = run_bench.parse_llama_output(out)
    assert got == {"total_s": 2.0, "ms_per_step": 20.0, "text": long}
    assert run_bench.parse_llama_output("nothing here") is None


def test_run_llama_still_refuses_a_cpu_fallback_and_still_parses(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    c = {"llama_cpp": str(rig.llama), "models": {"gguf": str(rig.model)}, "steps": 4}
    good = run_bench.run_llama(c, "hi", 2, 21, cuda=True)
    assert good["ok"] is True and good["generate_s"] == pytest.approx(12.3456)
    assert good["ms_per_step"] == 48.2 and good["forward_passes"] is None and good["steps"] == 4
    monkeypatch.setenv("FAKE_MODE", "cpu_fallback")
    bad = run_bench.run_llama(c, "hi", 2, 21, cuda=True)
    assert bad["ok"] is False and "every layer" in bad["error"]
    monkeypatch.setenv("FAKE_MODE", "fail")
    assert run_bench.run_llama(c, "hi", 2, 21)["ok"] is False


# resume


def test_resume_skips_answered_problems(rig: Rig) -> None:
    assert rig.run(limit=2) == 0
    assert rig.run(limit=2) == 0
    assert len(rig.rows()) == 2 and len(rig.calls()) == 2


@pytest.mark.parametrize("change", [["--steps", "8"], ["--gen-length", "32"], ["--schedule", "block"],
                                    ["--threads", "3"], ["--", "--diffusion-algorithm", "1"]])
def test_resume_with_a_changed_flag_is_refused(rig: Rig, change: list[str]) -> None:
    assert rig.run(limit=1) == 0
    # a later flag overrides an earlier one in argparse, so `change` replaces the default of rig.run
    with pytest.raises(SystemExit) as e:
        rig.run(*change, limit=2)
    assert "arm 'arm' was started with" in str(e.value) and "Use a new arm name" in str(e.value)
    assert len(rig.rows()) == 1


def test_resume_with_a_new_llamacpp_commit_is_refused(rig: Rig) -> None:
    assert rig.run(limit=1) == 0
    _git(rig.llama, "commit", "-q", "--allow-empty", "-m", "newer")
    with pytest.raises(SystemExit) as e:
        rig.run(limit=2)
    assert "version=" in str(e.value)
    assert len(rig.rows()) == 1


def test_resume_with_another_model_is_refused(rig: Rig) -> None:
    assert rig.run(limit=1) == 0
    other = rig.root / "models" / "Dream-test.gguf"
    other.write_bytes(b"GGUF")
    with pytest.raises(SystemExit) as e:
        rig.run(model=other, limit=2)
    assert "Dream-test.gguf" in str(e.value)
    assert len(rig.rows()) == 1


def test_pause_file_holds_the_next_problem(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.runs.mkdir(parents=True)
    (rig.runs / "PAUSE").write_text("")
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        (rig.runs / "PAUSE").unlink()

    monkeypatch.setattr(eval_gsm8k, "time", SimpleNamespace(sleep=fake_sleep))   # not the global time.sleep
    assert rig.run(limit=1) == 0
    assert sleeps == [10] and len(rig.rows()) == 1


# compare


def _arm(runs: Path, name: str, passes: list, correct: list[bool], model: str = "LLaDA-test.Q4_K_M.gguf") -> None:
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / f"{name}.jsonl").open("w") as f:
        for i, (p, c) in enumerate(zip(passes, correct)):
            f.write(json.dumps(dict(index=i, correct=c, forward_passes=p, generate_s=10.0 + i, flags=["--x"],
                                    version="v", model=model)) + "\n")


def test_compare_averages_over_the_rows_that_have_a_count(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    _arm(rig.runs, "a", [10, None, 20], [True, True, False])
    _arm(rig.runs, "b", [8, 8, 8], [True, False, False])
    assert eval_gsm8k.cmd_compare("a", "b") == 0
    doc = json.loads(next((rig.root / "bench" / "eval").glob("*-a-vs-b.json")).read_text())
    assert doc["forward_passes_mean"] == {"a": 15.0, "b": 8.0}
    assert doc["forward_passes_rows"] == {"a": 2, "b": 3}
    assert "forward_passes_rows" in capsys.readouterr().out


def test_compare_with_no_counts_at_all_reports_none(rig: Rig) -> None:
    _arm(rig.runs, "a", [None, None], [True, False])
    _arm(rig.runs, "b", [5, 7], [True, True])
    assert eval_gsm8k.cmd_compare("a", "b") == 0
    doc = json.loads(next((rig.root / "bench" / "eval").glob("*-a-vs-b.json")).read_text())
    assert doc["forward_passes_mean"] == {"a": None, "b": 6.0}
    assert doc["forward_passes_rows"] == {"a": 0, "b": 2}


def test_compare_keeps_its_other_keys(rig: Rig) -> None:
    _arm(rig.runs, "a", [1, 2], [True, False])
    _arm(rig.runs, "b", [1, 2], [True, True])
    assert eval_gsm8k.cmd_compare("a", "b") == 0
    doc = json.loads(next((rig.root / "bench" / "eval").glob("*-a-vs-b.json")).read_text())
    assert set(doc) == {"date", "problems", "arms", "limit", "problems_in_arm", "config", "accuracy", "difference",
                        "ci95", "ci95_method", "discordant", "mcnemar_p", "forward_passes_mean", "forward_passes_measured",
                        "forward_passes_rows", "generate_s_mean", "generate_s_mean_clean", "clean_rows", "noisy_rows", "unchecked_rows",
                        "models", "weights_differ"}


def test_compare_accepts_an_engine_arm_next_to_a_mojo_arm(rig: Rig) -> None:
    assert rig.run(arm="llamacpp-arm") == 0
    mojo_rows = [dict(r, flags=["--max-tokens", "16", "--json"], version="abc1234", forward_passes=3,
                      stopped_early=True) for r in rig.rows("llamacpp-arm")]
    (rig.runs / "mojo-arm.jsonl").write_text("".join(json.dumps(r) + "\n" for r in mojo_rows))
    assert eval_gsm8k.cmd_compare("mojo-arm", "llamacpp-arm") == 0
    doc = json.loads(next((rig.root / "bench" / "eval").glob("*-mojo-arm-vs-llamacpp-arm.json")).read_text())
    assert doc["problems"] == 2
    assert doc["forward_passes_mean"] == {"mojo-arm": 3.0, "llamacpp-arm": None}, "a configured count is not a mean"
    assert doc["forward_passes_measured"] == {"mojo-arm": True, "llamacpp-arm": False}
    assert doc["config"]["llamacpp-arm"]["flags"]["forward_passes_measured"] is False


# the fake scripts are valid shell and the tests above did select something


def test_fake_scripts_pass_bash_n(tmp_path: Path) -> None:
    for name, body in (("llama", FAKE_LLAMA), ("mojo", FAKE_MOJO)):
        path = tmp_path / name
        path.write_text(body)
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    assert not re.search(r"\\x[0-9a-f]{2}", FAKE_LLAMA + FAKE_MOJO), "generate scripts from heredocs, not escapes"
    assert os.access(str(ROOT / "bench" / "eval_engine.py"), os.R_OK)


# defaults, power, version, timeouts, token count


def test_steps_default_to_the_generation_length(rig: Rig) -> None:
    assert rig.run(steps=None, limit=1) == 0
    argv = rig.calls()[0]
    assert argv[argv.index("--diffusion-steps") + 1] == "16"
    assert rig.rows()[0]["flags"]["diffusion_steps"] == 16 and rig.rows()[0]["steps"] == 16


@pytest.mark.parametrize("battery", [True, False])
def test_allow_battery_reaches_run_on_mains_as_given(rig: Rig, monkeypatch: pytest.MonkeyPatch,
                                                      battery: bool) -> None:
    seen: list[bool] = []

    def stub(fn, *, allow_battery, log):
        seen.append(allow_battery)
        return fn(), {"power_retries": 0, "on_battery": False, "power_source": "mains"}

    monkeypatch.setattr(power, "run_on_mains", stub)
    assert rig.run(limit=1, battery=battery) == 0
    assert seen == [battery]


def test_failing_git_does_not_record_an_empty_or_partial_version(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    shim = rig.root / "shim"
    _script(shim / "git", "#!/usr/bin/env bash\necho deadbee\necho 'fatal: broken' >&2\nexit 128\n")
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "llama.cpp commit" in str(e.value)
    assert rig.rows() == [] and rig.calls() == []


def test_not_a_git_tree_is_refused(rig: Rig) -> None:
    subprocess.run(["rm", "-rf", str(rig.llama / ".git")], check=True)
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "llama.cpp commit" in str(e.value)


def test_generate_timeout_names_the_problem_and_stores_nothing(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "sleep")
    with pytest.raises(SystemExit) as e:
        rig.run("--timeout-s", "1")
    msg = str(e.value)
    assert f"problem {_first_problems(1)[0]['index']}" in msg and "timed out after 1" in msg
    assert rig.rows() == []


def test_tokenize_timeout_is_a_failure(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llamacpp, "SHORT_TIMEOUT_S", 1)
    monkeypatch.setenv("FAKE_MOJO_SLEEP", "30")
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "tokenize" in str(e.value) and "timed out" in str(e.value)
    assert rig.rows() == [] and rig.calls() == []


def test_git_timeout_is_a_failure(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    shim = rig.root / "shim"
    _script(shim / "git", "#!/usr/bin/env bash\nexec sleep 30\n")
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(llamacpp, "SHORT_TIMEOUT_S", 1)
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "timed out" in str(e.value)


def test_timeout_default_is_generous_and_recorded_nowhere_in_flags(rig: Rig) -> None:
    assert eval_engine.DEFAULT_TIMEOUT_S >= 1800
    assert rig.run(limit=1) == 0
    assert "timeout" not in json.dumps(rig.rows()[0]["flags"])      # a timeout does not change what was measured


# isolation: the runner preflights the arm, runs every problem in a window and records it (bench/isolation.py)


def test_rows_record_a_clean_window_and_the_arm_writes_its_env_file(rig: Rig) -> None:
    assert rig.run() == 0
    for r in rig.rows():
        assert r["isolation"]["clean"] is True and r["isolation"]["retries"] == 0 and r["isolation"]["reasons"] == []
    env = json.loads((rig.runs / "arm.env.json").read_text())
    s = env["sessions"][0]
    assert s["preflight"]["ok"] is True and s["tool_versions"]["llamacpp"] == rig.rows()[0]["version"]
    assert "snapshot_start" in s and "snapshot_end" in s and s["device"] == "gpu"


def test_a_failed_preflight_stops_before_the_first_problem(rig: Rig, fake_machine) -> None:
    fake_machine.profile = fake_machine.platform_profile = "balanced"
    fake_machine.flush()
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "profile" in str(e.value) and "--tune" in str(e.value)
    assert rig.rows() == [] and rig.calls() == []
    assert json.loads((rig.runs / "arm.env.json").read_text())["sessions"][0]["preflight"]["ok"] is False


def test_allow_noisy_runs_after_a_failed_preflight_and_records_it(rig: Rig, fake_machine) -> None:
    fake_machine.profile = fake_machine.platform_profile = "balanced"
    fake_machine.flush()
    assert rig.run("--allow-noisy") == 0 and len(rig.rows()) == 2
    assert json.loads((rig.runs / "arm.env.json").read_text())["sessions"][0]["preflight_failed_allowed"] is True


def test_tune_sets_performance_for_the_run_and_restores_it(rig: Rig, fake_machine) -> None:
    fake_machine.profile = fake_machine.platform_profile = "balanced"
    fake_machine.flush()
    assert rig.run("--tune") == 0
    assert fake_machine.profile == "balanced"
    sets = [c[2] for c in fake_machine.calls if c[:2] == ["powerprofilesctl", "set"]]
    assert sets == ["performance", "balanced"]
    assert json.loads((rig.runs / "arm.env.json").read_text())["sessions"][0]["tuned"]["previous"] == "balanced"


def test_a_noisy_window_repeats_the_problem_up_to_the_retry_option(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    class Noisy:
        def __init__(self, **kw: object) -> None:
            self.result = {"clean": False, "reasons": ["other-cpu-mean 12.0% > 5%"], "other_cpu_share_max": 0.3}

        def __enter__(self) -> "Noisy":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(isolation, "Window", Noisy)
    assert rig.run("--isolation-retries", "1", limit=1) == 0
    assert len(rig.calls()) == 2                                  # the first answer and one repeat
    r = rig.rows()[0]
    assert r["isolation"]["clean"] is False and r["isolation"]["retries"] == 1 and r["isolation"]["noisy"] is True
    assert r["isolation"]["reasons"] == ["other-cpu-mean 12.0% > 5%"] and r["text"]      # kept and still scored


def test_a_resumed_arm_does_not_preflight_when_nothing_is_left(rig: Rig, fake_machine) -> None:
    assert rig.run() == 0
    calls_before = len(fake_machine.calls)
    assert rig.run() == 0
    assert len(rig.rows()) == 2 and len(fake_machine.calls) == calls_before
