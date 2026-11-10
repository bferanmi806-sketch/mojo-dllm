"""`eval_gsm8k.py run` and `run_bench.py` run inside bench/isolation.py: a preflight, a window per problem, repeats.

The mojo-dllm binary, nvidia-smi and powerprofilesctl are fake scripts; the machine is conftest.FakeMachine.
Nothing here starts a real benchmark, touches the GPU or changes the real power profile.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import eval_gsm8k  # noqa: E402
import isolation  # noqa: E402
import run_bench  # noqa: E402
from conftest import FakeMachine  # noqa: E402

FAKE_MOJO_RUN = r"""#!/usr/bin/env bash
# fake mojo-dllm: `tokenize` prints ids, `run --json` prints one JSON line.
if [ "$1" = tokenize ]; then printf 'loading\n[1,2,3]\n'; exit 0; fi
printf '{"text": "The answer is 18.", "forward_passes": 4, "generate_s": 1.5, "ms_per_step": 375.0, '
printf '"prompt_tokens": 3, "gen_length": 16, "gemm_ms_per_forward": 1.0, "attention_ms_per_forward": 1.0, '
printf '"lm_head_ms_per_forward": 1.0, "forward_ms": 2.0, "stopped_early": false}\n'
"""
FAKE_PPD = """#!/usr/bin/env bash
state="${FAKE_PPD_STATE:?}"
case "$1" in
  get) cat "$state" ;;
  set) printf '%s\\n' "$2" > "$state" ;;
  *) exit 1 ;;
esac
"""
FAKE_SMI = "#!/usr/bin/env bash\nprintf 'Fake GPU, 595.00, 6144\\n'\n"


def _script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    return path


@pytest.fixture
def live(fake_machine: FakeMachine) -> FakeMachine:
    fake_machine.auto = True
    return fake_machine


def test_fake_scripts_pass_bash_n(tmp_path: Path) -> None:
    for i, body in enumerate((FAKE_MOJO_RUN, FAKE_PPD, FAKE_SMI)):
        _script(tmp_path / f"s{i}", body)


# eval_gsm8k.py run


@pytest.fixture
def gsm8k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live: FakeMachine) -> Path:
    _script(tmp_path / "build/mojo-dllm", FAKE_MOJO_RUN)
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(eval_gsm8k, "ROOT", tmp_path)
    monkeypatch.setattr(eval_gsm8k, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(eval_gsm8k, "PAUSE", tmp_path / "runs/PAUSE")
    return tmp_path


def run_gsm8k(monkeypatch: pytest.MonkeyPatch, root: Path, *extra: str, arm: str = "m1") -> int:
    monkeypatch.setattr(sys, "argv", ["eval_gsm8k.py", "run", "--model", str(root / "model.gguf"), "--arm", arm,
                                     "--device", "cpu", "--limit", "2", "--allow-battery", *extra])
    return eval_gsm8k.main()


def test_the_mojo_arm_records_a_window_per_problem_and_an_env_file(gsm8k: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert run_gsm8k(monkeypatch, gsm8k) == 0
    rows = [json.loads(l) for l in (gsm8k / "runs/m1.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(r["isolation"]["clean"] is True for r in rows)
    assert {"index", "gold", "pred", "correct", "forward_passes", "generate_s", "stopped_early", "text", "flags",
            "version", "model", "power_retries", "on_battery", "power_source"} <= set(rows[0])
    env = json.loads((gsm8k / "runs/m1.env.json").read_text())["sessions"][0]
    assert env["preflight"]["ok"] is True and env["device"] == "cpu" and "snapshot_end" in env


def test_the_mojo_arm_stops_on_a_failed_preflight_and_obeys_allow_noisy(gsm8k: Path, live: FakeMachine,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    live.gpu_apps = []
    live.add_proc(777, 1, "omp", rate=120)
    with pytest.raises(SystemExit) as e:
        run_gsm8k(monkeypatch, gsm8k)
    assert "omp (777)" in str(e.value) and not (gsm8k / "runs/m1.jsonl").exists()
    assert run_gsm8k(monkeypatch, gsm8k, "--allow-noisy", arm="m2") == 0
    rows = [json.loads(l) for l in (gsm8k / "runs/m2.jsonl").read_text().splitlines()]
    assert all("isolation" in r for r in rows)


def test_the_mojo_arm_repeats_a_noisy_problem(gsm8k: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Noisy:
        def __init__(self, **kw: object) -> None:
            self.result = {"clean": False, "reasons": ["swap 4 pages"]}

        def __enter__(self) -> "Noisy":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(isolation, "Window", Noisy)
    assert run_gsm8k(monkeypatch, gsm8k, "--isolation-retries", "2") == 0
    rows = [json.loads(l) for l in (gsm8k / "runs/m1.jsonl").read_text().splitlines()]
    assert [r["isolation"]["retries"] for r in rows] == [2, 2] and rows[0]["isolation"]["reasons"] == ["swap 4 pages"]


def test_the_mojo_arm_tunes_and_restores(gsm8k: Path, live: FakeMachine, monkeypatch: pytest.MonkeyPatch) -> None:
    live.profile = live.platform_profile = "power-saver"
    live.flush()
    assert run_gsm8k(monkeypatch, gsm8k, "--tune") == 0
    assert live.profile == "power-saver"


# run_bench.py


def test_run_bench_keeps_its_quiet_wait_names_and_meaning(live: FakeMachine) -> None:
    assert run_bench.cpu_busy(1.0) == 0.0
    busy, waited = run_bench.wait_for_quiet(0.1, 300)
    assert busy == 0.0 and 2.0 <= waited < 5.0                      # one two-second sample, no second try
    live.extra_busy = 100                                           # 25% of four cores, over the 10% threshold
    busy, waited = run_bench.wait_for_quiet(0.1, 20)
    assert busy == pytest.approx(0.25) and waited > 20              # gave up after max_quiet_wait_s, as before


def test_run_bench_main_records_the_window_and_writes_the_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                    live: FakeMachine) -> None:
    bin_dir = tmp_path / "bin"
    live.profile = live.platform_profile = "balanced"
    _script(bin_dir / "nvidia-smi", FAKE_SMI)
    _script(tmp_path / "build/mojo-dllm", FAKE_MOJO_RUN)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    cfg = {"models": {"gguf": str(model)}, "llama_cpp": str(tmp_path), "diffuse_cpp": str(tmp_path),
           "runtimes": ["mojo-dllm"], "prompts": ["hello"], "gen_length": 16, "steps": 4, "block_length": 4,
           "threads": 2, "busy_threshold": 0.1, "max_quiet_wait_s": 5, "kill_pytest": False}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(run_bench, "ROOT", tmp_path)
    monkeypatch.setattr(run_bench, "mojo_version", lambda: "Mojo 1.1.0 (fake)")
    out = tmp_path / "results" / "x.json"
    monkeypatch.setattr(sys, "argv", ["run_bench.py", "--config", str(tmp_path / "config.json"), "--reps", "1",
                                     "--out", str(out), "--allow-battery"])
    assert run_bench.main() == 0
    doc = json.loads(out.read_text())
    run = doc["runs"][0]
    assert run["ok"] is True and run["isolation"]["clean"] is True and run["cpu_busy_before"] == 0.0
    assert {"cpu_busy_before", "waited_s"} <= set(run)
    assert live.profile == "balanced"                                    # restored, as before
    env = json.loads((tmp_path / "results/env/x.env.json").read_text())["sessions"][0]
    assert env["preflight"]["ok"] is True and env["thresholds"]["cpu_busy_max"] == 0.1
    assert list((tmp_path / "results").glob("*.json")) == [out]          # bench_table reads results/*.json
