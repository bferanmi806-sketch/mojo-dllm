"""eval_gsm8k run: a per-problem timeout, and clear refusals for a damaged arm file (#238)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import eval_gsm8k  # noqa: E402
from test_bench_isolation_wiring import _script, gsm8k, live, run_gsm8k  # noqa: E402,F401  (fixtures)


def arm_file(root: Path) -> Path:
    return root / "runs" / "m1.jsonl"


def first_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    assert run_gsm8k(monkeypatch, root) == 0
    lines = arm_file(root).read_text().splitlines()
    assert len(lines) == 2
    return lines


def test_a_truncated_last_line_is_refused_with_the_arm_and_the_line(
    gsm8k: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = first_run(gsm8k, monkeypatch)
    arm_file(gsm8k).write_text(lines[0] + "\n" + lines[1][:60])
    with pytest.raises(SystemExit) as e:
        run_gsm8k(monkeypatch, gsm8k)
    message = str(e.value)
    assert "'m1'" in message and "line 2" in message and "not JSON" in message and "Delete that line" in message


def test_a_row_without_flags_is_refused(gsm8k: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = first_run(gsm8k, monkeypatch)
    row = json.loads(lines[1])
    del row["flags"]
    arm_file(gsm8k).write_text(lines[0] + "\n" + json.dumps(row) + "\n")
    with pytest.raises(SystemExit) as e:
        run_gsm8k(monkeypatch, gsm8k)
    assert "'m1'" in str(e.value) and "line 2" in str(e.value) and "'flags'" in str(e.value)


def test_a_duplicate_index_is_refused(gsm8k: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = first_run(gsm8k, monkeypatch)
    arm_file(gsm8k).write_text("\n".join([*lines, lines[0]]) + "\n")
    with pytest.raises(SystemExit) as e:
        run_gsm8k(monkeypatch, gsm8k)
    assert "line 3" in str(e.value) and "repeats problem" in str(e.value)


def test_a_clean_file_resumes(gsm8k: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_run(gsm8k, monkeypatch)
    assert run_gsm8k(monkeypatch, gsm8k) == 0, "nothing left to answer"


def test_compare_names_a_damaged_arm_file(gsm8k: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines = first_run(gsm8k, monkeypatch)
    arm_file(gsm8k).write_text(lines[0] + "\n" + lines[1][:60])
    with pytest.raises(SystemExit) as e:
        eval_gsm8k.load_arm("m1")
    assert "'m1'" in str(e.value) and "line 2" in str(e.value)


def test_a_hung_binary_is_stopped_by_the_timeout(gsm8k: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _script(gsm8k / "build/mojo-dllm", "#!/usr/bin/env bash\nexec sleep 60\n")
    started = time.monotonic()
    with pytest.raises(SystemExit) as e:
        run_gsm8k(monkeypatch, gsm8k, "--timeout-s", "1")
    assert time.monotonic() - started < 30
    assert "--timeout-s 1" in str(e.value) and "resumes" in str(e.value)
    assert not arm_file(gsm8k).exists(), "a stopped problem is not scored"


def test_the_default_timeout_is_the_engine_runners() -> None:
    import eval_engine

    assert eval_gsm8k.DEFAULT_TIMEOUT_S == eval_engine.DEFAULT_TIMEOUT_S == 1800.0


def test_a_dream_arm_does_not_pass_block_length_and_a_llada_arm_does(gsm8k: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert run_gsm8k(monkeypatch, gsm8k, "--arch", "dream", arm="d1") == 0
    assert run_gsm8k(monkeypatch, gsm8k, arm="l1") == 0
    dream = json.loads((gsm8k / "runs/d1.jsonl").read_text().splitlines()[0])["flags"]
    llada = json.loads((gsm8k / "runs/l1.jsonl").read_text().splitlines()[0])["flags"]
    assert "--block-length" not in dream and "--block-length" in llada
