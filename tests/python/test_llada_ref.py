"""tools/ref/llada_ref.cpp checks its inputs and its output (#249).

The tool is built with the host C++ compiler against a stand-in for llama.cpp (tests/python/stubs): a
deterministic tiny "model" whose failures the environment steers. Nothing here loads a model or needs libllama.
The compiler is looked for as c++, g++ or clang++; without one the tests skip, with that reason shown by `-rs`.
LLADA_REF_SOURCE points the build at another copy of the source (used to prove the tests fail on the old tool).
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STUBS = Path(__file__).resolve().parent / "stubs"
SOURCE = Path(os.environ.get("LLADA_REF_SOURCE", ROOT / "tools/ref/llada_ref.cpp"))
COMPILER = next((c for c in (shutil.which("c++"), shutil.which("g++"), shutil.which("clang++")) if c), None)
VOCAB = 8


@pytest.fixture(scope="module")
def tool(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if COMPILER is None:
        pytest.skip("no C++ compiler (c++, g++ or clang++) on PATH: tools/ref/llada_ref.cpp was not built or tested")
    out = tmp_path_factory.mktemp("llada_ref") / "llada_ref"
    p = subprocess.run([COMPILER, "-std=c++17", "-O1", "-Wall", "-I", str(STUBS), str(SOURCE), str(STUBS / "llama_stub.cpp"),
                        "-o", str(out)], capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stderr
    return out


def run(tool: Path, *args: str, tmp: Path, limit_bytes: int | None = None, **env: str) -> subprocess.CompletedProcess[str]:
    def prepare() -> None:
        if limit_bytes is not None:
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)       # a write past the limit fails with EFBIG
            resource.setrlimit(resource.RLIMIT_FSIZE, (limit_bytes, limit_bytes))

    return subprocess.run([str(tool), *args], capture_output=True, text=True, timeout=60, cwd=tmp,
                          env={**os.environ, **env}, preexec_fn=prepare)


def logit(row: int, v: int) -> float:
    return struct.unpack("f", struct.pack("f", ((row * 7 + v * 3) % 11) / 10.0))[0]


def read_rows(path: Path) -> list[list[float]]:
    data = path.read_bytes()
    floats = struct.unpack(f"{len(data) // 4}f", data)
    return [list(floats[i:i + VOCAB]) for i in range(0, len(floats), VOCAB)]


def logits(tool: Path, tmp: Path, tokens: str = "1,2,3", rows: str = "0,2", out: str = "out.f32", threads: str = "2",
           **kw: object) -> subprocess.CompletedProcess[str]:
    return run(tool, "logits", "m.gguf", tokens, rows, str(tmp / out) if not out.startswith("/") else out, threads, tmp=tmp,
               **kw)  # type: ignore[arg-type]


def refused(p: subprocess.CompletedProcess[str], code: int, *words: str) -> None:
    assert p.returncode == code, (p.returncode, p.stderr)
    assert "llada_ref:" in p.stderr, p.stderr
    for w in words:
        assert w in p.stderr, (w, p.stderr)


# logits -----------------------------------------------------------------------------------------------------

def test_logits_writes_the_requested_rows(tool: Path, tmp_path: Path) -> None:
    p = logits(tool, tmp_path)
    assert p.returncode == 0, p.stderr
    assert read_rows(tmp_path / "out.f32") == [[logit(0, v) for v in range(VOCAB)], [logit(2, v) for v in range(VOCAB)]]
    assert "wrote 2 rows x 8 vocab" in p.stderr and not (tmp_path / "out.f32.part").exists()


@pytest.mark.parametrize("tokens", ["1,x,3", "abc", "1,2z", "1.5,2", "1,,2", "1,2,", ",1", "", " 1", "99999999999999999999"])
def test_a_token_that_is_not_an_integer_is_refused(tool: Path, tmp_path: Path, tokens: str) -> None:
    p = logits(tool, tmp_path, tokens=tokens)
    refused(p, 1, "TOKENS")
    assert not (tmp_path / "out.f32").exists()


@pytest.mark.parametrize("rows", ["0,x", "", "1;2"])
def test_a_row_that_is_not_an_integer_is_refused(tool: Path, tmp_path: Path, rows: str) -> None:
    refused(logits(tool, tmp_path, rows=rows), 1, "ROWS")


@pytest.mark.parametrize("rows", ["3", "0,3", "-1", "100"])
def test_a_row_outside_the_token_count_is_refused_and_named(tool: Path, tmp_path: Path, rows: str) -> None:
    p = logits(tool, tmp_path, tokens="1,2,3", rows=rows)
    refused(p, 1, "outside the 3 tokens")
    assert not (tmp_path / "out.f32").exists()


def test_the_last_row_is_accepted(tool: Path, tmp_path: Path) -> None:
    assert logits(tool, tmp_path, rows="2").returncode == 0


@pytest.mark.parametrize("token", ["8", "-1", "1000"])
def test_a_token_outside_the_vocabulary_is_refused(tool: Path, tmp_path: Path, token: str) -> None:
    refused(logits(tool, tmp_path, tokens=f"1,{token},3"), 1, "outside the vocabulary")


@pytest.mark.parametrize("threads", ["x", "0", "-2", "2x", ""])
def test_a_thread_count_that_is_not_a_positive_integer_is_refused(tool: Path, tmp_path: Path, threads: str) -> None:
    refused(logits(tool, tmp_path, threads=threads), 1, "THREADS")


def test_an_unwritable_output_path_is_refused(tool: Path, tmp_path: Path) -> None:
    p = logits(tool, tmp_path, out=str(tmp_path / "no-such-dir" / "out.f32"))
    refused(p, 2, "cannot open")


def test_a_write_that_fails_is_refused_and_leaves_no_file(tool: Path, tmp_path: Path) -> None:
    p = logits(tool, tmp_path, tokens="1,2,3", rows="0,1,2", limit_bytes=20)         # a row is 32 bytes
    refused(p, 2)
    assert "short write" in p.stderr or "could not close" in p.stderr
    assert not (tmp_path / "out.f32").exists() and not (tmp_path / "out.f32.part").exists()


@pytest.mark.parametrize("env", [{"STUB_NAN_ROW": "2"}, {"STUB_INF_ROW": "2"}])
def test_a_non_finite_logit_in_a_requested_row_is_refused_and_leaves_no_file(tool: Path, tmp_path: Path,
                                                                           env: dict[str, str]) -> None:
    p = logits(tool, tmp_path, **env)
    refused(p, 3, "non-finite", "row 2")
    assert not (tmp_path / "out.f32").exists() and not (tmp_path / "out.f32.part").exists()


def test_a_non_finite_logit_in_a_row_that_is_not_requested_does_not_matter(tool: Path, tmp_path: Path) -> None:
    assert logits(tool, tmp_path, rows="0,2", STUB_NAN_ROW="1").returncode == 0


def test_no_logits_from_llama_is_refused(tool: Path, tmp_path: Path) -> None:
    refused(logits(tool, tmp_path, STUB_NO_LOGITS="1"), 3, "no logits")


def test_a_failed_decode_is_refused(tool: Path, tmp_path: Path) -> None:
    refused(logits(tool, tmp_path, STUB_DECODE_FAIL="1"), 3, "llama_decode failed")


def test_a_model_that_does_not_load_is_refused(tool: Path, tmp_path: Path) -> None:
    refused(logits(tool, tmp_path, STUB_LOAD_FAIL="1"), 2, "failed to load")


def test_an_old_output_is_not_replaced_by_a_failed_run(tool: Path, tmp_path: Path) -> None:
    (tmp_path / "out.f32").write_bytes(b"previous")
    assert logits(tool, tmp_path, STUB_NAN_ROW="0").returncode == 3
    assert (tmp_path / "out.f32").read_bytes() == b"previous"


# generate ---------------------------------------------------------------------------------------------------

def generate(tool: Path, tmp: Path, prompt: str = "1,2", gen: str = "4", block: str = "2", steps: str = "4",
             threads: str = "1", **env: str) -> subprocess.CompletedProcess[str]:
    return run(tool, "generate", "m.gguf", prompt, gen, block, steps, threads, tmp=tmp, **env)


def generate_dream(tool: Path, tmp: Path, prompt: str = "1,2", gen: str = "4", steps: str = "4", threads: str = "1",
                   **env: str) -> subprocess.CompletedProcess[str]:
    return run(tool, "generate-dream", "m.gguf", prompt, gen, steps, threads, tmp=tmp, **env)


@pytest.mark.parametrize("call", ["generate", "dream"])
def test_a_good_generation_prints_the_canvas(tool: Path, tmp_path: Path, call: str) -> None:
    p = generate(tool, tmp_path) if call == "generate" else generate_dream(tool, tmp_path)
    assert p.returncode == 0, p.stderr
    ids = [int(x) for x in p.stdout.strip().split(",")]
    assert len(ids) == 6 and ids[:2] == [1, 2]


@pytest.mark.parametrize("field,value", [("gen", "x"), ("gen", "0"), ("gen", "4x"), ("block", "0"), ("block", "x"),
                                         ("steps", "x"), ("steps", "0"), ("threads", "x"), ("threads", "0"),
                                         ("prompt", "1,y")])
def test_generate_refuses_a_number_that_is_not_one(tool: Path, tmp_path: Path, field: str, value: str) -> None:
    p = generate(tool, tmp_path, **{field: value})
    assert p.returncode == 1 and "llada_ref:" in p.stderr, (p.returncode, p.stderr)      # atoi gave 0: a division by zero


@pytest.mark.parametrize("field,value", [("gen", "x"), ("gen", "0"), ("steps", "x"), ("threads", "0"), ("prompt", "")])
def test_generate_dream_refuses_a_number_that_is_not_one(tool: Path, tmp_path: Path, field: str, value: str) -> None:
    p = generate_dream(tool, tmp_path, **{field: value})
    assert p.returncode == 1 and "llada_ref:" in p.stderr, (p.returncode, p.stderr)


def test_a_prompt_token_outside_the_vocabulary_is_refused(tool: Path, tmp_path: Path) -> None:
    refused(generate(tool, tmp_path, prompt="1,50"), 1, "outside the vocabulary")
    refused(generate_dream(tool, tmp_path, prompt="1,50"), 1, "outside the vocabulary")


@pytest.mark.parametrize("mask", ["x", "99", "-1"])
def test_a_mask_id_that_is_not_a_token_is_refused(tool: Path, tmp_path: Path, mask: str) -> None:
    assert generate(tool, tmp_path, STUB_MASK_ID=mask).returncode == 1
    assert generate_dream(tool, tmp_path, STUB_MASK_ID=mask).returncode == 1


def test_a_model_without_a_mask_id_is_refused(tool: Path, tmp_path: Path) -> None:
    refused(generate(tool, tmp_path, STUB_NO_MASK="1"), 1, "mask_token_id")


@pytest.mark.parametrize("call", ["generate", "dream"])
def test_a_non_finite_logit_in_a_row_the_sampler_reads_is_refused(tool: Path, tmp_path: Path, call: str) -> None:
    row = "2" if call == "generate" else "1"                    # canvas row 2 (generate) or row p-1 = 1 (dream)
    p = generate(tool, tmp_path, STUB_NAN_ROW=row) if call == "generate" else generate_dream(tool, tmp_path, STUB_NAN_ROW=row)
    refused(p, 3, "non-finite")
    assert p.stdout == ""


@pytest.mark.parametrize("call", ["generate", "dream"])
def test_generate_with_a_failed_decode_is_refused(tool: Path, tmp_path: Path, call: str) -> None:
    p = generate(tool, tmp_path, STUB_DECODE_FAIL="1") if call == "generate" else generate_dream(tool, tmp_path, STUB_DECODE_FAIL="1")
    refused(p, 3, "llama_decode failed")


def test_a_gen_length_that_is_not_a_multiple_of_the_block_is_refused(tool: Path, tmp_path: Path) -> None:
    assert generate(tool, tmp_path, gen="5", block="2").returncode == 1


def test_usage_errors_exit_non_zero(tool: Path, tmp_path: Path) -> None:
    assert run(tool, tmp=tmp_path).returncode == 1
    assert run(tool, "nonsense", tmp=tmp_path).returncode == 1
    assert run(tool, "logits", "m", tmp=tmp_path).returncode == 1


def test_an_output_path_that_cannot_be_replaced_is_refused_and_leaves_no_part_file(tool: Path, tmp_path: Path) -> None:
    (tmp_path / "taken").mkdir()
    p = logits(tool, tmp_path, out="taken")
    refused(p, 2, "cannot move")
    assert not (tmp_path / "taken.part").exists() and (tmp_path / "taken").is_dir()


def test_a_row_too_large_for_the_buffer_that_fails_to_write_is_a_short_write(tool: Path, tmp_path: Path) -> None:
    """A row of 20 kB goes past stdio's buffer, so fwrite itself (not only fclose) is what fails."""
    p = logits(tool, tmp_path, tokens="1,2,3", rows="0,1", limit_bytes=100, STUB_VOCAB="5000")
    refused(p, 2, "short write")
    assert not (tmp_path / "out.f32").exists() and not (tmp_path / "out.f32.part").exists()
