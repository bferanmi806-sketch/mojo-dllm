"""bench/engines/hf.py: the authors' own LLaDA and Dream sampling code as a GSM8K engine.

Nothing here loads weights, imports torch or transformers, or touches the GPU. The LLaDA checkout is a temporary git
repository whose `generate.py` returns canned token ids, the Dream model is a stub with a `diffusion_generate` method,
and the tokenizers decode canned ids to text. `hf.load_model` is replaced by a function that hands back the stubs;
`test_load_model_*` run the real one against fake `torch` and `transformers` modules to pin the arguments it passes.

Entry-point signatures were read from the sources (see the module docstring of bench/engines/hf.py):
ML-GSAI/LLaDA generate.py at 9182493 and Dream-org/Dream-v0-Instruct-7B generation_utils.py at 05334cb.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import engines  # noqa: E402
import eval_engine  # noqa: E402
import eval_gsm8k  # noqa: E402
from engines import hf  # noqa: E402

REAL_LOAD_NEED = hf.HF.load_need_gb
REAL_AFTER_LOAD = hf.HF.ram_gb_after_load
SHA = "05334cb9faaf763692dcf9d8737c642be2b2a6ae"
ANSWER = "The answer is 18."
RECORD_KEYS = {"index", "gold", "pred", "correct", "forward_passes", "generate_s", "stopped_early", "text",
               "flags", "version", "model", "power_retries", "on_battery", "power_source", "isolation"}
VERSIONS = {"transformers": "4.38.2", "torch": "2.2.0", "bitsandbytes": None}


def ids_of(text: str) -> list[int]:
    return [ord(c) for c in text]


def text_of(ids: list[int]) -> str:
    return "".join(chr(i) for i in ids)


class Row(list):
    def tolist(self) -> list:
        return list(self)

    def __getitem__(self, k):  # type: ignore[override]
        v = list.__getitem__(self, k)
        return Row(v) if isinstance(k, slice) else v


class T:
    """A tensor stand-in: rows of ints, `.to`, `.shape`, `[rows, cols]`, `[row]`."""

    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = [list(r) for r in rows]
        self.device: str | None = None

    def to(self, device: str) -> "T":
        self.device = device
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, k):
        if isinstance(k, tuple):
            rs, cs = k
            return T([r[cs] for r in self.rows[rs]])
        return Row(self.rows[k])

    def with_columns(self, ids: list[int]) -> "T":
        return T([r + ids for r in self.rows])


class Param:
    def __init__(self, kind: str) -> None:
        self.device = SimpleNamespace(type=kind)


class Tok:
    """Both families' tokenizer: a prompt becomes five ids, canned ids decode to text."""

    pad_token_id = 1
    padding_side = "right"
    eos_token = "<|endoftext|>"

    def __init__(self) -> None:
        self.template_kw: list[dict] = []
        self.decode_kw: list[dict] = []

    def apply_chat_template(self, messages, **kw):
        self.template_kw.append(kw)
        assert messages[0]["role"] == "user" and "Solve the problem" in messages[0]["content"]
        if kw.get("tokenize") is False:
            return "<chat>" + messages[0]["content"]
        return SimpleNamespace(input_ids=T([[1, 2, 3, 4, 5]]), attention_mask=T([[1, 1, 1, 1, 1]]))

    def __call__(self, text, **kw):
        assert text.startswith("<chat>") and kw.get("add_special_tokens") is False
        return {"input_ids": T([[1, 2, 3, 4, 5]]), "attention_mask": T([[1, 1, 1, 1, 1]])}

    def batch_decode(self, out, skip_special_tokens=False):
        self.decode_kw.append({"skip_special_tokens": skip_special_tokens})
        return [text_of(r) for r in out.rows]

    def decode(self, ids, **kw):
        self.decode_kw.append(kw)
        return text_of(ids)


class DreamStub:
    """`diffusion_generate` as Dream's remote code exposes it; every call is kept in `calls`."""

    def __init__(self, answer: str = ANSWER, device: str = "cuda") -> None:
        self.calls: list[dict] = []
        self.answer = answer
        self.kind = device

    def parameters(self):
        return iter([Param(self.kind)])

    def diffusion_generate(self, inputs, **kw):
        self.calls.append(dict(kw, n_prompt=inputs.shape[1]))
        return SimpleNamespace(sequences=inputs.with_columns(ids_of(self.answer)))


class LLaDAStub:
    def __init__(self, device: str = "cuda") -> None:
        self.kind = device

    def parameters(self):
        return iter([Param(self.kind)])


GENERATE_PY = '''
import json, os, time

def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, **more):
    with open(os.environ["FAKE_GEN_LOG"], "a") as f:
        f.write(json.dumps(dict(steps=steps, gen_length=gen_length, block_length=block_length,
                                temperature=temperature, cfg_scale=cfg_scale, remasking=remasking,
                                mask_id=mask_id, n_prompt=prompt.shape[1],
                                has_mask=attention_mask is not None)) + "\\n")
    mode = os.environ.get("FAKE_MODE", "ok")
    if mode == "raise":
        raise RuntimeError("CUDA out of memory (stub)")
    if mode == "sleep":
        time.sleep(30)
    return prompt.with_columns([ord(c) for c in os.environ.get("FAKE_ANSWER", "The answer is 18.")])
'''

OLD_GENERATE_PY = "def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.):\n    return prompt\n"


def _git(repo: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    p = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def make_llada_repo(path: Path, body: str = GENERATE_PY, commit: bool = True) -> Path:
    path.mkdir(parents=True)
    (path / "generate.py").write_text(textwrap.dedent(body))
    if commit:
        _git(path, "init", "-q")
        _git(path, "add", "generate.py")
        _git(path, "commit", "-q", "-m", "pin")
    return path


class Rig(SimpleNamespace):
    def run(self, *extra: str, family: str = "llada", device: str = "gpu", limit: int = 2, arm: str = "arm",
            revision: str | None = SHA, quant: str | None = None, gen_length: str = "32",
            llada_repo: bool = True, steps: str | None = None, timeout: str | None = None) -> int:
        model = {"llada": "GSAI-ML/LLaDA-8B-Instruct", "dream": "Dream-org/Dream-v0-Instruct-7B"}.get(family, family)
        argv = ["run", "--engine", "hf", "--model", model, "--arm", arm, "--device", device,
                "--gen-length", gen_length, "--limit", str(limit), "--allow-battery"]
        if revision is not None:
            argv += ["--model-revision", revision]
        if quant is not None:
            argv += ["--quant", quant]
        if llada_repo and family == "llada":
            argv += ["--llada-repo", str(self.repo)]
        if steps is not None:
            argv += ["--steps", steps]
        if timeout is not None:
            argv += ["--timeout-s", timeout]
        return eval_engine.main(argv + list(extra))

    def rows(self, arm: str = "arm") -> list[dict]:
        path = self.runs / f"{arm}.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def gen_calls(self) -> list[dict]:
        p = self.root / "gen.log"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    repo = make_llada_repo(tmp_path / "LLaDA")
    runs = tmp_path / "runs"
    (tmp_path / "bench" / "eval").mkdir(parents=True)
    monkeypatch.setattr(eval_gsm8k, "RUNS", runs)
    monkeypatch.setattr(eval_gsm8k, "PAUSE", runs / "PAUSE")
    monkeypatch.setattr(eval_gsm8k, "ROOT", tmp_path)
    monkeypatch.setenv("FAKE_GEN_LOG", str(tmp_path / "gen.log"))
    for k in ("FAKE_MODE", "FAKE_ANSWER"):
        monkeypatch.delenv(k, raising=False)
    dream = DreamStub()
    loads: list[hf.Spec] = []
    toks: list[Tok] = []

    def load_model(spec: hf.Spec):
        loads.append(spec)
        tok = Tok()
        toks.append(tok)
        return (dream if spec.family == "dream" else LLaDAStub()), tok

    monkeypatch.setattr(hf, "load_model", load_model)
    monkeypatch.setattr(hf, "package_version", lambda name: VERSIONS[name])
    # a machine with plenty of memory for the load guard; the guard itself is pinned by the test_ram_* tests
    monkeypatch.setattr(hf, "mem_available_gb", lambda machine=None: 100.0)
    return Rig(repo=repo, runs=runs, root=tmp_path, dream=dream, loads=loads, toks=toks)


# registry, lazy import


def test_hf_is_registered_and_satisfies_the_protocol() -> None:
    assert engines.get("hf") is hf.HF
    assert "hf" in engines.names()
    args = SimpleNamespace(hf_model=None, llada_repo=None, quant="none", model_revision=None, allow_unpinned=False)
    assert isinstance(hf.HF(args), engines.Engine)


def test_importing_the_adapter_imports_neither_torch_nor_transformers(tmp_path: Path) -> None:
    # decoy packages that record being imported: the frozen environment has neither, and a top-level import
    # of either would break CI, so the adapter must not run them before a model is loaded
    for name in ("torch", "transformers", "bitsandbytes"):
        (tmp_path / f"{name}.py").write_text("raise SystemExit('imported %s')\n" % name)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import engines.hf, engines; "
            "assert not {'torch', 'transformers', 'bitsandbytes'} & set(sys.modules), sorted(sys.modules); "
            "print('lazy ok')")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run([sys.executable, "-c", code, str(ROOT / "bench")], capture_output=True, text=True,
                       cwd=tmp_path, env=dict(env, PYTHONPATH=str(tmp_path)))
    assert p.returncode == 0 and "lazy ok" in p.stdout, p.stdout + p.stderr


# a LLaDA run


def test_llada_record_has_exactly_the_keys_of_the_other_arms(rig: Rig) -> None:
    assert rig.run() == 0
    rows = rig.rows()
    assert len(rows) == 2
    assert all(set(r) == RECORD_KEYS for r in rows)


def test_llada_record_values(rig: Rig) -> None:
    assert rig.run() == 0
    r = rig.rows()[0]
    assert r["model"] == "LLaDA-8B-Instruct"          # the basename of --model, as eval_gsm8k records it
    assert r["text"] == ANSWER
    assert r["forward_passes"] == 32                  # steps; the authors' loop reports no count of its own
    assert r["stopped_early"] is None
    assert isinstance(r["generate_s"], float) and r["generate_s"] >= 0
    f = r["flags"]
    assert f["engine"] == "hf" and f["family"] == "llada" and f["weights"] == "bf16"
    assert f["forward_passes_measured"] is False


def test_llada_calls_the_authors_generate_with_the_like_for_like_settings(rig: Rig) -> None:
    assert rig.run() == 0
    calls = rig.gen_calls()
    assert len(calls) == 2
    for c in calls:
        assert (c["steps"], c["gen_length"], c["block_length"]) == (32, 32, 32)   # steps = generation length
        assert c["temperature"] == 0 and c["cfg_scale"] == 0 and c["remasking"] == "low_confidence"
        assert c["mask_id"] == 126336 and c["n_prompt"] == 5 and c["has_mask"] is True


def test_llada_block_length_and_steps_come_from_the_runner_options(rig: Rig) -> None:
    assert rig.run("--block-length", "16", gen_length="32", steps="16") == 0
    c = rig.gen_calls()[0]
    assert (c["steps"], c["gen_length"], c["block_length"]) == (16, 32, 16)
    assert rig.rows()[0]["flags"]["block_length"] == 16


def test_llada_flags_record_the_sampler_settings(rig: Rig) -> None:
    assert rig.run() == 0
    f = rig.rows()[0]["flags"]
    assert f["remasking"] == "low_confidence" and f["temperature"] == 0.0 and f["cfg_scale"] == 0.0
    assert (f["gen_length"], f["steps"], f["block_length"], f["mask_id"]) == (32, 32, 32, 126336)
    assert f["hf_model"] == "GSAI-ML/LLaDA-8B-Instruct" and f["quant"] == "none" and f["device"] == "gpu"


def test_the_tokenizer_is_set_to_left_padding_as_the_authors_do(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Tok] = []
    real = hf.load_model

    def load(spec):
        m, t = real(spec)
        seen.append(t)
        return m, t

    monkeypatch.setattr(hf, "load_model", load)
    assert rig.run() == 0
    assert seen[0].padding_side == "left"


# versions


def test_version_records_the_revision_the_checkout_and_the_libraries(rig: Rig) -> None:
    assert rig.run() == 0
    v = json.loads(rig.rows()[0]["version"])
    assert v == {"model_revision": SHA, "llada_commit": _git(rig.repo, "rev-parse", "HEAD"),
                 "transformers": "4.38.2", "torch": "2.2.0", "bitsandbytes": None,
                 "model_source": "hub", "model_dir": None, "model_revision_verified": True}
    assert len(v["llada_commit"]) == 40


def test_version_records_null_for_a_revision_nobody_gave(rig: Rig) -> None:
    assert rig.run("--allow-unpinned", revision=None) == 0
    assert json.loads(rig.rows()[0]["version"])["model_revision"] is None
    assert rig.loads[0].revision is None              # and nothing was guessed for the download either


def test_the_revision_reaches_the_loader(rig: Rig) -> None:
    assert rig.run() == 0
    assert rig.loads[0].revision == SHA and rig.loads[0].model_id == "GSAI-ML/LLaDA-8B-Instruct"


def test_version_records_the_bitsandbytes_version_when_it_is_installed(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(VERSIONS, "bitsandbytes", "0.43.1")
    assert rig.run(quant="4bit") == 0
    v = json.loads(rig.rows()[0]["version"])
    assert v["bitsandbytes"] == "0.43.1"
    assert rig.rows()[0]["flags"]["weights"] == "bnb-nf4" and rig.loads[0].quant == "4bit"


def test_a_resume_with_another_revision_is_refused(rig: Rig) -> None:
    assert rig.run(limit=1) == 0
    with pytest.raises(SystemExit) as e:
        rig.run(limit=2, revision="a" * 40)
    assert "configuration" in str(e.value) or "version" in str(e.value)
    assert len(rig.rows()) == 1


def test_a_git_dir_in_the_environment_does_not_redirect_the_checkout_commit(rig: Rig, tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    other = make_llada_repo(tmp_path / "other")        # a pre-commit hook exports GIT_DIR of ITS repo
    (other / "x").write_text("x")
    _git(other, "add", "x")
    _git(other, "commit", "-q", "-m", "second")
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    assert rig.run() == 0
    got = json.loads(rig.rows()[0]["version"])["llada_commit"]
    monkeypatch.delenv("GIT_DIR")
    assert got == _git(rig.repo, "rev-parse", "HEAD") != _git(other, "rev-parse", "HEAD")


# a Dream run


def test_dream_record_and_the_authors_entry_point(rig: Rig) -> None:
    assert rig.run(family="dream", gen_length="32") == 0
    rows = rig.rows()
    assert all(set(r) == RECORD_KEYS for r in rows) and rows[0]["text"] == ANSWER
    assert rows[0]["model"] == "Dream-v0-Instruct-7B"
    call = rig.dream.calls[0]
    assert call["max_new_tokens"] == 32 and call["steps"] == 32          # steps = generation length
    assert call["temperature"] == 0 and call["alg"] == "entropy" and call["alg_temp"] == 0
    assert call.get("top_p") is None and call.get("top_k") is None
    assert call["return_dict_in_generate"] is True and call["attention_mask"] is not None
    v = json.loads(rows[0]["version"])
    assert v["llada_commit"] is None and v["model_revision"] == SHA


def test_dream_flags_record_the_departure_from_the_readme(rig: Rig) -> None:
    assert rig.run(family="dream", gen_length="32") == 0
    f = rig.rows()[0]["flags"]
    assert f["family"] == "dream" and f["alg"] == "entropy" and f["alg_temp"] == 0.0 and f["top_p"] is None
    assert f["readme_example"] == {"temperature": 0.2, "top_p": 0.95, "steps": 512, "alg": "entropy", "alg_temp": 0.0}
    assert f["library_default_alg"] == "origin"
    assert f["departs_from_readme"] == ["steps", "temperature", "top_p"]


def test_dream_at_the_readme_step_count_does_not_claim_a_step_departure(rig: Rig) -> None:
    assert rig.run(family="dream", gen_length="512") == 0
    assert rig.rows()[0]["flags"]["departs_from_readme"] == ["temperature", "top_p"]


def test_dream_text_stops_at_the_end_of_text_token(rig: Rig) -> None:
    rig.dream.answer = ANSWER + "<|endoftext|>garbage after the end"
    assert rig.run(family="dream", limit=1) == 0
    assert rig.rows()[0]["text"] == ANSWER


# refusals


@pytest.mark.parametrize("family", ["llada", "dream"])
def test_a_raising_entry_point_stops_the_run_and_stores_nothing(rig: Rig, monkeypatch: pytest.MonkeyPatch,
                                                               family: str) -> None:
    if family == "llada":
        monkeypatch.setenv("FAKE_MODE", "raise")
    else:
        def boom(*a, **k):
            raise RuntimeError("CUDA out of memory (stub)")
        rig.dream.diffusion_generate = boom           # type: ignore[method-assign]
    with pytest.raises(SystemExit) as e:
        rig.run(family=family)
    msg = str(e.value)
    assert "hf failed" in msg and "RuntimeError" in msg and "out of memory" in msg and "problem" in msg
    assert rig.rows() == []


def test_a_generation_over_the_timeout_stops_the_run_and_names_the_problem(rig: Rig,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_MODE", "sleep")
    with pytest.raises(SystemExit) as e:
        rig.run(timeout="0.3")
    assert "timed out after 0.3 s" in str(e.value) and "problem" in str(e.value)
    assert rig.rows() == []


def test_a_load_that_raises_is_refused_before_the_first_problem(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    def load(spec):
        raise MemoryError("not enough memory (stub)")

    monkeypatch.setattr(hf, "load_model", load)
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "hf:" in str(e.value) and "MemoryError" in str(e.value)
    assert rig.rows() == [] and rig.gen_calls() == []


def test_a_model_that_is_not_on_the_requested_device_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf, "load_model", lambda spec: (LLaDAStub(device="cpu"), Tok()))
    with pytest.raises(SystemExit) as e:
        rig.run(device="gpu")
    assert "not on the GPU" in str(e.value) and "cpu" in str(e.value)
    assert rig.gen_calls() == []


def test_a_model_with_no_parameters_is_not_taken_for_one_on_the_device(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    class Empty:
        def parameters(self):
            return iter(())

    monkeypatch.setattr(hf, "load_model", lambda spec: (Empty(), Tok()))
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "not on the GPU" in str(e.value)


def test_the_cpu_device_needs_cpu_parameters(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf, "load_model", lambda spec: (LLaDAStub(device="cpu"), Tok()))
    assert rig.run(device="cpu", limit=1) == 0
    monkeypatch.setattr(hf, "load_model", lambda spec: (LLaDAStub(device="cuda"), Tok()))
    with pytest.raises(SystemExit) as e:
        rig.run(device="cpu", arm="other", limit=1)
    assert "not on the CPU" in str(e.value)


def test_a_pad_token_equal_to_the_mask_token_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    class BadTok(Tok):
        pad_token_id = 126336

    monkeypatch.setattr(hf, "load_model", lambda spec: (LLaDAStub(), BadTok()))
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "pad token" in str(e.value) and rig.gen_calls() == []


# options


def _refused(rig: Rig, fragment: str, **kw) -> None:
    extra = kw.pop("extra", [])
    with pytest.raises(SystemExit) as e:
        rig.run(*extra, **kw)
    assert fragment in str(e.value), str(e.value)
    assert rig.rows() == [] and rig.loads == [] and rig.gen_calls() == []     # refused before any load


def test_llada_without_a_checkout_is_refused(rig: Rig) -> None:
    _refused(rig, "--llada-repo", llada_repo=False)


def test_a_checkout_without_generate_py_is_refused(rig: Rig, tmp_path: Path) -> None:
    (rig.repo / "generate.py").unlink()
    _refused(rig, "generate.py")


def test_a_checkout_whose_generate_lacks_a_parameter_is_refused(rig: Rig, tmp_path: Path) -> None:
    rig.repo = make_llada_repo(tmp_path / "old-llada", OLD_GENERATE_PY)
    _refused(rig, "attention_mask")


def test_a_checkout_that_is_not_a_git_repository_is_refused(rig: Rig, tmp_path: Path) -> None:
    rig.repo = make_llada_repo(tmp_path / "tarball", commit=False)
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "commit" in str(e.value) and rig.rows() == []


def test_a_checkout_that_is_not_a_git_repository_is_refused_before_any_load(rig: Rig, tmp_path: Path) -> None:
    rig.repo = make_llada_repo(tmp_path / "tarball2", commit=False)
    with pytest.raises(SystemExit):
        rig.run()
    assert rig.gen_calls() == []


def test_4bit_on_the_cpu_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(VERSIONS, "bitsandbytes", "0.43.1")     # installed, so only the device can refuse
    _refused(rig, "--quant 4bit needs --device gpu", quant="4bit", device="cpu")


def test_4bit_without_bitsandbytes_is_refused(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(VERSIONS, "bitsandbytes", None)
    _refused(rig, "bitsandbytes", quant="4bit")


def test_a_revision_that_is_not_a_full_commit_sha_is_refused(rig: Rig) -> None:
    for bad in ("main", "05334cb", SHA.upper(), SHA[:-1] + "g"):
        _refused(rig, "40", revision=bad)


def test_a_model_of_neither_family_is_refused(rig: Rig) -> None:
    _refused(rig, "LLaDA or Dream", family="meta-llama/Llama-3-8B")


def test_llada_needs_a_block_length_that_divides_the_generation_length(rig: Rig) -> None:
    _refused(rig, "block length", extra=["--block-length", "24"])


def test_extra_engine_flags_are_refused(rig: Rig) -> None:
    _refused(rig, "extra", extra=["--", "--top-p", "0.9"])


def test_unknown_quant_is_rejected_by_the_parser(rig: Rig) -> None:
    with pytest.raises(SystemExit):
        rig.run(quant="8bit")


# the real loader, against fake torch and transformers modules


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def to(self, d):
        self.calls.append(f"to:{d}")
        return self

    def eval(self):
        self.calls.append("eval")
        return self


@pytest.fixture
def fake_libs(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    seen = SimpleNamespace(model_kw=None, tok_kw=None, bnb_kw=None, model=FakeModel(), cuda=True)

    class AutoModel:
        @staticmethod
        def from_pretrained(name, **kw):
            seen.model_kw = dict(kw, name=name)
            return seen.model

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kw):
            seen.tok_kw = dict(kw, name=name)
            return Tok()

    class BitsAndBytesConfig:
        def __init__(self, **kw):
            seen.bnb_kw = kw

    torch = SimpleNamespace(bfloat16="bf16-dtype", cuda=SimpleNamespace(is_available=lambda: seen.cuda))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=AutoModel, AutoTokenizer=AutoTokenizer, BitsAndBytesConfig=BitsAndBytesConfig))
    return seen


def _spec(**kw) -> hf.Spec:
    base = dict(family="llada", model_id="GSAI-ML/LLaDA-8B-Instruct", revision=SHA, quant="none", device="gpu")
    return hf.Spec(**{**base, **kw})


def test_load_model_bf16_on_the_gpu(fake_libs: SimpleNamespace) -> None:
    model, tok = hf.load_model(_spec())
    kw = fake_libs.model_kw
    assert kw["trust_remote_code"] is True and kw["revision"] == SHA and kw["torch_dtype"] == "bf16-dtype"
    assert kw["low_cpu_mem_usage"] is True
    assert "quantization_config" not in kw
    assert fake_libs.tok_kw["revision"] == SHA and fake_libs.tok_kw["trust_remote_code"] is True
    assert model.calls == ["to:cuda", "eval"]


def test_load_model_4bit_is_nf4_with_bf16_compute_on_gpu_zero(fake_libs: SimpleNamespace) -> None:
    model, _ = hf.load_model(_spec(quant="4bit", family="dream", model_id="Dream-org/Dream-v0-Instruct-7B"))
    assert fake_libs.bnb_kw == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bf16-dtype"}
    assert fake_libs.model_kw["device_map"] == {"": 0} and "quantization_config" in fake_libs.model_kw
    assert fake_libs.model_kw["low_cpu_mem_usage"] is True
    assert "to:cuda" not in model.calls and model.calls == ["eval"]       # a 4-bit model cannot be moved


def test_load_model_on_the_cpu(fake_libs: SimpleNamespace) -> None:
    model, _ = hf.load_model(_spec(device="cpu"))
    assert model.calls == ["to:cpu", "eval"]


def test_load_model_refuses_a_gpu_run_without_cuda(fake_libs: SimpleNamespace) -> None:
    fake_libs.cuda = False
    with pytest.raises(engines.EngineError) as e:
        hf.load_model(_spec())
    assert "CUDA" in str(e.value) and fake_libs.model_kw is None


def test_load_model_without_a_pinned_revision_passes_none_not_a_guess(fake_libs: SimpleNamespace) -> None:
    hf.load_model(_spec(revision=None))
    assert fake_libs.model_kw["revision"] is None and fake_libs.tok_kw["revision"] is None


# compare --across-weights


def _arm(runs: Path, name: str, correct: list[bool], model: str) -> None:
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / f"{name}.jsonl").open("w") as f:
        for i, c in enumerate(correct):
            f.write(json.dumps(dict(index=i, correct=c, forward_passes=None if i else 4, generate_s=1.0 + i,
                                    flags={"k": 1}, version="v", model=model)) + "\n")


def _doc(rig: Rig, a: str, b: str) -> dict:
    return json.loads(next((rig.root / "bench" / "eval").glob(f"*-{a}-vs-{b}.json")).read_text())


def test_compare_still_refuses_different_models_by_default(rig: Rig) -> None:
    _arm(rig.runs, "gguf", [True, False], "LLaDA-8B-Instruct.Q4_K_M.gguf")
    _arm(rig.runs, "hf", [True, True], "LLaDA-8B-Instruct")
    with pytest.raises(SystemExit) as e:
        eval_gsm8k.cmd_compare("gguf", "hf")
    assert "different models" in str(e.value) and "--across-weights" in str(e.value)
    assert list((rig.root / "bench" / "eval").glob("*.json")) == []


def test_compare_across_weights_records_both_models_and_the_flag(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    _arm(rig.runs, "gguf", [True, False], "LLaDA-8B-Instruct.Q4_K_M.gguf")
    _arm(rig.runs, "hf", [True, True], "LLaDA-8B-Instruct")
    assert eval_gsm8k.cmd_compare("gguf", "hf", across_weights=True) == 0
    doc = _doc(rig, "gguf", "hf")
    assert doc["weights_differ"] is True
    assert doc["models"] == {"gguf": "LLaDA-8B-Instruct.Q4_K_M.gguf", "hf": "LLaDA-8B-Instruct"}
    assert doc["accuracy"] == {"gguf": 0.5, "hf": 1.0}
    assert '"weights_differ": true' in capsys.readouterr().out


def test_compare_across_weights_on_one_model_says_the_weights_do_not_differ(rig: Rig) -> None:
    _arm(rig.runs, "a", [True, False], "m.gguf")
    _arm(rig.runs, "b", [True, True], "m.gguf")
    assert eval_gsm8k.cmd_compare("a", "b", across_weights=True) == 0
    assert _doc(rig, "a", "b")["weights_differ"] is False


def test_compare_without_the_flag_on_one_model_writes_weights_differ_false(rig: Rig) -> None:
    _arm(rig.runs, "a", [True, False], "m.gguf")
    _arm(rig.runs, "b", [True, True], "m.gguf")
    assert eval_gsm8k.cmd_compare("a", "b") == 0
    doc = _doc(rig, "a", "b")
    assert doc["weights_differ"] is False and doc["models"] == {"a": "m.gguf", "b": "m.gguf"}


def test_compare_across_weights_keeps_the_null_tolerant_forward_pass_mean(rig: Rig) -> None:
    _arm(rig.runs, "gguf", [True, False], "x.gguf")
    _arm(rig.runs, "hf", [True, True], "x")
    assert eval_gsm8k.cmd_compare("gguf", "hf", across_weights=True) == 0
    doc = _doc(rig, "gguf", "hf")
    assert doc["forward_passes_mean"] == {"gguf": 4.0, "hf": 4.0} and doc["forward_passes_rows"] == {"gguf": 1, "hf": 1}


def test_the_across_weights_flag_reaches_compare_from_the_command_line(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    _arm(rig.runs, "gguf", [True, False], "x.gguf")
    _arm(rig.runs, "hf", [True, True], "x")
    monkeypatch.setattr(sys, "argv", ["eval_gsm8k.py", "compare", "gguf", "hf", "--across-weights"])
    assert eval_gsm8k.main() == 0
    assert _doc(rig, "gguf", "hf")["weights_differ"] is True
    monkeypatch.setattr(sys, "argv", ["eval_gsm8k.py", "compare", "gguf", "hf"])
    with pytest.raises(SystemExit):
        eval_gsm8k.main()


# chat template and decode arguments the authors use


def test_llada_asks_for_the_generation_prompt_and_skips_special_tokens(rig: Rig) -> None:
    assert rig.run(limit=1) == 0
    tok = rig.toks[0]
    assert tok.template_kw == [{"add_generation_prompt": True, "tokenize": False}]
    assert tok.decode_kw == [{"skip_special_tokens": True}]


def test_dream_asks_for_the_generation_prompt(rig: Rig) -> None:
    assert rig.run(family="dream", limit=1) == 0
    kw = rig.toks[0].template_kw[0]
    assert kw["add_generation_prompt"] is True and kw["return_dict"] is True and kw["return_tensors"] == "pt"


def test_dream_without_an_eos_token_is_refused_instead_of_split_on_whitespace(rig: Rig,
                                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    class NoEos(Tok):
        eos_token = None

    monkeypatch.setattr(hf, "load_model", lambda spec: (rig.dream, NoEos()))
    with pytest.raises(SystemExit) as e:
        rig.run(family="dream", limit=1)
    assert "end-of-text" in str(e.value) and rig.rows() == []


# remote code: an unpinned hub repository is refused


def test_a_hub_model_without_a_revision_is_refused(rig: Rig) -> None:
    for family in ("llada", "dream"):
        with pytest.raises(SystemExit) as e:
            rig.run(family=family, revision=None)
        assert "--model-revision" in str(e.value) and "--allow-unpinned" in str(e.value)
    assert rig.rows() == [] and rig.loads == []


def test_allow_unpinned_runs_warns_and_records_it(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:
    assert rig.run("--allow-unpinned", revision=None) == 0
    assert "not pinned" in capsys.readouterr().err
    assert rig.rows()[0]["flags"]["unpinned"] is True
    assert json.loads(rig.rows()[0]["version"])["model_revision"] is None


def test_a_pinned_run_does_not_say_unpinned(rig: Rig) -> None:
    assert rig.run() == 0
    assert rig.rows()[0]["flags"]["unpinned"] is False


def test_a_local_directory_needs_no_revision_and_is_recorded_as_unverified(rig: Rig, tmp_path: Path) -> None:
    d = tmp_path / "LLaDA-local"
    d.mkdir()
    with (d / "model.safetensors").open("wb") as f:
        f.truncate(1_000_000_000)
    assert rig.run("--hf-model", str(d), revision=None) == 0
    v = json.loads(rig.rows()[0]["version"])
    assert v["model_source"] == "local" and v["model_dir"] == str(d) and v["model_revision"] is None
    assert v["model_revision_verified"] is False
    assert rig.rows()[0]["flags"]["unpinned"] is False


def test_a_hub_model_records_its_source(rig: Rig) -> None:
    assert rig.run(limit=1) == 0
    v = json.loads(rig.rows()[0]["version"])
    assert v["model_source"] == "hub" and v["model_dir"] is None and v["model_revision_verified"] is True


# a dirty LLaDA checkout


def test_a_locally_edited_checkout_is_recorded_as_dirty(rig: Rig) -> None:
    (rig.repo / "generate.py").write_text((rig.repo / "generate.py").read_text() + "\n# local edit\n")
    assert rig.run(limit=1) == 0
    sha = _git(rig.repo, "rev-parse", "HEAD")
    assert json.loads(rig.rows()[0]["version"])["llada_commit"] == sha + "-dirty"


def test_an_untracked_file_in_the_checkout_does_not_make_it_dirty(rig: Rig) -> None:
    (rig.repo / "notes.txt").write_text("x")
    assert rig.run(limit=1) == 0
    assert json.loads(rig.rows()[0]["version"])["llada_commit"] == _git(rig.repo, "rev-parse", "HEAD")


# quantization is part of the weights identity


def test_a_bf16_arm_and_a_4bit_arm_of_one_model_differ_in_weights(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(VERSIONS, "bitsandbytes", "0.43.1")
    assert rig.run(arm="bf16", limit=2) == 0
    assert rig.run(arm="q4", limit=2, quant="4bit") == 0
    assert rig.rows("bf16")[0]["model"] == "LLaDA-8B-Instruct"
    assert rig.rows("q4")[0]["model"] == "LLaDA-8B-Instruct|4bit"
    with pytest.raises(SystemExit) as e:
        eval_gsm8k.cmd_compare("bf16", "q4")
    assert "different models" in str(e.value)
    assert eval_gsm8k.cmd_compare("bf16", "q4", across_weights=True) == 0
    assert _doc(rig, "bf16", "q4")["weights_differ"] is True


def test_two_bf16_arms_of_one_model_do_not_differ_in_weights(rig: Rig) -> None:
    assert rig.run(arm="a") == 0
    assert rig.run(arm="b") == 0
    assert eval_gsm8k.cmd_compare("a", "b") == 0
    assert _doc(rig, "a", "b")["weights_differ"] is False


# the memory the preflight asks for


GIB = 1.073741824


def _hf(**kw) -> hf.HF:
    base = dict(hf_model=None, llada_repo=None, quant="none", model_revision=None, allow_unpinned=False)
    return hf.HF(SimpleNamespace(**{**base, **kw}))


def _cfg(model: str) -> engines.GenConfig:
    return engines.GenConfig(model=model)


def test_load_need_bf16_is_one_quarter_over_the_weights_plus_three_gb() -> None:
    assert REAL_LOAD_NEED(_hf(), _cfg("GSAI-ML/LLaDA-8B-Instruct")) == pytest.approx(14.93 * GIB * 1.25 + 3, abs=0.01)
    assert REAL_LOAD_NEED(_hf(), _cfg("Dream-org/Dream-v0-Instruct-7B")) == pytest.approx(14.19 * GIB * 1.25 + 3, abs=0.01)


def test_ram_4bit_is_the_weights_over_three_and_a_half_plus_three_gb() -> None:
    assert REAL_LOAD_NEED(_hf(quant="4bit"), _cfg("GSAI-ML/LLaDA-8B-Instruct")) == pytest.approx(14.93 * GIB / 3.5 + 3, abs=0.01)


def test_ram_of_a_local_directory_sums_its_safetensors(tmp_path: Path) -> None:
    for i in range(2):
        with (tmp_path / f"model-{i}.safetensors").open("wb") as f:
            f.truncate(1_000_000_000)
    (tmp_path / "other.bin").write_bytes(b"x" * 5000)
    assert REAL_LOAD_NEED(_hf(hf_model=str(tmp_path)), _cfg("m")) == pytest.approx(2.0 * 1.25 + 3, abs=0.001)


def test_ram_fails_closed_for_an_unknown_repository_and_a_directory_without_weights(tmp_path: Path) -> None:
    assert REAL_LOAD_NEED(_hf(), _cfg("someone/LLaDA-custom")) is None
    assert REAL_LOAD_NEED(_hf(hf_model=str(tmp_path)), _cfg("m")) is None


def test_the_figure_asked_after_the_load_is_the_working_memory_only() -> None:
    assert REAL_AFTER_LOAD(_hf(), _cfg("GSAI-ML/LLaDA-8B-Instruct")) == 3.0
    assert REAL_AFTER_LOAD(_hf(quant="4bit"), _cfg("Dream-org/Dream-v0-Instruct-7B")) == 3.0
    assert REAL_AFTER_LOAD(_hf(), _cfg("someone/LLaDA-custom")) is None


def test_an_unknown_model_size_is_refused_before_the_loader_runs(rig: Rig) -> None:
    with pytest.raises(SystemExit) as e:
        rig.run("--llada-repo", str(rig.repo), family="someone/LLaDA-custom", llada_repo=False)
    assert "memory" in str(e.value) and "size" in str(e.value)
    assert rig.loads == [] and rig.rows() == []


NEED_LLADA_BF16 = 14.93 * GIB * 1.25 + 3


def test_the_load_guard_refuses_below_the_need_and_never_calls_the_loader(rig: Rig,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf, "mem_available_gb", lambda machine=None: NEED_LLADA_BF16 - 0.5)
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "MemAvailable" in str(e.value) and "before loading" in str(e.value)
    assert rig.loads == [] and rig.rows() == []


def test_the_load_guard_passes_at_exactly_the_need(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf, "mem_available_gb", lambda machine=None: REAL_LOAD_NEED(_hf(), _cfg("GSAI-ML/LLaDA-8B-Instruct")))
    assert rig.run(limit=1) == 0
    assert len(rig.loads) == 1


def test_the_4bit_load_guard_uses_the_smaller_need(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(VERSIONS, "bitsandbytes", "0.43.1")
    monkeypatch.setattr(hf, "mem_available_gb", lambda machine=None: 8.0)      # 4-bit needs about 7.6, bf16 23
    assert rig.run(limit=1, quant="4bit") == 0


def test_an_unreadable_meminfo_refuses_the_load(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf, "mem_available_gb", lambda machine=None: None)
    with pytest.raises(SystemExit) as e:
        rig.run()
    assert "cannot read MemAvailable" in str(e.value) and rig.loads == []


def test_mem_available_reads_the_machine_meminfo_in_gb(tmp_path: Path) -> None:
    import isolation

    (tmp_path / "meminfo").write_text("MemTotal: 33000000 kB\nMemAvailable: 10485760 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")
    assert hf.mem_available_gb(isolation.Machine(proc=tmp_path)) == pytest.approx(10485760 * 1024 / 1e9)
    assert hf.mem_available_gb(isolation.Machine(proc=tmp_path / "absent")) is None


def test_arm_session_sizes_from_the_file_unless_the_engine_gives_the_figure(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    import isolation

    seen: list = []

    class Started:
        def finish(self) -> None:
            pass

    monkeypatch.setattr(isolation, "begin_arm", lambda arm, **kw: seen.append(kw["need_ram_gb"]) or Started())
    args = SimpleNamespace(isolation_retries=0, allow_noisy=False, tune=False)
    f = tmp_path / "m.gguf"
    f.write_bytes(b"x" * 1000)
    for model, kw in ((f, {}), (tmp_path / "absent.gguf", {}), (f, {"ram_gb": None}), (f, {"ram_gb": 7.5})):
        with isolation.arm_session(args, arm="a", device="cpu", model=model, runs_dir=tmp_path, **kw):
            pass
    assert seen == [isolation.need_ram_gb(f), None, None, 7.5]
    assert seen[0] is not None
