"""The gemma4 tokenizer goldens: how scripts/gen_tokenizer_fixture.py makes them, and what is committed.

The generator runs here against a stand-in `llama-tokenize`, so the tests prove what the generator checks
and records. The committed ids themselves can only be verified against the real model file, which is what
the generator's own invariants and the provenance record (tests/fixtures/gemma4-tokenizer-golden.provenance.json) do.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"


def load_generator():
    spec = importlib.util.spec_from_file_location("gen_tokenizer_fixture", ROOT / "scripts/gen_tokenizer_fixture.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["gen_tokenizer_fixture"] = module
    spec.loader.exec_module(module)
    return module


gen = load_generator()

# A stand-in for llama-tokenize: the ids are the UTF-8 bytes, plus a BOS (id 2) unless --no-bos.
FAKE = """#!/usr/bin/env python3
import sys
import os
args = sys.argv[1:]
if os.environ.get("FAKE_LOG"):
    open(os.environ["FAKE_LOG"], "a").write(" ".join(args) + "\\n")
text = open(args[args.index("-f") + 1], "rb").read()
if "--no-escape" not in args:
    text = text.replace(b"\\\\n", b"\\n")
ids = list(text)
if "--no-bos" not in args:
    ids = [2] + ids
print("[" + ", ".join(map(str, ids)) + "]")
"""


def fake_tokenizer(folder: Path, body: str = FAKE) -> Path:
    path = folder / "src" / "build" / "bin" / "llama-tokenize"
    path.parent.mkdir(parents=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    subprocess.run(["git", "init", "-q", "-b", "main", str(folder / "src")], check=True)
    subprocess.run(
        ["git", "-C", str(folder / "src"), "-c", "user.name=t", "-c", "user.email=t@example.com",
         "commit", "-q", "--allow-empty", "-m", "one"],
        check=True,
    )
    return path


def head(folder: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(folder / "src"), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(gen, "OUT", out)
    (tmp_path / "model.gguf").write_bytes(b"not a real model")
    return tmp_path


def test_gemma4_goldens_are_written_in_the_existing_format(workdir: Path) -> None:
    tok = fake_tokenizer(workdir)
    gen.golden(workdir / "model.gguf", str(tok), "gemma4", expect_commit=head(workdir))
    rows = (workdir / "out" / "gemma4-tokenizer-golden.tsv").read_text().splitlines()
    assert len(rows) == len(gen.GEMMA4_CASES) >= 30
    for text, row in zip(gen.GEMMA4_CASES, rows, strict=True):
        hexed, ids = row.split("\t")
        assert bytes.fromhex(hexed).decode("utf-8") == text
        assert ids == ",".join(str(b) for b in text.encode("utf-8")), "ids are written without BOS, comma separated"


def test_the_gemma4_run_passes_no_escape_and_keeps_literal_backslashes(workdir: Path) -> None:
    tok = fake_tokenizer(workdir)
    gen.golden(workdir / "model.gguf", str(tok), "gemma4", expect_commit=head(workdir))
    assert any("\\" in c for c in gen.GEMMA4_CASES), "a case must hold a literal backslash"
    rows = (workdir / "out" / "gemma4-tokenizer-golden.tsv").read_text().splitlines()
    by_text = {bytes.fromhex(r.split("\t")[0]).decode(): r.split("\t")[1] for r in rows}
    case = next(c for c in gen.GEMMA4_CASES if "\\n" in c)
    assert by_text[case] == ",".join(str(b) for b in case.encode()), "llama-tokenize must not turn \\n into a newline"


def test_the_existing_goldens_keep_their_flags(workdir: Path) -> None:
    tok = fake_tokenizer(workdir)
    gen.golden(workdir / "model.gguf", str(tok), "llada")
    rows = (workdir / "out" / "llada-tokenizer-golden.tsv").read_text().splitlines()
    assert len(rows) == len(gen.CASES) == 36
    assert (workdir / "out" / "llada-tokenizer-golden.provenance.json").exists()
    assert not (workdir / "out" / "gemma4-tokenizer-golden.provenance.json").exists()


def test_a_tokenizer_that_adds_bos_despite_no_bos_is_refused(workdir: Path) -> None:
    tok = fake_tokenizer(workdir, FAKE.replace('if "--no-bos" not in args:', "if True:"))
    with pytest.raises(SystemExit) as e:
        gen.golden(workdir / "model.gguf", str(tok), "gemma4", expect_commit=head(workdir))
    assert "BOS" in str(e.value)
    assert not (workdir / "out" / "gemma4-tokenizer-golden.tsv").exists()


def test_a_tokenizer_at_another_commit_is_refused(workdir: Path) -> None:
    tok = fake_tokenizer(workdir)
    with pytest.raises(SystemExit) as e:
        gen.golden(workdir / "model.gguf", str(tok), "gemma4", expect_commit="0" * 40)
    assert head(workdir) in str(e.value) and "0" * 40 in str(e.value)
    assert not (workdir / "out" / "gemma4-tokenizer-golden.tsv").exists()


def test_an_empty_answer_is_refused(workdir: Path) -> None:
    tok = fake_tokenizer(workdir, "#!/usr/bin/env bash\necho '[]'\n")
    with pytest.raises(SystemExit):
        gen.golden(workdir / "model.gguf", str(tok), "gemma4", expect_commit=head(workdir))


def test_provenance_records_the_commit_the_model_and_the_command(workdir: Path) -> None:
    tok = fake_tokenizer(workdir)
    gen.golden(workdir / "model.gguf", str(tok), "gemma4", expect_commit=head(workdir))
    prov = json.loads((workdir / "out" / "gemma4-tokenizer-golden.provenance.json").read_text())
    assert prov["llama_cpp"]["commit"] == head(workdir)
    assert prov["llama_cpp"]["pull_request"] == "ggml-org/llama.cpp#24423"
    assert prov["model"]["file"] == "model.gguf"
    assert prov["model"]["sha256"] == hashlib.sha256(b"not a real model").hexdigest()
    assert prov["model"]["size_bytes"] == len(b"not a real model")
    assert prov["bos_id"] == 2
    assert prov["cases"] == len(gen.GEMMA4_CASES)
    tsv = (workdir / "out" / "gemma4-tokenizer-golden.tsv").read_bytes()
    assert prov["golden_sha256"] == hashlib.sha256(tsv).hexdigest()
    for flag in ("--ids", "--no-bos", "--no-escape", "-f"):
        assert flag in prov["command"]
    assert "/home/" not in json.dumps(prov), "an absolute home path must not enter a tracked file"


@pytest.mark.parametrize("prefix", ["llada", "dream"])
def test_the_llada_and_dream_goldens_record_their_origin(workdir: Path, prefix: str) -> None:
    tok = fake_tokenizer(workdir)
    gen.golden(workdir / "model.gguf", str(tok), prefix)
    prov = json.loads((workdir / "out" / f"{prefix}-tokenizer-golden.provenance.json").read_text())
    assert prov["llama_cpp"] == {"commit": head(workdir)}
    assert prov["model"] == {
        "file": "model.gguf",
        "size_bytes": len(b"not a real model"),
        "sha256": hashlib.sha256(b"not a real model").hexdigest(),
    }
    assert prov["command"] == "llama-tokenize -m MODEL.gguf -f CASE.txt --ids --no-bos --log-disable"
    assert prov["regenerate"].endswith(f"llama-tokenize {prefix}") and "--golden-only" in prov["regenerate"]
    assert prov["cases"] == len(gen.CASES)
    tsv = (workdir / "out" / f"{prefix}-tokenizer-golden.tsv").read_bytes()
    assert prov["golden_sha256"] == hashlib.sha256(tsv).hexdigest()
    assert "bos_id" not in prov and "/home/" not in json.dumps(prov)


def test_the_generator_still_writes_the_committed_gemma4_provenance_in_its_form(workdir: Path) -> None:
    """Key order, command and regenerate line: a regeneration changes the commit, the model and the hashes only."""
    tok = fake_tokenizer(workdir)
    gen.golden(workdir / "model.gguf", str(tok), "gemma4", expect_commit=head(workdir))
    fresh = json.loads((workdir / "out" / "gemma4-tokenizer-golden.provenance.json").read_text())
    committed = json.loads((FIXTURES / "gemma4-tokenizer-golden.provenance.json").read_text())
    assert list(fresh) == list(committed)
    assert list(fresh["llama_cpp"]) == list(committed["llama_cpp"]) and list(fresh["model"]) == list(committed["model"])
    for key in ("command", "regenerate", "bos_id", "bos_check", "cases"):
        assert fresh[key] == committed[key], key


def test_a_tokenizer_outside_a_git_checkout_is_refused_because_its_commit_is_unknown(workdir: Path) -> None:
    tok = workdir / "bin" / "llama-tokenize"
    tok.parent.mkdir()
    tok.write_text(FAKE)
    tok.chmod(tok.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(SystemExit, match="git checkout"):
        gen.golden(workdir / "model.gguf", str(tok), "llada")
    assert not (workdir / "out" / "llada-tokenizer-golden.tsv").exists()


# What is committed ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", ["llada", "dream", "gemma4"])
def test_each_committed_golden_has_provenance_that_describes_it(prefix: str) -> None:
    tsv = FIXTURES / f"{prefix}-tokenizer-golden.tsv"
    prov = json.loads((FIXTURES / f"{prefix}-tokenizer-golden.provenance.json").read_text())
    assert prov["golden_sha256"] == hashlib.sha256(tsv.read_bytes()).hexdigest(), "the golden changed after it was recorded"
    assert prov["cases"] == len(tsv.read_text().splitlines())
    assert re.fullmatch(r"[0-9a-f]{40}", prov["llama_cpp"]["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", prov["model"]["sha256"]) and prov["model"]["size_bytes"] > 1_000_000
    assert prov["regenerate"].endswith(prefix) and "--golden-only" in prov["regenerate"]
    assert "/home/" not in json.dumps(prov)

TSV = FIXTURES / "gemma4-tokenizer-golden.tsv"
PROVENANCE = FIXTURES / "gemma4-tokenizer-golden.provenance.json"


def committed_rows() -> list[tuple[str, list[int]]]:
    rows = []
    for line in TSV.read_text().splitlines():
        hexed, ids = line.split("\t")
        rows.append((bytes.fromhex(hexed).decode("utf-8"), [int(x) for x in ids.split(",")]))
    return rows


def test_the_committed_goldens_are_the_generators_cases_in_order() -> None:
    tree = ast.parse((ROOT / "scripts/gen_tokenizer_fixture.py").read_text())
    cases = next(
        ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "GEMMA4_CASES"
    )
    assert [t for t, _ in committed_rows()] == cases, "the file was edited by hand or the cases drifted"


def test_the_committed_ids_are_plausible_for_the_262144_vocabulary() -> None:
    rows = committed_rows()
    assert len(rows) >= 30
    assert len({t for t, _ in rows}) == len(rows), "duplicate cases"
    for text, ids in rows:
        assert ids and all(0 <= i < 262144 for i in ids), text
        # Only a <bos> typed into the text may start a row with token 2: the BOS llama.cpp adds is not stored.
        assert ids[0] != 2 or text.startswith("<bos>"), "goldens are stored without BOS, as for LLaDA and Dream"


def test_the_goldens_cover_what_issue_62_names() -> None:
    texts = [t for t, _ in committed_rows()]
    covers = {
        "a run of spaces": lambda t: "   " in t and "\n" not in t,
        "leading space": lambda t: t.startswith(" ") and t.strip() != "",
        "trailing space": lambda t: t.endswith(" ") and t.strip() != "",
        "a run of newlines": lambda t: "\n\n" in t,
        "a lone newline": lambda t: t == "\n",
        "tab": lambda t: "\t" in t,
        "CJK": lambda t: re.search(r"[一-鿿぀-ヿ가-힯]", t) is not None,
        "emoji": lambda t: re.search("[\U0001f300-\U0001faff]", t) is not None,
        "code": lambda t: "def " in t or "{" in t,
        "numbers": lambda t: re.search(r"\d{4,}", t) is not None,
        "a literal backslash": lambda t: "\\" in t,
        "the lower-one-eighth block U+2581": lambda t: "▁" in t,
        "a byte-token lookalike": lambda t: re.search(r"<0x[0-9A-F]{2}>", t) is not None,
        "a special token": lambda t: re.search(r"<[^<>\s]*\|?[a-z_]+\|?[^<>\s]*>", t) is not None,
    }
    missing = [name for name, test in covers.items() if not any(test(t) for t in texts)]
    assert missing == []


def test_the_provenance_matches_the_pin_and_the_file() -> None:
    prov = json.loads(PROVENANCE.read_text())
    pin = re.search(r'^PINNED_COMMIT="([0-9a-f]{40})"$', (ROOT / "scripts/build_dg_ref.sh").read_text(), flags=re.M)
    assert pin is not None
    assert prov["llama_cpp"]["commit"] == pin.group(1), "the goldens come from a different commit than the build script pins"
    assert prov["golden_sha256"] == hashlib.sha256(TSV.read_bytes()).hexdigest()
    assert prov["cases"] == len(committed_rows())
    assert prov["model"]["sha256"] == "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5"
    assert prov["model"]["size_bytes"] == 16806810208
    assert prov["bos_id"] == 2


def calls(log: Path) -> list[list[str]]:
    return [line.split() for line in log.read_text().splitlines()]


def test_llada_and_dream_keep_exactly_their_original_command(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = workdir / "calls.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    tok = fake_tokenizer(workdir)
    gen.golden(workdir / "model.gguf", str(tok), "dream")
    seen = calls(log)
    assert len(seen) == len(gen.CASES)
    for argv in seen:
        # [-m MODEL -f CASE --ids --no-bos --log-disable]: nothing added, nothing reordered.
        assert argv[:2] == ["-m", str(workdir / "model.gguf")] and argv[2] == "-f"
        assert argv[4:] == ["--ids", "--no-bos", "--log-disable"]


def test_gemma4_asks_twice_per_case_once_with_and_once_without_bos(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = workdir / "calls.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    tok = fake_tokenizer(workdir)
    gen.golden(workdir / "model.gguf", str(tok), "gemma4", expect_commit=head(workdir))
    seen = calls(log)
    assert len(seen) == 2 * len(gen.GEMMA4_CASES)
    assert sum("--no-bos" in a for a in seen) == len(gen.GEMMA4_CASES)
    assert all("--no-escape" in a for a in seen)


def test_a_bos_id_that_changes_between_cases_is_refused(workdir: Path) -> None:
    # BOS is 2 for texts of even length and 7 for odd length: each answer is [bos] + ids, so only the
    # comparison of the BOS ids across cases can notice.
    varying = FAKE.replace("ids = [2] + ids", "ids = [2 + 5 * (len(text) % 2)] + ids")
    assert varying != FAKE
    tok = fake_tokenizer(workdir, varying)
    with pytest.raises(SystemExit) as e:
        gen.golden(workdir / "model.gguf", str(tok), "gemma4", expect_commit=head(workdir))
    assert "BOS id differs" in str(e.value)
