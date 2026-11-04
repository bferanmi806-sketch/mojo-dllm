#!/usr/bin/env python3
"""Build the tokenizer fixtures from a real LLaDA GGUF and llama.cpp.

usage: uv run --only-group fixtures python scripts/gen_tokenizer_fixture.py \\
           MODEL.gguf PATH/TO/llama-tokenize [PREFIX]

PREFIX names the output files (default "llada"; Dream uses "dream"; "gemma4" is
DiffusionGemma, from the llama.cpp branch of ggml-org/llama.cpp#24423).

Outputs:
  tests/fixtures/<PREFIX>-tokenizer.gguf         tokenizer metadata only, no tensors
  tests/fixtures/<PREFIX>-tokenizer-golden.tsv   hex(utf-8 text) <TAB> ids from llama-tokenize

The golden ids come from llama.cpp's tokenizer on the full model file with
special-token parsing on and no BOS added, which is how mojo-dllm encodes a
formatted chat prompt.

gemma4 differs in three ways. Its tokenizer metadata (262,144 tokens and 514,906
merges) is not copied, so it needs --golden-only; the tokenizer fixture waits for
the implementation that reads it. llama-tokenize gets --no-escape, so a literal
backslash-n in a case stays two characters. And the generator checks, for every
case, that the ids with the BOS the model adds on its own equal [bos] + the ids
without it, that the llama.cpp checkout is the commit scripts/build_dg_ref.sh pins,
and it writes tests/fixtures/gemma4-tokenizer-golden.provenance.json: the commit,
the model file and its sha256, and the command.

  uv run --only-group fixtures python scripts/gen_tokenizer_fixture.py --golden-only \\
      MODEL.gguf $DG_LLAMA_SRC/build/bin/llama-tokenize gemma4
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures"

CASES = [
    "Hello world",
    "Explain speculative decoding in two sentences.",
    "  leading spaces",
    "trailing spaces   ",
    "tabs\tand\nnewlines\n\n\nend",
    "I'm you're they've he'll she'd it's WE'LL DON'T",
    "'s'S'll'LL'Re'x",
    "numbers 12345 and 3.14159 and 1,000,000",
    "abc123def456",
    "café naïve résumé Ñandú",
    "ÀÉÎÕÜ àéîõü ßæøå",
    "combining é accents",
    "中文测试，你好世界！",
    "日本語のテキスト",
    "안녕하세요 세계",
    "Привет, мир!",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "emoji 😀🎉👍🏽 and 1️⃣",
    "code: def f(x):\n    return x**2 + 1  # comment",
    "mixed\r\nwindows\r\nlines",
    "symbols $+<=>^`|~ and punct !@#%&*()[]{}",
    "a b nbsp and em space",
    "!!!???...,,,",
    " ",
    "\n",
    "x  \n  y",
    " ?!? test",
    "Supercalifragilisticexpialidocious",
    "<|start_header_id|>user<|end_header_id|>\n\nHi there<|eot_id|>",
    "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nWhat is paged attention?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "Ünïcödé 🤖 robots",
    "line one\n\n  indented\n\t\ttabbed",
    "URL https://example.com/path?q=1&r=2#frag",
    "math: ∑ x² ≤ ∞, π ≈ 3.14",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
]


# DiffusionGemma's gemma4 tokenizer (issue #62): SentencePiece-style spaces (U+2581), raw UTF-8 with no
# GPT-2 byte map, a pre-split into runs of newlines and runs of everything else, byte fallback through the
# 256 <0xNN> tokens, and BOS added by llama.cpp whether or not the vocabulary asks for it.
GEMMA4_CASES = [
    # spaces
    "Hello world",
    "Explain speculative decoding in two sentences.",
    "  leading spaces",
    "trailing spaces   ",
    "multiple    spaces    inside",
    " ",
    "                ",
    "a\u00a0b nbsp\u2003em space\u3000ideographic",
    # newlines and other whitespace
    "\n",
    "\n\n",
    "\n\n\n\n\n\n\n\n",
    "line one\nline two\n\nline four\n\n\n\nline eight",
    "x  \n  y",
    " \n \n ",
    "tabs\tand\t\tmore\ttabs",
    "mixed\r\nwindows\r\nlines",
    "line one\n\n  indented\n\t\ttabbed",
    # numbers
    "numbers 12345 and 3.14159 and 1,000,000",
    "abc123def456",
    "2026-11-04T04:30:00Z 0x1F4A9 -42 +7 1e-9",
    # CJK and other scripts
    "中文测试，你好世界！",
    "日本語のテキスト、カタカナも。",
    "안녕하세요 세계",
    "Привет, мир!",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "café naïve résumé Ñandú",
    "combining e\u0301 accents",
    # emoji
    "emoji 😀🎉👍🏽 and 1️⃣",
    "family 👨\u200d👩\u200d👧\u200d👦 flag 🇷🇴",
    "Ünïcödé 🤖 robots",
    # code and text with symbols
    "def f(x):\n    return x**2 + 1  # comment",
    'fn main() {\n    println!("{}", 1 + 2);\n}\n',
    "SELECT * FROM t WHERE a >= 10 AND b <> 'x';",
    "if (a && b) { return a->c[0]; } // done",
    r'print("tab:\t newline:\n") C:\Users\x \d+\s*',
    "URL https://example.com/path?q=1&r=2#frag",
    "math: ∑ x² ≤ ∞, π ≈ 3.14",
    "symbols $+<=>^`|~ and punct !@#%&*()[]{}",
    "!!!???...,,,",
    "Supercalifragilisticexpialidocious",
    # the tokenizer's own conventions. The first one cannot round-trip: decoding turns every U+2581 into a space.
    "lower\u2581block \u2581\u2581 doubled",
    "<0x41> is a byte-token lookalike, <0xFF> too",
    "byte fallback: \ue000 \u0378 \x01 \U0010ffff",
    "<unused0> and <unused99> and [multimodal] are ordinary vocabulary entries",
    # The chat template and its special tokens. The first three are what the GGUF's own tokenizer.chat_template
    # renders for a user turn, a system and a user turn, and a three-message conversation, each with
    # add_generation_prompt. A <bos> inside the text is parsed to token 2, on top of the BOS llama.cpp adds.
    "<bos><|turn>user\nWhat is paged attention?<turn|>\n<|turn>model\n",
    "<bos><|turn>system\nYou are a helpful assistant.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
    "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\n"
    "Explain speculative decoding in two sentences.<turn|>\n<|turn>model\n",
    "<|turn>user\nno bos in the text<turn|>\n",
    "<bos><bos>",
    "a<mask>b<eos>c<pad>d<unk>e",
    '<|tool_call>call:f{x:<|"|>a<|"|>}<tool_call|>',
    "<|channel>thought\n<channel|>The answer is 4.",
    "<turn|><turn|>\n\n<|turn>",
]


def copy_tokenizer(model: pathlib.Path, prefix: str) -> None:
    # Imported here so the golden-only path (and its tests) need no gguf package.
    from gguf import GGUFReader, GGUFValueType, GGUFWriter

    r = GGUFReader(str(model))
    w = GGUFWriter(str(OUT / f"{prefix}-tokenizer.gguf"), prefix)
    for name, field in r.fields.items():
        if not name.startswith("tokenizer."):
            continue
        vt = field.types[0]
        if vt == GGUFValueType.STRING:
            w.add_string(name, bytes(field.parts[field.data[0]]).decode("utf-8"))
        elif vt == GGUFValueType.ARRAY:
            et = field.types[1]
            if et == GGUFValueType.STRING:
                vals = [bytes(field.parts[i]).decode("utf-8", errors="surrogateescape") for i in field.data]
                w.add_array(name, vals)
            else:
                w.add_array(name, [int(field.parts[i][0]) for i in field.data])
        elif vt == GGUFValueType.BOOL:
            w.add_bool(name, bool(field.parts[field.data[0]][0]))
        else:
            w.add_uint32(name, int(field.parts[field.data[0]][0]))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


PINNED_RE = re.compile(r'^PINNED_COMMIT="([0-9a-f]{40})"$', re.M)

# Per prefix: the cases and the llama-tokenize flags beyond the common ones. A prefix not listed here is
# LLaDA or Dream, whose goldens were made with exactly the default flags and must stay byte-for-byte.
PROFILES = {"gemma4": dict(cases=GEMMA4_CASES, flags=["--no-escape"], gemma=True)}
DEFAULT_PROFILE = dict(cases=CASES, flags=[], gemma=False)


def _ids(tokenize: str, model: pathlib.Path, path: str, flags: list[str], bos: bool) -> list[int]:
    cmd = [tokenize, "-m", str(model), "-f", path, "--ids", *flags, "--log-disable"]
    if not bos:
        cmd.insert(cmd.index("--ids") + 1, "--no-bos")
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    last = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    ids = [int(x) for x in re.findall(r"\d+", last)] if re.fullmatch(r"\[[\d,\s]*\]", last) else []
    if not ids:
        raise SystemExit(f"llama-tokenize returned no ids for {path}: {res.stdout[-200:]!r}")
    return ids


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def golden(model: pathlib.Path, tokenize: str, prefix: str, expect_commit: str | None = None) -> None:
    profile = PROFILES.get(prefix, DEFAULT_PROFILE)
    commit = None
    if profile["gemma"]:
        if expect_commit is None:
            pins = PINNED_RE.findall((ROOT / "scripts" / "build_dg_ref.sh").read_text())
            if len(pins) != 1:
                raise SystemExit("scripts/build_dg_ref.sh does not hold exactly one full PINNED_COMMIT")
            expect_commit = pins[0]
        src = pathlib.Path(tokenize).resolve().parents[2]
        commit = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if commit != expect_commit:
            raise SystemExit(f"the llama.cpp checkout {src} is at {commit}, the pinned commit is {expect_commit}")
    lines = []
    bos_ids = set()
    for text in profile["cases"]:
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(text.encode("utf-8"))
            path = f.name
        ids = _ids(tokenize, model, path, profile["flags"], bos=False)
        if profile["gemma"]:
            with_bos = _ids(tokenize, model, path, profile["flags"], bos=True)
            if with_bos[1:] != ids:
                raise SystemExit(
                    f"BOS check failed for {text!r}: with BOS {with_bos}, without {ids}; "
                    "--no-bos must remove the BOS and change nothing else"
                )
            bos_ids.add(with_bos[0])
        lines.append(text.encode("utf-8").hex() + "\t" + ",".join(map(str, ids)))
    if profile["gemma"] and len(bos_ids) != 1:
        raise SystemExit(f"the BOS id differs between cases: {sorted(bos_ids)}")
    tsv = OUT / f"{prefix}-tokenizer-golden.tsv"
    tsv.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} golden cases")
    if profile["gemma"]:
        provenance = dict(
            llama_cpp=dict(
                repository="https://github.com/danielhanchen/llama.cpp",
                pull_request="ggml-org/llama.cpp#24423",
                commit=commit,
            ),
            model=dict(file=model.name, size_bytes=model.stat().st_size, sha256=_sha256(model)),
            command="llama-tokenize -m MODEL.gguf -f CASE.txt --ids --no-bos --no-escape --log-disable",
            regenerate=(
                "uv run --only-group fixtures python scripts/gen_tokenizer_fixture.py --golden-only "
                "MODEL.gguf $DG_LLAMA_SRC/build/bin/llama-tokenize gemma4"
            ),
            bos_id=bos_ids.pop(),
            bos_check="for every case the ids with BOS equal [bos_id] + the ids without it",
            cases=len(lines),
            golden_sha256=hashlib.sha256(tsv.read_bytes()).hexdigest(),
        )
        (OUT / f"{prefix}-tokenizer-golden.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--golden-only"]
    golden_only = len(args) != len(sys.argv) - 1
    model = pathlib.Path(args[0])
    prefix = args[2] if len(args) > 2 else "llada"
    if prefix in PROFILES and not golden_only:
        raise SystemExit(f"{prefix}: pass --golden-only; its tokenizer metadata is not copied into a fixture")
    if not golden_only:
        copy_tokenizer(model, prefix)
    golden(model, args[1], prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
