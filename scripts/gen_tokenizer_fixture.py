#!/usr/bin/env python3
"""Build the tokenizer fixtures from a real LLaDA GGUF and llama.cpp.

usage: uv run --only-group fixtures python scripts/gen_tokenizer_fixture.py \\
           MODEL.gguf PATH/TO/llama-tokenize [PREFIX]

PREFIX names the output files (default "llada"; Dream uses "dream").

Outputs:
  tests/fixtures/<PREFIX>-tokenizer.gguf         tokenizer metadata only, no tensors
  tests/fixtures/<PREFIX>-tokenizer-golden.tsv   hex(utf-8 text) <TAB> ids from llama-tokenize

The golden ids come from llama.cpp's tokenizer on the full model file with
special-token parsing on and no BOS added, which is how mojo-dllm encodes a
formatted chat prompt.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

from gguf import GGUFReader, GGUFValueType, GGUFWriter

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


def copy_tokenizer(model: pathlib.Path, prefix: str) -> None:
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


def golden(model: pathlib.Path, tokenize: str, prefix: str) -> None:
    lines = []
    for text in CASES:
        with tempfile.NamedTemporaryFile("wb", delete=False) as f:
            f.write(text.encode("utf-8"))
            path = f.name
        res = subprocess.run(
            [tokenize, "-m", str(model), "-f", path, "--ids", "--no-bos", "--log-disable"],
            capture_output=True,
            text=True,
            check=True,
        )
        ids = res.stdout.strip().splitlines()[-1].strip("[]").replace(" ", "")
        if not ids:
            raise SystemExit(f"llama-tokenize returned no ids for {text!r}")
        lines.append(text.encode("utf-8").hex() + "\t" + ids)
    (OUT / f"{prefix}-tokenizer-golden.tsv").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} golden cases")


def main() -> int:
    model = pathlib.Path(sys.argv[1])
    prefix = sys.argv[3] if len(sys.argv) > 3 else "llada"
    copy_tokenizer(model, prefix)
    golden(model, sys.argv[2], prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
