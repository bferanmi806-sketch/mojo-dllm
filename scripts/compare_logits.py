#!/usr/bin/env python3
"""Compare two builds' GPU logits on a real model, with the f32 CPU path as reference.

usage: python3 scripts/compare_logits.py --model M.gguf --base CHECKOUT --new CHECKOUT --out bench/checks/DATE-NAME.json
                                         [--prompt TEXT] [--masked 64]

Each checkout's build/mojo-dllm computes logits for every row of the chat-formatted
prompt followed by `--masked` mask tokens (the canvas a first denoising step sees).
The base checkout's CPU run is the reference. For each build the result gives the
per-row cosine to the reference (median, minimum, minimum over masked rows) and how
many rows keep the reference argmax, then the two builds against each other. Use it
for a kernel change that is not exact, before and after.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from array import array

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "bench"))
import spec  # noqa: E402  (bench/spec.py)

MASK = spec.ARCH_IDS["llada"]["mask"]


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd[:3])} ...\n{(p.stderr or p.stdout)[-600:]}")
    return p.stdout


def version(root: pathlib.Path) -> str:
    v = run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"]).strip()
    dirty = run(["git", "-C", str(root), "status", "--porcelain", "--", "src"]).strip()
    return v + ("-dirty" if dirty else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=spec.PROMPTS[0])
    ap.add_argument("--masked", type=int, default=64)
    args = ap.parse_args()
    model = os.path.expanduser(args.model)
    base = pathlib.Path(args.base).resolve()
    new = pathlib.Path(args.new).resolve()
    toks = json.loads(run([str(base / "build/mojo-dllm"), "tokenize", "--model", model, "--prompt", args.prompt]).strip().splitlines()[-1])
    ids = toks + [MASK] * args.masked
    rows = list(range(len(ids)))

    def logits(root: pathlib.Path, device: str) -> list[array]:
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "logits.f32"
            run([str(root / "build/mojo-dllm"), "logits", "--model", model, "--tokens", ",".join(map(str, ids)),
                 "--rows", ",".join(map(str, rows)), "--out", str(f), "--device", device])
            a = array("f")
            a.frombytes(f.read_bytes())
        if len(a) % len(rows):
            raise SystemExit(f"logits file holds {len(a)} floats, not a multiple of {len(rows)} rows")
        vocab = len(a) // len(rows)
        return [a[r * vocab:(r + 1) * vocab] for r in rows]

    def cos(x: array, y: array) -> float:
        return sum(p * q for p, q in zip(x, y)) / math.sqrt(sum(p * p for p in x) * sum(q * q for q in y))

    def argmax(x: array) -> int:
        return max(range(len(x)), key=x.__getitem__)

    ref = logits(base, "cpu")
    ref_top = [argmax(r) for r in ref]
    A, B = logits(base, "gpu"), logits(new, "gpu")

    def summary(X: list[array]) -> dict:
        c = [cos(X[r], ref[r]) for r in rows]
        s = sorted(c)
        return dict(median_cosine=s[len(s) // 2], min_cosine=s[0], masked_rows_min_cosine=min(c[len(toks):]),
                    argmax_agree_with_cpu=sum(argmax(X[r]) == ref_top[r] for r in rows))

    ab = sorted(cos(A[r], B[r]) for r in rows)
    doc = dict(
        date=dt.date.today().isoformat(), model=pathlib.Path(model).name, rows=len(rows), prompt_rows=len(toks),
        method="logits for every row of the prompt plus masked tokens; per-row cosine against the base checkout's f32 CPU run",
        base=dict(version=version(base), **summary(A)), new=dict(version=version(new), **summary(B)),
        base_vs_new=dict(median_cosine=ab[len(ab) // 2], min_cosine=ab[0],
                         argmax_agree=sum(argmax(A[r]) == argmax(B[r]) for r in rows)),
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
