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
import ab_evidence  # noqa: E402  (bench/ab_evidence.py)
import spec  # noqa: E402  (bench/spec.py)

MASK = spec.ARCH_IDS["llada"]["mask"]


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd[:3])} ...\n{(p.stderr or p.stdout)[-600:]}")
    return p.stdout


def version(root: pathlib.Path) -> str:
    return ab_evidence.head_label(root, run, short=True)


def require_finite(rows: list[array], label: str) -> None:
    """Refuse a NaN or an infinity in any row: a broken kernel must not leave a healthy-looking record."""
    for r, values in enumerate(rows):
        if not all(map(math.isfinite, values)):
            raise SystemExit(f"compare_logits: {label} logits hold a non-finite value in row {r}")


def cos(x: array, y: array) -> float:
    nx, ny = sum(p * p for p in x), sum(q * q for q in y)
    if nx == 0.0 or ny == 0.0:
        raise SystemExit("compare_logits: a logit row has zero norm, so its cosine is undefined")
    return sum(p * q for p, q in zip(x, y)) / math.sqrt(nx * ny)


def argmax(x: array) -> int:
    return max(range(len(x)), key=x.__getitem__)


def median(xs: list[float]) -> float:
    """The middle value, or the mean of the middle pair for an even count."""
    if not xs:
        raise SystemExit("compare_logits: no values to take a median of")
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def summarize(X: list[array], ref: list[array], prompt_rows: int) -> dict:
    """Per-row cosine to the reference (median, minimum, minimum over the masked rows) and the argmax agreement."""
    c = [cos(x, r) for x, r in zip(X, ref)]
    return dict(median_cosine=median(c), min_cosine=min(c), masked_rows_min_cosine=min(c[prompt_rows:]),
                argmax_agree_with_cpu=sum(argmax(x) == argmax(r) for x, r in zip(X, ref)))


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
    roots = {"base": base, "new": new}
    heads = {k: version(r) for k, r in roots.items()}
    hashes = {k: ab_evidence.binary_sha256(r) for k, r in roots.items()}
    ab_evidence.require_distinct_builds(roots, heads, hashes)
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
        out = [a[r * vocab:(r + 1) * vocab] for r in rows]
        require_finite(out, f"{root.name} {device}")
        return out

    ref = logits(base, "cpu")
    A, B = logits(base, "gpu"), logits(new, "gpu")

    def summary(X: list[array]) -> dict:
        return summarize(X, ref, len(toks))

    ab = [cos(A[r], B[r]) for r in rows]
    doc = dict(
        date=dt.date.today().isoformat(), model=pathlib.Path(model).name, rows=len(rows), prompt_rows=len(toks),
        method="logits for every row of the prompt plus masked tokens; per-row cosine against the base checkout's f32 CPU run",
        base=dict(version=heads["base"], binary_sha256=hashes["base"], **summary(A)),
        new=dict(version=heads["new"], binary_sha256=hashes["new"], **summary(B)),
        base_vs_new=dict(median_cosine=median(ab), min_cosine=min(ab),
                         argmax_agree=sum(argmax(A[r]) == argmax(B[r]) for r in rows)),
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
