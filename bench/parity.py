#!/usr/bin/env python3
"""Logit and generation parity between mojo-dllm and llama.cpp on a real model.

usage: python3 bench/parity.py --model MODEL.gguf [--ref build/llada_ref] [--threads 12] [--device cpu|gpu]
                              [--ref2 build/llada_ref_cuda --ref2-label "llama.cpp CUDA"]

--ref2 adds a second reference for the logits only: the result then also
records mojo-dllm against it and the two references against each other, which
shows how far two builds of llama.cpp disagree on the same model.

Writes bench/parity/<date>-<arch>-<model-stem>[-gpu].json. scripts/bench_table.py renders it into
docs/correctness.md. Requires build/mojo-dllm and build/llada_ref
(see docs/correctness.md for how to build the latter).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import tempfile
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPECS = {"llada": dict(vocab=126464, mask=126336), "dream": dict(vocab=152064, mask=151666)}

class LogitCase(NamedTuple):
    prompt: str
    masks: int


class GenCase(NamedTuple):
    prompt: str
    gen: int
    block: int
    steps: int


LOGIT_CASES = [
    LogitCase("Explain speculative decoding in two sentences.", 44),
    LogitCase("List three differences between diffusion language models and autoregressive language models.", 100),
]
GEN_CASES = [
    GenCase("Explain speculative decoding in two sentences.", 32, 32, 32),
    GenCase("Write a short Python function that checks whether a number is prime.", 64, 32, 32),
]


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd[:3])} ...\n{p.stderr[-800:]}")
    return p.stdout


def tokenize(model: str, prompt: str) -> list[int]:
    out = run([str(ROOT / "build/mojo-dllm"), "tokenize", "--model", model, "--prompt", prompt])
    return [int(x) for x in out.strip().splitlines()[-1].strip("[]").split(",")]


def result_path(model: str, arch: str, device: str, day: dt.date) -> pathlib.Path:
    """Different checkpoints of one architecture must not overwrite each other."""
    suffix = "-gpu" if device == "gpu" else ""
    return ROOT / "bench" / "parity" / f"{day.isoformat()}-{arch}-{pathlib.Path(model).stem}{suffix}.json"


def compare(a: np.ndarray, b: np.ndarray, first_mask: int) -> dict:
    import numpy as np

    cos = np.array([float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(a, b)])
    rel = np.array([float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(y**2))) for x, y in zip(a, b)])
    top5 = [len(set(np.argsort(-x)[:5]) & set(np.argsort(-y)[:5])) for x, y in zip(a, b)]
    agree = int(sum(int(np.argmax(x) == np.argmax(y)) for x, y in zip(a, b)))
    worst = np.argsort(cos)[:3]
    return dict(rows=len(cos), min_cosine=float(cos.min()), median_cosine=float(np.median(cos)),
                masked_median_cosine=float(np.median(cos[first_mask:])),
                max_rel_rms=float(rel.max()), median_rel_rms=float(np.median(rel)), argmax_agree=agree,
                mean_top5_overlap=sum(top5) / len(top5), worst_rows=[int(i) for i in worst])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default=str(ROOT / "build/llada_ref"))
    ap.add_argument("--threads", default="12")
    ap.add_argument("--arch", choices=sorted(SPECS), default="llada")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--ref2", default=None, help="a second reference binary, logits only")
    ap.add_argument("--ref2-label", default="llama.cpp CUDA")
    args = ap.parse_args()
    import numpy as np

    VOCAB = SPECS[args.arch]["vocab"]
    MASK = SPECS[args.arch]["mask"]
    logits = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in LOGIT_CASES:
            toks = tokenize(args.model, case.prompt) + [MASK] * case.masks
            L = len(toks)
            # Every row: a sample of 8 let the published worst case depend on
            # which positions happened to be picked.
            rows = list(range(L))
            tcsv = ",".join(map(str, toks))
            rcsv = ",".join(map(str, rows))
            a, b, b2 = f"{tmp}/mojo.f32", f"{tmp}/ref.f32", f"{tmp}/ref2.f32"
            run([str(ROOT / "build/mojo-dllm"), "logits", "--model", args.model, "--tokens", tcsv,
                 "--rows", rcsv, "--out", a, "--threads", args.threads, "--device", args.device])
            run([args.ref, "logits", args.model, tcsv, rcsv, b, args.threads])
            x = np.fromfile(a, dtype=np.float32).reshape(-1, VOCAB).astype(np.float64)
            y = np.fromfile(b, dtype=np.float32).reshape(-1, VOCAB).astype(np.float64)
            if x.size == 0 or x.shape != y.shape:
                raise SystemExit("refusing: logit dumps are empty or differ in shape")
            first = len(toks) - case.masks
            entry = dict(prompt=case.prompt, seq_len=L, **compare(x, y, first))
            if args.ref2:
                run([args.ref2, "logits", args.model, tcsv, rcsv, b2, args.threads])
                z = np.fromfile(b2, dtype=np.float32).reshape(-1, VOCAB).astype(np.float64)
                if z.shape != y.shape:
                    raise SystemExit("refusing: the second reference's dump differs in shape")
                entry["vs_ref2"] = compare(x, z, first)
                entry["ref_vs_ref2"] = compare(y, z, first)
            logits.append(entry)
            print("logits", logits[-1])
    gens = []
    for case in GEN_CASES:
        toks = tokenize(args.model, case.prompt)
        out = run([str(ROOT / "build/mojo-dllm"), "run", "--model", args.model, "--prompt", case.prompt,
                   "--max-tokens", str(case.gen), "--block-length", str(case.block),
                   "--steps", str(case.steps), "--threads", args.threads, "--device", args.device, "--json",
                   "--no-early-stop"])
        j = json.loads(next(l for l in out.splitlines() if l.startswith("{")))
        mine = j["tokens"][len(toks):]
        if args.arch == "dream":
            ref_out = run([args.ref, "generate-dream", args.model, ",".join(map(str, toks)), str(case.gen),
                           str(case.steps), args.threads])
        else:
            ref_out = run([args.ref, "generate", args.model, ",".join(map(str, toks)), str(case.gen),
                           str(case.block), str(case.steps), args.threads])
        ref = [int(x) for x in ref_out.strip().splitlines()[-1].split(",")][len(toks):]
        if len(mine) != case.gen or len(ref) != case.gen:
            raise SystemExit("refusing: a generation came back with the wrong length")
        same = sum(int(p == q) for p, q in zip(mine, ref))
        first = next((i for i, (p, q) in enumerate(zip(mine, ref)) if p != q), None)
        gens.append(dict(prompt=case.prompt, gen_length=case.gen, block_length=case.block,
                         steps=case.steps, matching_tokens=same, first_divergence=first,
                         mojo_text=j["text"]))
        print("generate", gens[-1])
    doc = dict(date=dt.datetime.now().isoformat(timespec="seconds"), arch=args.arch, device=args.device,
               model=pathlib.Path(args.model).name,
               versions={"mojo-dllm": run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).strip()},
               reference="llama.cpp libllama via tools/ref/llada_ref.cpp", logits=logits, generation=gens)
    if args.ref2:
        doc["reference2"] = args.ref2_label
    out = result_path(args.model, args.arch, args.device, dt.date.today())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
