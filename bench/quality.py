#!/usr/bin/env python3
"""How output quality and time move with the number of denoising steps.

usage: python3 bench/quality.py --model MODEL.gguf [--steps 32,64,128] [--gen 128] [--threads 12]
                                [--allow-battery] [--force]

A diffusion model commits gen_length / steps tokens per step. Few steps are
fast and commit several tokens in parallel, which shows up as repeated or
dropped words; more steps cost proportionally more time. This records both
for the same prompts so the trade-off is visible, and writes
bench/quality/<date>-<model>.json for scripts/bench_table.py to render.
On battery each run waits for the charger, and a run the charger left midway
is repeated (--allow-battery skips both and records `on_battery`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import atomic_write  # noqa: E402  (bench/atomic_write.py)
import power  # noqa: E402  (bench/power.py)
import spec  # noqa: E402  (bench/spec.py)

PROMPTS = list(spec.PROMPTS[:2])


def repeated_words(text: str) -> int:
    """Count immediate word repetitions ("the the"), a cheap proxy for parallel-decoding damage."""
    words = re.findall(r"[A-Za-z']+", text.lower())
    return sum(1 for a, b in zip(words, words[1:]) if a == b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", default="32,64,128")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--threads", default="12")
    ap.add_argument("--allow-battery", action="store_true", help="measure on battery instead of pausing")
    ap.add_argument("--force", action="store_true", help="replace an existing result file of the same name")
    args = ap.parse_args()
    model = os.path.expanduser(args.model)
    name = pathlib.Path(model).name.replace(".gguf", "")
    out_path = ROOT / "bench" / "quality" / f"{dt.date.today().isoformat()}-{name}.json"
    atomic_write.refuse_existing(out_path, args.force)   # before the runs, not after them
    rows = []
    for prompt in PROMPTS:
        for steps in [int(s) for s in args.steps.split(",")]:
            cmd = [str(ROOT / "build/mojo-dllm"), "run", "--model", model, "--prompt", prompt,
                   "--max-tokens", str(args.gen), "--steps", str(steps), "--threads", args.threads, "--json",
                   "--no-early-stop"]
            out, pinfo = power.run_on_mains(lambda: subprocess.run(cmd, capture_output=True, text=True),
                                            allow_battery=args.allow_battery, log=lambda m: print(m, flush=True))
            line = next((l for l in out.stdout.splitlines() if l.startswith("{")), None)
            if out.returncode != 0 or line is None:
                raise SystemExit(f"run failed for steps={steps}: {out.stderr[-400:]}")
            j = json.loads(line)
            rows.append(dict(
                prompt=prompt, steps=steps, tokens_per_step=args.gen / steps, generate_s=j["generate_s"],
                ms_per_step=j["ms_per_step"], repeated_words=repeated_words(j["text"]), text=j["text"],
                arch=j.get("arch", "llada"), power_retries=pinfo["power_retries"], on_battery=pinfo["on_battery"],
                power_source=pinfo["power_source"],
            ))
            print(f"steps {steps:4d}  {j['generate_s']:7.1f} s  repeats {rows[-1]['repeated_words']}  {j['text'][:80]!r}")
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    doc = dict(model=pathlib.Path(model).name, gen_length=args.gen, **spec.result_meta({"mojo-dllm": git.stdout.strip()}, args.threads), runs=rows)
    atomic_write.write_atomic(out_path, json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
