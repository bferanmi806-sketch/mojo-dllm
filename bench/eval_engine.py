#!/usr/bin/env python3
"""GSM8K for engines other than mojo-dllm: the same 200 problems, prompt, scoring and arm file.

usage:
  python3 bench/eval_engine.py run --engine NAME --model M.gguf --arm ARM [--device cpu|gpu]
                                   [--gen-length 256] [--block-length 32] [--steps N] [--limit K]
                                   [engine options] [-- extra engine flags]
  python3 bench/eval_gsm8k.py compare MOJO_ARM ARM

`run` answers the problems in bench/eval/gsm8k-200.jsonl through the engine and appends one line per problem to
bench/eval/runs/ARM.jsonl, with the keys `eval_gsm8k.py run` writes, so `eval_gsm8k.py compare` pairs the arm with a
mojo-dllm arm on the same model file. The prompt, the answer extraction, the PAUSE file, the mains-power wait
and the resume rule (an arm keeps one configuration) are `eval_gsm8k`'s own, and so is the isolation
(`bench/isolation.py`): a preflight at the start of the arm, ARM.env.json, a window and repeats for a noisy
problem, `--tune`, `--allow-noisy` and `--isolation-retries`.

A problem is never scored when the engine exits non-zero, prints no timing, or returns no text: the run stops and
names the problem. Engines are in bench/engines/; `--engine llamacpp --help` lists llama.cpp's options.
`--steps` defaults to the generation length (one token per step), `--block-length` is the block size of the
`block` schedule.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import eval_gsm8k  # noqa: E402  (bench/eval_gsm8k.py)
import engines  # noqa: E402
import isolation  # noqa: E402  (bench/isolation.py)
from engines import EngineError, GenConfig  # noqa: E402


DEFAULT_TIMEOUT_S = 1800.0     # a 256-step LLaDA problem takes about 1 to 2 minutes on the GPU, CPU about 25


def cmd_run(args: argparse.Namespace, extra: list[str]) -> int:
    problems = [json.loads(l) for l in eval_gsm8k.DATA.read_text().splitlines()]
    if args.limit:
        problems = problems[: args.limit]
    cfg = GenConfig(model=args.model, device=args.device, gen_length=args.gen_length,
                    steps=args.steps or args.gen_length, block_length=args.block_length,
                    schedule=getattr(args, "schedule", "timestep"), extra=tuple(extra), timeout_s=args.timeout_s)
    engine = engines.get(args.engine)(args)
    try:
        engine.check(cfg)
        version = engine.version()
    except EngineError as e:
        raise SystemExit(f"{args.engine}: {e}")
    flags = engine.flags(cfg)
    # An engine that loads its weights in `check` (hf) has no model file to size: it says what it still needs.
    sizing = {"ram_gb": engine.ram_gb_after_load(cfg)} if hasattr(engine, "ram_gb_after_load") else {}
    # `model` is what `compare` reads to tell weights apart; an engine may add to it (hf adds the quantization)
    model = engine.model_name(cfg) if hasattr(engine, "model_name") else pathlib.Path(args.model).name
    eval_gsm8k.RUNS.mkdir(parents=True, exist_ok=True)
    out = eval_gsm8k.RUNS / f"{args.arm}.jsonl"
    done = eval_gsm8k.answered(out, args.arm, flags, version, model)
    todo = [(k, p) for k, p in enumerate(problems) if p["index"] not in done]
    if not todo:
        return 0
    with isolation.arm_session(args, arm=args.arm, device=args.device, model=args.model, runs_dir=eval_gsm8k.RUNS,
                               versions={args.engine: version}, **sizing) as arm:
        for k, p in todo:
            eval_gsm8k.wait_if_paused()
            prompt = eval_gsm8k.PROMPT.format(question=p["question"])
            try:
                g, pinfo = arm.run(lambda: engine.generate(prompt, cfg))
            except EngineError as e:
                raise SystemExit(f"problem {p['index']}: {args.engine} failed: {e}")
            if not isinstance(g.get("text"), str) or not g["text"].strip():
                raise SystemExit(f"problem {p['index']}: {args.engine} returned no text; not scored")
            pred = eval_gsm8k.extract(g["text"])
            rec = dict(index=p["index"], gold=p["answer"], pred=pred, correct=pred == eval_gsm8k.canon(p["answer"]),
                       forward_passes=g["forward_passes"], generate_s=g["generate_s"],
                       stopped_early=g.get("stopped_early"), text=g["text"], flags=flags, version=version,
                       model=model, power_retries=pinfo["power_retries"], on_battery=pinfo["on_battery"],
                       power_source=pinfo["power_source"], isolation=pinfo["isolation"])
            with out.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            noisy = "" if pinfo["isolation"]["clean"] else " NOISY: " + "; ".join(pinfo["isolation"]["reasons"])
            print(f"[{k + 1}/{len(problems)}] #{p['index']} {'ok ' if rec['correct'] else 'BAD'} "
                  f"pred={pred} gold={p['answer']} passes={g['forward_passes']} {g['generate_s']:.1f}s{noisy}",
                  flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--engine")
    name = pre.parse_known_args(argv)[0].engine
    ap = argparse.ArgumentParser(description="GSM8K for engines other than mojo-dllm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", required=True, help=f"one of: {', '.join(engines.names())}")
    r.add_argument("--model", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    r.add_argument("--gen-length", type=int, default=256)
    r.add_argument("--block-length", type=int, default=32)
    r.add_argument("--steps", type=int, default=0)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S,
                   help="fail the run when one generation takes longer than this (default %(default)s)")
    r.add_argument("--allow-battery", action="store_true", help="answer on battery instead of pausing")
    isolation.add_arguments(r)
    if name:
        engines.get(name).add_arguments(r)
    args = ap.parse_args(argv)
    return cmd_run(args, extra)


if __name__ == "__main__":
    sys.exit(main())
