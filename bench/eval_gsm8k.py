#!/usr/bin/env python3
"""Paired GSM8K evaluation: does a sampler or cache change cost accuracy?

usage:
  python3 bench/eval_gsm8k.py run --model M.gguf --arm NAME [--device gpu] [--gen-length 256]
                                  [--block-length 32] [--steps N] [--limit K] [-- extra mojo-dllm flags]
  python3 bench/eval_gsm8k.py compare ARM_A ARM_B [--across-weights]
  python3 bench/eval_gsm8k.py --self-test

`run` answers the 200 problems in bench/eval/gsm8k-200.jsonl with `mojo-dllm
run --json` and appends one line per problem to bench/eval/runs/ARM.jsonl; a
stopped run resumes where it left off. The default is the clean baseline: one
token per denoising step (steps = generation length), greedy, LLaDA blocks of
32. While bench/eval/runs/PAUSE exists, `run` waits between problems. On
battery it waits for the charger, and a problem the charger left midway is
answered again, so `generate_s` is a mains-power time (--allow-battery skips
both and records `on_battery`). Every problem also runs inside an isolation window (`bench/isolation.py`): the
arm starts with a preflight of the machine and writes bench/eval/runs/ARM.env.json, a problem whose window
saw other CPU, GPU, swap or thermal activity is run again up to `--isolation-retries` times, and each row
records `isolation`. `--tune` sets the power profile to performance for the run; `--allow-noisy` continues
after a failed preflight.
Scoring takes the number after "The answer is", else the last number in
the output, and compares it with the gold answer.

`compare` pairs two arms problem by problem and writes
bench/eval/DATE-ARM_A-vs-ARM_B.json: both accuracies, the difference with a
95% confidence interval, McNemar's exact test on the discordant problems, and
the forward passes and seconds each arm spent. Passes average over the problems
that have a count (an engine that cannot count them writes null), and
`forward_passes_rows` says how many that is. The interval's width depends on how
many problems the arms disagree on: with 10 of 200 it is about +/- 3 points, so a smaller change is "not detectable", not "zero". An arm keeps one
configuration (flags, code version, model): resuming with another is refused,
and so is comparing arms that ran different models, unless `compare --across-weights` says the weights differ on
purpose (a GGUF against the authors' bf16 or 4-bit weights): the output then holds both model names and
`"weights_differ": true`, and the accuracy difference includes the quantization.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "eval" / "gsm8k-200.jsonl"
RUNS = ROOT / "bench" / "eval" / "runs"
sys.path.insert(0, str(ROOT / "bench"))
import isolation  # noqa: E402  (bench/isolation.py)
import power  # noqa: E402  (bench/power.py)

# While this file exists, `run` waits before starting the next problem, so a
# timing measurement can have the GPU to itself without stopping the arm.
PAUSE = RUNS / "PAUSE"
PROMPT = (
    "{question}\n\nSolve the problem step by step. End with a line of the form "
    '"The answer is N." where N is the final number.'
)
NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def extract(text: str) -> str | None:
    """The model's final answer as a canonical number string, or None."""
    m = re.findall(r"answer is\s*:?\s*\$?\s*(" + NUMBER + ")", text, flags=re.IGNORECASE)
    if not m:
        m = re.findall(NUMBER, text)
    if not m:
        return None
    return canon(m[-1])


def canon(s: str) -> str | None:
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else repr(v)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: binomial test of b vs c at p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_ci(b: int, c: int, n: int) -> tuple[float, float, float]:
    """Accuracy difference (B - A) and its 95% Wald interval for paired outcomes."""
    d = (c - b) / n
    se = math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n
    return d, d - 1.96 * se, d + 1.96 * se


def answered(out: pathlib.Path, arm: str, flags: object, version: str, model: str) -> set[int]:
    """The problem indexes already in an arm's file, refusing a file made with another configuration.

    A resumed arm must keep one configuration, or a paired comparison would mix two of them without saying so.
    """
    done: set[int] = set()
    if out.exists():
        for l in out.read_text().splitlines():
            r = json.loads(l)
            if (r.get("flags"), r.get("version"), r.get("model")) != (flags, version, model):
                raise SystemExit(
                    f"arm {arm!r} was started with flags={r.get('flags')} version={r.get('version')} "
                    f"model={r.get('model')}; this run has flags={flags} version={version} model={model}. "
                    "Use a new arm name.")
            done.add(r["index"])
    return done


def wait_if_paused() -> None:
    if PAUSE.exists():
        print(f"paused: waiting for {PAUSE.relative_to(ROOT)} to be removed", flush=True)
        while PAUSE.exists():
            time.sleep(10)


def cmd_run(args: argparse.Namespace, extra: list[str]) -> int:
    problems = [json.loads(l) for l in DATA.read_text().splitlines()]
    if args.limit:
        problems = problems[: args.limit]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"{args.arm}.jsonl"
    steps = args.steps or args.gen_length
    flags = ["--max-tokens", str(args.gen_length), "--block-length", str(args.block_length),
             "--steps", str(steps), "--device", args.device, "--json"] + extra
    version = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True).stdout.strip()
    version += "-dirty" if dirty else ""
    model = pathlib.Path(args.model).name
    done = answered(out, args.arm, flags, version, model)
    todo = [(k, p) for k, p in enumerate(problems) if p["index"] not in done]
    if not todo:
        return 0
    with isolation.arm_session(args, arm=args.arm, device=args.device, model=args.model, runs_dir=RUNS,
                               versions={"mojo-dllm": version}) as arm:
        for k, p in todo:
            wait_if_paused()
            cmd = [str(ROOT / "build/mojo-dllm"), "run", "--model", args.model,
                   "--prompt", PROMPT.format(question=p["question"])] + flags
            r, pinfo = arm.run(lambda: subprocess.run(cmd, capture_output=True, text=True))
            line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
            if r.returncode != 0 or line is None:
                raise SystemExit(f"problem {p['index']}: mojo-dllm failed (rc={r.returncode}): {r.stderr[-400:]}")
            j = json.loads(line)
            pred = extract(j["text"])
            rec = dict(index=p["index"], gold=p["answer"], pred=pred, correct=pred == canon(p["answer"]),
                       forward_passes=j["forward_passes"], generate_s=j["generate_s"],
                       stopped_early=j.get("stopped_early"), text=j["text"], flags=flags, version=version,
                       model=model, power_retries=pinfo["power_retries"], on_battery=pinfo["on_battery"],
                       power_source=pinfo["power_source"], isolation=pinfo["isolation"])
            with out.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            noisy = "" if pinfo["isolation"]["clean"] else " NOISY: " + "; ".join(pinfo["isolation"]["reasons"])
            print(f"[{k + 1}/{len(problems)}] #{p['index']} {'ok ' if rec['correct'] else 'BAD'} "
                  f"pred={pred} gold={p['answer']} passes={j['forward_passes']} {j['generate_s']:.1f}s{noisy}",
                  flush=True)
    return 0


def load_arm(name: str) -> dict[int, dict]:
    path = RUNS / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"no runs for arm {name!r} at {path}")
    return {r["index"]: r for r in map(json.loads, path.read_text().splitlines())}


def arm_config(name: str, rows: dict[int, dict]) -> dict:
    """The arm's one configuration; refuse an arm that mixes several."""
    configs = {(json.dumps(r.get("flags")), r.get("version"), r.get("model")) for r in rows.values()}
    if len(configs) != 1:
        raise SystemExit(f"arm {name!r} mixes configurations: {sorted(configs)}")
    flags, version, model = configs.pop()
    return dict(flags=json.loads(flags), version=version, model=model)


def first_rows(rows: dict[int, dict], limit: int) -> dict[int, dict]:
    """The first `limit` problems in run order (all of them for 0): one round of a
    rerun-safe arm, e.g. the round that had the machine to itself."""
    if limit < 0:
        raise SystemExit("--limit must be 0 (all) or positive")
    return dict(list(rows.items())[:limit]) if limit else rows


def passes_of(rows: dict[int, dict], common: list[int]) -> list[int]:
    """The forward-pass counts the arm recorded; an engine that cannot count them writes null."""
    return [rows[i]["forward_passes"] for i in common if rows[i]["forward_passes"] is not None]


def passes_mean(rows: dict[int, dict], common: list[int]) -> float | None:
    """Mean passes over the problems that have a count, or None when none has."""
    got = passes_of(rows, common)
    return sum(got) / len(got) if got else None


def is_clean(row: dict) -> bool:
    """True only for a row whose isolation window said clean; a row with no window is not known to be clean."""
    return (row.get("isolation") or {}).get("clean") is True


def clean_summary(A: dict[int, dict], B: dict[int, dict], common: list[int],
                  names: tuple[str, str] = ("a", "b")) -> dict:
    """Timing over the problems whose window was clean in both arms, and how many rows each arm had that were not.

    `generate_s_mean_clean` is None for an arm when no problem is clean in both (never 0). Rows written before
    windows existed count as `unchecked_rows`. Accuracy is not touched: it stays over every common row."""
    both = [i for i in common if is_clean(A[i]) and is_clean(B[i])]
    na, nb = names

    def mean(rows: dict[int, dict]) -> float | None:
        return sum(rows[i]["generate_s"] for i in both) / len(both) if both else None

    def noisy(rows: dict[int, dict]) -> int:
        return sum(1 for i in common if "isolation" in rows[i] and not is_clean(rows[i]))

    def unchecked(rows: dict[int, dict]) -> int:
        return sum(1 for i in common if "isolation" not in rows[i])

    return {"generate_s_mean_clean": {na: mean(A), nb: mean(B)}, "clean_rows": len(both),
            "noisy_rows": {na: noisy(A), nb: noisy(B)}, "unchecked_rows": {na: unchecked(A), nb: unchecked(B)}}


def cmd_compare(a_name: str, b_name: str, limit: int = 0, across_weights: bool = False) -> int:
    A, B = first_rows(load_arm(a_name), limit), first_rows(load_arm(b_name), limit)
    ca, cb = arm_config(a_name, A), arm_config(b_name, B)
    weights_differ = ca["model"] != cb["model"]
    if weights_differ and not across_weights:
        raise SystemExit(f"the arms ran different models: {ca['model']} vs {cb['model']}; "
                         "pass --across-weights to compare different weights (quantization then counts in the difference)")
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit("the two arms share no problem")
    n = len(common)
    b = sum(1 for i in common if A[i]["correct"] and not B[i]["correct"])
    c = sum(1 for i in common if not A[i]["correct"] and B[i]["correct"])
    d, lo, hi = paired_ci(b, c, n)
    doc = dict(
        date=dt.datetime.now().isoformat(timespec="seconds"), problems=n, arms=[a_name, b_name], limit=limit,
        problems_in_arm={a_name: len(A), b_name: len(B)},
        config={a_name: ca, b_name: cb},
        models={a_name: ca["model"], b_name: cb["model"]}, weights_differ=weights_differ,
        accuracy={a_name: sum(A[i]["correct"] for i in common) / n, b_name: sum(B[i]["correct"] for i in common) / n},
        difference=d, ci95=[lo, hi], discordant={"a_only": b, "b_only": c}, mcnemar_p=mcnemar_exact(b, c),
        forward_passes_mean={a_name: passes_mean(A, common), b_name: passes_mean(B, common)},
        forward_passes_rows={a_name: len(passes_of(A, common)), b_name: len(passes_of(B, common))},
        generate_s_mean={a_name: sum(A[i]["generate_s"] for i in common) / n,
                         b_name: sum(B[i]["generate_s"] for i in common) / n},
        **clean_summary(A, B, common, (a_name, b_name)),
    )
    suffix = f"-first{limit}" if limit else ""
    out = ROOT / "bench" / "eval" / f"{dt.date.today().isoformat()}-{a_name}-vs-{b_name}{suffix}.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: doc[k] for k in ("problems", "models", "weights_differ", "accuracy", "difference", "ci95", "discordant",
                                           "mcnemar_p", "forward_passes_mean", "forward_passes_rows",
                                           "generate_s_mean", "generate_s_mean_clean", "clean_rows",
                                           "noisy_rows", "unchecked_rows")}, indent=1))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


def self_test() -> int:
    cases = [
        ("so 5 + 3 = 8. The answer is 8.", "8"),
        ("The answer is $1,250.", "1250"),
        ("the answer is: 12.50", "12.5"),
        ("first 3, then 7, finally 42", "42"),
        ("The answer is 18. Then I add 2", "18"),
        ("no number here", None),
        ("The answer is -4.", "-4"),
    ]
    bad = [(t, want, extract(t)) for t, want in cases if extract(t) != want]
    assert canon("18.00") == "18" and canon("1,000") == "1000", "canon"
    assert abs(mcnemar_exact(0, 10) - 2 / 1024) < 1e-12, "mcnemar 0/10"
    assert mcnemar_exact(5, 5) == 1.0 and mcnemar_exact(0, 0) == 1.0, "mcnemar ties"
    d, lo, hi = paired_ci(2, 6, 200)
    assert abs(d - 0.02) < 1e-12 and lo < d < hi, "paired ci"
    order = {7: {"i": 0}, 3: {"i": 1}, 9: {"i": 2}}
    assert list(first_rows(order, 2)) == [7, 3], "first_rows keeps run order"
    assert first_rows(order, 0) is order and len(first_rows(order, 5)) == 3, "first_rows bounds"
    lines = [json.loads(l) for l in DATA.read_text().splitlines()]
    assert len(lines) == 200 and all(canon(r["answer"]) is not None for r in lines), "dataset"
    if bad:
        for t, want, got in bad:
            print(f"extract({t!r}) = {got!r}, want {want!r}")
        return 1
    print(f"eval_gsm8k self-test: {len(cases)} extraction cases, statistics and dataset ok")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    argv = sys.argv[1:]
    extra: list[str] = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--device", default="gpu")
    r.add_argument("--gen-length", type=int, default=256)
    r.add_argument("--block-length", type=int, default=32)
    r.add_argument("--steps", type=int, default=0)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--allow-battery", action="store_true", help="answer on battery instead of pausing")
    isolation.add_arguments(r)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--limit", type=int, default=0, help="compare only each arm's first N problems in run order")
    c.add_argument("--across-weights", action="store_true",
                   help="compare arms that ran different model files (a GGUF against bf16 weights); the output says "
                        "weights_differ and an accuracy difference then includes quantization")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args, extra)
    return cmd_compare(args.a, args.b, args.limit, args.across_weights)


if __name__ == "__main__":
    sys.exit(main())
