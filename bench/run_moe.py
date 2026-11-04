#!/usr/bin/env python3
"""Record every sample from bench_moe, including full CPU/machine metadata."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command(args: list[str]) -> str:
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False)
        return (result.stdout + result.stderr).strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def parse_run(text: str, reps: int, expected: dict) -> dict:
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    setup = [row for row in rows if row.get("kind") == "setup"]
    routing = [row for row in rows if row.get("kind") == "routing"]
    samples = [row for row in rows if row.get("kind") == "sample"]
    if len(setup) != 1 or len(routing) != 1 or len(rows) != reps + 3:
        raise ValueError("benchmark output must contain setup, routing and every sample")
    if any(setup[0].get(key) != value for key, value in expected.items()):
        raise ValueError("benchmark measured a different shape or thread count")
    if [row.get("index") for row in samples] != list(range(-1, reps)):
        raise ValueError("missing, duplicate or reordered benchmark samples")
    for row in samples:
        if row.get("warmup") is not (row["index"] == -1):
            raise ValueError("warmup marker disagrees with sample index")
        if not all(math.isfinite(row[key]) for key in ("seconds", "checksum", "max_abs")):
            raise ValueError("non-finite benchmark timing or output")
        if row["seconds"] <= 0 or row["max_abs"] <= 0:
            raise ValueError("benchmark timing and output magnitude must be positive")
    if len({row["checksum"] for row in samples}) != 1:
        raise ValueError("repeated deterministic inputs produced different checksums")
    if routing[0].get("token_expert_pairs") != expected["tokens"] * expected["top_k"]:
        raise ValueError("routing did not preserve all token-expert pairs")
    if not 1 <= routing[0]["active_experts"] <= expected["experts"]:
        raise ValueError("invalid active-expert count")
    if not 1 <= routing[0]["max_tokens_per_expert"] <= expected["tokens"]:
        raise ValueError("invalid per-expert token count")
    seconds = [row["seconds"] for row in samples if not row["warmup"]]
    return {
        "setup": setup[0],
        "routing": routing[0],
        "samples": samples,
        "median_seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/bench_moe"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=2816)
    parser.add_argument("--ffn", type=int, default=704)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads, args.tokens, args.reps, args.hidden, args.ffn, args.experts, args.top_k) <= 0:
        parser.error("all dimensions, thread counts and repetition counts must be positive")
    if args.hidden % 256 or args.ffn % 32 or args.top_k > args.experts:
        parser.error("hidden must divide by 256, FFN by 32, and top-k must not exceed experts")
    if args.out.exists():
        parser.error(f"output already exists: {args.out}; choose a new path")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    report = {
        "schema_version": 1,
        "benchmark": "synthetic_moe_layer",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "measurement": "router plus grouped experts, including activation quantization",
        "excluded": "weight generation, packing, workspace allocation, one warmup",
        "synthetic_seed": [630001, 630002, 630003],
        "machine": {
            "platform": platform.platform(),
            "lscpu": command(["lscpu"]),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "load_average_before": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "mojo": command([os.environ.get("MOJO", "mojo"), "--version"]),
            "git_head": command(["git", "-C", str(ROOT), "--work-tree", str(ROOT), "rev-parse", "HEAD"]),
            "git_status": command(["git", "-C", str(ROOT), "--work-tree", str(ROOT), "status", "--short"]),
            "omp_num_threads": env["OMP_NUM_THREADS"],
        },
        "runs": [],
    }
    for down_type, label in ((6, "Q5_0"), (8, "Q8_0")):
        expected = {
            "threads": args.threads, "tokens": args.tokens, "hidden": args.hidden,
            "ffn": args.ffn, "experts": args.experts, "top_k": args.top_k,
            "down_type": down_type,
        }
        argv = [str(args.binary.resolve()), str(args.threads), str(args.tokens),
                str(args.reps), str(down_type), str(args.hidden), str(args.ffn),
                str(args.experts), str(args.top_k)]
        result = subprocess.run(argv, env=env, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"{label} benchmark exited {result.returncode}\n{result.stdout}\n{result.stderr}")
        parsed = parse_run(result.stdout, args.reps, expected)
        report["runs"].append({"down_format": label, "command": argv,
                               "stdout": result.stdout, "stderr": result.stderr,
                               "exit_code": result.returncode, **parsed})
        print(f"{label}: median {parsed['median_seconds'] * 1000:.3f} ms "
              f"over {args.reps} samples", flush=True)
    report["machine"]["load_average_after"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(args.out)


if __name__ == "__main__":
    main()
