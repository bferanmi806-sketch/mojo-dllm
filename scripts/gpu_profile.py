#!/usr/bin/env python3
"""Per-kernel GPU time of a `mojo-dllm run --device gpu`, from an Nsight Systems trace.

usage: python3 scripts/gpu_profile.py --model MODEL.gguf --label NAME [--steps 4] [--max-tokens 128]
                                      [--prompt TEXT] [--out bench/profiles/DATE-NAME.json]
                                      [--runtime mojo-dllm|llama.cpp] [--seq-len N]

The run is traced with `nsys profile --trace=cuda`, which records every kernel's
device start and end without synchronizing the stream, unlike
MOJO_DLLM_GPU_PROFILE=1. It needs no admin rights. The result groups kernels by
name (and the matrix multiplies by output size), divides by the number of
forward passes, and records host<->device copies, so a change to one kernel
can be compared before and after from committed evidence.

`--runtime llama.cpp` traces llama.cpp's `llama-diffusion-cli` (the CUDA build
in ~/src/llama.cpp/build-cuda, every layer offloaded) on the same prompt, so
the two runtimes' kernels can be compared group by group. llama.cpp replays
its forward pass as a CUDA graph, so the trace expands graph nodes; its
canvas is `--seq-len` tokens (mojo-dllm's prompt plus `--max-tokens`).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import power  # noqa: E402  (bench/power.py)
import spec  # noqa: E402  (bench/spec.py)
IMPORTERS = [
    "/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter",
    "/opt/nvidia/nsight-systems/host-linux-x64/QdstrmImporter",
]
GROUPS = [
    # Mojo truncates kernel symbols to six characters of the function name.
    ("q4k_mm", "gemm_q4k"),
    ("q6k_mm", "gemm_q6k"),
    ("attent", "attention"),
    ("quanti", "quantize"),
    ("rmsnor", "rmsnorm"),
    ("rope", "rope"),
    ("silu_m", "silu_mul"),
    ("add_bi", "add_bias"),
    ("add_ke", "add"),
    ("copy_r", "copy_rows"),
    ("mma_ti", "mma_tile"),
]
LLAMA_GROUPS = [
    ("mul_mat_q<(ggml_type)12", "gemm_q4k"),
    ("mul_mat_q<(ggml_type)14", "gemm_q6k"),
    ("mul_mat_q_stream_k_fixup", "gemm_fixup"),
    ("flash_attn", "attention"),
    ("quantize_mmq", "quantize"),
    ("rms_norm", "rmsnorm"),
    ("rope", "rope"),
    ("unary_gated_op_kernel<&op_silu", "silu_mul"),
    ("k_bin_bcast<&op_add", "add"),
    ("cpy_scalar", "copy_f16"),
    ("k_get_rows", "get_rows"),
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd[:4])} ...\n{(p.stderr or p.stdout)[-800:]}")
    return p


def group_of(name: str, groups: list[tuple[str, str]] = GROUPS) -> str:
    for key, label in groups:
        if key in name:
            return label
    return "other:" + re.sub(r"_[0-9a-f]{16}$", "", name)[:60]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--prompt", default=spec.PROMPTS[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=str(ROOT),
                    help="checkout whose build/mojo-dllm runs and whose commit is recorded (for A/B runs)")
    ap.add_argument("--runtime", choices=["mojo-dllm", "llama.cpp"], default="mojo-dllm")
    ap.add_argument("--seq-len", type=int, default=0, help="llama.cpp canvas length (prompt + generated)")
    ap.add_argument("--llama-cpp", default=os.path.expanduser("~/src/llama.cpp"))
    ap.add_argument("--allow-battery", action="store_true", help="profile on battery instead of pausing")
    args = ap.parse_args()
    importer = next((p for p in IMPORTERS if os.path.exists(p)), None)
    if importer is None:
        raise SystemExit("gpu_profile: QdstrmImporter not found; install nsight-systems")
    out_dir = ROOT / "scratch" / "prof"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / args.label
    root = pathlib.Path(args.root).resolve()
    llama = args.runtime == "llama.cpp"
    if llama:
        if args.seq_len <= 0:
            raise SystemExit("gpu_profile: --runtime llama.cpp needs --seq-len (prompt tokens + --max-tokens)")
        cmd = [args.llama_cpp + "/build-cuda/bin/llama-diffusion-cli", "-m", os.path.expanduser(args.model),
               "-p", args.prompt, "-c", str(args.seq_len), "-b", str(args.seq_len), "-ub", str(args.seq_len),
               "--diffusion-steps", str(args.steps), "--diffusion-eps", "0.001", "--temp", "0",
               "--seed", "42", "-ngl", "99", "-v"]
        graph = ["--cuda-graph-trace=node"]
    else:
        cmd = [str(root / "build/mojo-dllm"), "run", "--model", os.path.expanduser(args.model), "--prompt", args.prompt,
               "--max-tokens", str(args.max_tokens), "--steps", str(args.steps), "--block-length",
               str(args.block_length), "--device", "gpu", "--json", "--no-early-stop"]
        graph = []
    # On battery the trace waits for the charger; a trace the charger left midway is taken again.
    p, pinfo = power.run_on_mains(
        lambda: run(["nsys", "profile", "-o", str(base), "--force-overwrite", "true", "--trace=cuda",
                     "--sample=none", "--cpuctxsw=none"] + graph + cmd),
        allow_battery=args.allow_battery, log=lambda m: print(m, flush=True))
    if llama:
        text = p.stdout + p.stderr
        devices = re.findall(r"layer\s+\d+ assigned to device (\w+)", text)
        if not devices or any(not d.startswith("CUDA") for d in devices):
            raise SystemExit(f"gpu_profile: llama.cpp did not put every layer on the GPU: {sorted(set(devices))}")
        n_layers = len(devices) - 1  # the output layer is listed too
        m = re.search(r"total time: ([0-9.]+)ms, time per step: ([0-9.]+)ms", text)
        if not m:
            raise SystemExit("gpu_profile: llama.cpp printed no timing line")
        # One forward pass per diffusion step; llama.cpp reports no separate forward time.
        j = dict(forward_passes=args.steps, forward_ms=None, ms_per_step=float(m.group(2)),
                 prompt_tokens=None, gen_length=None)
    else:
        line = next((l for l in p.stdout.splitlines() if l.startswith("{")), None)
        if line is None:
            raise SystemExit("gpu_profile: the traced run printed no JSON")
        j = json.loads(line)
    forwards = j["forward_passes"]
    if forwards <= 0:
        raise SystemExit("gpu_profile: the run made no forward pass")
    rep = base.with_suffix(".nsys-rep")
    if rep.exists():
        rep.unlink()
    run([importer, "-i", str(base.with_suffix(".qdstrm"))])
    sqlite = base.with_suffix(".sqlite")
    if sqlite.exists():
        sqlite.unlink()
    # gpukerngbsum has one row per kernel and grid shape (GridXYZ, BlockXYZ), which
    # the GEMM grouping below reads; nsys 2023 cut those columns from gpukernsum.
    # --force-export: nsys 2023 reports the sqlite its first call wrote as stale to the second.
    stats = ["nsys", "stats", "--force-export=true", "--format", "csv", "--output", "-"]
    kern = run(stats + ["--report", "gpukerngbsum", str(rep)]).stdout
    mem = run(stats + ["--report", "gpumemtimesum", str(rep)]).stdout

    def rows(text: str) -> list[dict]:
        lines = [l for l in text.splitlines() if l.startswith(("Time", '"', "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))]
        if not lines or not lines[0].startswith("Time"):
            raise SystemExit("gpu_profile: nsys stats printed no table")
        return list(csv.DictReader(io.StringIO("\n".join(lines))))

    groups: dict[str, dict] = {}
    gemm_shapes: dict[str, dict] = {}
    for r in rows(kern):
        name = r["Name"]
        g = group_of(name, LLAMA_GROUPS if llama else GROUPS)
        ns = float(r["Total Time (ns)"])
        n = int(r["Instances"])
        med = float(r["Med (ns)"]) * n
        e = groups.setdefault(g, dict(ms_per_forward=0.0, median_ms_per_forward=0.0, launches_per_forward=0.0))
        e["ms_per_forward"] += ns / 1e6 / forwards
        # Median instance x launches: robust to a few slow instances when
        # something else briefly holds the GPU or the clocks dip.
        e["median_ms_per_forward"] += med / 1e6 / forwards
        e["launches_per_forward"] += n / forwards
        if g.startswith("gemm"):
            key = f"{g} grid {' x '.join(r['GridXYZ'].split())}"
            s = gemm_shapes.setdefault(key, dict(ms_per_forward=0.0, median_ms_per_forward=0.0,
                                                 launches_per_forward=0.0))
            s["ms_per_forward"] += ns / 1e6 / forwards
            s["median_ms_per_forward"] += med / 1e6 / forwards
            s["launches_per_forward"] += n / forwards
    # Copy totals cover the whole run, including the one-time weight upload,
    # so they are reported per run, not per forward pass.
    copies = {}
    for r in rows(mem):
        copies[r["Operation"].strip('"[]')] = dict(
            ms_total=float(r["Total Time (ns)"]) / 1e6, count=int(r["Count"]))
    if llama:
        # The forward count is assumed, not reported: attention runs once per
        # layer, so check it ran layers x steps times, or every per-forward
        # figure would be off.
        att = groups.get("attention", {}).get("launches_per_forward", 0)
        if abs(att - n_layers) > 1e-9:
            raise SystemExit(f"gpu_profile: {att} attention launches per forward, expected {n_layers}; "
                             f"the trace is not {forwards} forward passes")
    kernel_ms = sum(e["ms_per_forward"] for e in groups.values())
    if llama:
        lroot = args.llama_cpp
        versions = {"llama.cpp": run(["git", "-C", lroot, "rev-parse", "--short", "HEAD"]).stdout.strip()}
    else:
        version = run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"]).stdout.strip()
        dirty = bool(run(["git", "-C", str(root), "status", "--porcelain", "--", "src"]).stdout.strip())
        versions = {"mojo-dllm": version + ("-dirty" if dirty else "")}
    gpu = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]).stdout.strip()
    doc = dict(
        date=dt.datetime.now().isoformat(timespec="seconds"), label=args.label,
        runtime=args.runtime, versions=versions, gpu=gpu, on_battery=pinfo["on_battery"],
        power_source=pinfo["power_source"],
        power_retries=pinfo["power_retries"], model=pathlib.Path(args.model).name,
        seq_len=args.seq_len if llama else j["prompt_tokens"] + j["gen_length"],
        steps=args.steps, forward_passes=forwards,
        forward_ms_measured=j["forward_ms"], ms_per_step_measured=j["ms_per_step"],
        kernel_ms_per_forward=round(kernel_ms, 3),
        kernel_median_ms_per_forward=round(sum(e["median_ms_per_forward"] for e in groups.values()), 3),
        groups={k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["ms_per_forward"])},
        gemm_by_grid={k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in sorted(gemm_shapes.items(), key=lambda kv: -kv[1]["ms_per_forward"])},
        copies=copies,
        note="Traced with nsys --trace=cuda (no stream synchronization). Kernel times are device time per "
             "forward pass; copies are totals for the run, weight upload included.",
    )
    out = pathlib.Path(args.out) if args.out else ROOT / "bench" / "profiles" / f"{dt.date.today().isoformat()}-{args.label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(dict(kernel_ms_per_forward=doc["kernel_ms_per_forward"], forward_ms=j["forward_ms"],
                          groups={k: v["ms_per_forward"] for k, v in doc["groups"].items()}), indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
