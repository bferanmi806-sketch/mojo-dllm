#!/usr/bin/env python3
"""End-to-end benchmark: mojo-dllm vs llama.cpp vs diffuse-cpp on LLaDA-8B Q4_K_M.

usage: python3 bench/run_bench.py --config bench/config.json [--reps 3] [--out bench/results/NAME.json]

Every runtime does the same work per prompt: a canvas of L = P + G tokens
(P prompt tokens, G generated), and the same number of full-canvas forward
passes S. Block layout and sampler details differ by runtime and are recorded
next to the numbers; the forward passes, which are over 95% of the time, are
identical in shape.

Measurement hygiene, all recorded in the output:
  * runs are interleaved (A B C, A B C, ...) so drift hits every runtime;
  * each run starts when a two-second sample shows the CPU under `busy_threshold`
    busy, or after `max_quiet_wait_s` seconds; the busy fraction is recorded as
    `cpu_busy_before`. The check is the shared one in bench/isolation.py, which also looks at pressure
    stalls, swap and, for a GPU runtime, the GPU's other processes;
  * each run happens inside an isolation window (bench/isolation.py): other CPU, swap, thermal or GPU
    activity makes the run `noisy` and it is repeated up to `--isolation-retries` times (default 2); the
    window is recorded as `isolation`, and the preflight and a system snapshot go to
    bench/results/env/NAME.env.json. `--allow-noisy` continues after a failed preflight;
  * a watchdog kills any process named exactly `pytest` while the benchmark
    runs (other sessions on this machine launch test suites that would
    otherwise compete for the CPU) and logs every kill. A config can turn it
    off with "kill_pytest": false, as the GPU configs do: a GPU run barely
    touches the CPU, and killing someone's tests for it is not worth it;
  * the power profile is set to `performance` and restored on exit.

Nothing here estimates a number. A run that fails or prints no timing is
recorded as failed and excluded, and the summary says how many were.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import platform
import re
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import isolation  # noqa: E402  (bench/isolation.py)
import power  # noqa: E402  (bench/power.py)


class Watchdog(threading.Thread):
    def __init__(self, name: str, log: list[dict]) -> None:
        super().__init__(daemon=True)
        self.target = name
        self.log = log
        self.stop = threading.Event()

    def run(self) -> None:
        me = os.getpid()
        while not self.stop.is_set():
            for d in pathlib.Path("/proc").iterdir():
                if not d.name.isdigit() or int(d.name) == me:
                    continue
                try:
                    comm = (d / "comm").read_text().strip()
                except OSError:
                    continue
                if comm != self.target:
                    continue
                pid = int(d.name)
                try:
                    os.kill(pid, signal.SIGKILL)
                    # Time and pid only: where the process ran is someone else's business.
                    self.log.append(dict(time=dt.datetime.now().isoformat(timespec="seconds"), pid=pid))
                except ProcessLookupError:
                    pass
            self.stop.wait(1.0)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def mojo_version() -> str:
    """The compiler version recorded with the results: the one uv.lock pins."""
    p = subprocess.run(["uv", "run", "--frozen", "mojo", "--version"], cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        raise SystemExit(f"run_bench: `uv run --frozen mojo --version` failed: {p.stderr.strip() or 'no output'}")
    return p.stdout.strip().splitlines()[-1]


def cpu_busy(sample_s: float = 2.0) -> float:
    """Fraction of all CPU time spent busy over a short sample (0..1); see isolation.cpu_busy."""
    return isolation.cpu_busy(sample_s)


def wait_for_quiet(threshold: float, max_wait: float) -> tuple[float, float]:
    """(busy fraction, seconds waited): start under `threshold`, or after `max_wait`; see isolation.wait_for_quiet."""
    return isolation.wait_for_quiet(threshold, max_wait)


def timed(cmd: list[str]) -> tuple[int, str, str, float, int]:
    """Run under /usr/bin/time; returns rc, stdout, stderr, wall s, peak RSS bytes."""
    with tempfile.NamedTemporaryFile("r", suffix=".time") as tf:
        full = ["/usr/bin/time", "-f", "%e %M", "-o", tf.name] + cmd
        p = subprocess.run(full, capture_output=True, text=True)
        parts = pathlib.Path(tf.name).read_text().strip().splitlines()[-1].split()
        return p.returncode, p.stdout, p.stderr, float(parts[0]), int(parts[1]) * 1024


def run_mojo(c: dict, prompt: str, threads: int, extra: list[str], n_prompt: int | None = None) -> dict:
    cmd = [
        str(ROOT / "build/mojo-dllm"), "run", "--model", c["models"]["gguf"], "--prompt", prompt,
        "--max-tokens", str(c["gen_length"]), "--steps", str(c["steps"]),
        "--block-length", str(c["block_length"]), "--threads", str(threads), "--json",
        # Every runtime does every step, so ms / step and generation time compare.
        "--no-early-stop",
    ] + extra
    rc, out, err, wall, rss = timed(cmd)
    line = next((l for l in out.splitlines() if l.startswith("{")), None)
    if rc != 0 or line is None:
        return dict(ok=False, rc=rc, error=(err or out)[-400:])
    j = json.loads(line)
    # Equal work is what the comparison rests on: every runtime does c["steps"] passes over the same prompt.
    if j["forward_passes"] != c["steps"]:
        return dict(ok=False, rc=rc, error=f"mojo-dllm did {j['forward_passes']} forward passes, not {c['steps']}; "
                                           "the runtimes would not do equal work")
    if n_prompt is not None and j["prompt_tokens"] != n_prompt:
        return dict(ok=False, rc=rc, error=f"mojo-dllm counted {j['prompt_tokens']} prompt tokens, "
                                           f"`tokenize` printed {n_prompt}")
    return dict(
        ok=True, wall_s=wall, peak_rss_bytes=rss, generate_s=j["generate_s"],
        ms_per_step=j["ms_per_step"], forward_passes=j["forward_passes"],
        startup_s=wall - j["generate_s"], seq_len=j["prompt_tokens"] + j["gen_length"],
        prompt_tokens=j["prompt_tokens"], text=j["text"],
        forward_ms=j.get("forward_ms"),
        gemm_ms_per_forward=j["gemm_ms_per_forward"], attention_ms_per_forward=j["attention_ms_per_forward"],
        lm_head_ms_per_forward=j["lm_head_ms_per_forward"],
    )


def cuda_layer_error(output: str) -> str | None:
    """Why a CUDA llama.cpp run does not count as one, or None when every layer was on the GPU.

    A CUDA build that fails to find the GPU falls back to the CPU and still prints a timing; that is
    refused rather than labelled CUDA. With -v, llama.cpp names the device of every layer it loads.
    """
    devices = re.findall(r"layer\s+\d+ assigned to device (\w+)", output)
    if not devices or any(not d.startswith("CUDA") for d in devices):
        return f"CUDA run did not put every layer on the GPU: {sorted(set(devices))}"
    return None


def parse_llama_output(output: str) -> dict | None:
    """llama-diffusion-cli's timing line and the text after it, or None when there is no timing line.

    diffusion-cli prints the timing line from inside the generator, then logs the text; its trailing
    llama_perf/teardown lines are dropped, and so are the timestamped log lines that -v (CUDA runs) adds,
    such as "0.01.487.312 D load_tensors: ...".
    """
    # newer llama.cpp appends ", sampling time per step: Zms" to the line; it is not text
    m = re.search(r"total time: ([0-9.]+)ms, time per step: ([0-9.]+)ms(?:, sampling time per step: [0-9.]+ms)?",
                  output)
    if not m:
        return None
    text = "\n".join(
        l for l in output[m.end():].splitlines()
        if not l.startswith(("llama_", "common_", "main:")) and not re.match(r"\d+\.\d+\.\d+\.\d+ [DIWE] ", l)
    ).strip()
    return dict(total_s=float(m.group(1)) / 1000.0, ms_per_step=float(m.group(2)), text=text)


def run_llama(c: dict, prompt: str, threads: int, seq_len: int, cuda: bool = False) -> dict:
    """llama.cpp's diffusion example; with cuda, the build-cuda tree and every layer offloaded."""
    build = "build-cuda" if cuda else "build"
    cmd = [
        c["llama_cpp"] + f"/{build}/bin/llama-diffusion-cli", "-m", c["models"]["gguf"], "-p", prompt,
        "-c", str(seq_len), "-b", str(seq_len), "-ub", str(seq_len),
        "--diffusion-steps", str(c["steps"]), "--diffusion-eps", "0.001",
        "--temp", "0", "-t", str(threads), "-tb", str(threads), "--seed", "42",
    ] + (["-ngl", "99", "-v"] if cuda else [])
    rc, out, err, wall, rss = timed(cmd)
    if cuda:
        bad = cuda_layer_error(err + out)
        if bad:
            return dict(ok=False, rc=rc, error=bad)
    parsed = parse_llama_output(err + out)
    if rc != 0 or not parsed:
        return dict(ok=False, rc=rc, error=(err or out)[-400:])
    total = parsed["total_s"]
    return dict(
        ok=True, wall_s=wall, peak_rss_bytes=rss, generate_s=total,
        ms_per_step=parsed["ms_per_step"], forward_passes=c["steps"], forward_passes_measured=False,
        startup_s=wall - total, seq_len=seq_len, text=parsed["text"][-600:],
    )


def run_llama_bench(c: dict, threads: int, seq_len: int, cuda: bool = False) -> dict:
    """One forward pass over seq_len tokens, from llama.cpp's own llama-bench.

    This is the kernels alone: no sampler and no logits beyond the last
    position, against mojo-dllm's measured forward pass. llama-bench applies a
    causal mask, which makes its attention a little cheaper than the
    bidirectional attention a diffusion model needs.
    """
    build = "build-cuda" if cuda else "build"
    cmd = [
        c["llama_cpp"] + f"/{build}/bin/llama-bench", "-m", c["models"]["gguf"], "-p", str(seq_len),
        "-n", "0", "-b", str(seq_len), "-ub", str(seq_len), "-t", str(threads), "-r", "3", "-o", "json",
    ] + (["-ngl", "99"] if cuda else ["-ngl", "0"])
    rc, out, err, wall, rss = timed(cmd)
    try:
        rows = json.loads(out)
        r = rows[0]
        if r["n_prompt"] != seq_len:
            raise ValueError(f"llama-bench ran {r['n_prompt']} tokens, not {seq_len}")
        if cuda and "CUDA" not in str(r.get("backends", "")):
            raise ValueError(f"llama-bench ran on {r.get('backends')!r}, not CUDA")
    except (ValueError, KeyError, IndexError) as e:
        return dict(ok=False, rc=rc, error=f"{e}: {(err or out)[-300:]}")
    if rc != 0:
        return dict(ok=False, rc=rc, error=(err or out)[-400:])
    return dict(ok=True, wall_s=wall, peak_rss_bytes=rss, forward_ms=r["avg_ns"] / 1e6,
                forward_samples_ms=[x / 1e6 for x in r["samples_ns"]], seq_len=seq_len,
                n_threads=r["n_threads"])


def run_diffuse(c: dict, token_ids: str, threads: int, prompt_tokens: int, mode: str) -> dict:
    """diffuse-cli prints no timings, so time its generate call from outside.

    It writes "Generating ..." to stderr right before diffuse_generate() and
    "Generated token IDs" right after; timestamping those two markers as the
    stream arrives brackets exactly the generation, excluding model load.
    """
    cmd = [
        c["diffuse_cpp"] + "/build/diffuse-cli", "-m", c["models"]["diffuse_gguf"], "--tokens", token_ids,
        "-n", str(c["gen_length"]), "-s", str(c["steps"]), "-t", str(threads),
    ]
    if mode == "equal":
        cmd += ["--no-cache", "--remasking", "low_confidence"]
    else:
        cmd += ["--remasking", "entropy_exit"]
    with tempfile.NamedTemporaryFile("r", suffix=".time") as tf:
        t_launch = time.perf_counter()
        proc = subprocess.Popen(["/usr/bin/time", "-f", "%e %M", "-o", tf.name] + cmd,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stderr is not None and proc.stdout is not None
        err = b""
        t_start = t_end = None
        last_step = 0
        while True:
            chunk = os.read(proc.stderr.fileno(), 65536)
            if not chunk:
                break
            now = time.perf_counter()
            err += chunk
            if t_start is None and b"Generating " in err:
                t_start = now
            if t_end is None and b"Generated token IDs" in err:
                t_end = now
        out = proc.stdout.read().decode(errors="replace")
        rc = proc.wait()
        wall_s, rss_kb = pathlib.Path(tf.name).read_text().strip().splitlines()[-1].split()
    text_err = err.decode(errors="replace")
    steps = re.findall(r"step (\d+)/(\d+)", text_err)
    if steps:
        last_step = int(steps[-1][0])
    if rc != 0 or t_start is None or t_end is None:
        return dict(ok=False, rc=rc, error=text_err[-600:])
    if last_step < 1:
        # ms / step would be divided by a step count nothing measured: refuse the run instead.
        return dict(ok=False, rc=rc, error="diffuse-cli printed no `step i/N` line, so its step count is unknown: "
                                           + text_err[-300:])
    total = t_end - t_start
    n_steps = last_step
    return dict(
        ok=True, wall_s=float(wall_s), peak_rss_bytes=int(rss_kb) * 1024, generate_s=total,
        ms_per_step=total * 1000.0 / n_steps, forward_passes=n_steps,
        startup_s=t_start - t_launch, seq_len=prompt_tokens + c["gen_length"],
        tokens=out.strip()[-800:],
    )


def parse_token_ids(out: str) -> tuple[str, int]:
    """The comma-separated ids on the last line of `mojo-dllm tokenize`, and how many there are.

    Anything but `[1, 2, 3]` (empty output, a log line) is refused: an empty line used to read as one token and
    gave llama.cpp a wrong context size.
    """
    lines = out.strip().splitlines()
    ids = lines[-1].strip().strip("[]").replace(" ", "") if lines else ""
    if not re.fullmatch(r"\d+(,\d+)*", ids):
        raise SystemExit(f"run_bench: `mojo-dllm tokenize` did not print token ids on its last line: {out[-200:]!r}")
    return ids, len(ids.split(","))


def select_runtimes(configured: list[str], only: str, reps: int) -> list[str]:
    """The runtimes to run, refusing a selection that would write a result file with nothing measured."""
    if reps < 1:
        raise SystemExit(f"run_bench: --reps must be 1 or more, got {reps}")
    wanted = [n for n in only.split(",") if n] if only else []
    if only and not wanted:
        raise SystemExit(f"run_bench: --only {only!r} names no runtime")
    unknown = [n for n in wanted if n not in configured]
    if unknown:
        raise SystemExit(f"run_bench: --only names {unknown}, which the config does not list: {configured}")
    chosen = [r for r in configured if not wanted or r in wanted]
    if not chosen:
        raise SystemExit("run_bench: no runtime selected; the config lists none")
    return chosen


def median_or_none(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def window_state(r: dict) -> str:
    """`clean`, `noisy` or `unchecked` for one run, from its isolation window.

    A run with no window (a file from before the windows existed, or a runner that did not record one) is
    `unchecked`: nothing says the machine was quiet, so it never counts as clean.
    """
    iso = r.get("isolation")
    if not isinstance(iso, dict):
        return "unchecked"
    return "clean" if iso.get("clean") is True and not iso.get("noisy") else "noisy"


def summarize_runs(results: list[dict], runtimes: list[str], gen_length: int) -> dict:
    """The per-runtime medians over every ok run, and how many of those runs were clean, noisy or unchecked.

    A noisy run stays in the medians (it is data, and dropping it would hide that the window was noisy);
    `noisy_runs` and `unchecked_runs` say how many there were, so the table can disclose them.
    """
    summary = {}
    for rt in runtimes:
        ok = [r for r in results if r["runtime"] == rt and r["ok"]]
        bad = [r for r in results if r["runtime"] == rt and not r["ok"]]
        states = [window_state(r) for r in ok]
        summary[rt] = dict(
            runs_ok=len(ok), runs_failed=len(bad),
            clean_runs=states.count("clean"), noisy_runs=states.count("noisy"),
            unchecked_runs=states.count("unchecked"),
            ms_per_step_median=median_or_none([r["ms_per_step"] for r in ok if "ms_per_step" in r]),
            generate_s_median=median_or_none([r["generate_s"] for r in ok if "generate_s" in r]),
            tokens_per_s_median=median_or_none(
                [gen_length / r["generate_s"] for r in ok if "generate_s" in r]),
            startup_s_median=median_or_none([r["startup_s"] for r in ok if "startup_s" in r]),
            peak_rss_gb_max=max((r["peak_rss_bytes"] for r in ok), default=0) / 1e9 or None,
            forward_passes_median=median_or_none([r["forward_passes"] for r in ok if "forward_passes" in r]),
            forward_ms_median=median_or_none([r["forward_ms"] for r in ok if r.get("forward_ms") is not None]),
        )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "bench/config.json"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default="", help="comma list of runtimes to run")
    ap.add_argument("--allow-battery", action="store_true",
                    help="measure on battery instead of pausing (each run records on_battery)")
    ap.add_argument("--allow-noisy", action="store_true",
                    help="continue after a failed isolation preflight (recorded in the env file)")
    ap.add_argument("--isolation-retries", type=int, default=2,
                    help="repeat a run whose isolation window was noisy up to this many times (default %(default)s)")
    args = ap.parse_args()
    c = json.loads(pathlib.Path(args.config).read_text())
    # Paths in the config are written with ~ so the committed file names no
    # home directory; expand them only for use.
    for key in list(c["models"]):
        c["models"][key] = os.path.expanduser(c["models"][key])
    c["llama_cpp"] = os.path.expanduser(c["llama_cpp"])
    c["diffuse_cpp"] = os.path.expanduser(c["diffuse_cpp"])
    runtimes = select_runtimes(c["runtimes"], args.only, args.reps)
    for key in ("gguf", "diffuse_gguf"):
        if key in c["models"] and not pathlib.Path(c["models"][key]).exists():
            raise SystemExit(f"model file missing: {c['models'][key]}")

    kills: list[dict] = []
    wd = Watchdog("pytest", kills)
    kill_pytest = c.get("kill_pytest", True)
    profile_before = sh("powerprofilesctl get")
    arm = None
    results: list[dict] = []
    out_path = pathlib.Path(args.out or ROOT / f"bench/results/{dt.date.today().isoformat()}-cpu.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    versions = {
        "mojo-dllm": sh(f"git -C {shlex.quote(str(ROOT))} rev-parse --short HEAD")
        + ("-dirty" if sh(f"git -C {shlex.quote(str(ROOT))} status --porcelain -- src") else ""),
        "mojo": mojo_version(),
        "llama.cpp": sh(f"git -C {shlex.quote(c['llama_cpp'])} rev-parse --short HEAD"),
        "diffuse-cpp": sh(f"git -C {shlex.quote(c['diffuse_cpp'])} rev-parse --short HEAD"),
    }

    def summarize() -> dict:
        return summarize_runs(results, runtimes, c["gen_length"])

    def write_doc(partial: bool) -> dict:
        summary = summarize()
        doc = dict(
            partial=partial,
            date=dt.datetime.now().isoformat(timespec="seconds"),
            machine=dict(cpu=sh("lscpu | sed -n 's/^Model name:\\s*//p'"), kernel=platform.release(),
                         ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
                         power_profile="performance", allow_battery=args.allow_battery, logical_cpus=os.cpu_count(),
                         gpu=sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
                         or None),
            versions=versions,
            config=json.loads(pathlib.Path(args.config).read_text()), reps=args.reps, summary=summary,
            watchdog_kills=kills, runs=results,
        )
        out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        return summary

    def on_term(signum: int, frame: object) -> None:
        raise SystemExit(f"stopped by signal {signum}")

    signal.signal(signal.SIGTERM, on_term)

    def one_run(rt: str, p: str, ids: str, n_prompt: int, seq_len: int, threads: int) -> tuple[dict, str]:
        """One timed run of one runtime on one prompt."""
        t_start = dt.datetime.now().isoformat(timespec="seconds")
        if rt == "mojo-dllm":
            r = run_mojo(c, p, threads, [], n_prompt)
        elif rt == "mojo-dllm-gpu":
            r = run_mojo(c, p, threads, ["--device", "gpu"], n_prompt)
        elif rt == "mojo-dllm-full-logits":
            r = run_mojo(c, p, threads, ["--full-logits"], n_prompt)
        elif rt == "llama.cpp":
            r = run_llama(c, p, threads, seq_len)
        elif rt == "llama.cpp-cuda":
            r = run_llama(c, p, threads, seq_len, cuda=True)
        elif rt.startswith("llama.cpp:t"):
            r = run_llama(c, p, int(rt.removeprefix("llama.cpp:t")), seq_len)
        elif rt.startswith("llama-bench:t"):
            r = run_llama_bench(c, int(rt.removeprefix("llama-bench:t")), seq_len)
        elif rt == "llama-bench-cuda":
            r = run_llama_bench(c, threads, seq_len, cuda=True)
        elif rt == "diffuse-cpp":
            r = run_diffuse(c, ids, threads, n_prompt, "equal")
        elif rt == "diffuse-cpp-cache-entropy-exit":
            r = run_diffuse(c, ids, threads, n_prompt, "default")
        else:
            raise SystemExit(f"unknown runtime {rt}")
        return r, t_start

    try:
        subprocess.run(["powerprofilesctl", "set", "performance"], check=True)
        if kill_pytest:
            wd.start()
        device = "gpu" if any("gpu" in r or "cuda" in r for r in runtimes) else "cpu"
        arm = isolation.begin_arm(
            out_path.stem, device=device, need_ram_gb=isolation.need_ram_gb(c["models"]["gguf"]),
            runs_dir=out_path.parent / "env", retries=args.isolation_retries, allow_noisy=args.allow_noisy,
            allow_battery=args.allow_battery, thresholds=isolation.Thresholds(cpu_busy_max=c["busy_threshold"]),
            tuned={"previous": profile_before, "set": "performance"}, log=lambda m: print(m, flush=True),
            versions=versions)
        threads = c["threads"]
        tok_cache: dict[str, tuple[str, int]] = {}
        for p in c["prompts"]:
            out = subprocess.run(
                [str(ROOT / "build/mojo-dllm"), "tokenize", "--model", c["models"]["gguf"], "--prompt", p],
                capture_output=True, text=True, check=True,
            ).stdout
            tok_cache[p] = parse_token_ids(out)
        for rep in range(args.reps):
            for pi, p in enumerate(c["prompts"]):
                ids, n_prompt = tok_cache[p]
                seq_len = n_prompt + c["gen_length"]
                for rt in runtimes:
                    # On battery the run waits for the charger (with the power profile
                    # restored meanwhile), and a run the charger left midway is repeated.
                    (r, t_start), pinfo = arm.run(
                        lambda: one_run(rt, p, ids, n_prompt, seq_len, threads),
                        max_quiet_wait_s=c["max_quiet_wait_s"], quiet_sample_s=2.0,
                        on_pause=lambda: subprocess.run(["powerprofilesctl", "set", profile_before or "power-saver"]),
                        on_resume=lambda: subprocess.run(["powerprofilesctl", "set", "performance"], check=True),
                    )
                    load, waited = pinfo["isolation"]["cpu_busy_before"], pinfo["isolation"]["waited_s"]
                    r.update(runtime=rt, prompt_index=pi, rep=rep, started=t_start,
                             cpu_busy_before=round(load, 3), waited_s=round(waited, 1),
                             power_waited_s=round(pinfo["power_waited_s"], 1),
                             power_retries=pinfo["power_retries"], on_battery=pinfo["on_battery"],
                             power_source=pinfo["power_source"], power_supplies=pinfo["power_supplies"],
                             isolation=pinfo["isolation"],
                             **({"power_root": pinfo["power_root"]} if "power_root" in pinfo else {}))
                    results.append(r)
                    write_doc(partial=True)
                    if not r["ok"]:
                        status = f"FAILED rc={r.get('rc')}"
                    elif "ms_per_step" in r:
                        status = f"{r['ms_per_step']:.0f} ms/step"
                    else:
                        status = f"{r['forward_ms']:.0f} ms/forward"
                    noisy = "" if pinfo["isolation"]["clean"] else "  NOISY: " + "; ".join(pinfo["isolation"]["reasons"])
                    print(f"rep {rep} prompt {pi} {rt:32s} {status}  busy before {load:.1%}{noisy}", flush=True)
    except isolation.IsolationError as e:
        raise SystemExit(str(e)) from e
    finally:
        if arm is not None:
            arm.finish()
        wd.stop.set()
        subprocess.run(["powerprofilesctl", "set", profile_before or "power-saver"])

    summary = write_doc(partial=False)
    print(json.dumps(summary, indent=2))
    print(f"watchdog killed {len(kills)} pytest process(es)" if kill_pytest else "watchdog off (kill_pytest: false)")
    print(f"wrote {out_path}")
    failed = sum(s["runs_failed"] for s in summary.values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
