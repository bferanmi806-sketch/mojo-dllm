"""llama.cpp's diffusion example (`llama-diffusion-cli`) as a GSM8K engine.

The command line is the one `bench/run_bench.py` times (`run_llama`), and a CUDA run is held to the same check:
every layer on the GPU. Flag names come from `llama-diffusion-cli --help` at the commit recorded as `version`.

llama.cpp has no early stop and prints no pass count, so a row's `forward_passes` is null, its `steps` is the
configured step count, and the arm's `flags` say `"forward_passes_measured": false`. Seconds per problem are the
comparable column.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

from . import EngineError, GenConfig, GenResult, register

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import run_bench  # noqa: E402  (bench/run_bench.py: the CUDA check and the output parser)

EPS = "0.001"     # the timestep schedule's epsilon, as in run_bench
TEMP = "0"
SHORT_TIMEOUT_S = 60    # git rev-parse and `mojo-dllm tokenize`: both finish in well under a second
SEED = "42"


def _run(cmd: list[str], timeout_s: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        what = " ".join([pathlib.Path(cmd[0]).name] + [w for w in cmd[1:3] if not w.startswith(("-", "/"))])
        raise EngineError(f"`{what}` timed out after {timeout_s:g} s") from None


def binary(llama_cpp: pathlib.Path, device: str) -> pathlib.Path:
    return llama_cpp / ("build-cuda" if device == "gpu" else "build") / "bin" / "llama-diffusion-cli"


@register
class LlamaCpp:
    name = "llamacpp"

    def __init__(self, args: argparse.Namespace) -> None:
        self.llama_cpp = pathlib.Path(args.llama_cpp).expanduser()
        self.mojo_dllm = pathlib.Path(args.mojo_dllm).expanduser()
        self.threads = args.threads

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--llama-cpp", default="~/src/llama.cpp", help="llama.cpp checkout with build/ and build-cuda/")
        parser.add_argument("--mojo-dllm", default=str(ROOT / "build" / "mojo-dllm"),
                            help="mojo-dllm binary; its `tokenize` sizes llama.cpp's canvas (prompt + generated tokens)")
        parser.add_argument("--threads", type=int, default=os.cpu_count() or 1, help="-t and -tb for llama.cpp")
        parser.add_argument("--schedule", choices=["timestep", "block"], default="timestep",
                            help="llama.cpp's transfer schedule; block needs a prompt length that is a multiple of "
                                 "--block-length, which GSM8K prompts are not")

    def version(self) -> str:
        p = _run(["git", "-C", str(self.llama_cpp), "rev-parse", "--short", "HEAD"], SHORT_TIMEOUT_S)
        if p.returncode != 0 or not p.stdout.strip():
            raise EngineError(f"cannot read the llama.cpp commit of {self.llama_cpp}: {p.stderr.strip() or 'no output'}")
        return p.stdout.strip()

    def check(self, cfg: GenConfig) -> None:
        if cfg.device not in ("cpu", "gpu"):
            raise SystemExit(f"llamacpp runs on --device cpu or gpu, not {cfg.device!r}")
        exe = binary(self.llama_cpp, cfg.device)
        if not (exe.is_file() and os.access(exe, os.X_OK)):
            raise SystemExit(f"{exe} is missing or not executable: build llama.cpp's {exe.parts[-3]} tree first")
        if not (self.mojo_dllm.is_file() and os.access(self.mojo_dllm, os.X_OK)):
            raise SystemExit(f"{self.mojo_dllm} is missing: `mojo-dllm tokenize` sizes the canvas (run ./mojo-dllm setup)")
        if not pathlib.Path(cfg.model).is_file():
            raise SystemExit(f"model file {cfg.model} does not exist")

    def flags(self, cfg: GenConfig) -> dict:
        schedule = {"schedule": cfg.schedule, "diffusion_eps": float(EPS)} if cfg.schedule == "timestep" else \
                   {"schedule": cfg.schedule, "diffusion_block_length": cfg.block_length}
        return {"engine": self.name, "device": cfg.device, "gen_length": cfg.gen_length,
                "diffusion_steps": cfg.steps, **schedule, "temp": float(TEMP), "seed": int(SEED),
                "threads": self.threads, "extra": list(cfg.extra), "forward_passes_measured": False}

    def prompt_tokens(self, prompt: str, cfg: GenConfig) -> int:
        """Tokens llama.cpp will see: the same chat-templated prompt mojo-dllm tokenizes, as run_bench counts them."""
        p = _run([str(self.mojo_dllm), "tokenize", "--model", cfg.model, "--prompt", prompt], SHORT_TIMEOUT_S)
        lines = p.stdout.strip().splitlines()
        if p.returncode != 0 or not lines:
            raise EngineError(f"`mojo-dllm tokenize` failed (exit {p.returncode}): {p.stderr[-300:]}")
        return len(lines[-1].strip("[]").replace(" ", "").split(","))

    def command(self, prompt: str, cfg: GenConfig, seq_len: int) -> list[str]:
        cmd = [str(binary(self.llama_cpp, cfg.device)), "-m", cfg.model, "-p", prompt,
               "-c", str(seq_len), "-b", str(seq_len), "-ub", str(seq_len),
               "--diffusion-steps", str(cfg.steps)]
        cmd += ["--diffusion-eps", EPS] if cfg.schedule == "timestep" else ["--diffusion-block-length", str(cfg.block_length)]
        cmd += ["--temp", TEMP, "-t", str(self.threads), "-tb", str(self.threads), "--seed", SEED]
        if cfg.device == "gpu":
            cmd += ["-ngl", "99", "-v"]
        return cmd + list(cfg.extra)

    def generate(self, prompt: str, cfg: GenConfig) -> dict:
        # Assumption, not a check: the canvas is sized from mojo-dllm's token count (chat template included, as
        # run_bench does). llama-diffusion-cli prints its own input token count only when the input is too long
        # (diffusion-cli.cpp, "input too long"), so a one-token disagreement, a BOS for example, cannot be
        # detected here. It would make llama.cpp generate one token fewer than --gen-length.
        seq_len = self.prompt_tokens(prompt, cfg) + cfg.gen_length
        p = _run(self.command(prompt, cfg, seq_len), cfg.timeout_s)
        output = p.stderr + p.stdout
        if p.returncode != 0:
            raise EngineError(f"llama-diffusion-cli exited {p.returncode}: {output[-400:]}")
        if cfg.device == "gpu":
            bad = run_bench.cuda_layer_error(output)
            if bad:
                raise EngineError(bad)
        parsed = run_bench.parse_llama_output(output)
        if parsed is None:
            raise EngineError(f"no `time per step` line in the output: {output[-400:]}")
        # llama.cpp does not report its pass count: the configured steps go in `steps`, `forward_passes` stays None.
        return GenResult(text=parsed["text"], generate_s=parsed["total_s"], forward_passes=None, steps=cfg.steps)
