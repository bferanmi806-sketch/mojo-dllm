"""The constants and prompts that several benchmark tools share, defined once.

A value that two tools repeat can change in one and stay behind in the other, and the comparison then runs on
unlike things. Each tool imports from here; tests/python/test_bench_spec.py fails when one of them carries its
own copy again.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Mapping

# Where a run happens. For vLLM and SGLang it names which isolation checks of THIS machine run, because the
# server is somewhere else.
Device = Literal["cpu", "gpu"]
DEVICES: tuple[str, ...] = ("cpu", "gpu")
# llama.cpp's transfer schedule: `timestep` (with an epsilon) or `block`.
Schedule = Literal["timestep", "block"]
SCHEDULES: tuple[str, ...] = ("timestep", "block")

# The mask and vocabulary ids of the two diffusion families (the model file's own tokenizer metadata).
ARCH_IDS: Mapping[str, dict[str, int]] = {
    "llada": dict(vocab=126464, mask=126336),
    "dream": dict(vocab=152064, mask=151666),
}

# The three benchmark prompts, in the order every config file lists them.
PROMPTS: tuple[str, ...] = (
    "Explain speculative decoding in two sentences.",
    "Write a short Python function that checks whether a number is prime.",
    "List three differences between diffusion language models and autoregressive language models.",
)

# llama.cpp's diffusion example, the settings every benchmark and engine gives it.
LLAMA_EPS = "0.001"
LLAMA_TEMP = "0"
LLAMA_SEED = "42"
LLAMA_GPU_LAYERS = "99"

# One generation or one problem may run this long before the run fails (it never scores a timeout).
GENERATION_TIMEOUT_S = 1800.0

# DiffusionGemma parity: the canvas and the noise are drawn from these seeds, and the first denoising step runs
# at this temperature (issue #60), so the second step's self-conditioning divides the previous logits by it.
DG_CANVAS_SEED = 66
DG_NOISE_SEED = 66
DG_FIRST_STEP_TEMP = 0.8

# compare_gpu.py: the workload of both builds, and the repetition index that marks a warm-up pair.
COMPARE_STEPS = 32
COMPARE_BLOCK_LENGTH = 32
COMPARE_THREADS = 8
WARMUP_REP = -1


def llama_diffusion_args(seq_len: int, steps: int, threads: int, gpu: bool, schedule: str = "timestep",
                         block_length: int = 32) -> list[str]:
    """The llama-diffusion-cli flags after `-m MODEL -p PROMPT`, for run_bench and the GSM8K engine alike."""
    cmd = ["-c", str(seq_len), "-b", str(seq_len), "-ub", str(seq_len), "--diffusion-steps", str(steps)]
    cmd += ["--diffusion-eps", LLAMA_EPS] if schedule == "timestep" else ["--diffusion-block-length", str(block_length)]
    cmd += ["--temp", LLAMA_TEMP, "-t", str(threads), "-tb", str(threads), "--seed", LLAMA_SEED]
    if gpu:
        cmd += ["-ngl", LLAMA_GPU_LAYERS, "-v"]
    return cmd


def result_meta(versions: Mapping[str, str], threads: int | str) -> dict:
    """`date`, `versions` and `threads` of a result document, in one convention.

    `threads` is an integer (a string such as "12" from a command line is converted, anything else refused),
    `versions` maps a component to its commit text, and `date` is the local time to the second.
    """
    try:
        n = int(threads)
    except (TypeError, ValueError):
        raise ValueError(f"threads must be an integer, got {threads!r}") from None
    if isinstance(threads, bool) or n < 1:
        raise ValueError(f"threads must be 1 or more, got {threads!r}")
    return dict(date=dt.datetime.now().isoformat(timespec="seconds"), versions=dict(versions), threads=n)
