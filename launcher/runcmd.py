"""`run` and `pull`: argument handling and the hand-off to the binary."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from launcher.build import BINARY, ensure_built, exec_binary
from launcher.errors import UsageError
from launcher.gpu import QUERY, Gpu, choose_device, parse_nvidia_smi
from launcher.hub import Hub, offline_requested
from launcher.models import DEFAULT_MODEL, ensure_model
from launcher.verified import default_store

# Flags of the binary that take no value. A test compares this list with
# boolean_flags() in src/main.mojo.
BOOLEAN_FLAGS = (
    "--verbose", "--full-logits", "--no-chat", "--dump-step-stats",
    "--json", "--visual", "--no-early-stop", "--no-suppress-mask", "--dual-cache",
)
EXAMPLE = './mojo-dllm run "Explain speculative decoding in two sentences."'


@dataclass(frozen=True)
class RunArgs:
    prompt: str
    model: str
    device: str
    offline: bool
    passthrough: list[str]


def parse_run_args(argv: list[str]) -> RunArgs:
    prompt: str | None = None
    model, device, offline = DEFAULT_MODEL, "auto", False
    passthrough: list[str] = []
    options_done = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if options_done:
            if prompt is not None:
                raise UsageError(f"unexpected argument {arg!r}: quote the prompt. Example: {EXAMPLE}")
            prompt = arg
        elif arg == "--":
            options_done = True
        elif arg == "--offline":
            offline = True
        elif arg == "--prompt":
            raise UsageError(f"pass the prompt as the first word, without --prompt. Example: {EXAMPLE}")
        elif arg.startswith("-"):
            if arg in BOOLEAN_FLAGS:
                passthrough.append(arg)
            else:
                if i + 1 >= len(argv):
                    raise UsageError(f"{arg} needs a value. Example: {EXAMPLE}")
                value = argv[i + 1]
                i += 1
                if arg in ("-m", "--model"):
                    model = value
                elif arg == "--device":
                    device = value
                else:
                    passthrough += [arg, value]
        elif prompt is None:
            prompt = arg
        else:
            raise UsageError(f"unexpected argument {arg!r}: quote the prompt. Example: {EXAMPLE}")
        i += 1
    if device not in ("auto", "cpu", "gpu"):
        raise UsageError(f"--device must be auto, cpu or gpu, not {device!r}")
    if prompt is None:
        raise UsageError(f"run needs a prompt. Example: {EXAMPLE}")
    return RunArgs(prompt, model, device, offline, passthrough)


def run_argv(binary: Path, model: Path, args: RunArgs, device: str) -> list[str]:
    return [
        str(binary), "run", "--model", str(model), "--prompt", args.prompt,
        "--device", device, *args.passthrough,
    ]


def detect_gpus() -> tuple[list[Gpu], str | None]:
    """The GPUs nvidia-smi lists, and why it could not say (None when it ran)."""
    if shutil.which("nvidia-smi") is None:
        return [], None
    try:
        done = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired as e:
        return [], f"timed out after {e.timeout:g} s"
    except OSError as e:
        return [], str(e)
    if done.returncode != 0:
        detail = done.stderr.strip().splitlines()
        return [], f"exit {done.returncode}: {detail[0]}" if detail else f"exit {done.returncode}"
    return parse_nvidia_smi(done.stdout), None


def cmd_run(argv: list[str]) -> int:
    args = parse_run_args(argv)
    ensure_built()
    hub = Hub(offline=offline_requested(args.offline, os.environ))
    # Progress and status go to stderr so stdout stays the model's text.
    model = ensure_model(args.model, hub, sys.stderr, announce_cached=False, store=default_store())
    gpus, smi_failure = detect_gpus() if args.device == "auto" else ([], None)
    device = choose_device(args.device, gpus, smi_failure)
    print(device.line, file=sys.stderr, flush=True)
    argv_out = run_argv(BINARY, model.resolve(), args, device.kind)
    exec_binary(argv_out)


def cmd_pull(argv: list[str]) -> int:
    offline = False
    refs: list[str] = []
    for arg in argv:
        if arg == "--offline":
            offline = True
        elif arg.startswith("-"):
            raise UsageError(f"unknown option {arg} for pull. Usage: ./mojo-dllm pull [MODEL] [--offline]")
        else:
            refs.append(arg)
    if len(refs) > 1:
        raise UsageError("pull takes one model. Usage: ./mojo-dllm pull [MODEL] [--offline]")
    hub = Hub(offline=offline_requested(offline, os.environ))
    ensure_model(
        refs[0] if refs else DEFAULT_MODEL, hub, sys.stdout, announce_cached=True, store=default_store()
    )
    return 0
