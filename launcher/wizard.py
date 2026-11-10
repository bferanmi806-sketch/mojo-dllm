"""`./mojo-dllm setup`: toolchain, build, model, smoke test."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from launcher import build
from launcher.doctor import preflight, render
from launcher.errors import LauncherError, UsageError, describe_os_error
from launcher.gpu import Device, choose_device, parse_nvidia_smi
from launcher.hub import offline_requested
from launcher.models import DEFAULT_MODEL, Hub, already_downloaded_line, fetch_model, parse_model_ref, plan_model, use_cached
from launcher.probes import Probes
from launcher.ui import Style, human_size
from launcher.verified import VerifiedStore, default_store

SMOKE_PROMPT = "Say hello."
SMOKE_TIMEOUT_S = 300


@dataclass(frozen=True)
class SetupArgs:
    model: str = DEFAULT_MODEL
    yes: bool = False
    no_model: bool = False
    offline: bool = False


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UsageError(f"{message}. Usage: ./mojo-dllm setup [--model MODEL] [--yes] [--no-model] [--offline]")


def parse_setup_args(argv: list[str]) -> SetupArgs:
    parser = _Parser(prog="./mojo-dllm setup", add_help=False)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--offline", action="store_true")
    ns = parser.parse_args(argv)
    return SetupArgs(model=ns.model, yes=ns.yes, no_model=ns.no_model, offline=ns.offline)


@dataclass(frozen=True)
class SetupEnv:
    probes: Probes
    hub: Hub
    store: VerifiedStore
    is_tty: bool
    ask: Callable[[str], str]
    sync: Callable[[], None]
    build: Callable[[], bool]  # True when it compiled, False when the binary was current
    binary_ready: Callable[[], bool]  # whether build/mojo-dllm exists after the build step
    smoke: Callable[[Path, str], float]
    style: Style
    out: TextIO
    err: TextIO


def real_sync() -> None:
    if subprocess.run(["uv", "sync", "--frozen"], cwd=build.ROOT, env=build.project_env()).returncode != 0:
        raise LauncherError("uv sync --frozen failed. Check the network, then run ./mojo-dllm setup again.")


def real_smoke(model: Path, device: str) -> float:
    cmd = [
        str(build.BINARY), "run", "--model", str(model.resolve()), "--prompt", SMOKE_PROMPT,
        "--max-tokens", "16", "--steps", "8", "--device", device,
    ]
    start = time.monotonic()
    try:
        done = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, timeout=SMOKE_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise LauncherError(
            f"the smoke test did not finish within {SMOKE_TIMEOUT_S} s. "
            "Run ./mojo-dllm doctor to see what is wrong."
        ) from e
    if done.returncode != 0:
        raise LauncherError("the smoke test failed. Run ./mojo-dllm doctor to see what is wrong.")
    if not done.stdout.strip():
        # Exit 0 with no text is not a working model.
        raise LauncherError("the smoke test printed nothing. Run ./mojo-dllm doctor to see what is wrong.")
    sys.stdout.write(done.stdout)
    return time.monotonic() - start


def _step(env: SetupEnv, n: int, title: str) -> None:
    print(env.style.bold(f"[{n}/4] {title}"), file=env.out, flush=True)


def run_setup(args: SetupArgs, env: SetupEnv) -> int:
    try:
        return _setup(args, env)
    except LauncherError as e:
        print(f"error: {e}", file=env.err)
        return e.code
    except OSError as e:
        print(f"error: {describe_os_error(e)}", file=env.err)
        return 1


def _setup(args: SetupArgs, env: SetupEnv) -> int:
    out = env.out
    outstanding: list[str] = []
    pull_cmd = "./mojo-dllm pull" + ("" if args.model == DEFAULT_MODEL else f" {args.model}")

    _step(env, 1, "Toolchain")
    failed = False
    for check in preflight(env.probes):
        render(check, env.style, out)
        failed = failed or check.status == "fail"
    if failed:
        raise LauncherError("this machine does not meet the requirements above; fix them, then run ./mojo-dllm setup again.")
    env.sync()

    _step(env, 2, "Build")
    if not env.build():
        print("build/mojo-dllm is up to date", file=out)
    if not env.binary_ready():
        raise LauncherError(
            "the build finished but build/mojo-dllm does not exist. Run bash scripts/build.sh and read its output."
        )

    _step(env, 3, "Model")
    model: Path | None = None
    if args.no_model:
        print("skipped (--no-model)", file=out)
    else:
        ref = parse_model_ref(args.model)
        if ref.path is not None:
            model = ref.path
            print(f"using {model}", file=out)
        else:
            plan = plan_model(ref, env.hub)
            if plan.cached is not None:
                model, verified = use_cached(plan, env.hub, out, env.store)
                print(f"{already_downloaded_line(verified)}\n{model}", file=out)
                if not verified:
                    outstanding.append(pull_cmd)
            else:
                size = f" ({human_size(plan.size)})" if plan.size is not None else ""
                question = f"Download {plan.label}{size} to {env.hub.cache_dir}? [Y/n]"
                if _confirm(args, env, question):
                    model = fetch_model(plan, env.hub, out, env.store)
                else:
                    why = "declined" if env.is_tty else "no terminal to ask on; pass --yes to download"
                    print(f"not downloaded ({why})", file=out)
                    outstanding.append(pull_cmd)

    _step(env, 4, "Check")
    if model is None:
        print("skipped: needs a model", file=out)
    else:
        if env.probes.gpu_text is None:
            # Missing, failed or timed out: no GPU was checked, which is not the same as no GPU.
            device = Device("cpu", "nvidia-smi missing or could not be read; GPU not checked")
        else:
            device = choose_device("auto", parse_nvidia_smi(env.probes.gpu_text))
        print(device.line, file=out, flush=True)
        seconds = env.smoke(model, device.kind)
        print(f"smoke test ok in {seconds:.1f} s", file=out)

    if outstanding:
        count = len(outstanding)
        print(f"\nSetup finished with {count} step{'s' if count > 1 else ''} outstanding. Run:", file=out)
        for cmd in outstanding:
            print(f"  {cmd}", file=out)
        return 3
    done = "Setup complete (no smoke test)." if args.no_model else "Setup complete."
    print(f"\n{env.style.bold(done)}\n\nTry it:", file=out)
    print('  ./mojo-dllm run "Explain speculative decoding in two sentences."', file=out)
    print("  ./mojo-dllm doctor", file=out)
    return 0


def _confirm(args: SetupArgs, env: SetupEnv, question: str) -> bool:
    if args.yes:
        return True
    if not env.is_tty:
        return False
    try:
        answer = env.ask(question)
    except EOFError:  # Ctrl-D at the prompt is a refusal
        print(file=env.out)
        return False
    return answer.strip().lower() in ("", "y", "yes")


def real_env(offline: bool = False) -> SetupEnv:
    from launcher.hub import Hub as RealHub
    from launcher.probes import real_probes

    return SetupEnv(
        probes=real_probes(),
        hub=RealHub(offline=offline_requested(offline, os.environ)),
        store=default_store(),
        is_tty=sys.stdin.isatty() and sys.stdout.isatty(),
        ask=input,
        sync=real_sync,
        build=build.ensure_built,
        binary_ready=build.BINARY.is_file,
        smoke=real_smoke,
        style=Style.for_stream(sys.stdout),
        out=sys.stdout,
        err=sys.stderr,
    )
