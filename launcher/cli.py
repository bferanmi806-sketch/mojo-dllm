"""Command dispatch and help text."""

from __future__ import annotations

import difflib
import sys

from launcher.build import BINARY, ROOT, ensure_built, exec_binary
from launcher.doctor import run_doctor
from launcher.errors import LauncherError, UsageError, describe_os_error
from launcher.probes import read_project_version, real_probes
from launcher.runcmd import cmd_pull, cmd_run
from launcher.ui import Style
from launcher.wizard import parse_setup_args, real_env, run_setup

LAUNCHER_COMMANDS = ("setup", "pull", "run", "doctor", "help")
# The binary's own commands. A test compares this list with command_spec() in src/main.mojo.
BINARY_COMMANDS = ("inspect", "tokenize", "logits")


def help_text(style: Style) -> str:
    b = style.bold

    def row(name: str, text: str) -> str:
        return f"  {name:<9}{text}"

    lines = [
        f"{b('mojo-dllm')}: diffusion language model inference in Mojo",
        "",
        b("Quick start"),
        "  ./mojo-dllm setup                              install the toolchain, build, fetch the model",
        '  ./mojo-dllm run "Explain diffusion models."    generate text',
        "  ./mojo-dllm doctor                             check this machine",
        "",
        b("Commands"),
        row("setup", "install the toolchain, build the binary, download a model, run a smoke test"),
        row("pull", "download a model into the Hugging Face cache (default: llada-8b)"),
        row("run", "generate text from a prompt; the binary's own flags pass through"),
        row("doctor", "check platform, toolchain, binary, GPU and model; changes nothing"),
        row("help", "show this text"),
        "  inspect, tokenize and logits go to the binary unchanged, after the build is current.",
        "  --version prints the version from pyproject.toml and exits.",
        "",
        b("Options"),
        "  setup [--model MODEL] [--yes] [--no-model] [--offline]",
        "  pull [MODEL] [--offline]",
        '  run "PROMPT" [-m MODEL] [--device auto|cpu|gpu] [--offline] [binary flags]',
        "",
        '  A prompt that starts with - goes after --: ./mojo-dllm run -- "-5 plus 3?"',
        "  MODEL is a path to a .gguf file, the alias llada-8b, or owner/repo[:QUANT].",
        "  Binary flags include --max-tokens, --steps, --block-length, --seed,",
        "  --temperature, --threads and --verbose.",
        "",
        b("Examples"),
        '  ./mojo-dllm run "Write a haiku about rain." --max-tokens 64 --steps 32',
        '  ./mojo-dllm run "Hello" -m ~/models/llada.gguf --device cpu',
        "  ./mojo-dllm pull someone/Some-Model-GGUF:Q4_K_M",
        "  ./mojo-dllm inspect ~/models/llada.gguf",
        "",
        b("Environment"),
        "  HF_TOKEN             Hugging Face token, for gated or private repos",
        "  HF_HOME              Hugging Face home; models live in $HF_HOME/hub",
        "  HF_HUB_CACHE         the model cache directory (overrides HF_HOME)",
        "  XDG_CACHE_HOME       cache root when HF_HOME is unset; mojo-dllm keeps verified.json under it",
        "  HF_HUB_OFFLINE=1     same as --offline: use the cache only",
        "  MOJO_DLLM_GPU_ARCH   GPU architecture to build for, such as sm_86",
        "  NO_COLOR             turn colour off",
    ]
    return "\n".join(lines)


def _unknown_command(command: str) -> str:
    close = difflib.get_close_matches(command, (*LAUNCHER_COMMANDS, *BINARY_COMMANDS), n=1)
    hint = f"; did you mean {close[0]}?" if close else "."
    return f"unknown command {command!r}{hint} Run ./mojo-dllm help for the commands."


def main(argv: list[str]) -> int:
    command, rest = (argv[0], argv[1:]) if argv else ("help", [])
    # After a bare "--" every word is data (a prompt may be "-h"), so help is looked for only before it.
    options = rest[: rest.index("--")] if "--" in rest else rest
    if command in ("help", "-h", "--help") or (
        command in LAUNCHER_COMMANDS and any(a in ("-h", "--help") for a in options)
    ):
        print(help_text(Style.for_stream(sys.stdout)))
        return 0
    try:
        if command == "--version":
            try:
                text = (ROOT / "pyproject.toml").read_text()
            except OSError as e:
                raise LauncherError(f"cannot read pyproject.toml: {e.strerror}. Restore it from git.") from e
            print(f"mojo-dllm {read_project_version(text)}")
            return 0
        if command == "setup":
            setup_args = parse_setup_args(rest)
            return run_setup(setup_args, real_env(offline=setup_args.offline))
        if command == "pull":
            return cmd_pull(rest)
        if command == "run":
            return cmd_run(rest)
        if command == "doctor":
            return run_doctor(real_probes(), sys.stdout, Style.for_stream(sys.stdout))
        if command not in BINARY_COMMANDS:
            raise UsageError(_unknown_command(command))
        ensure_built()
        exec_binary([str(BINARY), *argv])
    except LauncherError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.code
    except OSError as e:
        print(f"error: {describe_os_error(e)}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
