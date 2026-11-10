# Quick start

You need Linux on x86-64 with AVX2, glibc 2.34 or newer, about 6 GB of free
memory, and [uv](https://docs.astral.sh/uv/). uv installs the pinned Mojo
toolchain (Mojo 1.1.0, max-core 26.6.0) into the project's `.venv`; nothing
goes system-wide.

## 1. Set up

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # skip if you have uv
git clone https://github.com/vladimirrott/mojo-dllm
cd mojo-dllm
./mojo-dllm setup
```

The uv installer puts `uv` in `~/.local/bin`. If your shell does not find it
afterwards, open a new terminal or run `source ~/.local/bin/env`.

`setup` runs four steps and prints each one:

1. **Toolchain.** It checks the platform, glibc and AVX2, stops
   on anything that cannot work, and runs `uv sync --frozen`.
2. **Build.** It compiles `build/mojo-dllm`, or reports that the binary is
   newer than its inputs and was built for this Mojo version and GPU architecture.
3. **Model.** It asks before downloading LLaDA-8B-Instruct Q4_K_M (4.9 GB)
   into the Hugging Face cache, and checks the file's sha256 against the one
   the hub publishes. `--yes` answers the question for you; `--no-model` skips
   the step; `--model` picks another model.
4. **Check.** It generates 16 tokens to prove the binary, the model and the
   device work together. The check fails when the binary exits non-zero, prints
   nothing, or runs longer than 300 seconds. With `--no-model` the step is
   skipped and setup ends with `Setup complete (no smoke test).`

`setup --offline` (or `HF_HUB_OFFLINE=1`) uses the model cache only, as `run` and `pull` do.
Without a terminal to ask on (a script, CI), `setup` does not download
anything unless you pass `--yes`. It then exits 3 and prints the command that
finishes the job.

`./mojo-dllm doctor` runs those checks without changing anything, and adds free
memory, the Mojo version, the binary's age, the GPU and the model cache. Every
line that is not `ok` says how to fix it, with the command where one exists; for a
missing GPU it says that the CPU will be used.

## 2. Generate

```bash
./mojo-dllm run "Explain speculative decoding in two sentences."
./mojo-dllm run "Write a haiku about rain." --max-tokens 64 --steps 32 --verbose
```

The prompt is the first word that is not a flag; quote it. `run` rebuilds the
binary if a source file, `pyproject.toml`, `uv.lock` or `scripts/build.sh` changed, or if the
Mojo pin or the GPU architecture (`MOJO_DLLM_GPU_ARCH`, else the local GPU) differs from the one
recorded in `build/mojo-dllm.stamp`, finds the model, picks the device and hands
the rest of the flags to `build/mojo-dllm run`. [Command line](cli.md) lists
them all.

A flag and its value are two arguments: `--steps 8`, not `--steps=8`. The launcher
refuses the `=` form and prints the two-argument form to use. A prompt that starts
with `-` goes after `--` (`./mojo-dllm run -- "-h"`); `-h` and `--help` after `--`
belong to the prompt and do not print the help.

`--max-tokens` sets the canvas size, `--steps` the number of denoising passes.
Fewer steps run faster and commit more tokens per step, which costs quality.
Each step runs the full model over prompt and canvas, so expect seconds per
step on a laptop CPU. `--verbose` prints load time, per-step time and peak
memory.

The prompt goes through the model's chat template. Pass `--no-chat` to feed
raw text instead.

### The device

`--device auto` is the default. It picks the first NVIDIA GPU with compute
capability 8.0 or newer that `nvidia-smi` lists, and the CPU when there is
none. The binary opens CUDA's default device, which can be another card than
the first one `nvidia-smi` lists, so the launcher sets `CUDA_VISIBLE_DEVICES` to
the UUID of the card it named (with `CUDA_DEVICE_ORDER=PCI_BUS_ID`) before it
starts the binary. A `CUDA_VISIBLE_DEVICES` you already set (even an empty one)
is kept: the launcher does not pin the card and prints a warning saying so.
`--device gpu` runs the same check: it pins the first
qualifying card, and prints a warning when `nvidia-smi`
finds no card or only one below compute capability 8.0. Status lines go to
stderr, so stdout holds only the model's text:

```text
device: gpu (NVIDIA GeForce RTX 3050 6GB Laptop GPU, compute 8.6, 585 MiB free)
```

`auto` does not compare the card's free memory with the model's size. In that
line another program held most of the card, so the run that followed would
have said how much memory it needs and exited. Pass `--device cpu` to run
anyway.

## 3. Other models

`-m` (or `--model`) takes one of three forms:

| Form | Example | What happens |
|---|---|---|
| a file | `-m ~/models/llada.gguf` | used as is |
| the alias | `-m llada-8b` | `mradermacher/LLaDA-8B-Instruct-GGUF`, Q4_K_M (the default) |
| a repo | `-m owner/Some-Model-GGUF:Q4_K_M` | the file matching that quant, downloaded once |

Without `:QUANT`, a repo resolves to its Q4_K_M file, then Q4_K_S, then Q6_K,
the quants the matrix kernels load. A repo with none of them, and a quant that
matches nothing, fail with the list of files the repo has, and `owner/repo:QUANT`
selects one. Split (sharded) GGUF files are not supported. A Q8_0 file downloads
when you ask for it with `:Q8_0` but does not load: the matrix kernels need Q4_K
or Q6_K.

Tested Q4_K_M checkpoints are LLaDA-8B-Instruct, LLaDA-1.5,
Dream-v0-Instruct-7B, Dream-Coder-v0-Instruct-7B and DiffuCoder-7B-cpGRPO.
The added checkpoints were checked on CPU; pass `--device cpu` to reproduce
those runs. See the smoke and [parity evidence](correctness.md) for the exact
files and generation agreement.
Other quantizations still need compatible tensor types. An unsupported
architecture stops with an error naming it.

For DiffuCoder, obtain the file separately under its `apple-amlr` license and
pass its local path with `--model`; it is not bundled with the project.

`./mojo-dllm pull MODEL` downloads without generating. Models live in the
Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
`$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`), shared
with every other tool that uses it. A gated or private repo needs `HF_TOKEN`.

The sha256 check runs on download. A file already in the cache is checked
against the hub's sha256 the first time it is used, and the launcher
remembers the result in `~/.cache/mojo-dllm/verified.json` (under
`$XDG_CACHE_HOME` when that is set). A file that fails the check is deleted,
and the next run downloads it again. With `--offline` the check cannot run,
so an unchecked file is used after a warning. `pull --offline` and `setup`
then print `already downloaded (not verified)` and exit with status 3, and
`doctor` marks the model with `!`. Run `./mojo-dllm pull` once online to check it.

## 4. Offline

```bash
./mojo-dllm pull                      # on a connected machine
./mojo-dllm run "Hello" --offline     # later, with no network
```

`--offline` (or `HF_HUB_OFFLINE=1`) uses the cache only and fails with the
`pull` command to run when the model is missing.

## 5. Look inside a model

```bash
./mojo-dllm pull                             # prints the model file's path
./mojo-dllm inspect PATH.gguf | head -40
```

You get the metadata, a count of tensor types, and every tensor with its
shape. The binary's own commands (`inspect`, `tokenize`, `logits`) go to
`build/mojo-dllm` unchanged, after the same rebuild check. Any other word stops
with an error, and names the closest command when one is near.

## Without the launcher

The launcher is a convenience. The binary works on its own:

```bash
uv sync --frozen
scripts/build.sh
build/mojo-dllm run --model PATH.gguf --prompt "Hello" --device cpu
```

`build/mojo-dllm` defaults to `--device cpu` and takes only file paths.
