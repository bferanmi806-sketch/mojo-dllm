# Development

## Setup

```bash
uv sync --frozen                      # the pinned Mojo toolchain, in .venv
scripts/ci-local.sh --install-hooks
```

`scripts/ci-local.sh` also needs shellcheck, markdownlint-cli2 (`npm install
--global markdownlint-cli2@0.23.2`), mdBook 0.4.40 and an authenticated `gh`. A missing tool
fails its step; it never skips it.

The hooks run `scripts/ci-local.sh --fast` before each commit (hygiene scan,
`mojo format`, `-Werror` build) and the full set before each push.

## The gates

`scripts/ci-local.sh` runs the gates of `.github/workflows/ci.yml` and, on a
machine with a GPU, the GPU tests that CI only builds:

| Step | Command | Fails when |
|---|---|---|
| hygiene | `scripts/check_hygiene.sh --tracked` | a credential pattern (a provider token, a private key block, a database URL with a password) or the author's employer name appears in tracked text, or a tracked Markdown file outside `bench/results/` holds an em or en dash; file names are read NUL separated, so a name with non-ASCII or special characters is scanned too) |
| format | `scripts/check_format.sh` (`mojo format -q src tests bench`, comparing every file before and after) | `mojo format` changes any file, an untracked one included (Mojo has no check mode) |
| build | `scripts/build.sh` | any compiler warning (`-Werror`) |
| tests | `scripts/run-tests.sh` | a test fails, a file runs zero tests, the total differs from `tests/test-count.txt`, or a file's count differs from `tests/test-counts.txt` |
| launcher tests | `uv run --locked --only-group test pytest tests/python -q` | a test of the `./mojo-dllm` launcher fails |
| script lock | `uv lock --script mojo-dllm --check` | `mojo-dllm.lock` is missing, or the `# dependencies` header of `./mojo-dllm` no longer matches it. After you change the pin, run `uv lock --script mojo-dllm` and commit the lock |
| shellcheck, yamllint, markdownlint | `shellcheck --severity=warning scripts/*.sh .githooks/*`; `uv run --frozen --only-group lint yamllint --strict .github`; `markdownlint-cli2 "*.md" "docs/**/*.md" "assets/*.md" ".github/*.md" "#node_modules"` | any of the three reports a finding |
| claims | `uv run --no-project python scripts/bench_table.py --check` | a table in the README, `docs/benchmarks.md` or `docs/correctness.md` differs from what `bench/results`, `bench/parity`, `bench/quality` and `bench/eval` render (a missing table marker, target file or quality or eval evidence passes, [#141](https://github.com/vladimirrott/mojo-dllm/issues/141); an unknown mode exits 0, [#142](https://github.com/vladimirrott/mojo-dllm/issues/142)) |
| docs | `mdbook build` + `scripts/check_links.py` | a relative link is broken |
| action pins | `scripts/verify-action-pins.sh` (needs an authenticated `gh`) | a pinned SHA is not the tag its comment names |
| eval self-test | `uv run --no-project python bench/eval_gsm8k.py --self-test` | answer extraction or the paired statistics break, or `gsm8k-200.jsonl` is not 200 rows with numeric answers |
| checker self-test | `uv run --frozen --only-group fixtures python scripts/check_weights.py --self-test` | a safetensors or index guard in `check_weights.py` stops rejecting its case, or stops naming it |
| no GPU | `scripts/check_no_gpu.sh` | `--device gpu` without a GPU does anything but exit 1 with a message |
| visual options | `scripts/check_visual.sh` | an invalid `--visual-interval` does not exit 1, or `--json` with `--visual` does not exit 2, with a message naming the flag before model loading |
| fixtures | `uv run --frozen --only-group fixtures python scripts/check_fixtures.py` | `scripts/gen_fixtures.py` or `scripts/gen_moe_fixture.py` writes a file that differs from the tracked one (they run into a scratch directory), a file that is not tracked, or a tracked fixture that no generator writes and that is not a tokenizer fixture, a tokenizer golden or `golden-generation.tsv` |
| GPU test count | `scripts/check_gpu_test_count.sh` | the `def test_` functions of `tests/gpu/test_*.mojo` number other than `tests/gpu/test-count.txt` pins or differ per file from `tests/gpu/test-counts.txt`, or a file defines none; needs no GPU, runs in CI and `ci-local.sh` |
| GPU tests | `scripts/run-gpu-tests.sh` (local only) | a GPU kernel or the GPU forward pass disagrees with the CPU or the f32 reference, or the count differs from `tests/gpu/test-count.txt` or `tests/gpu/test-counts.txt` |

GitHub's runners have no GPU, so CI builds the GPU suite without running it.
`ci-local.sh` runs it when `nvidia-smi` lists a GPU and otherwise builds it and
prints a `SKIP` line for the run, never a `PASS`; `run-gpu-tests.sh` on its own exits 1 without a
GPU.

The launcher tests run `pytest -rs`, so every skip is listed in the summary,
and `tests/python/skip_policy.py` fails the run on a skip that
`tests/python/allowed-skips.txt` does not name (a test id and the text of the
reason). Add a line there in the commit that adds a skip. `--locked` makes
`uv run` fail when `uv.lock` no longer matches `pyproject.toml`; `--frozen` does
not compare them.

Branch protection on `main` requires four checks: `hygiene`, `mojo` and `docs` of the CI
workflow and `trufflehog` of the secret scan. The job `build` of the
"Deploy docs to GitHub Pages" workflow skips while the repository is private,
and a skipped job satisfies a required check, so it must not be required. CI
already runs the same `mdbook build` in `docs`.

The test count is pinned because Mojo's `TestSuite` exits 0 after running zero
tests. If you add tests, run `UPDATE_TEST_COUNT=1 scripts/run-tests.sh` (for
`tests/gpu/`, `UPDATE_TEST_COUNT=1 scripts/run-gpu-tests.sh`) and commit the
new count with them.

## Tests

`scripts/run-tests.sh` compiles each `tests/test_*.mojo` and runs it. Fixtures
live in `tests/fixtures/` and come from scripts, never from hand:

| Fixture | Made by | Checks |
|---|---|---|
| `kquants.gguf`, `bad-magic.gguf`, `tiny-llada.gguf`, `tiny-llada.expected.f32`, `tiny-dream.gguf`, `tiny-dream.expected.f32`, `malformed/`, `metadata/`, `gen_fixtures.manifest` | `scripts/gen_fixtures.py` | GGUF parsing and rejection, decoders, `ModelConfig` validation, the forward pass against a numpy reference |
| `tiny-diffusion-gemma.gguf`, `diffusion-gemma/` | `scripts/gen_fixtures.py` | the `diffusion-gemma` config (seed 610061): a 2-layer model with one sliding and one global layer, 4 experts, the tensor names and types of the real file, and one file per refusal (metadata only, or zero-filled tensors) |
| `tiny-diffusion-gemma.p0sc.case`, `.p9.case`, `.p9sc.case`, `.p150.case`, `.p150sc.case` (`tiny-diffusion-gemma.*.case`) | `scripts/gen_fixtures.py` (the forward pass is `scripts/dg_numpy.py`) | the tokens and numpy logits that `logits` on the tiny DiffusionGemma file must reproduce, for a prompt of 0, 9 and 150 tokens, with and without self-conditioning (`sc`) |
| `entropy-rows.tsv` | `scripts/gen_fixtures.py` (seed 65065, so no other fixture changes) | logits rows with the numpy f64 argmax, entropy and inverse-CDF draw that `tests/test_entropy_bound.mojo` pins |
| `utf8-cases.tsv` | `scripts/gen_fixtures.py` | byte strings with the code points and the first invalid offset that Python's own UTF-8 decoder gives, for `utf8_decode` |
| `dequant-types.gguf`, `unsupported-embedding.gguf` | `scripts/gen_fixtures.py` | Q8_0/Q5_K/BF16/Q5_0 references (seed 340034) and embedding rejection before CPU/GPU allocation (no RNG) |
| `llada-tokenizer.gguf`, `llada-tokenizer-golden.tsv`, `llada-tokenizer-golden.provenance.json`, `dream-tokenizer.gguf`, `dream-tokenizer-golden.tsv`, `dream-tokenizer-golden.provenance.json` | `scripts/gen_tokenizer_fixture.py` (`--golden-only` rewrites the golden and its provenance) | tokenizer ids against `llama-tokenize`. Each provenance file records the llama.cpp commit, the model file and its sha256, the command and the golden's sha256. The two goldens were committed with the first import, and regenerating them with llama.cpp `2e715c0639` on 2026-11-09 gave the same bytes |
| `gemma4-tokenizer.gguf` | `scripts/gen_tokenizer_fixture.py ... gemma4` | the 262,144 tokens of the DiffusionGemma file and the 472 of its 514,906 merges that apply to the 54 golden strings, tokenizer keys only and without the scores (5.5 MB) |
| `gemma4-tiny/` | `scripts/gen_fixtures.py` | a 279-token gemma4 vocabulary whose ids can be worked out by hand, and variants with one thing wrong or unusual each (a missing byte token, a merge with no space, mismatched counts, nested special tokens, an unknown-type token, an end-of-turn token the vocabulary lacks) |
| `gpt2-tiny/` | `scripts/gen_fixtures.py` | a six-token byte-level BPE vocabulary with padding, unknown and EOS ids and no end-of-turn token: the ids that LLaDA's and Dream's files leave out |
| `gemma4-tokenizer-golden.tsv`, `gemma4-tokenizer-golden.provenance.json` | `scripts/gen_tokenizer_fixture.py ... gemma4` (`--golden-only` rewrites only these) | DiffusionGemma tokenizer ids against `llama-tokenize` from the ggml-org/llama.cpp#24423 branch at the commit `scripts/build_dg_ref.sh` pins; the provenance file records that commit, the model file's sha256 and the command |
| `golden-generation.tsv` | `UPDATE_GOLDEN=1 build/tests/test_golden` | tiny LLaDA/Dream token sequences, forward-pass counts, early stopping and per-step commits |
| `src/mojo_dllm/tokenizer/unicode_tables.mojo` | `scripts/gen_unicode_tables.py` | code point classes from llama.cpp's own tables |
| `moe-q5.bin`, `moe-q8.bin`, `moe-q4-q5.bin`, `moe-q4-q8.bin` | `scripts/gen_moe_fixture.py` | deterministic raw expert tensors for scalar MoE comparisons, including mixed activation block sizes |
| GGUF/source tensor shape report | `scripts/check_weights.py` (`--self-test`) | LLaDA and Dream name mapping and safetensors header shapes; run with `uv run --only-group fixtures python scripts/check_weights.py MODEL.gguf INDEX.json` (the second argument may be one `.safetensors` file) |

`tests/gpu/test_gpu.mojo` holds the GPU tests: each kernel against the CPU
function it replaces, and the GPU forward pass on both tiny models against the
f32 reference.

The tiny model is 2 layers of hidden size 256 with random K-quant blocks. Its
expected logits come from a float numpy forward pass over the weights that
the `gguf` package dequantizes, so the Mojo forward pass is checked against
the reference formats, not against itself.

### Sampler goldens

`tests/test_golden.mojo` pins current behavior on the tiny fixtures. These
outputs come from mojo-dllm itself: they detect changes, not independent
correctness. The numpy forward-pass tests and real-model parity runs provide
that evidence. No model download or GPU is needed.

Run from the repository root:

```bash
bash scripts/build.sh tests/test_golden.mojo build/tests/test_golden
build/tests/test_golden
# Only after reviewing an intentional behavior change:
UPDATE_GOLDEN=1 build/tests/test_golden
```

Only `UPDATE_GOLDEN=1` writes the file, and `scripts/run-tests.sh` drops that variable
from the environment (and says so) unless you run `scripts/run-tests.sh --update-golden`:
the test rewrites the golden and then compares against what it wrote, so an exported
variable would make every run pass. Normal runs reject missing, empty,
duplicate, unknown or incomplete cases and compare tokens and schedule metadata
exactly. A mismatch names the case, the first differing token position (including
the prompt), both token lists and both complete rows. A golden diff in a PR needs
a sentence explaining why behavior changed, and `test_model.mojo` must still
pass. Do not regenerate merely to silence a failure on another machine; inspect
the differing logits and select a case without an unstable near-tie.

The 14 cases cover LLaDA block geometry, uneven transfer counts, serial commits,
random remasking, temperature and early stop, plus Dream's three confidence
rules, uneven timesteps, nondefault `eps` and early stop. LLaDA uses prompt
`[5, 17, 300]`; Dream uses `[448, 50, 134]` because the original prompt gave
identical all-49 outputs for D1 and D5. To avoid duplicate token lists, L5 uses
seed 1, D2 uses `eps=0.2`, D3 uses a 12-token canvas and D4 uses `eps=0.1`.
D1/D5 retain the same prompt, length and steps and differ only in `eps`.
L8/D6 use fixed stop ids 347/49 taken from the baseline outputs, never selected
at runtime. All cases disable threshold and dual-cache acceleration.

## Heavy jobs

A job that holds more memory than the machine can spare makes the kernel's global OOM killer pick a victim.
With systemd's default `OOMPolicy=stop`, systemd then stops every process in the victim's scope, and a job
started from a terminal lives in that terminal's scope: the shell, the editor session and every other session in the
window end with it. A measured run of DiffusionGemma-26B on the CPU peaked at about 25.6 GiB resident, and on a
30 GiB machine with a 2 GiB swapfile that run ended in exactly this way.

`scripts/capped.sh` starts a command in a scope of its own, under a hard cap, so only that scope can fail:

```bash
scripts/capped.sh --mem 26G -- build/mojo-dllm run --model model.gguf --prompt "hello"
```

It sets four things, each for a reason you can check in the documentation of the kernel's cgroup v2 memory
controller and of systemd's resource control:

| Setting | Effect |
|---|---|
| `MemoryMax=26G` | the hard cap (`memory.max`); the kernel OOM killer runs inside the scope when it is reached and kills nothing outside it |
| `MemorySwapMax=0` | no swap, so a job over its cap dies instead of dragging the desktop through swap |
| `OOMPolicy=kill` | one OOM kill ends the whole scope (`memory.oom.group`), not half of it |
| `CPUQuota=N00%` | only with `--cpus N`, so a job leaves cores for the desktop |

It never sets `MemoryHigh`. Below the cap, `MemoryHigh` throttles a process that has nothing to reclaim until it
stands still (state `D`, no progress); a job that needs its memory should be killed, not frozen.

The script also refuses to start a job in two cases:

- `/proc/meminfo` reports less `MemAvailable` than `--need` (default: `--mem`). A cap larger than the available
  memory does not protect the machine, because the global OOM killer still runs. Exit status 75.
- The systemd user manager cannot enforce a memory limit (no cgroup v2, or the `memory` controller is not
  delegated to it). It never runs the command uncapped. Exit status 69.

`--dry-run` runs both checks and prints the `systemd-run` command without running it. When a capped job dies with
exit status 137, the cap was most likely reached; `journalctl -k | grep oom-kill` tells the cases apart:
`constraint=CONSTRAINT_MEMCG` is the cap working, while `constraint=CONSTRAINT_NONE` with `global_oom` means the
whole machine ran out of memory.

The test suite checks the limits the kernel reports for a capped scope (`memory.max`, `memory.swap.max`,
`memory.oom.group`). The test that kills a process at its cap runs only with `CAPPED_TEST_KILL=1`, because GNOME's
`gsd-housekeeping` answers every such kill with an "Application Stopped" notification.

For a container, set the same two limits on the container: `podman run --memory 12g --memory-swap 12g`
(`--memory-swap` counts memory plus swap, so equal values mean no swap).

What does not isolate a job on its own:

- `ulimit -v` limits address space, which counts mapped files such as a 16 GB model that is read lazily, and it
  does not stop a process that stays below the limit but takes the machine's last free page.
- `systemd-oomd` reacts to memory pressure and swap use over seconds, and a job that fills memory in one
  allocation burst can reach the kernel's killer first.
- A swapfile only delays the kill and slows the machine while it lasts.

References: the kernel's [Control Group v2](https://docs.kernel.org/admin-guide/cgroup-v2.html) memory
interface, systemd's
[resource control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html) and
[scope units](https://www.freedesktop.org/software/systemd/man/latest/systemd.scope.html) (`OOMPolicy=`).

## Benchmarks

```bash
uv run --no-project python bench/run_bench.py --config bench/config.json --reps 3
uv run --no-project python scripts/bench_table.py --write
```

`bench/config.json` runs every CPU runtime in one pass and names the model
files and the llama.cpp and diffuse-cpp checkouts. The committed CPU benchmark
tables came from `bench/config-main.json`, `bench/config-diffuse.json` and
`bench/config-dream.json`; the thread sweep came from `bench/config-fair-cpu.json`. `scripts/bench_table.py` refuses a runtime that
appears in two result files, so move the files a new run replaces into
`bench/results/archive/` before `--write`. The harness waits for an idle CPU before each run, interleaves the
runtimes, switches the power profile to `performance` for the duration, and
writes a result file with every raw run. On a laptop it measures on mains
power only: unplugged, it pauses before the next run and prints that it is
waiting for the charger, and a run the charger left midway is repeated. The
eval, the quality table and `scripts/gpu_profile.py` do the same through
`bench/power.py`; `--allow-battery` measures on battery instead and records
it. The benchmark harness kills any process named
`pytest` while it runs, because test suites from other work on the
development machine kept starting mid-benchmark. Set `"kill_pytest": false`
in a config to turn that off; the GPU configs (`bench/config-gpu.json`,
`bench/config-dream-gpu.json`) do, since a GPU run leaves the CPU nearly idle.
They compare `--device gpu` with llama.cpp's diffusion example built with
CUDA in `build-cuda/`:

```bash
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda --target llama-diffusion-cli
```

### GPU kernels

`scripts/gpu_profile.py` traces one `run --device gpu` with Nsight Systems
(`nsys --trace=cuda`, which ran here without root although the driver keeps
hardware counters for admins) and writes the device time per
forward pass of each kernel group to `bench/profiles/`. It does not
synchronize the stream, unlike `MOJO_DLLM_GPU_PROFILE=1`. `--root` points it at
another checkout's build for before/after comparisons, and it reports a
median-based total next to the plain sum, because a laptop GPU's clocks move
under sustained load. `--runtime llama.cpp --seq-len N` traces llama.cpp's
CUDA `llama-diffusion-cli` on the same prompt instead (it replays its forward
pass as a CUDA graph, so the trace expands graph nodes), with the same kernel
groups. `bench/bench_gpu_gemm.mojo` times each LLaDA weight
shape alone, adding up the per-forward total from the tensor types in the
GGUF.

On the development laptop the GPU is capped at 30 W and its clocks drop under
sustained load (`bench/checks/2026-10-12-gpu-power-under-load.txt`), so
compare kernels with interleaved runs (A, B, A, B) in one session rather than
against a number from another day.

### Comparing GPU dispatch changes

`bench/compare_gpu.py` compares two already-built checkouts through
`run_bench.run_mojo`, interleaving A/B and B/A rounds on one assigned GPU.
It excludes a warmup pair per generation length, records every raw run, and
rejects failures or mismatched forward counts and canvas sizes. It leaves the
system power profile and other processes alone. Keep each pair on the same
GPU; another GPU can run a different model's independent comparison.

Both tools, and `scripts/compare_logits.py`, refuse a base and a new that are
the same checkout, the same clean commit or byte-identical binaries, and record
the SHA-256 of each `build/mojo-dllm` in the evidence, because a stale build
carries the label of whatever commit is checked out. `compare_gpu.py` sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` so that `--gpu` (an index or a `GPU-` uuid)
names the same card to CUDA and to `nvidia-smi`, records the index and the
uuid read back from `nvidia-smi`, and runs the isolation preflight and a
window around each run of `bench/isolation.py`
(`--allow-noisy` continues after a failed preflight). `check_gpu_dispatch.py`
records `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` as it found them, so
export `CUDA_DEVICE_ORDER=PCI_BUS_ID` there too.

```bash
python3 bench/compare_gpu.py --base ../base --new . --model MODEL.gguf \
  --gpu GPU-UUID --lengths 128,256,512 --reps 3 --prompts 3 \
  --out bench/results/dispatch/run.json
CUDA_VISIBLE_DEVICES=GPU-UUID python3 bench/check_gpu_dispatch.py \
  --base ../base --new . --model MODEL.gguf --arch llada \
  --out bench/checks/dispatch-exact.json
```

Use `--arch dream` for Dream. The exact check uses that model's mask token,
compares all finite logit values byte for byte at several canvas sizes, and
compares complete generated token lists. It checks preservation of the base
GPU computation, not correctness against an independent runtime. The timing
comparison runs without the synchronizing `MOJO_DLLM_GPU_PROFILE` mode.

### Quality

`bench/eval_gsm8k.py` answers a fixed 200-problem GSM8K subset
(`bench/eval/gsm8k-200.jsonl`, see `bench/eval/SOURCE.md`) and compares two
arms problem by problem: accuracy difference with a 95% interval and
McNemar's exact test. Any change that is not exact (a sampler that commits
more per step, a K/V cache) needs a comparison against the one-token-per-step
baseline before it becomes a default.

```bash
uv run --no-project python bench/eval_gsm8k.py run --model M.gguf --arm baseline
uv run --no-project python bench/eval_gsm8k.py run --model M.gguf --arm candidate -- --some-flag
uv run --no-project python bench/eval_gsm8k.py compare baseline candidate
uv run --no-project python bench/eval_gsm8k.py compare baseline candidate --limit 50
```

`compare` writes `bench/eval/<date>-<a>-vs-<b>.json`, and
`scripts/bench_table.py` turns those files into the accuracy table in
[Benchmarks](benchmarks.md). Time per problem depends on what else the
machine runs, so `--limit N` compares each arm's first N problems in run
order and writes a `-firstN` file; the table takes its times from that file.
Run that round with nothing else on the machine.

## Toolchain

Mojo is pinned to 1.1.0 and max-core to 26.6.0 in `pyproject.toml`, locked in
`uv.lock`; both come from PyPI. Bump them together by editing the two pins,
running `uv lock`, and committing that on its own with a green run: Mojo 1.1
removed `fn`, `alias` and `@parameter` closures, and each release has moved
APIs this code uses. `max-core` supplies `max.algorithm.parallelize` and the
GPU API (`max.gpu`, `max.gpu.host`). The Mojo wheels need glibc 2.34 or newer.

The Python scripts run through uv too. `uv run --no-project python script.py`
runs a script that uses only the standard library. The fixture generators and
`bench/parity.py` need numpy and gguf, pinned in the `fixtures` group:
`uv run --only-group fixtures python scripts/gen_fixtures.py`.
