# Development

## Setup

```bash
uv sync --frozen                      # the pinned Mojo toolchain, in .venv
scripts/ci-local.sh --install-hooks
```

`scripts/ci-local.sh` also needs shellcheck, markdownlint-cli2 (`npm install
--global markdownlint-cli2`), mdBook and an authenticated `gh`. A missing tool
fails its step; it never skips it.

The hooks run `scripts/ci-local.sh --fast` before each commit (hygiene scan,
`mojo format`, `-Werror` build) and the full set before each push.

## The gates

`scripts/ci-local.sh` runs the same steps as `.github/workflows/ci.yml`:

| Step | Command | Fails when |
|---|---|---|
| hygiene | `scripts/check_hygiene.sh` | a credential pattern or the author's employer name appears in tracked text |
| format | `uv run --frozen mojo format -q src tests bench`, then `git diff` | `mojo format` changes any file (Mojo has no check mode) |
| build | `scripts/build.sh` | any compiler warning (`-Werror`) |
| tests | `scripts/run-tests.sh` | a test fails, a file runs zero tests, or the total differs from `tests/test-count.txt` |
| shellcheck, yamllint, markdownlint | | lint |
| claims | `uv run --no-project python scripts/bench_table.py --check` | a benchmark table differs from what `bench/results/*.json` renders |
| docs | `mdbook build` + `scripts/check_links.py` | a relative link is broken |
| action pins | `scripts/verify-action-pins.sh` (CI) | a pinned SHA is not the tag its comment names |
| eval self-test | `uv run --no-project python bench/eval_gsm8k.py --self-test` | answer extraction or the paired statistics break, or `gsm8k-200.jsonl` is not 200 rows with numeric answers |
| checker self-test | `uv run --only-group fixtures python scripts/check_weights.py --self-test` | a safetensors or index guard in `check_weights.py` stops rejecting its case, or stops naming it |
| no GPU | `scripts/check_no_gpu.sh` | `--device gpu` without a GPU does anything but exit 1 with a message |
| visual options | `scripts/check_visual.sh` | an invalid `--visual-interval` does not exit 1 with a message naming the flag before model loading |
| GPU tests | `scripts/run-gpu-tests.sh` (local only) | a GPU kernel or the GPU forward pass disagrees with the CPU or the f32 reference, or the count differs from `tests/gpu/test-count.txt` |

GitHub's runners have no GPU, so CI builds the GPU suite without running it.
`ci-local.sh` runs it when `nvidia-smi` lists a GPU and otherwise prints a
`SKIP` line, never a `PASS`; `run-gpu-tests.sh` on its own exits 1 without a
GPU.

The test count is pinned because Mojo's `TestSuite` exits 0 after running zero
tests. If you add tests, run `UPDATE_TEST_COUNT=1 scripts/run-tests.sh` and
commit the new count with them.

## Tests

`scripts/run-tests.sh` compiles each `tests/test_*.mojo` and runs it. Fixtures
live in `tests/fixtures/` and come from scripts, never from hand:

| Fixture | Made by | Checks |
|---|---|---|
| `kquants.gguf`, `tiny-llada.gguf`, `tiny-llada.expected.f32` | `scripts/gen_fixtures.py` | GGUF parsing, decoders, the forward pass against a numpy reference |
| `dequant-types.gguf`, `unsupported-embedding.gguf` | `scripts/gen_fixtures.py` | Q8_0/Q5_K/BF16 references (seed 340034) and embedding rejection before CPU/GPU allocation (no RNG) |
| `llada-tokenizer.gguf`, `tokenizer-golden.tsv` | `scripts/gen_tokenizer_fixture.py` | tokenizer ids against `llama-tokenize` |
| `src/mojo_dllm/tokenizer/unicode_tables.mojo` | `scripts/gen_unicode_tables.py` | code point classes from llama.cpp's own tables |
| GGUF/source tensor shape report | `scripts/check_weights.py` (`--self-test`) | LLaDA and Dream name mapping and safetensors header shapes; run with `uv run --only-group fixtures python scripts/check_weights.py` |

`tests/gpu/test_gpu.mojo` holds the GPU tests: each kernel against the CPU
function it replaces, and the GPU forward pass on both tiny models against the
f32 reference.

The tiny model is 2 layers of hidden size 256 with random K-quant blocks. Its
expected logits come from a float numpy forward pass over the weights that
the `gguf` package dequantizes, so the Mojo forward pass is checked against
the reference formats, not against itself.

## Benchmarks

```bash
uv run --no-project python bench/run_bench.py --config bench/config.json --reps 3
uv run --no-project python scripts/bench_table.py --write
```

`bench/config.json` names the model files and the llama.cpp and diffuse-cpp
checkouts. The harness waits for an idle CPU before each run, interleaves the
runtimes, switches the power profile to `performance` for the duration, and
writes a result file with every raw run. It also kills any process named
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
