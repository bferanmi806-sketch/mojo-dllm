# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
semantic versioning (0.y: minor bumps may break things).

## [Unreleased]

### Added

- `bench/engines/vllm.py`, an engine for `bench/eval_engine.py` that answers GSM8K through the OpenAI-compatible
  endpoint of a running vLLM server (issue #179): `--engine vllm --base-url URL --served-model NAME --quantization Q`
  (and the declared `--checkpoint`, `--attention-backend`, `--chat-template`). It sends one chat message at
  temperature 0 and never starts or stops a server. `version` records the release `GET /version` reports plus the
  declared checkpoint, quantization and attention backend, and the arm's `model` carries the quantization so
  `compare --across-weights` sees different weights. `forward_passes` is the change of vLLM's
  `vllm:diffusion_num_denoising_steps` counter (read under that name and under its `_total` exposition name) across
  one request, null when the server has none, and refused unless `--max-num-seqs 1` is declared, because the counter
  is server-wide. Server text in errors is stripped of control characters and capped, and the arm's `--model` must
  equal the server's reported `root` or served name. The
  run is refused for a served model the server does not list, an HTTP error, an empty completion or a response
  with no token counts. `flags` says `"server": "remote"`: the isolation checks see only the client machine.
  `bench/engines/_http.py` is the shared standard-library client (no credentials, no cookies, no redirects, no
  retries, no environment proxy unless asked, a size bound and one deadline per request). `docs/benchmarks.md` has
  the runbook (rent, serve, run, destroy) with the recipe page's serve commands as read on 2026-11-09. No server
  was run and the evaluation on a rented GPU has not happened.
- `bench/engines/hf.py`, an engine for `bench/eval_engine.py` that runs GSM8K through the authors' own sampling
  code (issue #178): `generate` from a checkout of ML-GSAI/LLaDA (`--llada-repo`, imported and never copied) and
  `diffusion_generate` from Dream's remote code, at temperature 0 with steps equal to the generation length.
  Options: `--hf-model`, `--quant none|4bit`, `--model-revision` and the shared `--device`. Each row's `version`
  records the model repository's commit sha (`null` when none was given), the LLaDA checkout's commit and the
  `transformers`, `torch` and `bitsandbytes` versions; `flags` records where a Dream run departs from its README.
  The adapter imports `torch` and `transformers` only when it loads a model, so the frozen environment and CI
  never need them. `eval_gsm8k.py compare --across-weights` compares arms that ran different model files and writes
  both names and `weights_differ` (`compare` writes `models` and `weights_differ` on every run now, and still
  refuses different models without the flag). `docs/benchmarks.md` explains why the arm's accuracy includes
  quantization and how to set up one environment per model. A Hugging Face repository now needs
  `--model-revision` (or `--allow-unpinned`), because the loader runs the repository's Python; the arm's `model`
  includes `|4bit` so `compare` sees quantization; the load is refused before `from_pretrained` when `MemAvailable` is below an estimate from the weights'
  size (and when that size or `/proc/meminfo` is unknown), and the preflight after the load asks for the 3 GB of
  working memory only. The memory-fit spike and the GSM8K arm of #178 have
  not run.
- `scripts/capped.sh` runs a command in its own memory-capped systemd scope: a hard cap, no swap and a
  whole-scope kill on an OOM, so a job that outgrows the machine dies alone instead of taking its terminal with
  it. It refuses to start when less memory is available than the job needs or when the user manager cannot
  enforce limits. `docs/development.md` ("Heavy jobs") explains the settings and what does not isolate a job.
- The DiffusionGemma reference (issues #66 and #62). `scripts/build_dg_ref.sh`
  builds the ggml-org/llama.cpp#24423 branch at a pinned commit, CPU only and
  offline, into its own tree beside the mainline checkout. `tools/ref/dg_ref.cpp`
  writes the canvas logits of one denoising step for a fixed prompt and canvas:
  the unified `[prompt | canvas]` pass or the prompt K/V path (prefill, then one
  decode step), each with or without self-conditioning from a given
  previous-step buffer. `bench/parity.py --arch diffusion-gemma` runs the
  reference phase, and a comparison phase that stops with a message until
  mojo-dllm's own outputs exist (#64). The comparison's bound comes from the
  LLaDA evidence plus a check on each row's largest logit, and every case
  records how far the reference lies from itself. `bench/check_dg_bound.py`
  shows the bound rejecting a missing final soft-cap and a dropped layer scale
  (`bench/checks/2026-11-04-dg-bound.json`). The tokenizer goldens for `gemma4`
  are in `tests/fixtures/gemma4-tokenizer-golden.tsv`, with their provenance.
- DiffusionGemma's entropy-bounded denoiser (`_generate_diffusion_gemma`,
  issue #65): a random canvas over the whole vocabulary, a per-position
  argmax, entropy and draw from one pass over each logits row, acceptance of
  the lowest-entropy positions while the entropies before them sum to at most
  the bound, and a stop once the argmax canvas holds and the mean entropy is
  small. Six flags (`--eb-steps`, `--eb-t-min`, `--eb-t-max`, `--eb-bound`,
  `--eb-stability`, `--eb-confidence`) default to the llama.cpp branch's
  values. `--json` gains `steps_taken` and `stop_reason` for every model. The
  forward pass is #64; tests use a scripted model.
- `scripts/check_weights.py` compares GGUF tensor shapes with safetensors
  headers, with LLaDA/Dream name mapping and a standalone self-test.
- Q8_0 and Q5_K reference decoders, with GGUF type sizing and reference tests
  for Q8_0, Q5_K and BF16. Matrix kernel support remains Q4_K and Q6_K.
- A Q5_0 reference decoder (22-byte blocks of 32 weights) with GGUF type sizing,
  a reference test against gguf-py's decoder and a hand-built block test of the
  high-bit layout. It is the first piece of the DiffusionGemma reader (#61).
- A `diffusion-gemma` branch of `ModelConfig` reads the DiffusionGemma
  metadata (per-layer sliding/global pattern and K/V head counts, head sizes and
  RoPE bases per kind, experts, canvas length, soft cap) and checks the tensor
  directory against it: `attn_v` exactly on sliding layers, K width, the 3-D
  expert tensors and the rope table. A wrong key or tensor is refused by name,
  as are non-zero `attention.shared_kv_layers` and
  `embedding_length_per_layer_input`. `inspect` prints the config. `DiffusionLM`
  and the GPU model stop on these files (#61). The key names come from the
  unmerged ggml-org/llama.cpp#24423.
- `DiffusionGemma`, the CPU forward pass of DiffusionGemma, in
  `models/diffusion_gemma.mojo` (#64): one unified pass over [prompt | canvas]
  with sliding and global layers, the region masks, RoPE with `rope_freqs`,
  attention scale 1, the dense MLP plus 128-expert MoE, the two embedding and
  layer-scalar regions, the tied Q6_K head and the soft cap. `mojo-dllm logits`
  runs it. A numpy forward pass written from the llama.cpp branch
  (`scripts/dg_numpy.py`) checks it on a tiny fixture: most rows agree to 2e-7,
  a few to 7e-3 where an int8 rounding step lands on a tie. `prefill` and `step`
  run the prompt once and then the canvas against its cached K and V; they agree
  with the unified pass bit for bit on the tiny model. `forward_sc` and `step_sc`
  add self-conditioning: the previous step's logits, softmaxed at the previous
  temperature, go through the transposed embedding table and the `self_cond`
  MLP into the canvas embedding. `logits --path cached --prev FILE
  --prev-temp-inv X` reaches all four. The real-file parity record follows in
  #64.
- Fused elementwise kernels on the GPU: the residual add, the RMSNorm and the
  activation quantization run as one launch (three per layer boundary plus
  the first norm), and `silu(g) * u` is quantized in the same launch as the
  product. A layer goes from nine elementwise launches to four. Each fused
  kernel evaluates the unfused kernels' f32 expressions in the same order, and
  the GPU tests compare each against its chain bit for bit (issue #10).
  LLaDA-8B at 148 tokens: 236.1 to 226.4 ms per forward (median of three
  interleaved traces, `bench/profiles/2026-10-27-fuse-*`). The fused kernels
  account for 5.7 ms of that and the rest is GEMM timing variation between
  runs. The forward logits are identical (`bench/checks/2026-10-27-fuse-*`).
- Greedy decoding scores each requested row on the GPU. One block per row
  reduces its logits to the first argmax, the maximum, the sum of
  exp(x - max), the runner-up logit and the sum of exp(x - max)(x - max); the
  host turns those numbers into the LLaDA probability or Dream's entropy,
  maskgit_plus and topk_margin confidence, so a step copies 36 bytes per
  requested row instead of the logits rows. `--temperature` above 0 and
  `--full-logits` keep the host path. Tests compare the kernel with the host
  loops (issue #9). LLaDA-8B at 148 tokens: 250.5 to 234.3 ms per step, with the
  forward pass unchanged (234.6 and 234.3 ms) and the forward logits identical
  to main's (median of three interleaved traces,
  `bench/profiles/2026-10-27-argmax-*`, `bench/checks/2026-10-27-argmax-*`).
- Generation stops once the printed answer is final (an end-of-text token
  after a fully committed answer). The output is identical; `--no-early-stop`
  runs every step, and the benchmark, parity, quality and profiling harnesses
  pass it so comparisons stay step for step. The GSM8K evaluation leaves early
  stop on.
- `scripts/gpu_profile.py` (per-kernel device time from an Nsight Systems
  trace), `bench/bench_gpu_gemm.mojo`, and `bench/eval_gsm8k.py`, a paired
  GSM8K-200 evaluation for changes that are not exact.
- `--threshold P` (LLaDA): each step also commits every masked token whose
  confidence reaches P, as Fast-dLLM's parallel decoding does.
- `--dual-cache` (LLaDA, CPU and GPU): keys and values for every position
  are cached at the first step of a block, and the block's other steps
  recompute only its own rows, as Fast-dLLM's DualCache does.
  [Benchmarks](docs/benchmarks.md) has the GSM8K-200 comparison of both
  options with the plain sampler, rendered from `bench/eval/` by
  `scripts/bench_table.py`; `compare --limit N` times one round.

- NVIDIA GPU backend, `--device gpu`, for compute capability 8.0 and newer.
  Projections read the GGUF Q4_K / Q6_K blocks on the GPU and multiply on the
  int8 tensor cores; attention runs flash-style. GPU tests check each kernel
  against its CPU twin, and the parity harness takes `--device gpu`.
- `MOJO_DLLM_GPU_PROFILE=1` splits `--verbose` timings on the GPU.
- `bench/config-gpu.json` and `bench/config-dream-gpu.json`, against
  llama.cpp built with CUDA, and their results. Results record runs that
  started on a busy CPU, and the tables give the quiet-run medians next to
  them. The first results (2026-10-09) had llama.cpp's CUDA build faster than
  `--device gpu` on the RTX 3050 tested; the kernel work since then reverses
  that for LLaDA (see Changed).
- Fairness runs for the CPU comparison: llama.cpp is fastest at 12 threads,
  the most the test CPU runs, and mojo-dllm's forward pass alone stays ahead
  of `llama-bench`.

### Changed

- The toolchain comes from uv (`uv sync --frozen`, `uv.lock`) instead of pixi, and
  the `./mojo-dllm` launcher wraps `setup`, `pull`, `run` and `doctor`. The
  binary exits 2 on a usage error and names the flag, command or value that
  is wrong.
- The GPU benchmark tables are refreshed against llama.cpp `7df8b87` (issue
  #15), on mains power, one result file per table
  (`bench/results/2026-11-03-*`; the 2026-10-12 files moved to
  `bench/results/archive/`). The step time of mojo-dllm over llama.cpp CUDA,
  as the median of paired runs, went from 1.660 to 0.906 for LLaDA and from
  1.759 to 1.022 for Dream (from 2.413 and 2.391 in the 2026-10-09 files). Both
  new ratios are rendered under their tables by `scripts/bench_table.py`, which
  also refuses a ratio from two result files and a summary that disagrees with
  its runs. `bench/checks/2026-11-03-gpu-power-limit.txt` records the 30 W limit
  and three builds run back to back: the earlier kernel gains hold, and the day
  to day swing in absolute milliseconds follows the SM clock.
- Timed runs happen on mains power. `bench/run_bench.py`,
  `bench/eval_gsm8k.py`, `bench/quality.py` and `scripts/gpu_profile.py` pause
  before the next run while the machine is on battery and resume when the
  charger is back; a run the charger left midway is thrown away and repeated.
  Each run records `power_retries` and `on_battery`, and `--allow-battery`
  measures on battery instead. A benchmark refresh started unplugged on
  2026-10-29, and nothing in the harness noticed.
- The GGUF reader rejects a `general.alignment` that is not a uint32 or not a
  non-zero power of two, and an array whose element count is negative or
  larger than the bytes left in the file, naming the key (#55, #56). An
  alignment of 0 used to point tensors at the file header.
- Model configuration rejects incompatible RoPE widths/scaling and attention
  key lengths, and unsupported-architecture errors list LLaDA and Dream.
- Matrix load errors name the tensor and distinguish an unsupported type from
  a decodable type without matrix kernels; unsupported embeddings fail at load.
- The Q6_K GPU GEMM copies each 224-byte block to shared memory as it is,
  unpacks the A fragments in registers and prefetches the next super-block,
  as the Q4_K kernel does. The staged block grows from 212 to 224 bytes (74.6
  MB more device memory on LLaDA-8B). The accumulation order is unchanged.
  LLaDA-8B at 148 tokens: 166.1 to 151.0 ms per forward, the Q6_K GEMM going
  from 44.5 to 27.3 ms of kernel time (median of three interleaved traces,
  `bench/profiles/2026-10-27-q6k-*`). The forward logits are identical
  (`bench/checks/2026-10-27-q6k-*`; issue #11).
- The Q4_K GPU GEMM keeps weights packed in shared memory and unpacks them
  in registers, staging with 16-byte copies; a block now covers 128 rows with
  8 warps. LLaDA-8B at 148 tokens: Q4_K GEMM 186.9 to 148.4 ms per forward
  in interleaved traces (`bench/profiles/2026-10-13-gemm3-*`), logits
  unchanged (`bench/checks/2026-10-13-gemm3-logits.json`).
- The Q4_K GPU GEMM reads the next super-block into registers while the
  tensor cores work on the current one. LLaDA-8B at 148 tokens: Q4_K GEMM
  148.8 to 87.2 ms per forward (median launch time times launches, middle
  value of three interleaved traces, `bench/profiles/2026-10-13-gemm4-*`),
  logits unchanged
  (`bench/checks/2026-10-13-gemm4-logits.json`).
- GPU attention runs on the f16 tensor cores (`mma.m16n8k16`, f32
  accumulation): each warp takes 16 queries, and P goes from the score mma to
  the P V mma without leaving the registers. LLaDA-8B at 148 tokens:
  attention 17.2 to 8.5 ms per forward in interleaved traces
  (`bench/profiles/2026-10-13-attn-*`). Logits stay as close to the f32 CPU
  path as with f32 attention (`bench/checks/2026-10-13-attention-f16-logits.json`).
- `scripts/compare_logits.py` compares two builds' GPU logits on a real model
  against the f32 CPU path, for kernel changes that are not exact.
- GPU activation quantization runs one warp per 32-value block instead of
  one thread, with the same output bit for bit. LLaDA-8B at 148 tokens:
  quantization 30.5 to 5.9 ms per forward, measured forward 397-401 to
  375-385 ms in interleaved traces (`bench/profiles/2026-10-12-quant-*`).
  `bench/profiles/2026-10-12-llama-cuda.json` traces llama.cpp's CUDA build
  on the same canvas for comparison.
- The GPU GEMM picks its token tile per call (32, 64 or 80 wide), so a
  148-token canvas unpacks each weight tile twice instead of five times, and
  the activation scales are laid out for 128-bit shared-memory loads. LLaDA-8B
  GEMM time per forward: 395 to 286 ms in interleaved traces
  (`bench/profiles/2026-10-12-gemmtile-*`); outputs change only by rounding
  (`bench/checks/2026-10-12-gemm-tiles-equivalence.json`).
- The GPU GEMM runs a weight tile's token tiles back to back, so they share
  it through L2. Logits are byte-identical
  (`bench/checks/2026-10-12-grid-order-logits.json`), and
  `bench/profiles/2026-10-12-gemm-*` hold four interleaved before/after traces.

- The samplers' per-row confidence pass runs on `--threads` workers. Each
  row's arithmetic is unchanged, so tokens do not depend on the thread count.

## [0.1.0] - 2026-10-04

First release: CPU inference for two diffusion language models from GGUF.
Measured speed against llama.cpp and diffuse-cpp is in
[docs/benchmarks.md](docs/benchmarks.md); agreement with llama.cpp is in
[docs/correctness.md](docs/correctness.md).

### Added

- LLaDA-8B and Dream-7B from Q4_K_M GGUF files (F32, Q4_K and Q6_K tensors).
- GGUF v2/v3 reader over a memory map, with `mojo-dllm inspect`.
- Int8-activation GEMM over weights repacked 8 rows per register, using
  AVX-VNNI when present, with a work-claiming thread pool for hybrid CPUs.
- Bidirectional attention with grouped-query heads; NORM and NEOX RoPE; Q/K/V
  bias; logits computed only for the positions a step samples.
- Byte-level BPE tokenizers (`bailingmoe`, `qwen2`) matching `llama-tokenize`
  id for id on a 36-case multilingual corpus each.
- LLaDA's block sampler (low-confidence or random remasking, Gumbel
  temperature) and Dream's timestep sampler (entropy, maskgit_plus,
  topk_margin).
- `mojo-dllm run`, `tokenize`, `inspect` and `logits`.
- Benchmark, parity and quality harnesses, with every published number
  rendered from their result files.
- Local-first CI (`scripts/ci-local.sh`, git hooks) mirrored in GitHub
  Actions; mdBook documentation.
