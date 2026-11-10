# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
semantic versioning (0.y: minor bumps may break things).

## [Unreleased]

### Added

- `bench/engines/sglang.py`, an engine for `bench/eval_engine.py` that answers GSM8K through the OpenAI-compatible
  endpoint of a running SGLang server (issue #180): `--engine sglang --base-url URL`, with the declared `--dtype`,
  `--algorithm`, `--checkpoint`, `--max-running-requests` and `--chat-template`. The request has four fields (`model`,
  one user message, `max_completion_tokens`, `stream` false) and never carries a field that the DiffusionGemma algorithm
  rejects with HTTP 400 (logprobs, a grammar, a penalty, a seed) or ignores (`temperature`, `top_p`, `top_k`). SGLang
  has no `/version` route, so the release,
  the diffusion algorithm, the dtype and the batch size come from `GET /server_info` and the checkpoint from
  `GET /model_info`. A declared value that contradicts a reported one is refused, a server that runs no diffusion
  algorithm or reports a quantization is refused, and the batch size must be 1 (reported or declared). The arm's
  `model` carries the dtype, so `compare --across-weights` sees different weights, and `flags.model_matches_server` is
  true only when `--model` is the checkpoint the server reports (false for an alias that matches the served name only).
  `forward_passes` is null in every
  row: SGLang exposes no per-request step count on `main` or in v0.5.23 (sgl-project/sglang#48675 was open and
  unmerged on 2026-11-09), and the engine never reads `/metrics`. The model-list and completion checks moved from the
  vLLM engine into `bench/engines/_openai.py`, unchanged, and both engines use them. The shared HTTP client also
  strips control characters from, and caps, the text of a connection or transport error, as it does for server text.
  `docs/benchmarks.md` has the runbook with the cookbook's serve command as read on 2026-11-09, the note that the bf16
  checkpoint is 48.1 GiB, the Vast.ai API key read into `VAST_API_KEY` instead of passed as an argument (a test scans
  the fenced blocks of the file for a key in an argument list), and a
  DiffusionGemma section that lists the llama.cpp reference, vLLM, SGLang and mojo-dllm rows, each marked
  `no committed run yet`. No server was run and the evaluation on a rented machine has not happened.
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
  forward pass is #64, and #277 tracks wiring it to this loop; tests use a scripted model.
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

- The transposed embedding table of DiffusionGemma's self-conditioning is no longer zeroed before it is written, and the
  `soft_embed` partial sums and the CPU attention scores are not zeroed where every element is written first
  (issue #226). `Floats` takes `zero=False` for that. At the real size (vocabulary 262144, hidden 2816, 256 canvas
  rows) building the table took 3.5 to 6.2 s before and 1.8 to 2.1 s after in one set of three interleaved runs (one
  before run overlapped other work), and 2.2 to 2.8 s before and 1.2 to 1.3 s after in a later set on an idle machine
  (about 1.8 times faster); the peak RSS is unchanged at 3.89 GB and a `soft_embed` call is unchanged within noise
  (2.0 to 2.1 s). Each 64-row scratch block of the table build is still zeroed. The numbers are bit for bit the same.
- The CPU MoE router names the token whose router scores are NaN or infinite (`non-finite router scores for token N`,
  issue #217). Before, every probability of that token was NaN, the selection picked expert 0 for every slot, and
  `grouped_experts` reported "MoE routing repeats expert 0", which hides the cause. `Routing` also keeps its capacity
  apart from the number of tokens in use: `set_len` refuses a length above the capacity, and `route_tokens` refuses a
  count the buffers cannot hold before it writes any id.
- `logits --prev-temp-inv` for DiffusionGemma refuses a value whose product with the final logit soft cap exceeds
  1e8 (issue #229). The old bound, 3e38, accepted values that turned the whole canvas to NaN without an error: the
  softmax subtracts the maximum from an unrounded product, and from about 1e9 (cap 30) the leftover residue makes the
  exponential infinite. Measured on 64 rows of logits in [-30, 30] at the real vocabulary width of 262144: NaN in 61 of 64 rows at 1e9 and in
  none up to 1e8. The onset depends on the row width: at width 4096, 21 of 64 rows are NaN at 1e8.
  With the real cap of 30 the largest accepted value is about 3e6, a temperature of 3e-7.
- Mapping a model file names the operating system's reason when it fails (issue #224). A failing `lseek` is no longer
  reported as "file is empty or unreadable" (an empty file now says `file is empty: PATH`; a size that cannot be read
  says `cannot get the size of PATH: <reason>`), and `mmap failed for PATH` carries the reason too. `getrusage`
  failures no longer read as a peak of 0: `run --json` prints `"peak_rss_bytes": null` and `--verbose` prints
  `peak RSS: unavailable (...)`.
- The sampler reads the mask id from the model's config, and the run keeps one thread count and one step count
  (issue #219). `GenConfig.mask_id` and its default of 126336, which every architecture inherited until the CLI
  overwrote it, are gone: `generate` and the Dream and LLaDA loops use `model.config().mask_id`. `GenResult.steps_taken`
  is a method that returns `forward_passes`, which it always equalled, and `_run` takes its thread count from the
  config it receives instead of a second parameter. The `--json` output is unchanged.
- `run --json` keeps stdout clean (issues #223 and #110): `--json` together with `--visual`, `--verbose` or
  `--dump-step-stats` is a usage error (`--json cannot be combined with --visual`, exit status 2, before the model
  loads), because each of them prints to stdout before the JSON object. Every `ERROR:` line, usage error or not, now
  goes to stderr; help text and `--version` stay on stdout. `docs/cli.md` no longer tells consumers to select the line
  that starts with `{`.
- The flag parser refuses what it used to accept silently (issue #222): a lookup of a flag name that the command
  does not declare (`a.int_or("max-token", 128)` returned the default), a value that is the name of a declared flag
  (`--prompt --verbose` took `--verbose` as the prompt), a value flag given twice (the last won) and an empty entry in
  `--tokens` or `--rows` (`--rows 0,,2` named two rows; a trailing comma stays legal). Each usage error names the
  flag and exits with status 2. The range of each numeric flag lives in one place, `int_flag_rule` and
  `float_flag_rule` in the sampler, which the command line and the sampler both use, so `--temperature` is now checked
  by the sampler too (a NaN used to read as greedy). `Decision` has `help`, `error` and `run` constructors and derives
  the exit status, and `CanvasPrinter` refuses an interval below 1 in its constructor.
- `logits` writes its `.f32` file through `FILE.tmp`, checks the size and renames it (issue #208). On a full disk
  `FileHandle.write_bytes` does not return an error: in Mojo 1.1.0 it calls `abort()` when `write()` fails
  (`std/io/file.mojo`, tag `mojo/v1.1.0`, line 598), so the old code ended with SIGABRT and a partial file under the
  final name. The new code uses `write_all` and `close`, which raise, then compares the size, and removes the
  temporary file on any error; the message is `cannot write FILE: <reason>` with exit status 1.
- `tokenize` and `run` refuse a chat prompt for an architecture other than `llada`, `dream` and `diffusion-gemma`
  (issue #207). `chat_prompt` fell through to the LLaDA header template for every other architecture, so
  `tokenize --model qwen2.gguf --prompt hi` printed the ids of text that model never saw and exited with status 0. The
  error names the architecture and suggests `--no-chat`, which still tokenizes the text unchanged.
- `--threads` reaches every parallel CPU loop (issue #206): `quantize_acts` and the weight repack ran on every core
  whatever the setting, so a `--threads 1` timing included all-core work. Both now use `parallel_for` with the worker
  count, and the model, DiffusionGemma and MoE paths pass it; each token and each 8-row group is independent, so the
  bytes are identical for every count. A Dream run refuses `--full-logits`, `--threshold` and `--dual-cache` by name
  before the model loads, and `run --json` reports `remasking` (Dream's `low_confidence` is `entropy`) and omits
  `block_length` and `full_logits` for Dream, which reads neither. Dream still accepts `--block-length`: five scripts
  in `bench/` and `scripts/` pass it for every architecture.
- `run` refuses a DiffusionGemma file as soon as its header is read, before the tokenizer is built and the prompt
  encoded, and says that the architecture runs through `logits` only until the generation loop is wired (issue #205,
  wiring is tracked by #277). `run --help` and `docs/cli.md` mark the `--eb-*` flags as having no effect yet. The
  entropy-bounded sampler also refuses a generation length that differs from the model's canvas length, which
  overflowed the logits buffer when smaller and moved the prompt boundary otherwise. The `check_runnable` docstring no
  longer says that no forward pass exists.
- The LLaDA and Dream loops stop with `non-finite logits at canvas position P, step S` when a row's softmax is not a
  number, on the CPU and the GPU statistics path (issue #196). A single Q4_K block with an f16 scale of NaN used to
  give a canvas of token 0 repeated, stop reason `complete` and exit status 0. The GPU kernel's tie and NaN rule is
  unchanged; the host checks the statistics it returns.
- `run` refuses a canvas longer than the file's context length before any weight is mapped (issue #201): the prompt
  tokens plus `--max-tokens` must not exceed `<arch>.context_length` (4096 when the key is absent). A
  `--max-tokens` of 2^62 used to wrap the activation sizes and write past a small buffer after the whole model had
  loaded. `Floats` and `alloc_bytes` also refuse a size that does not fit in an Int, with a named error. No script
  or document in `bench/`, `scripts/` or `docs/` asks for a canvas near 4096: the largest generation length any of
  them passes or defaults to is 256.
- `run` and `logits` check every option that the flags alone decide before the model file is opened (issue #202):
  `--threshold` (finite, 0 to 1), `--prev-temp-inv` (finite, above 0), `--device`, `--path`, `--remasking`, and a
  `--block-length` or `--steps` that does not divide `--max-tokens`. Each exits with status 2 and names the flag.
  A default block length that does not divide `--max-tokens`, and a `--remasking` the architecture does not take, are
  refused once the header is read, before any weight is mapped. `logits` on a DiffusionGemma file checks `--rows` and
  the size of `--prev` against the header's canvas length before the weights load.
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
- The GGUF reader compares a length with the bytes left in the file instead of
  adding it to the read position, so a string length between 2^63 minus the
  position and 2^63 no longer wraps past the bounds check and aborts the
  process; a `Reader` that starts outside the file refuses to exist (#194).
- The GGUF reader checks every tensor of the directory before any loader sees
  it: each used dimension is at least 1, the element and byte counts are
  computed by dividing first so they cannot wrap, a row of a quantized type
  ends on a block boundary, the offset is not negative and a multiple of the
  alignment, and the bytes lie inside the file. A repeated tensor name or
  metadata key is refused (the first used to win without a message). `TensorInfo`
  has a `supported` field in place of the byte count of -1 that marked a type
  the reader cannot size, and `ggml_row_bytes` refuses a partial block (#195).
- `PackedMatrix` and `ExpertWeights` take the byte count of their source along
  with its address and refuse a source that does not hold exactly the rows the
  shape needs. Callers that build them by hand (the benchmarks, the tests) pass
  the size of the buffer (#218).
- `Mapping.drop` and `Mapping.populate` take an offset into the file in place of
  an absolute address and raise unless the range lies inside the mapping; a
  bad range used to reach `madvise`, where `MADV_DONTNEED` outside the map
  zero-fills heap pages. `Mapping.ptr()` returns a pointer with an immutable
  origin, so a store through it no longer compiles (#216).
- For LLaDA and Dream, `ModelConfig` checks that `token_embd.weight` and, when
  the file has one, `output.weight` are `[hidden, vocab]`, with the vocabulary
  taken from `<arch>.vocab_size` or from the rows of `output.weight`. A table
  with fewer rows than the vocabulary let a token id pass the guard and read
  past the tensor. The check runs before any loader and replaces the width test
  that the CPU and the GPU loader each carried; the error names the tensor,
  both shapes and the keys that set them (#198).
- For LLaDA and Dream, `ModelConfig` refuses a block count, hidden width, head
  count, KV head count, feed-forward width or vocabulary size that is zero or
  negative, and an epsilon or RoPE base that is zero, negative, NaN or
  infinite, naming the key. A zero block count used to load a model with no
  layers, a zero KV head count one with no KV width, and a negative head count
  a negative head dimension (#199).
- `inspect` and `run --verbose` list each metadata key that the file lacks and
  the reader replaced by a default, as `llada.rope.freq_base = 10000 (default,
  key absent)`: the KV head count, the context length, the rope base and the
  logit shift for LLaDA and Dream, and the context length and the final logit
  softcap for DiffusionGemma. A converter that drops `rope.freq_base` no longer
  runs on a value nobody sees. The defaults match llama.cpp's except two, which
  stay as they were: llama.cpp requires `context_length`, where this reader
  assumes 4096 and enforces it only for `run` (the prompt plus `--max-tokens` must fit, issue #201), and its diffusion example reads a missing
  `diffusion.shift_logits` as true for every architecture, where this reader
  takes false for LLaDA and true for Dream. `inspect` of a LLaDA or Dream file
  also prints `config: refused:` with the reason when its metadata is invalid
  (#204).
- `utf8_decode`, which the byte-level tokenizer applies to every prompt fragment
  and to every token text, requires the continuation bytes of a sequence and
  refuses overlong forms, surrogates and values above U+10FFFF. A byte that
  does not begin a valid sequence becomes U+FFFD and decoding goes on at the
  next byte, as in llama.cpp's `unicode_cpts_from_utf8`. It used to take the
  next one to three bytes whatever they were: the Latin-1 text `caf\xe9 au
  lait` decoded `E9 20 61` to U+9821 and lost the space and the `a`. The check
  lives in the new `formats/utf8.mojo`. `run` and `tokenize` refuse a
  `--prompt` that is not valid UTF-8 with exit status 2 and the offset of the
  first bad byte (`flag --prompt is not valid UTF-8: invalid byte at offset 3`);
  the Mojo runtime does not refuse such an argument itself (#197).
- The byte-level tokenizer loader and the gemma4 vocabulary share one
  `validate_token_table` (new `tokenizer/vocab.mojo`) and one `special_token_id`.
  The byte-level loader now refuses a token type array of another length than
  the token array, a repeated token text (the last id used to win), a special
  token id outside the vocabulary, a missing EOS id for LLaDA and Dream, and
  `add_bos_token` without a BOS id, each naming the key. `run` compares the
  tokenizer's vocabulary size with the model's before it loads the weights
  (#200).
- The GGUF reader checks that every string it turns into a `String` (a key, a
  tensor name, a string value, a token text) is valid UTF-8 and raises with the
  entry or key, the file offset and the bad byte; `inspect` prints
  `<string, N bytes, not valid UTF-8 at byte B>` for such a value (#225).
- Three validity gaps close (#227): `Gemma4Vocab` and the byte-level loader refuse
  a negative special token id other than -1, `Gemma4Vocab` refuses a token type
  outside 0 to 6 (it used to read it as attribute 0, which decodes to empty
  text), a DiffusionGemma file with `self_cond_gate.weight` and a vocabulary
  that is not a multiple of 8 is refused at load and not at the first
  self-conditioned step, and `Rng.below` raises for a bound of 0 or less in
  place of returning `n - 1`.
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
- The launcher tells the causes of a Hugging Face failure apart (#254). A gated repo and a missing repo keep their
  messages. Any other HTTP status error reports the status code and says to check the repo name and `HF_TOKEN`. A
  failure to write into the cache (a full disk, a read-only `HF_HOME`) names the cache directory. Only a transport
  failure (refused, timed out) says "could not reach Hugging Face" and suggests `--offline`.
- Without `:QUANT`, the launcher picks Q4_K_M, then Q4_K_S, then Q6_K, the quants the binary loads (#255). It used to
  fall back to Q8_0 and then to the first `.gguf` in the repo, which downloaded gigabytes the binary then refused. A
  repo with none of the three fails with the file listing and the `owner/repo:QUANT` syntax.
- A cached model file that has not been checked against its sha256 is no longer reported as healthy (#256).
  `pull --offline` and `setup` print `already downloaded (not verified)` and exit with status 3 (`setup` lists
  `./mojo-dllm pull` as the outstanding step), and `doctor` marks the `model` line `!` with the pull command.
- The launcher checks a cached file against the hub's hash at the commit the file was cached at (the name of its
  snapshot directory), not at the newest `main` (#257). A file that another tool cached at commit A, and that upstream
  re-uploaded later, is kept. A file that fails its check loses its snapshot link and its blob, and a link whose target
  lies outside the Hugging Face cache loses only the link.
- The launcher rebuilds the binary when `pyproject.toml`, `uv.lock` or `scripts/build.sh` is newer than it, and when
  the pinned Mojo version or the GPU architecture differs from the stamp `build/mojo-dllm.stamp` that a launcher build
  writes (#258). A binary without a stamp is rebuilt once. A `src/` with no `.mojo` file is an error instead of a
  fresh binary, and `doctor` says which of these made the binary stale.
- `./mojo-dllm setup` no longer reports success on paths where a step did not happen (#259). The smoke test must print
  text and finish within 300 seconds (a binary that exits 0 with no output, or hangs, is an error). The build step
  checks that `build/mojo-dllm` exists. `setup --no-model` ends with `Setup complete (no smoke test).`. `setup --offline`
  and `HF_HUB_OFFLINE=1` use the cache only, as `run` and `pull` do. Ctrl-D at the download question declines it instead
  of printing a traceback.
- `./mojo-dllm doctor` keeps the reason a probe failed (#260). The `toolchain` line says whether `uv run` timed out,
  could not start or exited non-zero (with the first line of its stderr), the `gpu` line carries the `nvidia-smi` failure
  (a driver and library mismatch is visible), an unreadable `/proc/cpuinfo` or `/proc/meminfo` is reported with the system's
  reason instead of "no AVX2", and a cache lookup that failed names its error instead of "not in the cache". The driver
  version comes from the same `nvidia-smi` call as the GPU list.
- The launcher prints one error line, not a Python traceback, for the failures that were not `LauncherError` (#261): an
  `OSError` (a read-only home directory, a full disk, a missing `bash`) names the path and the system's reason, an empty
  `get_paths_info` answer says the hub has no such file, and Ctrl-C prints `interrupted` and exits 130. A failed write of
  `verified.json` says that the download matched its sha256 and will be checked again. The size and mtime recorded for a
  verified file are the ones taken before it was hashed, so a file replaced during the hash is not recorded as verified.
- `--device auto` now names the GPU the binary will open (#262). The launcher queries the index and UUID of each card
  and sets `CUDA_VISIBLE_DEVICES=<uuid>` and `CUDA_DEVICE_ORDER=PCI_BUS_ID` before it starts the binary (and for the
  `setup` smoke test), so a machine with an older card first in the `nvidia-smi` list no longer prints one GPU and runs
  on another. An explicit `--device gpu` runs the same detection, pins the first qualifying card, and prints a warning on
  stderr when `nvidia-smi` fails, finds no card or finds only cards below compute capability 8.0.
- `mojo-dllm.lock` locks the dependency tree of the `./mojo-dllm` script, which `uv run --script` uses (#263). Before,
  the first run resolved `huggingface-hub`'s transitive packages (httpx2, httpcore2, hf-xet, certifi, filelock, tqdm) at
  their latest versions without hashes. CI and `scripts/ci-local.sh` run `uv lock --script mojo-dllm --check`, and
  `tests/python/test_launcher_script.py` checks that the header pin equals the `huggingface-hub` version in `pyproject.toml`,
  `uv.lock` and the script lock, that every locked package has sha256 hashes, and that the script starts.
- The launcher refuses `--flag=value` (for example `--device=cpu` or `--steps=8`) with a message that shows the
  two-argument form, instead of swallowing the next word as the flag's value, and `-h` and `--help` after a bare `--`
  are part of the prompt instead of a help request, so `./mojo-dllm run -- "-h"` runs with the prompt `-h` (#264).
- `./mojo-dllm setup` uses the CPU, and says why in the `device:` line, when `nvidia-smi` prints a listing the launcher
  cannot read, instead of stopping with an error after the model download (`doctor` already treated that case as a
  warning) (#267).
- The launcher's `doctor` accepts a pinned Mojo version with a suffix (`1.1.0.dev1`) when `mojo --version` prints the
  same suffix; it compared only the first three numbers and reported a mismatch. `nvidia-smi` rows with a non-finite or
  non-positive compute capability or a negative memory figure are treated as unreadable (`inf` used to pass the Ampere
  check). `./mojo-dllm help` lists `XDG_CACHE_HOME`, which the launcher reads (#269).
- `scripts/ci-local.sh` builds `tests/gpu/test_gpu.mojo` on a machine without a GPU, as CI does, so a compile error in
  the GPU suite stops the pre-push hook (#265). markdownlint runs over `assets/*.md` and `.github/*.md` in both gates
  and CI installs it even after an earlier step failed. The `hygiene` job of `ci.yml` is now `checks`, and the secret
  scan runs on `ubuntu-24.04`. `scripts/check_hygiene.sh` also refuses private key blocks and database URLs that carry
  a password (TruffleHog's verified-only scan cannot confirm either), and stops with an error when `grep` itself fails
  instead of reading that as a clean file.
- The launcher tests run with `pytest -rs` and `uv run --locked` (#266). `tests/python/skip_policy.py` fails the run on
  a skipped test that `tests/python/allowed-skips.txt` does not name (a test id and part of the skip reason), so a test
  that stops running no longer looks like a pass. `tests/python/test_workflows.py` checks that every command `ci.yml`
  runs is in `scripts/ci-local.sh` and the reverse (the GPU suite run aside), and runs `ci-local.sh` against stub tools to
  check its exit code and summary lines. `docs/development.md` lists the four required checks.
- `scripts/check_gpu_test_count.sh` compares the number of `def test_` functions in `tests/gpu/test_*.mojo` with
  `tests/gpu/test-count.txt` in CI and in `scripts/ci-local.sh`, on every machine (#145). Before, a GPU test added
  without re-pinning, or a file that stopped defining tests, passed both gates wherever there was no GPU.
- The refusal of `run` on a `diffusion-gemma` file now points at #277, which tracks wiring the model into the generation
  loop, instead of #64, which is the forward pass (written, and so is the denoiser, #65) (#270). `run` and
  `ModelConfig.check_runnable` raise one text from one function, `gemma_run_refusal`, and `docs/cli.md`,
  `docs/introduction.md`, `docs/model-porting.md` and `docs/benchmarks.md` use the same three numbers. `docs/cli.md` no
  longer says that DiffusionGemma's `--json` reports `stopped_early` today.
- Documentation facts checked against the code (#271). The source map in `docs/architecture.md` lists every file under
  `src/` and names DiffusionGemma's loop in the sampler row. `docs/cli.md` gives the limits of `--eb-steps` (at least
  1), `--eb-t-max` (above 0), `--eb-bound` and `--eb-confidence` (at least 0). The README, the introduction and the
  architecture page list the eight tensor types the decoders accept in one order. The version line under the first CPU
  table names `bdd268b` (cpu-llada) and `dd0a647` (cpu-diffuse, diffuse-cpp rows only), which
  `scripts/bench_table.py` now renders from each result file instead of the last one read. `tests/python/test_docs_facts.py`
  checks each of these against `src/` and `bench/results`.
- Prose rules enforced (#272). `scripts/check_hygiene.sh` fails on an em or en dash in a tracked Markdown file (the
  result directories under `bench/results/` are exempt: the benchmark tools write their tables), naming the file and
  line. `docs/spec.md` had 18 and no longer does, so it joins the markdownlint run. Filler adverbs went from
  `docs/development.md` and `ci.yml`, an en dash from `docs/model-porting.md`, and a coding tool's name from the
  list of noisy processes in `docs/benchmarks.md`.
- Fixtures are checked against their generators (#230). `scripts/check_fixtures.py`, a step of both gates, runs
  `scripts/gen_fixtures.py` and `scripts/gen_moe_fixture.py` into a scratch directory (they read `FIXTURES_OUT`) and
  fails on a file that differs by one byte, a generated file that is not tracked, or a tracked fixture that no
  generator writes and that is not a tokenizer fixture, a tokenizer golden or `golden-generation.tsv`. All 239 files
  regenerate byte for byte today. The LLaDA and Dream tokenizer goldens now have the provenance file the gemma4 golden
  had (llama.cpp commit, model file and sha256, command, golden sha256), written by `scripts/gen_tokenizer_fixture.py
  --golden-only`; regenerating them from the local models gave the committed bytes. `scripts/run-tests.sh` drops an
  exported `UPDATE_GOLDEN` and says so (the golden test rewrites its file and passes against its own output), unless it
  gets `--update-golden`. `docs/development.md` lists `entropy-rows.tsv`, `utf8-cases.tsv` and
  `tiny-diffusion-gemma.*.case`, and a test fails when a fixture is missing from its table.

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
