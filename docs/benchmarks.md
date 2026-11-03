# Benchmarks

<!-- bench:start -->
**LLaDA-8B-Instruct**, Q4_K_M. 13th Gen Intel(R) Core(TM) i5-13420H, 12 threads, 33.3 GB RAM, `performance` power profile. Each prompt: 128 generated tokens, 32 denoising steps; median of 3 runs x 3 prompts.

| Runtime | ms / step | tokens / s | startup | peak RSS | language |
|---|---:|---:|---:|---:|---|
| **mojo-dllm** | 3,005 | 1.33 | 1.8 s | 5.9 GB | Mojo |
| mojo-dllm, logits for every position (ablation) | 3,268 | 1.22 | 1.8 s | 5.9 GB | Mojo |
| llama.cpp `llama-diffusion-cli` | 5,807 | 0.69 | 2.5 s | 8.4 GB | C/C++ |
| diffuse-cpp, cache off | 9,996 | 0.40 | 2.3 s | 8.2 GB | C++ |
| diffuse-cpp, cache + `entropy_exit` † | 4,380 | 1.12 | 2.6 s | 8.4 GB | C++ |

Versions: mojo-dllm `bdd268b`, Mojo 1.1.0 (8189361e), llama.cpp `836d571`, diffuse-cpp `1d2bd6a`. Evidence: `bench/results/2026-10-03-cpu-diffuse.json`, `bench/results/2026-10-03-cpu-llada.json`.

† Not the same work: the inter-step cache reuses stale K/V and `entropy_exit` can stop early. Shown because it is diffuse-cpp's recommended mode.

**LLaDA-8B-Instruct** on the GPU, Q4_K_M. NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB, driver 595.99.02), host 13th Gen Intel(R) Core(TM) i5-13420H. Each prompt: 128 generated tokens, 32 denoising steps; median of 3 runs x 3 prompts. Peak RSS is host memory.

| Runtime | ms / step | tokens / s | startup | peak RSS | language |
|---|---:|---:|---:|---:|---|
| **mojo-dllm** `--device gpu` | 221 | 18.13 | 2.2 s | 1.1 GB | Mojo |
| llama.cpp `llama-diffusion-cli`, CUDA, all layers offloaded | 243 | 16.46 | 1.4 s | 5.3 GB | C/C++, CUDA |

Step time of mojo-dllm over llama.cpp: median 0.906, range 0.896 to 0.908 across 9 paired runs, with mojo-dllm's step shorter in 9 of them. A ratio below 1 means mojo-dllm's step is shorter.

Versions: mojo-dllm `69e4024`, Mojo 1.1.0 (8189361e), llama.cpp `7df8b87`. Evidence: `bench/results/2026-11-03-gpu-llada.json`.

**Dream-v0-Instruct-7B**, Q4_K_M. 13th Gen Intel(R) Core(TM) i5-13420H, 12 threads, 33.3 GB RAM, `performance` power profile. Each prompt: 128 generated tokens, 32 denoising steps; median of 3 runs x 3 prompts.

| Runtime | ms / step | tokens / s | startup | peak RSS | language |
|---|---:|---:|---:|---:|---|
| **mojo-dllm** | 2,967 | 1.35 | 1.7 s | 5.6 GB | Mojo |
| llama.cpp `llama-diffusion-cli` | 5,566 | 0.72 | 2.5 s | 8.0 GB | C/C++ |

Versions: mojo-dllm `bdd268b`, Mojo 1.1.0 (8189361e), llama.cpp `836d571`. Evidence: `bench/results/2026-10-03-cpu-dream.json`.

**Dream-v0-Instruct-7B** on the GPU, Q4_K_M. NVIDIA GeForce RTX 3050 6GB Laptop GPU (6144 MiB, driver 595.99.02), host 13th Gen Intel(R) Core(TM) i5-13420H. Each prompt: 128 generated tokens, 32 denoising steps; median of 3 runs x 3 prompts. Peak RSS is host memory.

| Runtime | ms / step | tokens / s | startup | peak RSS | language |
|---|---:|---:|---:|---:|---|
| **mojo-dllm** `--device gpu` | 219 | 18.29 | 2.0 s | 1.2 GB | Mojo |
| llama.cpp `llama-diffusion-cli`, CUDA, all layers offloaded | 214 | 18.66 | 1.3 s | 5.0 GB | C/C++, CUDA |

Step time of mojo-dllm over llama.cpp: median 1.022, range 1.000 to 1.037 across 9 paired runs, with mojo-dllm's step shorter in 1 of them. A ratio below 1 means mojo-dllm's step is shorter.

Versions: mojo-dllm `69e4024`, Mojo 1.1.0 (8189361e), llama.cpp `7df8b87`. Evidence: `bench/results/2026-11-03-gpu-dream.json`.

**LLaDA-8B-Instruct, llama.cpp thread sweep and forward pass alone**: 13th Gen Intel(R) Core(TM) i5-13420H; median of 2 runs, one prompt, canvas 148 tokens.

| Runtime | threads | ms / step | forward pass alone |
|---|---:|---:|---:|
| mojo-dllm | 12 | 3,014 | 3,011 ms |
| llama.cpp | 4 | 7,284 | 7,098 ms |
| llama.cpp | 6 | 6,416 | 7,046 ms |
| llama.cpp | 8 | 6,140 | 6,242 ms |
| llama.cpp | 12 | 5,703 | 5,748 ms |

The ms / step column runs `llama-diffusion-cli`. The forward pass alone is mojo-dllm's measured forward pass against llama.cpp's `llama-bench` at the same token count. `llama-bench` applies a causal mask and computes logits for one position; mojo-dllm's figure includes logits for up to 32 positions and copying the canvas in. CPU busy before a run: at most 6.5%. Evidence: `bench/results/2026-10-09-fair-cpu-llada.json`.

**LLaDA-8B-Instruct on the GPU, forward pass alone**: NVIDIA GeForce RTX 3050 6GB Laptop GPU; median of 3 runs, one prompt, canvas 148 tokens.

| Runtime | ms / step | forward pass alone |
|---|---:|---:|
| mojo-dllm `--device gpu` | 225 | 225 ms |
| llama.cpp CUDA, `llama-bench -p 148` | | 243 ms |

The forward pass alone is mojo-dllm's measured forward pass against llama.cpp's `llama-bench` at the same token count. `llama-bench` applies a causal mask and computes logits for one position; mojo-dllm's figure includes logits for up to 32 positions and copying the canvas in. CPU busy before a run: at most 3.1%. Evidence: `bench/results/2026-11-03-fair-gpu-llada.json`.
<!-- bench:end -->

## What is compared

For LLaDA, all three runtimes load LLaDA-8B-Instruct Q4_K_M and denoise a
canvas of L = prompt + 128 tokens in 32 steps. For Dream, mojo-dllm and
llama.cpp (diffuse-cpp's Dream support needs its own converted file, not
benchmarked here) do the same with Dream-v0-Instruct-7B Q4_K_M; both run
Dream's timestep schedule over the whole canvas. A step is one full forward pass over the
canvas in each of them, so the step counts and shapes match.

| Runtime | How it runs | Schedule |
|---|---|---|
| mojo-dllm | `run --max-tokens 128 --steps 32 --block-length 32` | LLaDA blocks of 32, low-confidence |
| mojo-dllm (ablation) | the same plus `--full-logits` | logits for every position, as the other two compute them |
| llama.cpp | `llama-diffusion-cli -ub L --diffusion-steps 32 --diffusion-eps 0.001` | timestep schedule |
| diffuse-cpp | `diffuse-cli -n 128 -s 32 --no-cache --remasking low_confidence` | one block, cosine schedule |
| diffuse-cpp † | `diffuse-cli -n 128 -s 32 --remasking entropy_exit` | its recommended mode |

diffuse-cpp cannot read llama.cpp's GGUF files, so it loads its own
conversion of the same model (`diffuse-cpp/LLaDA-8B-Instruct-GGUF`,
`llada-8b-q4km.gguf`, 5.5 GB): F16 embeddings and a slightly different Q4_K /
Q6_K mix. It has no tokenizer and receives the token ids mojo-dllm produces.
It prints no timings either, so the harness timestamps the two lines it writes
around its generate call.

llama.cpp runs its timestep schedule because its block mode needs the prompt
length to be a multiple of the block size. The schedules commit tokens in a
different order, which changes the text but not the cost of a step.

† diffuse-cpp's recommended mode reuses K/V between steps and stops early when
the canvas has settled. It does less work, so it is not a like-for-like row.

## On the GPU

`bench/config-gpu.json` and `bench/config-dream-gpu.json` compare
`mojo-dllm --device gpu` with the same `llama-diffusion-cli` built with CUDA
and run with `-ngl 99`; the harness refuses a CUDA run unless llama.cpp
reports every layer on the GPU. Canvas, steps and prompts match the CPU runs.
These configs leave the pytest watchdog off (`"kill_pytest": false`), so each
result records how busy the CPU was before every run. Peak RSS is host memory;
neither runtime reports GPU memory. The GPU tables above come from these
configs.

The test GPU is a laptop part held to 30 W, whose clocks fall under
sustained load (`bench/checks/2026-10-12-gpu-power-under-load.txt` and
`bench/checks/2026-11-03-gpu-power-limit.txt`) and vary from day to day: the
same llama.cpp build ran LLaDA steps much slower on 2026-10-12 than on
2026-10-09 (compare `bench/results/archive/2026-10-09-gpu-llada.json` with
`bench/results/archive/2026-10-12-gpu-llada.json`). The 2026-11-03 file
records three builds run back to back at about 1.1 GHz and 30 W: absolute
milliseconds follow the clock, and ratios inside one file do not.
Both runtimes in one result file run interleaved, prompt by prompt, so
compare them within a file, not across dates.

`bench/config-fair-cpu.json` and `bench/config-fair-gpu.json` address two
questions about fairness: whether llama.cpp does better below 12 threads on a
hybrid CPU (a sweep over `-t 4, 6, 8, 12`), and how the forward passes compare
without either runtime's sampler (mojo-dllm's measured forward pass against
llama.cpp's `llama-bench` at the same token count).

## Method

`bench/run_bench.py` does the measuring:

- three prompts and, by default, three repetitions (the fairness configs use
  one prompt, and the CPU thread sweep ran two repetitions), runtimes
  interleaved within each repetition;
- every run waits until a two-second sample shows the CPU under 10% busy, and
  starts anyway after 300 seconds (60 in the fairness configs); the busy
  fraction before each run is recorded as `cpu_busy_before`, and the tables
  disclose runs that started above the threshold;
- the power profile is `performance` for the duration and restored after;
- on battery, where a laptop caps CPU and GPU power, it pauses before the
  next run until the charger is back, and a run the charger left midway is
  thrown away and repeated (`bench/power.py`; each run records
  `power_waited_s`, `power_retries`, `on_battery`, `power_source` and a snapshot
  of the supplies it read, and `--allow-battery` measures on battery instead).
  The check fails closed: a supply file it cannot read stops the run, and a
  battery with the adapter offline counts as battery whatever its status says;
- a watchdog kills any `pytest` process (other work on the machine started
  test suites mid-run), and the kills are logged in the result file;
- generation time excludes model loading, which is reported as startup;
- peak RSS comes from `/usr/bin/time`.

The table above is generated from `bench/results/*.json` by
`scripts/bench_table.py`, and CI fails if the two disagree.

## Steps and quality

A diffusion model commits `tokens / steps` tokens per step. The table above
uses 4 tokens per step, which is fast and shows in the text as repeated or
dropped words. More steps cost proportionally more time:

<!-- quality:start -->
**LLaDA-8B-Instruct.Q4_K_M.gguf**, 128 generated tokens.

| Prompt | steps | tokens / step | time | repeated words | output (first 120 characters) |
|---|---:|---:|---:|---:|---|
| Explain speculative decoding in two sentences. | 32 | 4 | 93 s | 99 | Speculative decoding is a process used in machine communication to improve the likelihood of a message being by by allow |
| Explain speculative decoding in two sentences. | 64 | 2 | 191 s | 0 | Speculative decoding is a technique used in machine translation to improve the accuracy of the translated text by allowi |
| Explain speculative decoding in two sentences. | 128 | 1 | 377 s | 0 | Speculative decoding is a technique used in machine translation to improve the accuracy of the translated text by allowi |
| Write a short Python function that checks whether a number is prime. | 32 | 4 | 97 s | 1 | def is_prime(n):     if n <= 1:         return False     if n <= 3:         return True     if n % 2 == 0 or n % 3 == 0: |
| Write a short Python function that checks whether a number is prime. | 64 | 2 | 194 s | 1 | def is_prime(n):     if n <= 1:         return False     if n <= 3:         return True     if n % 2 == 0 or n % 3 == 0: |
| Write a short Python function that checks whether a number is prime. | 128 | 1 | 388 s | 0 | Here is a short Python function that checks whether a number is prime: ``` def is_prime(n):     if n <= 1:         retur |

Evidence: `bench/quality/2026-10-03-LLaDA-8B-Instruct.Q4_K_M.json` (mojo-dllm `bdd268b`).
<!-- quality:end -->

"Repeated words" counts immediate repetitions such as "the the", a rough
proxy for parallel-decoding damage. `bench/quality.py` records the runs and
`scripts/bench_table.py` renders this table.

### Accuracy of `--threshold` and `--dual-cache`

Both options change what a step computes, so their answers can differ from
the plain sampler's. A paired GSM8K run checks how often they still get the
answer right (`bench/eval_gsm8k.py`, see [Development](development.md)):

<!-- eval:start -->
**LLaDA-8B-Instruct.Q4_K_M.gguf**, GSM8K, 200 problems, flags `--max-tokens 256 --block-length 32 --steps 256 --device gpu --json`, mojo-dllm `3b2dac4`.

| Arm | accuracy | vs baseline | 95% CI | discordant | McNemar p | passes / problem | time / problem |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 80.5% | | | | | 237.0 | 79.7 s |
| dualcache | 81.5% | +1.0 pt | -4.4 to +6.4 pt | 14 / 16 | 0.86 | 236.6 | 15.8 s |
| threshold09 | 80.0% | -0.5 pt | -2.2 to +1.2 pt | 2 / 1 | 1.00 | 73.4 | 24.1 s |

Discordant counts problems only the baseline solved / only the arm solved. Time per problem is the mean over the first 50 problems of each arm, a round that ran with nothing else on the machine.

Evidence: `bench/eval/2026-10-29-baseline-vs-dualcache.json`, `bench/eval/2026-10-29-baseline-vs-threshold09.json`, `bench/eval/2026-10-29-baseline-vs-dualcache-first50.json`, `bench/eval/2026-10-29-baseline-vs-threshold09-first50.json`.
<!-- eval:end -->

## Reading the numbers

- **ms / step** is the cost of one denoising pass over the canvas.
- **tokens / s** is 128 divided by generation time. A diffusion model's
  tokens/s depends on the step count you choose, so compare ms / step first.
- The ablation row shows what computing logits only for candidate positions
  saves.
- On the GPU, read ms / step, the step-time ratio line under each GPU table
  and peak RSS together. The ratio pairs the two runtimes run by run, and one
  GPU with one quantization (Q4_K_M) stands behind it. mojo-dllm also holds
  far less in host memory, because it uploads the weights once and releases
  the file's pages behind it.
- The thread sweep and the forward-pass-alone tables test the CPU comparison.
  If llama.cpp ran faster below 12 threads, or mojo-dllm's lead came only from
  skipping logits, those tables would show it. The sweep stops at 12 because
  the test CPU runs 12 threads.
- A note under a table counts the runs that started on a busy CPU and gives
  the medians over the quiet runs alone.
