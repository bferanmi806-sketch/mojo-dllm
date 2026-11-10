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
different order, which changes the text but not, we expect, the cost of a step.
That is an argument. The measurement that does not depend on the schedule is the
"forward pass alone" table below, which times one pass at the same token count.

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
`scripts/bench_table.py`, and CI fails if a table differs from the evidence file
it names. Two gaps are open: `--check` passes when a table marker is missing, a
target file is missing, or the quality or eval evidence is deleted ([#141](https://github.com/vladimirrott/mojo-dllm/issues/141)), and an
unknown mode exits 0 ([#142](https://github.com/vladimirrott/mojo-dllm/issues/142)).

## Benchmark environment

A timed run reports whether it had the machine to itself. `bench/isolation.py` does the checking for
`bench/run_bench.py`, `bench/eval_gsm8k.py run` and `bench/eval_engine.py run`; `bench/power.py` still owns the
mains-power check. Each check below is a constant in `bench/isolation.py`, and an arm can change one only by
passing a `Thresholds` object, which is recorded in the arm's environment file.

**Before an arm starts** (`preflight`, one 5 second sample). A FAIL stops the arm and names the offenders; `--allow-noisy`
continues and records that it did. A source that cannot be read is a WARN that names the file, never a pass.

| Check | FAIL when | Source |
|---|---|---|
| power | on battery (`--allow-battery` turns it into a WARN) | `bench/power.py` |
| profile | `powerprofilesctl get` or `/sys/firmware/acpi/platform_profile` is not `performance` | LLVM "Benchmarking tips", pyperf "Tune the system for benchmarks" |
| governor | a CPU is not on `performance`, unless it is `powersave` under `intel_pstate` active with preference `performance` | kernel `admin-guide/pm/intel_pstate` |
| CPU busy | more than 10% of all CPUs busy over the sample | `/proc/stat` |
| CPU pressure | PSI CPU `some` avg10 above 5% | kernel `accounting/psi` |
| swap movement | any page swapped in or out over the sample | `/proc/vmstat` `pswpin`, `pswpout` |
| memory | `MemAvailable` below 1.25 times the model file, or the model file's size cannot be read | `/proc/meminfo` |
| thermal | a core or package throttle counter rose over the sample | kernel `admin-guide/thermal/intel_thermal_throttle` |
| GPU processes (GPU arm) | any compute process on the GPU | `nvidia-smi --query-compute-apps` |
| GPU memory (GPU arm) | more than 300 MiB used | `nvidia-smi --query-gpu=memory.used` |
| GPU utilization (GPU arm) | mean of three samples above 5% | `nvidia-smi --query-gpu=utilization.gpu` |
| GPU clock reasons (GPU arm) | any event reason other than idle (0x1), a software power cap (0x4) and the display clock setting (0x100) | NVML clocks event reasons, `clocks_event_reasons.active` (`clocks_throttle_reasons.active` on older drivers) |

Swap in use is a WARN, because a run that touches swapped pages measures the disk. The preflight also records the
CPU governor and frequency, `intel_pstate` status and `no_turbo`, the energy performance preference, the load
average, PSI for CPU and memory, `MemAvailable`, the top CPU processes, the GPU's state (persistence mode,
compute mode, power draw, temperature, SM clock, maximum SM clock), the systemd timers due within the next hour
(names only) and the number of lines in the user crontab.

**Around every problem** (`Window`, sampled every 0.5 s). The window subtracts the CPU time of the benchmark's own
process tree, read from `/proc/PID/stat`, from the busy time in `/proc/stat`, so a benchmark that uses every core is
not contention. A problem is noisy when:

| Check | Noisy when | Per-window value recorded |
|---|---|---|
| other CPU, mean | processes outside the tree use more than 5% of the machine's CPU capacity | `other_cpu_share_mean` |
| other CPU, peak | the same share exceeds 25% in any 0.5 s sample | `other_cpu_share_max` |
| swap | any page moved | `swap_pages` |
| thermal | a throttle counter rose | `thermal_events` |
| GPU clock reasons (GPU arm) | hardware slowdown (0x8), software or hardware thermal (0x20, 0x40), hardware power brake (0x80), applications clocks (0x2) or sync boost (0x10) was active | `gpu_reasons_seen` |
| GPU processes (GPU arm) | a compute process outside the benchmark tree appeared | `gpu_foreign_pids` |
| sampler | the sampler thread died or did not stop | reason `sampler-failed` |
| unmeasured | a source that exists on this machine could not be read during the window | reason `unmeasured:SOURCE` |

Recorded in every window and never judged: PSI CPU and memory `some` stall time (`psi_cpu_stall_ms`,
`psi_memory_stall_ms`), because a 12-thread run on 12 CPUs stalls 9% of the time by itself (measured on the
reference laptop: 24 ms in 10 s idle, 908 ms with 12 busy threads), so no limit separates the run from a neighbour;
a software power cap (0x4, `gpu_power_capped`), which the laptop's 30 W GPU hits under any compute load; the display
clock setting (0x100); the SM clock minimum and maximum; the window length; and the processes outside the tree that
used the most CPU (`other_top_processes`, percent of one core).

A source that does not exist on the host (no NVIDIA GPU on a CPU arm, no `thermal_throttle` files on an AMD CPU) is
not checked.

The child that exits inside the window is handled: a tree member's ticks are its own `utime + stime` plus the
`cutime + cstime` of the children it has waited for, and the window compares the sum over the tree between two
samples. When a child exits and is reaped, its ticks move from the child to its parent and the sum does not drop.
The Window's `pid` must outlive the window (the runner's own pid does).

The GPU is read every 2 s inside a window (two `nvidia-smi` processes per read, about 0.2 s of wall time and 0.1 s of
CPU time per read, so about 5% of one core of 12 while a GPU arm runs; they are children of the runner, so they sit in
the tree and are not counted as other). A GPU event reason that starts and ends between two reads is missed.

**What the checks cannot see.** Kernel threads and interrupt time are inside `busy` and are flagged as other CPU, but
they cannot be named. Hypervisor and firmware (SMM) time, fan and embedded-controller policy and thermal paste are
invisible. GPU use by the display server is not a compute app and counts only through the 300 MiB memory limit and
the utilization sample. Other users' processes are seen, but only at the 0.5 s resolution, and a burst shorter than
that, or shorter than the 10 ms tick, can fall between samples. A foreign GPU process that starts and ends between
two GPU reads is missed. Processes in another pid namespace (containers) can differ from the pids `nvidia-smi`
prints, so a GPU arm inside a container cannot match its own pid. A window can also miss one stolen core for a part
of a long problem; the numbers are in the next paragraph.

**Sensitivity.** The limits (5% of capacity as a mean, 25% in any sample) were kept after measuring the baseline on
the reference laptop with the shared machine in its normal state: 60 windows of 10 s around a child that burns one
core, other-process share of total capacity: mean of the window means 9.0%, median 8.0%, 95th percentile 20.6%,
maximum 25.3%; peak share mean 19.5%, 95th percentile 39.7%, maximum 47.5%. Only 14 of the 60 windows came out clean at
these limits, because other sessions were using the machine; the windows named them (`claude`, `python3`, `qemu`).
Since the baseline is far above 1%, the limits are not tightened. The consequence is a blind spot: one stolen
core for 30 s of a 90 s problem is 2.8% mean and 8.3% peak of 12 CPUs, and the CPU rules do not flag it. On a quiet
machine (`scripts/bench-quiet.sh`, no other session) rerun the baseline and lower the limits by passing a `Thresholds`.

Before a problem the runner waits up to 60 seconds for a quiet moment (CPU busy, swap movement and, for a GPU arm, GPU
processes), then runs it inside a window. A noisy problem is run again, up to `--isolation-retries`
times (default 2); after that the row is kept with its reasons. `bench/run_bench.py` keeps its own `busy_threshold`
and `max_quiet_wait_s` config keys for this wait, and records `cpu_busy_before` as before.

**Reading `ARM.env.json`.** The eval runners write `bench/eval/runs/ARM.env.json` and `run_bench.py` writes
`bench/results/env/NAME.env.json`. It holds one entry per start of the arm under `sessions`: the preflight report
(each check with its status and detail), `snapshot_start` and `snapshot_end`, the `thresholds` used, `tuned` (the
profile before and after `--tune`), `bench_quiet` (true when the run was started under `scripts/bench-quiet.sh`),
`preflight_failed_allowed` (true when `--allow-noisy` continued past a FAIL), a `session_id`, the retry limit and tool
versions. Each row of an arm file carries `isolation`: the window values above, `clean`, `reasons`, `retries` (how
many repeats this problem took), `noisy`, `cpu_busy_before`, `waited_s`, `session` (the `session_id` of the start
that wrote the row), `allow_noisy` and `preflight_failed_allowed`.

`eval_gsm8k.py compare` adds `generate_s_mean_clean` (the mean over the problems whose window was clean in both
arms, `null` when there is none), `clean_rows`, `noisy_rows` per arm and `unchecked_rows` per arm (rows written
before windows existed). Accuracy is still computed over every row.

**Quieting the machine.** `--tune` sets the power profile to `performance` through `powerprofilesctl`, reads it back,
and restores the previous profile on exit, on an exception, on Ctrl-C and on SIGTERM. `sudo scripts/bench-quiet.sh --
COMMAND` additionally stops the system timers `sysstat-collect`, `apt-daily`, `apt-daily-upgrade`, `fwupd-refresh`,
`anacron`, `man-db` and `motd-news`, turns on NVIDIA persistence mode and locks the GPU clocks where the driver
allows it, runs the command, and puts everything back; it never edits cron files. After a SIGKILL or a power loss no
trap can run: `sudo scripts/bench-quiet.sh --restore` undoes the script's changes from its state file, and `--tune`
(which keeps no state file) leaves the profile at `performance`, so run `powerprofilesctl set PREVIOUS` with the
profile in `tuned.previous` of the arm's environment file. The one-minute user cron job is not
stopped: a window that sees it is noisy and the problem is repeated. Not covered: kernel boot parameters (`isolcpus`,
`nohz_full`) and BIOS settings. The rows keep the 12 threads they were measured with, so no core is left to pin.

**Rows measured before this change.** The published rows above (and the arms under `bench/eval/` written before
`isolation` existed) were not checked this way. They have `cpu_busy_before` for the CPU rows and no window. Do
not read a ratio from them closer than their recorded spread.

## Steps and quality

A diffusion model commits `tokens / steps` tokens per step. The table above
uses 4 tokens per step, which is fast and shows in the text as repeated or
dropped words. More steps cost proportionally more time, so the headline
tokens / s above is not a quality-matched figure. In the first prompt below,
the repeated words disappear at 64 steps, which takes about twice the time of
32 steps, so by that count mojo-dllm's quality-matched speed on that prompt is
about half the headline rate. Other runtimes use other schedules. The code prompt shows no such damage at 32 steps.

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

### The authors' own code (Transformers arm)

`bench/eval_engine.py run --engine hf` answers the same 200 problems through the sampling code the model authors
publish: `generate` from `generate.py` in a checkout of [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) for
LLaDA, and `diffusion_generate` from the remote code of
[Dream-org/Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) for Dream. The adapter
imports the LLaDA file from your checkout and never copies it (the repository has no license file).

**This arm runs different weights.** The llama.cpp and mojo-dllm arms load the Q4_K_M GGUF. The Transformers arm
loads the authors' bf16 safetensors, or a bitsandbytes 4-bit (nf4) copy of them with `--quant 4bit`. An accuracy
difference between the arms therefore includes quantization as well as any difference between the
implementations, and `eval_gsm8k.py compare` refuses such a pair unless you pass `--across-weights`. The output
then holds both model names and `"weights_differ": true`.

Like-for-like settings: temperature 0, steps equal to the generation length, block length 32 and low-confidence
remasking for LLaDA, entropy remasking for Dream (which has no blocks). The Dream README example uses temperature
0.2, top_p 0.95 and 512 steps, and the library default algorithm is `origin`; the arm departs from that, and the
`flags` of every row say so. `version` records the model repository's commit sha (`--model-revision`, a full
40-character sha; `null` when you give none, and then the weights and the remote code are unpinned), the LLaDA
checkout's commit, and the `transformers`, `torch` and `bitsandbytes` versions. The Python sampling loop is not a
speed baseline; read accuracy from this arm, not seconds.

The two READMEs pin different environments (LLaDA: `transformers==4.38.2`; Dream: `transformers==4.46.2` and
`torch==2.5.1`), so each model gets its own virtual environment. The repository's frozen environment never
installs them. The runner needs only the standard library besides the model libraries, so run it with the
environment's own interpreter. The LLaDA README pins no `torch`; I pin 2.5.1 as for Dream, because an unpinned
install resolves a CUDA 13 build that may not match the laptop's driver. Both lines also pin `numpy<2`; that numpy 1.x
works with these `transformers` versions is an assumption I did not test. I checked each install line below with `uv pip install --dry-run` only (it resolves: `torch==2.5.1` with the
CUDA 12.4 libraries, `numpy==1.26.4`, and `bitsandbytes==0.50.2` for the GPU lines); nothing was installed and
nothing was run.

```bash
# LLaDA, GPU (4-bit). For the CPU path replace the install line with:
#   uv pip install --python "$V/bin/python" --extra-index-url https://download.pytorch.org/whl/cpu \
#       --index-strategy unsafe-best-match "torch==2.5.1+cpu" transformers==4.38.2 accelerate "numpy<2"
V=~/.cache/mojo-dllm-hf/llada
uv venv $V --python 3.12
uv pip install --python "$V/bin/python" torch==2.5.1 transformers==4.38.2 accelerate bitsandbytes "numpy<2"
git clone https://github.com/ML-GSAI/LLaDA ~/src/LLaDA
REV=$(curl -s https://huggingface.co/api/models/GSAI-ML/LLaDA-8B-Instruct | jq -r .sha)
scripts/capped.sh --mem 26G -- "$V/bin/python" bench/eval_engine.py run --engine hf --model GSAI-ML/LLaDA-8B-Instruct \
    --model-revision "$REV" --llada-repo ~/src/LLaDA --quant 4bit --device gpu --arm hf-llada-4bit --limit 50

# Dream, GPU (4-bit); the CPU line is the same with the CPU install line above and transformers==4.46.2
V=~/.cache/mojo-dllm-hf/dream
uv venv $V --python 3.12
uv pip install --python "$V/bin/python" torch==2.5.1 transformers==4.46.2 accelerate bitsandbytes "numpy<2"
REV=$(curl -s https://huggingface.co/api/models/Dream-org/Dream-v0-Instruct-7B | jq -r .sha)
scripts/capped.sh --mem 26G -- "$V/bin/python" bench/eval_engine.py run --engine hf --model Dream-org/Dream-v0-Instruct-7B \
    --model-revision "$REV" --quant 4bit --device gpu --arm hf-dream-4bit --limit 50

python3 bench/eval_gsm8k.py compare baseline hf-llada-4bit --across-weights
```

**The repository's code runs on your machine.** The adapter loads with `trust_remote_code=True`, which executes
the Python files of the model repository (for Dream, `modeling_dream.py` and `generation_utils.py`) at load. Only a
commit sha says which code that is, so `--model-revision` is required for a Hugging Face repository, and the run
refuses without it unless you pass `--allow-unpinned` (the rows then say `unpinned`). Look up the sha when you start
the arm and keep it: the `curl` lines above print the current one. A local directory is not on the hub, so it needs
no revision; its path is recorded and the sha is marked unverified, because nothing ties the directory to a
commit. `version` also records the LLaDA checkout's commit with `-dirty` when a tracked file differs from it.

**Which arms count as different weights.** `compare --across-weights` reads each arm's `model`. A 4-bit arm records
`LLaDA-8B-Instruct|4bit`, so it differs from a bf16 arm of the same repository. Two revisions of one repository
record the same `model`; they differ in `version.model_revision`, which `compare` does not read.

**Memory.** Two figures, both estimates and not measurements. Before it loads, the adapter reads `MemAvailable`
from `/proc/meminfo` and refuses to start the load when it is below the load need: bf16 is 1.25 times the weights
plus 3 GB of working memory (LLaDA's float64 Gumbel noise on the CPU is about 0.5 GB per tensor), 4-bit is the
weights over 3.5 plus 3 GB. After the load the isolation preflight asks for the 3 GB of working memory only,
because the weights are resident by then and `MemAvailable` already excludes them. The weights' size comes from a
table for the two repositories, else from the `*.safetensors` of a local directory; for any other model the size is
unknown and the run refuses before loading, as it does when `/proc/meminfo` cannot be read. The guard does not
replace a cap: run the commands above under `scripts/capped.sh` (see [Development](development.md)). The loader
also passes `low_cpu_mem_usage=True`.

The weights are 14.93 GiB (LLaDA) and 14.19 GiB (Dream) of safetensors, so one model fits a nearly full disk and
both do not. The adapter loads the model before the preflight and the first problem, and refuses a model that is
not on the requested device (a model that fell back to the CPU would still answer, slowly).

**Not done yet.** The generation timeout leaves the generation thread running until the stopped process exits. The tests run against stubs: no weights were downloaded and no GPU was used for this arm. The
memory-fit spike (issue #178, scope item 1: peak RAM, peak VRAM and seconds per step for bf16 on the CPU and for
4-bit on the GPU) and the GSM8K arm itself (scope item 4) have not run, because the laptop had 4 GB of disk free
and has to stay quiet for the timed benchmark runs. Whether 4-bit LLaDA fits 6 GB of VRAM is unmeasured, and so
is the CPU step time. There are no Transformers rows in this document until the arm is committed under
`bench/eval/`.

### A running vLLM server (DiffusionGemma)

`bench/eval_engine.py run --engine vllm` answers the same 200 problems through the OpenAI-compatible endpoint of a vLLM
server that is already running. vLLM serves one diffusion model, DiffusionGemma (issue #179). The engine is a client:
it never starts or stops a server, loads no weights and installs nothing. Rent the GPU, start the server there,
point the client at it, and destroy the instance after. The shared client is `bench/engines/_http.py` (standard
library only); the SGLang engine reuses it.

What the client sends: `POST /v1/chat/completions` with one user message holding the prompt text every engine uses,
`temperature` 0 and `max_tokens` equal to `--gen-length`. The server applies the model's chat template. The client
sends no credentials and no cookies, follows no redirects, never retries, ignores `http_proxy` and its siblings unless
you pass `--use-env-proxy`, refuses a response over 4 MiB, and gives each request one deadline (`--timeout-s`) that
covers connect, headers and body. Text the server sends back (an error message, a model id) is stripped of control
characters and capped at 400 characters before it reaches your terminal, so a server cannot write escape sequences
into it.

**This arm runs different weights.** The recipe's FP8 checkpoint is not the Q4_K_M GGUF of the llama.cpp arm. The
server does not report its quantization, so `--quantization` is required and goes into the arm's `model` (for example
`diffusiongemma-26B-A4B-it|fp8-dynamic`). `eval_gsm8k.py compare` then refuses the pair unless you pass
`--across-weights`, and an accuracy or speed difference includes the quantization.

**What a row records.** `version` is JSON with the release `GET /version` reports (`{"version": ...}`), and the
checkpoint, quantization and attention backend. Only the release comes from the server. `--quantization` and
`--attention-backend` are declared, and `flags.declared_not_verified` lists them. `--checkpoint` is compared with the
`root` that `GET /v1/models` reports for the served name, and a difference is refused; with no `--checkpoint`, the
reported `root` is the checkpoint. `--model` is the arm's label, and it must equal that `root` or the served name:
a label for other weights is refused, so pass the checkpoint you served (for the FP8 run,
`RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`), not the base model's name. `flags.model_matches_server` records it. The chat template is the server's: `flags.chat_template` says
`applied_by: server`, `verified: false`, and keeps what you declare with `--chat-template`. The run is refused
before the first problem when the server does not list `--served-model`, when `/version` answers no version, and
for an HTTP error, an empty completion, a response that names another model, or a response without token counts.
The server's address is in neither `version` nor `flags`.

**Steps.** The response carries no denoising steps. vLLM's source defines the counter
`vllm:diffusion_num_denoising_steps` for diffusion models (`vllm/v1/spec_decode/metrics.py`, read 2026-11-09). The
source has no `_total`; prometheus_client adds that suffix when it exposes a counter, so `/metrics` should show
`vllm:diffusion_num_denoising_steps_total`. **That `_total` name is inferred from prometheus_client, not seen on a
live `/metrics`**, and the engine accepts both names. The counter is server-wide and sums every request the server
runs. The engine reads it before and after each generation and records the change, which is this request's steps only
when nothing else runs on the server. So the run is **refused** when the server exposes the counter and you have
not declared `--max-num-seqs 1`, the value you started the server with (the client cannot read it; the recipe's own
command sets 4). Run one client at a time even then. When `/metrics` or the counter is missing, `forward_passes` is
null and `flags.forward_passes_measured` is false, and `--max-num-seqs` is not needed; the engine never fills in a
number. Any other error from `/metrics` (403, 500) stops the run: only a 404 means "no such endpoint". A counter that
does not move across a request also stops it. **I have not checked this counter against a running server.** Steps per
canvas vary from request to request, because the recipe describes an entropy-bound sampler; the recipe guide's "up to
48 steps per block" is the figure it prints, not an expected value for this server, so read the first rows for their
spread and do not check them against a constant. `--steps`, `--block-length` and `--schedule` do not reach the
server: the step rule is set when the server starts (`--diffusion-config`), and a `--steps` different from
`--gen-length` is refused.

**The isolation checks see the client only.** `bench/isolation.py` runs on the machine that runs the client. It does
not observe the server's GPU, clocks, power or neighbours, so a clean preflight says nothing about the rented card.
Every row says `"server": "remote"` in `flags`. `--device cpu|gpu` selects which local checks apply; run the client
with `--device cpu` on a machine without a GPU. `generate_s` is the wall time of one request, which includes the
network round trip and the server's queue.

#### Runbook: rent, serve, run, pull, destroy

**TODO for the maintainer: the budget.** The issue asks for the price and the hours in a pull request before any
renting. They are not in this document. Name the card, the hourly price from the offer you pick, the hours and the
total cost in the pull request that adds the arm.

**Which card.** The recipe lists FP8 as verified on H100, H200 and DGX Spark GB10, with a 32 GB minimum
(`recipes/models/Google/diffusiongemma-26B-A4B-it.yaml` in vllm-project/recipes, read 2026-11-09), and vLLM 0.24.0 or
newer. An RTX A6000 is not a verified target. The FP8-dynamic checkpoint is 25.33 GiB (Hugging Face API, 2026-11-09).

1. **One time on your machine.** Install the Vast.ai CLI, read your API key into an environment variable and create a
   throwaway SSH key. The key prompt reads without echo and the key never appears in a command line; cancel with
   Ctrl-C and nothing is written.

   ```bash
   uv tool install vastai
   ssh-keygen -t ed25519 -N '' -C vast-dgemma-throwaway -f ~/.ssh/vast_dgemma
   echo "== Vast.ai API key (console.vast.ai, Account, API Keys) =="
   read -rs -p "key: " VAST_API_KEY; echo; export VAST_API_KEY
   vastai create ssh-key "$(cat ~/.ssh/vast_dgemma.pub)"
   ```

   Run every later `vastai` command in this same shell, and close it when you are done: the key lives in that shell's
   environment, and every program you start from it inherits the variable. This runbook does not use
   `vastai set api-key KEY` or `--api-key KEY`, because both put the key in the command's argument list, which every user
   of the machine can read through `/proc/PID/cmdline` (mode 0444 on this machine, checked with
   `ls -l /proc/self/cmdline`) while the command runs. A process's environment file, `/proc/PID/environ`, is mode 0400
   here, readable by its owner and root. The CLI reads `VAST_API_KEY` when no `--api-key` is given: I read the default of
   `--api-key` in `vast.py` of vast-ai/vast-cli at commit `4727235e75470ab26b231c1bdb1415f6ca815d1d`
   (`os.getenv("VAST_API_KEY", ...)`), and the Context7 documentation of `/vast-ai/vast-cli` gives the same order
   (explicit argument, then `VAST_API_KEY`, then the key file), both on 2026-11-09. I have not run the CLI, and I did not
   check which version `uv tool install vastai` installs. If your version ignores the variable, the first call fails
   with an authentication error and nothing is stored.

   This touches `~/.ssh/vast_dgemma*` and your account's key list (the CLI also creates `~/.config/vastai`). To undo it,
   close the shell, delete `~/.ssh/vast_dgemma*` and remove the SSH key in the Vast.ai console, and revoke the API key
   there if you made it for this run. I have not run `vastai create ssh-key` here; check it with
   `vastai create ssh-key --help`.

2. **Find an offer.** Filter by card, reliability, disk, download speed and driver. The field names come from the
   Vast.ai CLI reference (`docs.vast.ai/cli/reference/search-offers`); fill in the card name from its listing.

   ```bash
   vastai search offers 'gpu_name=<CARD> num_gpus=1 reliability>0.98 disk_space>=100 inet_down>500 rented=False' -o 'dph'
   ```

   Pick by the total, not the hourly price: the hourly price times your hours, plus the download (`inet_down_cost`
   per GB for about 26 GB of weights and the image). Add `--raw` to see the cost fields.

3. **Rent it.** The recipe's image is `vllm/vllm-openai:gemma`. Billing starts when the instance runs.

   ```bash
   vastai create instance OFFER_ID --image vllm/vllm-openai:gemma --disk 100 --ssh --direct --label dgemma-vllm
   vastai show instances          # wait for "running"; note the instance ID
   vastai ssh-url INSTANCE_ID     # prints ssh://root@HOST:PORT
   ```

   `--disk 100` is my estimate (25.33 GiB of weights, the image and the cache), not a measurement. I have not run
   these lines on Vast.ai, and I do not know whether its `--ssh` mode keeps the image's `vllm` entry point out of
   the way. Run `vastai create instance --help` first; if the server starts by itself, read the instance's logs
   before you start a second one.

4. **Serve.** On the instance (`ssh -p PORT -i ~/.ssh/vast_dgemma root@HOST`), start the server with the command from
   the recipe page. The page `https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it` was read on 2026-11-09. It
   prints this `vllm serve` command for its default configuration, verbatim:

   ```
   vllm serve google/diffusiongemma-26B-A4B-it \
     --max-num-seqs 4 \
     --tensor-parallel-size 1 \
     --enable-auto-tool-choice \
     --tool-call-parser gemma4 \
     --chat-template examples/tool_chat_template_gemma4.jinja \
     --reasoning-parser gemma4
   ```

   and this `docker run` for "Single GPU (H100/H200, BF16)", verbatim:

   ```
   docker run -itd --name diffusiongemma \
       --ipc=host --network host --gpus all \
       -v ~/.cache/huggingface:/root/.cache/huggingface \
       vllm/vllm-openai:gemma \
           --model google/diffusiongemma-26B-A4B-it \
           --max-model-len 262144 \
           --max-num-seqs 4 \
           --gpu-memory-utilization 0.85 \
           --host 0.0.0.0 --port 8000
   ```

   The page prints no command for the FP8 checkpoint. Its deployment-notes table also lists
   `--generation-config vllm` (the checkpoint's `generation_config.json` sets `max_tokens` 256 and would override the
   per-request limit), `--gpu-memory-utilization 0.85` and `--diffusion-config '{"canvas_length": 256}'` as flags
   DiffusionGemma needs, and neither command above has them all. Which exact command to run for this arm is the
   maintainer's decision. The FP8 recipe variant names the model `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`;
   using that id in place of `google/diffusiongemma-26B-A4B-it` is my substitution, not the page's, and is not
   verified on hardware. To record steps the server must also run with `--max-num-seqs 1` (the recipe's value is 4;
   see Steps above). Write the command you ran, as run, into the pull request. Without Docker on the instance
   (Vast.ai runs the image as the container), take the arguments after the image name and run them as
   `vllm serve MODEL ...`. If you pass `--served-model-name`, give that name to `--served-model` below.
   **The `docker run` block binds `--host 0.0.0.0`**, and a vLLM server started without `--api-key` has no
   authentication. On a rented host that port may be reachable by anyone. Replace it with `--host 127.0.0.1` (the
   tunnel in step 5 reaches it) or close the port with the host's firewall, and never leave an unauthenticated server
   exposed. **The recipe's `--reasoning-parser gemma4` and the Gemma 4 chat template can put the answer in the
   response's `reasoning_content` and leave `message.content` empty.** The engine then stops with `the server
   returned an empty completion`. To see it, `curl` one chat request by hand against the tunnel and look at
   `choices[0].message`. What to change on the server is a question for the recipe: check the recipe page for how
   it turns thinking on and off, and do not guess a flag. The `--generation-config vllm` line in the page's
   deployment notes is mentioned above: the checkpoint's `generation_config.json` sets `max_tokens` 256, which would
   override the per-request limit.

5. **Run the arm from your machine.** Open the tunnel in a second terminal, check that the server answers, then run.
   Results are written on your machine, to `bench/eval/runs/ARM.jsonl`, so nothing needs pulling off the instance.
   The client holds no key, and the instance never receives one.

   ```bash
   ssh -N -L 8000:127.0.0.1:8000 -p PORT -i ~/.ssh/vast_dgemma root@HOST      # terminal 2; Ctrl-C closes it
   curl -s http://127.0.0.1:8000/version                                      # {"version":"0.24.0"} or newer
   curl -s http://127.0.0.1:8000/v1/models                                    # note the id and root
   python3 bench/eval_engine.py run --engine vllm --model RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic \
       --arm vllm-dgemma-fp8 --device cpu --gen-length 256 --max-num-seqs 1 \
       --base-url http://127.0.0.1:8000 --served-model SERVED_NAME \
       --quantization fp8-dynamic --checkpoint CHECKPOINT_AS_SERVED --attention-backend ATTENTION_BACKEND_AS_SERVED
   ```

   Add `--limit 3` first to look at the rows. Run one client at a time: the step counter is server-wide. After the
   run, `ss -ltnp 'sport = :8000'` must show no tunnel of yours.

6. **Compare.** `python3 bench/eval_gsm8k.py compare MOJO_OR_LLAMACPP_ARM vllm-dgemma-fp8 --across-weights`.

7. **Destroy the instance.** A stopped instance keeps billing for its disk.

   ```bash
   vastai destroy instance INSTANCE_ID
   vastai show instances          # must list nothing
   ```

   Then read the billing page in the console: the final charge should be within the total you computed in step 2.
   Delete the throwaway SSH key from the account, and remove `~/.ssh/vast_dgemma*`.

| Symptom | Likely cause | Action |
|---|---|---|
| the run refuses with `the server reports the models ...` | `--served-model` differs from `--served-model-name` | pass the id `/v1/models` lists |
| `--checkpoint is ... but the server reports root ...` | the label does not match what was served | fix the label, or the command |
| `forward_passes` is null | the server exposes no `vllm:diffusion_num_denoising_steps` counter | none: the column stays null |
| the run refuses with `--max-num-seqs 1` | the server exposes the counter and you did not declare 1 | restart the server with `--max-num-seqs 1`, declare it |
| the run stops on a counter that did not move | another client, or `/metrics` is not this model's | stop other clients; read `/metrics` by hand |
| `--model ... is neither the root ... nor the id` | the label names other weights than the server runs | pass the served checkpoint as `--model` |
| `returned an empty completion` | the answer is in `reasoning_content` | `curl` one request; check the recipe |
| `answered HTTP 4xx` | for example `max_tokens` over the server's limit | the message is the server's own |
| out of memory on the server | `--max-num-seqs` above 4 | the recipe sets 4 because diffusion state buffers are large (1 uses less) |

**Not done yet.** No server was run for this engine: the tests use a fake server on 127.0.0.1, and no vLLM, GPU or
weights were involved. The fields above were read from vLLM's source and the recipe page, not observed on a live
server (`/version`, `/v1/models`, `/metrics`, the counter's per-request meaning). The evaluation (issue #179, scope
item 6: batch size 1, the same 200 problems, a 256-token canvas, on a rented card) has not run, so there are no vLLM
rows in this document. If the harness's number differs from vLLM's published figures by more than 25 percent on the
same card class, explain it here (prompt length, canvas, checkpoint) when the arm lands.

### A running SGLang server (DiffusionGemma)

`bench/eval_engine.py run --engine sglang` answers the same 200 problems through the OpenAI-compatible endpoint of an
SGLang server that is already running (issue #180). SGLang serves DiffusionGemma with its dLLM framework and the
`Gemma4Renoise` algorithm, and its cookbook says the model is "Not benchmarked for speed". The engine is a client in
the same way as the vLLM one: it never starts or stops a server, loads no weights and installs nothing. It uses
the shared `bench/engines/_http.py` (no credentials, no cookies, no redirects, no retries, one deadline per request,
a 4 MiB bound, server text stripped of control characters and capped at 400 characters; see the vLLM section above)
and `bench/engines/_openai.py` (the model-list and completion checks the two engines share).

Every field name below comes from SGLang's source, read on 2026-11-09 at `main` commit
`dfaca5681ddc20e31d50f2dfe5dfada269f12bbd`, plus the cookbook page
`https://docs.sglang.io/cookbook/autoregressive/Google/DiffusionGemma.md` read the same day. The routes `/v1/models`,
`/model_info` and `/server_info` and the `Gemma4Renoise` algorithm were also compared with the release tag `v0.5.23`
(2026-11-02) and give the same field names and the same request rules; the other files were read at `main` only. The issue text and
the source differ in two places, and the engine follows the source: SGLang has no `GET /version` (the version is a
field of `GET /server_info`), and the issue's pull request #23722 for sequence-level forward counts was closed
without a merge. **No field was seen on a live server.**

**What the client sends.** `POST /v1/chat/completions` with four fields: `model`, one user message holding the prompt
text every engine uses, `max_completion_tokens` equal to `--gen-length`, and `stream` false (the cookbook says
streaming is block-level, one denoised canvas per chunk, and the tokenizer manager's `_wait_one_response` yields an
unfinished output only for a streaming request). `Gemma4Renoise.validate_request` returns an error for a request that
carries logprobs, a grammar (`json_schema`, `regex`, `ebnf`, `structural_tag`), a penalty, `min_new_tokens`, `min_p`, a
seed, a logit bias, a custom logit processor or a sampling mask, and the cookbook says such a request is rejected with
a 400. The engine sends none of them, and a test scans every request body for them.
`temperature`, `top_p` and `top_k` are accepted and have no effect, because the renoise schedule governs sampling;
the engine sends none of them either, and `flags.sampling` says so. A rejected request stops the run with the
server's own message. The server applies the chat template; `flags.chat_template` says `applied_by: server`,
`verified: false`, and holds the `chat_template` that `GET /server_info` reports (null means the checkpoint's own
template) and what you declare with `--chat-template`.

SGLang's own measurement, `benchmark/dllm/bench_diffusion_gemma.py`, sends pretokenized prompts to `/v1/completions`
with EOS stopping off and a fixed output length, and its README says the token rate it reports includes canvas
positions after EOS. This engine sends chat messages and lets the model stop, and it times a whole answer. The two
numbers are not the same measurement and should not be compared.

**This arm runs bf16 weights.** `Gemma4Renoise.configure_server_args` refuses a quantized checkpoint ("DiffusionGemma
does not support quantized loading"), and the engine refuses a server whose `/server_info` reports a `quantization`.
The arm's `model` is `<name>|<dtype>`, for example `diffusiongemma-26B-A4B-it|bfloat16`, so `eval_gsm8k.py compare`
refuses the pair with a Q4_K_M GGUF arm or the vLLM FP8 arm unless you pass `--across-weights`, and a difference in
accuracy or speed includes the precision.

**What a row records, and where each value comes from.** `version` is JSON with the SGLang release, the diffusion
algorithm, the checkpoint and the dtype.

| Value | Read from | If the server cannot say |
|---|---|---|
| release (`version`) | `GET /server_info`, field `version` | the run is refused; so is an empty string and the development fallback `0.0.0.dev0`, which names no release or commit |
| checkpoint | `GET /model_info`, field `model_path` | `--checkpoint` is recorded as declared and the label stays unconfirmed |
| algorithm | `GET /server_info`, field `dllm_algorithm` | `--algorithm` is needed; a null field means the server runs no diffusion model and the run is refused |
| dtype | `GET /server_info`, field `dtype` | `auto` and a missing field mean "cannot say": `--dtype` is needed (`half` and `float16` are one dtype, as are `float` and `float32`) |
| batch size | `GET /server_info`, field `max_running_requests` | null means "cannot say": `--max-running-requests 1` is needed |

A value the server reports is the value. A declared value that differs from a reported one is refused, and a declared
value the server could not confirm is listed in `flags.declared_not_verified`. `--model` must equal the reported
`model_path` or the served name (SGLang's `/v1/models` puts the served name in both `id` and `root`, so unlike vLLM
`root` cannot confirm the checkpoint). `flags.model_matches_server` is true when `--model` is the reported `model_path`,
false when it matches only the served name (an alias, so the label may name other weights than the checkpoint recorded
in `flags.checkpoint_reported_by_server`), and null when the server reports no `model_path`. The server's own text is never
copied beyond these named fields: `/server_info` also holds a `launch_command` and the scheduler's internal state, and
neither reaches an arm file. `flags.algorithm_config_set` says whether the server was started with a
`--dllm-algorithm-config` file (its content is not readable from the client). The server's address is in neither
`version` nor `flags`.

**Batch size 1 is required.** `generate_s` is the wall time of one request, and a second client would add its queue time
to it, so the run is refused unless the batch size is 1, reported by the server or declared with
`--max-running-requests 1`. SGLang's own LLaDA 2.1 cookbook serves with `--max-running-requests 1`
(`https://docs.sglang.io/cookbook/autoregressive/InclusionAI/LLaDA-2.1.md`, read 2026-11-09). Run one client at a time.

**Steps are not recorded.** SGLang exposes no per-request step or forward-pass count on `main` or in `v0.5.23`. Its
Prometheus metrics (`python/sglang/srt/observability/metrics_collector.py`) have no dLLM counter, and the only pass
counter, `sglang:cuda_graph_passes_total`, is server-wide and counts passes by CUDA-graph mode, so a change of it across
one request is not that request's denoising steps. Pull request sgl-project/sglang#48675 ("Expose cumulative
request-level forward metrics", which would add `meta_info.dllm_metrics` to `/generate`) was open and unmerged when I
read it on 2026-11-09, and its description opens with "dLLM responses do not report how many model forwards a request
uses". SGLang's own benchmark records `--denoising-steps` as metadata only. I read the metrics collector at both `main`
and `v0.5.23`, and neither has a dLLM metric. So `forward_passes` is
null in every row, `flags.forward_passes_measured` is false, and the engine never reads `/metrics` and never infers a
count. `--steps`, `--block-length` and `--schedule` do not reach the server (`--dllm-algorithm-config` sets the step
rule when the server starts), and a `--steps` different from `--gen-length` is refused.

**The isolation checks see the client only.** `bench/isolation.py` runs on the machine that runs the client, and
every row says `"server": "remote"` in `flags`. Run the client with `--device cpu` on a machine without a GPU.

#### Runbook: serve, run, destroy

**TODO for the maintainer: the budget.** The issue asks for the price and the hours in a pull request before any
renting. They are not in this document, and I name no card and no price. State the card (or cards), the hourly price
from the offer you pick, the hours and the total in the pull request that adds the arm.

**Which hardware.** The bf16 checkpoint `google/diffusiongemma-26B-A4B-it` has 51 647 701 024 bytes of `.safetensors`
files, 48.1 GiB (Hugging Face API, revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`, read 2026-11-09). That is more
than a 48 GB card holds before the cache is allocated, so a 48 GB card cannot run this arm alone; the issue's reading
is two cards or tensor parallelism. The cookbook prints no memory or parallelism figure and no command with tensor
parallelism. `ServerArgs` has `--tp-size` (alias `--tensor-parallel-size`), and I have not run it with this model.
Whether one 80 GB card holds the weights and the cache is not settled by any page I read. Settle the hardware with a
short trial of a few problems before the full run. SGLang's roadmap issue #39499 lists quantized checkpoints
(FP8 first) for DiffusionGemma as an unchecked item, and the algorithm refuses quantized loading today, so plan for
bf16.

1. **Rent the machine.** Steps 1 to 3 of the vLLM runbook above apply (Vast.ai CLI, key, offer, instance), with these
   changes. The install guide (`https://docs.sglang.io/docs/get-started/install.md`, read 2026-11-09) publishes the
   Docker image `lmsysorg/sglang` on Docker Hub, says `latest` is the newest stable release, and says to use a
   specific release or dated nightly tag for a reproducible deployment. I did not check which tag matches `v0.5.23`,
   and I have not run the image on Vast.ai. Size the disk from the 48.1 GiB of weights plus the image; I have no
   measurement of the image. Billing starts when the instance runs.

2. **Serve.** On the instance, start the server with the command from the cookbook. The page
   `https://docs.sglang.io/cookbook/autoregressive/Google/DiffusionGemma.md` was read on 2026-11-09. Section 3.1 prints
   this command, verbatim (the issue quotes the same flags on one line):

   ```bash
   sglang serve \
     --model-path google/diffusiongemma-26B-A4B-it \
     --dllm-algorithm Gemma4Renoise \
     --trust-remote-code \
     --host 0.0.0.0 \
     --port 30000
   ```

   I compared this block with `curl -s` of the page's markdown source on 2026-11-09 and the bytes are identical.
   The page says the Triton attention backend, eager mode and unchunked prefill are selected automatically for
   `Gemma4Renoise`, so none is passed. Three changes to it are mine and are not on the page. Add
   `--max-running-requests 1`, so the batch size is 1 and the server reports it. Replace `--host 0.0.0.0` with
   `--host 127.0.0.1`: **the page's command binds every interface, and a server started without `--api-key` has no
   authentication, so on a rented host that port may be reachable by anyone**; the tunnel in step 3 reaches a loopback
   bind. Do not add `--api-key`, because the client sends no credential and every request would stop with HTTP 401.
   Optionally add `--dtype bfloat16`: without it `/server_info` reports `auto`, and you declare `--dtype bfloat16` in
   step 4 (the argument's help text says `auto` uses BF16 for a BF16 model), which the row then lists as declared and
   not verified. Write the command you ran, as run, into the pull request.

   Leave `--dllm-algorithm-config` unset for the reference arm, so the checkpoint's defaults apply (the cookbook
   lists 48 denoising steps, an entropy bound of 0.1 and temperatures from 0.4 to 0.8). The cookbook's example
   config marks `seed` as optional and as what makes the sampling reproducible, so without one two runs of the same
   problem can answer differently. The page's command has no `--reasoning-parser`; if you add one, the answer may move
   to `reasoning_content` and the engine then stops with `the server returned an empty completion`.

3. **Run the arm from your machine.** Open the tunnel in a second terminal and look at what the server reports.
   Results are written on your machine to `bench/eval/runs/ARM.jsonl`, and the instance never receives a key.

   ```bash
   ssh -N -L 30000:127.0.0.1:30000 -p PORT -i ~/.ssh/vast_dgemma root@HOST    # terminal 2; Ctrl-C closes it
   curl -s http://127.0.0.1:30000/v1/models                                   # note the id
   curl -s http://127.0.0.1:30000/model_info | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("model_path"), d.get("served_model_name"))'
   curl -s http://127.0.0.1:30000/server_info | python3 -c 'import json,sys; d=json.load(sys.stdin); print({k: d.get(k) for k in ("version", "dllm_algorithm", "dtype", "quantization", "max_running_requests", "chat_template")})'
   ```

4. **Run.** `--model` is the checkpoint you served.

   ```bash
   python3 bench/eval_engine.py run --engine sglang --model google/diffusiongemma-26B-A4B-it \
       --arm sglang-dgemma-bf16 --device cpu --gen-length 256 \
       --base-url http://127.0.0.1:30000 --dtype bfloat16
   ```

   Add `--limit 3` first to look at the rows. Add `--served-model NAME` when the server uses `--served-model-name`.
   After the run, `ss -ltnp 'sport = :30000'` must show no tunnel of yours.

5. **Compare.** `python3 bench/eval_gsm8k.py compare MOJO_OR_LLAMACPP_ARM sglang-dgemma-bf16 --across-weights`.

6. **Destroy the instance.** Step 7 of the vLLM runbook above: `vastai destroy instance INSTANCE_ID`, then
   `vastai show instances` must list nothing, then read the billing page.

| Symptom | Likely cause | Action |
|---|---|---|
| the run refuses with `the server reports the models ...` | the served name differs from `--model` | pass `--served-model` with the id `/v1/models` lists |
| `GET /server_info ... HTTP 401` | the server was started with `--api-key` | restart it without one and keep the loopback bind |
| `GET /server_info did not report a version string` or `0.0.0.dev0` | a source install without version metadata | run a release, or a checkout with its git history |
| `the server reports no diffusion algorithm` | the server runs an autoregressive model, or no `--dllm-algorithm` | serve with the cookbook command |
| `the server does not report its dtype` | `/server_info` says `auto` | pass `--dtype bfloat16` (declared), or serve with `--dtype bfloat16` |
| `max_running_requests` refusal | the server was not started with `--max-running-requests 1` | restart it with the flag |
| `--model ... is neither the model path ...` | the label names other weights than the server runs | pass the served checkpoint as `--model` |
| `answered HTTP 400 ... does not support request-level ...` | the server rejected a request field | the message names the field; report it, the engine sends none of them |
| `returned an empty completion` | the answer is in `reasoning_content`, or the model produced nothing | `curl` one chat request by hand and look at `choices[0].message` |
| out of memory on the server | the weights plus the cache do not fit the card | see Which hardware |

**Not done yet.** No server was run for this engine: the tests use a fake server on 127.0.0.1, and no SGLang, GPU or
weights were involved. The fields above were read from SGLang's source and cookbook, not observed on a live server
(`/model_info`, `/server_info`, the `dtype` and `max_running_requests` values a real launch reports, the
`usage.completion_tokens` of a dLLM answer, the `finish_reason`). The evaluation (issue #180, scope item 5: batch size
1, the same 200 problems, a 256-token canvas, on a rented machine) has not run, so there are no SGLang rows in this
document.

### DiffusionGemma across engines

The four engines that can answer DiffusionGemma are meant to run the same 200 problems
(`bench/eval/gsm8k-200.jsonl`), with the same prompt, the same scoring and the same arm-file format, so
`eval_gsm8k.py compare --across-weights` pairs any two of them. This table is a skeleton and holds no measured value.
A row gets a number only from a committed arm file under `bench/eval/runs/`; until then it says `no committed run yet`.
Weights differ between rows, so every comparison includes the precision as well as the engine.

| Arm | Weights (as the arm's `model` records them) | Algorithm | How it runs | Committed run |
|---|---|---|---|---|
| llama.cpp reference (the branch of ggml-org/llama.cpp#24423) | `diffusiongemma-26B-A4B-it-Q4_K_M.gguf`, the file `docs/correctness.md` names | that branch's own loop; I did not read it for this table | the build is `scripts/build_dg_ref.sh`; the GSM8K runner for llama.cpp's diffusion CLI is issue #177 (open), and I have not checked that `bench/engines/llamacpp.py` drives this build | no committed run yet |
| vLLM | `diffusiongemma-26B-A4B-it\|fp8-dynamic`, or the bf16 checkpoint (the choice is the maintainer's, see the vLLM runbook) | vLLM's entropy-bound sampler, as the recipe page describes it | `--engine vllm` (issue #179), runbook above | no committed run yet |
| SGLang | `diffusiongemma-26B-A4B-it\|bfloat16` | `Gemma4Renoise` | `--engine sglang` (issue #180), runbook above | no committed run yet |
| mojo-dllm | the Q4_K_M GGUF of the llama.cpp row | its own entropy-bound loop, `docs/diffusion.md` ("DiffusionGemma's loop") | `mojo-dllm run` does not run DiffusionGemma yet: wiring the model into the generation loop is issue #277 (open), see `docs/introduction.md` | no committed run yet |

The six defaults in `docs/cli.md` (`--eb-steps` 48, `--eb-t-min` 0.4, `--eb-t-max` 0.8, `--eb-bound` 0.1,
`--eb-stability` 1, `--eb-confidence` 0.005) are the values the SGLang cookbook lists for `Gemma4Renoise`
(`max_denoising_steps`, `temperature_schedule`, `entropy_bound`, `stability_threshold`, `confidence_threshold`; read
2026-11-09). I did not check that the two loops agree step for step. When the arms exist, name each one's commit,
server release and algorithm settings next to its row, from the arm's own `version` and `flags`.

## Limits of this evidence

- One machine: a 13th Gen Intel i5-13420H laptop for the CPU tables, with an
  RTX 3050 6 GB laptop GPU for the GPU tables. Another CPU, memory speed or
  GPU can change the ratios.
- The four main tables use the median of 3 runs on each of 3 prompts (9 runs);
  the forward-pass tables use 2 or 3 runs on one prompt; the quality table is a
  single run per cell. Each table has one model file and one quantization
  (Q4_K_M).
- The schedules differ between runtimes (see above), so tokens / s compares
  runtimes at one step count, not at one output quality.
- Each table names its commit in a `Versions` or `Evidence` line, except the two
  forward-pass tables, whose commits are in their evidence files. They differ,
  because the evidence was recorded as the work went on; no single commit holds
  all of it, and the release tag is not one of them.
- The quality column counts immediate repeated words on two prompts. It is not
  a perplexity or a task score. The only task evidence is the GSM8K run for
  `--threshold` and `--dual-cache` on LLaDA.
- diffuse-cpp loads its own conversion of the model, so its rows are indicative.
- No binary is published yet ([#105](https://github.com/vladimirrott/mojo-dllm/issues/105)): you build from source.

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
