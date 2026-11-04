# CPU MoE layer benchmark

`bench_moe.mojo` measures one layer at the DiffusionGemma shape: 256 tokens,
hidden width 2816, FFN width 704, 128 experts and eight experts selected per
token. It uses deterministic synthetic Q4_K gate/up weights and runs both
Q5_0 and Q8_0 down projections. The router uses a nonconstant scale vector
and an F32 matrix. All generated quantization scales are finite.

Build with the repository's pinned Mojo toolchain, then run:

```bash
bash scripts/build.sh bench/bench_moe.mojo build/bench_moe
python3 bench/run_moe.py --threads 8 --reps 5 \
  --out bench/checks/moe-cpu.json
```

Each sample includes router normalization, projection and top-eight selection,
expert grouping, activation quantization, batched gate/up and down GEMMs,
GELU and weighted scatter-add. Weight generation, repacking and the reusable
expert workspace are set up once before timing. One full warmup is recorded
but excluded from the median. Router temporary allocation is included in the
timed call. Both the raw weight bytes and their packed copies remain resident.

The JSON retains every sample, setup time, output checksum, active-expert
count and maximum tokens received by an expert. It also records the CPU/ISA
listing, CPU affinity, thread count, compiler version and system load before
and after the run. The runner refuses an existing output path, incomplete
samples, nonfinite or changing outputs, and shape mismatches. Raw samples and
their median are reported separately for the two weight formats.

A small execution check avoids allocating the full layer:

```bash
python3 bench/run_moe.py --threads 2 --tokens 16 --hidden 256 --ffn 32 \
  --experts 4 --top-k 2 --reps 2 --out /tmp/moe-smoke.json
python3 -m unittest discover -s tests/python -p test_moe_bench.py
```

Use an available CPU allocation for full measurements; this runner does not
change power settings, CPU affinity, or other processes. This benchmark
reports the latency of one synthetic layer with the dimensions above.

A [recorded local run](checks/2026-11-04-moe-cpu-local.json) includes both
formats on an Intel Core Ultra 7 258V under WSL2, with eight threads and CPU
affinity 0-7. Its raw samples and metadata are the source for the summary.
