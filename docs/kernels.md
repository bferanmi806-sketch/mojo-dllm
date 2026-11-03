# The quantized GEMM

Most of a denoising step is matrix multiplication: seven projections per
layer, 32 layers in LLaDA-8B, every step (without `--dual-cache`), over the
whole canvas. `run --verbose` prints the split for your machine. This page explains the
kernel that does it.

## The shape of the problem

A step multiplies each weight matrix W (N rows × K columns, 4 to 6 bits per
weight) by an activation matrix X (L rows × K). L is the canvas length, a few
hundred at most. That is a skinny GEMM, and it is compute-bound: each weight
is read once per step and used L times.

## Activations: int8 with a scale per 256

Before a projection, each activation row is quantized to int8 with one f32
scale per 256 values, and the sum of each group of 16 values is kept as a
float. This is llama.cpp's Q8_K scheme, which keeps the comparison with it
fair. Integer dot products are 4× denser than f32 FMAs on AVX2.

## Weights: 8 rows per register

GGUF stores each row contiguously. A row-at-a-time dot product needs a
horizontal sum per output, and the block scales arrive one scalar at a time.
So the loader repacks every matrix once:

```text
for each group of 8 rows, for each 256-weight super-block:
    Q4_K:  d[8], dmin[8] as f32 | scales[8][8], mins[8][8] as u8 | 1024 bytes of nibbles
    Q6_K:  d[8] as f32          | scales[16][8] as i8           | 2048 bytes, one value per byte
```

Inside the weight bytes, a 32-byte load holds 4 consecutive weights of each of
the 8 rows. One `vpdpbusd` (AVX-VNNI) multiplies those 32 unsigned weights by
4 activation bytes broadcast to every lane and adds into 8 int32
accumulators: 8 rows × 4 multiply-adds in one instruction. The scale of a
sub-block is an 8-lane vector, one entry per row, applied once per 16 or 32
weights.

The zero points do not touch the inner loop. Q4_K computes
`w = d·sc·q − dmin·m`, so the second term becomes `dmin·m` times the
activation sum of the sub-block, which the quantizer already stored. Q6_K
stores `q + 32`, and the 32 comes off the same way.

## Tiling and threads

A work item is 16 output rows. It walks the tokens 8 at a time for Q4_K and 4
at a time for Q6_K, keeping that many int32 and f32 accumulators in registers,
so each loaded weight vector feeds 8 or 4 tokens. Items go to threads through an atomic counter (see
[Architecture](architecture.md#threads)).

## Checks

`tests/test_quant.mojo` compares the decoders with the `gguf` Python package's
dequantization, and both GEMMs with an f64 reference over the same dequantized
weights. Swapping the nibble halves or dropping the Q6_K zero point fails
these tests; that mutation check runs before changes to this file land.

`bench/bench_qgemm.mojo` measures the kernel alone on LLaDA's layer shapes.

## The GPU path

`src/mojo_dllm/gpu/kernels.mojo` does the same arithmetic on an NVIDIA GPU,
with a layout chosen for tensor cores instead of AVX registers.

**Weights stay in GGUF blocks.** Upload copies each Q4_K matrix as it is and
pads each 210-byte Q6_K block to 224 bytes, so every 16-byte chunk of a block
is aligned and the kernels can load 16 bytes at a time.

**Activations get one scale per 32.** The GPU quantizes activations to int8
with one f32 scale per 32 values (llama.cpp's Q8_1 shape) and keeps
`scale x sum(q)` next to it. One warp quantizes one 32-value block, a value
per lane, and reduces its max and integer sum with shuffles, so loads and
stores are coalesced. A 32-weight Q4_K sub-block then meets exactly one
activation scale:

```text
sum w x = d*sc*dx * sum(q * qx)  -  dmin*m * (dx * sum(qx))
```

**One mma per scale group.** A Q6_K thread block computes 64 weight rows by a
tile of 32, 64 or 80 tokens. For each 256-wide super-block it copies the
weights into shared memory (see the Q6_K paragraph below), copies the matching
activations next to them, and each of its 4 warps runs one `mma.m16n8k16.s8`
per 16-weight scale group for its 16 rows and every 8 tokens. (A Q4_K block
scales every 32 weights, so it runs `mma.m16n8k32.s8` per sub-block instead.)
The int32 result goes to f32 with the two scales and, for Q4_K, the min term.

**Q4_K: packed in shared memory, unpacked in registers.** Removing the staging
work cut the GEMM roughly in half
(`bench/checks/2026-10-13-gemm-cost-split.json`: with staging done once, the
GEMM took about 100 ms instead of about 220), while removing the mma saved
only a few milliseconds. So a Q4_K block copies each row's 144-byte
super-block as it is, in 16-byte loads: the 16-byte header (d, dmin and the
packed scales) to one array, the 128 bytes of nibbles to another. A warp
unpacks its A fragments in registers: word i of nibble pair p holds sub-block
2p in its low nibbles and 2p + 1 in its high ones at the same position, so one
shared load feeds two mmas, and the 6-bit scales are decoded from the header
per sub-block. With the weights packed, a block holds 128 rows and runs 8
warps in the same 48 KiB, which halves the activation copies per output. The
arithmetic is unchanged, so the logits are too
(`bench/checks/2026-10-13-gemm3-logits.json`).

**Prefetching the next super-block.** On the kernel before the packed layout,
removing staging saved about 120 ms of the GEMM and removing the mma 4 to
7 ms, both measured against the integer-accumulation variant
(`bench/checks/2026-10-13-gemm-cost-split.json`). The packed layout (128-row
blocks) took that harness from 258.2 to 215.0 ms (median of three runs, `v3_vs_base`
in the same file) without a breakdown of the
staging share. The loop still copied a super-block, waited at a barrier, and
only then computed. Now each thread
reads super-block b + 1 from global memory into registers (weights,
activation bytes and activation scales) before the mma loop of b, and stores
it to shared memory at the top of the next iteration, after the barrier that
ends b. The Q4_K GEMM went from 148.8 to 87.2 ms per forward (median
launch time times launches, middle value of three interleaved traces,
`bench/profiles/2026-10-13-gemm4-*`), and the
logits match main on all 84 rows (`bench/checks/2026-10-13-gemm4-logits.json`).
That the gain comes from overlapping the loads with compute is the reading
the numbers fit; no hardware counters were collected to confirm it.

**Q6_K: the same layout and prefetch.** The Q6_K kernel took the route the
Q4_K one did. A block of 64 rows copies each row's 224 bytes (low nibbles,
high bits, 16 int8 group scales, d) into shared memory as they are, in 16-byte
loads; a warp rebuilds each 16-weight group's A fragment in registers
(`_q6k_frag`: nibble, two high bits, minus 32 by an xor and a sign extension),
and the next super-block waits in registers while the mma loop runs. Groups
accumulate in the same order as before, so the result is unchanged. The
padded block is 12 bytes larger than the 212-byte padding it replaces: with
6,217,728 Q6_K blocks in LLaDA-8B Q4_K_M (output, 16 attn_v and 16 ffn_down
tensors) that is 74.6 MB (71.2 MiB) more device memory.

**Tile width per call.** The default rule minimizes token tiles, breaking
ties toward the narrower width. RTX Blackwell (compute capability major 12)
uses a shape-aware rule for Q4_K: start from that default, then choose the
narrowest width that does not increase the estimated number of block waves
across the device's SMs. The estimate assumes one resident block per SM;
it is a dispatch heuristic, not an occupancy measurement. It can trade more
blocks for less work within each block when a wide GPU has spare parallelism.
Q6_K and other device families keep their existing rule. The SM count is read
when activation buffers are created, not on every GEMM call.

The LM head's small row count still uses the narrow tile. The available token
widths remain 32, 64 and 80; 80 is the widest whose shared memory fits the
48 KiB static allocation limit. Weight layout, activation quantization and
accumulation order are unchanged. Token tiles of one weight tile run back to
back, so they read it through L2. The exact cross-width GPU test also covers
ragged token counts and partial weight-row tiles.

**Shared-memory layout.** Rows are padded so the 8 rows a fragment load reads
fall in different banks: activation rows are 272 bytes apart (256 plus 16),
Q4_K nibble rows 144 bytes (128 plus 16) and Q6_K block rows 240 bytes (224
plus 16). Activation scales are stored by sub-block, then token, then (scale,
sum), so the pair of tokens a thread folds after an mma is one 128-bit load.
`tests/gpu/test_gpu.mojo` checks the fragment layout against a CPU matmul
on random int8 matrices, and the whole kernel against an f64 GEMM.

**Row statistics on the device.** A greedy step needs, per masked row, the
argmax and a confidence. Copying the logits to the host (in LLaDA, 32 rows of
126,464 floats for a 32-token block) and looping over them there is the largest piece
of a step outside the forward pass. One block per row now reads its logits
twice: the first pass finds the first index of the maximum (a tie goes to the
smaller index, as the host loop picks it), the second accumulates, in f64,
the sum of exp(x - max), the sum of exp(x - max)(x - max) and the largest
logit at any other index. The sampler finishes from those four numbers:
p = 1 / z for LLaDA and maskgit_plus, p minus the runner-up's probability
for topk_margin, and sum e (x - max) / z - log z for Dream's entropy.
The exponentials are f32, so a confidence differs from the host's f64 loop in
the sixth digit; tests bound it at 1e-5 relative.

**Fused elementwise kernels.** The quantized GEMMs want int8 activations, so
each norm and the FFN activation end in a quantize. Run separately, a layer
wrote and re-read the normalized rows and the product of the gate and up
projections through global memory. Three kernels remove that traffic:
`add_rmsnorm_quantize` adds the projection output into the residual stream,
normalizes the sum and quantizes it; `rmsnorm_quantize` does the same for
layer 0, whose input has no add; `silu_mul_quantize` quantizes silu(g) * u
without storing it. The norm kernels give thread t the values t, t + 256, ...
so a warp holds 32 consecutive values at each step, which is one
quantization block, and the quantize step is the same `_quantize_lane` code
the plain kernel runs. The float expressions and their order match the
unfused kernels', so the fused output is identical to the chain's, which the
tests assert exactly. The attention output still goes through the plain
quantize kernel.

**Attention on f16 tensor cores.** Each warp owns 16 queries and walks the
keys 32 at a time. Q is scaled by 1/sqrt(d) and held in registers as f16
fragments; K is staged row-major and V transposed, both as f16, so every
operand load is one 32-bit word. `mma.m16n8k16` computes S = Q K^T in f32,
the four threads that share a row reduce its max with two shuffles, and the
online softmax rescales the running sums. S's accumulator for 16 keys has the
same register layout as the A operand of the next mma, so P goes from S to
P V without leaving the registers. On LLaDA-8B the logits stay as close to
the f32 CPU path as with f32 attention
(`bench/checks/2026-10-13-attention-f16-logits.json`). The kernel never
stores a score row, so the canvas length has no fixed limit.
