# Architecture

```text
                 mojo-dllm run
                      |
        +-------------+--------------+
        |             |              |
   GGUF reader    tokenizer     diffusion loop
   (mmap)         (BPE)         (diffusion/sampler.mojo)
        |                            |
        |                       model forward
        |                       (models/transformer.mojo)
        |                            |
        +----> repacked weights ---> quantized GEMM, RMSNorm, RoPE,
               (kernels/packed.mojo)  attention, SwiGLU (kernels/)
```

## Source map

| Path | What it does |
|---|---|
| `src/main.mojo`, `src/cli/` | command line: `run`, `tokenize`, `inspect`, `logits` |
| `src/mojo_dllm/formats/gguf.mojo` | GGUF v2/v3 reader over a memory map, bounds-checked |
| `src/mojo_dllm/quant/kquants.mojo` | scalar F32 / F16 / BF16 / Q5_0 / Q8_0 / Q4_K / Q5_K / Q6_K decoders (the test oracle, and embedding rows) |
| `src/mojo_dllm/kernels/packed.mojo` | weights repacked 8 rows per SIMD register; the GEMM used for every projection |
| `src/mojo_dllm/kernels/qgemm.mojo` | activation quantization (int8, one scale per 256) and a row-major reference GEMM |
| `src/mojo_dllm/kernels/ops.mojo` | RMSNorm, NORM and NEOX RoPE, bidirectional attention, SwiGLU |
| `src/mojo_dllm/models/transformer.mojo` | config from GGUF metadata, weight loading, the CPU forward pass, and the `DenoisingModel` trait the samplers drive |
| `src/mojo_dllm/gpu/kernels.mojo` | GPU kernels: int8 tensor-core GEMM over Q4_K / Q6_K blocks, flash-style attention, RMSNorm, RoPE, SwiGLU |
| `src/mojo_dllm/gpu/model.mojo` | the forward pass on the GPU: weights uploaded once, activations kept on the device |
| `src/mojo_dllm/tokenizer/bpe.mojo` | byte-level BPE with the `bailingmoe` and `qwen2` pre-tokenizers, and the `Tokenizer` that picks it or `gemma4` from `tokenizer.ggml.model` |
| `src/mojo_dllm/tokenizer/gemma4.mojo` | DiffusionGemma's tokenizer (U+2581 spaces, newline-run split, merges by rank, `<0xNN>` byte fallback) and its chat template |
| `src/mojo_dllm/diffusion/sampler.mojo` | LLaDA's and Dream's denoising loops and the PRNG |
| `src/mojo_dllm/sys/` | aligned buffers, mmap, a dynamically scheduled parallel-for |

## A forward pass

For a canvas of L tokens and a list of positions whose logits the caller
needs:

1. Embedding rows are decoded from the memory-mapped embedding table (Q4_K in
   a Q4_K_M file), one row per token.
2. For each layer (32 in LLaDA-8B): RMSNorm, then the activations are quantized to int8
   once and reused by the Q, K and V projections. RoPE rotates Q and K;
   attention runs over all L positions with no mask; the output projection
   and residual follow. The FFN quantizes once for gate and up, applies SiLU,
   quantizes again for down.
3. In the last layer, only the requested positions continue past attention.
4. Final RMSNorm and the output projection run on the requested rows only.

The forward pass allocates nothing: every activation buffer is sized once for
the longest canvas the run will use.

## Memory

The GGUF file is memory-mapped. At load, every projection matrix is repacked
into a heap block (see [The quantized GEMM](kernels.md)) and its mapped pages
are released, so the model is resident once. Q4_K stays 4-bit after
repacking. Q6_K expands to one byte per weight, which costs about 0.4 GB on
LLaDA-8B and buys a simpler inner loop. The embedding table stays mapped and
faults in as needed. Norm vectors and biases are decoded into f32 arrays at
load.

## Threads

`sys/parallel.mojo` hands out work items from a shared atomic counter. On a
hybrid CPU (performance and efficiency cores) a static split makes every
parallel region wait for the slowest core; claiming items one at a time lets
the fast cores take more. On the development laptop the static split left
every GEMM waiting on the four efficiency cores.
