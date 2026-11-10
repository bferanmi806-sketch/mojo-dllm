# mojo-dllm: Technical Specification

**Status:** Original specification, draft v0.1, written on 2026-10-03 before the code. It is not kept up to date. The [development plan](plan.md) lists the corrections that research made, [Architecture](architecture.md) maps the source as built, and [Command line](cli.md) lists the current flags.  
**Project type:** Open-source ML inference systems project  
**Primary language:** Mojo  
**Initial model:** LLaDA-8B, quantized GGUF  
**Initial target:** x86 CPU  
**Secondary target:** NVIDIA consumer GPU  
**Long-term targets:** Apple Silicon, AMD GPU  
**License target:** Apache-2.0 or MIT-compatible, subject to dependency review

---

## 1. Summary

`mojo-dllm` is a lightweight, Mojo-native runtime for local inference of diffusion language models (dLLMs).

The project will begin with **LLaDA-8B** in quantized **GGUF** form and implement the complete inference path in Mojo, including:

- GGUF model loading;
- tokenization;
- quantized matrix multiplication;
- transformer execution;
- bidirectional attention;
- diffusion masking and remasking;
- confidence/entropy-based token selection;
- iterative denoising;
- deterministic sampling;
- CLI inference.

The first milestone is deliberately CPU-only and correctness-focused.

The primary long-term technical thesis is:

> A single Mojo inference implementation can execute diffusion language models efficiently across heterogeneous consumer hardware, while sharing substantially more kernel/runtime code than traditional C++ + CUDA + Metal + ROCm stacks.

The project is **not** intended to introduce a new diffusion-model algorithm initially. Its novelty is primarily systems-oriented: portable, quantized, local dLLM inference in Mojo.

---

## 2. Motivation

Most current high-performance language-model runtimes are optimized around autoregressive decoding.

Diffusion language models use a fundamentally different inference loop:

1. initialize part or all of the generation region with mask tokens;
2. execute a bidirectional transformer pass;
3. estimate token distributions for unresolved positions;
4. select some positions to commit;
5. optionally remask low-confidence positions;
6. repeat until the generation region is resolved.

This creates runtime requirements that differ from ordinary autoregressive inference:

- repeated whole-sequence or block-level transformer evaluation;
- bidirectional attention rather than a purely causal mask;
- diffusion-specific scheduling state;
- mask/remask operations;
- per-position confidence or entropy calculations;
- opportunities for inter-step state reuse;
- potentially different memory/computation tradeoffs.

Existing implementations primarily use:

- PyTorch;
- C/C++;
- CUDA;
- Triton;
- specialized serving frameworks.

Mojo is a strong candidate for this workload because it aims to support portable low-level CPU and accelerator programming from a common language and type system.

---

## 3. Project Goals

### 3.1 v0.1 goals

The first release MUST:

- run **LLaDA-8B Q4_K_M GGUF** locally;
- require **no Python at inference time**;
- execute entirely on x86 CPU;
- parse GGUF directly in Mojo;
- tokenize prompts locally;
- perform iterative diffusion generation;
- expose a simple CLI;
- support deterministic generation with a fixed seed;
- produce numerically reasonable parity with a reference runtime;
- provide reproducible benchmark tooling.

Target invocation:

```bash
mojo-dllm run \
  --model ./LLaDA-8B-Q4_K_M.gguf \
  --prompt "Explain speculative decoding." \
  --max-tokens 128 \
  --steps 128 \
  --seed 42
```

### 3.2 v0.2 goals

- optimized CPU SIMD;
- thread-parallel transformer execution;
- faster Q4_K matrix multiplication;
- fused normalization;
- improved sampling kernels;
- benchmark parity against `diffuse-cpp` and `llama.cpp`.

### 3.3 v0.3 goals

- NVIDIA GPU backend;
- RTX 30-series support as an initial consumer target;
- device-side quantized GEMM;
- device-side attention;
- device-side diffusion sampling operations;
- optional CPU/GPU split execution.

### 3.4 v0.4+ goals

- Dream model support;
- Apple Silicon GPU backend;
- AMD GPU backend;
- inter-step cache experiments;
- block diffusion;
- runtime integration layer for projects such as `molla`;
- API/library mode in addition to CLI mode.

---

## 4. Non-Goals

The initial project MUST NOT attempt to implement:

- an OpenAI-compatible HTTP server;
- multi-user continuous batching;
- distributed inference;
- tensor parallelism;
- speculative decoding;
- training;
- fine-tuning;
- arbitrary Hugging Face model loading;
- every GGUF quantization format;
- MAX serving integration;
- custom model download infrastructure;
- a general-purpose LLM runtime;
- novel dLLM research algorithms before baseline correctness.

This project should remain a **small inference core**, not become another general serving framework.

---

## 5. Initial Model

### 5.1 Model

Initial target:

**LLaDA-8B**

Preferred representation:

**GGUF Q4_K_M**

Reasons:

- laptop-friendly model size;
- existing public quantized artifacts;
- simple enough transformer structure;
- multiple independent reference implementations;
- existing C/C++ implementations provide useful correctness oracles;
- bidirectional transformer execution makes it a representative dLLM baseline.

### 5.2 Second model

The second supported model SHOULD be:

**Dream-7B**

Adding Dream forces the runtime to generalize beyond a single LLaDA-specific model layout and helps validate the abstraction around:

- GQA;
- model metadata;
- tokenizer configuration;
- diffusion scheduler differences;
- transformer architecture differences.

---

## 6. High-Level Architecture

```text
                    +----------------------+
                    |      mojo-dllm       |
                    +----------+-----------+
                               |
             +-----------------+-----------------+
             |                 |                 |
             v                 v                 v
        GGUF Loader        Tokenizer      Diffusion Runtime
             |                                   |
             |                          +--------+--------+
             |                          |                 |
             v                          v                 v
       Tensor Store              Diffusion State      Scheduler
             |                          |                 |
             +-------------+------------+-----------------+
                           |
                           v
                    Model Execution
                           |
             +-------------+-------------+
             |                           |
             v                           v
       Quantized Linear           Bidirectional Attention
             |                           |
             +-------------+-------------+
                           |
                           v
                     Device Backend
                  CPU / NVIDIA / ...
```

---

## 7. Repository Layout

Suggested initial structure:

This is the layout the draft proposed. The code differs: modules live under `src/mojo_dllm/`, tests are in `tests/`, benchmarks in `bench/`, and `pyproject.toml` pins the Mojo toolchain. [Architecture](architecture.md#source-map) maps the files as built.

```text
mojo-dllm/
├── README.md
├── LICENSE
├── pyproject.toml              # only if needed for tooling; not runtime
├── mojoproject.toml
├── src/
│   ├── main.mojo
│   ├── cli/
│   │   └── run.mojo
│   ├── formats/
│   │   ├── gguf.mojo
│   │   └── tensor_store.mojo
│   ├── tokenizer/
│   │   ├── tokenizer.mojo
│   │   └── bpe.mojo
│   ├── quant/
│   │   ├── q4_k.mojo
│   │   ├── q8_0.mojo
│   │   └── dequant.mojo
│   ├── kernels/
│   │   ├── matmul.mojo
│   │   ├── rmsnorm.mojo
│   │   ├── rope.mojo
│   │   ├── attention.mojo
│   │   ├── softmax.mojo
│   │   └── sampling.mojo
│   ├── models/
│   │   ├── model.mojo
│   │   └── llada.mojo
│   ├── diffusion/
│   │   ├── state.mojo
│   │   ├── scheduler.mojo
│   │   ├── masking.mojo
│   │   ├── confidence.mojo
│   │   └── sampler.mojo
│   └── runtime/
│       ├── device.mojo
│       ├── cpu.mojo
│       └── executor.mojo
├── tests/
│   ├── test_gguf.mojo
│   ├── test_quant.mojo
│   ├── test_attention.mojo
│   ├── test_scheduler.mojo
│   └── test_llada.mojo
├── benchmarks/
│   ├── bench_matmul.mojo
│   ├── bench_attention.mojo
│   ├── bench_generation.mojo
│   └── scripts/
└── docs/
    ├── architecture.md
    ├── model-porting.md
    └── benchmarks.md
```

---

## 8. Core Data Structures

The structs, traits and signatures in sections 8, 10.2, 13, 17.1 and 24 are sketches written before the code. The code has `ModelConfig` and the `DenoisingModel` trait in `src/mojo_dllm/models/transformer.mojo`, `TensorInfo` in `src/mojo_dllm/formats/gguf.mojo`, and `GenConfig` with `generate` in `src/mojo_dllm/diffusion/sampler.mojo`. It has no `DiffusionState`, `DiffusionScheduler`, `Device` or `DiffusionModel`, and it uses `def` where the sketches use `fn`.

### 8.1 Model configuration

```mojo
struct ModelConfig:
    var vocab_size: Int
    var hidden_size: Int
    var intermediate_size: Int
    var num_layers: Int
    var num_attention_heads: Int
    var num_kv_heads: Int
    var head_dim: Int
    var max_position_embeddings: Int
    var rope_theta: Float64
    var mask_token_id: Int
```

### 8.2 Diffusion state

```mojo
struct DiffusionState:
    var tokens: Tensor[Int32]
    var masked: BitSet
    var committed: BitSet
    var confidence: Tensor[Float32]
    var step: Int
    var total_steps: Int
```

Future versions MAY add:

```mojo
var dirty_positions: BitSet
var block_state: BlockState
var cache_state: DiffusionCache
```

but cache-related structures SHOULD NOT be required for v0.1.

### 8.3 Tensor descriptor

```mojo
struct TensorInfo:
    var name: String
    var shape: List[Int]
    var ggml_type: Int
    var offset: UInt64
    var byte_length: UInt64
```

---

## 9. GGUF Support

### 9.1 Required v0.1 support

The GGUF reader MUST support:

- GGUF header validation;
- metadata parsing;
- tensor metadata;
- tensor offsets;
- memory-mapped or direct file-backed reads;
- little-endian decoding;
- required tokenizer metadata;
- LLaDA tensor-name mapping.

Initial quantization support:

- `Q4_K_M`;
- optionally `Q8_0`;
- optionally `F16` for correctness tests.

### 9.2 Unsupported formats

Unknown tensor types MUST fail explicitly:

```text
unsupported GGUF tensor type: IQ3_XS (tensor blk.0.attn_q.weight)
```

No silent fallback.

A type with a reference decoder but without a matrix kernel currently fails
at matrix load, for example:

```text
tensor blk.0.attn_q.weight is Q8_0: matrix kernels exist for Q4_K and Q6_K
```

### 9.3 Loading strategy

Prefer memory mapping where practical:

```text
GGUF file
   |
   v
memory map
   |
   +--> metadata
   |
   +--> tensor descriptors
   |
   +--> quantized tensor views
```

Avoid eagerly dequantizing the entire model into FP32.

---

## 10. Quantized Compute

### 10.1 Initial requirement

Implement Q4_K_M matrix multiplication directly against quantized weights.

Conceptually:

```text
Q4_K_M weights
      |
      v
block decode
      |
      v
vectorized dot product
      |
      v
FP32/BF16 accumulation
```

The baseline MAY initially use a simple reference implementation.

The optimized implementation SHOULD:

- exploit SIMD;
- operate blockwise;
- avoid full-tensor dequantization;
- reuse scales efficiently;
- minimize temporary allocations.

### 10.2 Kernel API

Example:

```mojo
fn linear_q4k[
    T: DType
](
    x: TensorView[T],
    weight: QuantizedTensorView,
    out: TensorView[T],
):
    ...
```

### 10.3 Benchmark requirements

Q4_K_M GEMM MUST be benchmarked independently.

Measurements:

- matrix dimensions;
- latency;
- effective memory bandwidth;
- effective FLOP/s where meaningful;
- allocations;
- thread count.

---

## 11. Transformer Execution

A LLaDA transformer layer will initially include:

```text
input
  |
RMSNorm
  |
QKV projection
  |
RoPE
  |
bidirectional attention
  |
output projection
  |
residual
  |
RMSNorm
  |
MLP
  |
residual
```

### 11.1 Critical difference from autoregressive execution

The attention mask is not purely causal.

The runtime must support full/bidirectional attention over the relevant diffusion region.

Therefore, the initial implementation SHOULD NOT assume an append-only KV cache.

---

## 12. Diffusion Inference Loop

The core execution loop should be explicit and readable.

Pseudo-code:

```text
tokens = tokenize(prompt)
canvas = append_masks(tokens, max_new_tokens)

for step in 0 .. num_steps:
    logits = model.forward(canvas)

    unresolved = get_masked_positions(canvas)

    scores = confidence(logits, unresolved)

    selected = scheduler.select(
        unresolved,
        scores,
        step,
        num_steps
    )

    sampled = sample(logits[selected])

    canvas[selected] = sampled

    if scheduler.requires_remasking:
        remask_low_confidence(canvas, scores)

return decode(canvas)
```

The scheduler MUST be modular.

---

## 13. Scheduler Interface

```mojo
trait DiffusionScheduler:
    fn initialize(
        tokens: Tensor[Int32],
        generated_length: Int
    ) -> DiffusionState

    fn select_positions(
        state: DiffusionState,
        logits: TensorView[Float32]
    ) -> Selection

    fn update(
        state: DiffusionState,
        selection: Selection,
        sampled_tokens: TensorView[Int32]
    )
```

Initial scheduler:

- reference implementation matching LLaDA inference behavior.

Future schedulers:

- confidence-based;
- entropy-based;
- fixed block count;
- adaptive steps;
- block diffusion.

---

## 14. Sampling

Required primitives:

- softmax;
- log-softmax where needed;
- entropy;
- argmax;
- top-k;
- categorical sampling;
- temperature;
- deterministic PRNG;
- masked-position gather/scatter.

Possible future fused kernel:

```text
logits
  |
softmax
  |
entropy/confidence
  |
top-k
  |
sample
  |
scatter
```

---

## 15. Tokenizer

v0.1 MUST perform tokenization without invoking Python.

Possible implementation paths:

1. reuse or adapt an existing Mojo BPE tokenizer;
2. implement the minimum tokenizer behavior required by the target model;
3. parse tokenizer metadata from GGUF where available.

The tokenizer MUST correctly support:

- prompt encoding;
- mask token;
- BOS/EOS if required;
- decoding generated IDs;
- byte fallback if required by the model tokenizer.

---

## 16. CPU Runtime

The initial backend is x86 CPU.

Priority order:

### Stage 1: correctness

- scalar/reference kernels;
- single thread;
- deterministic output.

### Stage 2: vectorization

- SIMD vector operations;
- vectorized dequantization;
- vectorized dot products;
- optimized softmax.

### Stage 3: threading

Potential parallel regions:

- output channels in linear layers;
- attention heads;
- MLP projections;
- batch-independent work.

Threading MUST be benchmark-driven.

---

## 17. NVIDIA Backend

The NVIDIA backend begins only after the CPU runtime passes correctness tests.

Initial GPU target:

- consumer NVIDIA RTX 30-series;
- FP16/BF16 where supported;
- quantized model weights;
- device-resident hot tensors.

Required kernels:

- Q4_K_M GEMM;
- RMSNorm;
- RoPE;
- attention;
- softmax;
- diffusion sampling operations.

### 17.1 Device abstraction

The runtime SHOULD expose:

```mojo
trait Device:
    fn allocate(...)
    fn copy_to_device(...)
    fn copy_from_device(...)
    fn synchronize(...)
```

Kernel implementation should avoid unnecessary backend-specific code where Mojo abstractions are sufficient.

---

## 18. CPU/GPU Hybrid Execution

Future laptop-focused mode:

```text
system RAM
   |
model weights
   |
+--+--------------------+
|                       |
CPU                  GPU VRAM
|                       |
fallback layers      hot layers
|                       |
+-----------+-----------+
            |
       diffusion state
```

Potential strategies:

- layer offload;
- operator offload;
- GPU-only attention;
- GPU-only matrix multiplication;
- dynamic VRAM budget.

This is NOT required for v0.1.

---

## 19. Correctness Strategy

The project MUST use at least two external reference implementations when available.

Recommended references:

- official PyTorch LLaDA implementation;
- `llama.cpp` diffusion implementation;
- `diffuse-cpp`.

Validation levels:

### Level 1: tensor-level

Compare:

- embeddings;
- norm outputs;
- Q/K/V projections;
- RoPE outputs;
- attention outputs;
- MLP outputs;
- final logits.

Use small contexts and fixed inputs.

### Level 2: diffusion-step parity

Compare:

- mask positions;
- confidence values;
- selected positions;
- sampled token IDs;
- updated canvas.

### Level 3: generation parity

With fixed:

- model;
- prompt;
- seed;
- temperature;
- diffusion steps;
- generation length;

compare final output and intermediate states.

Exact parity may not always be possible across floating-point implementations, so tolerances MUST be documented.

---

## 20. Testing

Minimum unit tests:

### File format

- valid GGUF header;
- invalid magic;
- metadata parsing;
- tensor offset calculations;
- Q4_K block parsing.

### Quantization

- known Q4_K block decode;
- scalar vs SIMD result;
- matrix-vector correctness.

### Transformer

- RMSNorm;
- RoPE;
- softmax;
- attention;
- MLP;
- full layer output.

### Diffusion

- mask initialization;
- confidence computation;
- selection logic;
- remasking;
- deterministic RNG.

### Integration

- tiny synthetic model;
- single transformer layer;
- short prompt;
- one diffusion step;
- full LLaDA run.

---

## 21. Benchmark Plan

### 21.1 Comparison targets

Primary:

- `llama.cpp`;
- `diffuse-cpp`.

Secondary:

- official PyTorch/Hugging Face implementation.

### 21.2 Metrics

Measure:

- time to first completed denoising step;
- time per denoising step;
- total generation latency;
- generated tokens per second;
- denoising steps per second;
- peak RSS;
- peak VRAM;
- model load time;
- CPU utilization;
- GPU utilization;
- energy consumption where practical.

### 21.3 Benchmark matrix

Example:

| Runtime | Device | Quant | Prompt | Output | Steps |
|---|---|---|---:|---:|---:|
| llama.cpp | CPU | Q4_K_M | 128 | 128 | 128 |
| diffuse-cpp | CPU | Q4_K_M | 128 | 128 | 128 |
| mojo-dllm | CPU | Q4_K_M | 128 | 128 | 128 |
| mojo-dllm | NVIDIA | Q4_K_M | 128 | 128 | 128 |

The published benchmarks use 128 generated tokens in 32 steps (4 tokens per step) on three short prompts, with `--block-length 32` for LLaDA. See [Benchmarks](benchmarks.md#what-is-compared).

---

## 22. Performance Philosophy

Correctness comes before optimization.

Optimization order:

```text
correct model execution
        |
        v
Q4_K compute
        |
        v
CPU SIMD
        |
        v
threading
        |
        v
fused kernels
        |
        v
GPU
        |
        v
inter-step caching
        |
        v
block diffusion
```

Do not introduce caching before a fully correct uncached baseline exists.

---

## 23. Future Caching Research

After the baseline runtime is stable, experiment with:

### 23.1 Inter-step cache

Track which intermediate state can be reused between adjacent denoising iterations.

### 23.2 Dirty-position tracking

```mojo
struct DirtyState:
    var changed_positions: BitSet
    var changed_blocks: BitSet
```

### 23.3 Layer-level reuse

Explore whether selected intermediate layer outputs can be reused under configurable error thresholds.

### 23.4 Block diffusion

Support models that denoise blocks rather than the entire generation canvas.

All cache optimizations MUST quantify:

- speedup;
- additional memory use;
- output divergence;
- benchmark quality impact.

---

## 24. Integration with Other Mojo Runtimes

The core should eventually expose a small library API.

Example:

```mojo
var model = DiffusionModel.load("model.gguf")

var config = GenerationConfig(
    max_new_tokens=128,
    steps=128,
    temperature=0.2,
    seed=42,
)

var output = model.generate(
    "Explain KV-cache quantization.",
    config
)
```

A runtime such as `molla` could then provide:

- HTTP serving;
- OpenAI-compatible APIs;
- model lifecycle;
- chat templates;
- request scheduling.

`mojo-dllm` should remain focused on dLLM execution.

---

## 25. CLI

Initial interface:

```text
mojo-dllm run [OPTIONS]
```

Required flags:

```text
--model PATH
--prompt TEXT
--max-tokens N
--steps N
--seed N
```

Only `--model` and `--prompt` are required. `--max-tokens` defaults to 128, `--steps` to the value of `--max-tokens`, and `--seed` to 42. `--device` takes `cpu` or `gpu`. The built binary is `build/mojo-dllm`, and the `./mojo-dllm` launcher takes the prompt as its first word. [Command line](cli.md) lists every flag.

Optional:

```text
--temperature FLOAT
--threads N
--device cpu
--verbose
--dump-step-stats
```

Example:

```bash
mojo-dllm run \
  --model ./llada-8b-q4_k_m.gguf \
  --prompt "What is paged attention?" \
  --max-tokens 128 \
  --steps 128 \
  --seed 1234 \
  --threads 8
```

---

## 26. Logging and Diagnostics

Verbose mode SHOULD print:

```text
model: LLaDA-8B
quantization: Q4_K_M
device: CPU
threads: 8
prompt tokens: 17
generation tokens: 128
diffusion steps: 128

load time: 1.21 s
step 1: 83.2 ms
step 2: 81.7 ms
...
generation: 10.4 s
peak RSS: 6.1 GB
```

This sample shows the format only. `run --verbose` prints the model, device, token counts, steps and load time, then the generation time, the per-step time with its GEMM, attention and LM head split, tokens per second and peak RSS; `--dump-step-stats` adds one line per step. The ms / step column of [Benchmarks](benchmarks.md) has the measured step time on the test laptop.

Debug mode MAY expose:

- selected token positions;
- confidence distributions;
- per-layer timings;
- kernel timings.

---

## 27. Error Handling

The runtime should fail with actionable errors.

Examples:

```text
ERROR: unsupported model architecture: diffusion_gemma (supported: llada, dream)
ERROR: unsupported GGUF tensor type: IQ2_XXS (tensor blk.0.attn_q.weight)
ERROR: required metadata key missing: tokenizer.ggml.model
ERROR: tensor shape mismatch for blk.0.attn_q.weight
ERROR: model requires 8.1 GB but available memory is 5.9 GB
```

The CPU path has no memory pre-check: a failed allocation prints `out of memory: could not allocate N bytes`. `--device gpu` prints `the model needs N MiB of GPU memory and M MiB are free`.

Avoid assertion-only failure paths in user-facing code.

---

## 28. Memory Strategy

For the initial CPU backend:

- memory-map GGUF;
- repack each projection matrix into a heap block at load and release its mapped pages (see [Architecture](architecture.md#memory)), keeping the embedding table mapped;
- allocate reusable activation buffers;
- avoid layer-by-layer heap churn;
- preallocate diffusion state;
- reuse attention workspaces.

Ideal high-level layout:

```text
mapped GGUF
    |
    +--> quantized weights

runtime arena
    |
    +--> hidden state
    +--> Q/K/V
    +--> attention workspace
    +--> MLP workspace
    +--> logits
    +--> diffusion scheduler state
```

---

## 29. Milestones

### M0: repository bootstrap

Deliverables:

- build system;
- CLI skeleton;
- CI;
- basic tests;
- coding conventions.

### M1: GGUF inspection

Deliverables:

```bash
mojo-dllm inspect model.gguf
```

Outputs:

- architecture;
- tensor count;
- tensor names;
- shapes;
- quantization formats;
- tokenizer metadata.

### M2: Q4_K_M primitives

Deliverables:

- parser;
- dequantization;
- reference matvec;
- numerical tests;
- microbenchmarks.

### M3: transformer primitives

Deliverables:

- RMSNorm;
- RoPE;
- softmax;
- attention;
- MLP;
- transformer layer tests.

### M4: LLaDA forward pass

Deliverables:

- model configuration;
- tensor mapping;
- embedding;
- transformer stack;
- logits;
- comparison against reference implementation.

### M5: diffusion generation

Deliverables:

- mask canvas;
- scheduler;
- confidence;
- sampling;
- iterative generation;
- complete text output.

At this milestone:

```bash
mojo-dllm run ...
```

must work.

### M6: CPU optimization

Deliverables:

- SIMD;
- multithreading;
- fused kernels;
- benchmark report.

### M7: NVIDIA backend

Deliverables:

- device allocation;
- GPU kernels;
- end-to-end GPU inference;
- consumer RTX benchmark.

### M8: second architecture

Deliverables:

- Dream support;
- model abstraction cleanup;
- generalized dLLM scheduler interface.

---

## 30. Initial Definition of Done

v0.1 is complete when all of the following are true:

This list records the 0.1 target. The changelog and the README status table record what shipped.

- [ ] LLaDA Q4_K_M GGUF loads directly in Mojo.
- [ ] No Python process is required during inference.
- [ ] Prompt tokenization works.
- [ ] Full transformer forward pass executes.
- [ ] Bidirectional attention is correct.
- [ ] Diffusion scheduler is implemented.
- [ ] The model generates readable text.
- [ ] Fixed-seed runs are deterministic.
- [ ] Tests cover quantization, attention and scheduler primitives.
- [ ] Peak memory fits within a reasonable consumer laptop RAM budget.
- [ ] Benchmark comparison with `diffuse-cpp` is published.
- [ ] Benchmark comparison with `llama.cpp` is published.
- [ ] README documents known differences and limitations.

---

## 31. Risks

### 31.1 Mojo ecosystem churn

Mojo APIs may still evolve quickly.

Mitigation:

- keep backend abstraction small;
- pin compiler/toolchain versions;
- isolate unstable APIs;
- use CI against a documented supported version.

### 31.2 Q4_K performance

Naive Q4_K dequantization may make CPU inference unusably slow.

Mitigation:

- optimize Q4_K early;
- benchmark it independently;
- borrow mathematical layout ideas from permissively licensed implementations while respecting licenses.

### 31.3 GGUF complexity

GGUF supports many metadata conventions and quantization formats.

Mitigation:

- support only the exact subset required by LLaDA initially;
- reject unsupported formats explicitly;
- expand only after v0.1.

### 31.4 Tokenizer mismatch

Small tokenizer differences can invalidate inference.

Mitigation:

- validate token IDs against a known reference tokenizer;
- add golden prompt/token test cases.

### 31.5 Floating-point divergence

Different kernel implementations may diverge slightly.

Mitigation:

- use tensor-level tolerances;
- compare logits statistically;
- separate numerical mismatch from algorithmic mismatch.

### 31.6 GPU portability may underperform

A single portable kernel may not match highly specialized vendor kernels.

Mitigation:

- measure rather than assume portability;
- allow compile-time specialization;
- permit small backend-specific sections if required;
- quantify shared vs backend-specific LOC.

---

## 32. Success Criteria

### Minimum success

A working pure-Mojo CPU implementation of LLaDA-8B Q4_K_M.

### Strong success

CPU performance within a reasonable factor of `diffuse-cpp` or `llama.cpp`, with a considerably smaller and more readable runtime.

### High-impact success

One shared Mojo implementation runs efficiently on:

- x86 CPU;
- NVIDIA GPU;
- Apple GPU;
- AMD GPU;

with limited backend-specific code.

### Research-quality success

The project demonstrates a useful dLLM-specific optimization, such as efficient inter-step reuse or portable block diffusion, that produces a measurable speedup while preserving output quality.

---

## 33. Project Positioning

Recommended one-line description:

> **Portable local inference for diffusion language models, written in Mojo.**

Alternative:

> **Run LLaDA and Dream locally from GGUF using a Mojo-native diffusion inference runtime.**

Avoid claims such as:

- "first diffusion inference engine";
- "first fast dLLM runtime";
- "novel diffusion caching";
- "faster than autoregressive LLMs";

unless validated by reproducible benchmarks and a fresh prior-art review.

A defensible early claim is:

> A standalone Mojo-native runtime for quantized local diffusion-language-model inference.

---

## 34. README Benchmark Target

The eventual repository front page should contain a table like:

| Runtime | Hardware | Model | Quant | Latency | Peak RAM | Runtime language |
|---|---|---|---|---:|---:|---|
| llama.cpp | CPU | LLaDA-8B | Q4_K_M | measured | measured | C/C++ |
| diffuse-cpp | CPU | LLaDA-8B | Q4_K_M | measured | measured | C++ |
| mojo-dllm | CPU | LLaDA-8B | Q4_K_M | measured | measured | Mojo |
| mojo-dllm | RTX 3050 | LLaDA-8B | Q4_K_M | measured | measured | Mojo |

No synthetic or estimated performance numbers should be published.

---

## 35. Immediate Implementation Order

The recommended sequence is:

```text
1. GGUF metadata reader
2. LLaDA tensor mapping
3. Q4_K_M block decoder
4. reference Q4_K matvec
5. RMSNorm
6. RoPE
7. softmax
8. bidirectional attention
9. MLP
10. single transformer layer
11. full transformer
12. tokenizer
13. diffusion canvas
14. confidence calculation
15. scheduler
16. sampling
17. full generation
18. correctness comparison
19. SIMD optimization
20. NVIDIA backend
```

This sequence minimizes the number of moving pieces during debugging.

---

## 36. First Engineering Spike

Before committing to the full runtime, implement a one-week feasibility spike:

### Deliverable A

```bash
mojo-dllm inspect llada.gguf
```

Correctly prints model metadata and tensor layout.

### Deliverable B

Load one Q4_K_M matrix and reproduce a reference matrix-vector multiplication within numerical tolerance.

### Deliverable C

Implement one complete LLaDA transformer layer and compare output tensors against a PyTorch or C++ reference.

If these three succeed cleanly, proceed with the full project.

If Q4_K_M or GGUF support proves disproportionately difficult because of current Mojo limitations, reconsider whether the first version should use Q8_0 before returning to Q4_K_M.

---

## 37. Core Principle

The project should optimize for one question:

> **Can Mojo provide a clean, performant, portable execution layer for diffusion language models on ordinary local hardware?**

Everything that does not help answer that question should remain outside the initial scope.
