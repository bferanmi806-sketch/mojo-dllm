<div class="hero">
  <img src="images/logo.svg" alt="mojo-dllm logo: masked tokens resolving into a flame">
  <div class="tagline">Local inference for diffusion language models, written in Mojo.</div>
</div>

# Introduction

mojo-dllm runs two families of diffusion language models, LLaDA and Dream, from
quantized GGUF files on your own CPU or on an NVIDIA GPU. LLaDA-8B and Dream-7B
are measured; LLaDA-1.5, Dream-Coder and DiffuCoder are tested on the CPU (see
[Quick start](quickstart.md#3-other-models)). The whole
inference path is Mojo: the GGUF reader, the tokenizer, the quantized matrix
multiply, the GPU kernels, the transformer and the denoising loop. The
`build/mojo-dllm` binary needs no Python at runtime, and the GPU is optional.
The `./mojo-dllm` launcher that builds and starts it is a Python script run
by uv.

A diffusion language model writes differently from the models you know. An
autoregressive model appends one token at a time. LLaDA starts from a canvas
of `[MASK]` tokens and fills it in over a fixed number of steps, committing the
tokens it is most sure about first. [Diffusion decoding in five
minutes](diffusion.md) walks through it with a picture.

## What works today

- `mojo-dllm run` generates text from LLaDA-8B-Instruct and
  Dream-v0-Instruct-7B (Q4_K_M) on x86 CPUs with AVX2, using AVX-VNNI when
  present.
- `--device gpu` runs both models on NVIDIA GPUs with compute capability 8.0
  or newer. It has been measured on an RTX 3050 Laptop GPU only.
- The LLaDA and Dream tokenizers match llama.cpp id for id on a 36-case
  multilingual corpus, and DiffusionGemma's `gemma4` tokenizer on 54 cases.
- Logits match llama.cpp's implementation of both models; the measured
  agreement is in [Correctness](correctness.md).
- [Benchmarks](benchmarks.md) against llama.cpp and diffuse-cpp on the same
  machine, with every measurement table generated from a result file in the
  repository.

## What does not work yet

- The GPU path is measured on one laptop part, an RTX 3050 held to 30 W,
  against llama.cpp's CUDA build; [Benchmarks](benchmarks.md) has the numbers.
  It supports NVIDIA only.
- Two architectures run: LLaDA and Dream. A `diffusion-gemma` file is read,
  `inspect` prints its config, `tokenize` prints the ids of its chat prompt,
  and `logits` runs it on the CPU: one unified pass over the prompt and the
  canvas, checked against a numpy forward pass on a tiny fixture (#64). The
  entropy-bounded denoiser is written too (#65), but `run` stops on this file
  with `diffusion-gemma does not run on this model` until the model is wired
  into the generation loop (#277), on the CPU and the GPU.
  Other architectures stop with
  `unsupported model architecture: <name> (runs: llada, dream; read only: diffusion-gemma)`.
- One quantization mix: Q4_K and Q6_K projection matrices, as in a Q4_K_M
  file, with an embedding table and norm vectors that the reference decoders
  read (F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K). A matrix of any other type
  stops the load with an error that names it. DiffusionGemma's `logits` also
  reads Q5_0 and Q8_0 matrices for the down projections.

## Where to go next

- Run it: [Quick start](quickstart.md).
- Read the numbers: [Benchmarks](benchmarks.md).
- Read the code: [Architecture](architecture.md), then the source under
  `src/mojo_dllm/`.
