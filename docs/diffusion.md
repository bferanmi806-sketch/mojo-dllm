# Diffusion decoding in five minutes

An autoregressive model such as Llama writes left to right. Each forward pass
produces one token, and an append-only KV cache makes the next pass cheap.

A masked diffusion model such as LLaDA writes the way you fill in a crossword.
It starts from a canvas of `[MASK]` tokens after the prompt and, at every step,
looks at the whole canvas at once, proposes a token for every blank, and
commits the few it is most confident about. The rest stay masked for the next
step.

```text
prompt: "Explain speculative decoding"           canvas (8 positions shown)

step 0   [M] [M]   [M]     [M]  [M]    [M] [M]     [M]
step 1   [M] [M]   [M]     [M]  [M]    [M] [M]     .
step 2   A   [M]   [M]     [M]  [M]    [M] [M]     .
step 3   A   draft [M]     [M]  [M]    [M] tokens  .
  ...
step 8   A   draft model   guesses     several     tokens .
```

Each step runs the full transformer over prompt plus canvas. Attention is
bidirectional: position 3 can read position 7, which a causal model forbids.

## The loop LLaDA uses

mojo-dllm implements the reference sampler from the LLaDA repository
(`generate.py`), with its defaults:

1. Split the generated region into blocks of `--block-length` positions and
   fill them left to right (semi-autoregressive decoding).
2. Give each block `steps / blocks` steps and spread its masked positions
   over them evenly: with 32 masks and 8 steps, 4 tokens per step.
3. At each step, for every masked position in the block, take the argmax
   token and its softmax probability.
4. Commit the k positions with the highest probability ("low-confidence
   remasking": the uncertain ones stay masked).

At temperature 0 the loop is deterministic. Above 0, LLaDA adds Gumbel noise
to the logits before the argmax; mojo-dllm does the same.

## Why the runtime looks different

- **No KV cache by default.** Any position can change between steps, so every
  step recomputes keys and values for the whole canvas. `--dual-cache` (LLaDA
  only) keeps a block-level cache and gives up exactness; see
  [Command line](cli.md). A step costs a full prompt pass, about 2 × 7 billion
  operations per token on the canvas in LLaDA-8B, plus 2 × 0.5 billion for each
  row whose logits are computed.
- **GEMMs, not GEMVs.** A step multiplies weight matrices by a hundred or more
  activation rows at once. The CPU kernel is built for that shape; see
  [The quantized GEMM](kernels.md).
- **Few rows need logits.** Only the masked positions the sampler can commit
  are candidates (the current block for LLaDA, every masked position for
  Dream), so mojo-dllm computes the output projection (126 464 columns for
  LLaDA, 152 064 for Dream) for those rows alone. The LLaDA benchmark ablation measures what that saves.
