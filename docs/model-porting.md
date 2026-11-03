# Porting a model

mojo-dllm supports the LLaDA and Dream architectures. Tested Q4_K_M checkpoints
are LLaDA-8B-Instruct, LLaDA-1.5, Dream-v0-Instruct-7B,
Dream-Coder-v0-Instruct-7B and DiffuCoder-7B-cpGRPO. The added checkpoints have
CPU smoke and [parity evidence](correctness.md); GPU evidence remains limited
to LLaDA-8B and Dream-v0. This checklist follows the path Dream took.

## 1. Read the metadata

Run `mojo-dllm inspect model.gguf` and note `general.architecture`, the
`<arch>.*` hyperparameters, `tokenizer.ggml.model` and `tokenizer.ggml.pre`,
the mask token id, and every tensor type. Dream differs from LLaDA in four
places that matter:

| | LLaDA-8B | Dream-7B |
|---|---|---|
| attention | 32 heads, MHA | 28 query heads, 4 KV heads (GQA) |
| QKV bias | none | yes (Qwen2 lineage) |
| logits | position i predicts token i | shifted: position i predicts token i+1 |
| sampler | low-confidence blocks | timestep schedule, entropy / margin variants |

### Metadata contract

`<arch>` is `llada` or `dream`. Required keys fail with a message naming the
missing key. This table describes `ModelConfig`; the tokenizer also needs
its own metadata.

| Key | Required/default | Interpretation |
|---|---|---|
| `general.architecture` | required | `llada` or `dream`; errors list the supported set |
| `general.name` | optional, `?` | display name |
| `<arch>.block_count` | required | transformer layer count |
| `<arch>.embedding_length` | required | hidden width |
| `<arch>.attention.head_count` | required | query head count |
| `<arch>.attention.head_count_kv` | optional, head count | KV head count; must divide query head count |
| `<arch>.feed_forward_length` | required | feed-forward width |
| `<arch>.context_length` | optional, 4096 | stored in config; currently does not constrain the canvas |
| `<arch>.attention.layer_norm_rms_epsilon` | required | RMSNorm epsilon |
| `<arch>.rope.freq_base` | optional, 10000 | RoPE frequency base |
| `<arch>.vocab_size` | optional, rows of `output.weight` | vocabulary size |
| `tokenizer.ggml.mask_token_id` | required | no fallback; tokenizer reads the same key |
| `diffusion.shift_logits` | optional, false for LLaDA / true for Dream | prediction-row shift |
| `<arch>.attention.causal` | ignored | attention stays bidirectional, as in llama.cpp for these architectures |
| `<arch>.attention.key_length` | optional, head dimension | other values are rejected |
| `<arch>.rope.dimension_count` | optional, head dimension | RoPE rotates the full head; other widths are rejected |
| `<arch>.rope.scaling.type` | optional, `none` | only `none` or `linear` with factor 1 are accepted |
| `<arch>.rope.scaling.factor` | optional, 1.0 | other factors are rejected; no scaled RoPE is implemented |

The head dimension is `embedding_length / head_count`, with exact divisibility
required. Other scaling metadata is not interpreted. These compatibility checks
do not add general validation of metadata ranges. Whether to constrain canvases
by `context_length` or reject `attention.causal = true` remains a separate policy
decision.

To convert a Hugging Face source with llama.cpp and quantize the result:

```bash
python convert_hf_to_gguf.py <hf-dir> --outtype f16 --outfile model-f16.gguf
llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

The converter accepts `f32`, `f16`, `bf16`, `q8_0`, `tq1_0`, `tq2_0`, and `auto`
as output types. Conversion success alone does not establish mojo-dllm support;
inspect the metadata and tensor types before running the model.

## 2. Load the weights

Extend `ModelConfig` and `Layer` in `models/transformer.mojo` (the architecture
check at the top of `ModelConfig.__init__` lists the supported names). Reuse
`PackedMatrix` for every projection. A tensor type it does not support stops
the load with `unsupported GGUF tensor type: <type> (tensor <name>)` when no
decoder exists. A decodable type without a matrix kernel reports
`tensor <name> is <type>: matrix kernels exist for Q4_K and Q6_K`.
Reference decoders accept F32, F16, BF16, Q5_0, Q8_0, Q4_K, Q5_K and Q6_K for
embedding rows and vectors; matrix kernels still require Q4_K or Q6_K.
Add the decoder and packed kernel with fixtures from `scripts/gen_fixtures.py`
before advertising a new matrix type as supported end to end.

Before loading a large conversion, compare its tensor shapes with the source:

```bash
uv run --only-group fixtures python scripts/check_weights.py \
  model-Q4_K_M.gguf <hf-dir>/model.safetensors.index.json
# For an unsharded source, pass model.safetensors instead of the index.
```

The checker reads safetensors headers only and reverses GGUF dimension order
before comparing. It reports mismatches, missing tensors on either side, and
unmapped source names; dtypes are shown for reference, not compared. Use
`--ignore 'rotary.*inv_freq'` for source buffers intentionally omitted during
conversion; `ignored N` counts these source entries, and their mapped GGUF
counterparts are also omitted. Index shard names must be bare `.safetensors`
filenames resolving inside the index directory. The checker accepts 1–1024
blocks to bound name-map construction. Malformed inputs return 2, including
JSON nesting errors. An output tensor absent from a tied source is still reported as
missing, with an explanation. Exit 0 means at least one tensor was compared and
all comparisons passed, 1 means findings, and 2 means no comparison or invalid
input. Shape agreement does not establish matching weights or logits.

## 3. Prove the forward pass

1. Extend `scripts/gen_fixtures.py` with a tiny model of the new layout and
   its numpy reference logits. Write the test first and watch it fail.
2. On the real model, compare logits with llama.cpp through
   `tools/ref/llada_ref.cpp` (`logits` mode) and `bench/parity.py`.

## 4. Tokenizer

If `tokenizer.ggml.pre` is new, add its pre-tokenizer to
`tokenizer/bpe.mojo` and golden cases to `scripts/gen_tokenizer_fixture.py`.
llama.cpp's regex for each pre-tokenizer is in `src/llama-vocab.cpp`.

## 5. Sampler

Add the model's reference sampler next to LLaDA's in
`diffusion/sampler.mojo`, with unit tests for its schedule.
