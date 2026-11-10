# Porting a model

mojo-dllm runs the LLaDA and Dream architectures, and reads the metadata of
`diffusion-gemma` (see below). Tested Q4_K_M checkpoints
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
| `<arch>.context_length` | optional, 4096 | the longest canvas (prompt plus `--max-tokens`) `run` accepts |
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
required. `block_count`, `embedding_length`, `head_count`, `head_count_kv`,
`feed_forward_length` and `vocab_size` must be positive, and the epsilon and
the RoPE base must be finite and above zero; a value outside those ranges
stops the load with `<key> must be positive, got N` or `<key> must be finite and
above zero, got X`. Other scaling metadata is not interpreted, and
`context_length` is not range checked. `run` refuses a canvas longer than it;
whether to reject `attention.causal = true` remains a separate policy decision.

### DiffusionGemma metadata

A `diffusion-gemma` file is read into a `ModelConfig`, and `mojo-dllm inspect`
prints the result, or the refusal and its key. `DiffusionLM` and the GPU model
stop on it with `diffusion-gemma does not run on this model`. `DiffusionGemma`
runs it on the CPU through `logits`, and `tokenize` prints the ids of its chat
prompt; `run` still waits for the model to be wired into the generation loop
(#277; the forward pass is #64 and the denoiser #65), so a clean `inspect`
does not mean `run` works. The names come from the unmerged
ggml-org/llama.cpp#24423 and appear once each, as the `GEMMA_*` constants at the
top of `models/transformer.mojo`. Keys carry the `diffusion-gemma.` prefix
except `diffusion.canvas_length` and `tokenizer.ggml.mask_token_id`.

| Key | Required/default | Interpretation |
|---|---|---|
| `block_count`, `attention.head_count`, `attention.key_length`, `attention.key_length_swa`, `attention.sliding_window`, `expert_count`, `expert_feed_forward_length`, `diffusion.canvas_length` | required, positive | layers, query heads, head size of global and of sliding layers, window, experts, expert width, canvas tokens |
| `embedding_length`, `feed_forward_length` | required | hidden width, dense feed-forward width |
| `attention.head_count_kv` | required, i32 array with one entry per block | K/V heads of each layer; one value per kind of layer, a divisor of `attention.head_count` |
| `attention.sliding_window_pattern` | required, bool array with one entry per block | `true` marks a sliding layer, `false` a global one |
| `expert_used_count` | required, from 1 to `expert_count` | experts per token |
| `rope.freq_base`, `rope.freq_base_swa` | required | RoPE base of global and of sliding layers |
| `attention.layer_norm_rms_epsilon`, `tokenizer.ggml.mask_token_id` | required | RMSNorm epsilon, mask token |
| `attention.value_length`, `attention.value_length_swa`, `rope.dimension_count`, `rope.dimension_count_swa` | required, equal to the key length of their kind | other widths are refused |
| `context_length` | optional, 4096 | stored in config; `run` does not take this architecture yet |
| `final_logit_softcapping` | optional, 30 when absent (llama.cpp's default); 0 turns it off | a negative value is refused |
| `attention.shared_kv_layers`, `embedding_length_per_layer_input` | optional, 0 | any other value is refused: no forward pass handles it |
| `attention.causal` | optional, false | true is refused; the canvas attends in both directions |

The config keeps the global layers' values in `n_kv_heads`, `head_dim` and
`theta`, and the sliding layers' in the `_swa` fields. `kv_heads(i)`,
`head_size(i)` and `rope_base(i)` pick the right one for layer `i`. The
vocabulary is the row count of `token_embd.weight`, because the file has no
`output.weight` (the logits are tied to the embedding).

The tensor directory has to agree with the flags and keys, and a mismatch names
the tensor and the keys that set its shape:

- `blk.N.attn_v.weight` exists on a sliding layer and on no global one (a global
  layer reuses K as V).
- `blk.N.attn_k.weight` is `[embedding_length, head_count_kv * key_length]` of
  that layer, with `key_length_swa` on a sliding layer.
- `blk.N.ffn_gate_up_exps.weight` is `[embedding_length, 2 * expert_feed_forward_length, expert_count]`
  and `blk.N.ffn_down_exps.weight` is `[expert_feed_forward_length, embedding_length, expert_count]`.
- `rope_freqs.weight` has `key_length / 2` entries when any layer is global, and
  `token_embd.weight` is `embedding_length` wide.

For LLaDA and Dream the same kind of check covers the two vocabulary-sized
tensors: `token_embd.weight` and, when the file has it, `output.weight` must be
`[embedding_length, vocabulary]`, where the vocabulary is `<arch>.vocab_size` or
the rows of `output.weight`. `ModelConfig` raises `tensor shape mismatch for
<name>: [..], expected [..]` before any loader runs, on the CPU and on the GPU.

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
filenames resolving inside the index directory. The checker accepts 1 to 1024
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
llama.cpp's regex for each pre-tokenizer is in `src/llama-vocab.cpp`. A model
whose `tokenizer.ggml.model` is not `gpt2` needs its own module, as `gemma4`
has in `tokenizer/gemma4.mojo`; `Tokenizer` picks it from that key.

## 5. Sampler

Add the model's reference sampler next to LLaDA's in
`diffusion/sampler.mojo`, with unit tests for its schedule.
