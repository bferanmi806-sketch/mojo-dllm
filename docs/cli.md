# Command line

```text
mojo-dllm run      --model PATH --prompt TEXT [options]
mojo-dllm tokenize --model PATH --prompt TEXT [--no-chat]
mojo-dllm inspect  PATH
mojo-dllm logits   --model PATH --tokens IDS --rows POSITIONS --out FILE [--threads N] [--device D]
                   [--path unified|cached] [--prev FILE --prev-temp-inv X]
```

This page describes the binary, `build/mojo-dllm`. The `./mojo-dllm run` launcher
takes the prompt as its first word, accepts `-m` for `--model`, defaults
`--device` to `auto` and passes the other flags on to the binary; see the
[Quick start](quickstart.md).

`mojo-dllm COMMAND --help` (or `-h`) prints that command's usage and exits 0.
Help wins over every other problem in the arguments, wherever it appears, except
as the value of a flag (`--prompt --help` sets the prompt to `--help`).
`mojo-dllm`, `mojo-dllm help` and `mojo-dllm --help` print the usage of every
command. Each command declares its flags: a flag the command does not declare
stops the run (see [Errors](#errors)).

`mojo-dllm --version` prints one line, `mojo-dllm` and the version, and exits
0. The same version is the `"version"` key of `run --json`.

## `run`

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | required | LLaDA or Dream GGUF file |
| `--prompt TEXT` | required | the user message |
| `--max-tokens N` | 128 | tokens to generate (the masked canvas). The prompt tokens plus N must not exceed the file's `<arch>.context_length` (4096 when the key is absent); the check runs after the header is read and before any weight is mapped |
| `--steps N` | `--max-tokens` | denoising steps; a multiple of the block count |
| `--block-length N` | 32, or `--max-tokens` when that is smaller | LLaDA's semi-autoregressive block size; must divide `--max-tokens`. Dream ignores it |
| `--temperature T` | 0 | LLaDA: 0 is greedy; above 0 adds Gumbel noise as LLaDA does. Dream supports only 0 and stops with an error above it |
| `--remasking S` | `low_confidence` | LLaDA: `low_confidence` or `random`. Dream: `entropy` (the default), `maskgit_plus`, `topk_margin` |
| `--threshold P` | 0 (off) | A number from 0 to 1. LLaDA: each step also commits every masked token whose confidence is at least P, so easy blocks finish in fewer steps (Fast-dLLM's parallel decoding). A step still commits at least what the schedule says. Not exact: see [Benchmarks](benchmarks.md#steps-and-quality) for its measured effect on accuracy |
| `--dual-cache` | off | LLaDA: cache every position's keys and values and, inside a block, recompute only that block's rows (Fast-dLLM's DualCache). The cache is refreshed at the first step of each block. Not exact: the prompt and later blocks see keys from the block's first step; see [Benchmarks](benchmarks.md#steps-and-quality) for its measured effect on accuracy. Cannot combine with `--full-logits` |
| `--seed N` | 42 | seeds this program's PRNG (it does not reproduce a PyTorch run). It fixes DiffusionGemma's random canvas and draws as well |
| `--eb-steps N` | 48 | DiffusionGemma: most denoising steps, at least 1 (use it instead of `--steps`, which that model refuses) (no effect yet, see the paragraph after the table) |
| `--eb-t-min T` | 0.4 | DiffusionGemma: the temperature the schedule approaches, 0 or more; the last of `--eb-steps` steps runs at `T + (t-max - T) / steps` (no effect yet, see the paragraph after the table) |
| `--eb-t-max T` | 0.8 | DiffusionGemma: the temperature of the first step, above 0; it falls linearly from there (no effect yet, see the paragraph after the table) |
| `--eb-bound B` | 0.1 | DiffusionGemma: a position is accepted while the entropies of the lower-entropy positions before it sum to at most B, which must be at least 0 (no effect yet, see the paragraph after the table) |
| `--eb-stability N` | 1 | DiffusionGemma: steps the argmax canvas must hold before the run may stop; 0 or more, and 0 lets the first step stop it (no effect yet, see the paragraph after the table) |
| `--eb-confidence C` | 0.005 | DiffusionGemma: the run may stop once the mean entropy over the canvas is below C, which must be at least 0 (no effect yet, see the paragraph after the table) |
| `--threads N` | all logical CPUs | worker threads of every parallel CPU loop, including the activation quantization and the weight repack at load (with `--device gpu`, only the sampler uses them) |
| `--device D` | `cpu` | `cpu`, or `gpu` for an NVIDIA GPU with compute capability 8.0 or newer |
| `--no-chat` | off | skip the chat template |
| `--full-logits` | off | LLaDA: compute logits for every position, as a generic runtime would (for benchmarking). Dream refuses it |
| `--no-early-stop` | off | run every step even after the answer is final (the benchmark and parity harnesses use it) |
| `--visual` | off | show the canvas: masked positions as teal blocks, tokens committed in that step in bold orange when colour is enabled |
| `--visual-interval N` | 1 | redraw every N steps and at the final step; requires `--visual` and N at least 1 |
| `--verbose` | off | model, device and load time before the text; step timings and peak memory after it |
| `--dump-step-stats` | off | one line per denoising step |
| `--json` | off | one JSON object with timings, tokens and text. `peak_rss_bytes` is `null` when the operating system could not report it, never 0. `remasking` names the strategy the run used (for Dream, `entropy` when the flag says `low_confidence`); `block_length` and `full_logits` appear for LLaDA only, because Dream reads neither. `steps_taken` counts the denoising steps run and `stop_reason` says why the loop ended: `complete`, `answer_fixed` (early stop), or for DiffusionGemma, once `run` accepts it (#277), `converged` or `max_steps` |

The flags `--threshold`, `--dual-cache` and `--full-logits` are LLaDA's; a Dream run refuses them by name before the model loads (`--threshold applies to llada only, not dream`). Dream also reads no `--block-length`, but still accepts it, because the benchmark and parity scripts pass it for every architecture.

The printed answer ends at the first end-of-text or end-of-turn token in the
canvas. Generation stops as soon as that answer is final: once the first
position after the prompt that is still masked or holds such a token holds
such a token, no later step can change the printed text, because steps only
fill masked positions. The result is the same text with fewer steps, and
`--json` reports `"stopped_early": true`. `--no-early-stop` runs every step.

The `--eb-*` flags belong to DiffusionGemma and every other model refuses them by name. The parser accepts them, but `run` stops on a `diffusion-gemma` file until the model is wired into the generation loop (see [Introduction](introduction.md)). It refuses the file once its header is read, before the tokenizer is built, with ``diffusion-gemma does not run on this model: `run` waits for the model to be wired into the generation loop (#277), so use `logits` (CPU) until then``, so the `--eb-*` flags have no effect yet; `run --help` says so. Only `logits` runs that architecture. The forward pass (#64) and the entropy-bounded loop (#65) are written; #277 tracks joining them. Once it lands, DiffusionGemma's `--json` reports `"stopped_early": true` when the stop rule ends the run before `--eb-steps` steps.

### Visual output

The final canvas always appears, including when generation stops early between
redraws. Its step number is the last observed step; any remaining masks stay
visible. A final frame that was already drawn is not repeated.

| stdout | `NO_COLOR` | colour codes | clear + home (`ESC[2J ESC[H`) |
|---|---|---|---|
| terminal | unset or empty | yes | yes |
| terminal | non-empty | no | yes |
| not a terminal | any | no | no; frames separated by one blank line |

On a terminal, `--visual` leaves the last canvas frame on screen and does not
print the answer a second time. Without `--visual`, or when stdout is not a
terminal, `run` prints the answer after generation.

Without colour, newly committed tokens have no highlight. Redirected frames
contain no ANSI escapes. `--visual`, `--verbose` and `--dump-step-stats` print to
stdout before the JSON object, so each of them is refused together with `--json`:
the command stops with the usage error `--json cannot be combined with --visual`
(or `--verbose`, `--dump-step-stats`) and exit status 2, before the model loads. With
`--json`, stdout holds the one JSON object and nothing else.

The checks that need only the flags run before the model file is opened, and each exits with status 2 and names the flag:
`--threshold` outside 0 to 1 or not finite, `--device` other than `cpu` or `gpu`, `--remasking` outside the five names
above, and `--threshold` above 0 with `--remasking random`. LLaDA also needs a `--block-length` that divides
`--max-tokens` and a `--steps` that is a multiple of the number of blocks (Dream reads neither flag and accepts any
value); that check depends on the architecture, which only the file's header says, so it runs once the header is read,
still before any weight is mapped, and exits with status 1, as does the check of the default block length (32, or
`--max-tokens` when smaller): `flag --max-tokens 100 is not a multiple of the default block
length 32`. The same header check refuses a `--remasking` value that the file's architecture does not take (`entropy`,
`maskgit_plus` and `topk_margin` for LLaDA, `random` for Dream).

Invalid `--visual-interval` values and use without `--visual` exit with status
1 before model loading. The messages name `--visual-interval`: `must be at
least 1`, `needs --visual`, or `needs an integer`.

## `tokenize`

Prints the token ids of the chat-formatted prompt, the same ids `run` feeds the
model. Useful to compare against `llama-tokenize`. For a `diffusion-gemma` file
the prompt is the file's chat template: `<bos>`, a user turn and an open model
turn, which gives one BOS token (id 2). It reads the tokenizer arrays of the
file and none of its weights, so it runs on a file too big to load.
Only `llada`, `dream` and `diffusion-gemma` files have a chat template here: for another
architecture the command stops with `no chat template for architecture 'qwen2' (llada, dream
and diffusion-gemma have one); pass --no-chat to tokenize the text as it is`, and `--no-chat`
tokenizes the text unchanged. `run` applies the same rule to its prompt.

## `inspect`

Prints GGUF metadata, a count per tensor type, and every tensor's type and
shape. It needs no supported architecture, so it works on any GGUF file. For
`diffusion-gemma` it also prints a `config:` block with what the reader took
from the file (layer kinds, K/V heads and head size per kind, experts, canvas
length), or `config: refused:` and the reason, such as the key that is wrong.
For `llada`, `dream` and `diffusion-gemma` it also lists each metadata key that
the file lacks and the reader replaced by a default, one per line under
`defaults (key absent from the file):`, as `llada.rope.freq_base = 10000
(default, key absent)`. A file that sets every such key prints no such block.
`run --verbose` prints the same block after the device line. The keys are the
KV head count, the context length, the rope base and the logit shift for LLaDA
and Dream, and the context length and the final logit softcap for
DiffusionGemma.

## `logits`

Runs one forward pass over the given token ids and writes f32 logits for the
requested positions to a file, `[positions, vocab]`, row major. The parity
tests use it; see [Correctness](correctness.md).

The file is written to `FILE.tmp`, checked there to hold exactly `positions * vocab * 4` bytes, and
only then renamed to `FILE`; any failure (a full disk, a quota, a missing or unwritable directory)
removes the temporary file, prints `ERROR: cannot write FILE: ...` with the system's reason, exits with
status 1 and leaves any earlier `FILE` as it was. `wrote N rows` appears only after the rename.

The rename replaces a symlink named `FILE` (the link goes and its target stays as it was) and a read-only
`FILE`. When `FILE` already exists and is not a regular file (`/dev/null`, `/dev/stdout`, a pipe), the bytes
are written to it directly: there is no name to rename to and no size to check.

On a `diffusion-gemma` file the tokens are the prompt followed by the canvas,
whose length the file declares (`diffusion.canvas_length`, 256 in the real
model). The model scores only the canvas, so `--rows` must list exactly those
positions, the last canvas-length ones, and the file holds `[canvas, vocab]`.
It runs on the CPU only; `--device gpu` is refused. Three flags apply to this
architecture alone and are refused on the others:

- `--path unified` (the default) runs the prompt and the canvas in one pass.
  `--path cached` prefills the prompt, keeps its keys and values, and steps the
  canvas against them. The two agree bit for bit.
- `--prev FILE` names the previous denoising step's canvas logits, `[canvas,
  vocab]` f32 as this command writes them, and turns self-conditioning on.
  `--prev-temp-inv X`, which must come with it, is 1 divided by the temperature
  of that previous step (1.25 for the first step's 0.8). It must be above 0 and
  its product with the model's final logit soft cap must not exceed 1e8 (about
  3e6 with the cap of 30), because the softmax turns the logits to NaN beyond
  that, and `--path` must be `unified` or `cached`; the path is checked before
  any model is read, the bound when the pass starts.

`--rows` and the size of the `--prev` file are checked against the canvas length
that the file's header declares, before a weight is mapped, so a wrong `--rows`
costs no model load.

## Errors

An error names the thing that is wrong. Usage errors exit with status 2 and
change nothing; every other error exits with status 1. Every `ERROR:` message goes to
stderr, so stdout stays free for the answer or the JSON object; help text and
`--version` go to stdout.

| Message | Cause and next step |
|---|---|
| `ERROR: unknown flag --stpes for 'run' (did you mean --steps?)` | The command does not declare the flag. The parenthesis appears when a declared flag is at most 2 edits away. The next line is `see: mojo-dllm run --help` |
| `ERROR: missing required flag --model for 'run'` | A required flag is absent. The usage of the command follows |
| `ERROR: missing model path for 'inspect'` | `inspect` takes the path as an argument. The usage follows |
| `ERROR: unknown command: bogus` | The first argument is not a command. The next line is `see: mojo-dllm --help` |
| `ERROR: unexpected argument 'x' for 'run'` | `run`, `tokenize` and `logits` take flags only, and `inspect` takes one path |
| `ERROR: unknown flag --model=x for 'run'` | Flags take their value as the next argument. The next line says `flags take their value as the next argument: --model x` |
| `ERROR: flag --prompt needs a value` | A value flag is the last argument |
| `ERROR: flag --steps needs an integer, got 'abc'` | A numeric flag holds text, or a number too large for 64 bits. `--temperature` needs a finite number and `--tokens` and `--rows` need comma-separated integers |
| `ERROR: flag --prompt needs a value, got the flag --verbose` | A value flag is followed by the name of a declared flag, which is almost always a forgotten value. `--help` stays a legal value |
| `ERROR: flag --steps is given twice` | A value flag appears twice; the command no longer keeps the last silently. A boolean flag repeated is harmless |
| `ERROR: flag --rows needs comma-separated integers, got '0,,2'` | An empty entry in `--tokens` or `--rows` (a trailing comma stays legal) |
| `ERROR: flag --steps must be at least 1, got 0` | `--max-tokens`, `--steps`, `--block-length`, `--threads` and `--eb-steps` need at least 1, and `--eb-stability` at least 0. `--temperature`, `--eb-t-min`, `--eb-bound` and `--eb-confidence` need a finite number of at least 0, `--eb-t-max` and `--prev-temp-inv` one above 0 (not only finite), and `--threshold` a number from 0 to 1. The check runs before the model is mapped |
| `ERROR: prompt 10 + --max-tokens 4096 = 4106 exceeds the context length 4096 (llada.context_length)` | The canvas is longer than the file's context length. Lower `--max-tokens`. Exit status 1, before any weight is mapped |
| `ERROR: cannot open model file PATH: No such file or directory` | The path is wrong. The reason is the operating system's. The next line is `to download a model, run: ./mojo-dllm pull` |
| `ERROR: cannot open model file PATH: Permission denied` | The open failed for a reason other than a missing file, such as `Not a directory`. The operating system names the reason. No download hint follows, because `pull` would not fix it |
| `ERROR: file is empty: PATH` | The model file has no bytes. Download it again |
| `ERROR: cannot get the size of PATH: Illegal seek` | The path opens but is not a regular file (a pipe or a FIFO). The operating system names the reason |
| `ERROR: mmap failed for PATH: No such device` | The file cannot be mapped (a device or a pseudo file). The operating system names the reason; `Cannot allocate memory` means the address space is exhausted |
| `ERROR: is a directory: PATH` | The model path names a directory |
| `ERROR: flag --prompt is not valid UTF-8: invalid byte at offset N` | `run` and `tokenize` refuse a prompt that is not UTF-8 (a Latin-1 text, for example) before the model loads and exit with status 2. N is the byte offset of the first bad byte. Save the text as UTF-8 |
| `ERROR: PATH is not a GGUF file (bad magic).` | A failed download leaves an HTML error page with a `.gguf` name. Delete the file and run `./mojo-dllm pull` again |

The remaining errors exit with status 1 and come from the model file itself:

```text
ERROR: unsupported GGUF tensor type: IQ2_XXS (tensor blk.0.attn_q.weight)
ERROR: tensor blk.0.attn_q.weight is Q8_0: matrix kernels exist for Q4_K and Q6_K
ERROR: unsupported model architecture: qwen2 (runs: llada, dream; read only: diffusion-gemma)
ERROR: required metadata key missing: tokenizer.ggml.model
ERROR: tensor blk.0.attn_q.weight extends past the end of the file
ERROR: tensor blk.0.attn_q.weight has offset 16, which is not a multiple of the alignment 32
ERROR: tensor blk.0.attn_q.weight is Q8_0 with 33 elements per row, which is not a multiple of its block size 32
ERROR: tensor blk.0.attn_q.weight is listed twice (entries 3 and 9)
ERROR: metadata key general.name is listed twice (entries 1 and 5)
ERROR: tensor shape mismatch for blk.0.attn_q.weight: [4096, 2048], expected [4096, 4096]
ERROR: non-finite logits at canvas position 3, step 1
ERROR: no GPU found: the GPU backend needs an NVIDIA GPU with compute capability 8.0 or newer (run with --device cpu)
```

`non-finite logits` comes from the LLaDA and Dream loops, on the CPU and the GPU: a canvas row whose softmax is not a
number (a NaN or `+inf` logit, or every logit `-inf`) stops the run instead of committing token 0 there. The position
counts from the start of the canvas and the step from 1. One `-inf` logit among finite ones is a token of probability 0
and passes for LLaDA; Dream's default `entropy` confidence has no value for such a row, so Dream refuses it too.

With `--device gpu`, setting `MOJO_DLLM_GPU_PROFILE=1` makes `--verbose` split
a step into GEMM, attention and LM head time. It synchronizes the GPU after
each kernel group to measure them, so leave it off when you benchmark.
