"""Masked-diffusion transformers: LLaDA and Dream, and the DiffusionGemma config.

Both are pre-norm decoder stacks run with bidirectional attention:

    h = x + Wo · attn(rope(Wq·n + bq), rope(Wk·n + bk), Wv·n + bv)    n = rmsnorm(x) * attn_norm
    x = h + Wd · (silu(Wg·m) * (Wu·m))                                  m = rmsnorm(h) * ffn_norm

followed by a final RMSNorm and an untied output projection. The two models
differ only in details this file reads from the GGUF:

| | LLaDA-8B (`llada`) | Dream-7B (`dream`) |
|---|---|---|
| attention | 32 heads, MHA | 28 query heads over 4 KV heads (GQA) |
| Q/K/V bias | none | present (Qwen2 lineage) |
| RoPE pairs | adjacent (llama.cpp NORM) | half-split (NEOX) |
| logits | row i scores token i | row i scores token i+1 (`diffusion.shift_logits`) |

Two departures from a generic forward pass, both exact (they change cost, not
results):

- `forward` takes the list of positions whose logits the caller needs. A
  denoising step samples only some positions, so the last layer's FFN and the
  output projection run on those rows alone.
- Every projection reads weights repacked at load time (`kernels/packed.mojo`);
  the GGUF stays memory-mapped for the embedding table, and norms and biases
  are decoded into f32 arrays at load.

`diffusion-gemma` files are read into a `ModelConfig` and checked against their
tensor directory, but `DiffusionLM` refuses to run them: there is no forward pass
for that architecture yet.
"""

from std.time import perf_counter_ns

from mojo_dllm.formats.gguf import GGUFFile, TensorInfo
from mojo_dllm.kernels.packed import PackedMatrix, pgemm
from mojo_dllm.kernels.qgemm import QActs, quantize_acts
from mojo_dllm.kernels.ops import (
    rmsnorm,
    rope_table,
    rope_apply,
    attention,
    silu_mul,
    add_inplace,
    add_bias,
)
from mojo_dllm.quant.kquants import (
    dequant_row,
    check_matrix_type,
    check_embedding_type,
)
from mojo_dllm.sys.mem import F32Ptr, Floats


# DiffusionGemma's metadata keys and tensor names. They come from the unmerged
# ggml-org/llama.cpp#24423 (gguf-py/gguf/constants.py, src/llama-arch.cpp) and
# are spelled here and nowhere else, so a rename when it merges is a one-line
# change. Keys are relative to `<arch>.`, except GEMMA_CANVAS.
comptime GEMMA_ARCH = "diffusion-gemma"
comptime GEMMA_BLOCK_COUNT = "block_count"
comptime GEMMA_CONTEXT_LENGTH = "context_length"
comptime GEMMA_EMBEDDING_LENGTH = "embedding_length"
comptime GEMMA_FFN_LENGTH = "feed_forward_length"
comptime GEMMA_HEAD_COUNT = "attention.head_count"
comptime GEMMA_HEAD_COUNT_KV = "attention.head_count_kv"
comptime GEMMA_KEY_LENGTH = "attention.key_length"
comptime GEMMA_KEY_LENGTH_SWA = "attention.key_length_swa"
comptime GEMMA_VALUE_LENGTH = "attention.value_length"
comptime GEMMA_VALUE_LENGTH_SWA = "attention.value_length_swa"
comptime GEMMA_WINDOW = "attention.sliding_window"
comptime GEMMA_PATTERN = "attention.sliding_window_pattern"
comptime GEMMA_CAUSAL = "attention.causal"
comptime GEMMA_SHARED_KV = "attention.shared_kv_layers"
comptime GEMMA_LAYER_INPUT = "embedding_length_per_layer_input"
comptime GEMMA_EPSILON = "attention.layer_norm_rms_epsilon"
comptime GEMMA_ROPE_BASE = "rope.freq_base"
comptime GEMMA_ROPE_BASE_SWA = "rope.freq_base_swa"
comptime GEMMA_ROPE_DIMS = "rope.dimension_count"
comptime GEMMA_ROPE_DIMS_SWA = "rope.dimension_count_swa"
comptime GEMMA_EXPERT_COUNT = "expert_count"
comptime GEMMA_EXPERT_USED = "expert_used_count"
comptime GEMMA_EXPERT_FFN = "expert_feed_forward_length"
comptime GEMMA_SOFTCAP = "final_logit_softcapping"
comptime GEMMA_CANVAS = "diffusion.canvas_length"
comptime GEMMA_TOKEN_EMBD = "token_embd.weight"
comptime GEMMA_ROPE_FREQS = "rope_freqs.weight"
comptime GEMMA_ATTN_K = "attn_k.weight"
comptime GEMMA_ATTN_V = "attn_v.weight"
comptime GEMMA_EXPERTS_UP = "ffn_gate_up_exps.weight"
comptime GEMMA_EXPERTS_DOWN = "ffn_down_exps.weight"


@fieldwise_init
struct ModelConfig(Copyable, ImplicitlyCopyable, Movable):
    """The hyperparameters of a GGUF model.

    LLaDA and Dream use one attention shape in every layer. `diffusion-gemma`
    has two, so for it `n_kv_heads`, `head_dim` and `theta` hold the global
    layers' values (the GGUF keys without a suffix) and the `_swa` fields the
    sliding layers'. Read them per layer through `kv_heads`, `head_size` and
    `rope_base`: a sliding layer that takes the global head size computes
    garbage without an error. `head_dim` is not `hidden / n_heads` there.
    The fields below `shift_logits` are 0 or empty for the other architectures.
    """

    var arch: String
    var n_layers: Int
    var hidden: Int
    var n_heads: Int
    var n_kv_heads: Int
    var head_dim: Int
    var ffn: Int
    var vocab: Int
    var ctx: Int
    var eps: Float32
    var theta: Float32
    var mask_id: Int
    var rope_neox: Bool
    var shift_logits: Bool
    var layer_kinds: String
    """One byte per layer, `s` for sliding-window attention and `g` for global."""
    var n_kv_heads_swa: Int
    var head_dim_swa: Int
    var theta_swa: Float32
    var sliding_window: Int
    var n_experts: Int
    var n_experts_used: Int
    var expert_ffn: Int
    var softcap: Float32
    """`final_logit_softcapping`: logits become softcap * tanh(logit / softcap); 0 is off."""
    var canvas: Int
    """`diffusion.canvas_length`: the tokens the model denoises after the prompt."""

    def __init__(out self, g: GGUFFile) raises:
        var arch = g.architecture()
        if arch == GEMMA_ARCH:
            self = _read_diffusion_gemma(g)
            return
        self.arch = arch
        if self.arch != "llada" and self.arch != "dream":
            raise Error(
                "unsupported model architecture: "
                + self.arch
                + " (runs: llada, dream; read only: "
                + GEMMA_ARCH
                + ")"
            )
        self.layer_kinds = String("")
        self.n_kv_heads_swa = 0
        self.head_dim_swa = 0
        self.theta_swa = 0.0
        self.sliding_window = 0
        self.n_experts = 0
        self.n_experts_used = 0
        self.expert_ffn = 0
        self.softcap = 0.0
        self.canvas = 0
        var a = self.arch + "."
        self.n_layers = g.get_int(a + "block_count")
        self.hidden = g.get_int(a + "embedding_length")
        self.n_heads = g.get_int(a + "attention.head_count")
        self.n_kv_heads = g.get_int_or(
            a + "attention.head_count_kv", self.n_heads
        )
        self.head_dim = self.hidden // self.n_heads
        self.ffn = g.get_int(a + "feed_forward_length")
        self.ctx = g.get_int_or(a + "context_length", 4096)
        self.eps = g.get_f32(a + "attention.layer_norm_rms_epsilon")
        self.theta = g.get_f32_or(a + "rope.freq_base", 10000.0)
        self.mask_id = g.get_int("tokenizer.ggml.mask_token_id")
        if g.has_key(a + "vocab_size"):
            self.vocab = g.get_int(a + "vocab_size")
        else:
            self.vocab = g.tensor("output.weight").n_rows()
        # llama.cpp: LLM_ARCH_LLADA -> ROPE_TYPE_NORM, LLM_ARCH_DREAM -> ROPE_TYPE_NEOX.
        self.rope_neox = self.arch == "dream"
        # LLaDA scores position i from row i; Dream (like its Qwen2 base)
        # from row i-1. The architecture sets the default and the GGUF key
        # `diffusion.shift_logits` overrides it. Defaulting to llama.cpp's
        # "shift when absent" would read the wrong rows of a LLaDA file
        # converted without the key.
        self.shift_logits = self.arch == "dream"
        if g.has_key("diffusion.shift_logits"):
            self.shift_logits = g.get_bool("diffusion.shift_logits")
        if self.head_dim * self.n_heads != self.hidden:
            raise Error("hidden size is not divisible by the head count")
        if self.n_heads % self.n_kv_heads != 0:
            raise Error("query heads are not a multiple of KV heads")
        var rotary_width = g.get_int_or(
            a + "rope.dimension_count", self.head_dim
        )
        if rotary_width != self.head_dim:
            raise Error(
                a
                + "rope.dimension_count must equal head dimension "
                + String(self.head_dim)
                + ", got "
                + String(rotary_width)
            )
        var key_length = g.get_int_or(a + "attention.key_length", self.head_dim)
        if key_length != self.head_dim:
            raise Error(
                a
                + "attention.key_length must equal head dimension "
                + String(self.head_dim)
                + ", got "
                + String(key_length)
            )
        var scaling = g.get_str_or(a + "rope.scaling.type", "none")
        if scaling != "none" and scaling != "linear":
            raise Error(
                a
                + "rope.scaling.type is unsupported: "
                + scaling
                + " (supported: none, linear with factor 1)"
            )
        var factor = g.get_f32_or(a + "rope.scaling.factor", 1.0)
        if factor != 1.0:
            raise Error(
                a + "rope.scaling.factor must be 1, got " + String(factor)
            )

    def kv_dim(self) -> Int:
        """Width of K or V when every layer has the same shape (LLaDA, Dream).
        """
        return self.n_kv_heads * self.head_dim

    def is_sliding(self, layer: Int) -> Bool:
        return (
            layer < self.layer_kinds.byte_length()
            and self.layer_kinds.as_bytes()[layer] == UInt8(ord("s"))
        )

    def kv_heads(self, layer: Int) -> Int:
        if self.is_sliding(layer):
            return self.n_kv_heads_swa
        return self.n_kv_heads

    def head_size(self, layer: Int) -> Int:
        """The width of one K or V head in `layer`."""
        if self.is_sliding(layer):
            return self.head_dim_swa
        return self.head_dim

    def rope_base(self, layer: Int) -> Float32:
        if self.is_sliding(layer):
            return self.theta_swa
        return self.theta

    def check_runnable(self) raises:
        """Refuses a config the loaders can read but no forward pass runs."""
        if self.arch == GEMMA_ARCH:
            raise Error(
                GEMMA_ARCH
                + " has no forward pass yet: the file can be inspected but not"
                " run"
            )


def _positive(g: GGUFFile, key: String) raises -> Int:
    var value = g.get_int(key)
    if value <= 0:
        raise Error(key + " must be positive, got " + String(value))
    return value


def _equal_to(g: GGUFFile, key: String, want: Int, other: String) raises:
    var got = g.get_int(key)
    if got != want:
        raise Error(
            key
            + " must equal "
            + other
            + " "
            + String(want)
            + ", got "
            + String(got)
        )


def _must_be_zero(g: GGUFFile, key: String, why: String) raises:
    var got = g.get_int_or(key, 0)
    if got != 0:
        raise Error(key + " must be 0, got " + String(got) + ": " + why)


def _shape_str(dims: List[Int]) -> String:
    var s = String("[")
    for i in range(len(dims)):
        s += ("" if i == 0 else ", ") + String(dims[i])
    return s + "]"


def _expect_shape(
    g: GGUFFile, name: String, dims: List[Int], why: String
) raises:
    """The tensor has exactly these dimensions; `why` says which keys set them.
    """
    var t = g.tensor(name)
    var have = [t.ne0, t.ne1, t.ne2, t.ne3]
    var same = t.n_dims == len(dims)
    for i in range(len(dims)):
        if i < 4 and have[i] != dims[i]:
            same = False
    if not same:
        raise Error(
            "tensor shape mismatch for "
            + name
            + ": "
            + t.shape_str()
            + ", expected "
            + _shape_str(dims)
            + " ("
            + why
            + ")"
        )


def _read_diffusion_gemma(g: GGUFFile) raises -> ModelConfig:
    """The `diffusion-gemma` branch of `ModelConfig`.

    Each per-layer array must have one entry per block, and the tensor
    directory must agree with the flags (see `_check_diffusion_gemma_tensors`).
    Whatever the forward pass would have to handle differently is refused by
    key: shared K/V layers, a per-layer input embedding, causal attention,
    value heads that differ from key heads, and partial rotary widths.
    """
    var a = String(GEMMA_ARCH) + "."
    var n_layers = _positive(g, a + GEMMA_BLOCK_COUNT)
    var hidden = g.get_int(a + GEMMA_EMBEDDING_LENGTH)
    var ffn = g.get_int(a + GEMMA_FFN_LENGTH)
    var n_heads = _positive(g, a + GEMMA_HEAD_COUNT)
    var head_dim = _positive(g, a + GEMMA_KEY_LENGTH)
    var head_dim_swa = _positive(g, a + GEMMA_KEY_LENGTH_SWA)
    _equal_to(g, a + GEMMA_VALUE_LENGTH, head_dim, String(a + GEMMA_KEY_LENGTH))
    _equal_to(
        g,
        a + GEMMA_VALUE_LENGTH_SWA,
        head_dim_swa,
        String(a + GEMMA_KEY_LENGTH_SWA),
    )
    _equal_to(g, a + GEMMA_ROPE_DIMS, head_dim, String(a + GEMMA_KEY_LENGTH))
    _equal_to(
        g,
        a + GEMMA_ROPE_DIMS_SWA,
        head_dim_swa,
        String(a + GEMMA_KEY_LENGTH_SWA),
    )

    var pattern_key = a + GEMMA_PATTERN
    var kv_key = a + GEMMA_HEAD_COUNT_KV
    var pattern = g.get_bool_array(pattern_key)
    if len(pattern) != n_layers:
        raise Error(
            pattern_key
            + " has length "
            + String(len(pattern))
            + ", expected "
            + String(n_layers)
            + " ("
            + a
            + GEMMA_BLOCK_COUNT
            + ")"
        )
    var kv = g.get_int_array(kv_key)
    if len(kv) != n_layers:
        raise Error(
            kv_key
            + " has length "
            + String(len(kv))
            + ", expected "
            + String(n_layers)
            + " ("
            + a
            + GEMMA_BLOCK_COUNT
            + ")"
        )
    # The forward pass takes one K/V head count per kind of layer, so a layer
    # that disagrees with the first of its kind is refused instead of averaged.
    var kinds = String("")
    var n_kv_heads = 0
    var n_kv_heads_swa = 0
    for i in range(n_layers):
        var heads = kv[i]
        if heads <= 0 or n_heads % heads != 0:
            raise Error(
                kv_key
                + "["
                + String(i)
                + "] is "
                + String(heads)
                + ", expected a positive divisor of "
                + a
                + GEMMA_HEAD_COUNT
                + " ("
                + String(n_heads)
                + ")"
            )
        var kind = "sliding" if pattern[i] else "global"
        var first = n_kv_heads_swa if pattern[i] else n_kv_heads
        if first != 0 and heads != first:
            raise Error(
                kv_key
                + "["
                + String(i)
                + "] is "
                + String(heads)
                + ", but every "
                + kind
                + " layer must have "
                + String(first)
                + " (the first one does)"
            )
        if pattern[i]:
            kinds += "s"
            n_kv_heads_swa = heads
        else:
            kinds += "g"
            n_kv_heads = heads

    var experts = _positive(g, a + GEMMA_EXPERT_COUNT)
    var used = g.get_int(a + GEMMA_EXPERT_USED)
    if used < 1 or used > experts:
        raise Error(
            a
            + GEMMA_EXPERT_USED
            + " must be between 1 and "
            + a
            + GEMMA_EXPERT_COUNT
            + " ("
            + String(experts)
            + "), got "
            + String(used)
        )
    var expert_ffn = _positive(g, a + GEMMA_EXPERT_FFN)
    var window = _positive(g, a + GEMMA_WINDOW)
    var canvas = _positive(g, GEMMA_CANVAS)
    _must_be_zero(
        g,
        a + GEMMA_SHARED_KV,
        "layers that reuse another layer's K/V are not implemented",
    )
    _must_be_zero(
        g,
        a + GEMMA_LAYER_INPUT,
        "per-layer input embeddings are not implemented",
    )
    if g.has_key(a + GEMMA_CAUSAL) and g.get_bool(a + GEMMA_CAUSAL):
        raise Error(
            a
            + GEMMA_CAUSAL
            + " is true: only the bidirectional canvas is implemented"
        )
    var softcap = g.get_f32_or(a + GEMMA_SOFTCAP, 0.0)
    if softcap < 0.0:
        raise Error(
            a + GEMMA_SOFTCAP + " must not be negative, got " + String(softcap)
        )

    var ctx = g.get_int_or(a + GEMMA_CONTEXT_LENGTH, 4096)
    var eps = g.get_f32(a + GEMMA_EPSILON)
    var theta = g.get_f32(a + GEMMA_ROPE_BASE)
    var theta_swa = g.get_f32(a + GEMMA_ROPE_BASE_SWA)
    var mask_id = g.get_int("tokenizer.ggml.mask_token_id")
    # Every key is read before the first tensor, so a missing key is named
    # first even in a file without tensors.
    var vocab = g.tensor(GEMMA_TOKEN_EMBD).n_rows()

    var c = ModelConfig(
        arch=String(GEMMA_ARCH),
        n_layers=n_layers,
        hidden=hidden,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        ffn=ffn,
        vocab=vocab,
        ctx=ctx,
        eps=eps,
        theta=theta,
        mask_id=mask_id,
        # llama.cpp: LLM_ARCH_DIFFUSION_GEMMA -> LLAMA_ROPE_TYPE_NEOX. The
        # canvas rows score their own positions, so logits are not shifted.
        rope_neox=True,
        shift_logits=False,
        layer_kinds=kinds,
        n_kv_heads_swa=n_kv_heads_swa,
        head_dim_swa=head_dim_swa,
        theta_swa=theta_swa,
        sliding_window=window,
        n_experts=experts,
        n_experts_used=used,
        expert_ffn=expert_ffn,
        softcap=softcap,
        canvas=canvas,
    )
    _check_diffusion_gemma_tensors(g, c)
    return c


def _check_diffusion_gemma_tensors(g: GGUFFile, c: ModelConfig) raises:
    """The tensor directory agrees with the config, layer by layer.

    A sliding layer has its own V projection and a global one reuses K, so
    `attn_v` must be there exactly when the pattern says sliding. K must be
    `head_count_kv` heads of the layer's key length, and the expert tensors
    must carry `expert_count` experts of `expert_feed_forward_length`.
    """
    var a = String(GEMMA_ARCH) + "."
    _expect_shape(
        g,
        GEMMA_TOKEN_EMBD,
        [c.hidden, c.vocab],
        a + GEMMA_EMBEDDING_LENGTH + " " + String(c.hidden),
    )
    if "g" in c.layer_kinds:
        _expect_shape(
            g,
            GEMMA_ROPE_FREQS,
            [c.head_dim // 2],
            "half of " + a + GEMMA_KEY_LENGTH + " " + String(c.head_dim),
        )
    for i in range(c.n_layers):
        var p = "blk." + String(i) + "."
        var kind = "sliding" if c.is_sliding(i) else "global"
        var v = p + GEMMA_ATTN_V
        if c.is_sliding(i) and not g.has_tensor(v):
            raise Error(
                "tensor "
                + v
                + " is missing: layer "
                + String(i)
                + " is sliding ("
                + a
                + GEMMA_PATTERN
                + " is true), so it has its own V projection"
            )
        if not c.is_sliding(i) and g.has_tensor(v):
            raise Error(
                "tensor "
                + v
                + " is present: layer "
                + String(i)
                + " is global ("
                + a
                + GEMMA_PATTERN
                + " is false), so V reuses K"
            )
        _expect_shape(
            g,
            p + GEMMA_ATTN_K,
            [c.hidden, c.kv_heads(i) * c.head_size(i)],
            "layer "
            + String(i)
            + " is "
            + kind
            + ": "
            + a
            + GEMMA_HEAD_COUNT_KV
            + " "
            + String(c.kv_heads(i))
            + " x "
            + a
            + (GEMMA_KEY_LENGTH_SWA if c.is_sliding(i) else GEMMA_KEY_LENGTH)
            + " "
            + String(c.head_size(i)),
        )
        _expect_shape(
            g,
            p + GEMMA_EXPERTS_UP,
            [c.hidden, 2 * c.expert_ffn, c.n_experts],
            "gate and up fused: 2 x "
            + a
            + GEMMA_EXPERT_FFN
            + " "
            + String(c.expert_ffn)
            + ", "
            + a
            + GEMMA_EXPERT_COUNT
            + " "
            + String(c.n_experts),
        )
        _expect_shape(
            g,
            p + GEMMA_EXPERTS_DOWN,
            [c.expert_ffn, c.hidden, c.n_experts],
            a
            + GEMMA_EXPERT_FFN
            + " "
            + String(c.expert_ffn)
            + ", "
            + a
            + GEMMA_EXPERT_COUNT
            + " "
            + String(c.n_experts),
        )


struct Layer(Movable):
    var attn_norm: Floats
    var ffn_norm: Floats
    var bq: Floats
    var bk: Floats
    var bv: Floats
    var has_bias: Bool
    var wq: PackedMatrix
    var wk: PackedMatrix
    var wv: PackedMatrix
    var wo: PackedMatrix
    var wg: PackedMatrix
    var wu: PackedMatrix
    var wd: PackedMatrix

    def __init__(out self, g: GGUFFile, i: Int, c: ModelConfig) raises:
        var p = "blk." + String(i) + "."
        var kvd = c.kv_dim()
        self.attn_norm = _load_vector(g, p + "attn_norm.weight", c.hidden)
        self.ffn_norm = _load_vector(g, p + "ffn_norm.weight", c.hidden)
        self.has_bias = g.has_tensor(p + "attn_q.bias")
        if self.has_bias:
            self.bq = _load_vector(g, p + "attn_q.bias", c.hidden)
            self.bk = _load_vector(g, p + "attn_k.bias", kvd)
            self.bv = _load_vector(g, p + "attn_v.bias", kvd)
        else:
            self.bq = Floats(0)
            self.bk = Floats(0)
            self.bv = Floats(0)
        self.wq = _load_matrix(g, p + "attn_q.weight", c.hidden, c.hidden)
        self.wk = _load_matrix(g, p + "attn_k.weight", kvd, c.hidden)
        self.wv = _load_matrix(g, p + "attn_v.weight", kvd, c.hidden)
        self.wo = _load_matrix(g, p + "attn_output.weight", c.hidden, c.hidden)
        self.wg = _load_matrix(g, p + "ffn_gate.weight", c.ffn, c.hidden)
        self.wu = _load_matrix(g, p + "ffn_up.weight", c.ffn, c.hidden)
        self.wd = _load_matrix(g, p + "ffn_down.weight", c.hidden, c.ffn)


def _load_vector(g: GGUFFile, name: String, n: Int) raises -> Floats:
    var t = g.tensor(name)
    if t.n_elements() != n:
        raise Error("tensor shape mismatch for " + name + ": " + t.shape_str())
    var v = Floats(n)
    dequant_row(t.ggml_type, g.data_addr(t), v.ptr(), n)
    return v^


def _load_matrix(
    g: GGUFFile, name: String, rows: Int, cols: Int
) raises -> PackedMatrix:
    var t = g.tensor(name)
    if t.ne0 != cols or t.n_rows() != rows:
        raise Error(
            "tensor shape mismatch for "
            + name
            + ": "
            + t.shape_str()
            + ", expected ["
            + String(cols)
            + ", "
            + String(rows)
            + "]"
        )
    check_matrix_type(t)
    g.map.populate(g.data_addr(t), t.n_bytes)
    var pm = PackedMatrix(g.data_addr(t), t.ggml_type, rows, cols)
    g.map.drop(g.data_addr(t), t.n_bytes)
    return pm^


def _timed_gemm(
    w: PackedMatrix, qa: QActs, n: Int, dst: F32Ptr, ldo: Int, threads: Int
) raises -> Int:
    var t0 = perf_counter_ns()
    pgemm(w, qa, n, dst, ldo, threads)
    return Int(perf_counter_ns() - t0)


struct Timings(Copyable, ImplicitlyCopyable, Movable):
    var gemm_ns: Int
    var attn_ns: Int
    var head_ns: Int
    var total_ns: Int
    var forwards: Int

    def __init__(out self):
        self.gemm_ns = 0
        self.attn_ns = 0
        self.head_ns = 0
        self.total_ns = 0
        self.forwards = 0


struct RowStats(Copyable, Movable):
    """Per requested row, what a greedy sampler needs from its logits.

    With m the row maximum and e_v = exp(logit_v - m): the first argmax, m,
    sum e_v, the largest logit at any other index, and sum e_v (logit_v - m).
    The probability of the argmax is 1 / sum e_v; every Dream confidence is a
    function of these four numbers.
    """

    var tok: List[Int]
    var mx: List[Float64]
    var z: List[Float64]
    var second: List[Float64]
    var weighted: List[Float64]

    def __init__(out self):
        self.tok = List[Int]()
        self.mx = List[Float64]()
        self.z = List[Float64]()
        self.second = List[Float64]()
        self.weighted = List[Float64]()


trait DenoisingModel:
    """What the samplers need from a model, whichever device runs it."""

    def config(self) -> ModelConfig:
        """The hyperparameters read from the GGUF."""
        ...

    def logit_row(self, position: Int) -> Int:
        """The forward-pass row whose logits score `position`."""
        ...

    def forward(
        mut self, tokens: List[Int], out_rows: List[Int], logits: F32Ptr
    ) raises:
        """Logits for `out_rows` (ascending forward-pass rows), written as [len(out_rows), vocab].
        """
        ...

    def forward_window(
        mut self,
        tokens: List[Int],
        lo: Int,
        hi: Int,
        out_rows: List[Int],
        logits: F32Ptr,
    ) raises:
        """Recompute rows [lo, hi) only, against the K/V cache (see DiffusionLM).
        """
        ...

    def device_row_stats(self) -> Bool:
        """Whether `forward_stats` reduces the logits on the device."""
        ...

    def forward_stats(
        mut self, tokens: List[Int], out_rows: List[Int], mut out: RowStats
    ) raises:
        """Like `forward`, but returns each row's `RowStats` and never copies
        the logits to the host."""
        ...

    def model_name(self) raises -> String:
        """`general.name` from the GGUF, or "?"."""
        ...

    def load_time(self) -> Float64:
        """Seconds the constructor took, weights included."""
        ...

    def stats(self) -> Timings:
        """Accumulated forward-pass timings."""
        ...


struct DiffusionLM(DenoisingModel, Movable):
    var gguf: GGUFFile
    var cfg: ModelConfig
    var layers: List[Layer]
    var out_norm: Floats
    var output: PackedMatrix
    var embd: TensorInfo
    var threads: Int
    var max_tokens: Int
    var load_seconds: Float64
    var timings: Timings
    # Activation workspace, sized once for max_tokens.
    var x: Floats
    var xn: Floats
    var q: Floats
    var k: Floats
    var v: Floats
    var att: Floats
    var h: Floats
    var g: Floats
    var u: Floats
    var rc: Floats
    var rs: Floats
    var qa_h: QActs
    var qa_f: QActs
    var kv_cache: Bool
    """Keep every layer's K and V across forward passes (forward_window)."""
    var kc: Floats
    var vc: Floats
    var cache_tokens: Int
    """Canvas length of the last full pass that filled the cache, or -1."""

    def __init__(
        out self,
        path: String,
        threads: Int,
        max_tokens: Int,
        kv_cache: Bool = False,
    ) raises:
        var t0 = perf_counter_ns()
        self.gguf = GGUFFile(path)
        self.cfg = ModelConfig(self.gguf)
        self.cfg.check_runnable()
        var c = self.cfg
        self.threads = threads
        self.max_tokens = max_tokens
        self.embd = self.gguf.tensor("token_embd.weight")
        check_embedding_type(self.embd)
        if self.embd.ne0 != c.hidden:
            raise Error(
                "tensor shape mismatch for token_embd.weight: "
                + self.embd.shape_str()
            )
        self.layers = List[Layer](capacity=c.n_layers)
        for i in range(c.n_layers):
            self.layers.append(Layer(self.gguf, i, c))
        self.out_norm = _load_vector(self.gguf, "output_norm.weight", c.hidden)
        self.output = _load_matrix(
            self.gguf, "output.weight", c.vocab, c.hidden
        )
        var L = max_tokens
        self.x = Floats(L * c.hidden)
        self.xn = Floats(L * c.hidden)
        self.q = Floats(L * c.hidden)
        self.k = Floats(L * c.kv_dim())
        self.v = Floats(L * c.kv_dim())
        self.att = Floats(L * c.hidden)
        self.h = Floats(L * c.hidden)
        self.g = Floats(L * c.ffn)
        self.u = Floats(L * c.ffn)
        self.rc = Floats(L * c.head_dim // 2)
        self.rs = Floats(L * c.head_dim // 2)
        rope_table(L, c.head_dim, c.theta, 0, self.rc.ptr(), self.rs.ptr())
        self.qa_h = QActs(L, c.hidden)
        self.qa_f = QActs(L, c.ffn)
        self.kv_cache = kv_cache
        var cache_n = c.n_layers * L * c.kv_dim() if kv_cache else 0
        self.kc = Floats(cache_n)
        self.vc = Floats(cache_n)
        self.cache_tokens = -1
        self.timings = Timings()
        self.load_seconds = Float64(perf_counter_ns() - t0) / 1e9

    def config(self) -> ModelConfig:
        return self.cfg

    def model_name(self) raises -> String:
        return self.gguf.get_str_or("general.name", "?")

    def load_time(self) -> Float64:
        return self.load_seconds

    def stats(self) -> Timings:
        return self.timings

    def logit_row(self, position: Int) -> Int:
        """The forward-pass row whose logits score `position`."""
        if self.cfg.shift_logits:
            return max(position - 1, 0)
        return position

    def device_row_stats(self) -> Bool:
        return False

    def forward_stats(
        mut self, tokens: List[Int], out_rows: List[Int], mut out: RowStats
    ) raises:
        raise Error("the CPU model has no device row statistics")

    def forward(
        mut self, tokens: List[Int], out_rows: List[Int], logits: F32Ptr
    ) raises:
        """Logits for `out_rows` (ascending forward-pass rows), written as [len(out_rows), vocab].
        """
        self.forward_window(tokens, 0, len(tokens), out_rows, logits)

    def forward_window(
        mut self,
        tokens: List[Int],
        lo: Int,
        hi: Int,
        out_rows: List[Int],
        logits: F32Ptr,
    ) raises:
        """Like `forward`, recomputing only rows [lo, hi) against cached K/V.

        A full window (0, len(tokens)) is an ordinary forward pass; with the
        K/V cache on it also refills the cache. A narrower window needs the
        cache: its rows get fresh Q, K and V (the K and V overwrite their
        cache rows) and attend to every cached row, so the rows outside the
        window keep the K/V of the last pass that computed them. That is
        Fast-dLLM's DualCache, approximate once a window row has changed.
        `out_rows` must lie inside the window.
        """
        var c = self.cfg
        var L = len(tokens)
        var n_out = len(out_rows)
        if L > self.max_tokens:
            raise Error(
                "input of "
                + String(L)
                + " tokens exceeds the runtime limit of "
                + String(self.max_tokens)
            )
        if lo < 0 or hi > L or lo >= hi:
            raise Error(
                "forward window must be a non-empty range inside the input"
            )
        var full = lo == 0 and hi == L
        if not full and not self.kv_cache:
            raise Error("a partial forward window needs the K/V cache")
        if not full and self.cache_tokens != L:
            raise Error(
                "the K/V cache does not hold this canvas; run a full pass first"
            )
        if n_out == 0:
            raise Error("forward called with no output rows")
        for j in range(n_out):
            if (
                out_rows[j] < lo
                or out_rows[j] >= hi
                or (j > 0 and out_rows[j] <= out_rows[j - 1])
            ):
                raise Error(
                    "output rows must be ascending positions inside the window"
                )
        # The cache holds no canvas until this pass completes: one that raises
        # past here may already have overwritten some of its rows.
        if self.kv_cache:
            self.cache_tokens = -1
        var d = c.hidden
        var kvd = c.kv_dim()
        var half = c.head_dim // 2
        var t_start = perf_counter_ns()
        var row_bytes = self.embd.row_bytes()
        var base = self.gguf.data_addr(self.embd)
        var rows = hi - lo
        for i in range(rows):
            var tok = tokens[lo + i]
            if tok < 0 or tok >= c.vocab:
                raise Error(
                    "token id " + String(tok) + " is outside the vocabulary"
                )
            dequant_row(
                self.embd.ggml_type,
                base + tok * row_bytes,
                self.x.ptr().unsafe_offset(i * d),
                d,
            )
        # Keys and values: the cache when it is on (rows lo..hi of each layer),
        # else the scratch buffers. Attention reads every cached row.
        var n_keys = L if self.kv_cache else rows
        var rc = self.rc.ptr().unsafe_offset(lo * half)
        var rs = self.rs.ptr().unsafe_offset(lo * half)

        for li in range(c.n_layers):
            var kbase = self.k.ptr()
            var vbase = self.v.ptr()
            if self.kv_cache:
                kbase = self.kc.ptr().unsafe_offset(li * self.max_tokens * kvd)
                vbase = self.vc.ptr().unsafe_offset(li * self.max_tokens * kvd)
            var kw = kbase.unsafe_offset(lo * kvd if self.kv_cache else 0)
            var vw = vbase.unsafe_offset(lo * kvd if self.kv_cache else 0)
            rmsnorm(
                self.x.ptr(),
                self.layers[li].attn_norm.ptr(),
                self.xn.ptr(),
                rows,
                d,
                c.eps,
                self.threads,
            )
            quantize_acts(self.xn.ptr(), rows, d, self.qa_h)
            self.timings.gemm_ns += _timed_gemm(
                self.layers[li].wq,
                self.qa_h,
                rows,
                self.q.ptr(),
                d,
                self.threads,
            )
            self.timings.gemm_ns += _timed_gemm(
                self.layers[li].wk, self.qa_h, rows, kw, kvd, self.threads
            )
            self.timings.gemm_ns += _timed_gemm(
                self.layers[li].wv, self.qa_h, rows, vw, kvd, self.threads
            )
            var ta = perf_counter_ns()
            if self.layers[li].has_bias:
                add_bias(self.q.ptr(), self.layers[li].bq.ptr(), rows, d)
                add_bias(kw, self.layers[li].bk.ptr(), rows, kvd)
                add_bias(vw, self.layers[li].bv.ptr(), rows, kvd)
            rope_apply(
                self.q.ptr(),
                rows,
                c.n_heads,
                c.head_dim,
                rc,
                rs,
                c.rope_neox,
                self.threads,
            )
            rope_apply(
                kw,
                rows,
                c.n_kv_heads,
                c.head_dim,
                rc,
                rs,
                c.rope_neox,
                self.threads,
            )
            attention(
                self.q.ptr(),
                kbase,
                vbase,
                self.att.ptr(),
                rows,
                c.n_heads,
                c.n_kv_heads,
                c.head_dim,
                self.threads,
                n_keys,
            )
            self.timings.attn_ns += Int(perf_counter_ns() - ta)
            quantize_acts(self.att.ptr(), rows, d, self.qa_h)
            self.timings.gemm_ns += _timed_gemm(
                self.layers[li].wo,
                self.qa_h,
                rows,
                self.h.ptr(),
                d,
                self.threads,
            )
            if li == c.n_layers - 1 and n_out < rows:
                # Only the requested positions continue past attention.
                for j in range(n_out):
                    var r = out_rows[j] - lo
                    if r != j:
                        for e in range(d):
                            self.x[j * d + e] = self.x[r * d + e]
                            self.h[j * d + e] = self.h[r * d + e]
                rows = n_out
            add_inplace(self.x.ptr(), self.h.ptr(), rows * d)
            rmsnorm(
                self.x.ptr(),
                self.layers[li].ffn_norm.ptr(),
                self.xn.ptr(),
                rows,
                d,
                c.eps,
                self.threads,
            )
            quantize_acts(self.xn.ptr(), rows, d, self.qa_h)
            self.timings.gemm_ns += _timed_gemm(
                self.layers[li].wg,
                self.qa_h,
                rows,
                self.g.ptr(),
                c.ffn,
                self.threads,
            )
            self.timings.gemm_ns += _timed_gemm(
                self.layers[li].wu,
                self.qa_h,
                rows,
                self.u.ptr(),
                c.ffn,
                self.threads,
            )
            silu_mul(self.g.ptr(), self.u.ptr(), rows * c.ffn, self.threads)
            quantize_acts(self.g.ptr(), rows, c.ffn, self.qa_f)
            self.timings.gemm_ns += _timed_gemm(
                self.layers[li].wd,
                self.qa_f,
                rows,
                self.h.ptr(),
                d,
                self.threads,
            )
            add_inplace(self.x.ptr(), self.h.ptr(), rows * d)

        if rows != n_out:
            for j in range(n_out):
                var r = out_rows[j] - lo
                if r != j:
                    for e in range(d):
                        self.x[j * d + e] = self.x[r * d + e]
        rmsnorm(
            self.x.ptr(),
            self.out_norm.ptr(),
            self.xn.ptr(),
            n_out,
            d,
            c.eps,
            self.threads,
        )
        quantize_acts(self.xn.ptr(), n_out, d, self.qa_h)
        var th = perf_counter_ns()
        pgemm(self.output, self.qa_h, n_out, logits, c.vocab, self.threads)
        var t_end = perf_counter_ns()
        if self.kv_cache:
            self.cache_tokens = L
        self.timings.head_ns += Int(t_end - th)
        self.timings.forwards += 1
        self.timings.total_ns += Int(t_end - t_start)
