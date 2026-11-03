"""Masked-diffusion transformers: LLaDA and Dream.

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


struct ModelConfig(Copyable, ImplicitlyCopyable, Movable):
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

    def __init__(out self, g: GGUFFile) raises:
        self.arch = g.architecture()
        if self.arch != "llada" and self.arch != "dream":
            raise Error(
                "unsupported model architecture: "
                + self.arch
                + " (supported: llada, dream)"
            )
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
        return self.n_kv_heads * self.head_dim


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
