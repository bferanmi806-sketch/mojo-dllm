"""DiffusionGemma on the CPU: layer kinds, region masks and the forward pass.

Every rule here was read from llama.cpp's `src/models/diffusion-gemma.cpp` and
`src/models/gemma4-common.h` (ggml-org/llama.cpp#24423 at 2e715c0639), not from
the issue text; `scripts/dg_numpy.py` repeats them in numpy for the tests.
"""


from std.math import sqrt, tanh
from std.time import perf_counter_ns

from mojo_dllm.formats.gguf import GGUFFile, TensorInfo
from mojo_dllm.kernels.moe import (
    ExpertWeights,
    MoEWorkspace,
    Routing,
    gelu_tanh,
    grouped_experts,
    route_tokens,
)
from mojo_dllm.kernels.ops import (
    add_inplace,
    attention_ranges,
    rmsnorm,
    rope_apply,
    rope_table_factors,
)
from mojo_dllm.kernels.packed import PackedMatrix, pgemm
from mojo_dllm.kernels.qgemm import QActs, quantize_acts
from mojo_dllm.models.transformer import (
    ModelConfig,
    _load_vector,
)
from mojo_dllm.quant.kquants import check_embedding_type, dequant_row
from mojo_dllm.sys.mem import Bytes, F32Ptr, Floats
from mojo_dllm.sys.parallel import parallel_for


@fieldwise_init
struct KeyRange(Copyable, ImplicitlyCopyable, Movable):
    """Keys [lo, hi) a query may attend to. Every allowed set is an interval."""

    var lo: Int
    var hi: Int


def key_range(
    q: Int, n_prompt: Int, window: Int, sliding: Bool, n_keys: Int
) -> KeyRange:
    """The keys query `q` (an absolute position) attends to in one layer.

    The first `n_prompt` positions are the prompt, the rest the canvas.
    A prompt row is causal over the prompt, clipped to `window` keys in a
    sliding layer (it sees q - window + 1 .. q). A canvas row sees the whole
    canvas, and of the prompt everything in a global layer but only the last
    window - 1 tokens in a sliding one (the branch's `canvas_prompt_lo`).
    """
    if q < n_prompt:
        if sliding:
            return KeyRange(max(0, q - window + 1), q + 1)
        return KeyRange(0, q + 1)
    if sliding:
        return KeyRange(max(0, n_prompt - window + 1), n_keys)
    return KeyRange(0, n_keys)


comptime ATTENTION_SCALE: Float32 = 1.0
"""`f_attention_scale` of the branch: Gemma 4 does not scale q . k."""


def _load_matrix(
    g: GGUFFile, name: String, rows: Int, cols: Int
) raises -> PackedMatrix:
    """Repack one weight matrix [rows, cols]: Q4_K, Q6_K, Q5_0 or Q8_0.

    `transformer._load_matrix` also demands a GPU kernel, which Q5_0 and Q8_0
    do not have; this model runs on the CPU only, and `PackedMatrix` refuses
    any other type itself.
    """
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
    g.map.populate(g.data_addr(t), t.n_bytes)
    var pm = PackedMatrix(g.data_addr(t), t.ggml_type, rows, cols)
    g.map.drop(g.data_addr(t), t.n_bytes)
    return pm^


def check_block_sizes(
    layer: Int,
    q: Int,
    k: Int,
    v: Int,
    o: Int,
    gate: Int,
    up: Int,
    down: Int,
    expert_gate_up: Int,
    expert_down: Int,
) raises:
    """The activation buffers cover one weight-type mix: 256-wide blocks (Q4_K,
    Q6_K) for attention, gate and up, and 32-wide blocks (Q5_0, Q8_0) for the
    down projections. Any other mix would read a buffer quantized for another
    block size, so it is refused when the model loads."""
    var names = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "expert gate/up",
        "expert down",
    ]
    var got = [q, k, v, o, gate, up, down, expert_gate_up, expert_down]
    var want = [256, 256, 256, 256, 256, 256, 32, 256, 32]
    for i in range(len(names)):
        if got[i] != want[i]:
            raise Error(
                "layer "
                + String(layer)
                + ": "
                + names[i]
                + " has "
                + String(got[i])
                + "-wide blocks where the activation buffers cover "
                + String(want[i])
                + " (attention, gate and up 256, down 32)"
            )


struct GemmaLayer(Movable):
    var sliding: Bool
    var n_kv: Int
    var head: Int
    var attn_norm: Floats
    var q_norm: Floats
    var k_norm: Floats
    var post_attn_norm: Floats
    var ffn_norm: Floats
    var post_ffn_norm_1: Floats
    var pre_ffn_norm_2: Floats
    var post_ffn_norm_2: Floats
    var post_ffn_norm: Floats
    var router_scale: Floats
    var router: Floats
    var out_scale: Float32
    var enc_scale: Float32
    var wq: PackedMatrix
    var wk: PackedMatrix
    var wv: List[PackedMatrix]
    """One matrix on a layer with its own V projection, none when V reuses K."""
    var wo: PackedMatrix
    var gate: PackedMatrix
    var up: PackedMatrix
    var down: PackedMatrix
    var experts: ExpertWeights

    def __init__(out self, g: GGUFFile, i: Int, c: ModelConfig) raises:
        var p = "blk." + String(i) + "."
        var d = c.hidden
        self.sliding = c.is_sliding(i)
        self.n_kv = c.kv_heads(i)
        self.head = c.head_size(i)
        self.attn_norm = _load_vector(g, p + "attn_norm.weight", d)
        self.q_norm = _load_vector(g, p + "attn_q_norm.weight", self.head)
        self.k_norm = _load_vector(g, p + "attn_k_norm.weight", self.head)
        self.post_attn_norm = _load_vector(
            g, p + "post_attention_norm.weight", d
        )
        self.ffn_norm = _load_vector(g, p + "ffn_norm.weight", d)
        self.post_ffn_norm_1 = _load_vector(g, p + "post_ffw_norm_1.weight", d)
        self.pre_ffn_norm_2 = _load_vector(g, p + "pre_ffw_norm_2.weight", d)
        self.post_ffn_norm_2 = _load_vector(g, p + "post_ffw_norm_2.weight", d)
        self.post_ffn_norm = _load_vector(g, p + "post_ffw_norm.weight", d)
        self.router_scale = _load_vector(g, p + "ffn_gate_inp.scale", d)
        self.router = _load_vector(
            g, p + "ffn_gate_inp.weight", c.n_experts * d
        )
        self.out_scale = _load_vector(g, p + "layer_output_scale.weight", 1)[0]
        self.enc_scale = _load_vector(
            g, p + "enc_layer_output_scale.weight", 1
        )[0]
        var q_rows = c.n_heads * self.head
        var kv_rows = self.n_kv * self.head
        self.wq = _load_matrix(g, p + "attn_q.weight", q_rows, d)
        self.wk = _load_matrix(g, p + "attn_k.weight", kv_rows, d)
        self.wv = List[PackedMatrix]()
        if g.has_tensor(p + "attn_v.weight"):
            self.wv.append(_load_matrix(g, p + "attn_v.weight", kv_rows, d))
        self.wo = _load_matrix(g, p + "attn_output.weight", d, q_rows)
        self.gate = _load_matrix(g, p + "ffn_gate.weight", c.ffn, d)
        self.up = _load_matrix(g, p + "ffn_up.weight", c.ffn, d)
        self.down = _load_matrix(g, p + "ffn_down.weight", d, c.ffn)
        var gu = g.tensor(p + "ffn_gate_up_exps.weight")
        var dn = g.tensor(p + "ffn_down_exps.weight")
        var scales = _load_vector(g, p + "ffn_down_exps.scale", c.n_experts)
        g.map.populate(g.data_addr(gu), gu.n_bytes)
        g.map.populate(g.data_addr(dn), dn.n_bytes)
        self.experts = ExpertWeights(
            g.data_addr(gu),
            gu.ggml_type,
            g.data_addr(dn),
            dn.ggml_type,
            d,
            c.expert_ffn,
            c.n_experts,
            scales.ptr(),
        )
        g.map.drop(g.data_addr(gu), gu.n_bytes)
        g.map.drop(g.data_addr(dn), dn.n_bytes)


struct DiffusionGemma(Movable):
    """The DiffusionGemma forward pass over [prompt | canvas], on the CPU.

    `forward` is the branch's default unified pass: one call over the prompt
    and the canvas, P = len(tokens) - canvas.
    """

    var gguf: GGUFFile
    var cfg: ModelConfig
    var layers: List[GemmaLayer]
    var out_norm: Floats
    var output: PackedMatrix
    var embd: TensorInfo
    var rope_cos: List[Floats]
    var rope_sin: List[Floats]
    """Index 0 is the sliding layers' table, 1 the global layers'."""
    var threads: Int
    var max_tokens: Int
    var x: Floats
    """Hidden state [rows, hidden]; after `run_layers`, the last layer's output."""
    var xn: Floats
    var h: Floats
    var q: Floats
    var k: Floats
    var v: Floats
    var att: Floats
    var g: Floats
    var u: Floats
    var mlp: Floats
    var moe_in: Floats
    var moe_out: Floats
    var ones: Floats
    var lo: Bytes
    var hi: Bytes
    var qa_h: QActs
    var qa_att_sliding: QActs
    var qa_att_global: QActs
    var qa_ffn: QActs
    var routing: Routing
    var moe_ws: MoEWorkspace

    def __init__(out self, path: String, threads: Int, max_tokens: Int) raises:
        self.gguf = GGUFFile(path)
        self.cfg = ModelConfig(self.gguf)
        if self.cfg.arch != "diffusion-gemma":
            raise Error(
                "DiffusionGemma runs diffusion-gemma files, not "
                + self.cfg.arch
            )
        var c = self.cfg
        if max_tokens < c.canvas:
            raise Error(
                "max_tokens "
                + String(max_tokens)
                + " is below the canvas length "
                + String(c.canvas)
            )
        self.threads = threads
        self.max_tokens = max_tokens
        self.embd = self.gguf.tensor("token_embd.weight")
        check_embedding_type(self.embd)
        self.layers = List[GemmaLayer](capacity=c.n_layers)
        for i in range(c.n_layers):
            self.layers.append(GemmaLayer(self.gguf, i, c))
        self.out_norm = _load_vector(self.gguf, "output_norm.weight", c.hidden)
        self.output = _load_matrix(
            self.gguf, "token_embd.weight", c.vocab, c.hidden
        )
        var d = c.hidden
        var width_q = 0
        var width_kv = 0
        for i in range(c.n_layers):
            width_q = max(width_q, c.n_heads * c.head_size(i))
            width_kv = max(width_kv, c.kv_heads(i) * c.head_size(i))
            ref L = self.layers[i]
            var v_block = L.wv[0].block_size if len(L.wv) > 0 else 256
            check_block_sizes(
                i,
                L.wq.block_size,
                L.wk.block_size,
                v_block,
                L.wo.block_size,
                L.gate.block_size,
                L.up.block_size,
                L.down.block_size,
                L.experts.gate_up[0].block_size,
                L.experts.down[0].block_size,
            )
        var T = max_tokens
        self.x = Floats(T * d)
        self.xn = Floats(T * d)
        self.h = Floats(T * d)
        self.q = Floats(T * width_q)
        self.k = Floats(T * width_kv)
        self.v = Floats(T * width_kv)
        self.att = Floats(T * width_q)
        self.g = Floats(T * c.ffn)
        self.u = Floats(T * c.ffn)
        self.mlp = Floats(T * d)
        self.moe_in = Floats(T * d)
        self.moe_out = Floats(T * d)
        var widest = max(d, max(c.head_dim, c.head_dim_swa))
        self.ones = Floats(widest)
        for i in range(widest):
            self.ones[i] = 1.0
        self.lo = Bytes(T * 4)
        self.hi = Bytes(T * 4)
        self.qa_h = QActs(T, d)
        self.qa_att_sliding = QActs(T, c.n_heads * c.head_dim_swa)
        self.qa_att_global = QActs(T, c.n_heads * c.head_dim)
        self.qa_ffn = QActs(T, c.ffn, 32)
        self.routing = Routing(T, c.n_experts, c.n_experts_used)
        self.moe_ws = MoEWorkspace(T, self.layers[0].experts)
        self.rope_cos = List[Floats]()
        self.rope_sin = List[Floats]()
        for kind in range(2):
            var sliding = kind == 0
            var hd = c.head_dim_swa if sliding else c.head_dim
            var theta = c.theta_swa if sliding else c.theta
            var cs = Floats(T * (hd // 2))
            var sn = Floats(T * (hd // 2))
            if sliding:
                rope_table_factors(
                    T, hd, theta, 0, self.ones.ptr(), False, cs.ptr(), sn.ptr()
                )
            else:
                var freqs = _load_vector(
                    self.gguf, "rope_freqs.weight", hd // 2
                )
                rope_table_factors(
                    T, hd, theta, 0, freqs.ptr(), True, cs.ptr(), sn.ptr()
                )
            self.rope_cos.append(cs^)
            self.rope_sin.append(sn^)

    def config(self) -> ModelConfig:
        return self.cfg

    def _check_tokens(self, tokens: List[Int]) raises -> Int:
        var n = len(tokens)
        var c = self.cfg
        if n < c.canvas:
            raise Error(
                "need at least the canvas length "
                + String(c.canvas)
                + " tokens, got "
                + String(n)
            )
        if n > self.max_tokens:
            raise Error(
                "input of "
                + String(n)
                + " tokens exceeds the runtime limit of "
                + String(self.max_tokens)
            )
        for i in range(n):
            if tokens[i] < 0 or tokens[i] >= c.vocab:
                raise Error(
                    "token id "
                    + String(tokens[i])
                    + " is outside the vocabulary"
                )
        return n - c.canvas

    def run_layers(mut self, tokens: List[Int], n_layers: Int) raises -> Int:
        """Embed [prompt | canvas] and run the first `n_layers` layers.

        Leaves the hidden state in `self.x` and returns the prompt length.
        """
        var c = self.cfg
        if n_layers < 0 or n_layers > c.n_layers:
            raise Error("layer count " + String(n_layers) + " is out of range")
        var P = self._check_tokens(tokens)
        var T = len(tokens)
        var d = c.hidden
        var th = self.threads
        var row_bytes = self.embd.row_bytes()
        var base = self.gguf.data_addr(self.embd)
        var embed_scale = sqrt(Float32(d))
        for r in range(T):
            var xr = self.x.ptr().unsafe_offset(r * d)
            dequant_row(
                self.embd.ggml_type, base + tokens[r] * row_bytes, xr, d
            )
            for e in range(d):
                xr.unsafe_store(e, xr.unsafe_load(e) * embed_scale)
        # Canvas rows: an RMS norm without weight on top of the scaled embedding.
        rmsnorm(
            self.x.ptr().unsafe_offset(P * d),
            self.ones.ptr(),
            self.x.ptr().unsafe_offset(P * d),
            T - P,
            d,
            c.eps,
            th,
        )
        for li in range(n_layers):
            self._layer(li, P, T)
        return P

    def _layer(mut self, li: Int, P: Int, T: Int) raises:
        var c = self.cfg
        var d = c.hidden
        var th = self.threads
        var nh = c.n_heads
        ref L = self.layers[li]
        var hd = L.head
        var nkv = L.n_kv
        var q_w = nh * hd
        var kv_w = nkv * hd
        rmsnorm(self.x.ptr(), L.attn_norm.ptr(), self.xn.ptr(), T, d, c.eps, th)
        quantize_acts(self.xn.ptr(), T, d, self.qa_h)
        pgemm(L.wq, self.qa_h, T, self.q.ptr(), q_w, th)
        pgemm(L.wk, self.qa_h, T, self.k.ptr(), kv_w, th)
        if len(L.wv) > 0:
            pgemm(L.wv[0], self.qa_h, T, self.v.ptr(), kv_w, th)
        else:
            # No V projection: V is the K projection before its norm and rope.
            for i in range(T * kv_w):
                self.v[i] = self.k[i]
        rmsnorm(
            self.q.ptr(), L.q_norm.ptr(), self.q.ptr(), T * nh, hd, c.eps, th
        )
        rmsnorm(
            self.k.ptr(), L.k_norm.ptr(), self.k.ptr(), T * nkv, hd, c.eps, th
        )
        rmsnorm(
            self.v.ptr(), self.ones.ptr(), self.v.ptr(), T * nkv, hd, c.eps, th
        )
        var kind = 0 if L.sliding else 1
        rope_apply(
            self.q.ptr(),
            T,
            nh,
            hd,
            self.rope_cos[kind].ptr(),
            self.rope_sin[kind].ptr(),
            True,
            th,
        )
        rope_apply(
            self.k.ptr(),
            T,
            nkv,
            hd,
            self.rope_cos[kind].ptr(),
            self.rope_sin[kind].ptr(),
            True,
            th,
        )
        for r in range(T):
            var kr = key_range(r, P, c.sliding_window, L.sliding, T)
            self.lo.i32().unsafe_store(r, Int32(kr.lo))
            self.hi.i32().unsafe_store(r, Int32(kr.hi))
        attention_ranges(
            self.q.ptr(),
            self.k.ptr(),
            self.v.ptr(),
            self.att.ptr(),
            self.lo.i32(),
            self.hi.i32(),
            T,
            T,
            nh,
            nkv,
            hd,
            ATTENTION_SCALE,
            th,
        )
        if L.sliding:
            quantize_acts(self.att.ptr(), T, q_w, self.qa_att_sliding)
            pgemm(L.wo, self.qa_att_sliding, T, self.h.ptr(), d, th)
        else:
            quantize_acts(self.att.ptr(), T, q_w, self.qa_att_global)
            pgemm(L.wo, self.qa_att_global, T, self.h.ptr(), d, th)
        rmsnorm(
            self.h.ptr(), L.post_attn_norm.ptr(), self.h.ptr(), T, d, c.eps, th
        )
        add_inplace(self.x.ptr(), self.h.ptr(), T * d)
        # Dense MLP (the shared expert).
        rmsnorm(self.x.ptr(), L.ffn_norm.ptr(), self.xn.ptr(), T, d, c.eps, th)
        quantize_acts(self.xn.ptr(), T, d, self.qa_h)
        pgemm(L.gate, self.qa_h, T, self.g.ptr(), c.ffn, th)
        pgemm(L.up, self.qa_h, T, self.u.ptr(), c.ffn, th)
        for i in range(T * c.ffn):
            self.g[i] = gelu_tanh(self.g[i]) * self.u[i]
        quantize_acts(self.g.ptr(), T, c.ffn, self.qa_ffn)
        pgemm(L.down, self.qa_ffn, T, self.mlp.ptr(), d, th)
        rmsnorm(
            self.mlp.ptr(),
            L.post_ffn_norm_1.ptr(),
            self.mlp.ptr(),
            T,
            d,
            c.eps,
            th,
        )
        # Mixture of experts. The router reads the raw post-attention residual.
        self.routing.n_tokens = T
        route_tokens(
            self.x.ptr(),
            d,
            L.router_scale.ptr(),
            L.router.ptr(),
            c.eps,
            self.routing,
            th,
        )
        rmsnorm(
            self.x.ptr(),
            L.pre_ffn_norm_2.ptr(),
            self.moe_in.ptr(),
            T,
            d,
            c.eps,
            th,
        )
        grouped_experts(
            L.experts,
            self.moe_in.ptr(),
            self.routing,
            self.moe_ws,
            self.moe_out.ptr(),
            th,
        )
        rmsnorm(
            self.moe_out.ptr(),
            L.post_ffn_norm_2.ptr(),
            self.moe_out.ptr(),
            T,
            d,
            c.eps,
            th,
        )
        add_inplace(self.mlp.ptr(), self.moe_out.ptr(), T * d)
        rmsnorm(
            self.mlp.ptr(),
            L.post_ffn_norm.ptr(),
            self.mlp.ptr(),
            T,
            d,
            c.eps,
            th,
        )
        add_inplace(self.x.ptr(), self.mlp.ptr(), T * d)
        # Layer output scalar: the encoder's on prompt rows, the decoder's on canvas rows.
        for r in range(T):
            var s = L.enc_scale if r < P else L.out_scale
            for e in range(d):
                self.x[r * d + e] = self.x[r * d + e] * s

    def forward(mut self, tokens: List[Int], logits: F32Ptr) raises:
        """Canvas logits [canvas, vocab] of one unified pass over [prompt | canvas].
        """
        var c = self.cfg
        var d = c.hidden
        var P = self.run_layers(tokens, c.n_layers)
        var C = len(tokens) - P
        rmsnorm(
            self.x.ptr().unsafe_offset(P * d),
            self.out_norm.ptr(),
            self.xn.ptr(),
            C,
            d,
            c.eps,
            self.threads,
        )
        quantize_acts(self.xn.ptr(), C, d, self.qa_h)
        pgemm(self.output, self.qa_h, C, logits, c.vocab, self.threads)
        if c.softcap != 0:
            _softcap(logits, C * c.vocab, c.softcap, self.threads)


def _softcap(x: F32Ptr, n: Int, cap: Float32, threads: Int) raises:
    """x = cap * tanh(x / cap), the branch's order: scale by 1/cap, tanh, scale by cap.
    """
    var chunk = 8192
    var items = (n + chunk - 1) // chunk
    var inv = 1.0 / cap

    def work(i: Int) {imm}:
        var hi = min((i + 1) * chunk, n)
        for j in range(i * chunk, hi):
            x.unsafe_store(j, cap * tanh(x.unsafe_load(j) * inv))

    parallel_for(work, items, threads)
