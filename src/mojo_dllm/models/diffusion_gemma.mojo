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
from mojo_dllm.kernels.softembed import EmbeddingT, soft_embed, softmax_rows
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
    var kc: List[Floats]
    var vc: List[Floats]
    """Per layer, the keys (after the K norm and RoPE) and values (after the
    weightless V norm) of every position: [max_tokens, kv heads * head size]."""
    var cached_prompt: Int
    """The prompt length `prefill` left in the cache, or -1 when there is none."""
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
    var sc_norm: Floats
    var sc_gate: List[PackedMatrix]
    var sc_up: List[PackedMatrix]
    var sc_down: List[PackedMatrix]
    """The self-conditioning MLP, one matrix each when the file has it, else none."""
    var sc_embt: List[EmbeddingT]
    """The transposed embedding table, built by the first self-conditioned pass."""

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
        for i in range(c.n_layers):
            width_q = max(width_q, c.n_heads * c.head_size(i))
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
        self.kc = List[Floats]()
        self.vc = List[Floats]()
        for i in range(c.n_layers):
            self.kc.append(Floats(T * c.kv_heads(i) * c.head_size(i)))
            self.vc.append(Floats(T * c.kv_heads(i) * c.head_size(i)))
        self.cached_prompt = -1
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
        self.sc_gate = List[PackedMatrix]()
        self.sc_up = List[PackedMatrix]()
        self.sc_down = List[PackedMatrix]()
        self.sc_embt = List[EmbeddingT]()
        if self.gguf.has_tensor("self_cond_gate.weight"):
            self.sc_norm = _load_vector(
                self.gguf, "self_cond_pre_norm.weight", d
            )
            self.sc_gate.append(
                _load_matrix(self.gguf, "self_cond_gate.weight", c.ffn, d)
            )
            self.sc_up.append(
                _load_matrix(self.gguf, "self_cond_up.weight", c.ffn, d)
            )
            self.sc_down.append(
                _load_matrix(self.gguf, "self_cond_down.weight", d, c.ffn)
            )
            if (
                self.sc_gate[0].block_size != 256
                or self.sc_up[0].block_size != 256
                or self.sc_down[0].block_size != 32
            ):
                raise Error(
                    "self_cond gate and up must be 256-wide quantized and down"
                    " 32-wide, like the dense MLP"
                )
        else:
            self.sc_norm = Floats(0)
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

    def _embed(
        mut self,
        tokens: List[Int],
        pos0: Int,
        P: Int,
        prev_addr: Int = 0,
        prev_temp_inv: Float32 = 1.0,
    ) raises:
        """Rows for the tokens at positions pos0 ..: embed * sqrt(n_embd); the
        rows at or past P (the canvas) then get the self-conditioning signal,
        if `prev_addr` holds the previous step's logits, and an RMS norm
        without weight."""
        var c = self.cfg
        var d = c.hidden
        var row_bytes = self.embd.row_bytes()
        var base = self.gguf.data_addr(self.embd)
        var embed_scale = sqrt(Float32(d))
        var rows = len(tokens)
        for r in range(rows):
            var xr = self.x.ptr().unsafe_offset(r * d)
            dequant_row(
                self.embd.ggml_type, base + tokens[r] * row_bytes, xr, d
            )
            for e in range(d):
                xr.unsafe_store(e, xr.unsafe_load(e) * embed_scale)
        var first_canvas = max(P - pos0, 0)
        if prev_addr != 0:
            self._self_condition(
                first_canvas, rows - first_canvas, prev_addr, prev_temp_inv
            )
        rmsnorm(
            self.x.ptr().unsafe_offset(first_canvas * d),
            self.ones.ptr(),
            self.x.ptr().unsafe_offset(first_canvas * d),
            rows - first_canvas,
            d,
            c.eps,
            self.threads,
        )

    def _self_condition(
        mut self, row0: Int, C: Int, prev_addr: Int, temp_inv: Float32
    ) raises:
        """Add the self-conditioning signal to canvas rows row0 ..< row0 + C of `x`.

        The previous step's logits [C, vocab] are replaced by their softmax at
        the previous temperature, in place, and multiplied into the embedding
        table; the result times sqrt(n_embd) goes through the weighted norm and
        the gated MLP, down(gelu(gate) * up), and is added to the embedding.
        """
        var c = self.cfg
        var d = c.hidden
        var th = self.threads
        if len(self.sc_gate) == 0:
            raise Error("this file has no self-conditioning tensors")
        if len(self.sc_embt) == 0:
            self.sc_embt.append(
                EmbeddingT(
                    self.gguf.data_addr(self.embd),
                    self.embd.ggml_type,
                    c.vocab,
                    d,
                    th,
                )
            )
        var prev = F32Ptr(unsafe_from_address=prev_addr)
        softmax_rows(prev, C, c.vocab, temp_inv, th)
        soft_embed(prev, self.sc_embt[0], self.moe_in.ptr(), C, th)
        var embed_scale = sqrt(Float32(d))
        for i in range(C * d):
            self.moe_in[i] = self.moe_in[i] * embed_scale
        rmsnorm(
            self.moe_in.ptr(),
            self.sc_norm.ptr(),
            self.xn.ptr(),
            C,
            d,
            c.eps,
            th,
        )
        quantize_acts(self.xn.ptr(), C, d, self.qa_h)
        pgemm(self.sc_gate[0], self.qa_h, C, self.g.ptr(), c.ffn, th)
        pgemm(self.sc_up[0], self.qa_h, C, self.u.ptr(), c.ffn, th)
        for i in range(C * c.ffn):
            self.g[i] = gelu_tanh(self.g[i]) * self.u[i]
        quantize_acts(self.g.ptr(), C, c.ffn, self.qa_ffn)
        pgemm(self.sc_down[0], self.qa_ffn, C, self.mlp.ptr(), d, th)
        add_inplace(self.x.ptr().unsafe_offset(row0 * d), self.mlp.ptr(), C * d)

    def run_layers(mut self, tokens: List[Int], n_layers: Int) raises -> Int:
        """Embed [prompt | canvas] and run the first `n_layers` layers.

        Leaves the hidden state in `self.x` and returns the prompt length.
        It overwrites every layer's cache rows, so a `prefill` does not survive it.
        """
        return self._run(tokens, n_layers, 0, 1.0)

    def _run(
        mut self,
        tokens: List[Int],
        n_layers: Int,
        prev_addr: Int,
        prev_temp_inv: Float32,
    ) raises -> Int:
        var c = self.cfg
        if n_layers < 0 or n_layers > c.n_layers:
            raise Error("layer count " + String(n_layers) + " is out of range")
        var P = self._check_tokens(tokens)
        self.cached_prompt = -1
        var T = len(tokens)
        self._embed(tokens, 0, P, prev_addr, prev_temp_inv)
        for li in range(n_layers):
            self._layer(li, T, 0, P, T)
        return P

    def _check_ids(self, tokens: List[Int]) raises:
        for i in range(len(tokens)):
            if tokens[i] < 0 or tokens[i] >= self.cfg.vocab:
                raise Error(
                    "token id "
                    + String(tokens[i])
                    + " is outside the vocabulary"
                )

    def prefill(mut self, prompt: List[Int]) raises:
        """Run the prompt once, causal over itself, and keep every layer's K and V.

        `step` then scores a canvas against them. The prompt rows depend on
        nothing after them, so this is what the unified pass computes for them.
        A prefill that raises leaves no prompt behind.
        """
        var c = self.cfg
        self.cached_prompt = -1
        var P = len(prompt)
        if P + c.canvas > self.max_tokens:
            raise Error(
                "a prompt of "
                + String(P)
                + " tokens leaves no room for the canvas of "
                + String(c.canvas)
                + " within the runtime limit of "
                + String(self.max_tokens)
            )
        self._check_ids(prompt)
        if P > 0:
            self._embed(prompt, 0, P)
            for li in range(c.n_layers):
                self._layer(li, P, 0, P, P)
        self.cached_prompt = P

    def step(mut self, canvas: List[Int], logits: F32Ptr) raises:
        """Canvas logits [canvas, vocab] against the K and V `prefill` kept.

        Only the canvas rows are computed: they sit at positions P ..< P + canvas,
        attend to the cached prompt and to each other, and overwrite the cache
        rows from P on. The prompt rows stay, so any number of steps can follow
        one prefill.
        """
        self._step(canvas, logits, 0, 1.0)

    def step_sc(
        mut self, canvas: List[Int], logits: F32Ptr, prev_temp_inv: Float32
    ) raises:
        """`step` with self-conditioning from the previous step's logits.

        `logits` holds the previous step's canvas logits [canvas, vocab] on
        entry and this step's on return. The previous ones are consumed first:
        softmaxed in place at `prev_temp_inv` (1 / the previous temperature),
        which is why one 268 MB buffer serves a whole run.
        """
        _check_temp(prev_temp_inv)
        self._step(canvas, logits, Int(logits), prev_temp_inv)

    def _step(
        mut self,
        canvas: List[Int],
        logits: F32Ptr,
        prev_addr: Int,
        prev_temp_inv: Float32,
    ) raises:
        var c = self.cfg
        if self.cached_prompt < 0:
            raise Error("run prefill before step: no prompt is cached")
        if len(canvas) != c.canvas:
            raise Error(
                "step takes the canvas of "
                + String(c.canvas)
                + " tokens, got "
                + String(len(canvas))
            )
        self._check_ids(canvas)
        var P = self.cached_prompt
        self._embed(canvas, P, P, prev_addr, prev_temp_inv)
        for li in range(c.n_layers):
            self._layer(li, c.canvas, P, P, P + c.canvas)
        self._head(0, c.canvas, logits)

    def _head(mut self, row0: Int, C: Int, logits: F32Ptr) raises:
        """Final norm, the tied head and the soft cap on rows row0 ..< row0 + C of `x`.
        """
        var c = self.cfg
        var d = c.hidden
        rmsnorm(
            self.x.ptr().unsafe_offset(row0 * d),
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

    def _layer(
        mut self, li: Int, rows: Int, pos0: Int, P: Int, n_keys: Int
    ) raises:
        """One layer over `rows` rows at absolute positions pos0 ..< pos0 + rows.

        The K and V of those rows go to the layer's cache at row pos0, and the
        attention reads cache rows [0, n_keys): the unified pass has pos0 = 0
        and n_keys = rows, a prefill the same with all rows prompt, and a step
        pos0 = P and n_keys = P + rows. Rows at positions below P are prompt.
        """
        var c = self.cfg
        var d = c.hidden
        var th = self.threads
        var nh = c.n_heads
        ref L = self.layers[li]
        var hd = L.head
        var nkv = L.n_kv
        var q_w = nh * hd
        var kv_w = nkv * hd
        var kp = self.kc[li].ptr().unsafe_offset(pos0 * kv_w)
        var vp = self.vc[li].ptr().unsafe_offset(pos0 * kv_w)
        var half = hd // 2
        var kind = 0 if L.sliding else 1
        var rc = self.rope_cos[kind].ptr().unsafe_offset(pos0 * half)
        var rs = self.rope_sin[kind].ptr().unsafe_offset(pos0 * half)
        rmsnorm(
            self.x.ptr(), L.attn_norm.ptr(), self.xn.ptr(), rows, d, c.eps, th
        )
        quantize_acts(self.xn.ptr(), rows, d, self.qa_h)
        pgemm(L.wq, self.qa_h, rows, self.q.ptr(), q_w, th)
        pgemm(L.wk, self.qa_h, rows, kp, kv_w, th)
        if len(L.wv) > 0:
            pgemm(L.wv[0], self.qa_h, rows, vp, kv_w, th)
        else:
            # No V projection: V is the K projection before its norm and rope.
            for i in range(rows * kv_w):
                vp.unsafe_store(i, kp.unsafe_load(i))
        rmsnorm(
            self.q.ptr(), L.q_norm.ptr(), self.q.ptr(), rows * nh, hd, c.eps, th
        )
        rmsnorm(kp, L.k_norm.ptr(), kp, rows * nkv, hd, c.eps, th)
        rmsnorm(vp, self.ones.ptr(), vp, rows * nkv, hd, c.eps, th)
        rope_apply(
            self.q.ptr(),
            rows,
            nh,
            hd,
            rc,
            rs,
            True,
            th,
        )
        rope_apply(
            kp,
            rows,
            nkv,
            hd,
            rc,
            rs,
            True,
            th,
        )
        for r in range(rows):
            var kr = key_range(pos0 + r, P, c.sliding_window, L.sliding, n_keys)
            self.lo.i32().unsafe_store(r, Int32(kr.lo))
            self.hi.i32().unsafe_store(r, Int32(kr.hi))
        attention_ranges(
            self.q.ptr(),
            self.kc[li].ptr(),
            self.vc[li].ptr(),
            self.att.ptr(),
            self.lo.i32(),
            self.hi.i32(),
            rows,
            n_keys,
            nh,
            nkv,
            hd,
            ATTENTION_SCALE,
            th,
        )
        if L.sliding:
            quantize_acts(self.att.ptr(), rows, q_w, self.qa_att_sliding)
            pgemm(L.wo, self.qa_att_sliding, rows, self.h.ptr(), d, th)
        else:
            quantize_acts(self.att.ptr(), rows, q_w, self.qa_att_global)
            pgemm(L.wo, self.qa_att_global, rows, self.h.ptr(), d, th)
        rmsnorm(
            self.h.ptr(),
            L.post_attn_norm.ptr(),
            self.h.ptr(),
            rows,
            d,
            c.eps,
            th,
        )
        add_inplace(self.x.ptr(), self.h.ptr(), rows * d)
        # Dense MLP (the shared expert).
        rmsnorm(
            self.x.ptr(), L.ffn_norm.ptr(), self.xn.ptr(), rows, d, c.eps, th
        )
        quantize_acts(self.xn.ptr(), rows, d, self.qa_h)
        pgemm(L.gate, self.qa_h, rows, self.g.ptr(), c.ffn, th)
        pgemm(L.up, self.qa_h, rows, self.u.ptr(), c.ffn, th)
        for i in range(rows * c.ffn):
            self.g[i] = gelu_tanh(self.g[i]) * self.u[i]
        quantize_acts(self.g.ptr(), rows, c.ffn, self.qa_ffn)
        pgemm(L.down, self.qa_ffn, rows, self.mlp.ptr(), d, th)
        rmsnorm(
            self.mlp.ptr(),
            L.post_ffn_norm_1.ptr(),
            self.mlp.ptr(),
            rows,
            d,
            c.eps,
            th,
        )
        # Mixture of experts. The router reads the raw post-attention residual.
        self.routing.n_tokens = rows
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
            rows,
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
            rows,
            d,
            c.eps,
            th,
        )
        add_inplace(self.mlp.ptr(), self.moe_out.ptr(), rows * d)
        rmsnorm(
            self.mlp.ptr(),
            L.post_ffn_norm.ptr(),
            self.mlp.ptr(),
            rows,
            d,
            c.eps,
            th,
        )
        add_inplace(self.x.ptr(), self.mlp.ptr(), rows * d)
        # Layer output scalar: the encoder's on prompt rows, the decoder's on canvas rows.
        for r in range(rows):
            var s = L.enc_scale if pos0 + r < P else L.out_scale
            for e in range(d):
                self.x[r * d + e] = self.x[r * d + e] * s

    def forward_sc(
        mut self, tokens: List[Int], logits: F32Ptr, prev_temp_inv: Float32
    ) raises:
        """`forward` with self-conditioning from the previous step's logits.

        `logits` holds the previous step's canvas logits [canvas, vocab] on
        entry and this step's on return; see `step_sc`.
        """
        _check_temp(prev_temp_inv)
        var c = self.cfg
        var P = self._run(tokens, c.n_layers, Int(logits), prev_temp_inv)
        self._head(P, len(tokens) - P, logits)

    def forward(mut self, tokens: List[Int], logits: F32Ptr) raises:
        """Canvas logits [canvas, vocab] of one unified pass over [prompt | canvas].
        """
        var c = self.cfg
        var P = self.run_layers(tokens, c.n_layers)
        self._head(P, len(tokens) - P, logits)


def _check_temp(inv: Float32) raises:
    if not (inv > 0) or inv > 3.0e38:
        raise Error(
            "the previous temperature must be positive and finite, got 1 / "
            + String(inv)
        )


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
