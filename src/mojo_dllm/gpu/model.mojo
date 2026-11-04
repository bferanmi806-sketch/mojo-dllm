"""The masked-diffusion forward pass on an NVIDIA GPU.

Same computation as `models/transformer.mojo` (read its docstring for the
math), with every weight resident on the device. Loading streams one tensor at
a time through a host staging buffer and releases the file's pages behind it,
so peak host memory stays near one tensor rather than the whole model.

The token embedding table stays on the host: a forward pass needs only L of
its rows, which are dequantized on the CPU and copied up with the canvas.
"""

from std.os import getenv
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext

from mojo_dllm.formats.gguf import GGUFFile, TensorInfo
from mojo_dllm.gpu.kernels import (
    GpuQActs,
    add_bias_gpu,
    attention_gpu,
    copy_rows_gpu,
    dev_f32,
    dev_f64,
    dev_i32,
    dev_u8,
    gemm_gpu,
    quantize_gpu,
    rmsnorm_quantize_gpu,
    add_rmsnorm_quantize_gpu,
    silu_mul_quantize_gpu,
    rope_gpu,
    row_stats_gpu,
    RS_FIELDS,
    stage_matrix,
    staged_matrix_bytes,
)
from mojo_dllm.kernels.ops import rope_table
from mojo_dllm.models.transformer import (
    DenoisingModel,
    ModelConfig,
    RowStats,
    Timings,
)
from mojo_dllm.quant.kquants import (
    dequant_row,
    check_matrix_type,
    check_embedding_type,
)
from mojo_dllm.sys.mem import Bytes, F32Ptr, Floats, I32Ptr, U8Ptr

comptime ALIGN = 256


@fieldwise_init
struct DevMatrix(Copyable, ImplicitlyCopyable, Movable):
    var off: Int
    var ggml_type: Int
    var rows: Int
    var cols: Int


@fieldwise_init
struct GpuLayer(Copyable, ImplicitlyCopyable, Movable):
    var attn_norm: Int
    var ffn_norm: Int
    var bq: Int
    var bk: Int
    var bv: Int
    var has_bias: Bool
    var wq: DevMatrix
    var wk: DevMatrix
    var wv: DevMatrix
    var wo: DevMatrix
    var wg: DevMatrix
    var wu: DevMatrix
    var wd: DevMatrix


def _align(n: Int) -> Int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


struct _Planner:
    """Assigns each tensor an aligned offset in the device weight buffer."""

    var size: Int

    def __init__(out self):
        self.size = 0

    def take(mut self, n_bytes: Int) -> Int:
        var off = self.size
        self.size = _align(off + n_bytes)
        return off


def _vec_len(g: GGUFFile, name: String, n: Int) raises -> Int:
    var t = g.tensor(name)
    if t.n_elements() != n:
        raise Error("tensor shape mismatch for " + name + ": " + t.shape_str())
    return n


def _plan_matrix(
    g: GGUFFile, mut plan: _Planner, name: String, rows: Int, cols: Int
) raises -> DevMatrix:
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
    var off = plan.take(staged_matrix_bytes(t.ggml_type, rows, cols))
    return DevMatrix(off, t.ggml_type, rows, cols)


def _largest_staging(
    layers: List[GpuLayer], output: DevMatrix, vocab: Int
) raises -> Int:
    var n = vocab * 4
    for i in range(len(layers)):
        var ly = layers[i]
        for m in [ly.wq, ly.wk, ly.wv, ly.wo, ly.wg, ly.wu, ly.wd]:
            n = max(n, staged_matrix_bytes(m.ggml_type, m.rows, m.cols))
    return max(
        n, staged_matrix_bytes(output.ggml_type, output.rows, output.cols)
    )


struct GpuDiffusionLM(DenoisingModel, Movable):
    var ctx: DeviceContext
    var gguf: GGUFFile
    var cfg: ModelConfig
    var embd: TensorInfo
    var layers: List[GpuLayer]
    var out_norm: Int
    var output: DevMatrix
    var weights: DeviceBuffer[DType.uint8]
    var max_tokens: Int
    var load_seconds: Float64
    var profile: Bool
    var timings: Timings
    var x: DeviceBuffer[DType.float32]
    var xn: DeviceBuffer[DType.float32]
    var q: DeviceBuffer[DType.float32]
    var k: DeviceBuffer[DType.float32]
    var v: DeviceBuffer[DType.float32]
    var att: DeviceBuffer[DType.float32]
    var h: DeviceBuffer[DType.float32]
    var g: DeviceBuffer[DType.float32]
    var u: DeviceBuffer[DType.float32]
    var logits: DeviceBuffer[DType.float32]
    var rc: DeviceBuffer[DType.float32]
    var rs: DeviceBuffer[DType.float32]
    var rows_idx: DeviceBuffer[DType.int32]
    var stat_tok: DeviceBuffer[DType.int32]
    var stat_val: DeviceBuffer[DType.float64]
    var qa_h: GpuQActs
    var qa_f: GpuQActs
    var host_x: Floats
    var host_idx: Bytes
    var kv_cache: Bool
    var kc: DeviceBuffer[DType.float32]
    var vc: DeviceBuffer[DType.float32]
    var cache_tokens: Int
    var host_stat_tok: Bytes
    var host_stat_val: Bytes

    def __init__(
        out self, path: String, max_tokens: Int, kv_cache: Bool = False
    ) raises:
        var t0 = perf_counter_ns()
        # Validate host metadata before creating a device or allocating buffers.
        self.gguf = GGUFFile(path)
        self.cfg = ModelConfig(self.gguf)
        self.cfg.check_runnable()
        var c = self.cfg
        self.max_tokens = max_tokens
        self.embd = self.gguf.tensor("token_embd.weight")
        check_embedding_type(self.embd)
        if DeviceContext.number_of_devices() == 0:
            raise Error(
                "no GPU found: the GPU backend needs an NVIDIA GPU with"
                " compute capability 8.0 or newer (run with --device cpu)"
            )
        self.ctx = DeviceContext()
        if self.ctx.compute_capability() < 80:
            raise Error(
                "the GPU backend needs compute capability 8.0 or newer (int8"
                " tensor cores); "
                + self.ctx.name()
                + " has "
                + String(self.ctx.compute_capability())
            )
        if self.embd.ne0 != c.hidden:
            raise Error(
                "tensor shape mismatch for token_embd.weight: "
                + self.embd.shape_str()
            )
        var kvd = c.kv_dim()

        # Plan the device layout, checking every shape before anything moves.
        var plan = _Planner()
        self.layers = List[GpuLayer](capacity=c.n_layers)
        for i in range(c.n_layers):
            var p = "blk." + String(i) + "."
            var has_bias = self.gguf.has_tensor(p + "attn_q.bias")
            var an = plan.take(
                4 * _vec_len(self.gguf, p + "attn_norm.weight", c.hidden)
            )
            var fnorm = plan.take(
                4 * _vec_len(self.gguf, p + "ffn_norm.weight", c.hidden)
            )
            var bq = -1
            var bk = -1
            var bv = -1
            if has_bias:
                bq = plan.take(
                    4 * _vec_len(self.gguf, p + "attn_q.bias", c.hidden)
                )
                bk = plan.take(4 * _vec_len(self.gguf, p + "attn_k.bias", kvd))
                bv = plan.take(4 * _vec_len(self.gguf, p + "attn_v.bias", kvd))
            self.layers.append(
                GpuLayer(
                    an,
                    fnorm,
                    bq,
                    bk,
                    bv,
                    has_bias,
                    _plan_matrix(
                        self.gguf, plan, p + "attn_q.weight", c.hidden, c.hidden
                    ),
                    _plan_matrix(
                        self.gguf, plan, p + "attn_k.weight", kvd, c.hidden
                    ),
                    _plan_matrix(
                        self.gguf, plan, p + "attn_v.weight", kvd, c.hidden
                    ),
                    _plan_matrix(
                        self.gguf,
                        plan,
                        p + "attn_output.weight",
                        c.hidden,
                        c.hidden,
                    ),
                    _plan_matrix(
                        self.gguf, plan, p + "ffn_gate.weight", c.ffn, c.hidden
                    ),
                    _plan_matrix(
                        self.gguf, plan, p + "ffn_up.weight", c.ffn, c.hidden
                    ),
                    _plan_matrix(
                        self.gguf, plan, p + "ffn_down.weight", c.hidden, c.ffn
                    ),
                )
            )
        self.out_norm = plan.take(
            4 * _vec_len(self.gguf, "output_norm.weight", c.hidden)
        )
        self.output = _plan_matrix(
            self.gguf, plan, "output.weight", c.vocab, c.hidden
        )

        var L = max_tokens
        var act_bytes = 4 * L * (
            6 * c.hidden + 2 * kvd + 2 * c.ffn + c.vocab + c.head_dim
        ) + 2 * L * (c.hidden + c.ffn)
        var cache_n = c.n_layers * L * kvd if kv_cache else 1
        if kv_cache:
            act_bytes += 2 * 4 * cache_n
        var free_bytes = Int(self.ctx.get_memory_info()[0])
        if plan.size + act_bytes > free_bytes:
            raise Error(
                "the model needs "
                + String((plan.size + act_bytes) // 1048576)
                + " MiB of GPU memory and "
                + String(free_bytes // 1048576)
                + " MiB are free"
            )

        self.weights = self.ctx.enqueue_create_buffer[DType.uint8](plan.size)
        self.x = self.ctx.enqueue_create_buffer[DType.float32](L * c.hidden)
        self.xn = self.ctx.enqueue_create_buffer[DType.float32](L * c.hidden)
        self.q = self.ctx.enqueue_create_buffer[DType.float32](L * c.hidden)
        self.k = self.ctx.enqueue_create_buffer[DType.float32](L * kvd)
        self.v = self.ctx.enqueue_create_buffer[DType.float32](L * kvd)
        self.att = self.ctx.enqueue_create_buffer[DType.float32](L * c.hidden)
        self.h = self.ctx.enqueue_create_buffer[DType.float32](L * c.hidden)
        self.g = self.ctx.enqueue_create_buffer[DType.float32](L * c.ffn)
        self.u = self.ctx.enqueue_create_buffer[DType.float32](L * c.ffn)
        self.logits = self.ctx.enqueue_create_buffer[DType.float32](L * c.vocab)
        var half = c.head_dim // 2
        self.rc = self.ctx.enqueue_create_buffer[DType.float32](L * half)
        self.rs = self.ctx.enqueue_create_buffer[DType.float32](L * half)
        var hc = Floats(L * half)
        var hs = Floats(L * half)
        rope_table(L, c.head_dim, c.theta, 0, hc.ptr(), hs.ptr())
        self.ctx.enqueue_copy(self.rc, hc.ptr())
        self.ctx.enqueue_copy(self.rs, hs.ptr())
        self.rows_idx = self.ctx.enqueue_create_buffer[DType.int32](L)
        self.stat_tok = self.ctx.enqueue_create_buffer[DType.int32](L)
        self.stat_val = self.ctx.enqueue_create_buffer[DType.float64](
            RS_FIELDS * L
        )
        self.qa_h = GpuQActs(self.ctx, L, c.hidden)
        self.qa_f = GpuQActs(self.ctx, L, c.ffn)
        self.host_x = Floats(L * c.hidden)
        self.host_idx = Bytes(4 * L)
        self.kv_cache = kv_cache
        self.kc = self.ctx.enqueue_create_buffer[DType.float32](cache_n)
        self.vc = self.ctx.enqueue_create_buffer[DType.float32](cache_n)
        self.cache_tokens = -1
        self.host_stat_tok = Bytes(4 * L)
        self.host_stat_val = Bytes(8 * RS_FIELDS * L)
        self.timings = Timings()
        self.load_seconds = 0
        # Synchronizing around each kernel group costs a little, so timing
        # is opt-in.
        self.profile = getenv("MOJO_DLLM_GPU_PROFILE") == "1"
        # Upload.
        var stage = Bytes(
            _largest_staging(self.layers, self.output, c.vocab), zero=False
        )
        for i in range(c.n_layers):
            var p = "blk." + String(i) + "."
            var ly = self.layers[i]
            self._upload_vec(p + "attn_norm.weight", ly.attn_norm, stage)
            self._upload_vec(p + "ffn_norm.weight", ly.ffn_norm, stage)
            if ly.has_bias:
                self._upload_vec(p + "attn_q.bias", ly.bq, stage)
                self._upload_vec(p + "attn_k.bias", ly.bk, stage)
                self._upload_vec(p + "attn_v.bias", ly.bv, stage)
            self._upload_matrix(p + "attn_q.weight", ly.wq, stage)
            self._upload_matrix(p + "attn_k.weight", ly.wk, stage)
            self._upload_matrix(p + "attn_v.weight", ly.wv, stage)
            self._upload_matrix(p + "attn_output.weight", ly.wo, stage)
            self._upload_matrix(p + "ffn_gate.weight", ly.wg, stage)
            self._upload_matrix(p + "ffn_up.weight", ly.wu, stage)
            self._upload_matrix(p + "ffn_down.weight", ly.wd, stage)
        var on = self.out_norm
        self._upload_vec("output_norm.weight", on, stage)
        var om = self.output
        self._upload_matrix("output.weight", om, stage)

        self.ctx.synchronize()
        _ = hc^
        _ = hs^
        self.load_seconds = Float64(perf_counter_ns() - t0) / 1e9

    def _upload(mut self, off: Int, n_bytes: Int, stage: Bytes) raises:
        var dst = self.weights.create_sub_buffer[DType.uint8](off, n_bytes)
        self.ctx.enqueue_copy(dst, stage.u8())
        # The staging buffer is reused by the next tensor.
        self.ctx.synchronize()

    def _upload_vec(mut self, name: String, off: Int, stage: Bytes) raises:
        var t = self.gguf.tensor(name)
        var n = t.n_elements()
        dequant_row(t.ggml_type, self.gguf.data_addr(t), stage.f32(), n)
        self._upload(off, 4 * n, stage)

    def _upload_matrix(
        mut self, name: String, m: DevMatrix, stage: Bytes
    ) raises:
        var t = self.gguf.tensor(name)
        self.gguf.map.populate(self.gguf.data_addr(t), t.n_bytes)
        stage_matrix(
            self.gguf.data_addr(t), m.ggml_type, m.rows, m.cols, stage.u8()
        )
        self.gguf.map.drop(self.gguf.data_addr(t), t.n_bytes)
        self._upload(
            m.off, staged_matrix_bytes(m.ggml_type, m.rows, m.cols), stage
        )

    def _f32(self, off: Int) -> F32Ptr:
        return dev_u8(self.weights).unsafe_offset(off).unsafe_bitcast[Float32]()

    def _u8(self, off: Int) -> U8Ptr:
        return dev_u8(self.weights).unsafe_offset(off)

    def _tick(self) raises -> Int:
        if self.profile:
            self.ctx.synchronize()
        return Int(perf_counter_ns())

    def _gemm(
        self, m: DevMatrix, qa: GpuQActs, n: Int, dst: F32Ptr, ldo: Int
    ) raises -> Int:
        """Runs one projection; returns its time when profiling, else ~0."""
        var t0 = self._tick()
        gemm_gpu(
            self.ctx,
            m.ggml_type,
            self._u8(m.off),
            m.rows,
            m.cols,
            qa,
            n,
            dst,
            ldo,
        )
        return self._tick() - t0 if self.profile else 0

    def config(self) -> ModelConfig:
        return self.cfg

    def model_name(self) raises -> String:
        return self.gguf.get_str_or("general.name", "?")

    def load_time(self) -> Float64:
        return self.load_seconds

    def stats(self) -> Timings:
        return self.timings

    def logit_row(self, position: Int) -> Int:
        if self.cfg.shift_logits:
            return max(position - 1, 0)
        return position

    def forward(
        mut self, tokens: List[Int], out_rows: List[Int], logits: F32Ptr
    ) raises:
        var t_start = self._forward_logits(tokens, 0, len(tokens), out_rows)
        self.ctx.enqueue_copy(
            logits,
            self.logits.create_sub_buffer[DType.float32](
                0, len(out_rows) * self.cfg.vocab
            ),
        )
        self.ctx.synchronize()
        self._cache_holds(len(tokens))
        self.timings.forwards += 1
        self.timings.total_ns += Int(perf_counter_ns() - t_start)

    def forward_window(
        mut self,
        tokens: List[Int],
        lo: Int,
        hi: Int,
        out_rows: List[Int],
        logits: F32Ptr,
    ) raises:
        """Rows [lo, hi) only, against the K/V cache; see DiffusionLM.forward_window.
        """
        var t_start = self._forward_logits(tokens, lo, hi, out_rows)
        self.ctx.enqueue_copy(
            logits,
            self.logits.create_sub_buffer[DType.float32](
                0, len(out_rows) * self.cfg.vocab
            ),
        )
        self.ctx.synchronize()
        self._cache_holds(len(tokens))
        self.timings.forwards += 1
        self.timings.total_ns += Int(perf_counter_ns() - t_start)

    def device_row_stats(self) -> Bool:
        return True

    def forward_stats(
        mut self, tokens: List[Int], out_rows: List[Int], mut out: RowStats
    ) raises:
        """`forward`, then the row reductions on the device: only the token and
        four numbers per row cross the bus."""
        var t_start = self._forward_logits(tokens, 0, len(tokens), out_rows)
        var n_out = len(out_rows)
        row_stats_gpu(
            self.ctx,
            dev_f32(self.logits),
            n_out,
            self.cfg.vocab,
            dev_i32(self.stat_tok),
            dev_f64(self.stat_val),
        )
        self.ctx.enqueue_copy(
            self.host_stat_tok.i32(),
            self.stat_tok.create_sub_buffer[DType.int32](0, n_out),
        )
        self.ctx.enqueue_copy(
            self.host_stat_val.f64(),
            self.stat_val.create_sub_buffer[DType.float64](
                0, RS_FIELDS * n_out
            ),
        )
        self.ctx.synchronize()
        self._cache_holds(len(tokens))
        var tp = self.host_stat_tok.i32()
        var sp = self.host_stat_val.f64()
        out.tok.clear()
        out.mx.clear()
        out.z.clear()
        out.second.clear()
        out.weighted.clear()
        for j in range(n_out):
            out.tok.append(Int(tp[unsafe_offset=j]))
            out.mx.append(sp[unsafe_offset=RS_FIELDS * j])
            out.z.append(sp[unsafe_offset=RS_FIELDS * j + 1])
            out.second.append(sp[unsafe_offset=RS_FIELDS * j + 2])
            out.weighted.append(sp[unsafe_offset=RS_FIELDS * j + 3])
        self.timings.forwards += 1
        self.timings.total_ns += Int(perf_counter_ns() - t_start)

    def _forward_logits(
        mut self, tokens: List[Int], lo: Int, hi: Int, out_rows: List[Int]
    ) raises -> Int:
        """Run the network up to the LM head over rows [lo, hi), against the K/V
        cache when the window is partial; the logits stay in self.logits (not
        synchronized). Returns the start time in ns."""
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
        # The cache holds no canvas until this pass has synchronized: one that
        # raises, here or at the sync, may already have overwritten some rows.
        if self.kv_cache:
            self.cache_tokens = -1
        var d = c.hidden
        var kvd = c.kv_dim()
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
                self.host_x.ptr().unsafe_offset(i * d),
                d,
            )
        var ip = self.host_idx.i32()
        for j in range(n_out):
            ip[unsafe_offset=j] = Int32(out_rows[j] - lo)
        var ctx = self.ctx
        ctx.enqueue_copy(
            self.x.create_sub_buffer[DType.float32](0, rows * d),
            self.host_x.ptr(),
        )
        ctx.enqueue_copy(
            self.rows_idx.create_sub_buffer[DType.int32](0, n_out), ip
        )
        var x = dev_f32(self.x)
        var xn = dev_f32(self.xn)
        var q = dev_f32(self.q)
        var k = dev_f32(self.k)
        var v = dev_f32(self.v)
        var att = dev_f32(self.att)
        var h = dev_f32(self.h)
        var g = dev_f32(self.g)
        var u = dev_f32(self.u)

        var half = c.head_dim // 2
        var rc = dev_f32(self.rc).unsafe_offset(lo * half)
        var rs = dev_f32(self.rs).unsafe_offset(lo * half)
        var n_keys = L if self.kv_cache else rows
        for li in range(c.n_layers):
            var ly = self.layers[li]
            # K and V go straight into the cache rows lo..hi when it is on.
            var kbase = k
            var vbase = v
            if self.kv_cache:
                kbase = dev_f32(self.kc).unsafe_offset(
                    li * self.max_tokens * kvd
                )
                vbase = dev_f32(self.vc).unsafe_offset(
                    li * self.max_tokens * kvd
                )
            var kw = kbase.unsafe_offset(lo * kvd if self.kv_cache else 0)
            var vw = vbase.unsafe_offset(lo * kvd if self.kv_cache else 0)
            if li == 0:
                # Later layers start from the previous layer's fused tail.
                rmsnorm_quantize_gpu(
                    ctx, x, self._f32(ly.attn_norm), self.qa_h, rows, d, c.eps
                )
            self.timings.gemm_ns += self._gemm(ly.wq, self.qa_h, rows, q, d)
            self.timings.gemm_ns += self._gemm(ly.wk, self.qa_h, rows, kw, kvd)
            self.timings.gemm_ns += self._gemm(ly.wv, self.qa_h, rows, vw, kvd)
            var ta = self._tick()
            if ly.has_bias:
                add_bias_gpu(ctx, q, self._f32(ly.bq), rows, d)
                add_bias_gpu(ctx, kw, self._f32(ly.bk), rows, kvd)
                add_bias_gpu(ctx, vw, self._f32(ly.bv), rows, kvd)
            rope_gpu(ctx, q, rows, c.n_heads, c.head_dim, rc, rs, c.rope_neox)
            rope_gpu(
                ctx, kw, rows, c.n_kv_heads, c.head_dim, rc, rs, c.rope_neox
            )
            attention_gpu(
                ctx,
                q,
                kbase,
                vbase,
                att,
                rows,
                c.n_heads,
                c.n_kv_heads,
                c.head_dim,
                n_keys,
            )
            self.timings.attn_ns += self._tick() - ta
            quantize_gpu(ctx, att, rows, d, self.qa_h)
            self.timings.gemm_ns += self._gemm(ly.wo, self.qa_h, rows, h, d)
            if li == c.n_layers - 1 and n_out < rows:
                # Only the requested positions continue past attention:
                # gather them into the front of x and h.
                copy_rows_gpu(ctx, x, xn, dev_i32(self.rows_idx), n_out, d)
                copy_rows_gpu(ctx, h, att, dev_i32(self.rows_idx), n_out, d)
                ctx.enqueue_copy(
                    self.x.create_sub_buffer[DType.float32](0, n_out * d),
                    self.xn.create_sub_buffer[DType.float32](0, n_out * d),
                )
                ctx.enqueue_copy(
                    self.h.create_sub_buffer[DType.float32](0, n_out * d),
                    self.att.create_sub_buffer[DType.float32](0, n_out * d),
                )
                rows = n_out
            add_rmsnorm_quantize_gpu(
                ctx, x, h, self._f32(ly.ffn_norm), self.qa_h, rows, d, c.eps
            )
            self.timings.gemm_ns += self._gemm(ly.wg, self.qa_h, rows, g, c.ffn)
            self.timings.gemm_ns += self._gemm(ly.wu, self.qa_h, rows, u, c.ffn)
            silu_mul_quantize_gpu(ctx, g, u, self.qa_f, rows, c.ffn)
            self.timings.gemm_ns += self._gemm(ly.wd, self.qa_f, rows, h, d)
            # Residual add, then the next layer's attention norm (or the
            # output norm after the last layer), quantized for its GEMMs.
            var next_norm = self.out_norm
            if li + 1 < c.n_layers:
                next_norm = self.layers[li + 1].attn_norm
            add_rmsnorm_quantize_gpu(
                ctx, x, h, self._f32(next_norm), self.qa_h, rows, d, c.eps
            )

        self.timings.head_ns += self._gemm(
            self.output, self.qa_h, n_out, dev_f32(self.logits), c.vocab
        )
        return Int(t_start)

    def _cache_holds(mut self, n_tokens: Int):
        """Mark the K/V cache as holding an n_tokens canvas. Call only after the
        pass that wrote it has synchronized."""
        if self.kv_cache:
            self.cache_tokens = n_tokens
