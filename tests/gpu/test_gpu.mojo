"""GPU kernels and the GPU forward pass, each checked against the CPU path.

Run with scripts/run-gpu-tests.sh, which refuses to report success on a
machine without a GPU. The CPU suite (tests/test_*.mojo) is the oracle here:
every kernel is compared with the CPU function it replaces.
"""

from std.math import sqrt
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.gpu.kernels import (
    GpuQActs,
    rmsnorm_gpu,
    rope_gpu,
    attention_gpu,
    silu_mul_gpu,
    add_gpu,
    add_bias_gpu,
    quantize_gpu,
    rmsnorm_quantize_gpu,
    add_rmsnorm_quantize_gpu,
    silu_mul_quantize_gpu,
    gemm_gpu,
    staged_matrix_bytes,
    dev_f32,
    dev_u8,
    stage_matrix,
    dev_i8,
    dev_i32,
    dev_f64,
    row_stats_gpu,
    RS_FIELDS,
    mma_tile_gpu,
    _pick_bn,
    _pick_q4_bn,
    _launch,
)
from mojo_dllm.gpu.model import GpuDiffusionLM
from mojo_dllm.kernels.ops import (
    rmsnorm,
    rope_table,
    rope_apply,
    attention,
    silu_mul,
    add_inplace,
    add_bias,
)
from mojo_dllm.diffusion.sampler import (
    GenConfig,
    _confidence,
    _confidence_from_stats,
    argmax_and_prob,
    generate,
)
from mojo_dllm.models.transformer import DiffusionLM, RowStats
from mojo_dllm.quant.kquants import dequant_row
from mojo_dllm.sys.mem import F32Ptr, I8Ptr, I32Ptr, U8Ptr, Floats, Bytes
from mojo_dllm.sys.mmap import Mapping

comptime FIX = "tests/fixtures/"


def _fill(mut x: Floats, seed: Int):
    var s = UInt64(seed * 2654435761 + 1)
    for i in range(x.n):
        s = s * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int((s >> 33) % 2001) - 1000) / 250.0


def _up(ctx: DeviceContext, x: Floats) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](x.n)
    ctx.enqueue_copy(b, x.ptr())
    return b


def _down(
    ctx: DeviceContext, b: DeviceBuffer[DType.float32], n: Int
) raises -> Floats:
    var h = Floats(n)
    ctx.enqueue_copy(h.ptr(), b.create_sub_buffer[DType.float32](0, n))
    ctx.synchronize()
    return h^


def _max_abs(a: Floats, b: Floats) -> Float64:
    var worst = Float64(0)
    for i in range(a.n):
        worst = max(worst, abs(Float64(a[i]) - Float64(b[i])))
    return worst


def _rel(got: Floats, want: Floats) -> Float64:
    var num = Float64(0)
    var den = Float64(0)
    for i in range(want.n):
        var d = Float64(got[i]) - Float64(want[i])
        num += d * d
        den += Float64(want[i]) * Float64(want[i])
    return sqrt(num / den)


def test_rmsnorm_matches_cpu() raises:
    var ctx = DeviceContext()
    var rows = 5
    var dim = 384
    var x = Floats(rows * dim)
    var w = Floats(dim)
    _fill(x, 1)
    _fill(w, 2)
    var want = Floats(rows * dim)
    rmsnorm(x.ptr(), w.ptr(), want.ptr(), rows, dim, 1e-5, 1)
    var dx = _up(ctx, x)
    var dw = _up(ctx, w)
    var dy = ctx.enqueue_create_buffer[DType.float32](rows * dim)
    rmsnorm_gpu(ctx, dev_f32(dx), dev_f32(dw), dev_f32(dy), rows, dim, 1e-5)
    var got = _down(ctx, dy, rows * dim)
    assert_true(
        _max_abs(got, want) < 1e-4, "rmsnorm " + String(_max_abs(got, want))
    )


def _check_rope(neox: Bool) raises:
    var ctx = DeviceContext()
    var L = 9
    var heads = 3
    var hd = 64
    var x = Floats(L * heads * hd)
    _fill(x, 3)
    var c = Floats(L * hd // 2)
    var s = Floats(L * hd // 2)
    rope_table(L, hd, 10000.0, 0, c.ptr(), s.ptr())
    var dx = _up(ctx, x)
    var dc = _up(ctx, c)
    var ds = _up(ctx, s)
    rope_apply(x.ptr(), L, heads, hd, c.ptr(), s.ptr(), neox, 1)
    rope_gpu(ctx, dev_f32(dx), L, heads, hd, dev_f32(dc), dev_f32(ds), neox)
    var got = _down(ctx, dx, x.n)
    assert_true(_max_abs(got, x) < 1e-5, "rope " + String(_max_abs(got, x)))


def test_rope_norm_matches_cpu() raises:
    _check_rope(False)


def test_rope_neox_matches_cpu() raises:
    _check_rope(True)


def _check_attention(L: Int, hd: Int) raises:
    # The GPU kernel multiplies in f16 (f32 accumulation), so it matches the
    # f32 CPU kernel to f16 rounding, not to 1e-4.
    var ctx = DeviceContext()
    var heads = 4
    var kvh = 2
    var q = Floats(L * heads * hd)
    var k = Floats(L * kvh * hd)
    var v = Floats(L * kvh * hd)
    _fill(q, 4)
    _fill(k, 5)
    _fill(v, 6)
    for i in range(q.n):
        q[i] = q[i] * 0.1
    var want = Floats(L * heads * hd)
    attention(q.ptr(), k.ptr(), v.ptr(), want.ptr(), L, heads, kvh, hd, 2)
    var dq = _up(ctx, q)
    var dk = _up(ctx, k)
    var dv = _up(ctx, v)
    var do_ = ctx.enqueue_create_buffer[DType.float32](want.n)
    attention_gpu(
        ctx,
        dev_f32(dq),
        dev_f32(dk),
        dev_f32(dv),
        dev_f32(do_),
        L,
        heads,
        kvh,
        hd,
    )
    var got = _down(ctx, do_, want.n)
    var rel = _rel(got, want)
    print("attention L", L, "head_dim", hd, "relative error", rel)
    assert_true(
        rel < 2e-3,
        "attention L="
        + String(L)
        + " hd="
        + String(hd)
        + " rel "
        + String(rel),
    )


def test_attention_gqa_matches_cpu() raises:
    _check_attention(37, 128)


def test_attention_head_dim_64_matches_cpu() raises:
    _check_attention(37, 64)


def test_attention_many_tiles_matches_cpu() raises:
    # 150 queries: three query blocks, the last ragged; five key tiles.
    _check_attention(150, 128)


def test_elementwise_ops_match_cpu() raises:
    var ctx = DeviceContext()
    var rows = 3
    var dim = 1000
    var n = rows * dim
    var g = Floats(n)
    var u = Floats(n)
    var b = Floats(dim)
    _fill(g, 7)
    _fill(u, 8)
    _fill(b, 9)
    var dg = _up(ctx, g)
    var du = _up(ctx, u)
    var db = _up(ctx, b)
    silu_mul(g.ptr(), u.ptr(), n, 1)
    silu_mul_gpu(ctx, dev_f32(dg), dev_f32(du), n)
    add_inplace(g.ptr(), u.ptr(), n)
    add_gpu(ctx, dev_f32(dg), dev_f32(du), n)
    add_bias(g.ptr(), b.ptr(), rows, dim)
    add_bias_gpu(ctx, dev_f32(dg), dev_f32(db), rows, dim)
    var got = _down(ctx, dg, n)
    assert_true(
        _max_abs(got, g) < 1e-4, "elementwise " + String(_max_abs(got, g))
    )


def _check_mma(K: Int) raises:
    var ctx = DeviceContext()
    var ha = Bytes(16 * K)
    var hb = Bytes(8 * K)
    var s = UInt64(12345 + K)
    for i in range(16 * K):
        s = s * 6364136223846793005 + 1442695040888963407
        ha.i8()[unsafe_offset=i] = Int8(Int((s >> 33) % 255) - 127)
    for i in range(8 * K):
        s = s * 6364136223846793005 + 1442695040888963407
        hb.i8()[unsafe_offset=i] = Int8(Int((s >> 33) % 255) - 127)
    var da = ctx.enqueue_create_buffer[DType.int8](16 * K)
    var db = ctx.enqueue_create_buffer[DType.int8](8 * K)
    var dc = ctx.enqueue_create_buffer[DType.int32](128)
    ctx.enqueue_copy(da, ha.i8())
    ctx.enqueue_copy(db, hb.i8())
    mma_tile_gpu(ctx, K, dev_i8(da), dev_i8(db), dev_i32(dc))
    var hc = Bytes(512)
    ctx.enqueue_copy(hc.i32(), dc)
    ctx.synchronize()
    for m in range(16):
        for n in range(8):
            var want = 0
            for k in range(K):
                want += Int(ha.i8()[unsafe_offset=m * K + k]) * Int(
                    hb.i8()[unsafe_offset=n * K + k]
                )
            assert_equal(
                Int(hc.i32()[unsafe_offset=m * 8 + n]),
                want,
                "k" + String(K) + " C[" + String(m) + ", " + String(n) + "]",
            )


def test_mma_k32_fragment_layout() raises:
    _check_mma(32)


def test_mma_k16_fragment_layout() raises:
    _check_mma(16)


def _check_quantize(n_tokens: Int, K: Int) raises:
    """GPU activation quantization against the formula, bit for bit."""
    var ctx = DeviceContext()
    var x = Floats(n_tokens * K)
    _fill(x, 21)
    # One all-zero block: its scale must be 0 and its values 0, not NaN.
    for i in range(32):
        x[32 + i] = 0
    var qa = GpuQActs(ctx, n_tokens, K)
    var dx = _up(ctx, x)
    quantize_gpu(ctx, dev_f32(dx), n_tokens, K, qa)
    var nb = n_tokens * K // 32
    var q = Bytes(n_tokens * K)
    ctx.enqueue_copy(q.i8(), qa.q.create_sub_buffer[DType.int8](0, q.size))
    var d = _down(ctx, qa.d, nb)
    var s = _down(ctx, qa.s, nb)
    var qp = q.i8()
    for b in range(nb):
        var amax = Float32(0)
        for i in range(32):
            amax = max(amax, abs(x[b * 32 + i]))
        var want_d = amax / 127.0
        var inv = Float32(0) if amax == 0 else 127.0 / amax
        var sum = 0
        for i in range(32):
            var v = Int(round(x[b * 32 + i] * inv))
            assert_equal(
                Int(qp[unsafe_offset=b * 32 + i]),
                v,
                "q block " + String(b) + " lane " + String(i),
            )
            sum += v
        assert_equal(d[b], want_d, "scale of block " + String(b))
        assert_equal(s[b], want_d * Float32(sum), "sum of block " + String(b))


def test_quantize_matches_formula() raises:
    _check_quantize(1, 256)
    _check_quantize(37, 512)


def _same_quant(
    ctx: DeviceContext,
    a: GpuQActs,
    b: GpuQActs,
    rows: Int,
    K: Int,
    what: String,
) raises:
    """Two quantized activation sets are identical: values, scales and sums."""
    var qa = Bytes(rows * K)
    var qb = Bytes(rows * K)
    ctx.enqueue_copy(qa.i8(), a.q.create_sub_buffer[DType.int8](0, qa.size))
    ctx.enqueue_copy(qb.i8(), b.q.create_sub_buffer[DType.int8](0, qb.size))
    var nb = rows * K // 32
    var da = _down(ctx, a.d, nb)
    var db = _down(ctx, b.d, nb)
    var sa = _down(ctx, a.s, nb)
    var sb = _down(ctx, b.s, nb)
    for i in range(rows * K):
        assert_equal(
            Int(qa.i8()[unsafe_offset=i]),
            Int(qb.i8()[unsafe_offset=i]),
            what + ": value " + String(i),
        )
    for i in range(nb):
        assert_equal(da[i], db[i], what + ": scale " + String(i))
        assert_equal(sa[i], sb[i], what + ": sum " + String(i))


def _check_rmsnorm_quantize(rows: Int, K: Int) raises:
    var ctx = DeviceContext()
    var x = Floats(rows * K)
    var w = Floats(K)
    _fill(x, 31)
    _fill(w, 32)
    var dw = _up(ctx, w)
    var dx = _up(ctx, x)
    var xn = ctx.enqueue_create_buffer[DType.float32](rows * K)
    var chain = GpuQActs(ctx, rows, K)
    rmsnorm_gpu(ctx, dev_f32(dx), dev_f32(dw), dev_f32(xn), rows, K, 1e-5)
    quantize_gpu(ctx, dev_f32(xn), rows, K, chain)
    var fused = GpuQActs(ctx, rows, K)
    rmsnorm_quantize_gpu(ctx, dev_f32(dx), dev_f32(dw), fused, rows, K, 1e-5)
    _same_quant(ctx, chain, fused, rows, K, "rmsnorm+quantize")


def test_fused_rmsnorm_quantize_matches_chain() raises:
    _check_rmsnorm_quantize(4, 1024)
    # Activation widths are multiples of 256 (GpuQActs), the block size, so the
    # quantize loop always runs whole passes: 768 takes three of them.
    _check_rmsnorm_quantize(5, 768)


def _check_add_rmsnorm_quantize(rows: Int, K: Int) raises:
    var ctx = DeviceContext()
    var x = Floats(rows * K)
    var h = Floats(rows * K)
    var w = Floats(K)
    _fill(x, 41)
    _fill(h, 42)
    _fill(w, 43)
    var dw = _up(ctx, w)
    var dh = _up(ctx, h)
    var x1 = _up(ctx, x)
    var xn = ctx.enqueue_create_buffer[DType.float32](rows * K)
    var chain = GpuQActs(ctx, rows, K)
    add_gpu(ctx, dev_f32(x1), dev_f32(dh), rows * K)
    rmsnorm_gpu(ctx, dev_f32(x1), dev_f32(dw), dev_f32(xn), rows, K, 1e-5)
    quantize_gpu(ctx, dev_f32(xn), rows, K, chain)
    var x2 = _up(ctx, x)
    var fused = GpuQActs(ctx, rows, K)
    add_rmsnorm_quantize_gpu(
        ctx, dev_f32(x2), dev_f32(dh), dev_f32(dw), fused, rows, K, 1e-5
    )
    _same_quant(ctx, chain, fused, rows, K, "add+rmsnorm+quantize")
    var a = _down(ctx, x1, rows * K)
    var b = _down(ctx, x2, rows * K)
    for i in range(rows * K):
        assert_equal(a[i], b[i], "residual " + String(i))


def test_fused_add_rmsnorm_quantize_matches_chain() raises:
    _check_add_rmsnorm_quantize(4, 1024)
    _check_add_rmsnorm_quantize(5, 768)


def test_fused_silu_mul_quantize_matches_chain() raises:
    var ctx = DeviceContext()
    var rows = 3
    var K = 768
    var g = Floats(rows * K)
    var u = Floats(rows * K)
    _fill(g, 51)
    _fill(u, 52)
    var du = _up(ctx, u)
    var g1 = _up(ctx, g)
    var chain = GpuQActs(ctx, rows, K)
    silu_mul_gpu(ctx, dev_f32(g1), dev_f32(du), rows * K)
    quantize_gpu(ctx, dev_f32(g1), rows, K, chain)
    var g2 = _up(ctx, g)
    var fused = GpuQActs(ctx, rows, K)
    silu_mul_quantize_gpu(ctx, dev_f32(g2), dev_f32(du), fused, rows, K)
    _same_quant(ctx, chain, fused, rows, K, "silu_mul+quantize")
    var left = _down(ctx, g2, rows * K)
    for i in range(rows * K):
        assert_equal(left[i], g[i], "g untouched " + String(i))


def _check_gemm(
    name: String, n_tokens: Int, rows: Int = 0, widest: Bool = False
) raises:
    """GPU int8 GEMM against an f32 GEMM over the reference dequantization."""
    var ctx = DeviceContext()
    var g = GGUFFile(FIX + "tiny-llada.gguf")
    var t = g.tensor(name)
    var K = t.ne0
    var all_rows = t.n_rows()
    # rows > 0 multiplies only the first `rows` weight rows, to cover a last
    # row tile that is partly outside the matrix.
    var N = rows if rows > 0 else all_rows
    var staged = Bytes(staged_matrix_bytes(t.ggml_type, all_rows, K))
    stage_matrix(g.data_addr(t), t.ggml_type, all_rows, K, staged.u8())
    var dw = ctx.enqueue_create_buffer[DType.uint8](staged.size)
    ctx.enqueue_copy(dw, staged.u8())
    var x = Floats(n_tokens * K)
    _fill(x, 11)
    var dx = _up(ctx, x)
    var qa = GpuQActs(ctx, n_tokens, K)
    quantize_gpu(ctx, dev_f32(dx), n_tokens, K, qa)
    var dout = ctx.enqueue_create_buffer[DType.float32](n_tokens * N)
    if widest:
        _launch[80](
            ctx, t.ggml_type, dev_u8(dw), N, K, qa, n_tokens, dev_f32(dout), N
        )
    else:
        gemm_gpu(
            ctx, t.ggml_type, dev_u8(dw), N, K, qa, n_tokens, dev_f32(dout), N
        )
    var got = _down(ctx, dout, n_tokens * N)
    var want = Floats(n_tokens * N)
    var w = Floats(K)
    for r in range(N):
        dequant_row(t.ggml_type, g.data_addr(t) + r * t.row_bytes(), w.ptr(), K)
        for m in range(n_tokens):
            var acc = Float64(0)
            for k in range(K):
                acc += Float64(w[k]) * Float64(x[m * K + k])
            want[m * N + r] = Float32(acc)
    var rel = _rel(got, want)
    assert_true(rel < 0.01, name + " relative error " + String(rel))


def test_gemm_q4k_ragged_tokens() raises:
    _check_gemm("blk.0.attn_q.weight", 7)


def test_gemm_q4k_many_tokens() raises:
    _check_gemm("blk.0.ffn_gate.weight", 37)


def test_gemm_q4k_partial_row_tile() raises:
    # 200 of 512 rows: the last row tile holds 72 rows, whatever its height.
    _check_gemm("blk.0.ffn_gate.weight", 37, 200)


def test_gemm_q6k_ragged_tokens() raises:
    _check_gemm("blk.0.attn_v.weight", 5)


def test_gemm_q6k_partial_row_tile() raises:
    # 100 of the rows: the second 64-row tile holds 36 rows.
    _check_gemm("blk.1.ffn_down.weight", 37, 100)


def test_gemm_q6k_two_superblocks() raises:
    _check_gemm("blk.1.ffn_down.weight", 19)


def test_gemm_q4k_widest_tile() raises:
    # Force the widest kernel even when device-aware dispatch prefers 32/64.
    assert_equal(_pick_bn(150), 80)
    _check_gemm("blk.0.ffn_gate.weight", 150, widest=True)


def test_gemm_q6k_widest_tile() raises:
    assert_equal(_pick_bn(150), 80)
    _check_gemm("blk.1.ffn_down.weight", 150)


def test_tile_width_rule() raises:
    assert_equal(_pick_bn(1), 32)
    assert_equal(_pick_bn(32), 32)
    assert_equal(_pick_bn(37), 64)
    assert_equal(_pick_bn(148), 80)
    assert_equal(_pick_bn(380), 80)


def test_q4_tile_width_uses_sm_waves() raises:
    # A 156-SM device can run 96 Q4_K blocks in one wave, but not 160.
    assert_equal(_pick_q4_bn(148, 4096, 156), 64)
    assert_equal(_pick_q4_bn(148, 12288, 156), 64)
    assert_equal(_pick_q4_bn(148, 1024, 156), 32)
    # Do not reduce tile width when it would add a wave.
    assert_equal(_pick_q4_bn(276, 4096, 156), 80)
    assert_equal(_pick_q4_bn(532, 4096, 156), 64)
    assert_equal(_pick_q4_bn(32, 126464, 156), 32)
    # Zero disables the new rule on devices outside its measured family.
    assert_equal(_pick_q4_bn(148, 4096, 0), 80)
    # Round up both the token and weight-row tile counts.
    assert_equal(_pick_q4_bn(148, 4097, 96), 80)


def test_q4_tile_widths_preserve_outputs() raises:
    var ctx = DeviceContext()
    var g = GGUFFile(FIX + "tiny-llada.gguf")
    var t = g.tensor("blk.0.ffn_gate.weight")
    var K = t.ne0
    var N = 200
    var staged = Bytes(staged_matrix_bytes(t.ggml_type, N, K))
    stage_matrix(g.data_addr(t), t.ggml_type, N, K, staged.u8())
    var dw = ctx.enqueue_create_buffer[DType.uint8](staged.size)
    ctx.enqueue_copy(dw, staged.u8())
    for rows in [7, 37, 148, 276]:
        var x = Floats(rows * K)
        _fill(x, 11)
        var dx = _up(ctx, x)
        var qa = GpuQActs(ctx, rows, K)
        quantize_gpu(ctx, dev_f32(dx), rows, K, qa)
        var a = ctx.enqueue_create_buffer[DType.float32](rows * N)
        var b = ctx.enqueue_create_buffer[DType.float32](rows * N)
        var c = ctx.enqueue_create_buffer[DType.float32](rows * N)
        _launch[32](ctx, t.ggml_type, dev_u8(dw), N, K, qa, rows, dev_f32(a), N)
        _launch[64](ctx, t.ggml_type, dev_u8(dw), N, K, qa, rows, dev_f32(b), N)
        _launch[80](ctx, t.ggml_type, dev_u8(dw), N, K, qa, rows, dev_f32(c), N)
        var x32 = _down(ctx, a, rows * N)
        var x64 = _down(ctx, b, rows * N)
        var x80 = _down(ctx, c, rows * N)
        for i in range(rows * N):
            assert_equal(x32[i], x64[i], "32 vs 64: " + String(i))
            assert_equal(x32[i], x80[i], "32 vs 80: " + String(i))


def _check_forward(path: String, vocab: Int, bound: Float64) raises:
    # The canvas tests/fixtures/*.expected.f32 was computed for.
    var toks: List[Int] = [5, 17, 300, 511, 511, 42, 511, 7]
    var rows = List[Int]()
    for i in range(len(toks)):
        rows.append(i)
    var cpu = DiffusionLM(FIX + path, threads=2, max_tokens=16)
    var want = Floats(len(toks) * vocab)
    cpu.forward(toks, rows, want.ptr())
    var gpu = GpuDiffusionLM(FIX + path, max_tokens=16)
    var got = Floats(len(toks) * vocab)
    gpu.forward(toks, rows, got.ptr())
    var expected = Mapping(FIX + path.replace(".gguf", ".expected.f32"))
    var ep = F32Ptr(unsafe_from_address=expected.addr)
    var ref_ = Floats(len(toks) * vocab)
    for i in range(ref_.n):
        ref_[i] = ep[unsafe_offset=i]
    var rel_gpu = _rel(got, ref_)
    var rel_cpu = _rel(want, ref_)
    print(
        path,
        "vs f32 reference: gpu",
        rel_gpu,
        "cpu",
        rel_cpu,
        "gpu-vs-cpu",
        _rel(got, want),
    )
    assert_true(
        rel_gpu < bound, path + " relative logit error " + String(rel_gpu)
    )
    for r in range(len(toks)):
        var bg = 0
        var bw = 0
        for v in range(vocab):
            if got[r * vocab + v] > got[r * vocab + bg]:
                bg = v
            if want[r * vocab + v] > want[r * vocab + bw]:
                bw = v
        assert_equal(bg, bw, path + " argmax differs at row " + String(r))


def test_forward_llada_matches_cpu() raises:
    # Attention runs in f16 on the GPU. This random-weight fixture is more
    # sensitive to f16 rounding of Q and K than a trained model: its error
    # against the f32 reference went from 0.017 (f32 attention) to 0.036. On
    # LLaDA-8B the f16 kernel tracks the CPU as closely as the f32 one did
    # (bench/checks/2026-10-13-attention-f16-logits.json).
    _check_forward("tiny-llada.gguf", 512, 0.05)


def test_forward_dream_matches_cpu() raises:
    _check_forward("tiny-dream.gguf", 512, 0.02)


def test_forward_subset_rows_match_full() raises:
    var toks: List[Int] = [5, 17, 300, 7, 9, 42, 100, 7]
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var some: List[Int] = [1, 4, 7]
    var m = GpuDiffusionLM(FIX + "tiny-llada.gguf", max_tokens=16)
    var full = Floats(len(toks) * 512)
    m.forward(toks, all_rows, full.ptr())
    var part = Floats(len(some) * 512)
    m.forward(toks, some, part.ptr())
    var worst = Float64(0)
    for j in range(len(some)):
        for v in range(512):
            worst = max(
                worst,
                abs(
                    Float64(part[j * 512 + v])
                    - Float64(full[some[j] * 512 + v])
                ),
            )
    assert_true(worst < 1e-3, "subset rows differ by " + String(worst))


def test_attention_more_keys_than_queries_matches_cpu() raises:
    # With the K/V cache, one block's rows attend over the whole canvas.
    var ctx = DeviceContext()
    var L = 9
    var NK = 40
    var heads = 4
    var kvh = 2
    var hd = 128
    var q = Floats(L * heads * hd)
    var k = Floats(NK * kvh * hd)
    var v = Floats(NK * kvh * hd)
    _fill(q, 7)
    _fill(k, 8)
    _fill(v, 9)
    for i in range(q.n):
        q[i] = q[i] * 0.1
    var want = Floats(L * heads * hd)
    attention(q.ptr(), k.ptr(), v.ptr(), want.ptr(), L, heads, kvh, hd, 2, NK)
    var dq = _up(ctx, q)
    var dk = _up(ctx, k)
    var dv = _up(ctx, v)
    var do_ = ctx.enqueue_create_buffer[DType.float32](want.n)
    attention_gpu(
        ctx,
        dev_f32(dq),
        dev_f32(dk),
        dev_f32(dv),
        dev_f32(do_),
        L,
        heads,
        kvh,
        hd,
        NK,
    )
    var got = _down(ctx, do_, want.n)
    # The f16 kernel's error against the f32 reference is relative to the
    # output scale, like the other attention tests (measured about 3e-4 there).
    var rel = _rel(got, want)
    assert_true(rel < 2e-3, "attention over cached keys, rel " + String(rel))


def _window_rows() -> List[Int]:
    var r: List[Int] = [3, 4, 6]
    return r^


def test_window_after_full_pass_matches_full() raises:
    # Right after a full pass the cache is fresh, so a window must give the
    # full pass's logits for its rows (up to the GEMM tile width, which
    # depends on the row count).
    var toks: List[Int] = [5, 17, 300, 511, 511, 42, 511, 7]
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var m = GpuDiffusionLM(
        FIX + "tiny-llada.gguf", max_tokens=16, kv_cache=True
    )
    var full = Floats(len(toks) * 512)
    m.forward(toks, all_rows, full.ptr())
    var win = _window_rows()
    var part = Floats(len(win) * 512)
    m.forward_window(toks, 3, 7, win, part.ptr())
    var worst = Float64(0)
    for j in range(len(win)):
        for v in range(512):
            worst = max(
                worst,
                abs(
                    Float64(part[j * 512 + v]) - Float64(full[win[j] * 512 + v])
                ),
            )
    assert_true(worst < 1e-3, "window rows differ by " + String(worst))


def test_window_on_a_stale_cache_matches_cpu() raises:
    # The case the sampler runs: a full pass, then tokens inside the window
    # change and only the window is recomputed. Keys outside it are stale on
    # both devices, so the GPU must agree with the CPU window.
    var toks: List[Int] = [5, 17, 300, 511, 511, 42, 511, 7]
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var cpu = DiffusionLM(
        FIX + "tiny-llada.gguf", threads=2, max_tokens=16, kv_cache=True
    )
    var gpu = GpuDiffusionLM(
        FIX + "tiny-llada.gguf", max_tokens=16, kv_cache=True
    )
    var scratch = Floats(len(toks) * 512)
    cpu.forward(toks, all_rows, scratch.ptr())
    gpu.forward(toks, all_rows, scratch.ptr())
    toks[3] = 250
    toks[6] = 99
    var win = _window_rows()
    var want = Floats(len(win) * 512)
    var got = Floats(len(win) * 512)
    cpu.forward_window(toks, 3, 7, win, want.ptr())
    gpu.forward_window(toks, 3, 7, win, got.ptr())
    var rel = _rel(got, want)
    assert_true(rel < 0.02, "stale-cache window gpu vs cpu " + String(rel))
    for j in range(len(win)):
        var bg = 0
        var bw = 0
        for v in range(512):
            if got[j * 512 + v] > got[j * 512 + bg]:
                bg = v
            if want[j * 512 + v] > want[j * 512 + bw]:
                bw = v
        assert_equal(bg, bw, "argmax differs at window row " + String(win[j]))


def test_a_failed_gpu_pass_invalidates_the_cache() raises:
    # The cache is marked valid only after the pass has synchronized, so a
    # pass that raises leaves a cache the next window refuses.
    var toks: List[Int] = [5, 17, 300, 511, 511, 42, 511, 7]
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var win = _window_rows()
    var full = Floats(len(toks) * 512)
    var out = Floats(len(win) * 512)
    var gpu = GpuDiffusionLM(
        FIX + "tiny-llada.gguf", max_tokens=16, kv_cache=True
    )
    gpu.forward(toks, all_rows, full.ptr())
    var bad = toks.copy()
    bad[5] = 512
    with assert_raises(contains="outside the vocabulary"):
        gpu.forward(bad, all_rows, full.ptr())
    with assert_raises(contains="run a full pass first"):
        gpu.forward_window(toks, 3, 7, win, out.ptr())
    gpu.forward(toks, all_rows, full.ptr())
    with assert_raises(contains="outside the vocabulary"):
        gpu.forward_window(bad, 3, 7, win, out.ptr())
    with assert_raises(contains="run a full pass first"):
        gpu.forward_window(toks, 3, 7, win, out.ptr())


def _close(got: Float64, want: Float64, rel: Float64, floor: Float64) -> Bool:
    return abs(got - want) <= rel * abs(want) + floor


def _check_row_stats(vocab: Int, n_rows: Int, seed: Int) raises:
    """The device reductions give the host's token and confidences, row by row.

    The grid of values _fill draws from repeats, so every row holds its maximum
    several times, and row 1 gets two extra copies of a larger value: the
    first index must win a tie.
    """
    var ctx = DeviceContext()
    var x = Floats(n_rows * vocab)
    _fill(x, seed)
    x[vocab + 5] = 9.0
    x[vocab + vocab // 2] = 9.0
    var dx = _up(ctx, x)
    var tok = ctx.enqueue_create_buffer[DType.int32](n_rows)
    var val = ctx.enqueue_create_buffer[DType.float64](RS_FIELDS * n_rows)
    row_stats_gpu(ctx, dev_f32(dx), n_rows, vocab, dev_i32(tok), dev_f64(val))
    var htok = Bytes(4 * n_rows)
    var hval = Bytes(8 * RS_FIELDS * n_rows)
    ctx.enqueue_copy(htok.i32(), tok)
    ctx.enqueue_copy(hval.f64(), val)
    ctx.synchronize()
    var st = RowStats()
    for j in range(n_rows):
        st.tok.append(Int(htok.i32()[unsafe_offset=j]))
        st.mx.append(hval.f64()[unsafe_offset=RS_FIELDS * j])
        st.z.append(hval.f64()[unsafe_offset=RS_FIELDS * j + 1])
        st.second.append(hval.f64()[unsafe_offset=RS_FIELDS * j + 2])
        st.weighted.append(hval.f64()[unsafe_offset=RS_FIELDS * j + 3])
    assert_equal(st.tok[1], 5)
    for j in range(n_rows):
        var row = x.ptr().unsafe_offset(j * vocab)
        var want = argmax_and_prob(row, vocab)
        var at = " (row " + String(j) + ", vocab " + String(vocab) + ")"
        assert_equal(st.tok[j], want[0], "argmax" + at)
        assert_true(
            _close(_confidence_from_stats(st, j, "llada"), want[1], 1e-5, 0.0),
            "probability" + at,
        )
        for algo in ["maskgit_plus", "topk_margin", "entropy"]:
            var w = _confidence(row, vocab, algo)
            assert_equal(st.tok[j], w[0], algo + " token" + at)
            assert_true(
                _close(_confidence_from_stats(st, j, algo), w[1], 1e-5, 1e-5),
                algo + " confidence" + at,
            )


def test_row_stats_match_host_small_vocab() raises:
    _check_row_stats(1000, 5, 3)


def test_row_stats_match_host_full_vocab() raises:
    _check_row_stats(126464, 3, 7)


def test_forward_stats_match_forward_logits() raises:
    var toks: List[Int] = [5, 17, 300, 7, 9, 42, 100, 7]
    var rows: List[Int] = [1, 4, 7]
    var m = GpuDiffusionLM(FIX + "tiny-llada.gguf", max_tokens=16)
    var lg = Floats(len(rows) * 512)
    m.forward(toks, rows, lg.ptr())
    var st = RowStats()
    m.forward_stats(toks, rows, st)
    assert_equal(len(st.tok), len(rows))
    for j in range(len(rows)):
        var want = argmax_and_prob(lg.ptr().unsafe_offset(j * 512), 512)
        assert_equal(st.tok[j], want[0])
        assert_true(
            _close(_confidence_from_stats(st, j, "llada"), want[1], 1e-5, 0.0),
            "probability of row " + String(j),
        )


def test_generation_with_device_stats_matches_host_scoring() raises:
    var m = GpuDiffusionLM(FIX + "tiny-llada.gguf", max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = GenConfig()
    c.gen_length = 8
    c.block_length = 4
    c.steps = 4
    c.mask_id = 511
    var device = generate(m, prompt, c)
    # full_logits copies every row to the host and scores it there.
    var h = c.copy()
    h.full_logits = True
    var host = generate(m, prompt, h)
    assert_equal(len(device.tokens), len(host.tokens))
    for i in range(len(host.tokens)):
        assert_equal(device.tokens[i], host.tokens[i], "token " + String(i))


def test_dual_cache_generation_on_the_host_logits_path() raises:
    # One step per block makes every pass a full refresh, so the cached run
    # (forward_window, logits on the host) matches the uncached run (device
    # row statistics). The sampler must not hand forward_window the one-float
    # logits buffer that the statistics path allocates.
    var prompt: List[Int] = [5, 17, 300]
    var c = GenConfig()
    c.gen_length = 8
    c.block_length = 4
    c.steps = 2
    c.mask_id = 511
    var plain = GpuDiffusionLM(FIX + "tiny-llada.gguf", max_tokens=16)
    var a = generate(plain, prompt, c)
    var cached = GpuDiffusionLM(
        FIX + "tiny-llada.gguf", max_tokens=16, kv_cache=True
    )
    c.dual_cache = True
    var b = generate(cached, prompt, c)
    assert_equal(a.forward_passes, b.forward_passes)
    assert_equal(len(a.tokens), len(b.tokens))
    for i in range(len(a.tokens)):
        assert_equal(a.tokens[i], b.tokens[i], "token " + String(i))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
