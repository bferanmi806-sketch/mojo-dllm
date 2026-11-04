"""Quantized GEMM: f32 activations x Q4_K / Q6_K weights.

A denoising step pushes the whole canvas (prompt + generation, L tokens)
through every linear layer, so this is a GEMM with L in the tens to hundreds,
not the GEMV an autoregressive decoder runs.

Activations are quantized per token to int8 with one f32 scale per 256 values
(llama.cpp's Q8_K scheme), and the per-16 sums are kept as floats. Weight
blocks are decoded in registers, never into memory: a Q4_K or Q6_K sub-block
becomes 32 unsigned bytes, `vpmaddubsw` multiplies them against the int8
activations into int16 pairs, and `vpdpwssd` (AVX-VNNI) multiplies those by the
integer block scale and accumulates into int32 in one instruction. That is two
instructions per 32 multiply-adds, with no horizontal sum until a row ends.

The zero point of each format becomes an offset dotted with the activation
sums:

  Q4_K:  w = d*sc*q - dmin*m       -> sum(w*x) = d*da*sum(sc*q*a) - sum_g (dmin*m_g) * s_g
  Q6_K:  w = d*sc*(u - 32)         -> sum(w*x) = d*da*sum(sc*u*a) - sum_g (32*d*sc_g) * s_g

where `a` are the int8 activations, `da` their scale, and `s_g = da * sum(a)`
over group g of 16.
"""

from std.math import round
from std.sys import CompilationTarget, llvm_intrinsic
from max.algorithm import parallelize

from mojo_dllm.formats.gguf import GGML_Q4_K, GGML_Q6_K, ggml_type_name
from mojo_dllm.quant.kquants import load_f16, q4k_scale_min
from mojo_dllm.sys.mem import Bytes, Floats, F32Ptr, I8Ptr, U8Ptr

comptime QK = 256
comptime GROUP = 16
comptime ROWS_PER_ITEM = 8
comptime TOK_TILE = 4


struct QActs(Movable):
    """Int8 activations for `n_tokens` rows of length `K`."""

    var n_tokens: Int
    var K: Int
    var block_size: Int
    var q: Bytes
    var d: Floats
    """One scale per block_size values (256 for K-quants, 32 for Q5_0/Q8_0)."""
    var s: Floats
    """d * sum(q) per 16 values."""

    def __init__(out self, n_tokens: Int, K: Int, block_size: Int = QK) raises:
        if block_size != 32 and block_size != QK:
            raise Error(
                "unsupported activation block size " + String(block_size)
            )
        if n_tokens < 0 or K <= 0:
            raise Error(
                "activation dimensions need nonnegative tokens and positive K"
            )
        if K % block_size != 0:
            raise Error(
                "activation length "
                + String(K)
                + " is not a multiple of "
                + String(block_size)
            )
        self.n_tokens = n_tokens
        self.K = K
        self.block_size = block_size
        self.q = Bytes(n_tokens * K)
        self.d = Floats(n_tokens * (K // block_size))
        self.s = Floats(n_tokens * (K // GROUP))


def quantize_acts(x: F32Ptr, n_tokens: Int, K: Int, mut qa: QActs) raises:
    if K != qa.K or n_tokens < 0 or n_tokens > qa.n_tokens:
        raise Error("activation buffer is too small for this matrix")
    if qa.block_size == 32:
        _quantize_acts[32](x, n_tokens, K, qa)
    else:
        _quantize_acts[QK](x, n_tokens, K, qa)


def _quantize_acts[
    BLOCK: Int
](x: F32Ptr, n_tokens: Int, K: Int, mut qa: QActs) raises:
    var qp = qa.q.i8()
    var dp = qa.d.ptr()
    var sp = qa.s.ptr()
    var xp = x
    var nb = K // BLOCK

    def work(m: Int) {imm}:
        for b in range(nb):
            var src = xp.unsafe_offset(m * K + b * BLOCK)
            var amax = SIMD[DType.float32, 8](0)
            for i in range(0, BLOCK, 8):
                amax = max(amax, abs(src.unsafe_load[width=8](i)))
            var mx = amax.reduce_max()
            var d = mx / 127.0
            var inv = Float32(127.0) / mx if mx > 0 else Float32(0)
            dp.unsafe_store(m * nb + b, d)
            var dst = qp.unsafe_offset(m * K + b * BLOCK)
            for g in range(BLOCK // GROUP):
                var v0 = round(src.unsafe_load[width=8](g * 16) * inv)
                var v1 = round(src.unsafe_load[width=8](g * 16 + 8) * inv)
                var q0 = v0.cast[DType.int32]()
                var q1 = v1.cast[DType.int32]()
                dst.unsafe_store(g * 16, q0.cast[DType.int8]())
                dst.unsafe_store(g * 16 + 8, q1.cast[DType.int8]())
                sp.unsafe_store(
                    m * (K // GROUP) + b * (BLOCK // GROUP) + g,
                    d * Float32((q0 + q1).reduce_add()),
                )

    parallelize(work, n_tokens)


@always_inline
def _maddubs(
    w: SIMD[DType.uint8, 32], a: SIMD[DType.int8, 32]
) -> SIMD[DType.int16, 16]:
    return llvm_intrinsic["llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]](
        w, a
    )


@always_inline
def _dpwssd(
    acc: SIMD[DType.int32, 8],
    p: SIMD[DType.int16, 16],
    s: SIMD[DType.int16, 16],
) -> SIMD[DType.int32, 8]:
    comptime if CompilationTarget.has_vnni():
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpwssd.256", SIMD[DType.int32, 8]
        ](acc, p, s)
    else:
        return acc + llvm_intrinsic[
            "llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]
        ](p, s)


@always_inline
def _pair_scale(a: Int16, b: Int16) -> SIMD[DType.int16, 16]:
    """Scale vector for 32 weights spanning two 16-groups."""
    return SIMD[DType.int16, 8](a).join(SIMD[DType.int16, 8](b))


@always_inline
def _q4k_tile[
    T: Int
](
    wrow: U8Ptr,
    nsb: Int,
    qa_q: I8Ptr,
    qa_d: F32Ptr,
    qa_s: F32Ptr,
    K: Int,
    tok0: Int,
) -> SIMD[DType.float32, T]:
    var facc = Array[SIMD[DType.float32, 8], T](fill=SIMD[DType.float32, 8](0))
    for sb in range(nsb):
        var blk = wrow.unsafe_offset(sb * 144)
        var d = load_f16(blk, 0)
        var dmin = load_f16(blk, 2)
        var scp = blk.unsafe_offset(4)
        var sc = SIMD[DType.int16, 8](0)
        var mn = SIMD[DType.float32, 8](0)
        comptime for j in range(8):
            var r = q4k_scale_min(scp, j)
            sc[j] = Int16(r[0])
            mn[j] = Float32(r[1])
        mn = mn * dmin
        var off_lo = SIMD[DType.float32, 8](
            mn[0], mn[0], mn[1], mn[1], mn[2], mn[2], mn[3], mn[3]
        )
        var off_hi = SIMD[DType.float32, 8](
            mn[4], mn[4], mn[5], mn[5], mn[6], mn[6], mn[7], mn[7]
        )
        var iacc = Array[SIMD[DType.int32, 8], T](fill=SIMD[DType.int32, 8](0))
        comptime for j in range(4):
            var qv = blk.unsafe_offset(16 + 32 * j).unsafe_load[width=32]()
            var lo = qv & 0x0F
            var hi = qv >> 4
            var s_lo = SIMD[DType.int16, 16](sc[2 * j])
            var s_hi = SIMD[DType.int16, 16](sc[2 * j + 1])
            comptime for t in range(T):
                var ap = qa_q.unsafe_offset((tok0 + t) * K + sb * QK + 64 * j)
                iacc[t] = _dpwssd(
                    iacc[t], _maddubs(lo, ap.unsafe_load[width=32](0)), s_lo
                )
                iacc[t] = _dpwssd(
                    iacc[t], _maddubs(hi, ap.unsafe_load[width=32](32)), s_hi
                )
        comptime for t in range(T):
            var tok = tok0 + t
            var da = qa_d.unsafe_load(tok * (K // QK) + sb)
            var sp = qa_s.unsafe_offset(tok * (K // GROUP) + sb * 16)
            facc[t] = (
                facc[t]
                + iacc[t].cast[DType.float32]() * (d * da)
                - off_lo * sp.unsafe_load[width=8](0)
                - off_hi * sp.unsafe_load[width=8](8)
            )
    var res = SIMD[DType.float32, T](0)
    comptime for t in range(T):
        res[t] = facc[t].reduce_add()
    return res


@always_inline
def _q6k_tile[
    T: Int
](
    wrow: U8Ptr,
    nsb: Int,
    qa_q: I8Ptr,
    qa_d: F32Ptr,
    qa_s: F32Ptr,
    K: Int,
    tok0: Int,
) -> SIMD[DType.float32, T]:
    var facc = Array[SIMD[DType.float32, 8], T](fill=SIMD[DType.float32, 8](0))
    for sb in range(nsb):
        var blk = wrow.unsafe_offset(sb * 210)
        var d = load_f16(blk, 208)
        var scb = blk.unsafe_offset(192).unsafe_load[width=16]()
        var sc = scb.cast[DType.int8]().cast[DType.int16]()
        var scf = sc.cast[DType.float32]() * (32.0 * d)
        var off_lo = scf.slice[8, offset=0]()
        var off_hi = scf.slice[8, offset=8]()
        var iacc = Array[SIMD[DType.int32, 8], T](fill=SIMD[DType.int32, 8](0))
        comptime for h in range(2):
            var qla = blk.unsafe_offset(64 * h).unsafe_load[width=32]()
            var qlb = blk.unsafe_offset(64 * h + 32).unsafe_load[width=32]()
            var qh = blk.unsafe_offset(128 + 32 * h).unsafe_load[width=32]()
            var c0 = (qla & 0x0F) | ((qh & 3) << 4)
            var c1 = (qlb & 0x0F) | (((qh >> 2) & 3) << 4)
            var c2 = (qla >> 4) | (((qh >> 4) & 3) << 4)
            var c3 = (qlb >> 4) | (((qh >> 6) & 3) << 4)
            var s0 = _pair_scale(sc[8 * h + 0], sc[8 * h + 1])
            var s1 = _pair_scale(sc[8 * h + 2], sc[8 * h + 3])
            var s2 = _pair_scale(sc[8 * h + 4], sc[8 * h + 5])
            var s3 = _pair_scale(sc[8 * h + 6], sc[8 * h + 7])
            comptime for t in range(T):
                var ap = qa_q.unsafe_offset((tok0 + t) * K + sb * QK + 128 * h)
                iacc[t] = _dpwssd(
                    iacc[t], _maddubs(c0, ap.unsafe_load[width=32](0)), s0
                )
                iacc[t] = _dpwssd(
                    iacc[t], _maddubs(c1, ap.unsafe_load[width=32](32)), s1
                )
                iacc[t] = _dpwssd(
                    iacc[t], _maddubs(c2, ap.unsafe_load[width=32](64)), s2
                )
                iacc[t] = _dpwssd(
                    iacc[t], _maddubs(c3, ap.unsafe_load[width=32](96)), s3
                )
        comptime for t in range(T):
            var tok = tok0 + t
            var da = qa_d.unsafe_load(tok * (K // QK) + sb)
            var sp = qa_s.unsafe_offset(tok * (K // GROUP) + sb * 16)
            facc[t] = (
                facc[t]
                + iacc[t].cast[DType.float32]() * (d * da)
                - off_lo * sp.unsafe_load[width=8](0)
                - off_hi * sp.unsafe_load[width=8](8)
            )
    var res = SIMD[DType.float32, T](0)
    comptime for t in range(T):
        res[t] = facc[t].reduce_add()
    return res


@always_inline
def _row[
    kind: Int
](
    wrow: U8Ptr,
    nsb: Int,
    qa: QActs,
    n_tokens: Int,
    dst: F32Ptr,
    ldo: Int,
    n: Int,
):
    var qq = qa.q.i8()
    var qd = qa.d.ptr()
    var qs = qa.s.ptr()
    var K = qa.K
    var t = 0
    while t + TOK_TILE <= n_tokens:
        var r: SIMD[DType.float32, TOK_TILE]
        comptime if kind == GGML_Q4_K:
            r = _q4k_tile[TOK_TILE](wrow, nsb, qq, qd, qs, K, t)
        else:
            r = _q6k_tile[TOK_TILE](wrow, nsb, qq, qd, qs, K, t)
        comptime for i in range(TOK_TILE):
            dst.unsafe_store((t + i) * ldo + n, r[i])
        t += TOK_TILE
    while t < n_tokens:
        var r: SIMD[DType.float32, 1]
        comptime if kind == GGML_Q4_K:
            r = _q4k_tile[1](wrow, nsb, qq, qd, qs, K, t)
        else:
            r = _q6k_tile[1](wrow, nsb, qq, qd, qs, K, t)
        dst.unsafe_store(t * ldo + n, r[0])
        t += 1


def qgemm(
    w_addr: Int,
    w_type: Int,
    N: Int,
    K: Int,
    qa: QActs,
    n_tokens: Int,
    dst: F32Ptr,
    ldo: Int,
    threads: Int,
) raises:
    """out[m * ldo + n] = sum_k W[n, k] * x[m, k] for n < N, m < n_tokens."""
    if qa.block_size != QK:
        raise Error("qgemm requires activation block size 256")
    if K != qa.K:
        raise Error(
            "activation length "
            + String(qa.K)
            + " does not match weight rows of "
            + String(K)
        )
    if K % QK != 0:
        raise Error(
            "weight row length " + String(K) + " is not a multiple of 256"
        )
    var nsb = K // QK
    var row_bytes: Int
    if w_type == GGML_Q4_K:
        row_bytes = nsb * 144
    elif w_type == GGML_Q6_K:
        row_bytes = nsb * 210
    else:
        raise Error(
            "qgemm does not support weight type " + ggml_type_name(w_type)
        )
    var items = (N + ROWS_PER_ITEM - 1) // ROWS_PER_ITEM
    var is_q4 = w_type == GGML_Q4_K

    def work(item: Int) {imm}:
        var r0 = item * ROWS_PER_ITEM
        var r1 = min(r0 + ROWS_PER_ITEM, N)
        for n in range(r0, r1):
            var wrow = U8Ptr(unsafe_from_address=w_addr + n * row_bytes)
            if is_q4:
                _row[GGML_Q4_K](wrow, nsb, qa, n_tokens, dst, ldo, n)
            else:
                _row[GGML_Q6_K](wrow, nsb, qa, n_tokens, dst, ldo, n)

    parallelize(work, items, threads)
