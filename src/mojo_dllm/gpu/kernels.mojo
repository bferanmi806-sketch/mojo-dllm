"""GPU kernels for the masked-diffusion forward pass (NVIDIA, via Mojo's DeviceContext).

Each kernel mirrors a CPU function in `kernels/ops.mojo` or `kernels/qgemm.mojo`,
and tests/gpu/test_gpu.mojo compares them one to one.

The linear layers read GGUF K-quant blocks in place. Activations are quantized
to int8 with one f32 scale per 32 values (llama.cpp's Q8_1 shape), so a
32-weight sub-block of either format meets exactly one activation scale:

    Q4_K sub-block j:   sum_i w_i x_i = d*sc_j*dx * sum_i q_i qx_i  -  dmin*m_j * (dx * sum_i qx_i)
    Q6_K 16-group g:    sum_i w_i x_i = d*sc_g*dx * sum_i (q_i - 32) qx_i

The integer sums run on the int8 tensor cores (`mma.sync`, sm_80 and newer).
Q6_K blocks are 210 bytes, which leaves every other block 2-byte aligned;
upload pads them to 224, so every 16-byte chunk of a block is aligned.
"""

from std.math import exp, sqrt, round
from std.memory import AddressSpace, bitcast, unsafe_stack_allocation
from std.sys import llvm_intrinsic
from std.sys.intrinsics import _RegisterPackType

from max.gpu import barrier, block_dim, block_idx, lane_id, thread_idx, warp_id
from max.gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext
from max.gpu.primitives.warp import (
    max as warp_max,
    shuffle_idx,
    shuffle_xor,
    sum as warp_sum,
)

from mojo_dllm.formats.gguf import GGML_Q4_K, GGML_Q6_K, ggml_type_name
from mojo_dllm.sys.mem import F32Ptr, F64Ptr, I8Ptr, I32Ptr, U8Ptr

comptime QK = 32
"""Activation values per int8 scale."""
comptime Q4K_BYTES = 144
comptime Q6K_BYTES = 210
comptime Q6K_STAGED = 224
comptime NT = 256
comptime SharedF32 = Pointer[
    Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]
comptime SharedF64 = Pointer[
    Float64, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@always_inline
def dev_f32(b: DeviceBuffer[DType.float32]) -> F32Ptr:
    """A device buffer's address as the untracked pointer kernels take."""
    return F32Ptr(unsafe_from_address=Int(b.unsafe_ptr()))


@always_inline
def dev_u8(b: DeviceBuffer[DType.uint8]) -> U8Ptr:
    return U8Ptr(unsafe_from_address=Int(b.unsafe_ptr()))


@always_inline
def dev_i8(b: DeviceBuffer[DType.int8]) -> I8Ptr:
    return I8Ptr(unsafe_from_address=Int(b.unsafe_ptr()))


@always_inline
def dev_i32(b: DeviceBuffer[DType.int32]) -> I32Ptr:
    return I32Ptr(unsafe_from_address=Int(b.unsafe_ptr()))


@always_inline
def dev_f64(b: DeviceBuffer[DType.float64]) -> F64Ptr:
    return F64Ptr(unsafe_from_address=Int(b.unsafe_ptr()))


@always_inline
def _f16(p: U8Ptr, off: Int) -> Float32:
    var bits = UInt16(p[unsafe_offset=off]) | (
        UInt16(p[unsafe_offset=off + 1]) << 8
    )
    return bitcast[DType.float16](bits).cast[DType.float32]()


@always_inline
def _cdiv(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def _block_sum(v: Float32, red: SharedF32) -> Float32:
    """Sum over a block of NT threads; every thread gets the total."""
    var s = warp_sum(v)
    if lane_id() == 0:
        red[unsafe_offset=Int(warp_id())] = s
    barrier()
    var t = Float32(0)
    for w in range(NT // 32):
        t += red[unsafe_offset=w]
    barrier()
    return t


# ---------------------------------------------------------------- RMSNorm


def _rmsnorm_kernel(x: F32Ptr, w: F32Ptr, y: F32Ptr, dim: Int32, eps: Float32):
    var red = unsafe_stack_allocation[
        NT // 32, Float32, address_space=AddressSpace.SHARED
    ]()
    var d = Int(dim)
    var row = x.unsafe_offset(Int(block_idx.x) * d)
    var out = y.unsafe_offset(Int(block_idx.x) * d)
    var ss = Float32(0)
    var i = Int(thread_idx.x)
    while i < d:
        var v = row[unsafe_offset=i]
        ss += v * v
        i += NT
    var total = _block_sum(ss, red)
    var inv = 1.0 / sqrt(total / Float32(d) + eps)
    i = Int(thread_idx.x)
    while i < d:
        out[unsafe_offset=i] = row[unsafe_offset=i] * inv * w[unsafe_offset=i]
        i += NT


def rmsnorm_gpu(
    ctx: DeviceContext,
    x: F32Ptr,
    w: F32Ptr,
    y: F32Ptr,
    rows: Int,
    dim: Int,
    eps: Float32,
) raises:
    ctx.enqueue_function[_rmsnorm_kernel](
        x, w, y, Int32(dim), eps, grid_dim=rows, block_dim=NT
    )


# ---------------------------------------------------------------- RoPE


def _rope_kernel(
    x: F32Ptr,
    n: Int32,
    heads: Int32,
    hd: Int32,
    c: F32Ptr,
    s: F32Ptr,
    neox: Int32,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n):
        return
    var half = Int(hd) // 2
    var i = idx % half
    var th = idx // half
    var t = th // Int(heads)
    var base = x.unsafe_offset(th * Int(hd))
    var ia = i if neox != 0 else 2 * i
    var ib = ia + (half if neox != 0 else 1)
    var cv = c[unsafe_offset=t * half + i]
    var sv = s[unsafe_offset=t * half + i]
    var a = base[unsafe_offset=ia]
    var b = base[unsafe_offset=ib]
    base[unsafe_offset=ia] = a * cv - b * sv
    base[unsafe_offset=ib] = a * sv + b * cv


def rope_gpu(
    ctx: DeviceContext,
    x: F32Ptr,
    rows: Int,
    heads: Int,
    head_dim: Int,
    cos_t: F32Ptr,
    sin_t: F32Ptr,
    neox: Bool,
) raises:
    var n = rows * heads * head_dim // 2
    ctx.enqueue_function[_rope_kernel](
        x,
        Int32(n),
        Int32(heads),
        Int32(head_dim),
        cos_t,
        sin_t,
        Int32(1 if neox else 0),
        grid_dim=_cdiv(n, NT),
        block_dim=NT,
    )


# ---------------------------------------------------------------- attention

comptime AQ = 64
"""Queries per attention block: 16 per warp, 4 warps."""
comptime AK = 32
"""Keys per shared-memory tile."""
comptime ATT_THREADS = 128
comptime KPAD = 8
"""Halves of padding per shared K row, so the 8 rows a fragment load reads
start 4 banks apart."""
comptime VPAD = 8
"""Halves of padding per shared transposed-V row, for the same reason."""
comptime H2 = SIMD[DType.float16, 2]


@always_inline
def _mma_f16(
    a0: H2,
    a1: H2,
    a2: H2,
    a3: H2,
    b0: H2,
    b1: H2,
    c: SIMD[DType.float32, 4],
) -> SIMD[DType.float32, 4]:
    """c + A B for one m16n8k16 tile: f16 inputs, f32 accumulation."""
    var r = llvm_intrinsic[
        "llvm.nvvm.mma.m16n8k16.row.col.f32.f32",
        _RegisterPackType[Float32, Float32, Float32, Float32],
    ](a0, a1, a2, a3, b0, b1, c[0], c[1], c[2], c[3])
    return SIMD[DType.float32, 4](r[0], r[1], r[2], r[3])


@always_inline
def _h2(x: Float32, y: Float32) -> H2:
    return H2(x.cast[DType.float16](), y.cast[DType.float16]())


def _attention_kernel[
    HD: Int
](
    q: F32Ptr,
    k: F32Ptr,
    v: F32Ptr,
    o: F32Ptr,
    L: Int32,
    Lk: Int32,
    heads: Int32,
    kv_heads: Int32,
    scale: Float32,
):
    """Flash attention on the f16 tensor cores, f32 accumulation.

    Warp w owns queries 16w..16w+15 of the block's AQ. Per tile of AK keys it
    computes S = Q K^T with m16n8k16 mmas, folds S into an online softmax,
    and adds P V with more mmas. The S accumulators of keys 16c..16c+15 are
    laid out exactly as the A operand of the P V mma, so P never leaves the
    registers. Thread (g, t) of a warp holds rows g and g + 8 of its tile.
    """
    comptime KC = HD // 16
    """k16 chunks of the head dimension (S = Q K^T)."""
    comptime DN = HD // 8
    """n8 tiles of the head dimension (O = P V)."""
    comptime SN = AK // 8
    """n8 tiles of keys in S."""
    comptime KS = HD + KPAD
    comptime VS = AK + VPAD
    var sK = unsafe_stack_allocation[
        AK * KS, Float16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var sVt = unsafe_stack_allocation[
        HD * VS, Float16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var nq = Int(L)
    var nk = Int(Lk)
    var h = Int(block_idx.y)
    var kvh = h // (Int(heads) // Int(kv_heads))
    var stride = Int(heads) * HD
    var kv_stride = Int(kv_heads) * HD
    var th = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp = Int(warp_id())
    var g = lane >> 2
    var t = lane & 3
    var r0 = Int(block_idx.x) * AQ + 16 * warp + g
    var r1 = r0 + 8
    # Q fragments for the whole head dimension, scaled, kept in registers.
    var qa = Array[H2, 4 * KC](fill=H2(0))
    comptime for c in range(KC):
        comptime for e in range(4):
            var r = r0 if (e & 1) == 0 else r1
            var col = c * 16 + 2 * t + (8 if e >= 2 else 0)
            if r < nq:
                var src = q.unsafe_offset(r * stride + h * HD + col)
                qa[c * 4 + e] = _h2(
                    src[unsafe_offset=0] * scale, src[unsafe_offset=1] * scale
                )
    var m0 = Float32.MIN
    var m1 = Float32.MIN
    var l0 = Float32(0)
    var l1 = Float32(0)
    var acc = Array[SIMD[DType.float32, 4], DN](fill=SIMD[DType.float32, 4](0))
    var kt = 0
    while kt < nk:
        barrier()
        # Stage AK keys: K row-major, V transposed, both as f16.
        comptime for i in range(AK * HD // ATT_THREADS):
            var idx = th + i * ATT_THREADS
            var j = idx // HD
            var d = idx % HD
            var kv_ = Float32(0)
            var vv = Float32(0)
            if kt + j < nk:
                var src = (kt + j) * kv_stride + kvh * HD + d
                kv_ = k[unsafe_offset=src]
                vv = v[unsafe_offset=src]
            sK[unsafe_offset=j * KS + d] = kv_.cast[DType.float16]()
            sVt[unsafe_offset=d * VS + j] = vv.cast[DType.float16]()
        barrier()
        # S = Q K^T for this warp's 16 queries and AK keys.
        var s = Array[SIMD[DType.float32, 4], SN](
            fill=SIMD[DType.float32, 4](0)
        )
        comptime for n in range(SN):
            var krow = sK.unsafe_offset((n * 8 + g) * KS + 2 * t)
            comptime for c in range(KC):
                var b0 = krow.unsafe_offset(c * 16).unsafe_load[
                    width=2, alignment=4
                ]()
                var b1 = krow.unsafe_offset(c * 16 + 8).unsafe_load[
                    width=2, alignment=4
                ]()
                s[n] = _mma_f16(
                    qa[c * 4],
                    qa[c * 4 + 1],
                    qa[c * 4 + 2],
                    qa[c * 4 + 3],
                    b0,
                    b1,
                    s[n],
                )
        # Online softmax over rows g (lanes 0, 1 of s[n]) and g + 8 (2, 3).
        var t0 = Float32.MIN
        var t1 = Float32.MIN
        comptime for n in range(SN):
            var sn = s[n]
            comptime for e in range(4):
                if kt + n * 8 + 2 * t + (e & 1) >= nk:
                    sn[e] = Float32.MIN
            s[n] = sn
            t0 = max(t0, max(sn[0], sn[1]))
            t1 = max(t1, max(sn[2], sn[3]))
        # The four threads of a quad hold one row between them.
        t0 = max(t0, shuffle_xor(t0, 1))
        t0 = max(t0, shuffle_xor(t0, 2))
        t1 = max(t1, shuffle_xor(t1, 1))
        t1 = max(t1, shuffle_xor(t1, 2))
        var n0 = max(m0, t0)
        var n1 = max(m1, t1)
        var c0 = exp(m0 - n0)
        var c1 = exp(m1 - n1)
        m0 = n0
        m1 = n1
        var ps0 = Float32(0)
        var ps1 = Float32(0)
        comptime for n in range(SN):
            var sn = s[n]
            comptime for e in range(4):
                var valid = kt + n * 8 + 2 * t + (e & 1) < nk
                var mrow = m0 if e < 2 else m1
                sn[e] = exp(sn[e] - mrow) if valid else Float32(0)
            s[n] = sn
            ps0 += sn[0] + sn[1]
            ps1 += sn[2] + sn[3]
        # l is a per-thread partial sum; the quad adds them up at the end.
        l0 = l0 * c0 + ps0
        l1 = l1 * c1 + ps1
        var corr = SIMD[DType.float32, 4](c0, c0, c1, c1)
        comptime for dn in range(DN):
            acc[dn] = acc[dn] * corr
        # O += P V: S's accumulators for keys 16c..16c+15 are P's A operand.
        comptime for c in range(AK // 16):
            var a0 = _h2(s[2 * c][0], s[2 * c][1])
            var a1 = _h2(s[2 * c][2], s[2 * c][3])
            var a2 = _h2(s[2 * c + 1][0], s[2 * c + 1][1])
            var a3 = _h2(s[2 * c + 1][2], s[2 * c + 1][3])
            comptime for dn in range(DN):
                var vrow = sVt.unsafe_offset((dn * 8 + g) * VS + c * 16 + 2 * t)
                var b0 = vrow.unsafe_load[width=2, alignment=4]()
                var b1 = vrow.unsafe_offset(8).unsafe_load[
                    width=2, alignment=4
                ]()
                acc[dn] = _mma_f16(a0, a1, a2, a3, b0, b1, acc[dn])
        kt += AK
    l0 += shuffle_xor(l0, 1)
    l0 += shuffle_xor(l0, 2)
    l1 += shuffle_xor(l1, 1)
    l1 += shuffle_xor(l1, 2)
    var inv0 = 1.0 / l0
    var inv1 = 1.0 / l1
    comptime for dn in range(DN):
        var col = h * HD + dn * 8 + 2 * t
        if r0 < nq:
            o[unsafe_offset=r0 * stride + col] = acc[dn][0] * inv0
            o[unsafe_offset=r0 * stride + col + 1] = acc[dn][1] * inv0
        if r1 < nq:
            o[unsafe_offset=r1 * stride + col] = acc[dn][2] * inv1
            o[unsafe_offset=r1 * stride + col + 1] = acc[dn][3] * inv1


def attention_gpu(
    ctx: DeviceContext,
    q: F32Ptr,
    k: F32Ptr,
    v: F32Ptr,
    o: F32Ptr,
    L: Int,
    heads: Int,
    kv_heads: Int,
    head_dim: Int,
    n_keys: Int = -1,
) raises:
    """o = softmax(q k^T / sqrt(d)) v for L query rows over n_keys key rows
    (default L); with a K/V cache the keys are the whole canvas."""
    var scale = Float32(1.0 / sqrt(Float64(head_dim)))
    var grid = (_cdiv(L, AQ), heads)
    var Li = Int32(L)
    var Lki = Int32(L if n_keys < 0 else n_keys)
    var hi = Int32(heads)
    var kvi = Int32(kv_heads)
    if head_dim == 128:
        ctx.enqueue_function[_attention_kernel[128]](
            q,
            k,
            v,
            o,
            Li,
            Lki,
            hi,
            kvi,
            scale,
            grid_dim=grid,
            block_dim=ATT_THREADS,
        )
    elif head_dim == 64:
        ctx.enqueue_function[_attention_kernel[64]](
            q,
            k,
            v,
            o,
            Li,
            Lki,
            hi,
            kvi,
            scale,
            grid_dim=grid,
            block_dim=ATT_THREADS,
        )
    else:
        raise Error(
            "GPU attention supports head dimensions 64 and 128, got "
            + String(head_dim)
        )


# ------------------------------------------------------------- row statistics

comptime RS_THREADS = 256
"""Threads per block of the row-statistics kernel (a power of two)."""
comptime RS_FIELDS = 4
"""f64 values written per row: max, sum, runner-up, weighted sum."""


@always_inline
def _block_sum_f64(red: SharedF64, v: Float64, th: Int) -> Float64:
    """Sum over RS_THREADS threads; every thread gets the total."""
    red[unsafe_offset=th] = v
    barrier()
    var s = RS_THREADS // 2
    while s > 0:
        if th < s:
            red[unsafe_offset=th] = (
                red[unsafe_offset=th] + red[unsafe_offset=th + s]
            )
        barrier()
        s //= 2
    var r = red[unsafe_offset=0]
    barrier()
    return r


@always_inline
def _block_max_f64(red: SharedF64, v: Float64, th: Int) -> Float64:
    """Max over RS_THREADS threads; every thread gets the result."""
    red[unsafe_offset=th] = v
    barrier()
    var s = RS_THREADS // 2
    while s > 0:
        if th < s:
            red[unsafe_offset=th] = max(
                red[unsafe_offset=th], red[unsafe_offset=th + s]
            )
        barrier()
        s //= 2
    var r = red[unsafe_offset=0]
    barrier()
    return r


def _row_stats_kernel(logits: F32Ptr, vocab: Int32, tok: I32Ptr, stats: F64Ptr):
    """Reduce one logits row (block b reads row b) to what the samplers score.

    tok[b] is the first index of the maximum, as the host argmax picks it.
    stats[4b ..] holds, with m the maximum and e_v = exp(logit_v - m):
    m, sum e_v, the largest logit at any index other than tok[b], and
    sum e_v (logit_v - m). Sums are f64; the exponentials are f32.
    """
    var sv = unsafe_stack_allocation[
        RS_THREADS, Float32, address_space=AddressSpace.SHARED
    ]()
    var si = unsafe_stack_allocation[
        RS_THREADS, Int32, address_space=AddressSpace.SHARED
    ]()
    var red = unsafe_stack_allocation[
        RS_THREADS, Float64, address_space=AddressSpace.SHARED
    ]()
    var th = Int(thread_idx.x)
    var n = Int(vocab)
    var row = logits.unsafe_offset(Int(block_idx.x) * n)
    # First index of the maximum: a larger value wins, a tie goes to the
    # smaller index.
    var bv = Float32.MIN
    var bi = Int32.MAX
    for v in range(th, n, RS_THREADS):
        var x = row[unsafe_offset=v]
        if x > bv:
            bv = x
            bi = Int32(v)
    sv[unsafe_offset=th] = bv
    si[unsafe_offset=th] = bi
    barrier()
    var s = RS_THREADS // 2
    while s > 0:
        if th < s:
            var v2 = sv[unsafe_offset=th + s]
            var i2 = si[unsafe_offset=th + s]
            var v1 = sv[unsafe_offset=th]
            var i1 = si[unsafe_offset=th]
            if v2 > v1 or (v2 == v1 and i2 < i1):
                sv[unsafe_offset=th] = v2
                si[unsafe_offset=th] = i2
        barrier()
        s //= 2
    var m = sv[unsafe_offset=0]
    var best = Int(si[unsafe_offset=0])
    if best == Int(Int32.MAX):
        # No value beat the sentinel (a row of NaN): the host loop's answer.
        best = 0
    barrier()
    var z = Float64(0)
    var a = Float64(0)
    var second = Float64(-1e300)
    for v in range(th, n, RS_THREADS):
        var x = row[unsafe_offset=v]
        var dx = x - m
        var e = Float64(exp(dx))
        z += e
        a += e * Float64(dx)
        if v != best:
            second = max(second, Float64(x))
    z = _block_sum_f64(red, z, th)
    a = _block_sum_f64(red, a, th)
    second = _block_max_f64(red, second, th)
    if th == 0:
        var b = Int(block_idx.x)
        tok[unsafe_offset=b] = Int32(best)
        stats[unsafe_offset=RS_FIELDS * b] = Float64(m)
        stats[unsafe_offset=RS_FIELDS * b + 1] = z
        stats[unsafe_offset=RS_FIELDS * b + 2] = second
        stats[unsafe_offset=RS_FIELDS * b + 3] = a


def row_stats_gpu(
    ctx: DeviceContext,
    logits: F32Ptr,
    n_rows: Int,
    vocab: Int,
    tok: I32Ptr,
    stats: F64Ptr,
) raises:
    """Greedy token and softmax statistics of each of n_rows logits rows."""
    ctx.enqueue_function[_row_stats_kernel](
        logits,
        Int32(vocab),
        tok,
        stats,
        grid_dim=n_rows,
        block_dim=RS_THREADS,
    )


# ---------------------------------------------------------------- elementwise


def _silu_mul_kernel(g: F32Ptr, u: F32Ptr, n: Int32):
    var i = Int(block_idx.x) * NT + Int(thread_idx.x)
    if i < Int(n):
        var a = g[unsafe_offset=i]
        g[unsafe_offset=i] = a / (1.0 + exp(-a)) * u[unsafe_offset=i]


def _add_kernel(a: F32Ptr, b: F32Ptr, n: Int32):
    var i = Int(block_idx.x) * NT + Int(thread_idx.x)
    if i < Int(n):
        a[unsafe_offset=i] = a[unsafe_offset=i] + b[unsafe_offset=i]


def _add_bias_kernel(x: F32Ptr, b: F32Ptr, n: Int32, dim: Int32):
    var i = Int(block_idx.x) * NT + Int(thread_idx.x)
    if i < Int(n):
        x[unsafe_offset=i] = x[unsafe_offset=i] + b[unsafe_offset=i % Int(dim)]


def _copy_rows_kernel(
    src: F32Ptr, dst: F32Ptr, idx: I32Ptr, n_rows: Int32, dim: Int32
):
    """dst[j] = src[idx[j]] for j < n_rows. src and dst must not overlap."""
    var i = Int(block_idx.x) * NT + Int(thread_idx.x)
    var d = Int(dim)
    if i < Int(n_rows) * d:
        var j = i // d
        dst[unsafe_offset=i] = src[
            unsafe_offset=Int(idx[unsafe_offset=j]) * d + i % d
        ]


def silu_mul_gpu(ctx: DeviceContext, g: F32Ptr, u: F32Ptr, n: Int) raises:
    ctx.enqueue_function[_silu_mul_kernel](
        g, u, Int32(n), grid_dim=_cdiv(n, NT), block_dim=NT
    )


def add_gpu(ctx: DeviceContext, a: F32Ptr, b: F32Ptr, n: Int) raises:
    ctx.enqueue_function[_add_kernel](
        a, b, Int32(n), grid_dim=_cdiv(n, NT), block_dim=NT
    )


def add_bias_gpu(
    ctx: DeviceContext, x: F32Ptr, b: F32Ptr, rows: Int, dim: Int
) raises:
    var n = rows * dim
    ctx.enqueue_function[_add_bias_kernel](
        x, b, Int32(n), Int32(dim), grid_dim=_cdiv(n, NT), block_dim=NT
    )


def copy_rows_gpu(
    ctx: DeviceContext,
    src: F32Ptr,
    dst: F32Ptr,
    idx: I32Ptr,
    n_rows: Int,
    dim: Int,
) raises:
    var n = n_rows * dim
    ctx.enqueue_function[_copy_rows_kernel](
        src,
        dst,
        idx,
        Int32(n_rows),
        Int32(dim),
        grid_dim=_cdiv(n, NT),
        block_dim=NT,
    )


# ---------------------------------------------------------------- activation quantization


struct GpuQActs(Movable):
    """int8 activations with one f32 scale and one scaled sum per 32 values.

    `s` holds dx * sum(qx) for each 32-block: the Q4_K min term needs it.
    """

    var q: DeviceBuffer[DType.int8]
    var d: DeviceBuffer[DType.float32]
    var s: DeviceBuffer[DType.float32]
    var max_tokens: Int
    var K: Int
    var q4_sm_count: Int

    def __init__(out self, ctx: DeviceContext, max_tokens: Int, K: Int) raises:
        if K % 256 != 0:
            raise Error("activation width must be a multiple of 256")
        self.max_tokens = max_tokens
        self.K = K
        # Query once per buffer, not once per projection. The wave rule is
        # measured on RTX Blackwell; preserve dispatch on other families.
        self.q4_sm_count = 0
        if ctx.get_attribute(DeviceAttribute.COMPUTE_CAPABILITY_MAJOR) == 12:
            self.q4_sm_count = ctx.get_attribute(
                DeviceAttribute.MULTIPROCESSOR_COUNT
            )
        self.q = ctx.enqueue_create_buffer[DType.int8](max_tokens * K)
        self.d = ctx.enqueue_create_buffer[DType.float32](max_tokens * K // QK)
        self.s = ctx.enqueue_create_buffer[DType.float32](max_tokens * K // QK)


@always_inline
def _quantize_lane(
    v: Float32, q: I8Ptr, ds: F32Ptr, ss: F32Ptr, i: Int, b: Int
):
    """Lane `lane_id()` of a warp quantizes value i of the 32-value block b.

    All 32 lanes of the warp call it together: max and an integer sum do not
    depend on the order of the reduction, so the result is the same as a
    sequential loop's. The plain and the fused kernels share this code, which
    is what keeps a fused kernel's output identical to the chain it replaces.
    """
    var amax = warp_max(abs(v))
    var d = amax / 127.0
    var inv = Float32(0) if amax == 0 else 127.0 / amax
    var qv = Int32(round(v * inv))
    q[unsafe_offset=i] = Int8(qv)
    var sum = warp_sum(qv)
    if lane_id() == 0:
        ds[unsafe_offset=b] = d
        ss[unsafe_offset=b] = d * Float32(sum)


def _quantize_kernel(
    x: F32Ptr, q: I8Ptr, ds: F32Ptr, ss: F32Ptr, n_blocks: Int32
):
    # One warp per 32-value block, one value per lane, so loads and stores
    # are coalesced. QK is 32, the warp width.
    var b = Int(block_idx.x) * (NT // 32) + Int(warp_id())
    if b >= Int(n_blocks):
        return
    var i = b * QK + Int(lane_id())
    _quantize_lane(x[unsafe_offset=i], q, ds, ss, i, b)


def quantize_gpu(
    ctx: DeviceContext, x: F32Ptr, n_tokens: Int, K: Int, qa: GpuQActs
) raises:
    if K != qa.K or n_tokens > qa.max_tokens:
        raise Error("activation buffer is the wrong shape")
    var nb = n_tokens * K // QK
    ctx.enqueue_function[_quantize_kernel](
        x,
        dev_i8(qa.q),
        dev_f32(qa.d),
        dev_f32(qa.s),
        Int32(nb),
        grid_dim=_cdiv(nb, NT // 32),
        block_dim=NT,
    )


# ------------------------------------------------- fused elementwise kernels
#
# Each fused kernel evaluates the same f32 expressions, in the same order, as
# the kernels it replaces, so its output matches the chain bit for bit
# (tests/gpu/test_gpu.mojo compares them exactly).


def _norm_quantize_pass(
    row: F32Ptr,
    w: F32Ptr,
    inv: Float32,
    q: I8Ptr,
    ds: F32Ptr,
    ss: F32Ptr,
    d: Int,
    row_off: Int,
):
    """Quantize row * inv * w. Thread t takes values t, t + NT, ...: a warp
    holds 32 consecutive values at each step, which is one quantization block.
    d is a multiple of 32, so a warp is either inside the row or outside it.
    """
    var k = 0
    while k < d:
        var base = k + Int(warp_id()) * 32
        if base < d:
            var i = base + Int(lane_id())
            var v = row[unsafe_offset=i] * inv * w[unsafe_offset=i]
            _quantize_lane(v, q, ds, ss, row_off + i, (row_off + base) // 32)
        k += NT


def _rmsnorm_quantize_kernel(
    x: F32Ptr,
    w: F32Ptr,
    q: I8Ptr,
    ds: F32Ptr,
    ss: F32Ptr,
    dim: Int32,
    eps: Float32,
):
    var red = unsafe_stack_allocation[
        NT // 32, Float32, address_space=AddressSpace.SHARED
    ]()
    var d = Int(dim)
    var row_off = Int(block_idx.x) * d
    var row = x.unsafe_offset(row_off)
    var sq = Float32(0)
    var i = Int(thread_idx.x)
    while i < d:
        var v = row[unsafe_offset=i]
        sq += v * v
        i += NT
    var total = _block_sum(sq, red)
    var inv = 1.0 / sqrt(total / Float32(d) + eps)
    _norm_quantize_pass(row, w, inv, q, ds, ss, d, row_off)


def _add_rmsnorm_quantize_kernel(
    x: F32Ptr,
    h: F32Ptr,
    w: F32Ptr,
    q: I8Ptr,
    ds: F32Ptr,
    ss: F32Ptr,
    dim: Int32,
    eps: Float32,
):
    var red = unsafe_stack_allocation[
        NT // 32, Float32, address_space=AddressSpace.SHARED
    ]()
    var d = Int(dim)
    var row_off = Int(block_idx.x) * d
    var row = x.unsafe_offset(row_off)
    var hrow = h.unsafe_offset(row_off)
    var sq = Float32(0)
    var i = Int(thread_idx.x)
    while i < d:
        var v = row[unsafe_offset=i] + hrow[unsafe_offset=i]
        row[unsafe_offset=i] = v
        sq += v * v
        i += NT
    # The second pass reads back the values this same thread wrote.
    var total = _block_sum(sq, red)
    var inv = 1.0 / sqrt(total / Float32(d) + eps)
    _norm_quantize_pass(row, w, inv, q, ds, ss, d, row_off)


def _silu_mul_quantize_kernel(
    g: F32Ptr, u: F32Ptr, q: I8Ptr, ds: F32Ptr, ss: F32Ptr, n_blocks: Int32
):
    var b = Int(block_idx.x) * (NT // 32) + Int(warp_id())
    if b >= Int(n_blocks):
        return
    var i = b * QK + Int(lane_id())
    var a = g[unsafe_offset=i]
    _quantize_lane(a / (1.0 + exp(-a)) * u[unsafe_offset=i], q, ds, ss, i, b)


def _check_quant_shape(qa: GpuQActs, n_tokens: Int, K: Int) raises:
    if K != qa.K or n_tokens > qa.max_tokens or K % QK != 0:
        raise Error("activation buffer is the wrong shape")


def rmsnorm_quantize_gpu(
    ctx: DeviceContext,
    x: F32Ptr,
    w: F32Ptr,
    qa: GpuQActs,
    rows: Int,
    dim: Int,
    eps: Float32,
) raises:
    """quantize(rmsnorm(x)) in one launch; the normalized f32 rows are not
    stored."""
    _check_quant_shape(qa, rows, dim)
    ctx.enqueue_function[_rmsnorm_quantize_kernel](
        x,
        w,
        dev_i8(qa.q),
        dev_f32(qa.d),
        dev_f32(qa.s),
        Int32(dim),
        eps,
        grid_dim=rows,
        block_dim=NT,
    )


def add_rmsnorm_quantize_gpu(
    ctx: DeviceContext,
    x: F32Ptr,
    h: F32Ptr,
    w: F32Ptr,
    qa: GpuQActs,
    rows: Int,
    dim: Int,
    eps: Float32,
) raises:
    """x += h, then quantize(rmsnorm(x)), in one launch. x keeps the sum (it
    is the residual stream); the normalized f32 rows are not stored."""
    _check_quant_shape(qa, rows, dim)
    ctx.enqueue_function[_add_rmsnorm_quantize_kernel](
        x,
        h,
        w,
        dev_i8(qa.q),
        dev_f32(qa.d),
        dev_f32(qa.s),
        Int32(dim),
        eps,
        grid_dim=rows,
        block_dim=NT,
    )


def silu_mul_quantize_gpu(
    ctx: DeviceContext,
    g: F32Ptr,
    u: F32Ptr,
    qa: GpuQActs,
    rows: Int,
    dim: Int,
) raises:
    """quantize(silu(g) * u) in one launch; g and u are left as they were."""
    _check_quant_shape(qa, rows, dim)
    var nb = rows * dim // QK
    ctx.enqueue_function[_silu_mul_quantize_kernel](
        g,
        u,
        dev_i8(qa.q),
        dev_f32(qa.d),
        dev_f32(qa.s),
        Int32(nb),
        grid_dim=_cdiv(nb, NT // 32),
        block_dim=NT,
    )


# ---------------------------------------------------------------- weights


def staged_row_bytes(ggml_type: Int, K: Int) raises -> Int:
    if K % 256 != 0:
        raise Error("row length " + String(K) + " is not a multiple of 256")
    if ggml_type == GGML_Q4_K:
        return K // 256 * Q4K_BYTES
    if ggml_type == GGML_Q6_K:
        return K // 256 * Q6K_STAGED
    raise Error(
        "the GPU backend has no matrix kernel for " + ggml_type_name(ggml_type)
    )


def staged_matrix_bytes(ggml_type: Int, N: Int, K: Int) raises -> Int:
    return N * staged_row_bytes(ggml_type, K)


def stage_matrix(
    src_addr: Int, ggml_type: Int, N: Int, K: Int, dst: U8Ptr
) raises:
    """Copy a GGUF matrix into its device layout (Q6_K padded to 224-byte blocks).
    """
    var src = U8Ptr(unsafe_from_address=src_addr)
    var blocks = N * K // 256
    if ggml_type == GGML_Q4_K:
        for i in range(blocks * Q4K_BYTES):
            dst[unsafe_offset=i] = src[unsafe_offset=i]
        return
    if ggml_type == GGML_Q6_K:
        for b in range(blocks):
            var s = src.unsafe_offset(b * Q6K_BYTES)
            var d = dst.unsafe_offset(b * Q6K_STAGED)
            for i in range(Q6K_BYTES):
                d[unsafe_offset=i] = s[unsafe_offset=i]
            for i in range(Q6K_BYTES, Q6K_STAGED):
                d[unsafe_offset=i] = 0
        return
    raise Error(
        "the GPU backend has no matrix kernel for " + ggml_type_name(ggml_type)
    )


# ---------------------------------------------------------------- GEMM
#
# Tensor-core GEMM. A block computes a BM x BN tile of the output (BM weight
# rows, BN tokens), walking K one 256-wide super-block at a time. Each step
# stages the super-block's weights as int8 (Q4_K: q in 0..15, Q6_K: q - 32)
# plus their scales in shared memory, and the matching activations next to
# them. Warp w owns rows 16w..16w+15 and all BN tokens: per 32-wide sub-block
# it runs one mma.m16n8k32 (Q6_K: two m16n8k16, one per 16-group scale) per
# 8 tokens and folds the int32 result into f32 with the sub-block's scales.
#
# Fragment layouts (verified against a CPU matmul, scratch/gpu/mma_layout):
#   g = lane / 4, t = lane % 4
#   A row-major 16 x 32: a0 (g, 4t..), a1 (g+8, 4t..), a2 (g, 16+4t..), a3 (g+8, 16+4t..)
#   B as 8 token rows x 32: b0 (n=g, 4t..), b1 (n=g, 16+4t..)
#   C 16 x 8: c0 (g, 2t), c1 (g, 2t+1), c2 (g+8, 2t), c3 (g+8, 2t+1)

comptime BM = 64
comptime BN_MAX = 80
"""Widest token tile; gemm_gpu picks the width per call (see _pick_bn)."""
comptime GEMM_THREADS = 128
comptime WS4 = 9
"""Shared row stride of the Q4_K sub-block scales (8 used)."""
comptime SW = 68
"""Shared row stride in 32-bit words: 256 bytes plus 16 of padding, so the 8
rows a fragment load touches fall in different banks."""


@always_inline
def _mma_k32(
    a0: Int32, a1: Int32, a2: Int32, a3: Int32, b0: Int32, b1: Int32
) -> SIMD[DType.int32, 4]:
    var z = Int32(0)
    var r = llvm_intrinsic[
        "llvm.nvvm.mma.m16n8k32.row.col.s8",
        _RegisterPackType[Int32, Int32, Int32, Int32],
    ](a0, a1, a2, a3, b0, b1, z, z, z, z)
    return SIMD[DType.int32, 4](r[0], r[1], r[2], r[3])


@always_inline
def _mma_k16(a0: Int32, a1: Int32, b0: Int32) -> SIMD[DType.int32, 4]:
    var z = Int32(0)
    var r = llvm_intrinsic[
        "llvm.nvvm.mma.m16n8k16.row.col.s8",
        _RegisterPackType[Int32, Int32, Int32, Int32],
    ](a0, a1, b0, z, z, z, z)
    return SIMD[DType.int32, 4](r[0], r[1], r[2], r[3])


comptime SharedI32 = Pointer[
    Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@always_inline
def _stage_acts[
    BN: Int
](
    aq: I8Ptr,
    ad: F32Ptr,
    as_: F32Ptr,
    k: Int,
    T: Int,
    t0: Int,
    b: Int,
    sA: SharedI32,
    sAds: SharedF32,
):
    """Copy BN tokens' activations for super-block b into shared memory.

    Scales go to sAds as [sub-block j][token][d, s]: the token pair a thread
    folds after an mma is then four consecutive floats, one 128-bit load,
    and the four lanes of a quad read disjoint banks.
    """
    var th = Int(thread_idx.x)
    comptime for rep in range((BN * 4 + GEMM_THREADS - 1) // GEMM_THREADS):
        var unit = th + rep * GEMM_THREADS
        if unit >= BN * 4:
            break
        var tok = unit >> 2
        var qtr = unit & 3
        var dst = sA.unsafe_offset(tok * SW + qtr * 16)
        if t0 + tok < T:
            var src = aq.unsafe_offset(
                (t0 + tok) * k + b * 256 + qtr * 64
            ).unsafe_bitcast[Int32]()
            comptime for i in range(16):
                dst[unsafe_offset=i] = src[unsafe_offset=i]
        else:
            comptime for i in range(16):
                dst[unsafe_offset=i] = 0
    var nsub = k // QK
    comptime for e in range((BN * 8 + GEMM_THREADS - 1) // GEMM_THREADS):
        var idx = th + e * GEMM_THREADS
        if idx >= BN * 8:
            break
        var tk = idx >> 3
        var j = idx & 7
        var o = (j * BN + tk) * 2
        if t0 + tk < T:
            sAds[unsafe_offset=o] = ad[
                unsafe_offset=(t0 + tk) * nsub + b * 8 + j
            ]
            sAds[unsafe_offset=o + 1] = as_[
                unsafe_offset=(t0 + tk) * nsub + b * 8 + j
            ]
        else:
            sAds[unsafe_offset=o] = 0
            sAds[unsafe_offset=o + 1] = 0


comptime BM4 = 128
"""Weight rows per Q4_K thread block: 16 per warp, 8 warps."""
comptime Q4K_THREADS = 256
comptime SWQ = 36
"""Shared row stride of the packed Q4_K nibbles in 32-bit words: 128 bytes
plus 16 of padding, so the 8 rows a fragment load reads fall in different
banks."""


@always_inline
def _stage_acts16[
    BN: Int, NTH: Int
](
    aq: I8Ptr,
    ad: F32Ptr,
    as_: F32Ptr,
    k: Int,
    T: Int,
    t0: Int,
    b: Int,
    sA: SharedI32,
    sAds: SharedF32,
):
    """`_stage_acts` with 128-bit copies and NTH threads."""
    var th = Int(thread_idx.x)
    comptime for rep in range((BN * 16 + NTH - 1) // NTH):
        var unit = th + rep * NTH
        if unit < BN * 16:
            var tok = unit >> 4
            var c = unit & 15
            var v = SIMD[DType.int32, 4](0)
            if t0 + tok < T:
                v = (
                    aq.unsafe_offset((t0 + tok) * k + b * 256 + c * 16)
                    .unsafe_bitcast[Int32]()
                    .unsafe_load[width=4, alignment=16]()
                )
            sA.unsafe_offset(tok * SW + c * 4).unsafe_store[alignment=16](v)
    var nsub = k // QK
    comptime for e in range((BN * 8 + NTH - 1) // NTH):
        var idx = th + e * NTH
        if idx < BN * 8:
            var tk = idx >> 3
            var j = idx & 7
            var o = (j * BN + tk) * 2
            if t0 + tk < T:
                sAds[unsafe_offset=o] = ad[
                    unsafe_offset=(t0 + tk) * nsub + b * 8 + j
                ]
                sAds[unsafe_offset=o + 1] = as_[
                    unsafe_offset=(t0 + tk) * nsub + b * 8 + j
                ]
            else:
                sAds[unsafe_offset=o] = 0
                sAds[unsafe_offset=o + 1] = 0


@always_inline
def _byte(w: Int32, i: Int) -> Int32:
    return (w >> Int32(8 * i)) & 0xFF


@always_inline
def _sb(h: SIMD[DType.int32, 4], i: Int) -> Int32:
    """Scale byte i (0..11) of a Q4_K header held as four words."""
    return _byte(h[1 + i // 4], i % 4)


@always_inline
def _q4k_scale(h: SIMD[DType.int32, 4], j: Int) -> Tuple[Float32, Float32]:
    """d * sc_j and -dmin * m_j of one Q4_K super-block from its 16-byte header.

    Word 0 holds d and dmin as f16; words 1..3 hold the 12 packed scale bytes.
    """
    var d = bitcast[DType.float16](UInt16(Int(h[0]) & 0xFFFF)).cast[
        DType.float32
    ]()
    var dmin = bitcast[DType.float16](UInt16((Int(h[0]) >> 16) & 0xFFFF)).cast[
        DType.float32
    ]()
    var sc: Int32
    var mn: Int32
    if j < 4:
        sc = _sb(h, j) & 63
        mn = _sb(h, j + 4) & 63
    else:
        sc = (_sb(h, j + 4) & 0xF) | ((_sb(h, j - 4) >> 6) << 4)
        mn = (_sb(h, j + 4) >> 4) | ((_sb(h, j) >> 6) << 4)
    return (d * Float32(sc), -dmin * Float32(mn))


comptime Q4K_WR = (BM4 * 9 + Q4K_THREADS - 1) // Q4K_THREADS
"""16-byte weight chunks each thread stages per super-block."""


@always_inline
def _q4k_load(
    w: U8Ptr, row_bytes: Int, r0: Int, n_rows: Int, b: Int, th: Int
) -> Array[SIMD[DType.int32, 4], Q4K_WR]:
    """This thread's 16-byte chunks of super-block b: header, then nibbles."""
    var v = Array[SIMD[DType.int32, 4], Q4K_WR](fill=SIMD[DType.int32, 4](0))
    comptime for rep in range(Q4K_WR):
        var unit = th + rep * Q4K_THREADS
        var row = unit // 9
        if unit < BM4 * 9 and r0 + row < n_rows:
            v[rep] = (
                w.unsafe_offset(
                    (r0 + row) * row_bytes + b * Q4K_BYTES + (unit % 9) * 16
                )
                .unsafe_bitcast[Int32]()
                .unsafe_load[width=4, alignment=16]()
            )
    return v^


@always_inline
def _q4k_store(
    v: Array[SIMD[DType.int32, 4], Q4K_WR],
    sWq: SharedI32,
    sWh: SharedI32,
    th: Int,
):
    comptime for rep in range(Q4K_WR):
        var unit = th + rep * Q4K_THREADS
        if unit < BM4 * 9:
            var row = unit // 9
            var c = unit % 9
            if c == 0:
                sWh.unsafe_offset(row * 4).unsafe_store[alignment=16](v[rep])
            else:
                sWq.unsafe_offset(row * SWQ + (c - 1) * 4).unsafe_store[
                    alignment=16
                ](v[rep])


@always_inline
def _acts_load[
    BN: Int, NTH: Int
](aq: I8Ptr, k: Int, T: Int, t0: Int, b: Int, th: Int) -> Array[
    SIMD[DType.int32, 4], (BN * 16 + NTH - 1) // NTH
]:
    """This thread's 16-byte chunks of BN tokens' activations, super-block b."""
    comptime R = (BN * 16 + NTH - 1) // NTH
    var v = Array[SIMD[DType.int32, 4], R](fill=SIMD[DType.int32, 4](0))
    comptime for rep in range(R):
        var unit = th + rep * NTH
        var tok = unit >> 4
        if unit < BN * 16 and t0 + tok < T:
            v[rep] = (
                aq.unsafe_offset((t0 + tok) * k + b * 256 + (unit & 15) * 16)
                .unsafe_bitcast[Int32]()
                .unsafe_load[width=4, alignment=16]()
            )
    return v^


@always_inline
def _acts_store[
    BN: Int, NTH: Int
](
    v: Array[SIMD[DType.int32, 4], (BN * 16 + NTH - 1) // NTH],
    sA: SharedI32,
    th: Int,
):
    comptime for rep in range((BN * 16 + NTH - 1) // NTH):
        var unit = th + rep * NTH
        if unit < BN * 16:
            sA.unsafe_offset((unit >> 4) * SW + (unit & 15) * 4).unsafe_store[
                alignment=16
            ](v[rep])


@always_inline
def _scales_load[
    BN: Int, NTH: Int
](ad: F32Ptr, as_: F32Ptr, k: Int, T: Int, t0: Int, b: Int, th: Int) -> Array[
    Float32, 2 * ((BN * 8 + NTH - 1) // NTH)
]:
    """(d, s) of this thread's (token, sub-block) pairs, super-block b."""
    comptime R = (BN * 8 + NTH - 1) // NTH
    var v = Array[Float32, 2 * R](fill=0)
    var nsub = k // QK
    comptime for e in range(R):
        var idx = th + e * NTH
        var tk = idx >> 3
        if idx < BN * 8 and t0 + tk < T:
            var src = (t0 + tk) * nsub + b * 8 + (idx & 7)
            v[2 * e] = ad[unsafe_offset=src]
            v[2 * e + 1] = as_[unsafe_offset=src]
    return v^


@always_inline
def _scales_store[
    BN: Int, NTH: Int
](v: Array[Float32, 2 * ((BN * 8 + NTH - 1) // NTH)], sAds: SharedF32, th: Int):
    comptime for e in range((BN * 8 + NTH - 1) // NTH):
        var idx = th + e * NTH
        if idx < BN * 8:
            var o = ((idx & 7) * BN + (idx >> 3)) * 2
            sAds[unsafe_offset=o] = v[2 * e]
            sAds[unsafe_offset=o + 1] = v[2 * e + 1]


def _q4k_mma_kernel[
    BN: Int
](
    w: U8Ptr,
    N: Int32,
    K: Int32,
    aq: I8Ptr,
    ad: F32Ptr,
    as_: F32Ptr,
    T: Int32,
    dst: F32Ptr,
    ldo: Int32,
):
    """Q4_K GEMM: BM4 weight rows by BN tokens per block, 8 warps.

    Staging is a plain 16-byte copy: each row's 128 bytes of packed nibbles
    go to sWq as they are, and its 16-byte header (d, dmin, 12 scale bytes)
    to sWh. A warp unpacks its A fragments in registers: packed word i of
    pair p holds sub-block 2p in its low nibbles and 2p + 1 in its high
    ones, at the same position, so one load feeds two mmas. The scale
    arithmetic is the same as when the scales were unpacked during staging.
    """
    var sWq = unsafe_stack_allocation[
        BM4 * SWQ, Int32, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var sWh = unsafe_stack_allocation[
        BM4 * 4, Int32, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var sA = unsafe_stack_allocation[
        BN * SW, Int32, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var sAds = unsafe_stack_allocation[
        BN * 16, Float32, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var k = Int(K)
    var n_rows = Int(N)
    var n_tok = Int(T)
    var r0 = Int(block_idx.y) * BM4
    var t0 = Int(block_idx.x) * BN
    var th = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp = Int(warp_id())
    var g = lane >> 2
    var t = lane & 3
    var row_bytes = (k // 256) * Q4K_BYTES
    comptime NT8 = BN // 8
    var acc = Array[Float32, 4 * NT8](fill=0)
    var ra = 16 * warp + g
    var rb = ra + 8
    # Super-block b + 1 is read from global memory into registers while b
    # runs on the tensor cores, so the loads can overlap with compute.
    var nb = k // 256
    var wv = _q4k_load(w, row_bytes, r0, n_rows, 0, th)
    var av = _acts_load[BN, Q4K_THREADS](aq, k, n_tok, t0, 0, th)
    var dv = _scales_load[BN, Q4K_THREADS](ad, as_, k, n_tok, t0, 0, th)
    for b in range(nb):
        _q4k_store(wv, sWq, sWh, th)
        _acts_store[BN, Q4K_THREADS](av, sA, th)
        _scales_store[BN, Q4K_THREADS](dv, sAds, th)
        barrier()
        if b + 1 < nb:
            wv = _q4k_load(w, row_bytes, r0, n_rows, b + 1, th)
            av = _acts_load[BN, Q4K_THREADS](aq, k, n_tok, t0, b + 1, th)
            dv = _scales_load[BN, Q4K_THREADS](ad, as_, k, n_tok, t0, b + 1, th)
        # ---- tensor cores
        var ha = sWh.unsafe_offset(ra * 4).unsafe_load[width=4, alignment=16]()
        var hb = sWh.unsafe_offset(rb * 4).unsafe_load[width=4, alignment=16]()
        comptime for p in range(4):
            var pa0 = sWq[unsafe_offset=ra * SWQ + p * 8 + t]
            var pb0 = sWq[unsafe_offset=rb * SWQ + p * 8 + t]
            var pa1 = sWq[unsafe_offset=ra * SWQ + p * 8 + 4 + t]
            var pb1 = sWq[unsafe_offset=rb * SWQ + p * 8 + 4 + t]
            comptime for hi in range(2):
                comptime j = 2 * p + hi
                var a0 = (pa0 >> Int32(4 * hi)) & 0x0F0F0F0F
                var a1 = (pb0 >> Int32(4 * hi)) & 0x0F0F0F0F
                var a2 = (pa1 >> Int32(4 * hi)) & 0x0F0F0F0F
                var a3 = (pb1 >> Int32(4 * hi)) & 0x0F0F0F0F
                var s0 = _q4k_scale(ha, j)
                var s1 = _q4k_scale(hb, j)
                var ws0 = s0[0]
                var wm0 = s0[1]
                var ws1 = s1[0]
                var wm1 = s1[1]
                comptime for nt in range(NT8):
                    var bo = (nt * 8 + g) * SW + j * 8 + t
                    var c = _mma_k32(
                        a0,
                        a1,
                        a2,
                        a3,
                        sA[unsafe_offset=bo],
                        sA[unsafe_offset=bo + 4],
                    )
                    # (d, s) of tokens 2t and 2t + 1 in one 128-bit load.
                    var q = sAds.unsafe_offset(
                        (j * BN + nt * 8 + 2 * t) * 2
                    ).unsafe_load[width=4, alignment=16]()
                    acc[nt * 4 + 0] += ws0 * q[0] * Float32(c[0]) + wm0 * q[1]
                    acc[nt * 4 + 1] += ws0 * q[2] * Float32(c[1]) + wm0 * q[3]
                    acc[nt * 4 + 2] += ws1 * q[0] * Float32(c[2]) + wm1 * q[1]
                    acc[nt * 4 + 3] += ws1 * q[2] * Float32(c[3]) + wm1 * q[3]
        barrier()
    _store_tile[BN](acc, dst, Int(ldo), r0, t0, n_rows, n_tok, warp, g, t)


@always_inline
def _store_tile[
    BN: Int
](
    acc: Array[Float32, 4 * (BN // 8)],
    dst: F32Ptr,
    ldo: Int,
    r0: Int,
    t0: Int,
    n_rows: Int,
    n_tok: Int,
    warp: Int,
    g: Int,
    t: Int,
):
    var ra = r0 + 16 * warp + g
    comptime for nt in range(BN // 8):
        var ta = t0 + nt * 8 + 2 * t
        comptime for e in range(4):
            var r = ra + (8 if e >= 2 else 0)
            var tk = ta + (e & 1)
            if r < n_rows and tk < n_tok:
                dst[unsafe_offset=tk * ldo + r] = acc[nt * 4 + e]


comptime Q6K_THREADS = GEMM_THREADS
comptime Q6K_CHUNKS = Q6K_STAGED // 16
"""16-byte chunks of a staged Q6_K block: 8 of low nibbles, 4 of high bits, one
of scales, one holding d."""
comptime SWQ6 = 60
"""Shared row stride of a Q6_K block in 32-bit words: 56 plus 4 of padding, so
the 8 rows a fragment load reads fall in different banks."""
comptime Q6K_WR = (BM * Q6K_CHUNKS + Q6K_THREADS - 1) // Q6K_THREADS
"""16-byte weight chunks each thread stages per super-block."""


@always_inline
def _q6k_load(
    w: U8Ptr, row_bytes: Int, r0: Int, n_rows: Int, b: Int, th: Int
) -> Array[SIMD[DType.int32, 4], Q6K_WR]:
    """This thread's 16-byte chunks of super-block b of rows r0..r0 + BM - 1."""
    var v = Array[SIMD[DType.int32, 4], Q6K_WR](fill=SIMD[DType.int32, 4](0))
    comptime for rep in range(Q6K_WR):
        var unit = th + rep * Q6K_THREADS
        var row = unit // Q6K_CHUNKS
        if unit < BM * Q6K_CHUNKS and r0 + row < n_rows:
            v[rep] = (
                w.unsafe_offset(
                    (r0 + row) * row_bytes
                    + b * Q6K_STAGED
                    + (unit % Q6K_CHUNKS) * 16
                )
                .unsafe_bitcast[Int32]()
                .unsafe_load[width=4, alignment=16]()
            )
    return v^


@always_inline
def _q6k_store(v: Array[SIMD[DType.int32, 4], Q6K_WR], sW: SharedI32, th: Int):
    comptime for rep in range(Q6K_WR):
        var unit = th + rep * Q6K_THREADS
        if unit < BM * Q6K_CHUNKS:
            sW.unsafe_offset(
                (unit // Q6K_CHUNKS) * SWQ6 + (unit % Q6K_CHUNKS) * 4
            ).unsafe_store[alignment=16](v[rep])


@always_inline
def _q6k_frag(ql: Int32, qh: Int32, gg: Int) -> Int32:
    """Four weights, minus 32, as sign-extended int8 lanes: group gg of a half
    (weights 32gg..32gg + 31) takes its low four bits from the low or high
    nibbles of a ql word and its top two from bits 2gg..2gg + 1 of qh."""
    var nib = (ql >> Int32(4 * (gg >> 1))) & 0x0F0F0F0F
    var hb = ((qh >> Int32(2 * gg)) & 0x03030303) << 4
    # q - 32 for q in 0..63 is q ^ 32 read as a 6-bit signed value; widen
    # bit 5 into bits 6 and 7 of each byte.
    var v = (nib | hb) ^ 0x20202020
    var sgn = v & 0x20202020
    return v | (sgn << 1) | (sgn << 2)


def _q6k_mma_kernel[
    BN: Int
](
    w: U8Ptr,
    N: Int32,
    K: Int32,
    aq: I8Ptr,
    ad: F32Ptr,
    as_: F32Ptr,
    T: Int32,
    dst: F32Ptr,
    ldo: Int32,
):
    """Q6_K GEMM: BM weight rows by BN tokens per block, 4 warps.

    Staging is a plain 16-byte copy of each row's 224-byte block (low nibbles,
    high bits, scales, d) into sW. A warp unpacks its A fragments in
    registers, one 16-weight group per mma.m16n8k16, with the scale of that
    group. The next super-block is read into registers while this one runs on
    the tensor cores. The accumulation order is the one the kernel had when it
    unpacked during staging, so the output is unchanged.
    """
    var sW = unsafe_stack_allocation[
        BM * SWQ6, Int32, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var sA = unsafe_stack_allocation[
        BN * SW, Int32, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var sAds = unsafe_stack_allocation[
        BN * 16, Float32, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var k = Int(K)
    var n_rows = Int(N)
    var n_tok = Int(T)
    var r0 = Int(block_idx.y) * BM
    var t0 = Int(block_idx.x) * BN
    var th = Int(thread_idx.x)
    var lane = Int(lane_id())
    var warp = Int(warp_id())
    var g = lane >> 2
    var t = lane & 3
    var row_bytes = (k // 256) * Q6K_STAGED
    comptime NT8 = BN // 8
    var acc = Array[Float32, 4 * NT8](fill=0)
    var ra = 16 * warp + g
    var rb = ra + 8
    var nb = k // 256
    var wv = _q6k_load(w, row_bytes, r0, n_rows, 0, th)
    var av = _acts_load[BN, Q6K_THREADS](aq, k, n_tok, t0, 0, th)
    var dv = _scales_load[BN, Q6K_THREADS](ad, as_, k, n_tok, t0, 0, th)
    for b in range(nb):
        _q6k_store(wv, sW, th)
        _acts_store[BN, Q6K_THREADS](av, sA, th)
        _scales_store[BN, Q6K_THREADS](dv, sAds, th)
        barrier()
        if b + 1 < nb:
            wv = _q6k_load(w, row_bytes, r0, n_rows, b + 1, th)
            av = _acts_load[BN, Q6K_THREADS](aq, k, n_tok, t0, b + 1, th)
            dv = _scales_load[BN, Q6K_THREADS](ad, as_, k, n_tok, t0, b + 1, th)
        # Row words: 0..31 low nibbles, 32..47 high bits, 48..51 the 16 group
        # scales as int8, 52 holds d as f16.
        var sa = sW.unsafe_offset(ra * SWQ6 + 48).unsafe_load[
            width=4, alignment=16
        ]()
        var sb = sW.unsafe_offset(rb * SWQ6 + 48).unsafe_load[
            width=4, alignment=16
        ]()
        var da = bitcast[DType.float16](
            UInt16(Int(sW[unsafe_offset=ra * SWQ6 + 52]) & 0xFFFF)
        ).cast[DType.float32]()
        var db = bitcast[DType.float16](
            UInt16(Int(sW[unsafe_offset=rb * SWQ6 + 52]) & 0xFFFF)
        ).cast[DType.float32]()
        comptime for J in range(16):
            # Group J holds weights 16J..16J + 15: half hf, 32-wide sub-block
            # gg of that half, 16-wide half lo of the sub-block.
            comptime hf = J // 8
            comptime gg = (J % 8) // 2
            comptime lo = J % 2
            var qla = sW[
                unsafe_offset=ra * SWQ6 + 16 * hf + 8 * (gg & 1) + 4 * lo + t
            ]
            var qlb = sW[
                unsafe_offset=rb * SWQ6 + 16 * hf + 8 * (gg & 1) + 4 * lo + t
            ]
            var qha = sW[unsafe_offset=ra * SWQ6 + 32 + 8 * hf + 4 * lo + t]
            var qhb = sW[unsafe_offset=rb * SWQ6 + 32 + 8 * hf + 4 * lo + t]
            var a0 = _q6k_frag(qla, qha, gg)
            var a1 = _q6k_frag(qlb, qhb, gg)
            var ws0 = da * Float32(
                bitcast[DType.int8](
                    UInt8((Int(sa[J // 4]) >> (8 * (J % 4))) & 0xFF)
                )
            )
            var ws1 = db * Float32(
                bitcast[DType.int8](
                    UInt8((Int(sb[J // 4]) >> (8 * (J % 4))) & 0xFF)
                )
            )
            comptime for nt in range(NT8):
                var c = _mma_k16(
                    a0, a1, sA[unsafe_offset=(nt * 8 + g) * SW + 4 * J + t]
                )
                var q = sAds.unsafe_offset(
                    ((J // 2) * BN + nt * 8 + 2 * t) * 2
                ).unsafe_load[width=4, alignment=16]()
                acc[nt * 4 + 0] += ws0 * q[0] * Float32(c[0])
                acc[nt * 4 + 1] += ws0 * q[2] * Float32(c[1])
                acc[nt * 4 + 2] += ws1 * q[0] * Float32(c[2])
                acc[nt * 4 + 3] += ws1 * q[2] * Float32(c[3])
        barrier()
    _store_tile[BN](acc, dst, Int(ldo), r0, t0, n_rows, n_tok, warp, g, t)


def _mma_tile32_kernel(a: I8Ptr, b: I8Ptr, c: I32Ptr):
    # A 16x32 row-major, B as 8 token rows of 32: the layout gemm_gpu stages.
    var l = Int(lane_id())
    var g = l >> 2
    var t = l & 3
    var aw = a.unsafe_bitcast[Int32]()
    var bw = b.unsafe_bitcast[Int32]()
    var r = _mma_k32(
        aw[unsafe_offset=g * 8 + t],
        aw[unsafe_offset=(g + 8) * 8 + t],
        aw[unsafe_offset=g * 8 + 4 + t],
        aw[unsafe_offset=(g + 8) * 8 + 4 + t],
        bw[unsafe_offset=g * 8 + t],
        bw[unsafe_offset=g * 8 + 4 + t],
    )
    _store_tile_c(c, g, t, r)


def _mma_tile16_kernel(a: I8Ptr, b: I8Ptr, c: I32Ptr):
    var l = Int(lane_id())
    var g = l >> 2
    var t = l & 3
    var aw = a.unsafe_bitcast[Int32]()
    var bw = b.unsafe_bitcast[Int32]()
    var r = _mma_k16(
        aw[unsafe_offset=g * 4 + t],
        aw[unsafe_offset=(g + 8) * 4 + t],
        bw[unsafe_offset=g * 4 + t],
    )
    _store_tile_c(c, g, t, r)


def _store_tile_c(c: I32Ptr, g: Int, t: Int, r: SIMD[DType.int32, 4]):
    c[unsafe_offset=g * 8 + 2 * t] = r[0]
    c[unsafe_offset=g * 8 + 2 * t + 1] = r[1]
    c[unsafe_offset=(g + 8) * 8 + 2 * t] = r[2]
    c[unsafe_offset=(g + 8) * 8 + 2 * t + 1] = r[3]


def mma_tile_gpu(
    ctx: DeviceContext, K: Int, a: I8Ptr, b: I8Ptr, c: I32Ptr
) raises:
    """One 16 x 8 int8 tile, c = a (16 x K) times b (8 x K) transposed, K = 32 or 16.

    It goes through the same fragment loads the GEMM uses. tests/gpu checks it
    against a CPU matmul, which pins the PTX fragment layout.
    """
    if K == 32:
        ctx.enqueue_function[_mma_tile32_kernel](
            a, b, c, grid_dim=1, block_dim=32
        )
    elif K == 16:
        ctx.enqueue_function[_mma_tile16_kernel](
            a, b, c, grid_dim=1, block_dim=32
        )
    else:
        raise Error("mma_tile_gpu takes K = 32 or 16")


def _launch[
    BN: Int
](
    ctx: DeviceContext,
    ggml_type: Int,
    w: U8Ptr,
    N: Int,
    K: Int,
    qa: GpuQActs,
    n_tokens: Int,
    dst: F32Ptr,
    ldo: Int,
) raises:
    if ggml_type == GGML_Q4_K:
        ctx.enqueue_function[_q4k_mma_kernel[BN]](
            w,
            Int32(N),
            Int32(K),
            dev_i8(qa.q),
            dev_f32(qa.d),
            dev_f32(qa.s),
            Int32(n_tokens),
            dst,
            Int32(ldo),
            grid_dim=(_cdiv(n_tokens, BN), _cdiv(N, BM4)),
            block_dim=Q4K_THREADS,
        )
        return
    if ggml_type == GGML_Q6_K:
        ctx.enqueue_function[_q6k_mma_kernel[BN]](
            w,
            Int32(N),
            Int32(K),
            dev_i8(qa.q),
            dev_f32(qa.d),
            dev_f32(qa.s),
            Int32(n_tokens),
            dst,
            Int32(ldo),
            grid_dim=(_cdiv(n_tokens, BN), _cdiv(N, BM)),
            block_dim=GEMM_THREADS,
        )
        return
    raise Error(
        "the GPU backend has no matrix kernel for " + ggml_type_name(ggml_type)
    )


def _pick_bn(n_tokens: Int) -> Int:
    """Token tile width: the fewest tiles, then the narrowest tile.

    A wider tile unpacks each weight tile once for more tokens, so it wins
    whenever it saves a tile (148 tokens: 80 wide, 2 tiles, against 5 at 32;
    bench/profiles/2026-10-12-gemmtile-* has the GEMM time before and after).
    A tile wider than the canvas only wastes work, so the 32 rows of the LM
    head keep 32. 80 is the widest whose shared memory fits the 48 KiB
    static limit.
    """
    var best = 32
    for bn in [32, 64, 80]:
        if _cdiv(n_tokens, bn) < _cdiv(n_tokens, best):
            best = bn
    return best


def _pick_q4_bn(n_tokens: Int, n_rows: Int, sm_count: Int) -> Int:
    """Use the narrowest Q4_K tile that does not add a block wave.

    Estimate one resident block per SM; this is a scheduling heuristic,
    not a measured occupancy guarantee. Fewer token tiles alone can waste
    lanes on a wide GPU: at
    148 tokens and 4096 weight rows, both BN=80 (64 blocks) and BN=64
    (96 blocks) fit in one 156-SM wave. The narrower tile does less work.
    Zero SMs preserves the existing policy for unmeasured device families.
    """
    var best = _pick_bn(n_tokens)
    if sm_count <= 0:
        return best
    var row_tiles = _cdiv(n_rows, BM4)
    var waves = _cdiv(row_tiles * _cdiv(n_tokens, best), sm_count)
    for bn in [32, 64, 80]:
        if _cdiv(row_tiles * _cdiv(n_tokens, bn), sm_count) <= waves:
            return bn
    return best


def gemm_gpu(
    ctx: DeviceContext,
    ggml_type: Int,
    w: U8Ptr,
    N: Int,
    K: Int,
    qa: GpuQActs,
    n_tokens: Int,
    dst: F32Ptr,
    ldo: Int,
) raises:
    """dst[t, r] = sum_k W[r, k] x[t, k] for r < N, t < n_tokens; W in its staged layout.
    """
    if K != qa.K or n_tokens > qa.max_tokens:
        raise Error("activation buffer is the wrong shape")
    var bn = _pick_bn(n_tokens)
    if ggml_type == GGML_Q4_K:
        bn = _pick_q4_bn(n_tokens, N, qa.q4_sm_count)
    if bn == 32:
        _launch[32](ctx, ggml_type, w, N, K, qa, n_tokens, dst, ldo)
    elif bn == 64:
        _launch[64](ctx, ggml_type, w, N, K, qa, n_tokens, dst, ldo)
    elif bn == 80:
        _launch[80](ctx, ggml_type, w, N, K, qa, n_tokens, dst, ldo)
    else:
        raise Error("no GEMM tile of width " + String(bn))
