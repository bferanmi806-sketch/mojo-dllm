"""Weights repacked at load time so one SIMD lane is one output row.

The GGUF layout stores each row contiguously, which suits a dot product per
row but forces a horizontal sum per output and leaves the block scales scalar.
Here every group of 8 rows is interleaved so that a 256-bit register holds
4 consecutive weights of each of 8 rows. One `vpdpbusd` against 4 broadcast
activation bytes then performs 8 rows x 4 multiply-adds, and the per-row
scales of a sub-block are an 8-lane vector applied once per sub-block. llama.cpp
does the same on CPU (its Q4_K 8x8 repack); the layout here is our own.

Per 8-row group and per 256-weight super-block:

  Q4_K (1216 bytes):
    f32 d[8]  | f32 dmin[8] | u8 sc[8 sub][8 rows] | u8 m[8 sub][8 rows]
    | 8 sub-blocks x 4 steps x 32 bytes of nibbles:
      byte 4r+k of step s in sub-block g: low  = w[r][32g + 8s + k]
                                          high = w[r][32g + 8s + 4 + k]
  Q6_K (2208 bytes):
    f32 d[8]  | i8 sc[16 groups][8 rows]
    | 16 groups x 4 steps x 32 bytes: byte 4r+k = w[r][16g + 4s + k] + 32

Per 8-row group and per 32-weight block:

  Q5_0 / Q8_0 (288 bytes):
    f32 d[8] | 8 steps x 32 bytes: byte 4r+k = w[r][4s+k] + zero_point
    zero_point is 16 for Q5_0 and 128 for Q8_0.

Values stay unsigned, which is what `vpdpbusd` multiplies
against signed int8 activations; the zero points become per-group offsets
dotted with the activation sums, as in `qgemm.mojo`.
"""

from std.memory import bitcast
from std.sys import CompilationTarget, llvm_intrinsic

from mojo_dllm.formats.gguf import (
    GGML_Q5_0,
    GGML_Q8_0,
    GGML_Q4_K,
    GGML_Q6_K,
    ggml_type_name,
    ggml_row_bytes,
)
from mojo_dllm.kernels.qgemm import QActs
from mojo_dllm.quant.kquants import load_f16, q4k_scale_min
from mojo_dllm.sys.mem import Bytes, F32Ptr, I8Ptr, U8Ptr
from mojo_dllm.sys.parallel import (
    parallel_for,
    parallel_for_counted,
    worker_count,
)

comptime QK = 256
comptime P4_SB = 1216
comptime P6_SB = 2208
comptime P32_SB = 288
"""Per 32 weights: f32 scales[8], then 8 steps x 4 bytes x 8 rows."""
comptime TOK_TILE_Q4 = 8
"""Tokens per register tile for Q4_K: 8 independent vpdpbusd chains hide its
latency."""
comptime TOK_TILE_Q6 = 4
"""Tokens per register tile for Q6_K, Q5_0 and Q8_0. Q6_K keeps 4 because its
unpacked byte layout needs more registers; the 32-wide types share the value."""
comptime GROUPS_PER_ITEM = 2


def matrix_bytes(row_bytes: Int, rows: Int) raises -> Int:
    """Bytes of `rows` rows of `row_bytes` each; raises when they do not fit an
    Int, so an untrusted shape cannot wrap into a small size."""
    if (
        rows < 0
        or row_bytes < 0
        or (row_bytes > 0 and rows > Int.MAX // row_bytes)
    ):
        raise Error(
            "a matrix of "
            + String(rows)
            + " rows of "
            + String(row_bytes)
            + " bytes is too large to size"
        )
    return rows * row_bytes


struct PackedMatrix(Movable):
    var kind: Int
    var N: Int
    var K: Int
    var block_size: Int
    var data: Bytes
    var group_bytes: Int
    """Bytes per 8-row group across all blocks."""
    var repack_workers: Int
    """Workers that ran the repack: the setting, capped by the 8-row groups."""

    def __init__(
        out self,
        src_addr: Int,
        src_bytes: Int,
        kind: Int,
        N: Int,
        K: Int,
        threads: Int = 0,
    ) raises:
        """Repacks `N` rows of `K` weights of type `kind` from `src_addr`, the
        8-row groups in parallel on `worker_count(threads)` workers. Each
        group is independent, so the count cannot change a byte.

        `src_bytes` is what the caller knows the source to hold. It must equal
        the bytes of N rows exactly, so a shape that disagrees with the tensor
        it was taken from is an error and not a read past the tensor.
        """
        if (
            kind != GGML_Q4_K
            and kind != GGML_Q6_K
            and kind != GGML_Q5_0
            and kind != GGML_Q8_0
        ):
            raise Error("cannot repack weight type " + ggml_type_name(kind))
        if N <= 0 or N % 8 != 0:
            raise Error(
                "repacked matrices need a multiple of 8 rows, got " + String(N)
            )
        var block_size = 32 if kind == GGML_Q5_0 or kind == GGML_Q8_0 else QK
        if K <= 0 or K % block_size != 0:
            raise Error(
                "row length "
                + String(K)
                + " is not a "
                + ("positive " if K <= 0 else "")
                + "multiple of "
                + String(block_size)
            )
        var row_bytes = ggml_row_bytes(kind, K)
        var want = matrix_bytes(row_bytes, N)
        if src_bytes != want:
            raise Error(
                "the source holds "
                + String(src_bytes)
                + " bytes, but "
                + String(N)
                + " rows of "
                + String(K)
                + " "
                + ggml_type_name(kind)
                + " values take "
                + String(want)
            )
        self.kind = kind
        self.N = N
        self.K = K
        self.block_size = block_size
        var nsb = K // block_size
        var sb_bytes = P4_SB if kind == GGML_Q4_K else P6_SB
        if block_size == 32:
            sb_bytes = P32_SB
        self.group_bytes = nsb * sb_bytes
        # Every byte is written by the repack below, so skip zeroing 5 GB.
        self.data = Bytes((N // 8) * self.group_bytes, zero=False)
        var dst_base = self.data.addr
        var gb = self.group_bytes
        var is_q4 = kind == GGML_Q4_K

        def work(grp: Int) {imm}:
            for sb in range(nsb):
                var dst = U8Ptr(
                    unsafe_from_address=dst_base + grp * gb + sb * sb_bytes
                )
                for r in range(8):
                    var src = U8Ptr(
                        unsafe_from_address=src_addr
                        + (grp * 8 + r) * row_bytes
                        + sb * (row_bytes // nsb)
                    )
                    if is_q4:
                        _pack_q4k_row(src, dst, r)
                    elif kind == GGML_Q6_K:
                        _pack_q6k_row(src, dst, r)
                    else:
                        _pack_32_row(src, dst, r, kind)

        self.repack_workers = parallel_for_counted(
            work, N // 8, worker_count(threads)
        )


def _pack_32_row(blk: U8Ptr, dst: U8Ptr, r: Int, kind: Int):
    dst.unsafe_bitcast[Float32]().unsafe_store(r, load_f16(blk, 0))
    for j in range(32):
        var u: UInt8
        if kind == GGML_Q5_0:
            var nib = (blk.unsafe_load(6 + j % 16) >> UInt8(4 * (j // 16))) & 15
            var high = (blk.unsafe_load(2 + j // 8) >> UInt8(j % 8)) & 1
            u = nib | (high << 4)
        else:
            # Signed Q8 bytes become unsigned with zero point 128.
            u = blk.unsafe_load(2 + j) ^ 128
        dst.unsafe_store(32 + (j // 4) * 32 + r * 4 + j % 4, u)


def _pack_q4k_row(blk: U8Ptr, dst: U8Ptr, r: Int):
    var df = dst.unsafe_bitcast[Float32]()
    df.unsafe_store(r, load_f16(blk, 0))
    df.unsafe_store(8 + r, load_f16(blk, 2))
    var scales = blk.unsafe_offset(4)
    for j in range(8):
        var sm = q4k_scale_min(scales, j)
        dst.unsafe_store(64 + 8 * j + r, UInt8(sm[0]))
        dst.unsafe_store(128 + 8 * j + r, UInt8(sm[1]))
    # Sub-block g holds elements 32g..32g+31: the low nibbles of qs[32*(g/2)..]
    # for even g, the high nibbles for odd g. Each 4-byte output word packs
    # elements 8s+k (low nibble) and 8s+4+k (high nibble), k = 0..3.
    comptime for g in range(8):
        var raw = blk.unsafe_offset(16 + 32 * (g // 2)).unsafe_load[width=32]()
        var w = (raw >> UInt8(4 * (g % 2))) & 0x0F
        comptime for s in range(4):
            var word = w.slice[4, offset=8 * s]() | (
                w.slice[4, offset=8 * s + 4]() << 4
            )
            dst.unsafe_offset(192 + 128 * g + 32 * s + 4 * r).unsafe_store(word)


def _pack_q6k_row(blk: U8Ptr, dst: U8Ptr, r: Int):
    var df = dst.unsafe_bitcast[Float32]()
    df.unsafe_store(r, load_f16(blk, 208))
    for g in range(16):
        dst.unsafe_store(32 + 8 * g + r, blk.unsafe_load(192 + g))
    # Decode 32 values at a time exactly as the GEMM kernel does (q + 32,
    # unsigned), then copy each run of 4 into its place: group g = 16
    # elements, step s = 4 of them.
    comptime for h in range(2):
        var qla = blk.unsafe_offset(64 * h).unsafe_load[width=32]()
        var qlb = blk.unsafe_offset(64 * h + 32).unsafe_load[width=32]()
        var qh = blk.unsafe_offset(128 + 32 * h).unsafe_load[width=32]()
        comptime for i in range(4):
            var lo = qla if i % 2 == 0 else qlb
            var nib = lo & 0x0F if i < 2 else lo >> 4
            var u = nib | (((qh >> UInt8(2 * i)) & 3) << 4)
            comptime for gg in range(2):
                comptime for s in range(4):
                    comptime g = 2 * (4 * h + i) + gg
                    dst.unsafe_offset(
                        160 + 128 * g + 32 * s + 4 * r
                    ).unsafe_store(u.slice[4, offset=16 * gg + 4 * s]())


@always_inline
def _bcast4(p: I8Ptr) -> SIMD[DType.uint8, 32]:
    var v = p.unsafe_bitcast[Int32]().unsafe_load()
    return bitcast[DType.uint8, 32](SIMD[DType.int32, 8](v))


@always_inline
def _dpbusd(
    acc: SIMD[DType.int32, 8],
    w: SIMD[DType.uint8, 32],
    a: SIMD[DType.uint8, 32],
) -> SIMD[DType.int32, 8]:
    comptime if CompilationTarget.has_vnni():
        return llvm_intrinsic[
            "llvm.x86.avx512.vpdpbusd.256", SIMD[DType.int32, 8]
        ](acc, w, a)
    else:
        var p = llvm_intrinsic[
            "llvm.x86.avx2.pmadd.ub.sw", SIMD[DType.int16, 16]
        ](w, bitcast[DType.int8, 32](a))
        return acc + llvm_intrinsic[
            "llvm.x86.avx2.pmadd.wd", SIMD[DType.int32, 8]
        ](p, SIMD[DType.int16, 16](1))


@always_inline
def _dpbusd_q8(
    acc: SIMD[DType.int32, 8],
    w: SIMD[DType.uint8, 32],
    a: SIMD[DType.uint8, 32],
) -> SIMD[DType.int32, 8]:
    comptime if CompilationTarget.has_vnni():
        return _dpbusd(acc, w, a)
    else:
        # pmaddubsw saturates pairs when full-range Q8 bytes multiply int8
        # activations. Splitting bit 7 keeps each pair within signed int16.
        var hi = _dpbusd(SIMD[DType.int32, 8](0), w >> 7, a)
        return _dpbusd(acc, w & 127, a) + hi * 128


@always_inline
def _p32_tile[
    kind: Int, T: Int
](
    wg: U8Ptr,
    nsb: Int,
    qq: I8Ptr,
    qd: F32Ptr,
    qs: F32Ptr,
    K: Int,
    tok0: Int,
    dst: F32Ptr,
    ldo: Int,
    row0: Int,
):
    var facc = Array[SIMD[DType.float32, 8], T](fill=SIMD[DType.float32, 8](0))
    for sb in range(nsb):
        var base = wg.unsafe_offset(sb * P32_SB)
        var dw = base.unsafe_bitcast[Float32]().unsafe_load[width=8]()
        var iacc = Array[SIMD[DType.int32, 8], T](fill=SIMD[DType.int32, 8](0))
        comptime for step in range(8):
            var w = base.unsafe_load[width=32](32 + step * 32)
            comptime for t in range(T):
                var a = _bcast4(
                    qq.unsafe_offset((tok0 + t) * K + sb * 32 + step * 4)
                )
                comptime if kind == GGML_Q8_0:
                    iacc[t] = _dpbusd_q8(iacc[t], w, a)
                else:
                    iacc[t] = _dpbusd(iacc[t], w, a)
        comptime for t in range(T):
            var da = qd.unsafe_load((tok0 + t) * nsb + sb)
            var si = (tok0 + t) * (K // 16) + sb * 2
            var sum_a = qs.unsafe_load(si) + qs.unsafe_load(si + 1)
            comptime ZERO = 128 if kind == GGML_Q8_0 else 16
            facc[t] = facc[t] + dw * (
                iacc[t].cast[DType.float32]() * da - Float32(ZERO) * sum_a
            )
    comptime for t in range(T):
        dst.unsafe_offset((tok0 + t) * ldo + row0).unsafe_store(facc[t])


@always_inline
def _p4_tile[
    T: Int
](
    wg: U8Ptr,
    nsb: Int,
    qq: I8Ptr,
    qd: F32Ptr,
    qs: F32Ptr,
    K: Int,
    tok0: Int,
    dst: F32Ptr,
    ldo: Int,
    row0: Int,
):
    var facc = Array[SIMD[DType.float32, 8], T](fill=SIMD[DType.float32, 8](0))
    for sb in range(nsb):
        var base = wg.unsafe_offset(sb * P4_SB)
        var dv = base.unsafe_bitcast[Float32]().unsafe_load[width=8]()
        var mv = base.unsafe_bitcast[Float32]().unsafe_load[width=8](8)
        var da = SIMD[DType.float32, T](0)
        comptime for t in range(T):
            da[t] = qd.unsafe_load((tok0 + t) * (K // QK) + sb)
        for g in range(8):
            var scv = (
                base.unsafe_load[width=8](64 + 8 * g).cast[DType.float32]() * dv
            )
            var mnv = (
                base.unsafe_load[width=8](128 + 8 * g).cast[DType.float32]()
                * mv
            )
            var iacc = Array[SIMD[DType.int32, 8], T](
                fill=SIMD[DType.int32, 8](0)
            )
            var wp = base.unsafe_offset(192 + 128 * g)
            comptime for s in range(4):
                var v = wp.unsafe_load[width=32](32 * s)
                var lo = v & 0x0F
                var hi = v >> 4
                comptime for t in range(T):
                    var ap = qq.unsafe_offset(
                        (tok0 + t) * K + sb * QK + 32 * g + 8 * s
                    )
                    iacc[t] = _dpbusd(iacc[t], lo, _bcast4(ap))
                    iacc[t] = _dpbusd(iacc[t], hi, _bcast4(ap.unsafe_offset(4)))
            comptime for t in range(T):
                var so = (tok0 + t) * (K // 16) + sb * 16 + 2 * g
                var sg = qs.unsafe_load(so) + qs.unsafe_load(so + 1)
                facc[t] = (
                    facc[t]
                    + scv * (iacc[t].cast[DType.float32]() * da[t])
                    - mnv * sg
                )
    comptime for t in range(T):
        dst.unsafe_offset((tok0 + t) * ldo + row0).unsafe_store(facc[t])


@always_inline
def _p6_tile[
    T: Int
](
    wg: U8Ptr,
    nsb: Int,
    qq: I8Ptr,
    qd: F32Ptr,
    qs: F32Ptr,
    K: Int,
    tok0: Int,
    dst: F32Ptr,
    ldo: Int,
    row0: Int,
):
    var facc = Array[SIMD[DType.float32, 8], T](fill=SIMD[DType.float32, 8](0))
    for sb in range(nsb):
        var base = wg.unsafe_offset(sb * P6_SB)
        var dv = base.unsafe_bitcast[Float32]().unsafe_load[width=8]()
        var da = SIMD[DType.float32, T](0)
        comptime for t in range(T):
            da[t] = qd.unsafe_load((tok0 + t) * (K // QK) + sb)
        for g in range(16):
            var scv = (
                base.unsafe_load[width=8](32 + 8 * g)
                .cast[DType.int8]()
                .cast[DType.float32]()
                * dv
            )
            var iacc = Array[SIMD[DType.int32, 8], T](
                fill=SIMD[DType.int32, 8](0)
            )
            var wp = base.unsafe_offset(160 + 128 * g)
            comptime for s in range(4):
                var v = wp.unsafe_load[width=32](32 * s)
                comptime for t in range(T):
                    var ap = qq.unsafe_offset(
                        (tok0 + t) * K + sb * QK + 16 * g + 4 * s
                    )
                    iacc[t] = _dpbusd(iacc[t], v, _bcast4(ap))
            comptime for t in range(T):
                var sg = qs.unsafe_load((tok0 + t) * (K // 16) + sb * 16 + g)
                facc[t] = facc[t] + scv * (
                    iacc[t].cast[DType.float32]() * da[t] - 32.0 * sg
                )
    comptime for t in range(T):
        dst.unsafe_offset((tok0 + t) * ldo + row0).unsafe_store(facc[t])


@always_inline
def _group_tokens[
    kind: Int, T: Int
](
    pm_addr: Int,
    gb: Int,
    grp: Int,
    nsb: Int,
    qq: I8Ptr,
    qd: F32Ptr,
    qs: F32Ptr,
    K: Int,
    tok0: Int,
    dst: F32Ptr,
    ldo: Int,
):
    var wg = U8Ptr(unsafe_from_address=pm_addr + grp * gb)
    comptime if kind == GGML_Q4_K:
        _p4_tile[T](wg, nsb, qq, qd, qs, K, tok0, dst, ldo, grp * 8)
    elif kind == GGML_Q6_K:
        _p6_tile[T](wg, nsb, qq, qd, qs, K, tok0, dst, ldo, grp * 8)
    else:
        _p32_tile[kind, T](wg, nsb, qq, qd, qs, K, tok0, dst, ldo, grp * 8)


def _run[
    kind: Int
](
    pm: PackedMatrix,
    qa: QActs,
    n_tokens: Int,
    dst: F32Ptr,
    ldo: Int,
    threads: Int,
) raises:
    var n_groups = pm.N // 8
    var items = (n_groups + GROUPS_PER_ITEM - 1) // GROUPS_PER_ITEM
    var addr = pm.data.addr
    var gb = pm.group_bytes
    var nsb = pm.K // pm.block_size
    var K = pm.K
    var qq = qa.q.i8()
    var qd = qa.d.ptr()
    var qs = qa.s.ptr()

    def work(item: Int) {imm}:
        var g0 = item * GROUPS_PER_ITEM
        var g1 = min(g0 + GROUPS_PER_ITEM, n_groups)
        var t = 0
        comptime TILE = TOK_TILE_Q4 if kind == GGML_Q4_K else TOK_TILE_Q6
        while t + TILE <= n_tokens:
            for grp in range(g0, g1):
                _group_tokens[kind, TILE](
                    addr, gb, grp, nsb, qq, qd, qs, K, t, dst, ldo
                )
            t += TILE
        while t < n_tokens:
            for grp in range(g0, g1):
                _group_tokens[kind, 1](
                    addr, gb, grp, nsb, qq, qd, qs, K, t, dst, ldo
                )
            t += 1

    parallel_for(work, items, threads)


def pgemm(
    pm: PackedMatrix,
    qa: QActs,
    n_tokens: Int,
    dst: F32Ptr,
    ldo: Int,
    threads: Int,
) raises:
    """dst[m * ldo + n] = sum_k W[n, k] * x[m, k] for every row n, m < n_tokens.
    """
    if qa.block_size != pm.block_size:
        raise Error(
            "activation block size "
            + String(qa.block_size)
            + " does not match "
            + ggml_type_name(pm.kind)
            + " block size "
            + String(pm.block_size)
        )
    if n_tokens < 0 or n_tokens > qa.n_tokens or ldo < pm.N:
        raise Error("pgemm activation rows or output stride are too small")
    if qa.K != pm.K:
        raise Error(
            "activation length "
            + String(qa.K)
            + " does not match weight rows of "
            + String(pm.K)
        )
    if pm.kind == GGML_Q4_K:
        _run[GGML_Q4_K](pm, qa, n_tokens, dst, ldo, threads)
    elif pm.kind == GGML_Q6_K:
        _run[GGML_Q6_K](pm, qa, n_tokens, dst, ldo, threads)
    elif pm.kind == GGML_Q5_0:
        _run[GGML_Q5_0](pm, qa, n_tokens, dst, ldo, threads)
    else:
        _run[GGML_Q8_0](pm, qa, n_tokens, dst, ldo, threads)
