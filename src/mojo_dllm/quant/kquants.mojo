"""Scalar decoders for the GGML block formats the runtime reads: F32, F16, BF16,
Q5_0, Q8_0, Q4_K, Q5_K and Q6_K.

They are written for clarity. The hot path never calls them for weight matrices:
`PackedMatrix` and `pgemm` in `kernels/packed.mojo` repack and multiply the
blocks. They decode embedding rows and norm vectors, and serve as the oracle
for tests.

Block layouts, from ggml-common.h:

  Q4_K, 144 bytes per 256 weights:
    d f16 | dmin f16 | scales[12] (8 x 6-bit scale, 8 x 6-bit min) | qs[128] nibbles
    w = d * sc[j] * q - dmin * m[j], sub-block j = 32 weights

  Q6_K, 210 bytes per 256 weights:
    ql[128] low 4 bits | qh[64] high 2 bits | scales[16] int8 | d f16
    w = d * sc[g] * (q - 32), group g = 16 weights
"""

from std.memory import bitcast

from mojo_dllm.formats.gguf import (
    GGML_F32,
    GGML_F16,
    GGML_BF16,
    GGML_Q5_0,
    GGML_Q8_0,
    GGML_Q4_K,
    GGML_Q5_K,
    GGML_Q6_K,
    TensorInfo,
    ggml_type_name,
)
from mojo_dllm.sys.mem import F32Ptr, U8Ptr

comptime QK_K = 256
comptime Q4K_BYTES = 144
comptime Q5K_BYTES = 176
comptime Q6K_BYTES = 210


@always_inline
def f16_to_f32(bits: UInt16) -> Float32:
    return bitcast[DType.float16](bits).cast[DType.float32]()


@always_inline
def load_f16(p: U8Ptr, off: Int) -> Float32:
    var lo = UInt16(p.unsafe_load(off))
    var hi = UInt16(p.unsafe_load(off + 1))
    return f16_to_f32(lo | (hi << 8))


@always_inline
def q4k_scale_min(q: U8Ptr, j: Int) -> Tuple[Int, Int]:
    """6-bit scale and min of sub-block j from the 12 packed scale bytes."""
    if j < 4:
        return (Int(q.unsafe_load(j) & 63), Int(q.unsafe_load(j + 4) & 63))
    var sc = (q.unsafe_load(j + 4) & 0xF) | ((q.unsafe_load(j - 4) >> 6) << 4)
    var m = (q.unsafe_load(j + 4) >> 4) | ((q.unsafe_load(j) >> 6) << 4)
    return (Int(sc), Int(m))


def dequant_q4k_block(blk: U8Ptr, dst: F32Ptr):
    var d = load_f16(blk, 0)
    var dmin = load_f16(blk, 2)
    var scales = blk.unsafe_offset(4)
    var qs = blk.unsafe_offset(16)
    for j in range(4):
        var a = q4k_scale_min(scales, 2 * j)
        var b = q4k_scale_min(scales, 2 * j + 1)
        var d1 = d * Float32(a[0])
        var m1 = dmin * Float32(a[1])
        var d2 = d * Float32(b[0])
        var m2 = dmin * Float32(b[1])
        for l in range(32):
            var byte = qs.unsafe_load(32 * j + l)
            dst.unsafe_store(64 * j + l, d1 * Float32(byte & 0xF) - m1)
            dst.unsafe_store(64 * j + 32 + l, d2 * Float32(byte >> 4) - m2)


def dequant_q6k_block(blk: U8Ptr, dst: F32Ptr):
    var d = load_f16(blk, 208)
    for h in range(2):
        var ql = blk.unsafe_offset(64 * h)
        var qh = blk.unsafe_offset(128 + 32 * h)
        var sc = blk.unsafe_offset(192 + 8 * h)
        var y = dst.unsafe_offset(128 * h)
        for l in range(32):
            var is_ = l // 16
            var hb = Int(qh.unsafe_load(l))
            var a = Int(ql.unsafe_load(l))
            var b = Int(ql.unsafe_load(l + 32))
            var q1 = ((a & 0xF) | ((hb & 3) << 4)) - 32
            var q2 = ((b & 0xF) | (((hb >> 2) & 3) << 4)) - 32
            var q3 = ((a >> 4) | (((hb >> 4) & 3) << 4)) - 32
            var q4 = ((b >> 4) | (((hb >> 6) & 3) << 4)) - 32
            var s0 = Float32(bitcast[DType.int8](sc.unsafe_load(is_ + 0)))
            var s2 = Float32(bitcast[DType.int8](sc.unsafe_load(is_ + 2)))
            var s4 = Float32(bitcast[DType.int8](sc.unsafe_load(is_ + 4)))
            var s6 = Float32(bitcast[DType.int8](sc.unsafe_load(is_ + 6)))
            y.unsafe_store(l, d * s0 * Float32(q1))
            y.unsafe_store(l + 32, d * s2 * Float32(q2))
            y.unsafe_store(l + 64, d * s4 * Float32(q3))
            y.unsafe_store(l + 96, d * s6 * Float32(q4))


def has_decoder(t: Int) -> Bool:
    """Types accepted by the scalar embedding/vector reference path."""
    return (
        t == GGML_F32
        or t == GGML_F16
        or t == GGML_BF16
        or t == GGML_Q5_0
        or t == GGML_Q8_0
        or t == GGML_Q4_K
        or t == GGML_Q5_K
        or t == GGML_Q6_K
    )


def has_matrix_kernel(t: Int) -> Bool:
    """Types with both CPU and GPU matrix kernels."""
    return t == GGML_Q4_K or t == GGML_Q6_K


def check_embedding_type(t: TensorInfo) raises:
    """Reject unsupported scalar decoding before accessing embedding data."""
    if not has_decoder(t.ggml_type):
        raise Error(
            "unsupported GGUF tensor type: "
            + t.type_name()
            + " (tensor "
            + t.name
            + ")"
        )


def check_matrix_type(t: TensorInfo) raises:
    if not has_decoder(t.ggml_type):
        raise Error(
            "unsupported GGUF tensor type: "
            + t.type_name()
            + " (tensor "
            + t.name
            + ")"
        )
    if not has_matrix_kernel(t.ggml_type):
        raise Error(
            "tensor "
            + t.name
            + " is "
            + t.type_name()
            + ": matrix kernels exist for Q4_K and Q6_K"
        )


def dequant_q5_0_block(blk: U8Ptr, dst: F32Ptr):
    """One 22-byte block: f16 scale, 32 high bits, 16 bytes of low nibbles.

    Weight j of the first half takes high bit j and weight j of the second half
    takes high bit j + 16. Each 5-bit value is offset by 16 before the scale.
    """
    var d = load_f16(blk, 0)
    var qh = (
        UInt32(blk.unsafe_load(2))
        | (UInt32(blk.unsafe_load(3)) << 8)
        | (UInt32(blk.unsafe_load(4)) << 16)
        | (UInt32(blk.unsafe_load(5)) << 24)
    )
    for j in range(16):
        var byte = Int(blk.unsafe_load(6 + j))
        var lo = (byte & 15) | (Int((qh >> UInt32(j)) & 1) << 4)
        var hi = (byte >> 4) | (Int((qh >> UInt32(j + 16)) & 1) << 4)
        dst.unsafe_store(j, Float32(lo - 16) * d)
        dst.unsafe_store(j + 16, Float32(hi - 16) * d)


def dequant_q8_0_block(blk: U8Ptr, dst: F32Ptr):
    """One 34-byte block: f16 scale, then 32 signed quants."""
    var d = load_f16(blk, 0)
    for i in range(32):
        var q = bitcast[DType.int8](blk.unsafe_load(2 + i))
        dst.unsafe_store(i, Float32(q) * d)


def dequant_q5k_block(blk: U8Ptr, dst: F32Ptr):
    """Q4_K scales/mins plus a separate high bit for each of 256 quants."""
    var d = load_f16(blk, 0)
    var dmin = load_f16(blk, 2)
    var scales = blk.unsafe_offset(4)
    var qh = blk.unsafe_offset(16)
    var qs = blk.unsafe_offset(48)
    for j in range(4):
        var a = q4k_scale_min(scales, 2 * j)
        var b = q4k_scale_min(scales, 2 * j + 1)
        var d1 = d * Float32(a[0])
        var m1 = dmin * Float32(a[1])
        var d2 = d * Float32(b[0])
        var m2 = dmin * Float32(b[1])
        for l in range(32):
            var byte = Int(qs.unsafe_load(32 * j + l))
            var high = Int(qh.unsafe_load(l))
            var q1 = (byte & 15) + (((high >> (2 * j)) & 1) << 4)
            var q2 = (byte >> 4) + (((high >> (2 * j + 1)) & 1) << 4)
            dst.unsafe_store(64 * j + l, d1 * Float32(q1) - m1)
            dst.unsafe_store(64 * j + 32 + l, d2 * Float32(q2) - m2)


def dequant_row(ggml_type: Int, src_addr: Int, dst: F32Ptr, n: Int) raises:
    """Decode `n` consecutive elements starting at `src_addr` into f32."""
    var src = U8Ptr(unsafe_from_address=src_addr)
    if ggml_type == GGML_F32:
        var s = F32Ptr(unsafe_from_address=src_addr)
        for i in range(n):
            dst.unsafe_store(i, s.unsafe_load(i))
        return
    if ggml_type == GGML_F16:
        for i in range(n):
            dst.unsafe_store(i, load_f16(src, 2 * i))
        return
    if ggml_type == GGML_BF16:
        for i in range(n):
            var lo = UInt32(src.unsafe_load(2 * i))
            var hi = UInt32(src.unsafe_load(2 * i + 1))
            dst.unsafe_store(i, bitcast[DType.float32]((lo | (hi << 8)) << 16))
        return
    if ggml_type == GGML_Q5_0:
        if n % 32 != 0:
            raise Error("row length " + String(n) + " is not a multiple of 32")
        for b in range(n // 32):
            dequant_q5_0_block(
                src.unsafe_offset(b * 22), dst.unsafe_offset(b * 32)
            )
        return
    if ggml_type == GGML_Q8_0:
        if n % 32 != 0:
            raise Error("row length " + String(n) + " is not a multiple of 32")
        for b in range(n // 32):
            dequant_q8_0_block(
                src.unsafe_offset(b * 34), dst.unsafe_offset(b * 32)
            )
        return
    if n % QK_K != 0:
        raise Error("row length " + String(n) + " is not a multiple of 256")
    if ggml_type == GGML_Q4_K:
        for b in range(n // QK_K):
            dequant_q4k_block(
                src.unsafe_offset(b * Q4K_BYTES), dst.unsafe_offset(b * QK_K)
            )
        return
    if ggml_type == GGML_Q5_K:
        for b in range(n // QK_K):
            dequant_q5k_block(
                src.unsafe_offset(b * Q5K_BYTES), dst.unsafe_offset(b * QK_K)
            )
        return
    if ggml_type == GGML_Q6_K:
        for b in range(n // QK_K):
            dequant_q6k_block(
                src.unsafe_offset(b * Q6K_BYTES), dst.unsafe_offset(b * QK_K)
            )
        return
    raise Error("unsupported GGUF tensor type: " + ggml_type_name(ggml_type))
