from std.math import sqrt
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.formats.gguf import (
    GGML_Q4_K,
    GGML_Q5_0,
    GGML_Q8_0,
    ggml_row_bytes,
)
from mojo_dllm.kernels.packed import PackedMatrix, pgemm
from mojo_dllm.kernels.qgemm import QActs, quantize_acts, qgemm
from mojo_dllm.quant.kquants import dequant_row
from mojo_dllm.sys.mem import Bytes, Floats


def _weights(kind: Int, N: Int, K: Int) raises -> Bytes:
    # Generate blocks algorithmically, including both Q5 halves and all Q8
    # signed-byte values. No downloaded model or hand-edited fixture is used.
    var raw = Bytes(N * ggml_row_bytes(kind, K))
    var stride = 22 if kind == GGML_Q5_0 else 34
    for row in range(N):
        for block in range(K // 32):
            var p = raw.u8().unsafe_offset((row * (K // 32) + block) * stride)
            # Alternate exact positive/negative f16 scales, and zero blocks.
            var bits = 0x2800 if block % 3 == 0 else 0xAC00
            if block % 7 == 6:
                bits = 0
            p.unsafe_store(0, UInt8(bits & 255))
            p.unsafe_store(1, UInt8(bits >> 8))
            for j in range(32):
                if kind == GGML_Q5_0:
                    var u = UInt8((row * 7 + block * 11 + j * 13) % 32)
                    var qoff = 6 + j % 16
                    p.unsafe_store(
                        qoff,
                        p.unsafe_load(qoff)
                        | ((u & 15) << UInt8(4 * (j // 16))),
                    )
                    var hoff = 2 + j // 8
                    p.unsafe_store(
                        hoff, p.unsafe_load(hoff) | ((u >> 4) << UInt8(j % 8))
                    )
                else:
                    p.unsafe_store(
                        2 + j, UInt8((row * 31 + block * 17 + j * 29) % 256)
                    )
    return raw^


def _check(kind: Int, K: Int, tokens: Int) raises:
    var N = 24
    var raw = _weights(kind, N, K)
    var pm = PackedMatrix(raw.addr, kind, N, K)
    assert_equal(pm.block_size, 32)
    var x = Floats(tokens * K)
    var state = UInt64(789)
    for i in range(x.n):
        state = state * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int((state >> 33) % 2001) - 1000) / 250.0
    var qa = QActs(tokens, K, 32)
    quantize_acts(x.ptr(), tokens, K, qa)
    var ldo = N + 3
    var out = Floats(tokens * ldo)
    for i in range(out.n):
        out[i] = 12345
    pgemm(pm, qa, tokens, out.ptr(), ldo, 3)
    var w = Floats(K)
    var num = Float64(0)
    var den = Float64(0)
    for row in range(N):
        dequant_row(kind, raw.addr + row * ggml_row_bytes(kind, K), w.ptr(), K)
        for tok in range(tokens):
            var want = Float64(0)
            for k in range(K):
                want += Float64(w[k]) * Float64(x[tok * K + k])
            var diff = Float64(out[tok * ldo + row]) - want
            num += diff * diff
            den += want * want
    assert_true(
        sqrt(num / den) < 0.01,
        "packed32 relative error " + String(sqrt(num / den)),
    )
    for tok in range(tokens):
        for col in range(N, ldo):
            assert_equal(out[tok * ldo + col], 12345, "output stride sentinel")


def test_q5_0_packed_704() raises:
    _check(GGML_Q5_0, 704, 9)


def test_q5_0_packed_2112() raises:
    _check(GGML_Q5_0, 2112, 1)


def test_q8_0_packed_704() raises:
    _check(GGML_Q8_0, 704, 5)


def test_q8_0_packed_2112() raises:
    _check(GGML_Q8_0, 2112, 9)


def test_q8_0_extreme_products_do_not_saturate() raises:
    var raw = Bytes(8 * 34)
    for row in range(8):
        var p = raw.u8().unsafe_offset(row * 34)
        p.unsafe_store(1, 0x3C)
        for k in range(32):
            p.unsafe_store(2 + k, UInt8(127 if row % 2 == 0 else 128))
    var pm = PackedMatrix(raw.addr, GGML_Q8_0, 8, 32)
    var x = Floats(2 * 32)
    for k in range(32):
        x[k] = 127
        x[32 + k] = -127
    var qa = QActs(2, 32, 32)
    quantize_acts(x.ptr(), 2, 32, qa)
    var out = Floats(16)
    pgemm(pm, qa, 2, out.ptr(), 8, 1)
    for tok in range(2):
        for row in range(8):
            var weight = 127 if row % 2 == 0 else -128
            var act = 127 if tok == 0 else -127
            assert_equal(out[tok * 8 + row], Float32(32 * weight * act))


def test_quantize32_zero_and_per_block_scale() raises:
    var x = Floats(64)
    for k in range(32, 64):
        x[k] = Float32(k - 32)
    var qa = QActs(1, 64, 32)
    quantize_acts(x.ptr(), 1, 64, qa)
    assert_equal(qa.d[0], 0)
    assert_equal(qa.d[1], Float32(31.0 / 127.0))
    for k in range(32):
        assert_equal(qa.q.i8().unsafe_load(k), 0)
    assert_equal(qa.q.i8().unsafe_load(63), 127)


def test_packed32_rejects_incompatible_activations() raises:
    var raw = _weights(GGML_Q8_0, 8, 256)
    var pm = PackedMatrix(raw.addr, GGML_Q8_0, 8, 256)
    var qa = QActs(1, 256)
    var out = Floats(8)
    with assert_raises(contains="activation block size"):
        pgemm(pm, qa, 1, out.ptr(), 8, 1)
    var qa32 = QActs(1, 256, 32)
    with assert_raises(contains="output stride"):
        pgemm(pm, qa32, 1, out.ptr(), 7, 1)
    with assert_raises(contains="activation rows"):
        pgemm(pm, qa32, 2, out.ptr(), 8, 1)
    var short_qa = QActs(1, 32, 32)
    with assert_raises(contains="does not match weight rows"):
        pgemm(pm, short_qa, 1, out.ptr(), 8, 1)


def test_quantize32_rejects_bad_dimensions() raises:
    with assert_raises(contains="block size"):
        _ = QActs(1, 704, 64)
    with assert_raises(contains="multiple of 32"):
        _ = QActs(1, 705, 32)
    with assert_raises(contains="positive K"):
        _ = QActs(1, 0, 32)
    with assert_raises(contains="nonnegative tokens"):
        _ = QActs(-1, 32, 32)
    var raw = Bytes(8 * 34)
    with assert_raises(contains="multiple of 8 rows"):
        _ = PackedMatrix(raw.addr, GGML_Q8_0, 0, 32)
    with assert_raises(contains="positive multiple of 32"):
        _ = PackedMatrix(raw.addr, GGML_Q8_0, 8, 0)
    var qa = QActs(1, 32, 32)
    var x = Floats(64)
    with assert_raises(contains="too small"):
        quantize_acts(x.ptr(), 2, 32, qa)


def test_packed32_rejects_incomplete_weight_block() raises:
    # Two blocks per row leave ample backing storage even if the constructor
    # incorrectly truncates K=33 to one block instead of rejecting it.
    var raw = Bytes(8 * 2 * 34)
    with assert_raises(contains="row length 33 is not a multiple of 32"):
        _ = PackedMatrix(raw.addr, GGML_Q8_0, 8, 33)
    _ = raw^


def test_qgemm_rejects_32_value_activation_blocks() raises:
    # The legacy Q4_K path expects one activation scale per 256 values.
    # Oversized zero buffers make a missing refusal fail by assertion rather
    # than by reading or writing beyond the allocated backing storage.
    var raw = Bytes(8 * 210)
    var qa = QActs(1, 256, 32)
    var out = Floats(16)
    with assert_raises(contains="qgemm requires activation block size 256"):
        qgemm(raw.addr, GGML_Q4_K, 8, 256, qa, 1, out.ptr(), 8, 1)
    _ = raw^
    _ = qa^
    _ = out^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
