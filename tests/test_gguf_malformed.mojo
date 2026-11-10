from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojo_dllm.formats.gguf import (
    GGUFFile,
    GGML_F32,
    GGML_Q8_0,
    GGML_Q4_K,
    Reader,
    ggml_block_bytes,
    ggml_row_bytes,
)
from mojo_dllm.quant.kquants import dequant_row
from mojo_dllm.kernels.packed import PackedMatrix
from mojo_dllm.sys.mem import Bytes, Floats

comptime FIX = "tests/fixtures/malformed/"


def test_base_fixture_is_valid() raises:
    var g = GGUFFile(FIX + "base.gguf")
    assert_equal(g.n_tensors(), 2)
    assert_equal(g.tensor("f32").n_elements(), 8)
    assert_equal(g.tensor("q4k").n_elements(), 256)
    assert_equal(g.tensor("q4k").ggml_type, GGML_Q4_K)
    assert_equal(len(g.get_int_array("test.array")), 3)
    assert_equal(g.alignment, 32)
    assert_equal(g.tensor("f32").data_offset % 32, 0)


def test_rejects_zero_alignment() raises:
    with assert_raises(contains="general.alignment has invalid value 0"):
        _ = GGUFFile(FIX + "alignment-zero.gguf")


def test_rejects_nonpower_alignment() raises:
    with assert_raises(contains="general.alignment has invalid value 24"):
        _ = GGUFFile(FIX + "alignment-nonpower.gguf")


def test_rejects_alignment_wrong_type() raises:
    with assert_raises(
        contains="general.alignment has type 0, expected UINT32"
    ):
        _ = GGUFFile(FIX + "alignment-type.gguf")


def test_rejects_alignment_array_type() raises:
    with assert_raises(
        contains="general.alignment has type 9, expected UINT32"
    ):
        _ = GGUFFile(FIX + "alignment-array.gguf")


def test_accepts_explicit_alignment() raises:
    var g = GGUFFile(FIX + "alignment-valid.gguf")
    assert_equal(g.alignment, 64)
    assert_equal(g.tensor("f32").data_offset % 64, 0)
    assert_equal(g.tensor("f32").n_elements(), 8)


def test_rejects_wrapped_scalar_count_on_open() raises:
    with assert_raises(
        contains="test.array has invalid array count 4611686018427387906"
    ):
        _ = GGUFFile(FIX + "array-count-wrapped.gguf")


def test_rejects_negative_string_count_on_open() raises:
    with assert_raises(
        contains="test.strings has invalid array count 18446744073709551615"
    ):
        _ = GGUFFile(FIX + "string-count-negative.gguf")


def test_rejects_oversized_string_count_on_open() raises:
    with assert_raises(
        contains="test.strings has invalid array count 4611686018427387904"
    ):
        _ = GGUFFile(FIX + "string-count-large.gguf")


def test_accepts_string_array() raises:
    var g = GGUFFile(FIX + "string-array.gguf")
    assert_equal(len(g.get_str_array("test.strings")), 2)


def test_rejects_empty() raises:
    with assert_raises(contains="file is empty: "):
        _ = GGUFFile(FIX + "empty.gguf")


def test_rejects_short_magic() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "magic-cut.gguf")


def test_rejects_header_tensor_count() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "header-12.gguf")


def test_rejects_header_kv_count() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "header-20.gguf")


def test_rejects_metadata_cut() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "metadata-cut.gguf")


def test_rejects_oversized_key() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "key-length.gguf")


def test_rejects_key_length_that_wraps_negative() raises:
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "key-length-negative.gguf")


def test_rejects_key_length_that_wraps_the_position_sum() raises:
    # 24 + 0x7FFFFFFFFFFFFFF0 wraps to a negative Int, so a check written as
    # `at + n > size` lets it through.
    with assert_raises(contains="GGUF file truncated"):
        _ = GGUFFile(FIX + "key-length-int64max.gguf")


def test_reader_skip_and_string_refuse_a_length_that_wraps() raises:
    # 8 + (Int.MAX - 4) overflows to a negative Int.
    var buf = Bytes(16)
    var r = Reader(buf.addr, 16, 8)
    with assert_raises(contains="GGUF file truncated: wanted"):
        r.skip(Int.MAX - 4)
    assert_equal(r.at, 8)
    with assert_raises(contains="GGUF file truncated: wanted"):
        r.skip(-1)
    r.skip(8)
    assert_equal(r.at, 16)


def test_reader_refuses_a_start_outside_the_file() raises:
    var buf = Bytes(16)
    with assert_raises(contains="GGUF read position -1 is outside the file"):
        _ = Reader(buf.addr, 16, -1)
    with assert_raises(contains="GGUF read position 17 is outside the file"):
        _ = Reader(buf.addr, 16, 17)
    var at_end = Reader(buf.addr, 16, 16)
    with assert_raises(contains="GGUF file truncated"):
        _ = at_end.u8()


def test_accepts_last_read_ending_exactly_at_eof() raises:
    var g = GGUFFile(FIX + "exact-eof.gguf")
    assert_equal(g.n_tensors(), 0)


def test_rejects_version_one() raises:
    with assert_raises(contains="unsupported GGUF version 1"):
        _ = GGUFFile(FIX + "version-1.gguf")


def test_rejects_version_four() raises:
    with assert_raises(contains="unsupported GGUF version 4"):
        _ = GGUFFile(FIX + "version-4.gguf")


def test_rejects_large_tensor_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "tensors-large.gguf")


def test_rejects_negative_tensor_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "tensors-negative.gguf")


def test_rejects_large_kv_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "kv-large.gguf")


def test_rejects_negative_kv_count() raises:
    with assert_raises(contains="GGUF header counts are implausible"):
        _ = GGUFFile(FIX + "kv-negative.gguf")


def test_rejects_unknown_scalar_type() raises:
    with assert_raises(contains="GGUF value type 99 has no fixed size"):
        _ = GGUFFile(FIX + "scalar-type.gguf")


def test_rejects_unknown_array_type() raises:
    with assert_raises(contains="GGUF value type 99 has no fixed size"):
        _ = GGUFFile(FIX + "array-type.gguf")


def test_rejects_nested_array() raises:
    with assert_raises(
        contains="nested GGUF arrays are not supported: test.array"
    ):
        _ = GGUFFile(FIX + "nested-array.gguf")


def test_rejects_zero_dims() raises:
    with assert_raises(contains="tensor f32 has 0 dims"):
        _ = GGUFFile(FIX + "dims-0.gguf")


def test_rejects_five_dims() raises:
    with assert_raises(contains="tensor f32 has 5 dims"):
        _ = GGUFFile(FIX + "dims-5.gguf")


def test_rejects_tensor_data_cut() raises:
    with assert_raises(contains="tensor q4k extends past the end of the file"):
        _ = GGUFFile(FIX + "tensor-cut.gguf")


def test_directory_control_file_is_valid() raises:
    # Every tensor-directory refusal below differs from this file in one field.
    var g = GGUFFile(FIX + "tensors-control.gguf")
    assert_equal(g.n_tensors(), 1)
    assert_equal(g.tensor("f32").n_elements(), 8)
    assert_equal(g.tensor("f32").n_bytes, 32)
    assert_equal(g.tensor("f32").data_offset % 32, 0)
    _ = GGUFFile(FIX + "kv-keys-distinct.gguf")


def test_rejects_negative_tensor_offset() raises:
    # -32 is a multiple of the alignment and points inside the file, before the
    # data section: only a sign check refuses it.
    with assert_raises(contains="tensor f32 has offset 18446744073709551584"):
        _ = GGUFFile(FIX + "tensor-offset-negative.gguf")


def test_rejects_tensor_offset_near_int64_max() raises:
    with assert_raises(contains="tensor f32 extends past the end of the file"):
        _ = GGUFFile(FIX + "tensor-offset-int64max.gguf")


def test_rejects_unaligned_tensor_offset() raises:
    with assert_raises(
        contains=(
            "tensor f32 has offset 16, which is not a multiple of the"
            " alignment 32"
        )
    ):
        _ = GGUFFile(FIX + "tensor-offset-unaligned.gguf")


def test_rejects_zero_dimension() raises:
    with assert_raises(contains="tensor f32 has dimension 1 of 0"):
        _ = GGUFFile(FIX + "tensor-dim-zero.gguf")


def test_rejects_negative_dimension() raises:
    with assert_raises(
        contains="tensor f32 has dimension 1 of 9223372036854775808"
    ):
        _ = GGUFFile(FIX + "tensor-dim-negative.gguf")


def test_rejects_dimensions_whose_product_wraps() raises:
    with assert_raises(contains="tensor q4k has too many elements"):
        _ = GGUFFile(FIX + "tensor-dim-wraps.gguf")


def test_rejects_byte_count_that_wraps() raises:
    with assert_raises(contains="tensor f32 extends past the end of the file"):
        _ = GGUFFile(FIX + "tensor-bytes-wrap.gguf")


def test_rejects_row_count_whose_bytes_wrap() raises:
    with assert_raises(contains="tensor f32 extends past the end of the file"):
        _ = GGUFFile(FIX + "tensor-rows-wrap.gguf")


def test_rejects_row_that_ends_in_a_partial_block() raises:
    with assert_raises(
        contains=(
            "tensor q8 is Q8_0 with 16 elements per row, which is not a"
            " multiple of its block size 32"
        )
    ):
        _ = GGUFFile(FIX + "tensor-row-partial-block.gguf")


def test_rejects_unsupported_tensor_past_the_end() raises:
    with assert_raises(contains="tensor iq2 extends past the end of the file"):
        _ = GGUFFile(FIX + "tensor-unsupported-past-end.gguf")


def test_rejects_duplicate_tensor_name() raises:
    with assert_raises(contains="tensor f32 is listed twice"):
        _ = GGUFFile(FIX + "tensor-name-duplicate.gguf")


def test_rejects_duplicate_metadata_key() raises:
    with assert_raises(contains="metadata key test.value is listed twice"):
        _ = GGUFFile(FIX + "kv-key-duplicate.gguf")


def test_lists_an_unsupported_type_without_a_byte_count() raises:
    var g = GGUFFile(FIX + "tensor-unsupported-type.gguf")
    var t = g.tensor("iq2")
    assert_equal(t.type_name(), "IQ2_XXS")
    assert_true(not t.supported)
    assert_equal(t.n_bytes, 0)
    assert_true(GGUFFile(FIX + "tensors-control.gguf").tensor("f32").supported)


def test_row_bytes_refuses_a_partial_block() raises:
    assert_equal(ggml_row_bytes(GGML_Q8_0, 64), 68)
    with assert_raises(
        contains="row length 33 is not a multiple of the Q8_0 block size 32"
    ):
        _ = ggml_row_bytes(GGML_Q8_0, 33)
    with assert_raises(contains="row length -4 is not a multiple"):
        _ = ggml_row_bytes(GGML_F32, -4)


def test_row_bytes_refuses_a_size_that_wraps() raises:
    assert_equal(ggml_row_bytes(GGML_F32, Int.MAX // 4), Int.MAX // 4 * 4)
    with assert_raises(contains="is too large to size"):
        _ = ggml_row_bytes(GGML_F32, Int.MAX // 4 + 1)
    with assert_raises(contains="is too large to size"):
        _ = ggml_row_bytes(GGML_Q8_0, (Int.MAX // 34 + 1) * 32)


def test_block_bytes_is_zero_for_a_type_it_cannot_size() raises:
    assert_equal(ggml_block_bytes(16), 0)
    assert_equal(ggml_block_bytes(GGML_Q4_K), 144)


def test_rejects_a_key_that_is_not_utf8() raises:
    with assert_raises(
        contains=(
            "metadata entry 1: string at file offset 76 is not valid UTF-8:"
            " byte 3 of 4 is 0xE9"
        )
    ):
        _ = GGUFFile(FIX + "string-invalid-key.gguf")


def test_rejects_a_tensor_name_that_is_not_utf8() raises:
    with assert_raises(contains="tensor entry 0: string at file offset"):
        _ = GGUFFile(FIX + "string-invalid-tensor-name.gguf")


def test_rejects_a_string_value_that_is_not_utf8() raises:
    var g = GGUFFile(FIX + "string-invalid-value.gguf")
    with assert_raises(
        contains=(
            "metadata key test.str: string at file offset 96 is not valid"
            " UTF-8: byte 3 of 4 is 0xE9"
        )
    ):
        _ = g.get_str("test.str")


def test_rejects_a_token_text_that_is_not_utf8() raises:
    var g = GGUFFile(FIX + "string-invalid-token.gguf")
    with assert_raises(
        contains="metadata key tokenizer.ggml.tokens[1]: string at file offset"
    ):
        _ = g.get_str_array("tokenizer.ggml.tokens")
    # The count and the other entries are still known without decoding.
    assert_equal(g.array_len("tokenizer.ggml.tokens"), 3)


def test_accepts_strings_of_every_width() raises:
    var g = GGUFFile(FIX + "string-valid.gguf")
    var wide = "a \u00e9 \u20ac \U0001f600"
    assert_equal(g.get_str("test.str"), wide)
    var tokens = g.get_str_array("tokenizer.ggml.tokens")
    assert_equal(tokens[0], wide)
    assert_equal(tokens[1], "\u00e9")
    assert_equal(g.tensor(wide).n_elements(), 8)


def test_rejects_unsupported_row_type() raises:
    with assert_raises(contains="unsupported GGUF tensor type: IQ2_XXS"):
        _ = ggml_row_bytes(16, 256)


def test_rejects_unsupported_dequant_type() raises:
    var src = Bytes(144)
    var dst = Floats(256)
    with assert_raises(contains="unsupported GGUF tensor type: Q2_K"):
        dequant_row(10, src.addr, dst.ptr(), 256)


def test_rejects_unsupported_packed_type() raises:
    var src = Bytes(8 * 144)
    with assert_raises(contains="cannot repack weight type Q2_K"):
        _ = PackedMatrix(src.addr, src.size, 10, 8, 256)


def test_rejects_incomplete_dequant_block() raises:
    # Keep backing memory valid even when the length guard is mutated away.
    var src = Bytes(144)
    var dst = Floats(300)
    with assert_raises(contains="row length 300 is not a multiple of 256"):
        dequant_row(GGML_Q4_K, src.addr, dst.ptr(), 300)


def test_rejects_incomplete_packed_block() raises:
    var src = Bytes(8 * 144)
    with assert_raises(contains="row length 300 is not a multiple of 256"):
        _ = PackedMatrix(src.addr, src.size, GGML_Q4_K, 8, 300)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
