from std.testing import assert_equal, assert_true, assert_raises, TestSuite
from mojo_dllm.formats.gguf import GGUFFile, TensorInfo
from mojo_dllm.quant.kquants import (
    has_decoder,
    has_matrix_kernel,
    check_matrix_type,
    check_embedding_type,
)
from mojo_dllm.models.transformer import _load_matrix, DiffusionLM
from mojo_dllm.gpu.model import _Planner, _plan_matrix, GpuDiffusionLM


def test_decoder_and_matrix_support() raises:
    for t in List[Int]([0, 1, 30, 6, 8, 12, 13, 14]):
        assert_true(has_decoder(t))
        assert_equal(has_matrix_kernel(t), t == 12 or t == 14)
    for t in List[Int]([16, 2, 7]):
        assert_true(not has_decoder(t))
        assert_true(not has_matrix_kernel(t))


def tensor_type(t: Int) -> TensorInfo:
    return TensorInfo("blk.0.attn_q.weight", 2, 256, 8, 1, 1, t, 0, 0)


def test_matrix_type_errors_name_tensor() raises:
    with assert_raises(
        contains=(
            "unsupported GGUF tensor type: IQ2_XXS (tensor blk.0.attn_q.weight)"
        )
    ):
        check_matrix_type(tensor_type(16))
    with assert_raises(
        contains=(
            "tensor blk.0.attn_q.weight is Q8_0: matrix kernels exist for Q4_K"
            " and Q6_K"
        )
    ):
        check_matrix_type(tensor_type(8))
    for t in List[Int]([0, 1, 30, 13]):
        with assert_raises(contains="matrix kernels exist for Q4_K and Q6_K"):
            check_matrix_type(tensor_type(t))
    check_matrix_type(tensor_type(12))
    check_matrix_type(tensor_type(14))


def test_cpu_loader_rejects_decodable_matrix() raises:
    var g = GGUFFile("tests/fixtures/dequant-types.gguf")
    with assert_raises(
        contains="tensor q8_0 is Q8_0: matrix kernels exist for Q4_K and Q6_K"
    ):
        _ = _load_matrix(g, "q8_0", 4, 96)


def test_gpu_planner_rejects_decodable_matrix_before_allocation() raises:
    var g = GGUFFile("tests/fixtures/dequant-types.gguf")
    var plan = _Planner()
    with assert_raises(
        contains="tensor q8_0 is Q8_0: matrix kernels exist for Q4_K and Q6_K"
    ):
        _ = _plan_matrix(g, plan, "q8_0", 4, 96)
    assert_equal(plan.size, 0)


def test_cpu_constructor_rejects_embedding_by_name() raises:
    with assert_raises(
        contains=(
            "unsupported GGUF tensor type: IQ2_XXS (tensor token_embd.weight)"
        )
    ):
        _ = DiffusionLM(
            "tests/fixtures/unsupported-embedding.gguf", threads=1, max_tokens=1
        )


def test_gpu_constructor_checks_embedding_before_device_creation() raises:
    with assert_raises(
        contains=(
            "unsupported GGUF tensor type: IQ2_XXS (tensor token_embd.weight)"
        )
    ):
        _ = GpuDiffusionLM(
            "tests/fixtures/unsupported-embedding.gguf", max_tokens=1
        )


def test_embedding_types_accept_all_decoders_and_name_unknown_tensor() raises:
    for t in List[Int]([0, 1, 30, 6, 8, 12, 13, 14]):
        check_embedding_type(
            TensorInfo("token_embd.weight", 2, 256, 1, 1, 1, t, 0, 0)
        )
    with assert_raises(
        contains=(
            "unsupported GGUF tensor type: IQ2_XXS (tensor token_embd.weight)"
        )
    ):
        check_embedding_type(
            TensorInfo("token_embd.weight", 2, 256, 1, 1, 1, 16, 0, 0)
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
