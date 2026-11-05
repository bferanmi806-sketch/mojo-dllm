from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.formats.gguf import (
    GGUFFile,
    GGML_F32,
    GGML_Q4_K,
    GGML_Q5_0,
    GGML_Q6_K,
    GGML_Q8_0,
)
from mojo_dllm.models.transformer import DiffusionLM, ModelConfig

comptime FIX = "tests/fixtures/"
comptime DG = FIX + "diffusion-gemma/"
comptime A = "diffusion-gemma."


def _refuses(name: String, needle: String) raises:
    """The config refuses fixture `name`, and the error says `needle`."""
    var g = GGUFFile(DG + name + ".gguf")
    try:
        _ = ModelConfig(g)
    except e:
        if needle not in String(e):
            raise Error(
                name + ": the error does not say '" + needle + "': " + String(e)
            )
        return
    raise Error(name + ": the config accepted the file")


def _required_keys() -> List[String]:
    return [
        A + "block_count",
        A + "embedding_length",
        A + "feed_forward_length",
        A + "attention.head_count",
        A + "attention.head_count_kv",
        A + "rope.freq_base",
        A + "rope.freq_base_swa",
        A + "attention.layer_norm_rms_epsilon",
        A + "expert_count",
        A + "expert_used_count",
        A + "attention.key_length",
        A + "attention.value_length",
        A + "attention.sliding_window",
        A + "attention.sliding_window_pattern",
        A + "attention.key_length_swa",
        A + "attention.value_length_swa",
        A + "expert_feed_forward_length",
        A + "rope.dimension_count",
        A + "rope.dimension_count_swa",
        "diffusion.canvas_length",
        "tokenizer.ggml.mask_token_id",
    ]


def _positive_keys() -> List[String]:
    return [
        A + "block_count",
        A + "attention.head_count",
        A + "attention.key_length",
        A + "attention.key_length_swa",
        A + "attention.sliding_window",
        A + "expert_count",
        A + "expert_feed_forward_length",
        "diffusion.canvas_length",
    ]


# ---- what the file declares ---------------------------------------------------


def test_config_reads_diffusion_gemma_metadata() raises:
    var c = ModelConfig(GGUFFile(FIX + "tiny-diffusion-gemma.gguf"))
    assert_equal(c.arch, "diffusion-gemma")
    assert_equal(c.n_layers, 2)
    assert_equal(c.hidden, 256)
    assert_equal(c.n_heads, 8)
    assert_equal(c.ffn, 512)
    assert_equal(c.vocab, 512)
    assert_equal(c.ctx, 8192)
    assert_equal(c.eps, Float32(1e-6))
    assert_equal(c.mask_id, 4)
    assert_equal(c.layer_kinds, "sg")
    # Without a suffix, the global layers' values; with `_swa`, the sliding ones.
    assert_equal(c.n_kv_heads, 1)
    assert_equal(c.n_kv_heads_swa, 4)
    assert_equal(c.head_dim, 128)
    assert_equal(c.head_dim_swa, 64)
    assert_equal(c.theta, Float32(1000000.0))
    assert_equal(c.theta_swa, Float32(10000.0))
    assert_equal(c.sliding_window, 128)
    assert_equal(c.n_experts, 4)
    assert_equal(c.n_experts_used, 2)
    assert_equal(c.expert_ffn, 64)
    assert_equal(c.softcap, Float32(30.0))
    assert_equal(c.canvas, 16)
    assert_true(c.rope_neox, "diffusion-gemma uses NEOX rope")
    assert_true(not c.shift_logits, "diffusion-gemma does not shift")


def test_each_layer_takes_the_values_of_its_kind() raises:
    var c = ModelConfig(GGUFFile(FIX + "tiny-diffusion-gemma.gguf"))
    assert_true(c.is_sliding(0), "layer 0 is sliding")
    assert_true(not c.is_sliding(1), "layer 1 is global")
    assert_equal(c.kv_heads(0), 4)
    assert_equal(c.kv_heads(1), 1)
    assert_equal(c.head_size(0), 64)
    assert_equal(c.head_size(1), 128)
    assert_equal(c.rope_base(0), Float32(10000.0))
    assert_equal(c.rope_base(1), Float32(1000000.0))


def test_the_other_architectures_have_no_sliding_layers() raises:
    var c = ModelConfig(GGUFFile(FIX + "tiny-dream.gguf"))
    for layer in range(c.n_layers):
        assert_true(not c.is_sliding(layer), "dream has no sliding layers")
        assert_equal(c.kv_heads(layer), 2)
        assert_equal(c.head_size(layer), 64)
        assert_equal(c.rope_base(layer), Float32(1000000.0))
    assert_equal(c.n_experts, 0)
    assert_equal(c.canvas, 0)


def test_the_expert_tensors_and_q5_0_are_in_the_directory() raises:
    var g = GGUFFile(FIX + "tiny-diffusion-gemma.gguf")
    var up = g.tensor("blk.1.ffn_gate_up_exps.weight")
    assert_equal(up.n_dims, 3)
    assert_equal(up.ne0, 256)
    assert_equal(up.ne1, 128)
    assert_equal(up.ne2, 4)
    assert_equal(up.ggml_type, GGML_Q4_K)
    assert_equal(up.n_bytes, 4 * 128 * 144)
    var q5 = g.tensor("blk.0.ffn_down_exps.weight")
    assert_equal(q5.shape_str(), "[64, 256, 4]")
    assert_equal(q5.ggml_type, GGML_Q5_0)
    assert_equal(q5.n_bytes, 4 * 256 * 2 * 22)
    var q8 = g.tensor("blk.1.ffn_down_exps.weight")
    assert_equal(q8.ggml_type, GGML_Q8_0)
    assert_equal(q8.n_bytes, 4 * 256 * 2 * 34)
    assert_equal(g.tensor("self_cond_down.weight").ggml_type, GGML_Q5_0)
    assert_equal(g.tensor("blk.0.attn_v.weight").ggml_type, GGML_Q6_K)
    assert_equal(g.tensor("blk.0.ffn_gate_inp.weight").ggml_type, GGML_F32)
    assert_true(g.has_tensor("blk.0.attn_v.weight"), "sliding layer has V")
    assert_true(
        not g.has_tensor("blk.1.attn_v.weight"), "global layer has no V"
    )


def test_per_layer_arrays_are_read_with_their_element_type() raises:
    var g = GGUFFile(FIX + "tiny-diffusion-gemma.gguf")
    var pattern = g.get_bool_array(A + "attention.sliding_window_pattern")
    assert_equal(len(pattern), 2)
    assert_true(pattern[0], "layer 0 is sliding")
    assert_true(not pattern[1], "layer 1 is global")
    var kv = g.get_int_array(A + "attention.head_count_kv")
    assert_equal(len(kv), 2)
    assert_equal(kv[0], 4)
    assert_equal(kv[1], 1)
    with assert_raises(contains=A + "attention.head_count"):
        _ = g.get_bool_array(A + "attention.head_count")
    with assert_raises(contains=A + "attention.head_count_kv"):
        _ = g.get_bool_array(A + "attention.head_count_kv")


def test_a_smaller_file_with_the_same_checks_passes() raises:
    var c = ModelConfig(GGUFFile(DG + "thin-valid.gguf"))
    assert_equal(c.layer_kinds, "sg")


def test_the_smallest_values_and_the_absent_optional_keys_pass() raises:
    var c = ModelConfig(GGUFFile(DG + "ones.gguf"))
    assert_equal(c.n_layers, 1)
    assert_equal(c.n_heads, 1)
    assert_equal(c.n_kv_heads_swa, 1)
    assert_equal(c.head_dim, 1)
    assert_equal(c.head_dim_swa, 1)
    assert_equal(c.sliding_window, 1)
    assert_equal(c.n_experts, 1)
    assert_equal(c.n_experts_used, 1)
    assert_equal(c.expert_ffn, 1)
    assert_equal(c.canvas, 1)
    # context_length, final_logit_softcapping and attention.causal are optional.
    # A missing soft cap is llama.cpp's default of 30, not "off" (llama-hparams.h).
    assert_equal(c.ctx, 4096)
    assert_equal(c.softcap, Float32(30.0))


def test_a_file_with_one_kind_of_layer_passes() raises:
    # No global layer: no rope table is required. No sliding layer: no attn_v.
    var s = ModelConfig(GGUFFile(DG + "all-sliding.gguf"))
    assert_equal(s.layer_kinds, "ss")
    assert_equal(s.n_kv_heads_swa, 4)
    assert_equal(s.n_kv_heads, 0)
    var g = ModelConfig(GGUFFile(DG + "all-global.gguf"))
    assert_equal(g.layer_kinds, "gg")
    assert_equal(g.n_kv_heads, 1)
    assert_equal(g.n_kv_heads_swa, 0)


# ---- refusals: one fixture per check ------------------------------------------


def test_required_keys_are_named_when_missing() raises:
    for key in _required_keys():
        _refuses("missing-" + key, "required metadata key missing: " + key)


def test_non_positive_values_are_refused_by_key() raises:
    for key in _positive_keys():
        _refuses("zero-" + key, key + " must be positive")


def test_refuses_a_pattern_of_the_wrong_length() raises:
    var key = A + "attention.sliding_window_pattern"
    _refuses("pattern-length", key)
    _refuses("pattern-length", "has length 3")
    _refuses("pattern-length-short", key)
    _refuses("pattern-length-short", "has length 1")


def test_refuses_kv_heads_of_the_wrong_length() raises:
    var key = A + "attention.head_count_kv"
    _refuses("kv-heads-length", key)
    _refuses("kv-heads-length", "has length 3")
    _refuses("kv-heads-length-short", key)
    _refuses("kv-heads-length-short", "has length 1")


def test_refuses_a_pattern_that_is_not_a_bool_array() raises:
    _refuses("pattern-type", A + "attention.sliding_window_pattern")
    _refuses("pattern-type", "bool array")


def test_refuses_kv_heads_that_are_not_an_integer_array() raises:
    _refuses("kv-heads-scalar", A + "attention.head_count_kv")
    _refuses("kv-heads-scalar", "not an array")
    _refuses("kv-heads-type", A + "attention.head_count_kv")
    _refuses("kv-heads-type", "integer array")
    _refuses("kv-heads-bool", A + "attention.head_count_kv")
    _refuses("kv-heads-bool", "integer array")
    _refuses("kv-heads-f64", A + "attention.head_count_kv")
    _refuses("kv-heads-f64", "integer array")
    _refuses("kv-heads-strings", A + "attention.head_count_kv")
    _refuses("kv-heads-strings", "integer array")


def test_refuses_kv_heads_that_do_not_divide_the_query_heads() raises:
    for name in ["kv-heads-divisor", "kv-heads-negative", "kv-heads-zero"]:
        _refuses(name, A + "attention.head_count_kv[")
        _refuses(name, "positive divisor of")


def test_refuses_kv_heads_that_differ_within_a_kind() raises:
    _refuses("kv-heads-uneven", A + "attention.head_count_kv[1]")
    _refuses("kv-heads-uneven", "every sliding layer")
    _refuses("kv-heads-uneven-global", A + "attention.head_count_kv[2]")
    _refuses("kv-heads-uneven-global", "every global layer")


def test_refuses_shared_kv_layers() raises:
    _refuses("shared-kv-layers", A + "attention.shared_kv_layers")
    _refuses("shared-kv-layers", "got 2")


def test_refuses_a_per_layer_input() raises:
    _refuses("per-layer-input", A + "embedding_length_per_layer_input")
    _refuses("per-layer-input", "got 64")


def test_refuses_causal_attention() raises:
    _refuses("causal", A + "attention.causal")


def test_refuses_value_lengths_that_differ_from_the_key_lengths() raises:
    _refuses("value-length", A + "attention.value_length")
    _refuses("value-length", A + "attention.key_length")
    _refuses("value-length-wide", A + "attention.value_length")
    _refuses("value-length-swa", A + "attention.value_length_swa")
    _refuses("value-length-swa", A + "attention.key_length_swa")


def test_refuses_a_rotary_width_that_is_not_the_head_size() raises:
    _refuses("rope-width", A + "rope.dimension_count")
    _refuses("rope-width", "must equal")
    _refuses("rope-width-swa", A + "rope.dimension_count_swa")
    _refuses("rope-width-swa", "must equal")


def test_refuses_expert_counts_that_do_not_fit() raises:
    _refuses("expert-used-over", A + "expert_used_count")
    _refuses("expert-used-zero", A + "expert_used_count")
    _refuses("expert-used-over", A + "expert_count")


def test_refuses_a_negative_softcap() raises:
    _refuses("softcap-negative", A + "final_logit_softcapping")


def test_refuses_a_global_layer_with_attn_v() raises:
    _refuses("global-with-attn-v", "blk.1.attn_v.weight")
    _refuses("global-with-attn-v", A + "attention.sliding_window_pattern")


def test_refuses_a_sliding_layer_without_attn_v() raises:
    _refuses("sliding-without-attn-v", "blk.0.attn_v.weight")
    _refuses("sliding-without-attn-v", A + "attention.sliding_window_pattern")


def test_refuses_a_k_width_that_is_not_heads_times_key_length() raises:
    _refuses(
        "k-width-sliding",
        "tensor shape mismatch for blk.0.attn_k.weight: [256, 512]",
    )
    _refuses("k-width-sliding", A + "attention.key_length_swa")
    _refuses(
        "k-width-global",
        "tensor shape mismatch for blk.1.attn_k.weight: [256, 256]",
    )
    _refuses("k-width-global", A + "attention.key_length")
    _refuses(
        "k-hidden",
        "tensor shape mismatch for blk.0.attn_k.weight: [512, 256]",
    )
    _refuses(
        "k-dims",
        "tensor shape mismatch for blk.1.attn_k.weight: [256, 128, 2]",
    )
    _refuses("k-missing", "tensor not found: blk.1.attn_k.weight")


def test_refuses_a_rope_table_of_the_wrong_width() raises:
    _refuses("rope-freqs-missing", "tensor not found: rope_freqs.weight")
    _refuses(
        "rope-freqs-width",
        "tensor shape mismatch for rope_freqs.weight: [128], expected [64]",
    )


def test_refuses_expert_tensors_that_disagree_with_the_expert_keys() raises:
    _refuses(
        "exps-gate-up",
        (
            "tensor shape mismatch for blk.0.ffn_gate_up_exps.weight: [256, 64,"
            " 4], expected [256, 128, 4]"
        ),
    )
    _refuses("exps-gate-up", A + "expert_feed_forward_length")
    _refuses(
        "exps-gate-up-dims",
        (
            "tensor shape mismatch for blk.0.ffn_gate_up_exps.weight: [256,"
            " 128, 4, 1]"
        ),
    )
    _refuses(
        "exps-down-count",
        (
            "tensor shape mismatch for blk.1.ffn_down_exps.weight: [64, 256,"
            " 8], expected [64, 256, 4]"
        ),
    )
    _refuses("exps-down-count", A + "expert_count")
    _refuses(
        "exps-down-width",
        "tensor shape mismatch for blk.1.ffn_down_exps.weight: [32, 256, 4]",
    )
    _refuses("exps-missing", "tensor not found: blk.1.ffn_gate_up_exps.weight")


def test_refuses_an_embedding_that_is_not_hidden_wide() raises:
    _refuses(
        "embd-width", "tensor shape mismatch for token_embd.weight: [512, 512]"
    )


# ---- the model does not run ---------------------------------------------------


def test_the_llada_model_does_not_run_diffusion_gemma() raises:
    with assert_raises(contains="diffusion-gemma does not run on this model"):
        _ = DiffusionLM(
            FIX + "tiny-diffusion-gemma.gguf", threads=1, max_tokens=4
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
