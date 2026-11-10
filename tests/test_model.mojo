from std.math import sqrt
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.models.transformer import (
    DiffusionLM,
    ModelConfig,
    check_window,
    _load_matrix,
    _load_vector,
)
from mojo_dllm.sys.mem import F32Ptr, Floats
from mojo_dllm.sys.mmap import Mapping

comptime FIX = "tests/fixtures/"
comptime VOCAB = 512


def _tokens() -> List[Int]:
    return [5, 17, 300, 511, 511, 42, 511, 7]


def test_config_reads_llada_metadata() raises:
    var g = GGUFFile(FIX + "tiny-llada.gguf")
    var c = ModelConfig(g)
    assert_equal(c.n_layers, 2)
    assert_equal(c.hidden, 256)
    assert_equal(c.n_heads, 2)
    assert_equal(c.head_dim, 128)
    assert_equal(c.ffn, 512)
    assert_equal(c.vocab, 512)
    assert_equal(c.mask_id, 511)


def test_forward_matches_numpy_reference() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=4, max_tokens=16)
    var toks = _tokens()
    var rows = List[Int]()
    for i in range(len(toks)):
        rows.append(i)
    var logits = Floats(len(toks) * VOCAB)
    m.forward(toks, rows, logits.ptr())
    var expected = Mapping(FIX + "tiny-llada.expected.f32")
    var ep = F32Ptr(unsafe_from_address=expected.addr)
    var num = Float64(0)
    var den = Float64(0)
    var argmax_agree = 0
    for r in range(len(toks)):
        var best_g = 0
        var best_e = 0
        for v in range(VOCAB):
            var got = Float64(logits[r * VOCAB + v])
            var want = Float64(ep.unsafe_load(r * VOCAB + v))
            num += (got - want) * (got - want)
            den += want * want
            if logits[r * VOCAB + v] > logits[r * VOCAB + best_g]:
                best_g = v
            if ep.unsafe_load(r * VOCAB + v) > ep.unsafe_load(
                r * VOCAB + best_e
            ):
                best_e = v
        if best_g == best_e:
            argmax_agree += 1
    var rel = sqrt(num / den)
    assert_true(rel < 0.02, "relative logit error " + String(rel))
    assert_equal(argmax_agree, len(toks))


def test_forward_subset_rows_equal_full_rows() raises:
    """Asking for logits of a subset of positions returns the same numbers."""
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=4, max_tokens=16)
    var toks = _tokens()
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var full = Floats(len(toks) * VOCAB)
    m.forward(toks, all_rows, full.ptr())
    var some: List[Int] = [3, 6]
    var part = Floats(2 * VOCAB)
    m.forward(toks, some, part.ptr())
    for j in range(2):
        for v in range(VOCAB):
            var a = full[some[j] * VOCAB + v]
            var b = part[j * VOCAB + v]
            assert_true(
                abs(a - b) <= 1e-4, "row " + String(some[j]) + " differs"
            )


def test_rejects_too_long_input() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=4)
    var toks = _tokens()
    var rows: List[Int] = [0]
    var out = Floats(VOCAB)
    with assert_raises(contains="exceeds"):
        m.forward(toks, rows, out.ptr())


def _refuses_window(
    mut m: DiffusionLM,
    toks: List[Int],
    lo: Int,
    hi: Int,
    rows: List[Int],
    needle: String,
) raises:
    """`forward_window` refuses, says `needle`, and writes nothing to the
    logits buffer: a guard that ran late would have written there first. The
    buffer holds a row for every position, so a pass that is wrongly accepted
    fails the assertions instead of overrunning it."""
    var out = Floats(9 * VOCAB)
    for i in range(out.n):
        out[i] = 7.0
    try:
        m.forward_window(toks, lo, hi, rows, out.ptr())
    except e:
        if needle not in String(e):
            raise Error(
                "window ("
                + String(lo)
                + ", "
                + String(hi)
                + "): the error does not say '"
                + needle
                + "': "
                + String(e)
            )
        for i in range(out.n):
            assert_equal(out[i], 7.0, "the refused pass wrote to the logits")
        return
    raise Error("window (" + String(lo) + ", " + String(hi) + "): accepted")


def test_forward_refuses_output_rows_that_are_not_ascending_inside_the_window() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var toks = _tokens()
    var ascending = "output rows must be ascending positions inside the window"
    var a: List[Int] = [3, 3]
    _refuses_window(m, toks, 0, 8, a, ascending)
    var b: List[Int] = [4, 3]
    _refuses_window(m, toks, 0, 8, b, ascending)
    var c: List[Int] = [8]
    _refuses_window(m, toks, 0, 8, c, ascending)
    var d: List[Int] = [-1]
    _refuses_window(m, toks, 0, 8, d, ascending)
    var e: List[Int] = [0, 2, 9]
    _refuses_window(m, toks, 0, 8, e, ascending)


def test_forward_refuses_no_output_rows() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var none = List[Int]()
    _refuses_window(
        m, _tokens(), 0, 8, none, "forward called with no output rows"
    )


def test_forward_refuses_a_window_that_is_not_a_range_inside_the_input() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var toks = _tokens()
    var rows: List[Int] = [3]
    var needle = "forward window must be a non-empty range inside the input"
    _refuses_window(m, toks, 0, 0, rows, needle)
    _refuses_window(m, toks, -1, 4, rows, needle)
    _refuses_window(m, toks, 2, 9, rows, needle)
    _refuses_window(m, toks, 5, 5, rows, needle)
    _refuses_window(m, toks, 6, 2, rows, needle)


def test_a_partial_window_refuses_rows_outside_it() raises:
    # With a warm cache a window (2, 6) is allowed, and its rows must lie in it.
    var m = DiffusionLM(
        FIX + "tiny-llada.gguf", threads=2, max_tokens=16, kv_cache=True
    )
    var toks = _tokens()
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var out = Floats(len(toks) * VOCAB)
    m.forward(toks, all_rows, out.ptr())
    var ascending = "output rows must be ascending positions inside the window"
    var below: List[Int] = [1]
    _refuses_window(m, toks, 2, 6, below, ascending)
    var at_hi: List[Int] = [6]
    _refuses_window(m, toks, 2, 6, at_hi, ascending)
    var inside: List[Int] = [2, 5]
    var two = Floats(2 * VOCAB)
    m.forward(toks, all_rows, out.ptr())
    m.forward_window(toks, 2, 6, inside, two.ptr())


def test_forward_refuses_a_token_id_outside_the_vocabulary() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var rows: List[Int] = [0]
    var below = _tokens()
    below[2] = -1
    var needle = "token id -1 is outside the vocabulary"
    var out = Floats(VOCAB)
    with assert_raises(contains=needle):
        m.forward(below, rows, out.ptr())
    var above = _tokens()
    above[2] = VOCAB
    with assert_raises(contains="token id 512 is outside the vocabulary"):
        m.forward(above, rows, out.ptr())
    # The model still runs afterwards.
    m.forward(_tokens(), rows, out.ptr())


def _window_logits(
    mut m: DiffusionLM, toks: List[Int], win: List[Int]
) raises -> Floats:
    var out = Floats(len(win) * VOCAB)
    m.forward_window(toks, 3, 7, win, out.ptr())
    return out^


def _full_pass(mut m: DiffusionLM, toks: List[Int]) raises -> Floats:
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var out = Floats(len(toks) * VOCAB)
    m.forward(toks, all_rows, out.ptr())
    return out^


def _check_stale_window(path: String, outside: Int) raises:
    # A window pass recomputes rows [3, 7) and reads every other row's K/V from
    # the cache. Changing a token outside the window after the full pass must
    # therefore change nothing: the cache is stale on purpose. A pass that
    # recomputed the outside rows from the tokens would see the new token.
    var m = DiffusionLM(FIX + path, threads=2, max_tokens=16, kv_cache=True)
    var toks = _tokens()
    var win: List[Int] = [3, 4, 6]
    _ = _full_pass(m, toks)
    var w1 = _window_logits(m, toks, win)

    var changed = _tokens()
    changed[outside] = 99
    _ = _full_pass(m, toks)
    var w2 = _window_logits(m, changed, win)
    for i in range(w1.n):
        assert_equal(
            w1[i],
            w2[i],
            path + ": window logit " + String(i) + " saw the changed token",
        )

    # The changed token does matter to a fresh full pass, so the equality
    # above is not an artifact of a token that no row reads.
    _ = _full_pass(m, changed)
    var fresh = _window_logits(m, changed, win)
    var differing = 0
    for i in range(w1.n):
        if fresh[i] != w1[i]:
            differing += 1
    assert_true(
        differing > 0,
        path + ": a fresh pass on the changed canvas equals the stale window",
    )


def test_a_window_reads_stale_cache_rows_llada() raises:
    _check_stale_window("tiny-llada.gguf", 1)
    _check_stale_window("tiny-llada.gguf", 7)


def test_a_window_reads_stale_cache_rows_dream() raises:
    _check_stale_window("tiny-dream.gguf", 1)
    _check_stale_window("tiny-dream.gguf", 7)


def test_the_input_length_limit_is_exact() raises:
    # n_tokens == max_tokens runs; one more raises. The limit sizes every
    # buffer, so a guard that lets one extra token through overruns them.
    var toks = _tokens()
    assert_equal(len(toks), 8)
    var rows: List[Int] = [0]
    # The guard itself, on its own: a model that wrongly accepted the ninth
    # token would overrun its buffers, so this case must fail by assertion.
    check_window(8, 8, 0, 8, rows, False, -1)
    with assert_raises(
        contains="input of 9 tokens exceeds the runtime limit of 8"
    ):
        check_window(9, 8, 0, 9, rows, False, -1)
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=8)
    var out = Floats(VOCAB)
    m.forward(toks, rows, out.ptr())
    var longer = _tokens()
    longer.append(7)
    with assert_raises(
        contains="input of 9 tokens exceeds the runtime limit of 8"
    ):
        m.forward(longer, rows, out.ptr())


def test_load_matrix_refuses_a_shape_that_is_not_the_tensors() raises:
    # kquants.gguf: q4k is 512 wide and 4 rows. A wrong row count or width
    # would let `PackedMatrix` read past the tensor.
    var g = GGUFFile(FIX + "kquants.gguf")
    var ll = GGUFFile(FIX + "tiny-llada.gguf")
    var ok = _load_matrix(ll, "blk.0.attn_q.weight", 256, 256, 1)
    assert_equal(ok.N, 256)
    with assert_raises(
        contains="tensor shape mismatch for q4k: [512, 4], expected [512, 8]"
    ):
        _ = _load_matrix(g, "q4k", 8, 512, 1)
    with assert_raises(
        contains="tensor shape mismatch for q4k: [512, 4], expected [256, 4]"
    ):
        _ = _load_matrix(g, "q4k", 4, 256, 1)
    with assert_raises(contains="tensor not found: absent"):
        _ = _load_matrix(g, "absent", 4, 512, 1)


def test_load_matrix_refuses_a_type_without_a_matrix_kernel() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    with assert_raises(
        contains="tensor f16 is F16: matrix kernels exist for Q4_K and Q6_K"
    ):
        _ = _load_matrix(g, "f16", 2, 64, 1)


def test_load_vector_refuses_a_length_that_is_not_the_tensors() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    var v = _load_vector(g, "f16", 128)
    assert_equal(v.n, 128)
    with assert_raises(contains="tensor shape mismatch for f16: [64, 2]"):
        _ = _load_vector(g, "f16", 64)
    with assert_raises(contains="tensor shape mismatch for f16: [64, 2]"):
        _ = _load_vector(g, "f16", 129)


def test_rejects_wrong_architecture() raises:
    with assert_raises(
        contains=(
            "unsupported model architecture: test (runs: llada, dream; read"
            " only: diffusion-gemma)"
        )
    ):
        _ = DiffusionLM(FIX + "kquants.gguf", threads=1, max_tokens=4)


def _check_against(
    path: String, expected: String, vocab: Int, tol: Float64
) raises:
    var m = DiffusionLM(FIX + path, threads=4, max_tokens=16)
    var toks = _tokens()
    var rows = List[Int]()
    for i in range(len(toks)):
        rows.append(i)
    var logits = Floats(len(toks) * vocab)
    m.forward(toks, rows, logits.ptr())
    var exp_map = Mapping(FIX + expected)
    var ep = F32Ptr(unsafe_from_address=exp_map.addr)
    var num = Float64(0)
    var den = Float64(0)
    var agree = 0
    for r in range(len(toks)):
        var bg = 0
        var be = 0
        for v in range(vocab):
            var got = Float64(logits[r * vocab + v])
            var want = Float64(ep.unsafe_load(r * vocab + v))
            num += (got - want) * (got - want)
            den += want * want
            if logits[r * vocab + v] > logits[r * vocab + bg]:
                bg = v
            if ep.unsafe_load(r * vocab + v) > ep.unsafe_load(r * vocab + be):
                be = v
        if bg == be:
            agree += 1
    var rel = sqrt(num / den)
    assert_true(rel < tol, path + " relative logit error " + String(rel))
    assert_equal(agree, len(toks))


def test_dream_config_reads_gqa_bias_and_rope_style() raises:
    var g = GGUFFile(FIX + "tiny-dream.gguf")
    var c = ModelConfig(g)
    assert_equal(c.arch, "dream")
    assert_equal(c.n_heads, 4)
    assert_equal(c.n_kv_heads, 2)
    assert_equal(c.head_dim, 64)
    assert_true(c.rope_neox, "dream uses NEOX rope")
    assert_true(c.shift_logits, "dream shifts logits")
    var l = ModelConfig(GGUFFile(FIX + "tiny-llada.gguf"))
    assert_true(not l.rope_neox, "llada uses NORM rope")
    assert_true(not l.shift_logits, "llada does not shift")


def test_dream_forward_matches_numpy_reference() raises:
    # The fixture's Q/K biases are large on purpose (dropping them must fail
    # loudly: 51% error). They also sharpen the attention softmax, which
    # amplifies int8 activation error to about 2.6%, hence 5% here.
    _check_against("tiny-dream.gguf", "tiny-dream.expected.f32", 512, 0.05)


def test_logit_row_follows_the_shift() raises:
    var d = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=8)
    assert_equal(d.logit_row(5), 4)
    var l = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=8)
    assert_equal(l.logit_row(5), 5)


def _check_window_exact(path: String) raises:
    # Right after a full pass nothing in the cache is stale, so recomputing a
    # window must give the full pass's logits for those rows, bit for bit:
    # every per-row computation is independent of how rows are grouped.
    var m = DiffusionLM(FIX + path, threads=2, max_tokens=16, kv_cache=True)
    var toks = _tokens()
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var full = Floats(len(toks) * VOCAB)
    m.forward(toks, all_rows, full.ptr())
    var win: List[Int] = [3, 4, 6]
    var part = Floats(len(win) * VOCAB)
    m.forward_window(toks, 3, 7, win, part.ptr())
    for j in range(len(win)):
        for v in range(VOCAB):
            assert_equal(
                part[j * VOCAB + v],
                full[win[j] * VOCAB + v],
                path + " row " + String(win[j]) + " vocab " + String(v),
            )


def test_window_after_full_pass_is_exact_llada() raises:
    _check_window_exact("tiny-llada.gguf")


def test_window_after_full_pass_is_exact_dream() raises:
    _check_window_exact("tiny-dream.gguf")


def test_window_refuses_without_a_matching_cache() raises:
    var toks = _tokens()
    var rows: List[Int] = [3]
    var out = Floats(VOCAB)
    var no_cache = DiffusionLM(
        FIX + "tiny-llada.gguf", threads=2, max_tokens=16
    )
    with assert_raises(contains="needs the K/V cache"):
        no_cache.forward_window(toks, 3, 7, rows, out.ptr())
    var m = DiffusionLM(
        FIX + "tiny-llada.gguf", threads=2, max_tokens=16, kv_cache=True
    )
    with assert_raises(contains="run a full pass first"):
        m.forward_window(toks, 3, 7, rows, out.ptr())


def test_a_failed_pass_invalidates_the_cache() raises:
    # A pass that raises may have rewritten some cache rows, so the cache no
    # longer holds any canvas: the next window must ask for a full pass.
    var toks = _tokens()
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var rows: List[Int] = [3]
    var full = Floats(len(toks) * VOCAB)
    var out = Floats(VOCAB)
    var m = DiffusionLM(
        FIX + "tiny-llada.gguf", threads=2, max_tokens=16, kv_cache=True
    )
    m.forward(toks, all_rows, full.ptr())
    var bad = toks.copy()
    bad[5] = VOCAB
    with assert_raises(contains="outside the vocabulary"):
        m.forward(bad, all_rows, full.ptr())
    with assert_raises(contains="run a full pass first"):
        m.forward_window(toks, 3, 7, rows, out.ptr())
    # The same holds for a failed window pass.
    m.forward(toks, all_rows, full.ptr())
    with assert_raises(contains="outside the vocabulary"):
        m.forward_window(bad, 3, 7, rows, out.ptr())
    with assert_raises(contains="run a full pass first"):
        m.forward_window(toks, 3, 7, rows, out.ptr())


def _refuses_config(name: String, message: String) raises:
    var g = GGUFFile(FIX + "metadata/" + name + ".gguf")
    with assert_raises(contains=message):
        _ = ModelConfig(g)


def test_config_accepts_embedding_and_output_that_match_the_vocabulary() raises:
    var c = ModelConfig(GGUFFile(FIX + "metadata/shapes-valid.gguf"))
    assert_equal(c.vocab, 8)
    assert_equal(c.hidden, 256)
    # A file without an output tensor loads here; the loader names it later.
    _ = ModelConfig(GGUFFile(FIX + "metadata/output-absent.gguf"))


def test_config_refuses_embedding_rows_short_of_the_vocabulary() raises:
    _refuses_config(
        "embd-rows-short",
        (
            "tensor shape mismatch for token_embd.weight: [256, 6], expected"
            " [256, 8] (llada.embedding_length 256, vocabulary 8 from"
            " llada.vocab_size)"
        ),
    )


def test_config_refuses_embedding_rows_beyond_the_vocabulary() raises:
    _refuses_config(
        "embd-rows-long",
        (
            "tensor shape mismatch for token_embd.weight: [256, 10], expected"
            " [256, 8]"
        ),
    )


def test_config_refuses_output_rows_short_of_the_vocabulary() raises:
    _refuses_config(
        "output-rows-short",
        "tensor shape mismatch for output.weight: [256, 6], expected [256, 8]",
    )


def test_config_refuses_an_embedding_of_the_wrong_width() raises:
    _refuses_config(
        "embd-width",
        (
            "tensor shape mismatch for token_embd.weight: [512, 8], expected"
            " [256, 8]"
        ),
    )


def test_vocabulary_read_from_the_output_rows_must_match_the_embedding() raises:
    _refuses_config(
        "vocab-from-output",
        (
            "tensor shape mismatch for token_embd.weight: [256, 6], expected"
            " [256, 8] (llada.embedding_length 256, vocabulary 8 from the rows"
            " of output.weight)"
        ),
    )


def test_cpu_loader_refuses_a_short_embedding_before_any_layer() raises:
    # The file has no layer tensors: without the check, the loader would stop at
    # blk.0.attn_norm.weight, and a token id of 7 would read past the 6 rows.
    with assert_raises(
        contains="tensor shape mismatch for token_embd.weight: [256, 6]"
    ):
        _ = DiffusionLM(
            FIX + "metadata/embd-rows-short.gguf", threads=1, max_tokens=4
        )


def test_config_refuses_a_zero_count_and_names_the_key() raises:
    for key in [
        "llada.block_count",
        "llada.embedding_length",
        "llada.attention.head_count",
        "llada.attention.head_count_kv",
        "llada.feed_forward_length",
        "llada.vocab_size",
    ]:
        _refuses_config(
            "zero-" + String(key), String(key) + " must be positive, got 0"
        )


def test_config_refuses_a_negative_count_and_names_the_key() raises:
    _refuses_config(
        "negative-llada.attention.head_count",
        "llada.attention.head_count must be positive, got -2",
    )
    _refuses_config(
        "negative-llada.attention.head_count_kv",
        "llada.attention.head_count_kv must be positive, got -1",
    )


def test_config_refuses_an_epsilon_that_is_not_finite_and_above_zero() raises:
    for label in ["zero", "negative", "nan", "inf"]:
        _refuses_config(
            String(label) + "-llada.attention.layer_norm_rms_epsilon",
            (
                "llada.attention.layer_norm_rms_epsilon must be finite and"
                " above zero"
            ),
        )


def test_config_refuses_a_rope_base_that_is_not_finite_and_above_zero() raises:
    for label in ["zero", "negative", "nan", "inf"]:
        _refuses_config(
            String(label) + "-llada.rope.freq_base",
            "llada.rope.freq_base must be finite and above zero",
        )


def test_config_rejects_missing_layers() raises:
    var g = GGUFFile(FIX + "metadata/missing-llada.block_count.gguf")
    with assert_raises(
        contains="required metadata key missing: llada.block_count"
    ):
        _ = ModelConfig(g)


def test_config_rejects_missing_hidden() raises:
    var g = GGUFFile(FIX + "metadata/missing-llada.embedding_length.gguf")
    with assert_raises(
        contains="required metadata key missing: llada.embedding_length"
    ):
        _ = ModelConfig(g)


def test_config_rejects_missing_heads() raises:
    var g = GGUFFile(FIX + "metadata/missing-llada.attention.head_count.gguf")
    with assert_raises(
        contains="required metadata key missing: llada.attention.head_count"
    ):
        _ = ModelConfig(g)


def test_config_rejects_missing_ffn() raises:
    var g = GGUFFile(FIX + "metadata/missing-llada.feed_forward_length.gguf")
    with assert_raises(
        contains="required metadata key missing: llada.feed_forward_length"
    ):
        _ = ModelConfig(g)


def test_config_rejects_missing_epsilon() raises:
    var g = GGUFFile(
        FIX + "metadata/missing-llada.attention.layer_norm_rms_epsilon.gguf"
    )
    with assert_raises(
        contains=(
            "required metadata key missing:"
            " llada.attention.layer_norm_rms_epsilon"
        )
    ):
        _ = ModelConfig(g)


def test_config_rejects_missing_mask() raises:
    var g = GGUFFile(FIX + "metadata/missing-tokenizer.ggml.mask_token_id.gguf")
    with assert_raises(
        contains="required metadata key missing: tokenizer.ggml.mask_token_id"
    ):
        _ = ModelConfig(g)


def test_config_rejects_kv_head_shape() raises:
    var g = GGUFFile(FIX + "metadata/kv-heads.gguf")
    with assert_raises(contains="query heads are not a multiple of KV heads"):
        _ = ModelConfig(g)


def test_config_rejects_hidden_shape() raises:
    var g = GGUFFile(FIX + "metadata/hidden-size.gguf")
    with assert_raises(
        contains="hidden size is not divisible by the head count"
    ):
        _ = ModelConfig(g)


def test_config_rejects_rope_width() raises:
    var g = GGUFFile(FIX + "metadata/rope-width.gguf")
    with assert_raises(
        contains=(
            "llada.rope.dimension_count must equal head dimension 128, got 64"
        )
    ):
        _ = ModelConfig(g)


def test_config_rejects_key_length() raises:
    var g = GGUFFile(FIX + "metadata/key-length.gguf")
    with assert_raises(contains="llada.attention.key_length"):
        _ = ModelConfig(g)


def test_config_rejects_rope_scaling_type() raises:
    var g = GGUFFile(FIX + "metadata/rope-type.gguf")
    with assert_raises(contains="llada.rope.scaling.type"):
        _ = ModelConfig(g)


def test_config_rejects_rope_scaling_factor() raises:
    var g = GGUFFile(FIX + "metadata/rope-factor.gguf")
    with assert_raises(contains="llada.rope.scaling.factor must be 1, got 4"):
        _ = ModelConfig(g)


def test_config_accepts_defaults() raises:
    var g = GGUFFile(FIX + "metadata/defaults.gguf")
    var c = ModelConfig(g)
    assert_equal(c.head_dim, 128)
    assert_equal(c.n_kv_heads, 4)
    assert_equal(c.vocab, 512)
    assert_equal(c.ctx, 4096)
    assert_equal(c.theta, 10000.0)
    assert_true(not c.shift_logits)


def test_config_accepts_compatible_none() raises:
    var g = GGUFFile(FIX + "metadata/compatible-none.gguf")
    var c = ModelConfig(g)
    assert_equal(c.head_dim, 128)
    assert_equal(c.n_kv_heads, 4)
    assert_equal(c.vocab, 512)
    assert_equal(c.ctx, 4096)
    assert_equal(c.theta, 10000.0)
    assert_true(not c.shift_logits)


def test_config_accepts_compatible_linear() raises:
    var g = GGUFFile(FIX + "metadata/compatible-linear.gguf")
    var c = ModelConfig(g)
    assert_equal(c.head_dim, 128)
    assert_equal(c.n_kv_heads, 4)
    assert_equal(c.vocab, 512)
    assert_equal(c.ctx, 4096)
    assert_equal(c.theta, 10000.0)
    assert_true(not c.shift_logits)


def test_config_wrong_rotary_width_type_names_key() raises:
    var g = GGUFFile(FIX + "metadata/rope-width-type.gguf")
    with assert_raises(contains="metadata key llada.rope.dimension_count"):
        _ = ModelConfig(g)


def test_config_wrong_scaling_factor_type_names_key() raises:
    var g = GGUFFile(FIX + "metadata/rope-factor-type.gguf")
    with assert_raises(contains="metadata key llada.rope.scaling.factor"):
        _ = ModelConfig(g)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
