from std.math import sqrt
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.models.diffusion_gemma import DiffusionGemma
from mojo_dllm.sys.mem import F32Ptr, I32Ptr, Floats
from mojo_dllm.sys.mmap import Mapping

comptime TINY = "tests/fixtures/tiny-diffusion-gemma.gguf"
comptime CANVAS = 16
comptime VOCAB = 512


def _model(max_tokens: Int) raises -> DiffusionGemma:
    return DiffusionGemma(TINY, threads=4, max_tokens=max_tokens)


def _tokens(n: Int, salt: Int) -> List[Int]:
    var t = List[Int]()
    for i in range(n):
        t.append((i * 37 + salt * 101) % 500 + 5)
    return t^


def _split(tokens: List[Int], p: Int) -> Tuple[List[Int], List[Int]]:
    var prompt = List[Int]()
    var canvas = List[Int]()
    for i in range(len(tokens)):
        if i < p:
            prompt.append(tokens[i])
        else:
            canvas.append(tokens[i])
    return (prompt^, canvas^)


def _differing(a: Floats, b: Floats, n: Int) -> Int:
    var count = 0
    for i in range(n):
        if a[i] != b[i]:
            count += 1
    return count


def _check(p: Int) raises:
    var m = _model(200)
    var tokens = _tokens(p + CANVAS, p)
    var parts = _split(tokens, p)
    var unified = Floats(CANVAS * VOCAB)
    m.forward(tokens, unified.ptr())
    var cached = Floats(CANVAS * VOCAB)
    m.prefill(parts[0])
    m.step(parts[1], cached.ptr())
    var n = _differing(unified, cached, CANVAS * VOCAB)
    assert_equal(
        n, 0, "prompt " + String(p) + ": " + String(n) + " logits differ"
    )


def _rel_error(got: F32Ptr, want: F32Ptr, n: Int) -> Float64:
    var num = Float64(0)
    var den = Float64(0)
    for i in range(n):
        var a = Float64(got.unsafe_load(i))
        var b = Float64(want.unsafe_load(i))
        num += (a - b) * (a - b)
        den += b * b
    return sqrt(num / den)


def _step_matches_numpy(name: String) raises:
    """prefill then step against the numpy logits of scripts/dg_numpy.py.

    The unified pass and the cached one share their layer code, so agreeing
    with each other cannot show a rule they both get wrong; the numpy pass can.
    Row by row, as in test_dg_forward: at least half the rows within 1e-5 and
    none beyond 2e-2 (measured worst: 5.6e-3 for prompt 9, 7.0e-3 for 150).
    """
    var mm = Mapping("tests/fixtures/tiny-diffusion-gemma." + name + ".case")
    var words = I32Ptr(unsafe_from_address=mm.addr)
    var n = Int(words.unsafe_load(0))
    var p = Int(words.unsafe_load(1))
    var tokens = List[Int]()
    for i in range(n):
        tokens.append(Int(words.unsafe_load(5 + i)))
    var want = F32Ptr(unsafe_from_address=mm.addr + 4 * (5 + n))
    var parts = _split(tokens, p)
    var m = _model(200)
    var got = Floats(CANVAS * VOCAB)
    m.prefill(parts[0])
    m.step(parts[1], got.ptr())
    var close = 0
    var worst = Float64(0)
    for r in range(CANVAS):
        var e = _rel_error(
            got.ptr().unsafe_offset(r * VOCAB),
            want.unsafe_offset(r * VOCAB),
            VOCAB,
        )
        if e < 1e-5:
            close += 1
        worst = max(worst, e)
    assert_true(
        close * 2 >= CANVAS,
        name + ": only " + String(close) + " rows within 1e-5",
    )
    assert_true(worst < 2e-2, name + ": worst row error " + String(worst))


def test_step_matches_numpy_prompt_9() raises:
    _step_matches_numpy("p9")


def test_step_matches_numpy_prompt_150() raises:
    _step_matches_numpy("p150")


def test_cached_path_equals_the_unified_pass_prompt_9() raises:
    _check(9)


def test_cached_path_equals_the_unified_pass_prompt_14() raises:
    _check(14)


def test_cached_path_equals_the_unified_pass_prompt_16() raises:
    _check(16)


def test_cached_path_equals_the_unified_pass_prompt_1_and_0() raises:
    _check(1)
    _check(0)


def test_cached_path_equals_the_unified_pass_past_the_window() raises:
    # 150 prompt tokens against a 128-token window: prompt rows are clipped
    # and canvas rows reach the last 127 prompt tokens in the sliding layer.
    _check(150)


def test_one_prefill_serves_many_steps() raises:
    var m = _model(200)
    var tokens = _tokens(20 + CANVAS, 3)
    var parts = _split(tokens, 20)
    m.prefill(parts[0])
    for salt in range(3):
        var canvas = _tokens(CANVAS, 50 + salt)
        var full = List[Int]()
        for t in parts[0]:
            full.append(t)
        for t in canvas:
            full.append(t)
        var cached = Floats(CANVAS * VOCAB)
        m.step(canvas, cached.ptr())
        var other = _model(200)
        var unified = Floats(CANVAS * VOCAB)
        other.forward(full, unified.ptr())
        assert_equal(_differing(cached, unified, CANVAS * VOCAB), 0)


def test_step_needs_a_prefill_and_the_unified_pass_drops_it() raises:
    var m = _model(64)
    var logits = Floats(CANVAS * VOCAB)
    with assert_raises(contains="run prefill before step"):
        m.step(_tokens(CANVAS, 1), logits.ptr())
    m.prefill(_tokens(8, 2))
    m.step(_tokens(CANVAS, 1), logits.ptr())
    # The unified pass overwrites the cache rows, so the prefill is gone.
    m.forward(_tokens(8 + CANVAS, 4), logits.ptr())
    with assert_raises(contains="run prefill before step"):
        m.step(_tokens(CANVAS, 1), logits.ptr())


def test_cache_guards() raises:
    var m = _model(30)
    var logits = Floats(CANVAS * VOCAB)
    with assert_raises(contains="leaves no room for the canvas"):
        m.prefill(_tokens(15, 1))
    # The room guard is exact: 14 + 16 fits in 30, 15 + 16 does not.
    m.prefill(_tokens(14, 1))
    # Token id 0 is a valid token, in a prompt and in a canvas.
    var zero_prompt = _tokens(10, 1)
    zero_prompt[0] = 0
    m.prefill(zero_prompt)
    var zero_canvas = _tokens(CANVAS, 1)
    zero_canvas[0] = 0
    m.step(zero_canvas, logits.ptr())
    with assert_raises(contains="canvas of 16 tokens"):
        m.step(_tokens(CANVAS - 1, 1), logits.ptr())
    var bad = _tokens(CANVAS, 1)
    bad[2] = VOCAB
    with assert_raises(contains="outside the vocabulary"):
        m.step(bad, logits.ptr())
    var bad_prompt = _tokens(5, 1)
    bad_prompt[1] = -1
    with assert_raises(contains="outside the vocabulary"):
        m.prefill(bad_prompt)
    # A refused prefill leaves no prompt behind.
    with assert_raises(contains="run prefill before step"):
        m.step(_tokens(CANVAS, 1), logits.ptr())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
