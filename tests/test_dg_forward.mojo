from std.math import cos, exp, sin, sqrt
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.kernels.ops import (
    attention_ranges,
    rope_apply,
    rope_table_factors,
)
from mojo_dllm.models.diffusion_gemma import (
    DiffusionGemma,
    KeyRange,
    check_block_sizes,
    key_range,
)
from mojo_dllm.sys.mem import Bytes, F32Ptr, I32Ptr, Floats
from mojo_dllm.sys.mmap import Mapping

comptime FIX = "tests/fixtures/"


# The branch's unified mask, one cell, transcribed from
# `llm_graph_input_attn_diffusion::set_input` (diffusion-gemma.cpp). Prompt rows
# are causal over the prompt, clipped to the window in sliding layers. Canvas rows
# see the whole canvas, every prompt token in a global layer, and in a sliding
# layer the prompt tokens from P - window + 1 on.
def _allowed(q: Int, k: Int, p: Int, window: Int, sliding: Bool) -> Bool:
    var q_canvas = q >= p
    var k_canvas = k >= p
    if q_canvas:
        if sliding:
            return k_canvas or k >= p - window + 1
        return True
    var ok = (not k_canvas) and k <= q
    if ok and sliding and q - k >= window:
        ok = False
    return ok


def test_key_range_is_the_branch_mask_cell_for_cell() raises:
    var cells = 0
    for p in [0, 1, 5, 9, 20]:
        for window in [1, 3, 8, 64]:
            for sliding in [False, True]:
                var n = p + 6
                for q in range(n):
                    var r = key_range(q, p, window, sliding, n)
                    for k in range(n):
                        var inside = k >= r.lo and k < r.hi
                        if inside != _allowed(q, k, p, window, sliding):
                            raise Error(
                                "mask differs at q="
                                + String(q)
                                + " k="
                                + String(k)
                                + " P="
                                + String(p)
                                + " window="
                                + String(window)
                                + " sliding="
                                + String(sliding)
                            )
                        cells += 1
    assert_true(cells > 5000, "the comparison covered " + String(cells))


def test_sliding_canvas_row_reaches_window_minus_one_prompt_tokens() raises:
    # P = 10, window 4: a canvas row sees prompt tokens 7, 8, 9 (the last
    # window - 1 = 3) and the canvas, not the 4 a prompt row would see.
    var r = key_range(12, 10, 4, True, 16)
    assert_equal(r.lo, 7)
    assert_equal(r.hi, 16)
    var g = key_range(12, 10, 4, False, 16)
    assert_equal(g.lo, 0)
    assert_equal(g.hi, 16)


def test_prompt_row_never_sees_the_canvas() raises:
    for sliding in [False, True]:
        var r = key_range(9, 10, 4, sliding, 16)
        assert_equal(r.hi, 10)
        assert_true(r.lo <= 9)
    var first = key_range(0, 10, 4, True, 16)
    assert_equal(first.lo, 0)
    assert_equal(first.hi, 1)
    var last = key_range(9, 10, 4, True, 16)
    assert_equal(last.lo, 6)
    assert_equal(last.hi, 10)


def test_rope_table_divides_the_angle_by_the_factor() raises:
    # head 8: four pairs. Factor 1e30 stops a pair from rotating; the others
    # rotate by p * theta^(-2i/8) / factor.
    var factors = Floats(4)
    factors[0] = 1.0
    factors[1] = 1e30
    factors[2] = 2.0
    factors[3] = 1e30
    var c = Floats(6 * 4)
    var s = Floats(6 * 4)
    rope_table_factors(6, 8, 1e6, 0, factors.ptr(), True, c.ptr(), s.ptr())
    var worst = Float64(0)
    for t in range(6):
        for i in range(4):
            var f = Float64(factors[i])
            var ang = (
                Float64(t) * (Float64(1e6) ** (-2.0 * Float64(i) / 8.0)) / f
            )
            worst = max(worst, abs(Float64(c[t * 4 + i]) - cos(ang)))
            worst = max(worst, abs(Float64(s[t * 4 + i]) - sin(ang)))
    assert_true(worst < 1e-6, "table error " + String(worst))
    for t in range(6):
        assert_equal(c[t * 4 + 1], 1.0)
        assert_true(abs(s[t * 4 + 1]) < 1e-20)
    # Pair 2 turns at half the rate it would without the factor.
    var plain = Floats(6 * 4)
    var plain_s = Floats(6 * 4)
    rope_table_factors(
        6, 8, 1e6, 0, factors.ptr(), False, plain.ptr(), plain_s.ptr()
    )
    assert_true(abs(plain_s[5 * 4 + 2] - s[5 * 4 + 2]) > 1e-4)


def test_neox_pairs_i_with_i_plus_half() raises:
    var c = Floats(2 * 4)
    var s = Floats(2 * 4)
    rope_table_factors(2, 8, 100.0, 0, c.ptr(), False, c.ptr(), s.ptr())
    var x = Floats(2 * 8)
    for i in range(16):
        x[i] = Float32(i % 8) + 1.0
    rope_apply(x.ptr(), 2, 1, 8, c.ptr(), s.ptr(), True, 1)
    for t in range(2):
        for i in range(4):
            var a = Float64(i) + 1.0
            var b = Float64(i + 4) + 1.0
            var ang = Float64(t) * (Float64(100.0) ** (-2.0 * Float64(i) / 8.0))
            var want_lo = a * cos(ang) - b * sin(ang)
            var want_hi = a * sin(ang) + b * cos(ang)
            assert_true(abs(Float64(x[t * 8 + i]) - want_lo) < 1e-5)
            assert_true(abs(Float64(x[t * 8 + i + 4]) - want_hi) < 1e-5)


def _attention_case(
    hd: Int, scale: Float32, n_q: Int, n_keys: Int
) raises -> Float64:
    var nh = 4
    var nkv = 2
    var q = Floats(n_q * nh * hd)
    var k = Floats(n_keys * nkv * hd)
    var v = Floats(n_keys * nkv * hd)
    for i in range(n_q * nh * hd):
        q[i] = Float32((i * 7919) % 97 - 48) / 40.0
    for i in range(n_keys * nkv * hd):
        k[i] = Float32((i * 104729) % 89 - 44) / 30.0
        v[i] = Float32((i * 1299709) % 83 - 41) / 20.0
    var lo = Bytes(n_q * 4)
    var hi = Bytes(n_q * 4)
    for i in range(n_q):
        # An empty-looking prefix, a single key, a middle window, everything.
        var a = [0, n_keys - 1, 2, 0][i % 4]
        var b = [3, n_keys, n_keys - 1, n_keys][i % 4]
        lo.i32().unsafe_store(i, Int32(a))
        hi.i32().unsafe_store(i, Int32(b))
    var o = Floats(n_q * nh * hd)
    attention_ranges(
        q.ptr(),
        k.ptr(),
        v.ptr(),
        o.ptr(),
        lo.i32(),
        hi.i32(),
        n_q,
        n_keys,
        nh,
        nkv,
        hd,
        scale,
        3,
    )
    var worst = Float64(0)
    for i in range(n_q):
        var a = Int(lo.i32().unsafe_load(i))
        var b = Int(hi.i32().unsafe_load(i))
        for h in range(nh):
            var kvh = h // (nh // nkv)
            var sc = List[Float64]()
            var mx = Float64(-1e30)
            for j in range(a, b):
                var dot = Float64(0)
                for d in range(hd):
                    dot += Float64(q[(i * nh + h) * hd + d]) * Float64(
                        k[(j * nkv + kvh) * hd + d]
                    )
                sc.append(dot * Float64(scale))
                mx = max(mx, sc[len(sc) - 1])
            var z = Float64(0)
            for t in range(len(sc)):
                sc[t] = exp(sc[t] - mx)
                z += sc[t]
            for d in range(hd):
                var want = Float64(0)
                for j in range(a, b):
                    want += sc[j - a] / z * Float64(v[(j * nkv + kvh) * hd + d])
                var got = Float64(o[(i * nh + h) * hd + d])
                worst = max(worst, abs(got - want))
    return worst


def test_attention_ranges_match_a_scalar_reference() raises:
    for hd in [16, 64]:
        var worst = _attention_case(hd, 1.0, 9, 12)
        assert_true(
            worst < 2e-5, "hd " + String(hd) + " error " + String(worst)
        )


def test_attention_scale_is_the_argument_not_one_over_root_d() raises:
    var worst = _attention_case(64, 0.5, 9, 12)
    assert_true(worst < 2e-5, "scale 0.5 error " + String(worst))
    var worst_root = _attention_case(64, 0.125, 9, 12)
    assert_true(worst_root < 2e-5, "scale 1/8 error " + String(worst_root))


struct Case(Movable):
    """A tiny-fixture case file written by scripts/gen_fixtures.py (layout in dg_cases).
    """

    var map: Mapping
    var n_tokens: Int
    var n_prompt: Int
    var n_trace: Int
    var tokens: List[Int]
    var expected_addr: Int
    var trace_addr: Int

    def __init__(out self, name: String, canvas: Int, vocab: Int) raises:
        self.map = Mapping(FIX + "tiny-diffusion-gemma." + name + ".case")
        var words = I32Ptr(unsafe_from_address=self.map.addr)
        self.n_tokens = Int(words.unsafe_load(0))
        self.n_prompt = Int(words.unsafe_load(1))
        self.n_trace = Int(words.unsafe_load(4))
        var sc = Int(words.unsafe_load(2))
        self.tokens = List[Int]()
        for i in range(self.n_tokens):
            self.tokens.append(Int(words.unsafe_load(5 + i)))
        var at = 5 + self.n_tokens
        if sc != 0:
            at += canvas * vocab
        self.expected_addr = self.map.addr + 4 * at
        self.trace_addr = self.map.addr + 4 * (at + canvas * vocab)

    def expected(self) -> F32Ptr:
        return F32Ptr(unsafe_from_address=self.expected_addr)

    def trace(self) -> F32Ptr:
        return F32Ptr(unsafe_from_address=self.trace_addr)


def _rel_error(got: F32Ptr, want: F32Ptr, n: Int) -> Float64:
    var num = Float64(0)
    var den = Float64(0)
    for i in range(n):
        var a = Float64(got.unsafe_load(i))
        var b = Float64(want.unsafe_load(i))
        num += (a - b) * (a - b)
        den += b * b
    return sqrt(num / den)


def _check_rows(
    what: String, got: F32Ptr, want: F32Ptr, rows: Int, width: Int
) raises:
    """Row by row, the model and the numpy pass agree to rounding, bar a few rows.

    The numpy pass repeats the int8 rounding of the activations, so most rows
    agree to about 2e-7 (f32 against f64 arithmetic). When a rounding step lands
    on a tie in one and not in the other, one activation moves by a whole step
    and that row by 1e-3 to 7e-3. Measured on the tiny model: prompt 9, layer 0:
    1 of 25 rows beyond 1e-5 (worst 6.1e-4); layer 1: 6 of 25 (worst 1.9e-3);
    logits: 6 of 16 rows (worst 5.6e-3). Prompt 150 logits: worst 7.0e-3. Prompt
    rows never see canvas ties and stay at 1e-7. So the test asks that at least
    a quarter of the rows agree within 1e-5 and that none is off by more than
    2e-2. The measured worst case leaves 10 of 16 rows within 1e-5, so a quarter
    (4 of 16) survives 12 rows flipping on another CPU, where "half" survived
    only 8. A wrong formula moves every row by far more than 1e-5, so it leaves
    none within the bound and still fails.
    """
    var close = 0
    var worst = Float64(0)
    for r in range(rows):
        var e = _rel_error(
            got.unsafe_offset(r * width), want.unsafe_offset(r * width), width
        )
        if e < 1e-5:
            close += 1
        worst = max(worst, e)
    assert_true(
        close * 4 >= rows,
        what
        + ": only "
        + String(close)
        + " of "
        + String(rows)
        + " rows agree within 1e-5",
    )
    assert_true(worst < 2e-2, what + ": worst row error " + String(worst))


def _model(max_tokens: Int) raises -> DiffusionGemma:
    return DiffusionGemma(
        FIX + "tiny-diffusion-gemma.gguf", threads=4, max_tokens=max_tokens
    )


def _check_case(name: String) raises:
    var m = _model(200)
    var c = m.config()
    var cs = Case(name, c.canvas, c.vocab)
    var logits = Floats(c.canvas * c.vocab)
    m.forward(cs.tokens, logits.ptr())
    _check_rows(
        name + " logits", logits.ptr(), cs.expected(), c.canvas, c.vocab
    )
    var agree = 0
    for r in range(c.canvas):
        var bg = 0
        var be = 0
        for v in range(c.vocab):
            if logits[r * c.vocab + v] > logits[r * c.vocab + bg]:
                bg = v
            if cs.expected().unsafe_load(
                r * c.vocab + v
            ) > cs.expected().unsafe_load(r * c.vocab + be):
                be = v
        if bg == be:
            agree += 1
    assert_true(
        agree >= c.canvas - 1, name + ": argmax agrees on " + String(agree)
    )


def test_layer_kinds_of_the_tiny_model() raises:
    var m = _model(32)
    assert_equal(len(m.layers), 2)
    assert_true(m.layers[0].sliding)
    assert_equal(m.layers[0].n_kv, 4)
    assert_equal(m.layers[0].head, 64)
    assert_equal(len(m.layers[0].wv), 1)
    assert_true(not m.layers[1].sliding)
    assert_equal(m.layers[1].n_kv, 1)
    assert_equal(m.layers[1].head, 128)
    assert_equal(len(m.layers[1].wv), 0)


def test_each_layer_output_matches_numpy_prompt_9() raises:
    var m = _model(200)
    var c = m.config()
    var cs = Case("p9", c.canvas, c.vocab)
    assert_equal(cs.n_trace, 2)
    for layers in [1, 2]:
        _ = m.run_layers(cs.tokens, layers)
        _check_rows(
            "layer " + String(layers - 1),
            m.x.ptr(),
            cs.trace().unsafe_offset((layers - 1) * cs.n_tokens * c.hidden),
            cs.n_tokens,
            c.hidden,
        )


def test_prompt_rows_match_numpy_to_rounding() raises:
    # Prompt rows never see the canvas, so no canvas rounding tie reaches them:
    # they agree to f32 noise (measured 1.1e-7 at layer 0, 1.6e-7 at layer 1).
    var m = _model(200)
    var c = m.config()
    var cs = Case("p9", c.canvas, c.vocab)
    for layers in [1, 2]:
        _ = m.run_layers(cs.tokens, layers)
        var e = _rel_error(
            m.x.ptr(),
            cs.trace().unsafe_offset((layers - 1) * cs.n_tokens * c.hidden),
            cs.n_prompt * c.hidden,
        )
        assert_true(
            e < 1e-5,
            "prompt rows, layer " + String(layers - 1) + ": " + String(e),
        )


def test_forward_matches_numpy_prompt_9() raises:
    _check_case("p9")


def test_forward_matches_numpy_prompt_150_past_the_window() raises:
    _check_case("p150")


def test_forward_refuses_bad_input() raises:
    var m = _model(20)
    var c = m.config()
    var logits = Floats(c.canvas * c.vocab)
    var short = List[Int]()
    for _ in range(c.canvas - 1):
        short.append(5)
    with assert_raises(contains="at least the canvas length"):
        m.forward(short, logits.ptr())
    var long = List[Int]()
    for _ in range(21):
        long.append(5)
    with assert_raises(contains="exceeds the runtime limit"):
        m.forward(long, logits.ptr())
    var bad = List[Int]()
    for _ in range(c.canvas):
        bad.append(5)
    bad[3] = c.vocab
    with assert_raises(contains="outside the vocabulary"):
        m.forward(bad, logits.ptr())
    bad[3] = -1
    with assert_raises(contains="token id -1 is outside the vocabulary"):
        m.forward(bad, logits.ptr())


def test_model_refuses_other_architectures_and_small_limits() raises:
    with assert_raises(contains="runs diffusion-gemma files, not llada"):
        _ = DiffusionGemma(FIX + "tiny-llada.gguf", threads=1, max_tokens=32)
    with assert_raises(contains="below the canvas length 16"):
        _ = DiffusionGemma(
            FIX + "tiny-diffusion-gemma.gguf", threads=1, max_tokens=15
        )


def test_run_layers_refuses_a_layer_count_out_of_range() raises:
    var m = _model(32)
    var toks = List[Int]()
    for _ in range(16):
        toks.append(5)
    with assert_raises(contains="layer count 3 is out of range"):
        _ = m.run_layers(toks, 3)
    with assert_raises(contains="layer count -1 is out of range"):
        _ = m.run_layers(toks, -1)


def test_attention_refuses_an_empty_or_outside_key_range() raises:
    var q = Floats(16)
    var k = Floats(16)
    var o = Floats(16)
    var lo = Bytes(4)
    var hi = Bytes(4)
    for bad in [[2, 2], [-1, 1], [0, 3]]:
        lo.i32().unsafe_store(0, Int32(bad[0]))
        hi.i32().unsafe_store(0, Int32(bad[1]))
        with assert_raises(contains="is empty or outside 2 keys"):
            attention_ranges(
                q.ptr(),
                k.ptr(),
                k.ptr(),
                o.ptr(),
                lo.i32(),
                hi.i32(),
                1,
                2,
                1,
                1,
                8,
                1.0,
                1,
            )
    lo.i32().unsafe_store(0, 0)
    hi.i32().unsafe_store(0, 1)
    with assert_raises(contains="multiple of 8"):
        attention_ranges(
            q.ptr(),
            k.ptr(),
            k.ptr(),
            o.ptr(),
            lo.i32(),
            hi.i32(),
            1,
            2,
            1,
            1,
            12,
            1.0,
            1,
        )
    with assert_raises(contains="multiple of 8"):
        attention_ranges(
            q.ptr(),
            k.ptr(),
            k.ptr(),
            o.ptr(),
            lo.i32(),
            hi.i32(),
            1,
            2,
            3,
            2,
            8,
            1.0,
            1,
        )


def test_block_size_check_names_the_matrix_that_does_not_fit() raises:
    check_block_sizes(0, 256, 256, 256, 256, 256, 256, 32, 256, 32)
    var names = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "expert gate/up",
        "expert down",
    ]
    for i in range(9):
        var sizes = [256, 256, 256, 256, 256, 256, 32, 256, 32]
        sizes[i] = 32 if sizes[i] == 256 else 256
        with assert_raises(contains="layer 7: " + names[i] + " has"):
            check_block_sizes(
                7,
                sizes[0],
                sizes[1],
                sizes[2],
                sizes[3],
                sizes[4],
                sizes[5],
                sizes[6],
                sizes[7],
                sizes[8],
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
