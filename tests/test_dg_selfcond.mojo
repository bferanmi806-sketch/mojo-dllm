from std.math import exp, sqrt
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.kernels.softembed import EmbeddingT, soft_embed, softmax_rows
from mojo_dllm.models.diffusion_gemma import DiffusionGemma
from mojo_dllm.quant.kquants import dequant_row
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.sys.mem import F32Ptr, I32Ptr, Floats
from mojo_dllm.sys.mmap import Mapping

comptime FIX = "tests/fixtures/"
comptime TINY = FIX + "tiny-diffusion-gemma.gguf"
comptime CANVAS = 16
comptime VOCAB = 512
comptime HIDDEN = 256


def test_softmax_rows_matches_a_scalar_reference() raises:
    var rows = 5
    var width = 300
    var x = Floats(rows * width)
    var want = Floats(rows * width)
    for r in range(rows):
        for i in range(width):
            # Spread of tens, as soft-capped logits are, plus one very low value.
            x[r * width + i] = (
                Float32((i * 7919 + r * 104729) % 601 - 300) / 10.0
            )
        x[r * width + 3] = -1e30
    # Row 4 runs to 900: without the row maximum subtracted its exponentials overflow.
    for i in range(width):
        x[4 * width + i] = Float32(i * 3)
    var inv = Float32(1.25)
    for r in range(rows):
        var mx = Float64(-1e300)
        for i in range(width):
            mx = max(mx, Float64(x[r * width + i]) * Float64(inv))
        var z = Float64(0)
        for i in range(width):
            z += exp(Float64(x[r * width + i]) * Float64(inv) - mx)
        for i in range(width):
            want[r * width + i] = Float32(
                exp(Float64(x[r * width + i]) * Float64(inv) - mx) / z
            )
    softmax_rows(x.ptr(), rows, width, inv, 3)
    var worst = Float64(0)
    for r in range(rows):
        var total = Float64(0)
        for i in range(width):
            worst = max(
                worst,
                abs(Float64(x[r * width + i]) - Float64(want[r * width + i])),
            )
            total += Float64(x[r * width + i])
        assert_true(abs(total - 1.0) < 1e-5, "row sums to " + String(total))
    assert_true(worst < 1e-6, "softmax error " + String(worst))
    assert_equal(x[3], 0.0)


def test_soft_embed_is_probabilities_times_the_embedding_table() raises:
    var g = GGUFFile(TINY)
    var t = g.tensor("token_embd.weight")
    var emb = EmbeddingT(g.data_addr(t), t.ggml_type, VOCAB, HIDDEN, 3)
    var rows = 7
    var probs = Floats(rows * VOCAB)
    for r in range(rows):
        for v in range(VOCAB):
            probs[r * VOCAB + v] = (
                Float32((v * 31 + r * 17) % 97) / 97.0 / Float32(VOCAB)
            )
    var out = Floats(rows * HIDDEN)
    soft_embed(probs.ptr(), emb, out.ptr(), rows, 3)
    var table = Floats(VOCAB * HIDDEN)
    var row_bytes = t.row_bytes()
    for v in range(VOCAB):
        dequant_row(
            t.ggml_type,
            g.data_addr(t) + v * row_bytes,
            table.ptr().unsafe_offset(v * HIDDEN),
            HIDDEN,
        )
    var worst = Float64(0)
    var scale = Float64(0)
    for r in range(rows):
        for e in range(HIDDEN):
            var want = Float64(0)
            for v in range(VOCAB):
                want += Float64(probs[r * VOCAB + v]) * Float64(
                    table[v * HIDDEN + e]
                )
            worst = max(worst, abs(Float64(out[r * HIDDEN + e]) - want))
            scale = max(scale, abs(want))
    assert_true(
        worst < 1e-5 * scale + 1e-7,
        "soft embed error " + String(worst) + " of " + String(scale),
    )


struct Case(Movable):
    var map: Mapping
    var n_tokens: Int
    var n_prompt: Int
    var sc: Int
    var temp_inv: Float32
    var tokens: List[Int]
    var prev_addr: Int
    var expected_addr: Int

    def __init__(out self, name: String) raises:
        self.map = Mapping(FIX + "tiny-diffusion-gemma." + name + ".case")
        var words = I32Ptr(unsafe_from_address=self.map.addr)
        self.n_tokens = Int(words.unsafe_load(0))
        self.n_prompt = Int(words.unsafe_load(1))
        self.sc = Int(words.unsafe_load(2))
        self.temp_inv = F32Ptr(unsafe_from_address=self.map.addr).unsafe_load(3)
        self.tokens = List[Int]()
        for i in range(self.n_tokens):
            self.tokens.append(Int(words.unsafe_load(5 + i)))
        var at = 5 + self.n_tokens
        self.prev_addr = self.map.addr + 4 * at
        if self.sc != 0:
            at += CANVAS * VOCAB
        self.expected_addr = self.map.addr + 4 * at

    def prev(self) -> F32Ptr:
        return F32Ptr(unsafe_from_address=self.prev_addr)

    def expected(self) -> F32Ptr:
        return F32Ptr(unsafe_from_address=self.expected_addr)


def _rel_error(got: F32Ptr, want: F32Ptr, n: Int) -> Float64:
    var num = Float64(0)
    var den = Float64(0)
    for i in range(n):
        var a = Float64(got.unsafe_load(i))
        var b = Float64(want.unsafe_load(i))
        num += (a - b) * (a - b)
        den += b * b
    return sqrt(num / den)


def _model(max_tokens: Int) raises -> DiffusionGemma:
    return DiffusionGemma(TINY, threads=4, max_tokens=max_tokens)


def _copy_prev(cs: Case) raises -> Floats:
    var buf = Floats(CANVAS * VOCAB)
    for i in range(CANVAS * VOCAB):
        buf[i] = cs.prev().unsafe_load(i)
    return buf^


def _check_sc_logits(name: String, max_tokens: Int) raises:
    var m = _model(max_tokens)
    var cs = Case(name)
    assert_equal(cs.sc, 1)
    var buf = _copy_prev(cs)
    m.forward_sc(cs.tokens, buf.ptr(), cs.temp_inv)
    # Rows agree to rounding but for a few where an int8 rounding step lands on
    # a tie (see test_dg_forward). The tie flips measured on the tiny model
    # leave 15 of 16 rows within 1e-5 for p9sc (worst 5.7e-3), 12 for p0sc
    # (worst 7.2e-3) and 14 for p150sc (worst 4.2e-3). A wrong
    # formula moves every row, so the test asks for a quarter of the rows
    # within 1e-5, which tolerates 12 of 16 flipping on another CPU, and for
    # no row beyond 2e-2.
    var close = 0
    var worst = Float64(0)
    for r in range(CANVAS):
        var e = _rel_error(
            buf.ptr().unsafe_offset(r * VOCAB),
            cs.expected().unsafe_offset(r * VOCAB),
            VOCAB,
        )
        if e < 1e-5:
            close += 1
        worst = max(worst, e)
    assert_true(
        close * 4 >= CANVAS,
        name + ": only " + String(close) + " rows agree within 1e-5",
    )
    assert_true(worst < 2e-2, name + ": worst row error " + String(worst))


def test_forward_sc_matches_numpy() raises:
    _check_sc_logits("p9sc", 64)


def test_forward_sc_matches_numpy_with_no_prompt() raises:
    # The canvas alone: no prompt row to attend to.
    _check_sc_logits("p0sc", 64)


def test_forward_sc_matches_numpy_past_the_sliding_window() raises:
    # 150 prompt tokens against a window of 128: the sliding layer clips the
    # prompt, and self-conditioning rides on top of that.
    _check_sc_logits("p150sc", 200)


def test_self_conditioning_changes_the_logits() raises:
    var m = _model(64)
    var cs = Case("p9sc")
    var plain = Floats(CANVAS * VOCAB)
    m.forward(cs.tokens, plain.ptr())
    var buf = _copy_prev(cs)
    m.forward_sc(cs.tokens, buf.ptr(), cs.temp_inv)
    var e = _rel_error(buf.ptr(), plain.ptr(), CANVAS * VOCAB)
    assert_true(
        e > 1e-2, "self-conditioning moved the logits by only " + String(e)
    )


def test_cached_step_with_self_conditioning_equals_the_unified_pass() raises:
    var m = _model(64)
    var cs = Case("p9sc")
    var unified = _copy_prev(cs)
    m.forward_sc(cs.tokens, unified.ptr(), cs.temp_inv)
    var prompt = List[Int]()
    var canvas = List[Int]()
    for i in range(cs.n_tokens):
        if i < cs.n_prompt:
            prompt.append(cs.tokens[i])
        else:
            canvas.append(cs.tokens[i])
    m.prefill(prompt)
    var cached = _copy_prev(cs)
    m.step_sc(canvas, cached.ptr(), cs.temp_inv)
    var differ = 0
    for i in range(CANVAS * VOCAB):
        if unified[i] != cached[i]:
            differ += 1
    assert_equal(differ, 0)


def test_cached_step_with_self_conditioning_equals_the_unified_pass_past_the_window() raises:
    var m = _model(200)
    var cs = Case("p150sc")
    var unified = _copy_prev(cs)
    m.forward_sc(cs.tokens, unified.ptr(), cs.temp_inv)
    var prompt = List[Int]()
    var canvas = List[Int]()
    for i in range(cs.n_tokens):
        if i < cs.n_prompt:
            prompt.append(cs.tokens[i])
        else:
            canvas.append(cs.tokens[i])
    m.prefill(prompt)
    var cached = _copy_prev(cs)
    m.step_sc(canvas, cached.ptr(), cs.temp_inv)
    var differ = 0
    for i in range(CANVAS * VOCAB):
        if unified[i] != cached[i]:
            differ += 1
    assert_equal(differ, 0)


def test_self_conditioning_guards() raises:
    var m = _model(64)
    var cs = Case("p9sc")
    var buf = _copy_prev(cs)
    # 5e6 passes a bound that ignores the cap (5e6 < 1e8) and fails the real one (1.5e8).
    for bad in [0.0, -1.0, 1e30 * 1e30, 3e38, 1e9, 5e6]:
        with assert_raises(contains="previous temperature"):
            m.forward_sc(cs.tokens, buf.ptr(), Float32(bad))
    # The message states the real bound, not "finite": 5e6 is finite.
    with assert_raises(
        contains="times the logit soft cap 30.0 at most 1e8, got 1 / 5000000.0"
    ):
        m.forward_sc(cs.tokens, buf.ptr(), 5e6)
    # Inside the bound (3e6 times the cap of 30 is 9e7) the logits stay finite.
    var ok = _copy_prev(cs)
    m.forward_sc(cs.tokens, ok.ptr(), 3e6)
    for i in range(CANVAS * VOCAB):
        assert_true(
            ok[i] == ok[i] and abs(ok[i]) < 1e6,
            "logit " + String(i) + " is not finite",
        )
    var canvas = List[Int]()
    for i in range(CANVAS):
        canvas.append(cs.tokens[cs.n_prompt + i])
    with assert_raises(contains="run prefill before step"):
        m.step_sc(canvas, buf.ptr(), cs.temp_inv)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
