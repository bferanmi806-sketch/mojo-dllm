"""DiffusionGemma's entropy-bounded denoiser against exact oracles.

The reference is `diffusion_generate_entropy_bound` in llama.cpp PR 24423
(2e715c0639). The DiffusionGemma forward pass (#64) is not a `DenoisingModel`
yet (#277), so a scripted model hands the sampler logits rows chosen so that every
entropy, argmax and accepted set is known in advance.
"""

from std.math import ceil, exp, log
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.diffusion.sampler import (
    GenConfig,
    GenResult,
    NoObserver,
    Rng,
    StepObserver,
    _generate_diffusion_gemma,
    eb_accept,
    eb_initial_canvas,
    eb_row,
    eb_should_stop,
    eb_temperature,
    generate,
)
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.models.transformer import (
    DenoisingModel,
    ModelConfig,
    RowStats,
    Timings,
)
from mojo_dllm.sys.mem import F32Ptr, Floats

comptime ROWS = "tests/fixtures/entropy-rows.tsv"
comptime REGENERATE = "regenerate with scripts/gen_fixtures.py"
comptime CONFIDENT = Float32(200.0)
"""A logit gap so wide that the softmax is one-hot to 1e-100 at any
temperature the schedule reaches, so the row's entropy is below 1e-90."""


# ---------------------------------------------------------------------------
# The scripted model
# ---------------------------------------------------------------------------


struct ScriptedModel(DenoisingModel, Movable):
    """Answers forward call k with script[k] (the last one repeats): rows laid
    out as [canvas row][vocab]. Records the tokens of every call."""

    var cfg: ModelConfig
    var script: List[List[Float32]]
    var seen: List[List[Int]]
    var asked: List[List[Int]]

    def __init__(out self, vocab: Int, var script: List[List[Float32]]) raises:
        self.cfg = ModelConfig(GGUFFile("tests/fixtures/tiny-llada.gguf"))
        self.cfg.arch = "diffusion-gemma"
        self.cfg.vocab = vocab
        self.cfg.shift_logits = False
        self.cfg.mask_id = 4
        self.cfg.canvas = len(script[0]) // vocab if len(script) > 0 else 0
        self.script = script^
        self.seen = List[List[Int]]()
        self.asked = List[List[Int]]()

    def config(self) -> ModelConfig:
        return self.cfg

    def logit_row(self, position: Int) -> Int:
        return position

    def forward(
        mut self, tokens: List[Int], out_rows: List[Int], logits: F32Ptr
    ) raises:
        var call = len(self.seen)
        self.seen.append(tokens.copy())
        self.asked.append(out_rows.copy())
        var rows = self.script[min(call, len(self.script) - 1)].copy()
        var vocab = self.cfg.vocab
        assert_equal(len(rows), len(out_rows) * vocab, "script rows")
        for i in range(len(rows)):
            logits[unsafe_offset=i] = rows[i]

    def forward_window(
        mut self,
        tokens: List[Int],
        lo: Int,
        hi: Int,
        out_rows: List[Int],
        logits: F32Ptr,
    ) raises:
        raise Error("the scripted model has no window")

    def device_row_stats(self) -> Bool:
        return False

    def forward_stats(
        mut self, tokens: List[Int], out_rows: List[Int], mut out: RowStats
    ) raises:
        raise Error("the scripted model has no device row statistics")

    def model_name(self) raises -> String:
        return "scripted"

    def load_time(self) -> Float64:
        return 0.0

    def stats(self) -> Timings:
        return Timings()


def _one_hot(vocab: Int, tok: Int) -> List[Float32]:
    """A row whose softmax is one-hot at `tok`: entropy below 1e-90."""
    var r = List[Float32](length=vocab, fill=Float32(0))
    r[tok] = CONFIDENT
    return r^


def _flat(vocab: Int) -> List[Float32]:
    """A row of equal logits: entropy ln(vocab) at every temperature."""
    return List[Float32](length=vocab, fill=Float32(0))


def _two_way(vocab: Int, gap: Float32) -> List[Float32]:
    """Token 1 leads token 0 by `gap`; everything else is far below."""
    var r = List[Float32](length=vocab, fill=Float32(-CONFIDENT))
    r[0] = 0.0
    r[1] = gap
    return r^


def _two_way_exact(gap: Float32) -> List[Float32]:
    var r = List[Float32]()
    r.append(0.0)
    r.append(gap)
    return r^


def _step(*rows: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for r in rows:
        for v in r:
            out.append(v)
    return out^


def _cfg(canvas: Int, steps: Int = 48) -> GenConfig:
    var c = GenConfig()
    c.gen_length = canvas
    c.eb_steps = steps
    c.seed = 7
    c.threads = 1
    return c^


struct Accepted(StepObserver):
    """The canvas positions (counted from 0) each step accepted."""

    var sets: List[List[Int]]
    var steps: List[Int]
    var totals: List[Int]
    var prompt: Int

    def __init__(out self, prompt: Int):
        self.sets = List[List[Int]]()
        self.steps = List[Int]()
        self.totals = List[Int]()
        self.prompt = prompt

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        var rel = List[Int]()
        for c in committed:
            rel.append(c - self.prompt)
        self.sets.append(rel^)
        self.steps.append(step)
        self.totals.append(total)


def _run(
    mut m: ScriptedModel, prompt: List[Int], cfg: GenConfig
) raises -> GenResult:
    var none = NoObserver()
    return _generate_diffusion_gemma(m, prompt, cfg, none)


def _slice(xs: List[Int], lo: Int, hi: Int) -> List[Int]:
    var out = List[Int]()
    for i in range(lo, hi):
        out.append(xs[i])
    return out^


def _sorted(xs: List[Int]) -> List[Int]:
    var out = xs.copy()
    for a in range(1, len(out)):
        var v = out[a]
        var b = a - 1
        while b >= 0 and out[b] > v:
            out[b + 1] = out[b]
            b -= 1
        out[b + 1] = v
    return out^


def _same(a: List[Int], b: List[Int], what: String) raises:
    assert_equal(len(a), len(b), what + ": length")
    for i in range(len(a)):
        assert_equal(a[i], b[i], what + " at " + String(i))


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------


def test_temperature_first_24th_and_48th_step() raises:
    # t = t_min + (t_max - t_min) * cur / S with cur = S, S-1, ..., 1.
    var first = eb_temperature(0, 48, 0.4, 0.8)
    assert_true(abs(first - 0.8) < 1e-15, "first step " + String(first))
    # The 24th step (index 23) has cur = 25: 0.4 + 0.4 * 25 / 48.
    var mid = eb_temperature(23, 48, 0.4, 0.8)
    assert_true(abs(mid - (0.4 + 0.4 * 25.0 / 48.0)) < 1e-15, String(mid))
    assert_true(abs(mid - 0.6083333333333333) < 1e-15, String(mid))
    # The 48th step (index 47) has cur = 1: 0.4 + 0.4 / 48, not t_min.
    var last = eb_temperature(47, 48, 0.4, 0.8)
    assert_true(abs(last - 0.4083333333333333) < 1e-15, String(last))
    assert_true(last > 0.4, "the schedule never reaches t_min")


def test_temperature_with_one_step_is_t_max() raises:
    assert_true(abs(eb_temperature(0, 1, 0.4, 0.8) - 0.8) < 1e-15)
    assert_true(abs(eb_temperature(0, 1, 0.3, 0.9) - 0.9) < 1e-15)


def test_temperature_falls_every_step() raises:
    var prev = eb_temperature(0, 48, 0.4, 0.8)
    for k in range(1, 48):
        var t = eb_temperature(k, 48, 0.4, 0.8)
        assert_true(t < prev, "temperature rose at step " + String(k))
        prev = t


# ---------------------------------------------------------------------------
# One row: argmax, entropy and the draw, against numpy in f64
# ---------------------------------------------------------------------------


def _floats(text: String) raises -> List[Float32]:
    var out = List[Float32]()
    for tok in text.split(" "):
        out.append(Float32(Float64(String(tok))))
    return out^


def test_row_matches_the_numpy_reference() raises:
    var text: String
    try:
        with open(ROWS, "r") as f:
            text = f.read()
    except:
        raise Error("missing or unreadable " + ROWS + "; " + REGENERATE)
    var cases = 0
    var widest = 0
    for line in text.split("\n"):
        var row = String(line)
        if row == "":
            continue
        var f = row.split("\t")
        assert_equal(len(f), 8, "entropy-rows: eight fields; " + REGENERATE)
        var name = String(f[0])
        var vocab = Int(String(f[1]))
        var temperature = Float64(String(f[2]))
        var u = Float64(String(f[3]))
        var vals = _floats(String(f[7]))
        assert_equal(len(vals), vocab, name + ": vocab")
        var buf = Floats(vocab)
        for i in range(vocab):
            buf[i] = vals[i]
        var got = eb_row(buf.ptr(), vocab, temperature, u)
        assert_equal(got.argmax, Int(String(f[4])), name + ": argmax")
        var want = Float64(String(f[5]))
        assert_true(
            abs(got.entropy - want) < 1e-9,
            name + ": entropy " + String(got.entropy) + " vs " + String(want),
        )
        assert_equal(got.sampled, Int(String(f[6])), name + ": sampled")
        cases += 1
        widest = max(widest, vocab)
    assert_equal(cases, 10, "entropy-rows: ten cases; " + REGENERATE)
    # Rows wider than one chunk exercise the chunk walk of the draw.
    assert_true(widest > 2048, "a row wider than two chunks")


def test_row_of_equal_logits_has_entropy_ln_vocab() raises:
    var vocab = 4096
    var buf = Floats(vocab)
    for i in range(vocab):
        buf[i] = 1.5
    var r = eb_row(buf.ptr(), vocab, 0.8, 0.5)
    assert_equal(r.argmax, 0)  # the first maximum wins
    assert_true(abs(r.entropy - log(Float64(vocab))) < 1e-12, String(r.entropy))
    assert_equal(r.sampled, 2047)  # cum >= 0.5 first holds at index 2047
    # u = 100 / 4096 is exact, so the target 100 equals the running sum after
    # index 99: "reaches" means at least, and index 99 is the answer.
    assert_equal(eb_row(buf.ptr(), vocab, 0.8, 0.0244140625).sampled, 99)


def test_row_at_the_full_vocabulary_width() raises:
    # 262 144 ids is 256 chunks. Equal logits: every sum is an exact integer in
    # f64, so the entropy is ln N and the draw at u is ceil(u N) - 1 exactly.
    var n = 262144
    var buf = Floats(n)
    for i in range(n):
        buf[i] = 2.0
    var flat = eb_row(buf.ptr(), n, 0.8, 0.3)
    assert_equal(flat.argmax, 0)
    assert_true(
        abs(flat.entropy - log(Float64(n))) < 1e-9, String(flat.entropy)
    )
    assert_equal(flat.sampled, 78643)  # ceil(0.3 * 262144) - 1
    # A single large logit in the last chunk carries nearly all the mass: the
    # running maximum moves in the final chunk, and the draw lands there.
    buf[n - 1] = 42.0
    var peak = eb_row(buf.ptr(), n, 0.8, 0.5)
    assert_equal(peak.argmax, n - 1)
    assert_equal(peak.sampled, n - 1)
    # Each other id has p = e^-50 / (1 + (n - 1) e^-50); numpy in f64 gives
    # H = 2.528e-15 for the row. The sum Z rounds to 1, so the entropy comes
    # from the weighted term and agrees to a few percent, not to 1e-12.
    assert_true(
        peak.entropy > 2.4e-15 and peak.entropy < 2.6e-15, String(peak.entropy)
    )


def test_row_draw_picks_the_first_index_that_reaches_the_target() raises:
    # Probabilities (1/2, 1/4, 1/4) from logits ln 2, 0, 0 at temperature 1.
    var buf = Floats(3)
    buf[0] = Float32(log(Float64(2.0)))
    buf[1] = 0.0
    buf[2] = 0.0
    assert_equal(eb_row(buf.ptr(), 3, 1.0, 0.1).sampled, 0)
    assert_equal(eb_row(buf.ptr(), 3, 1.0, 0.49).sampled, 0)
    assert_equal(eb_row(buf.ptr(), 3, 1.0, 0.51).sampled, 1)
    assert_equal(eb_row(buf.ptr(), 3, 1.0, 0.74).sampled, 1)
    assert_equal(eb_row(buf.ptr(), 3, 1.0, 0.76).sampled, 2)
    assert_equal(eb_row(buf.ptr(), 3, 1.0, 0.99).sampled, 2)


def test_row_temperature_scales_the_logits() raises:
    # Two logits 0 and 1: p = sigmoid(1 / t), H = -p ln p - (1 - p) ln (1 - p).
    var buf = Floats(2)
    buf[0] = 0.0
    buf[1] = 1.0
    for t in List[Float64]([0.4, 0.8, 2.0]):
        var p = 1.0 / (1.0 + exp(-1.0 / t))
        var h = -p * log(p) - (1.0 - p) * log(1.0 - p)
        var r = eb_row(buf.ptr(), 2, t, 0.5)
        assert_true(
            abs(r.entropy - h) < 1e-9,
            "t="
            + String(t)
            + " got "
            + String(r.entropy)
            + " want "
            + String(h),
        )
        assert_equal(r.argmax, 1)


def test_row_with_a_non_finite_logit_is_refused_by_name() raises:
    var vocab = 6
    var script = List[List[Float32]]()
    var rows = _one_hot(vocab, 2)
    rows[3] = Float32.MAX * 2  # inf: the row has no finite softmax
    script.append(rows^)
    var m = ScriptedModel(vocab, script^)
    var prompt: List[Int] = [1, 2]
    var none = NoObserver()
    with assert_raises(contains="non-finite"):
        _ = _generate_diffusion_gemma(m, prompt, _cfg(1), none)


# ---------------------------------------------------------------------------
# The acceptance rule
# ---------------------------------------------------------------------------


def _set(flags: List[Bool]) -> List[Int]:
    var out = List[Int]()
    for i in range(len(flags)):
        if flags[i]:
            out.append(i)
    return out^


def test_accept_every_confident_position() raises:
    var e: List[Float64] = [0.001, 0.002, 0.003, 0.004]
    _same(_set(eb_accept(e, 0.1)), [0, 1, 2, 3], "all confident")


def test_accept_only_the_lowest_when_none_is_confident() raises:
    # Each entropy is far above the bound, so only the position with nothing
    # before it passes: the lowest entropy.
    var e: List[Float64] = [2.0, 1.5, 3.0, 1.4]
    _same(_set(eb_accept(e, 0.1)), [3], "none confident")


def test_accept_bound_zero_takes_exactly_the_minimum() raises:
    var e: List[Float64] = [0.3, 0.1, 0.2]
    _same(_set(eb_accept(e, 0.0)), [1], "bound 0")


def test_accept_tie_at_the_bound_is_accepted() raises:
    # Sorted: 0.0625, 0.0625, 0.5. The third has 0.0625 + 0.0625 = 0.125 before
    # it, exactly the bound, and the rule is "at most".
    var e: List[Float64] = [0.5, 0.0625, 0.0625]
    _same(_set(eb_accept(e, 0.125)), [0, 1, 2], "tie at the bound")
    # One ulp under the bound refuses it.
    _same(_set(eb_accept(e, 0.12499999999999999)), [1, 2], "just under")


def test_accept_the_sum_counts_only_what_comes_before() raises:
    # Sorted 0.06, 0.06, 0.06. Before them: 0, 0.06, 0.12. A bound of 0.1 takes
    # the first two. Counting the position itself (0.06, 0.12, 0.18) would
    # take only the first.
    var e: List[Float64] = [0.06, 0.06, 0.06]
    _same(_set(eb_accept(e, 0.1)), [0, 1], "strictly before")


def test_accept_a_bound_that_admits_exactly_one() raises:
    var e: List[Float64] = [0.06, 0.07, 0.9]
    _same(_set(eb_accept(e, 0.05)), [0], "bound 0.05")
    _same(_set(eb_accept(e, 0.06)), [0, 1], "bound 0.06 admits the second")


def test_accept_orders_by_entropy_and_ties_go_to_the_lower_index() raises:
    # Four equal entropies of 0.04 with bound 0.09: before = 0, .04, .08, .12.
    var e: List[Float64] = [0.04, 0.04, 0.04, 0.04]
    _same(_set(eb_accept(e, 0.09)), [0, 1, 2], "ties by index")
    var f: List[Float64] = [0.9, 0.01, 0.9, 0.02]
    _same(_set(eb_accept(f, 0.015)), [1, 3], "order by entropy")


# ---------------------------------------------------------------------------
# The stop rule
# ---------------------------------------------------------------------------


def test_stop_needs_stability_and_confidence() raises:
    assert_true(eb_should_stop(1, 1, 0.004, 0.005), "both hold")
    assert_true(not eb_should_stop(0, 1, 0.004, 0.005), "not yet stable")
    assert_true(
        not eb_should_stop(1, 1, 0.005, 0.005), "mean entropy is < not <="
    )
    assert_true(not eb_should_stop(1, 1, 0.9, 0.005), "not confident")
    assert_true(eb_should_stop(3, 3, 0.0, 0.005), "held reaches the threshold")
    assert_true(not eb_should_stop(2, 3, 0.0, 0.005), "held short of it")


def test_run_stops_at_step_two_when_stable_and_confident() raises:
    # Every step is one-hot at the same tokens: step 1 has no earlier argmax to
    # hold against, step 2 holds once, so the run takes exactly two steps.
    var vocab = 6
    var step = _step(_one_hot(vocab, 3), _one_hot(vocab, 1), _one_hot(vocab, 5))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [1, 2]
    var obs = Accepted(len(prompt))
    var res = _generate_diffusion_gemma(m, prompt, _cfg(3), obs)
    assert_equal(res.steps_taken(), 2)
    assert_equal(res.forward_passes, 2)
    assert_equal(res.stop_reason, "converged")
    assert_true(res.stopped_early)
    _same(obs.steps, [1, 2], "observer step numbers")
    _same(obs.totals, [48, 48], "observer total is the step budget")
    _same(res.tokens, [1, 2, 3, 1, 5], "the output is prompt + argmax canvas")
    assert_equal(len(m.seen), 2)


def test_the_canvas_must_match_the_models_canvas_length() raises:
    # The model scores [prompt | canvas] with the canvas it declares; a
    # generation length that differs would overflow its logits buffer or move
    # the prompt boundary.
    var vocab = 6
    var step = _step(_one_hot(vocab, 3), _one_hot(vocab, 1), _one_hot(vocab, 5))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [1, 2]
    for bad in List[Int]([2, 4]):
        var obs = Accepted(len(prompt))
        with assert_raises(
            contains=(
                "generation length "
                + String(bad)
                + " must equal the model's canvas length 3"
            )
        ):
            _ = _generate_diffusion_gemma(m, prompt, _cfg(bad), obs)
    assert_equal(len(m.seen), 0)
    var ok = Accepted(len(prompt))
    _ = _generate_diffusion_gemma(m, prompt, _cfg(3), ok)


def test_run_does_not_stop_when_stable_but_not_confident() raises:
    var vocab = 6
    # The same argmax every step (token 1), but a gap of 2 leaves the mean
    # entropy far above 0.005 at every temperature the schedule reaches.
    var step = _step(_two_way(vocab, 2.0), _two_way(vocab, 2.0))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [1]
    var obs = Accepted(len(prompt))
    var res = _generate_diffusion_gemma(m, prompt, _cfg(2, 6), obs)
    assert_equal(res.steps_taken(), 6)
    assert_equal(res.stop_reason, "max_steps")
    assert_true(not res.stopped_early)
    _same(res.tokens, [1, 1, 1], "argmax canvas of the last step")


def test_run_does_not_stop_when_confident_but_not_stable() raises:
    var vocab = 6
    # One-hot every step, but the argmax token alternates between steps.
    var a = _step(_one_hot(vocab, 2), _one_hot(vocab, 3))
    var b = _step(_one_hot(vocab, 4), _one_hot(vocab, 5))
    var m = ScriptedModel(
        vocab, [a.copy(), b.copy(), a.copy(), b.copy(), a.copy()]
    )
    var prompt: List[Int] = [1]
    var obs = Accepted(len(prompt))
    var res = _generate_diffusion_gemma(m, prompt, _cfg(2, 5), obs)
    assert_equal(res.steps_taken(), 5)
    assert_equal(res.stop_reason, "max_steps")
    _same(res.tokens, [1, 2, 3], "step 5 is the a-step")


def test_run_with_stability_three_needs_four_steps() raises:
    var vocab = 6
    var step = _step(_one_hot(vocab, 3), _one_hot(vocab, 1))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [1]
    var c = _cfg(2)
    c.eb_stability = 3
    var obs = Accepted(len(prompt))
    var res = _generate_diffusion_gemma(m, prompt, c, obs)
    assert_equal(res.steps_taken(), 4)
    assert_equal(res.stop_reason, "converged")


def test_run_stop_rule_can_fire_on_the_last_step() raises:
    # Stable and confident only once step 2 of 2 completes: the run took every
    # step it was allowed and still reports why it ended.
    var vocab = 6
    var step = _step(_one_hot(vocab, 3), _one_hot(vocab, 1))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [1]
    var res = _run(m, prompt, _cfg(2, 2))
    assert_equal(res.steps_taken(), 2)
    assert_equal(res.stop_reason, "converged")
    assert_true(not res.stopped_early, "no step was skipped")


def test_run_confidence_threshold_follows_the_temperature_schedule() raises:
    # Two logits 0 and a = 3.4 on a vocabulary of two, argmax 1 every step.
    # With z = a / t the row's entropy is h(sigmoid(z)) in nats (numpy, f64,
    # a read as float32). t = 0.4 + 0.4 * cur / 48: cur = 7 gives t = 0.45833,
    # z = 7.418, h = 0.005050 (still above 0.005); cur = 6 gives t = 0.45,
    # z = 7.556, h = 0.004474 (below). cur = 6 is step index 42, the 43rd step,
    # and the argmax is stable, so the run ends there. A schedule running the
    # other way round starts at t = 0.4 and would stop at step 2.
    var vocab = 2
    var step = _step(_two_way_exact(3.4), _two_way_exact(3.4))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [1]
    var res = _run(m, prompt, _cfg(2))
    assert_equal(res.steps_taken(), 43)
    assert_equal(res.stop_reason, "converged")


# ---------------------------------------------------------------------------
# The loop: acceptance, renoising, the draws and the output
# ---------------------------------------------------------------------------


def test_run_accepts_the_lowest_entropies_within_the_bound() raises:
    # Entropies: positions 0 and 2 about 0, positions 1 and 3 ln 4. Sorted:
    # 0, 2, then the flat pair. The flat position at rank 3 has ~0 before it
    # and passes, the one at rank 4 has ln 4 = 1.386 before it and does not.
    var vocab = 4
    var step = _step(
        _one_hot(vocab, 2), _flat(vocab), _one_hot(vocab, 0), _flat(vocab)
    )
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [9, 9]
    var obs = Accepted(len(prompt))
    _ = _generate_diffusion_gemma(m, prompt, _cfg(4, 1), obs)
    assert_equal(len(obs.sets), 1)
    _same(_sorted(obs.sets[0]), [0, 1, 2], "accepted set")


def test_run_with_every_position_confident_accepts_all() raises:
    var vocab = 5
    var step = _step(_one_hot(vocab, 1), _one_hot(vocab, 2), _one_hot(vocab, 3))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [0]
    var obs = Accepted(len(prompt))
    _ = _generate_diffusion_gemma(m, prompt, _cfg(3, 1), obs)
    _same(_sorted(obs.sets[0]), [0, 1, 2], "all accepted")


def test_run_with_no_confident_position_accepts_the_lowest_only() raises:
    # A flat row (ln 5), a gap-0.5 row, and a gap-1.5 row: none is confident.
    # The sharpest (gap 1.5) has the lowest entropy and is the only one with
    # nothing before it that fits in the bound.
    var vocab = 5
    var step = _step(_flat(vocab), _two_way(vocab, 0.5), _two_way(vocab, 1.5))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [0]
    var obs = Accepted(len(prompt))
    _ = _generate_diffusion_gemma(m, prompt, _cfg(3, 1), obs)
    _same(_sorted(obs.sets[0]), [2], "only the lowest entropy")


def test_run_bound_zero_accepts_only_the_minimum() raises:
    # With every entropy positive, a bound of 0 leaves only the first in the
    # sorted order, where nothing precedes it.
    var vocab = 4
    var step = _step(
        _two_way(vocab, 3.0), _two_way(vocab, 5.0), _two_way(vocab, 4.0)
    )
    var m = ScriptedModel(vocab, [step.copy()])
    var c = _cfg(3, 1)
    c.eb_bound = 0.0
    var prompt: List[Int] = [0]
    var obs = Accepted(len(prompt))
    _ = _generate_diffusion_gemma(m, prompt, c, obs)
    _same(_sorted(obs.sets[0]), [1], "bound 0")
    var wide = _cfg(3, 1)
    wide.eb_bound = 1000.0
    var obs2 = Accepted(len(prompt))
    var m2 = ScriptedModel(vocab, [step.copy()])
    _ = _generate_diffusion_gemma(m2, prompt, wide, obs2)
    _same(_sorted(obs2.sets[0]), [0, 1, 2], "a huge bound accepts all")


def test_run_a_change_resets_the_stability_count() raises:
    # Argmax per step: a, a, b, b, b with stability 2 (the canvas is one
    # position). held runs 0, 1, 0 (changed), 1, 2: it reaches 2 at step 5. A
    # count that is not reset would reach 2 at step 4.
    var vocab = 6
    var a = _step(_one_hot(vocab, 2))
    var b = _step(_one_hot(vocab, 4))
    var m = ScriptedModel(
        vocab, [a.copy(), a.copy(), b.copy(), b.copy(), b.copy()]
    )
    var prompt: List[Int] = [1]
    var c = _cfg(1)
    c.eb_stability = 2
    var res = _run(m, prompt, c)
    assert_equal(res.steps_taken(), 5)
    assert_equal(res.stop_reason, "converged")
    _same(res.tokens, [1, 4], "the output is the last argmax")


def test_run_output_is_the_argmax_canvas_not_the_drawn_one() raises:
    # Eight positions of equal entropy (about 0.27): only the first in the
    # order is accepted, the other seven are renoised with random ids over 50.
    # The output still shows each position's argmax, token 1.
    var vocab = 50
    var rows = List[Float32]()
    for _ in range(8):
        for v in _two_way(vocab, 2.0):
            rows.append(v)
    var m = ScriptedModel(vocab, [rows.copy()])
    var prompt: List[Int] = [7, 7]
    var res = _run(m, prompt, _cfg(8, 1))
    var want: List[Int] = [7, 7, 1, 1, 1, 1, 1, 1, 1, 1]
    _same(res.tokens, want, "argmax canvas")
    # The canvas the next step would see is not that: the renoised ids differ.
    var m2 = ScriptedModel(vocab, [rows.copy()])
    var c = _cfg(8, 2)
    c.eb_stability = 100
    _ = _run(m2, prompt, c)
    var other = 0
    for i in range(2, 10):
        if m2.seen[1][i] != 1:
            other += 1
    assert_true(other >= 4, "renoised positions are random: " + String(other))


def test_run_renoises_and_draws_in_the_references_order() raises:
    # The reference draws the canvas first (one id per position), then at each
    # step, for each position, the uniform u and then the renoise id. Rows are
    # one-hot, flat, one-hot, flat: ranks by entropy are 0, 2, 1, 3, so
    # positions 0, 2 and 1 are accepted and position 3 is renoised.
    var vocab = 64
    var step = _step(
        _one_hot(vocab, 2), _flat(vocab), _one_hot(vocab, 0), _flat(vocab)
    )
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [9, 8]
    var c = _cfg(4, 2)
    c.eb_stability = 100  # never stop: look at the second step's input
    _ = _run(m, prompt, c)
    var rng = Rng(c.seed)
    var canvas0 = eb_initial_canvas(rng, 4, vocab)
    _same(
        _slice(m.seen[0], 2, len(m.seen[0])),
        canvas0,
        "step 1 sees the random canvas",
    )
    _same(_slice(m.seen[0], 0, 2), prompt, "the prompt is passed through")
    var u = List[Float64]()
    var renoise = List[Int]()
    for _ in range(4):
        u.append(rng.uniform())
        renoise.append(rng.below(vocab))
    # The flat position 1 draws by inverse CDF over equal masses: the first
    # index v with (v + 1) / vocab >= u, that is ceil(vocab * u) - 1.
    var flat_pick = Int(ceil(u[1] * Float64(vocab))) - 1
    var second = _slice(m.seen[1], 2, len(m.seen[1]))
    assert_equal(second[0], 2, "accepted one-hot position takes its token")
    assert_equal(second[1], flat_pick, "accepted flat position takes its draw")
    assert_equal(second[2], 0, "accepted one-hot position takes its token")
    assert_equal(second[3], renoise[3], "rejected position takes the renoise")


def test_run_uses_the_given_t_min_t_max_and_confidence() raises:
    # Logits [0, 3.4] as in the schedule test; numpy (f64) puts the first step
    # from step 2 on whose mean entropy is under the threshold at:
    #   defaults (0.4, 0.8, 0.005)  step 43
    #   (0.3, 0.6, 0.02)           step 4
    #   t_min back at 0.4           step 6
    #   t_max back at 0.8           step 22
    #   confidence back at 0.005    step 24
    # so each of the three values moves the stop step on its own.
    var step = _step(_two_way_exact(3.4), _two_way_exact(3.4))
    var m = ScriptedModel(2, [step.copy()])
    var prompt: List[Int] = [1]
    var c = _cfg(2)
    c.eb_t_min = 0.3
    c.eb_t_max = 0.6
    c.eb_confidence = 0.02
    var res = _run(m, prompt, c)
    assert_equal(res.steps_taken(), 4)
    assert_equal(res.stop_reason, "converged")


def test_run_with_stability_zero_stops_at_step_one() raises:
    var vocab = 6
    var step = _step(_one_hot(vocab, 3), _one_hot(vocab, 1))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [1]
    var c = _cfg(2)
    c.eb_stability = 0
    var res = _run(m, prompt, c)
    assert_equal(res.steps_taken(), 1)
    assert_equal(res.stop_reason, "converged")
    assert_true(res.stopped_early)


def test_run_asks_the_model_for_the_canvas_rows_only() raises:
    var vocab = 4
    var step = _step(_one_hot(vocab, 2), _one_hot(vocab, 1))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [9, 8, 7]
    _ = _run(m, prompt, _cfg(2, 1))
    _same(m.asked[0], [3, 4], "canvas rows")
    assert_equal(len(m.seen[0]), 5)


# ---------------------------------------------------------------------------
# The seed and the initial canvas
# ---------------------------------------------------------------------------


def test_initial_canvas_covers_the_whole_vocabulary() raises:
    var rng = Rng(1)
    var canvas = eb_initial_canvas(rng, 256, 262144)
    assert_equal(len(canvas), 256)
    var top = 0
    var above_half = 0
    for t in canvas:
        assert_true(t >= 0 and t < 262144, "id " + String(t))
        top = max(top, t)
        if t >= 131072:
            above_half += 1
    # Special tokens and the upper half of the range are drawn too: 256 uniform
    # draws all landing below 131072 has probability 2^-256.
    assert_true(above_half > 64 and above_half < 192, String(above_half))
    assert_true(top > 250000, "largest id " + String(top))


def test_same_seed_gives_the_same_run_and_another_seed_another_canvas() raises:
    var vocab = 64
    var rows = List[Float32]()
    for p in range(8):
        for v in _two_way_big(vocab, p):
            rows.append(v)
    var prompt: List[Int] = [1, 2, 3]
    var c = _cfg(8, 4)
    c.eb_stability = 100
    var a = ScriptedModel(vocab, [rows.copy()])
    var b = ScriptedModel(vocab, [rows.copy()])
    var ra = _run(a, prompt, c)
    var rb = _run(b, prompt, c)
    _same(ra.tokens, rb.tokens, "same seed, same tokens")
    assert_equal(len(a.seen), len(b.seen))
    for k in range(len(a.seen)):
        _same(
            a.seen[k], b.seen[k], "same seed, same canvas at call " + String(k)
        )
    var other = c.copy()
    other.seed = c.seed + 1
    var d = ScriptedModel(vocab, [rows.copy()])
    _ = _run(d, prompt, other)
    var differ = 0
    for i in range(3, 11):
        if a.seen[0][i] != d.seen[0][i]:
            differ += 1
    assert_true(
        differ >= 5, "another seed draws another canvas: " + String(differ)
    )


def _two_way_big(vocab: Int, p: Int) -> List[Float32]:
    var r = List[Float32](length=vocab, fill=Float32(0))
    r[p] = Float32(1.0 + Float32(p) * 0.25)
    return r^


def test_thread_count_does_not_change_the_run() raises:
    var vocab = 64
    var rows = List[Float32]()
    for p in range(8):
        for v in _two_way_big(vocab, p):
            rows.append(v)
    var prompt: List[Int] = [1, 2, 3]
    var one = _cfg(8, 5)
    one.eb_stability = 100
    var four = one.copy()
    four.threads = 4
    var a = ScriptedModel(vocab, [rows.copy()])
    var b = ScriptedModel(vocab, [rows.copy()])
    var ra = _run(a, prompt, one)
    var rb = _run(b, prompt, four)
    _same(ra.tokens, rb.tokens, "threads 1 against 4")
    for k in range(len(a.seen)):
        _same(a.seen[k], b.seen[k], "canvas at call " + String(k))


# ---------------------------------------------------------------------------
# Arguments and dispatch
# ---------------------------------------------------------------------------


def test_bad_tunables_are_refused_by_name() raises:
    var vocab = 4
    var m = ScriptedModel(vocab, [_step(_one_hot(vocab, 1))])
    var prompt: List[Int] = [1]
    var c = _cfg(1)
    c.eb_steps = 0
    with assert_raises(contains="--eb-steps"):
        _ = _run(m, prompt, c)
    c = _cfg(1)
    c.eb_t_min = -0.1
    with assert_raises(contains="--eb-t-min"):
        _ = _run(m, prompt, c)
    c = _cfg(1)
    c.eb_t_max = -1.0
    with assert_raises(contains="--eb-t-max"):
        _ = _run(m, prompt, c)
    c = _cfg(1)
    c.eb_t_max = 0.0  # the CLI refuses it first; a library caller gets this
    with assert_raises(contains="--eb-t-max"):
        _ = _run(m, prompt, c)
    # Zero is valid for t_min and for stability (the reference allows both).
    c = _cfg(1)
    c.eb_t_min = 0.0
    assert_equal(_run(m, prompt, c).steps_taken(), 2)
    c = _cfg(1)
    c.eb_stability = 0
    assert_equal(_run(m, prompt, c).steps_taken(), 1)
    c = _cfg(1)
    c.eb_bound = -0.1
    with assert_raises(contains="--eb-bound"):
        _ = _run(m, prompt, c)
    c = _cfg(1)
    c.eb_stability = -1
    with assert_raises(contains="--eb-stability"):
        _ = _run(m, prompt, c)
    c = _cfg(1)
    c.eb_confidence = -0.1
    with assert_raises(contains="--eb-confidence"):
        _ = _run(m, prompt, c)
    c = _cfg(0)
    with assert_raises(contains="generation length"):
        _ = _run(m, prompt, c)
    var empty = List[Int]()
    with assert_raises(contains="prompt"):
        _ = _run(m, empty, _cfg(1))


def test_defaults_are_the_branchs() raises:
    var c = GenConfig()
    assert_equal(c.eb_steps, 48)
    assert_equal(c.eb_t_min, 0.4)
    assert_equal(c.eb_t_max, 0.8)
    assert_equal(c.eb_bound, 0.1)
    assert_equal(c.eb_stability, 1)
    assert_equal(c.eb_confidence, 0.005)


def test_generate_dispatches_on_the_architecture() raises:
    var vocab = 4
    var step = _step(_one_hot(vocab, 2))
    var m = ScriptedModel(vocab, [step.copy()])
    var prompt: List[Int] = [1]
    var res = generate(m, prompt, _cfg(1))
    assert_equal(res.stop_reason, "converged")
    assert_equal(res.steps_taken(), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
