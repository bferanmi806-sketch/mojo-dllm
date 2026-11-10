from std.math import exp, log
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.diffusion.sampler import (
    GenConfig,
    num_transfer_tokens,
    select_top,
    argmax_and_prob,
    generate,
    Rng,
    _confidence,
    _confidence_from_stats,
    _greedy_rows,
    _gumbel_argmax,
    _prob_of,
    StepObserver,
    generate_observed,
)
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.models.transformer import (
    DenoisingModel,
    DiffusionLM,
    ModelConfig,
    RowStats,
    Timings,
)
from mojo_dllm.sys.mem import F32Ptr, Floats

comptime FIX = "tests/fixtures/"


def test_transfer_schedule_matches_generate_py() raises:
    # LLaDA: base = m // steps, the first (m % steps) steps take one more.
    var s = num_transfer_tokens(32, 8)
    assert_equal(len(s), 8)
    for v in s:
        assert_equal(v, 4)
    var r = num_transfer_tokens(10, 4)
    assert_equal(r[0], 3)
    assert_equal(r[1], 3)
    assert_equal(r[2], 2)
    assert_equal(r[3], 2)
    var total = 0
    for v in num_transfer_tokens(31, 7):
        total += v
    assert_equal(total, 31)


def test_select_top_is_stable_on_ties() raises:
    var conf: List[Float64] = [0.5, 0.9, 0.5, 0.9, 0.1]
    var sel = select_top(conf, 3)
    assert_equal(len(sel), 3)
    assert_equal(sel[0], 1)
    assert_equal(sel[1], 3)
    assert_equal(sel[2], 0)


def test_select_top_caps_at_candidates() raises:
    var conf: List[Float64] = [0.2, 0.3]
    assert_equal(len(select_top(conf, 5)), 2)


def test_argmax_and_prob_in_f64() raises:
    var row = Floats(4)
    row[0] = 1.0
    row[1] = 3.0
    row[2] = 2.0
    row[3] = 3.0
    var r = argmax_and_prob(row.ptr(), 4)
    assert_equal(r[0], 1)  # first maximum wins
    var z = exp(Float64(-2.0)) + 1.0 + exp(Float64(-1.0)) + 1.0
    assert_true(abs(r[1] - 1.0 / z) < 1e-12, "prob " + String(r[1]))


def test_rng_below_refuses_a_bound_that_is_not_positive() raises:
    var r = Rng(5)
    assert_equal(r.below(1), 0)
    for _ in range(50):
        var v = r.below(7)
        assert_true(v >= 0 and v < 7)
    with assert_raises(contains="Rng.below needs a positive bound, got 0"):
        _ = r.below(0)
    with assert_raises(contains="Rng.below needs a positive bound, got -3"):
        _ = r.below(-3)


def test_rng_is_deterministic() raises:
    var a = Rng(42)
    var b = Rng(42)
    for _ in range(10):
        assert_equal(a.next_u64(), b.next_u64())
    var r7 = Rng(7)
    var u = r7.uniform()
    assert_true(u > 0.0 and u < 1.0, "uniform in (0,1)")


def _tiny_cfg() -> GenConfig:
    var c = GenConfig()
    c.gen_length = 8
    c.block_length = 4
    c.steps = 4
    return c^


def _tiny_dream_cfg() -> GenConfig:
    # Same shape as `_tiny_cfg`; the mask id comes from the model.
    var c = GenConfig()
    c.gen_length = 8
    c.block_length = 4
    c.steps = 4
    return c^


def test_generation_resolves_every_mask() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var out = generate(m, prompt, _tiny_cfg())
    assert_equal(len(out.tokens), 11)
    for i in range(3):
        assert_equal(out.tokens[i], prompt[i])
    for i in range(3, 11):
        assert_true(out.tokens[i] != 511, "mask left at position " + String(i))
    assert_equal(out.forward_passes, 4)


def test_generation_is_deterministic_with_temperature() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = _tiny_cfg()
    c.temperature = 0.7
    c.seed = 1234
    var a = generate(m, prompt, c)
    var b = generate(m, prompt, c)
    for i in range(len(a.tokens)):
        assert_equal(a.tokens[i], b.tokens[i])


def test_full_logits_ablation_gives_same_tokens() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var a = generate(m, prompt, _tiny_cfg())
    var c = _tiny_cfg()
    c.full_logits = True
    var b = generate(m, prompt, c)
    for i in range(len(a.tokens)):
        assert_equal(a.tokens[i], b.tokens[i])


def test_sampler_threads_do_not_change_tokens() raises:
    # The per-row confidence pass runs rows in parallel. Each row's sums
    # keep their order, so the thread count must not move a single token.
    for path in ["tiny-llada.gguf", "tiny-dream.gguf"]:
        var m = DiffusionLM(FIX + path, threads=2, max_tokens=16)
        var prompt: List[Int] = [5, 17, 300]
        var one = _tiny_cfg()
        one.threads = 1
        var four = _tiny_cfg()
        four.threads = 4
        var a = generate(m, prompt, one)
        var b = generate(m, prompt, four)
        assert_equal(len(a.tokens), len(b.tokens))
        for i in range(len(a.tokens)):
            assert_equal(
                a.tokens[i], b.tokens[i], path + " position " + String(i)
            )


def test_greedy_rows_match_the_sequential_scorers() raises:
    var vocab = 700
    var n = 9
    var logits = Floats(n * vocab)
    var rng = Rng(3)
    for i in range(n * vocab):
        logits[i] = Float32(rng.uniform() * 12.0 - 6.0)
    # Score rows out of order, as full-logits mode does.
    var row_of: List[Int] = [8, 0, 3, 5, 1, 7, 2, 6, 4]
    for algorithm in ["llada", "entropy", "maskgit_plus", "topk_margin"]:
        var got = _greedy_rows(logits.ptr(), row_of, vocab, algorithm, 4)
        for j in range(n):
            var row = logits.ptr().unsafe_offset(row_of[j] * vocab)
            var want: Tuple[Int, Float64]
            if algorithm == "llada":
                want = argmax_and_prob(row, vocab)
            else:
                want = _confidence(row, vocab, algorithm)
            assert_equal(
                got[0][j], want[0], algorithm + " token, row " + String(j)
            )
            assert_equal(
                got[1][j], want[1], algorithm + " confidence, row " + String(j)
            )


def _visible(tokens: List[Int], P: Int, stop: Int) -> List[Int]:
    """The generated tokens a reader sees: up to the first stop token."""
    var out = List[Int]()
    for i in range(P, len(tokens)):
        if tokens[i] == stop:
            break
        out.append(tokens[i])
    return out^


def _check_early_stop(path: String, mask: Int, block: Int) raises:
    var m = DiffusionLM(FIX + path, threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var P = len(prompt)
    var c = GenConfig()
    c.gen_length = 16
    c.block_length = block
    c.steps = 16
    c.early_stop = False
    var full = generate(m, prompt, c)
    var saved_any = False
    # Each token of the full answer in turn plays the end-of-text token.
    for i in range(P, len(full.tokens)):
        var stop = full.tokens[i]
        var e = c.copy()
        e.early_stop = True
        e.stop_ids = [stop]
        var early = generate(m, prompt, e)
        var want = _visible(full.tokens, P, stop)
        var got = _visible(early.tokens, P, stop)
        assert_equal(
            len(got), len(want), path + " visible length, stop " + String(stop)
        )
        for j in range(len(want)):
            assert_equal(got[j], want[j], path + " visible token " + String(j))
        assert_true(
            early.forward_passes <= full.forward_passes,
            "early stop ran more passes",
        )
        if early.forward_passes < full.forward_passes:
            saved_any = True
    assert_true(
        saved_any,
        path + ": no stop token ever saved a pass, so the check is vacuous",
    )


def test_early_stop_keeps_the_visible_answer_llada() raises:
    _check_early_stop("tiny-llada.gguf", 511, 4)


def test_early_stop_keeps_the_visible_answer_dream() raises:
    _check_early_stop("tiny-dream.gguf", 510, 16)


def _check_stop_report(path: String, mask: Int, block: Int) raises:
    # `--json` reports steps_taken and stop_reason for every model: the passes
    # run, and why the loop ended ("answer_fixed" exactly when it stopped early).
    var m = DiffusionLM(FIX + path, threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var c = GenConfig()
    c.gen_length = 16
    c.block_length = block
    c.steps = 16
    c.early_stop = False
    var full = generate(m, prompt, c)
    assert_equal(full.steps_taken(), full.forward_passes)
    assert_equal(full.stop_reason, "complete", path)
    assert_true(not full.stopped_early)
    var fixed = 0
    for i in range(len(prompt), len(full.tokens)):
        var e = c.copy()
        e.early_stop = True
        e.stop_ids = [full.tokens[i]]
        var early = generate(m, prompt, e)
        assert_equal(early.steps_taken(), early.forward_passes)
        if early.stopped_early:
            fixed += 1
            assert_equal(early.stop_reason, "answer_fixed", path)
            break
        assert_equal(early.stop_reason, "complete", path)
    assert_true(
        fixed > 0, path + ": no run stopped early, so the check is vacuous"
    )


def test_llada_json_members_report_steps_and_reason() raises:
    _check_stop_report("tiny-llada.gguf", 511, 4)


def test_dream_json_members_report_steps_and_reason() raises:
    _check_stop_report("tiny-dream.gguf", 510, 16)


def _threshold_cfg(threshold: Float64) -> GenConfig:
    var c = GenConfig()
    c.gen_length = 16
    c.block_length = 8
    c.steps = 16
    c.early_stop = False
    c.threshold = threshold
    return c^


def test_threshold_off_or_unreachable_changes_nothing() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var base = generate(m, prompt, _threshold_cfg(0.0))
    var never = generate(m, prompt, _threshold_cfg(1.5))
    assert_equal(base.forward_passes, 16)
    assert_equal(never.forward_passes, base.forward_passes)
    for i in range(len(base.tokens)):
        assert_equal(never.tokens[i], base.tokens[i], "position " + String(i))


def test_threshold_zero_fills_each_block_in_one_step() raises:
    # Every candidate clears a threshold of 0, so each block is one pass.
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var all_in = generate(m, prompt, _threshold_cfg(1e-300))
    assert_equal(all_in.forward_passes, 2)
    for i in range(3, len(all_in.tokens)):
        assert_true(all_in.tokens[i] != 511, "mask left at " + String(i))


def test_threshold_never_takes_more_passes() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var base = generate(m, prompt, _threshold_cfg(0.0))
    for tau in [0.9, 0.5, 0.1, 0.01]:
        var r = generate(m, prompt, _threshold_cfg(tau))
        assert_true(
            r.forward_passes <= base.forward_passes, "threshold " + String(tau)
        )
        for i in range(3, len(r.tokens)):
            assert_true(r.tokens[i] != 511, "mask left at " + String(i))


def test_dual_cache_with_one_step_per_block_is_exact() raises:
    # One step per block makes every pass a refresh, so the cached run must
    # match the uncached one token for token.
    var prompt: List[Int] = [5, 17, 300]
    var c = _threshold_cfg(0.0)
    c.steps = 2
    var plain = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=24)
    var a = generate(plain, prompt, c)
    var cached = DiffusionLM(
        FIX + "tiny-llada.gguf", threads=2, max_tokens=24, kv_cache=True
    )
    c.dual_cache = True
    var b = generate(cached, prompt, c)
    assert_equal(a.forward_passes, b.forward_passes)
    for i in range(len(a.tokens)):
        assert_equal(a.tokens[i], b.tokens[i], "position " + String(i))


def test_dual_cache_resolves_every_mask() raises:
    var prompt: List[Int] = [5, 17, 300]
    var c = _threshold_cfg(0.0)
    c.dual_cache = True
    var m = DiffusionLM(
        FIX + "tiny-llada.gguf", threads=2, max_tokens=24, kv_cache=True
    )
    var r = generate(m, prompt, c)
    assert_equal(r.forward_passes, 16)
    for i in range(3, len(r.tokens)):
        assert_true(r.tokens[i] != 511, "mask left at " + String(i))


def test_rejects_bad_block_geometry() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_cfg()
    c.block_length = 3
    with assert_raises(contains="multiple of the block length"):
        _ = generate(m, prompt, c)
    c = _tiny_cfg()
    c.steps = 3
    with assert_raises(contains="multiple of the number of blocks"):
        _ = generate(m, prompt, c)


def test_generate_refuses_a_temperature_that_is_not_a_number_or_negative() raises:
    # `nan > 0` is false, so a NaN temperature read as greedy decoding.
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_cfg()
    c.temperature = Float64("nan")
    with assert_raises(contains="--temperature must be a finite number, got"):
        _ = generate(m, prompt, c)
    c = _tiny_cfg()
    c.temperature = -0.5
    with assert_raises(contains="--temperature must be at least 0, got -0.5"):
        _ = generate(m, prompt, c)


# The eleven refusals below each pin one guard in the sampler. Every config
# starts from a tiny cfg that generates cleanly, then flips the one field the
# guard reads, so the assert fails if that guard is removed. #144.


def test_llada_rejects_nonpositive_geometry() raises:
    # sampler.mojo `_generate_llada`: gen_length/block_length/steps must be
    # positive. The block_length case also guards the `gen_length %
    # block_length` modulo just below it from dividing by zero.
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_cfg()
    c.gen_length = 0
    with assert_raises(contains="block length and steps must be positive"):
        _ = generate(m, prompt, c)
    c = _tiny_cfg()
    c.block_length = 0
    with assert_raises(contains="block length and steps must be positive"):
        _ = generate(m, prompt, c)
    c = _tiny_cfg()
    c.steps = 0
    with assert_raises(contains="block length and steps must be positive"):
        _ = generate(m, prompt, c)


def test_llada_rejects_dual_cache_with_full_logits() raises:
    # The K/V cache recomputes only one block, so it cannot report every row's
    # logits; asking for both is refused rather than silently truncated.
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_cfg()
    c.dual_cache = True
    c.full_logits = True
    with assert_raises(contains="cannot give every row's logits"):
        _ = generate(m, prompt, c)


def test_llada_rejects_negative_threshold() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_cfg()
    c.threshold = -0.5
    with assert_raises(contains="threshold must not be negative"):
        _ = generate(m, prompt, c)


def test_llada_rejects_threshold_with_random_remasking() raises:
    # A confidence threshold ranks candidates by confidence; random remasking
    # has no confidence to rank, so the combination is meaningless.
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_cfg()
    c.threshold = 0.5
    c.remasking = "random"
    with assert_raises(contains="needs low_confidence remasking"):
        _ = generate(m, prompt, c)


def test_llada_rejects_unknown_remasking() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_cfg()
    c.remasking = "bogus"
    with assert_raises(contains="unknown remasking strategy"):
        _ = generate(m, prompt, c)


def test_dream_rejects_nonpositive_geometry() raises:
    # sampler.mojo `_generate_dream`: gen_length and steps must be positive.
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_dream_cfg()
    c.gen_length = 0
    with assert_raises(contains="generation length and steps must be positive"):
        _ = generate(m, prompt, c)
    c = _tiny_dream_cfg()
    c.steps = 0
    with assert_raises(contains="generation length and steps must be positive"):
        _ = generate(m, prompt, c)


def test_dream_rejects_unknown_algorithm() raises:
    # Dream maps low_confidence -> entropy, then accepts only its three
    # algorithms; anything else is refused by name.
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_dream_cfg()
    c.remasking = "bogus"
    with assert_raises(contains="unknown Dream algorithm"):
        _ = generate(m, prompt, c)


def test_dream_rejects_positive_temperature() raises:
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_dream_cfg()
    c.temperature = 0.7
    with assert_raises(contains="temperature > 0 is not implemented"):
        _ = generate(m, prompt, c)


def test_dream_rejects_threshold() raises:
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_dream_cfg()
    c.threshold = 0.5
    with assert_raises(
        contains="confidence threshold is implemented for LLaDA only"
    ):
        _ = generate(m, prompt, c)


def test_dream_rejects_dual_cache() raises:
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_dream_cfg()
    c.dual_cache = True
    with assert_raises(contains="K/V cache is implemented for LLaDA only"):
        _ = generate(m, prompt, c)


def test_dream_rejects_empty_prompt() raises:
    # The sharpest of the eleven: Dream scores a position from the row before
    # it, so an empty prompt would index row -1 without this guard.
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = []
    with assert_raises(contains="at least one prompt token"):
        _ = generate(m, prompt, _tiny_dream_cfg())


def test_dream_confidences() raises:
    var row = Floats(3)
    row[0] = 0.0
    row[1] = 0.0
    row[2] = 0.0
    var e = _confidence(row.ptr(), 3, "entropy")
    assert_equal(e[0], 0)
    # uniform over 3: sum p log p = log(1/3)
    assert_true(
        abs(e[1] - log(Float64(1.0) / 3.0)) < 1e-9,
        "neg entropy " + String(e[1]),
    )
    row[2] = 2.0
    var m = _confidence(row.ptr(), 3, "maskgit_plus")
    assert_equal(m[0], 2)
    var z = 2.0 + exp(Float64(2.0))
    assert_true(abs(m[1] - exp(Float64(2.0)) / z) < 1e-12, "p(x0)")
    var k = _confidence(row.ptr(), 3, "topk_margin")
    assert_true(abs(k[1] - (exp(Float64(2.0)) - 1.0) / z) < 1e-12, "margin")


def test_dream_generation_resolves_every_mask() raises:
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = GenConfig()
    c.gen_length = 8
    c.steps = 4
    var out = generate(m, prompt, c)
    assert_equal(len(out.tokens), 11)
    for i in range(3, 11):
        assert_true(out.tokens[i] != 510, "mask left at position " + String(i))
    assert_true(out.forward_passes <= 4, "at most one pass per step")


struct Recorder(StepObserver):
    var steps: Int
    var committed: Int
    var last_total: Int

    def __init__(out self):
        self.steps = 0
        self.committed = 0
        self.last_total = 0

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        self.steps += 1
        self.committed += len(committed)
        self.last_total = total


def test_observer_sees_every_step_and_commit() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var rec = Recorder()
    var out = generate_observed(m, prompt, _tiny_cfg(), rec)
    assert_equal(rec.steps, out.forward_passes)
    assert_equal(rec.committed, 8)
    assert_equal(rec.last_total, 4)


def test_observer_sees_dream_commits() raises:
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = GenConfig()
    c.gen_length = 8
    c.steps = 4
    var rec = Recorder()
    _ = generate_observed(m, prompt, c, rec)
    assert_equal(rec.committed, 8)


def test_confidence_from_row_statistics_matches_the_full_row() raises:
    # The device kernel hands the sampler four numbers per row; the sampler
    # must score the row from them exactly as the loops over the whole row do.
    var vocab = 300
    var row = Floats(vocab)
    var s = UInt64(12345)
    for v in range(vocab):
        s = s * 6364136223846793005 + 1442695040888963407
        row[v] = Float32(Int((s >> 33) % 2001) - 1000) / 125.0
    row[40] = 9.0
    row[200] = 9.0  # the first of two equal maxima wins
    var best = 0
    for v in range(1, vocab):
        if row[v] > row[best]:
            best = v
    var st = RowStats()
    var m = Float64(row[best])
    var z = Float64(0)
    var a = Float64(0)
    var second = Float64(-1e300)
    for v in range(vocab):
        var e = exp(Float64(row[v]) - m)
        z += e
        a += e * (Float64(row[v]) - m)
        if v != best and Float64(row[v]) > second:
            second = Float64(row[v])
    st.tok.append(best)
    st.mx.append(m)
    st.z.append(z)
    st.second.append(second)
    st.weighted.append(a)
    assert_equal(best, 40)
    var p = argmax_and_prob(row.ptr(), vocab)
    assert_true(abs(_confidence_from_stats(st, 0, "llada") - p[1]) < 1e-12)
    for algo in ["maskgit_plus", "topk_margin", "entropy"]:
        var w = _confidence(row.ptr(), vocab, algo)
        var got = _confidence_from_stats(st, 0, algo)
        assert_true(
            abs(got - w[1]) < 1e-9,
            algo + ": " + String(got) + " against " + String(w[1]),
        )


def test_rng_matches_the_xoshiro256_starstar_reference() raises:
    # Known answers from an independent Python implementation of splitmix64
    # (the seed expansion) and xoshiro256** (Blackman and Vigna's definition).
    var a = Rng(0)
    assert_equal(a.s0, UInt64(0xE220A8397B1DCDAF))
    assert_equal(a.s1, UInt64(0x6E789E6AA1B965F4))
    assert_equal(a.s2, UInt64(0x06C45D188009454F))
    assert_equal(a.s3, UInt64(0xF88BB8A8724C81EC))
    assert_equal(a.next_u64(), UInt64(0x99EC5F36CB75F2B4))
    assert_equal(a.next_u64(), UInt64(0xBF6E1F784956452A))
    assert_equal(a.next_u64(), UInt64(0x1A5F849D4933E6E0))
    var b = Rng(42)
    assert_equal(b.s0, UInt64(0xBDD732262FEB6E95))
    assert_equal(b.s1, UInt64(0x28EFE333B266F103))
    assert_equal(b.s2, UInt64(0x47526757130F9F52))
    assert_equal(b.s3, UInt64(0x581CE1FF0E4AE394))
    assert_equal(b.next_u64(), UInt64(0x15780B2E0C2EC716))
    assert_equal(b.next_u64(), UInt64(0x6104D9866D113A7E))
    assert_equal(b.next_u64(), UInt64(0xAE17533239E499A1))
    assert_equal(b.next_u64(), UInt64(0xECB8AD4703B360A1))


def test_rng_uniform_and_below_scale_the_53_high_bits() raises:
    var r = Rng(0)
    assert_equal(r.uniform(), 0.6012629994179048)
    assert_equal(r.uniform(), 0.7477740925472398)
    assert_equal(r.uniform(), 0.10301998939503637)
    var s = Rng(0)
    var want: List[Int] = [4, 5, 0, 2, 5, 6]
    for w in want:
        assert_equal(s.below(7), w)
    var t = Rng(0)
    var want_k: List[Int] = [601, 747, 103, 416, 732, 999]
    for w in want_k:
        assert_equal(t.below(1000), w)


def _gumbel_frequency(
    logits: Floats, temperature: Float64, draws: Int, seed: Int, index: Int
) -> Float64:
    var rng = Rng(seed)
    var hits = 0
    for _ in range(draws):
        if _gumbel_argmax(logits.ptr(), logits.n, temperature, rng) == index:
            hits += 1
    return Float64(hits) / Float64(draws)


def test_gumbel_draw_follows_softmax_of_logits_over_temperature() raises:
    # With logits [0, T ln 3] the draw picks index 1 with probability
    # exp(ln 3) / (1 + exp(ln 3)) = 3/4, whatever T is. A noise scale other
    # than T moves the frequency out of the band (no scale: 0.683).
    var t = Float64(0.7)
    var logits = Floats(2)
    logits[0] = 0.0
    logits[1] = Float32(t * log(3.0))
    var f = _gumbel_frequency(logits, t, 20000, 1, 1)
    assert_true(f >= 0.74 and f <= 0.76, "frequency " + String(f))
    # Same logits at T = 2: 3^(0.35) / (1 + 3^(0.35)) = 0.595.
    var f2 = _gumbel_frequency(logits, 2.0, 20000, 1, 1)
    assert_true(
        f2 >= 0.575 and f2 <= 0.615, "frequency at T = 2: " + String(f2)
    )
    # The draw is a function of the seed only.
    assert_equal(f, _gumbel_frequency(logits, t, 20000, 1, 1))


def test_gumbel_draw_with_three_candidates_follows_the_softmax() raises:
    # Two candidates cannot tell the noise from its mirror image (the
    # difference of two Gumbel variables is symmetric); three can. Logits
    # T ln 1, T ln 2, T ln 3 give 1/6, 1/3 and 1/2; a draw that subtracted
    # the noise instead of adding it gives about 0.13, 0.34 and 0.53.
    var t = Float64(0.7)
    var logits = Floats(3)
    for i in range(3):
        logits[i] = Float32(t * log(Float64(i + 1)))
    for i in range(3):
        var want = Float64(i + 1) / 6.0
        var f = _gumbel_frequency(logits, t, 20000, 3, i)
        assert_true(
            abs(f - want) < 0.015,
            "index " + String(i) + ": frequency " + String(f),
        )


def test_gumbel_draw_at_a_vanishing_temperature_is_the_argmax() raises:
    var logits = Floats(5)
    logits[0] = 0.3
    logits[1] = -1.0
    logits[2] = 2.5
    logits[3] = 2.4
    logits[4] = 0.0
    var rng = Rng(7)
    for _ in range(200):
        assert_equal(_gumbel_argmax(logits.ptr(), 5, 1e-6, rng), 2)


# ---- which positions the LLaDA and Dream samplers commit (#213) ---------------
#
# `tests/fixtures/golden-generation.tsv` holds what this implementation
# produced. A sampler that commits the wrong positions, or takes a candidate
# index for a position, can still match it. Here a scripted model gives every
# candidate a row of known confidence, so the committed positions are worked
# out by hand.

comptime SCRIPT_VOCAB = 8
comptime SCRIPT_MASK = 4


struct _ScriptedModel(DenoisingModel, Movable):
    """Answers forward call k with script[k]: rows in the order of `out_rows`,
    [row][vocab]. Records the rows each call asked for."""

    var cfg: ModelConfig
    var script: List[List[Float32]]
    var asked: List[List[Int]]

    def __init__(
        out self, arch: String, var script: List[List[Float32]]
    ) raises:
        self.cfg = ModelConfig(GGUFFile(FIX + "tiny-llada.gguf"))
        self.cfg.arch = arch
        self.cfg.vocab = SCRIPT_VOCAB
        self.cfg.mask_id = SCRIPT_MASK
        self.cfg.shift_logits = arch == "dream"
        self.script = script^
        self.asked = List[List[Int]]()

    def config(self) -> ModelConfig:
        return self.cfg

    def logit_row(self, position: Int) -> Int:
        return position - 1 if self.cfg.shift_logits else position

    def forward(
        mut self, tokens: List[Int], out_rows: List[Int], logits: F32Ptr
    ) raises:
        var call = len(self.asked)
        self.asked.append(out_rows.copy())
        assert_true(call < len(self.script), "more forward calls than scripted")
        var rows = self.script[call].copy()
        assert_equal(
            len(rows), len(out_rows) * SCRIPT_VOCAB, "script rows of call"
        )
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


struct _Steps(Movable, StepObserver):
    var committed: List[List[Int]]
    var canvas: List[List[Int]]

    def __init__(out self):
        self.committed = List[List[Int]]()
        self.canvas = List[List[Int]]()

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        self.committed.append(committed.copy())
        self.canvas.append(tokens.copy())


def _scripted_rows(tops: List[Int], margins: List[Int]) -> List[Float32]:
    """One row per candidate: zero everywhere but `margin` at its top token,
    so the argmax is `top` and a larger margin is a higher confidence under
    every confidence measure the samplers use."""
    var out = List[Float32]()
    for j in range(len(tops)):
        for v in range(SCRIPT_VOCAB):
            out.append(Float32(margins[j]) if v == tops[j] else Float32(0))
    return out^


def _assert_list(got: List[Int], want: List[Int], what: String) raises:
    assert_equal(len(got), len(want), what + ": length")
    for i in range(len(want)):
        assert_equal(got[i], want[i], what + " [" + String(i) + "]")


def test_llada_commits_the_most_confident_positions() raises:
    # Prompt [1, 2], four masked slots at positions 2..5, two steps. Step 1
    # candidates are 2, 3, 4, 5 with margins 1, 4, 2, 3: the two most confident
    # are candidates 1 and 3, that is positions 3 and 5 (not 1 and 3, the
    # candidate indices). Step 2 has candidates 2 and 4 with margins 2 and 5.
    var script = List[List[Float32]]()
    script.append(_scripted_rows([5, 6, 7, 5], [1, 4, 2, 3]))
    script.append(_scripted_rows([7, 6], [2, 5]))
    var m = _ScriptedModel("llada", script^)
    var c = GenConfig()
    c.gen_length = 4
    c.block_length = 4
    c.steps = 2
    c.early_stop = False
    var obs = _Steps()
    var prompt: List[Int] = [1, 2]
    var r = generate_observed(m, prompt, c, obs)
    assert_equal(len(obs.committed), 2)
    _assert_list(obs.committed[0], [3, 5], "step 1 committed")
    _assert_list(obs.committed[1], [4, 2], "step 2 committed")
    _assert_list(obs.canvas[0], [1, 2, 4, 6, 4, 5], "canvas after step 1")
    _assert_list(r.tokens, [1, 2, 7, 6, 6, 5], "final canvas")
    _assert_list(m.asked[0], [2, 3, 4, 5], "rows of step 1")
    _assert_list(m.asked[1], [2, 4], "rows of step 2")
    assert_equal(r.forward_passes, 2)


def test_llada_commits_per_block_and_keeps_earlier_blocks() raises:
    # Two blocks of two slots, two steps: one commit per step inside a block.
    var script = List[List[Float32]]()
    script.append(_scripted_rows([5, 6], [3, 1]))
    script.append(_scripted_rows([7], [1]))
    script.append(_scripted_rows([6, 5], [1, 4]))
    script.append(_scripted_rows([1], [1]))
    var m = _ScriptedModel("llada", script^)
    var c = GenConfig()
    c.gen_length = 4
    c.block_length = 2
    c.steps = 4
    c.early_stop = False
    var obs = _Steps()
    var prompt: List[Int] = [1, 2]
    var r = generate_observed(m, prompt, c, obs)
    _assert_list(obs.committed[0], [2], "block 0 step 1")
    _assert_list(obs.committed[1], [3], "block 0 step 2")
    _assert_list(obs.committed[2], [5], "block 1 step 1")
    _assert_list(obs.committed[3], [4], "block 1 step 2")
    _assert_list(r.tokens, [1, 2, 5, 7, 1, 5], "final canvas")
    _assert_list(m.asked[2], [4, 5], "rows of block 1 step 1")


def test_dream_commits_its_schedule_in_confidence_order() raises:
    # Four slots at positions 2..5 and three steps. n_transfer is
    # int(n_masked * (1 - s/t)): 1 of 4, 1 of 3, then the last 2. Dream's
    # logits are shifted, so the rows asked for are position - 1.
    var script = List[List[Float32]]()
    script.append(_scripted_rows([5, 6, 7, 5], [1, 4, 2, 3]))
    script.append(_scripted_rows([7, 5, 6], [3, 1, 2]))
    script.append(_scripted_rows([6, 1], [2, 5]))
    var m = _ScriptedModel("dream", script^)
    var c = GenConfig()
    c.gen_length = 4
    c.steps = 3
    c.early_stop = False
    var obs = _Steps()
    var prompt: List[Int] = [1, 2]
    var r = generate_observed(m, prompt, c, obs)
    assert_equal(len(obs.committed), 3)
    _assert_list(obs.committed[0], [3], "step 1 committed")
    _assert_list(obs.committed[1], [2], "step 2 committed")
    _assert_list(obs.committed[2], [5, 4], "step 3 committed")
    _assert_list(m.asked[0], [1, 2, 3, 4], "rows of step 1")
    _assert_list(m.asked[1], [1, 3, 4], "rows of step 2")
    _assert_list(m.asked[2], [3, 4], "rows of step 3")
    _assert_list(r.tokens, [1, 2, 7, 6, 6, 1], "final canvas")


def _check_mask_score_row(mask_id: Int, runner_up: Int) raises:
    var vocab = 700
    var logits = Floats(vocab)
    for v in range(vocab):
        logits[v] = -5.0
    logits[runner_up] = 5.0
    logits[mask_id] = 10.0
    var row = logits.ptr()

    var plain = argmax_and_prob(row, vocab)
    assert_equal(plain[0], mask_id, "unsuppressed positional call")

    var z = 1.0 + Float64(vocab - 2) * exp(Float64(-10.0))
    var expected_p = 1.0 / z
    var expected_other_p = exp(Float64(-10.0)) / z
    var expected_entropy = expected_p * log(expected_p) + Float64(
        vocab - 2
    ) * expected_other_p * log(expected_other_p)
    var expected_margin = expected_p - expected_other_p

    var argmax = argmax_and_prob(row, vocab, mask_id)
    assert_equal(argmax[0], runner_up)
    assert_true(abs(argmax[1] - expected_p) < 1e-12)

    var probability = _prob_of(row, vocab, runner_up, mask_id)
    assert_true(abs(probability - expected_p) < 1e-12)

    var plus = _confidence(row, vocab, "maskgit_plus", mask_id)
    assert_equal(plus[0], runner_up)
    assert_true(abs(plus[1] - expected_p) < 1e-12)

    var margin = _confidence(row, vocab, "topk_margin", mask_id)
    assert_equal(margin[0], runner_up)
    assert_true(abs(margin[1] - expected_margin) < 1e-12)

    var entropy = _confidence(row, vocab, "entropy", mask_id)
    assert_equal(entropy[0], runner_up)
    assert_true(
        abs(entropy[1] - expected_entropy) < 1e-9,
        "entropy "
        + String(entropy[1])
        + " against allowed-row value "
        + String(expected_entropy),
    )

    var greedy = _greedy_rows(row, [0], vocab, "llada", 1, mask_id)
    assert_equal(greedy[0][0], runner_up)
    assert_true(abs(greedy[1][0] - expected_p) < 1e-12)

    var greedy_plus = _greedy_rows(row, [0], vocab, "maskgit_plus", 1, mask_id)
    assert_equal(greedy_plus[0][0], runner_up)
    assert_true(abs(greedy_plus[1][0] - expected_p) < 1e-12)


def test_mask_token_is_excluded_from_greedy_scores() raises:
    # Include index 0 and the final index to cover scorers' old start values.
    _check_mask_score_row(0, 350)
    _check_mask_score_row(350, 0)
    _check_mask_score_row(699, 350)


def _check_gumbel_mask(mask_id: Int) raises:
    var vocab = 700
    var runner_up = 0 if mask_id != 0 else 1
    var logits = Floats(vocab)
    for v in range(vocab):
        logits[v] = -100.0
    logits[runner_up] = 0.0
    logits[mask_id] = 1000.0
    var suppressed_rng = Rng(991)
    var plain_rng = Rng(991)
    var selected = _gumbel_argmax(
        logits.ptr(), vocab, 1.0, suppressed_rng, mask_id
    )
    var plain = _gumbel_argmax(logits.ptr(), vocab, 1.0, plain_rng)
    assert_equal(plain, mask_id)
    assert_true(selected != mask_id, "mask candidate won Gumbel sampling")
    assert_equal(suppressed_rng.next_u64(), plain_rng.next_u64())


def test_gumbel_mask_skip_preserves_rng_draws() raises:
    _check_gumbel_mask(0)
    _check_gumbel_mask(350)
    _check_gumbel_mask(699)


def test_mask_suppression_preserves_fixture_generations() raises:
    var defaults = GenConfig()
    assert_true(defaults.suppress_mask, "mask suppression should default on")
    for path in ["tiny-llada.gguf", "tiny-dream.gguf"]:
        var model = DiffusionLM(FIX + path, threads=2, max_tokens=16)
        var prompt: List[Int] = [5, 17, 300]
        var mask = 510 if path == "tiny-dream.gguf" else 511
        var on = _tiny_cfg()
        on.mask_id = mask
        on.suppress_mask = True
        var off = on.copy()
        off.suppress_mask = False
        var suppressed = generate(model, prompt, on)
        var unchanged = generate(model, prompt, off)
        assert_equal(suppressed.forward_passes, unchanged.forward_passes)
        assert_equal(len(suppressed.tokens), len(unchanged.tokens))
        for i in range(len(suppressed.tokens)):
            assert_equal(
                suppressed.tokens[i],
                unchanged.tokens[i],
                path + " position " + String(i),
            )


struct FirstCommit(StepObserver):
    var step_one_commits: Int
    var step_one_tokens: List[Int]

    def __init__(out self):
        self.step_one_commits = -1
        self.step_one_tokens = List[Int]()

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        if step == 1:
            self.step_one_commits = len(committed)
            self.step_one_tokens = tokens.copy()


def _dream_cfg(gen_length: Int, steps: Int, seed: Int) -> GenConfig:
    var c = GenConfig()
    c.gen_length = gen_length
    c.steps = steps
    c.mask_id = 510
    c.seed = seed
    c.remasking = "origin"
    return c^


def test_dream_origin_commits_everything_on_final_step() raises:
    var model = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = _dream_cfg(8, 1, 42)
    # Keep the normal transfer probability below one; the last step must
    # explicitly override it to one.
    c.eps = 0.9
    var recorder = FirstCommit()
    var result = generate_observed(model, prompt, c, recorder)
    assert_equal(result.forward_passes, 1)
    assert_equal(recorder.step_one_commits, 8)
    for i in range(len(prompt), len(result.tokens)):
        assert_true(result.tokens[i] != c.mask_id, "mask left at " + String(i))


def test_dream_origin_first_step_matches_binomial_mean() raises:
    var model = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=70)
    var prompt: List[Int] = [5, 17, 300]
    var total_first_step = 0
    for seed in range(200):
        var c = _dream_cfg(64, 4, seed)
        c.eps = 1e-3
        var recorder = FirstCommit()
        _ = generate_observed(model, prompt, c, recorder)
        total_first_step += recorder.step_one_commits
    var mean = Float64(total_first_step) / 200.0
    print("origin step-one mean over 200 seeds:", mean)
    assert_true(
        abs(mean - 15.98) < 1.0,
        "origin step-one mean " + String(mean) + " against 15.98",
    )


def test_dream_origin_and_random_are_seeded_and_resolve_masks() raises:
    var model = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=70)
    var prompt: List[Int] = [5, 17, 300]
    for algorithm in ["origin", "random"]:
        var c = _dream_cfg(64, 4, 2026)
        c.remasking = algorithm
        var a = generate(model, prompt, c)
        var b = generate(model, prompt, c)
        assert_equal(a.forward_passes, b.forward_passes, algorithm)
        for i in range(len(a.tokens)):
            assert_equal(
                a.tokens[i], b.tokens[i], algorithm + " position " + String(i)
            )
            if i >= len(prompt):
                assert_true(
                    a.tokens[i] != c.mask_id, algorithm + " left a mask"
                )

    var recorder = FirstCommit()
    var random_cfg = _dream_cfg(64, 4, 2026)
    random_cfg.remasking = "random"
    _ = generate_observed(model, prompt, random_cfg, recorder)
    assert_equal(recorder.step_one_commits, 15)
    var expected_rng = Rng(2026)
    var scores = List[Float64]()
    for _ in range(64):
        scores.append(expected_rng.uniform())
    var expected_positions = select_top(scores, 15)
    for j in range(64):
        var expected = False
        for selected in expected_positions:
            if selected == j:
                expected = True
        var committed = (
            recorder.step_one_tokens[len(prompt) + j] != random_cfg.mask_id
        )
        assert_equal(
            committed,
            expected,
            "random step-one position " + String(j),
        )


def test_remasking_names_are_model_specific() raises:
    var prompt: List[Int] = [5, 17, 300]
    var llada = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var c = _tiny_cfg()
    c.remasking = "origin"
    with assert_raises(contains="unknown remasking strategy"):
        _ = generate(llada, prompt, c)

    var dream = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    c = _dream_cfg(8, 4, 42)
    c.remasking = "unknown"
    with assert_raises(contains="unknown Dream algorithm"):
        _ = generate(dream, prompt, c)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
