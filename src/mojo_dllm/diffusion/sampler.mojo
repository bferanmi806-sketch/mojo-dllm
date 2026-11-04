"""Denoising loops: LLaDA's `generate.py`, Dream's `diffusion_generate` and
DiffusionGemma's entropy-bounded denoiser.

LLaDA (semi-autoregressive blocks, below) and Dream (one block, timestep
schedule, see `_generate_dream`) share the model call; `generate` picks the
loop from the model's architecture. DiffusionGemma starts from random tokens
instead of masks and has its own loop, `_generate_diffusion_gemma`.

LLaDA (ML-GSAI/LLaDA `generate.py`):

    canvas = prompt ++ [MASK] * gen_length
    for each block of `block_length` generated positions, left to right:
        m = masked positions in the block
        spread m over steps/num_blocks steps:  m // s each, +1 for the first m % s
        each step:
            logits = model(canvas)                    # bidirectional, whole canvas
            x0[i]  = argmax(logits[i] (+ Gumbel noise if temperature > 0))
            conf[i] = softmax(logits[i])[x0[i]]       # "low_confidence" remasking
                      or U(0,1)                       # "random" remasking
            commit the k most confident masked positions of the block

For LLaDA only the masked positions of the current block are candidates, so
only their logits are requested from the model (`full_logits` turns that off,
to measure what the saving is worth). Dream requests every masked position of
the canvas. Probabilities are computed in f64, as the reference
does.

Seeds drive this module's own PRNG (splitmix64 + xoshiro256**); a seed here
does not reproduce a PyTorch run. At temperature 0 with low-confidence
remasking the algorithm is deterministic and token-for-token comparable with
the reference.
"""

from std.math import exp, isfinite, log
from std.time import perf_counter_ns

from mojo_dllm.models.transformer import DenoisingModel, RowStats
from mojo_dllm.sys.mem import Bytes, F32Ptr, Floats
from mojo_dllm.sys.parallel import parallel_for


struct GenConfig(Copyable, Movable):
    """Sampling options. Mask suppression is on by default. Dream keeps the
    entropy default; origin uses independent transfers, random ranks uniforms.
    """

    var gen_length: Int
    var block_length: Int
    var steps: Int
    var temperature: Float64
    var remasking: String
    var seed: Int
    var mask_id: Int
    var suppress_mask: Bool
    """Exclude mask_id from token selection and softmax; disable for parity."""
    var full_logits: Bool
    var eps: Float64
    """Dream's final timestep: timesteps run linspace(1, eps, steps + 1)."""
    var threads: Int
    """Workers for the per-row confidence pass; each row's arithmetic is the
    same whatever the count, so results do not depend on it."""
    var early_stop: Bool
    """Stop once the visible answer is fixed (see `answer_is_fixed`)."""
    var stop_ids: List[Int]
    """End-of-text tokens: the CLI prints the canvas up to the first of them."""
    var dual_cache: Bool
    """LLaDA: refresh the model's K/V cache at each block's first step and
    recompute only the block's rows on the others (Fast-dLLM's DualCache).
    The model must be built with its K/V cache on. Approximate."""
    var threshold: Float64
    """LLaDA: also commit every candidate whose confidence reaches this value
    (Fast-dLLM's parallel decoding). 0 turns it off. A step never commits
    fewer tokens than the schedule, so a block never takes more steps."""
    var eb_steps: Int
    """DiffusionGemma: most denoising steps (the reference's 48)."""
    var eb_t_min: Float64
    """DiffusionGemma: the temperature the schedule approaches at the last
    step (0.4). The last step runs at t_min + (t_max - t_min) / steps."""
    var eb_t_max: Float64
    """DiffusionGemma: the temperature of the first step (0.8)."""
    var eb_bound: Float64
    """DiffusionGemma: positions are accepted while the entropies before them,
    lowest first, sum to at most this (0.1)."""
    var eb_stability: Int
    """DiffusionGemma: steps the argmax canvas must hold (1)."""
    var eb_confidence: Float64
    """DiffusionGemma: stop once the mean entropy is below this (0.005)."""

    def __init__(out self):
        self.gen_length = 128
        self.block_length = 32
        self.steps = 128
        self.temperature = 0.0
        self.remasking = "low_confidence"
        self.seed = 42
        self.mask_id = 126336
        self.suppress_mask = True
        self.full_logits = False
        self.eps = 1e-3
        self.threads = 1
        self.early_stop = True
        self.stop_ids = List[Int]()
        self.threshold = 0.0
        self.dual_cache = False
        self.eb_steps = 48
        self.eb_t_min = 0.4
        self.eb_t_max = 0.8
        self.eb_bound = 0.1
        self.eb_stability = 1
        self.eb_confidence = 0.005


struct GenResult(Movable):
    var tokens: List[Int]
    var step_ms: List[Float64]
    var forward_passes: Int
    var seconds: Float64
    var stopped_early: Bool
    var steps_taken: Int
    """Denoising steps run: one forward pass each."""
    var stop_reason: String
    """Why the loop ended: "complete" (the schedule ran out), "answer_fixed"
    (LLaDA and Dream early stop), "converged" (DiffusionGemma's argmax held
    and its mean entropy fell below the threshold) or "max_steps"
    (DiffusionGemma used every step)."""

    def __init__(out self):
        self.tokens = List[Int]()
        self.step_ms = List[Float64]()
        self.forward_passes = 0
        self.seconds = 0.0
        self.stopped_early = False
        self.steps_taken = 0
        self.stop_reason = "complete"


struct Rng(Movable):
    """xoshiro256** seeded through splitmix64."""

    var s0: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64

    def __init__(out self, seed: Int):
        var x = UInt64(seed)
        self.s0 = Rng._splitmix(x)
        self.s1 = Rng._splitmix(x)
        self.s2 = Rng._splitmix(x)
        self.s3 = Rng._splitmix(x)

    @staticmethod
    def _splitmix(mut x: UInt64) -> UInt64:
        x += 0x9E3779B97F4A7C15
        var z = x
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    @staticmethod
    def _rotl(x: UInt64, k: UInt64) -> UInt64:
        return (x << k) | (x >> (64 - k))

    def next_u64(mut self) -> UInt64:
        var result = Rng._rotl(self.s1 * 5, 7) * 9
        var t = self.s1 << 17
        self.s2 ^= self.s0
        self.s3 ^= self.s1
        self.s1 ^= self.s2
        self.s0 ^= self.s3
        self.s2 ^= t
        self.s3 = Rng._rotl(self.s3, 45)
        return result

    def uniform(mut self) -> Float64:
        """Uniform in the open interval (0, 1)."""
        return (Float64(self.next_u64() >> 11) + 0.5) * (
            1.0 / 9007199254740992.0
        )

    def below(mut self, n: Int) -> Int:
        """Uniform in [0, n): the 53 high bits scaled, unbiased to 2^-53."""
        var x = Float64(self.next_u64() >> 11) * (1.0 / 9007199254740992.0)
        return min(Int(x * Float64(n)), n - 1)


trait StepObserver:
    """Receives the canvas after every denoising step (for live display)."""

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        ...


struct NoObserver(StepObserver):
    def __init__(out self):
        pass

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        pass


def num_transfer_tokens(m: Int, steps: Int) -> List[Int]:
    var out = List[Int](length=steps, fill=m // steps)
    for i in range(m % steps):
        out[i] += 1
    return out^


def select_top(conf: List[Float64], k: Int) -> List[Int]:
    """Indices of the k largest values; ties go to the lower index."""
    var idx = List[Int](capacity=len(conf))
    for i in range(len(conf)):
        idx.append(i)
    # Insertion sort: candidates per step are at most one block (tens of items).
    for a in range(1, len(idx)):
        var v = idx[a]
        var b = a - 1
        while b >= 0 and (conf[idx[b]] < conf[v]):
            idx[b + 1] = idx[b]
            b -= 1
        idx[b + 1] = v
    var n = min(k, len(idx))
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(idx[i])
    return out^


def argmax_and_prob(
    row: F32Ptr, vocab: Int, mask_id: Int = -1
) -> Tuple[Int, Float64]:
    """First argmax and its softmax probability, excluding mask_id, in f64."""
    var best = 1 if mask_id == 0 else 0
    var mx = row.unsafe_load(best)
    for v in range(1, vocab):
        if v == mask_id:
            continue
        var x = row.unsafe_load(v)
        if x > mx:
            mx = x
            best = v
    var z = Float64(0)
    var m64 = Float64(mx)
    for v in range(vocab):
        if v == mask_id:
            continue
        z += exp(Float64(row.unsafe_load(v)) - m64)
    return (best, 1.0 / z)


def _gumbel_argmax(
    row: F32Ptr,
    vocab: Int,
    temperature: Float64,
    mut rng: Rng,
    mask_id: Int = -1,
) -> Int:
    """argmax of exp(logits) / (-log U)^temperature, the reference's noise, in f64.

    Compared in log space, which ranks identically and does not overflow:
    logit - temperature * log(-log U). mask_id consumes a draw but cannot win.
    """
    var best = 1 if mask_id == 0 else 0
    var best_v = Float64(-1e300)
    for v in range(vocab):
        var u = rng.uniform()
        # Keep one draw per vocabulary entry, even for the excluded mask.
        if v == mask_id:
            continue
        var score = Float64(row.unsafe_load(v)) - temperature * log(-log(u))
        if score > best_v:
            best_v = score
            best = v
    return best


def _greedy_rows(
    logits: F32Ptr,
    row_of: List[Int],
    vocab: Int,
    algorithm: String,
    threads: Int,
    mask_id: Int = -1,
) raises -> Tuple[List[Int], List[Float64]]:
    """Greedy token and confidence for each requested logits row, rows in parallel.

    algorithm "llada" scores with `argmax_and_prob`; anything else is a Dream
    algorithm for `_confidence`. Every row is computed exactly as the
    sequential loop would, so the thread count cannot change a generation.
    """
    var n = len(row_of)
    var tok_buf = Bytes(8 * max(n, 1))
    var conf_buf = Bytes(8 * max(n, 1))
    var tp = Pointer[Int64, MutAnyOrigin](unsafe_from_address=tok_buf.addr)
    var cp = Pointer[Float64, MutAnyOrigin](unsafe_from_address=conf_buf.addr)

    def work(j: Int) {imm}:
        var row = logits.unsafe_offset(row_of[j] * vocab)
        if algorithm == "llada":
            var r = argmax_and_prob(row, vocab, mask_id)
            tp[unsafe_offset=j] = Int64(r[0])
            cp[unsafe_offset=j] = r[1]
        else:
            var r = _confidence(row, vocab, algorithm, mask_id)
            tp[unsafe_offset=j] = Int64(r[0])
            cp[unsafe_offset=j] = r[1]

    parallel_for(work, n, threads)
    var toks = List[Int](capacity=n)
    var confs = List[Float64](capacity=n)
    for j in range(n):
        toks.append(Int(tp[unsafe_offset=j]))
        confs.append(cp[unsafe_offset=j])
    # tp and cp point into these; keep them alive until the last read.
    _ = tok_buf^
    _ = conf_buf^
    return (toks^, confs^)


def answer_is_fixed(
    tokens: List[Int], P: Int, mask_id: Int, stop_ids: List[Int]
) -> Bool:
    """True once no later step can change the text a reader sees.

    The visible answer is the canvas from P up to the first stop token. Steps
    only ever fill masked positions, so once the first position that is
    masked or a stop token is a stop token, every visible token is final.
    Stopping there is exact for the printed text; the positions after it
    stay masked in the returned canvas.
    """
    for i in range(P, len(tokens)):
        var t = tokens[i]
        if t == mask_id:
            return False
        for s in stop_ids:
            if t == s:
                return True
    return False


def generate[
    M: DenoisingModel
](mut model: M, prompt: List[Int], cfg: GenConfig) raises -> GenResult:
    var none = NoObserver()
    return generate_observed(model, prompt, cfg, none)


def generate_observed[
    M: DenoisingModel, O: StepObserver
](
    mut model: M, prompt: List[Int], cfg: GenConfig, mut obs: O
) raises -> GenResult:
    """Like `generate`, calling `obs.on_step` after every denoising step."""
    if model.config().arch == "dream":
        return _generate_dream(model, prompt, cfg, obs)
    if model.config().arch == "diffusion-gemma":
        return _generate_diffusion_gemma(model, prompt, cfg, obs)
    return _generate_llada(model, prompt, cfg, obs)


def _generate_llada[
    M: DenoisingModel, O: StepObserver
](
    mut model: M, prompt: List[Int], cfg: GenConfig, mut obs: O
) raises -> GenResult:
    if cfg.gen_length <= 0 or cfg.block_length <= 0 or cfg.steps <= 0:
        raise Error(
            "generation length, block length and steps must be positive"
        )
    if cfg.gen_length % cfg.block_length != 0:
        raise Error(
            "generation length "
            + String(cfg.gen_length)
            + " is not a multiple of the block length "
            + String(cfg.block_length)
        )
    var num_blocks = cfg.gen_length // cfg.block_length
    if cfg.steps % num_blocks != 0:
        raise Error(
            "steps "
            + String(cfg.steps)
            + " is not a multiple of the number of blocks "
            + String(num_blocks)
        )
    if cfg.dual_cache and cfg.full_logits:
        raise Error(
            "the K/V cache recomputes one block, so it cannot give every row's"
            " logits"
        )
    if cfg.threshold < 0:
        raise Error("the confidence threshold must not be negative")
    if cfg.threshold > 0 and cfg.remasking == "random":
        raise Error(
            "a confidence threshold needs low_confidence remasking; random"
            " remasking has no confidence"
        )
    if cfg.remasking != "low_confidence" and cfg.remasking != "random":
        raise Error("unknown remasking strategy: " + cfg.remasking)
    var steps_per_block = cfg.steps // num_blocks
    var P = len(prompt)
    var L = P + cfg.gen_length
    var vocab = model.config().vocab
    var mask_id = cfg.mask_id if cfg.suppress_mask else -1
    var res = GenResult()
    for t in prompt:
        res.tokens.append(t)
    for _ in range(cfg.gen_length):
        res.tokens.append(cfg.mask_id)
    var rng = Rng(cfg.seed)
    # Device reductions include every vocabulary entry. Suppression needs the
    # full row on the host, as does Gumbel noise. With both off, only the token
    # and four statistics per row leave the device (`full_logits` disables it).
    var use_stats = (
        cfg.temperature <= 0
        and not cfg.full_logits
        and not cfg.dual_cache
        and not cfg.suppress_mask
        and model.device_row_stats()
    )
    var logits = Floats(
        1 if use_stats else (L if cfg.full_logits else cfg.block_length) * vocab
    )
    var stats = RowStats()
    var t_start = perf_counter_ns()

    for b in range(num_blocks):
        var b0 = P + b * cfg.block_length
        var b1 = b0 + cfg.block_length
        var m = 0
        for i in range(b0, b1):
            if res.tokens[i] == cfg.mask_id:
                m += 1
        var schedule = num_transfer_tokens(m, steps_per_block)
        for s in range(steps_per_block):
            var t_step = perf_counter_ns()
            var cand = List[Int]()
            for i in range(b0, b1):
                if res.tokens[i] == cfg.mask_id:
                    cand.append(i)
            if len(cand) == 0:
                break
            var rows = List[Int]()
            if cfg.full_logits:
                for i in range(L):
                    rows.append(i)
            else:
                rows = cand.copy()
            if cfg.dual_cache:
                # The block's first step refreshes every row's K/V; the rest
                # recompute the block's rows against that cache.
                if s == 0:
                    model.forward_window(res.tokens, 0, L, rows, logits.ptr())
                else:
                    model.forward_window(res.tokens, b0, b1, rows, logits.ptr())
            elif use_stats:
                model.forward_stats(res.tokens, rows, stats)
            else:
                model.forward(res.tokens, rows, logits.ptr())
            res.forward_passes += 1
            var x0 = List[Int](capacity=len(cand))
            var conf = List[Float64](capacity=len(cand))
            var row_of = List[Int](capacity=len(cand))
            for j in range(len(cand)):
                row_of.append(cand[j] if cfg.full_logits else j)
            if cfg.temperature > 0:
                # Gumbel noise draws from one RNG stream in row order, so this
                # path stays sequential.
                for j in range(len(cand)):
                    var row = logits.ptr().unsafe_offset(row_of[j] * vocab)
                    var tok = _gumbel_argmax(
                        row, vocab, cfg.temperature, rng, mask_id
                    )
                    x0.append(tok)
                    if cfg.remasking == "random":
                        conf.append(rng.uniform())
                    else:
                        conf.append(_prob_of(row, vocab, tok, mask_id))
            elif use_stats:
                x0 = stats.tok.copy()
                for j in range(len(cand)):
                    if cfg.remasking == "random":
                        conf.append(rng.uniform())
                    else:
                        conf.append(_confidence_from_stats(stats, j, "llada"))
            else:
                var greedy = _greedy_rows(
                    logits.ptr(), row_of, vocab, "llada", cfg.threads, mask_id
                )
                x0 = greedy[0].copy()
                if cfg.remasking == "random":
                    for _ in range(len(cand)):
                        conf.append(rng.uniform())
                else:
                    conf = greedy[1].copy()
            var committed = List[Int]()
            var k = schedule[s]
            if cfg.threshold > 0:
                var sure = 0
                for c in conf:
                    if c >= cfg.threshold:
                        sure += 1
                k = min(max(k, sure), len(cand))
            for j in select_top(conf, k):
                res.tokens[cand[j]] = x0[j]
                committed.append(cand[j])
            res.step_ms.append(Float64(perf_counter_ns() - t_step) / 1e6)
            obs.on_step(res.tokens, committed, res.forward_passes, cfg.steps)
            if cfg.early_stop and answer_is_fixed(
                res.tokens, P, cfg.mask_id, cfg.stop_ids
            ):
                res.stopped_early = True
                break
        if res.stopped_early:
            break
    res.seconds = Float64(perf_counter_ns() - t_start) / 1e9
    res.steps_taken = res.forward_passes
    if res.stopped_early:
        res.stop_reason = "answer_fixed"
    return res^


def _prob_of(row: F32Ptr, vocab: Int, tok: Int, mask_id: Int = -1) -> Float64:
    """Softmax probability of tok, with mask_id excluded from the distribution.
    """
    if tok == mask_id:
        return 0.0
    var first = 1 if mask_id == 0 else 0
    var mx = row.unsafe_load(first)
    for v in range(1, vocab):
        if v == mask_id:
            continue
        mx = max(mx, row.unsafe_load(v))
    var z = Float64(0)
    for v in range(vocab):
        if v == mask_id:
            continue
        z += exp(Float64(row.unsafe_load(v)) - Float64(mx))
    return exp(Float64(row.unsafe_load(tok)) - Float64(mx)) / z


def _confidence(
    row: F32Ptr, vocab: Int, algorithm: String, mask_id: Int = -1
) -> Tuple[Int, Float64]:
    """Greedy token and Dream's confidence for it, in f64.

    entropy:      sum_v p_v log p_v (negative entropy; larger is surer)
    maskgit_plus: p(x0)
    topk_margin:  p(top1) - p(top2)
    mask_id is excluded from the argmax and every probability sum.
    """
    var best = 1 if mask_id == 0 else 0
    var mx = row.unsafe_load(best)
    for v in range(1, vocab):
        if v == mask_id:
            continue
        var x = row.unsafe_load(v)
        if x > mx:
            mx = x
            best = v
    var m64 = Float64(mx)
    var z = Float64(0)
    var second = Float64(-1e300)
    for v in range(vocab):
        if v == mask_id:
            continue
        var x = Float64(row.unsafe_load(v))
        z += exp(x - m64)
        if v != best and x > second:
            second = x
    var p_best = 1.0 / z
    if algorithm == "maskgit_plus":
        return (best, p_best)
    if algorithm == "topk_margin":
        return (best, p_best - exp(second - m64) / z)
    var logz = log(z)
    var neg_h = Float64(0)
    for v in range(vocab):
        if v == mask_id:
            continue
        var lp = Float64(row.unsafe_load(v)) - m64 - logz
        neg_h += exp(lp) * lp
    return (best, neg_h)


def _confidence_from_stats(st: RowStats, j: Int, algorithm: String) -> Float64:
    """`_confidence` (or, for "llada", `argmax_and_prob`'s probability) of row
    j from its device reductions: p = 1 / z, the margin subtracts the
    runner-up's probability, and the negative entropy is
    sum e (x - m) / z - log z."""
    var z = st.z[j]
    var p_best = 1.0 / z
    if algorithm == "llada" or algorithm == "maskgit_plus":
        return p_best
    if algorithm == "topk_margin":
        return p_best - exp(st.second[j] - st.mx[j]) / z
    return st.weighted[j] / z - log(z)


def _generate_dream[
    M: DenoisingModel, O: StepObserver
](
    mut model: M, prompt: List[Int], cfg: GenConfig, mut obs: O
) raises -> GenResult:
    """Dream-org/Dream `diffusion_generate`, greedy (temperature 0, alg_temp 0).

    The whole generation region is one canvas. At step i of S, with
    t = 1 - i(1-eps)/S and s = 1 - (i+1)(1-eps)/S, Dream commits
    int(n_masked * (1 - s/t)) positions by confidence, or uniform random
    confidence for llama.cpp's random rule. Dream's origin rule instead makes
    independent transfers with probability 1 - s/t, without flooring a count.
    Every remaining position commits at the last step. Logits are shifted:
    the row before a position scores it.
    """
    if cfg.gen_length <= 0 or cfg.steps <= 0:
        raise Error("generation length and steps must be positive")
    var algorithm = cfg.remasking
    if algorithm == "low_confidence":
        algorithm = "entropy"
    if (
        algorithm != "entropy"
        and algorithm != "maskgit_plus"
        and algorithm != "topk_margin"
        and algorithm != "origin"
        and algorithm != "random"
    ):
        raise Error(
            "unknown Dream algorithm: "
            + cfg.remasking
            + " (entropy, maskgit_plus, topk_margin, origin, random)"
        )
    if cfg.temperature > 0:
        raise Error(
            "Dream sampling with temperature > 0 is not implemented yet; use"
            " --temperature 0"
        )
    if cfg.threshold > 0:
        raise Error("the confidence threshold is implemented for LLaDA only")
    if cfg.dual_cache:
        raise Error("the block K/V cache is implemented for LLaDA only")
    var P = len(prompt)
    if P == 0:
        raise Error(
            "Dream needs at least one prompt token (logits are shifted)"
        )
    var vocab = model.config().vocab
    var mask_id = cfg.mask_id if cfg.suppress_mask else -1
    var res = GenResult()
    for t in prompt:
        res.tokens.append(t)
    for _ in range(cfg.gen_length):
        res.tokens.append(cfg.mask_id)
    var rng = Rng(cfg.seed)
    # origin and random choose positions themselves, not by entropy.
    var scoring = (
        "maskgit_plus" if algorithm == "origin"
        or algorithm == "random" else algorithm
    )
    var use_stats = not cfg.suppress_mask and model.device_row_stats()
    var logits = Floats(1 if use_stats else cfg.gen_length * vocab)
    var stats = RowStats()
    var t_start = perf_counter_ns()
    for i in range(cfg.steps):
        var t_step = perf_counter_ns()
        var cand = List[Int]()
        for p in range(P, P + cfg.gen_length):
            if res.tokens[p] == cfg.mask_id:
                cand.append(p)
        if len(cand) == 0:
            break
        var rows = List[Int]()
        for p in cand:
            rows.append(model.logit_row(p))
        var x0 = List[Int]()
        var conf = List[Float64]()
        if use_stats:
            model.forward_stats(res.tokens, rows, stats)
            res.forward_passes += 1
            x0 = stats.tok.copy()
            for j in range(len(cand)):
                conf.append(_confidence_from_stats(stats, j, scoring))
        else:
            model.forward(res.tokens, rows, logits.ptr())
            res.forward_passes += 1
            var row_of = List[Int](capacity=len(cand))
            for j in range(len(cand)):
                row_of.append(j)
            var greedy = _greedy_rows(
                logits.ptr(), row_of, vocab, scoring, cfg.threads, mask_id
            )
            x0 = greedy[0].copy()
            conf = greedy[1].copy()
        var p_transfer = Float64(1.0)
        if i < cfg.steps - 1:
            var t = 1.0 - Float64(i) * (1.0 - cfg.eps) / Float64(cfg.steps)
            var s = 1.0 - Float64(i + 1) * (1.0 - cfg.eps) / Float64(cfg.steps)
            p_transfer = 1.0 - s / t
        var committed = List[Int]()
        if algorithm == "origin":
            for j in range(len(cand)):
                if rng.uniform() < p_transfer:
                    res.tokens[cand[j]] = x0[j]
                    committed.append(cand[j])
        else:
            if algorithm == "random":
                for j in range(len(cand)):
                    conf[j] = rng.uniform()
            var n_transfer = Int(Float64(len(cand)) * p_transfer)
            if n_transfer > 0:
                for j in select_top(conf, n_transfer):
                    res.tokens[cand[j]] = x0[j]
                    committed.append(cand[j])
        res.step_ms.append(Float64(perf_counter_ns() - t_step) / 1e6)
        obs.on_step(res.tokens, committed, res.forward_passes, cfg.steps)
        if cfg.early_stop and answer_is_fixed(
            res.tokens, P, cfg.mask_id, cfg.stop_ids
        ):
            res.stopped_early = True
            break
    res.seconds = Float64(perf_counter_ns() - t_start) / 1e9
    res.steps_taken = res.forward_passes
    if res.stopped_early:
        res.stop_reason = "answer_fixed"
    return res^


comptime EB_CHUNK = 1024
"""Positions per chunk in `eb_row`: the draw walks chunk sums, then one chunk."""


struct EbRow(Copyable, Movable):
    """One canvas position's answers from its logits row."""

    var argmax: Int
    """First index of the largest logit."""
    var entropy: Float64
    """Entropy in nats of softmax(logits / temperature)."""
    var sampled: Int
    """The index an inverse-CDF draw at u picks from that softmax."""

    def __init__(out self, argmax: Int, entropy: Float64, sampled: Int):
        self.argmax = argmax
        self.entropy = entropy
        self.sampled = sampled


def eb_row(row: F32Ptr, vocab: Int, temperature: Float64, u: Float64) -> EbRow:
    """Argmax, entropy and one draw for a logits row, reading it once.

    The reference scans the row three times: for the maximum, for the sum Z of
    exp(z - max) and then for the entropy and the draw (z = logit / t). Here
    one pass over `EB_CHUNK`-wide chunks keeps, per chunk, its own maximum m_c,
    sum Z_c = sum exp(z - m_c) and A_c = sum exp(z - m_c)(z - m_c), and merges
    them into the row's (m, Z, A) with exp(m_c - m) as the common factor. Then
    H = ln Z - A / Z. The draw needs Z, so it finds the chunk holding
    u * Z from the chunk sums and rescans only that chunk (at most
    `EB_CHUNK` values) for the first index whose running sum reaches the
    target, as the reference does over the whole row. Sums are in f64.
    A row with a non-finite value gives a non-finite entropy.
    """
    var inv = 1.0 / temperature
    var n_chunks = (vocab + EB_CHUNK - 1) // EB_CHUNK
    var cm = List[Float64](length=n_chunks, fill=0.0)
    var cz = List[Float64](length=n_chunks, fill=0.0)
    var best = 0
    var best_raw = row.unsafe_load(0)
    var gm = Float64(0)
    var gz = Float64(0)
    var ga = Float64(0)
    for c in range(n_chunks):
        var lo = c * EB_CHUNK
        var hi = min(lo + EB_CHUNK, vocab)
        var first = row.unsafe_load(lo)
        if first > best_raw:
            best_raw = first
            best = lo
        var m = Float64(first) * inv
        var z = Float64(1)
        var a = Float64(0)
        for v in range(lo + 1, hi):
            var raw = row.unsafe_load(v)
            if raw > best_raw:
                best_raw = raw
                best = v
            var x = Float64(raw) * inv
            if x > m:
                var d = m - x
                var k = exp(d)
                a = k * (a + d * z)
                z = k * z + 1.0
                m = x
            else:
                var e = exp(x - m)
                z += e
                a += e * (x - m)
        cm[c] = m
        cz[c] = z
        if c == 0:
            gm = m
            gz = z
            ga = a
        else:
            var nm = max(gm, m)
            var kg = exp(gm - nm)
            var kc = exp(m - nm)
            ga = kg * (ga + (gm - nm) * gz) + kc * (a + (m - nm) * z)
            gz = kg * gz + kc * z
            gm = nm
    var entropy = log(gz) - ga / gz
    var target = u * gz
    var cum = Float64(0)
    var chunk = n_chunks - 1
    for c in range(n_chunks):
        var w = cz[c] * exp(cm[c] - gm)
        if cum + w >= target:
            chunk = c
            break
        cum += w
    var lo = chunk * EB_CHUNK
    var hi = min(lo + EB_CHUNK, vocab)
    var sampled = hi - 1
    for v in range(lo, hi):
        cum += exp(Float64(row.unsafe_load(v)) * inv - gm)
        if cum >= target:
            sampled = v
            break
    return EbRow(best, entropy, sampled)


def eb_temperature(
    step: Int, steps: Int, t_min: Float64, t_max: Float64
) -> Float64:
    """Temperature of the 0-based `step`: t_min + (t_max - t_min) * cur / steps
    with cur = steps - step, so t_max on the first step and t_min + (t_max -
    t_min) / steps on the last. It does not depend on when the run stops."""
    var cur = steps - step
    return t_min + (t_max - t_min) * (Float64(cur) / Float64(steps))


def eb_initial_canvas(mut rng: Rng, n: Int, vocab: Int) -> List[Int]:
    """`n` ids drawn uniformly from the whole vocabulary, special tokens
    included."""
    var out = List[Int](capacity=n)
    for _ in range(n):
        out.append(rng.below(vocab))
    return out^


def eb_accept(entropy: List[Float64], bound: Float64) -> List[Bool]:
    """Which positions the step accepts.

    Positions are taken in ascending entropy (ties to the lower index) and a
    position is accepted when the entropies of the positions before it sum to
    at most `bound`, so the lowest-entropy position always goes in. The sum
    is the reference's: a running total less the position's own entropy.
    """
    var n = len(entropy)
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    # Stable insertion sort; a canvas is a few hundred positions.
    for a in range(1, n):
        var v = order[a]
        var b = a - 1
        while b >= 0 and entropy[order[b]] > entropy[v]:
            order[b + 1] = order[b]
            b -= 1
        order[b + 1] = v
    var accepted = List[Bool](length=n, fill=False)
    var total = Float64(0)
    for k in range(n):
        var pos = order[k]
        total += entropy[pos]
        if total - entropy[pos] <= bound:
            accepted[pos] = True
    return accepted^


def eb_should_stop(
    held: Int, stability: Int, mean_entropy: Float64, confidence: Float64
) -> Bool:
    """The argmax canvas has held for `stability` steps and the mean entropy
    is below `confidence`."""
    return held >= stability and mean_entropy < confidence


def _eb_rows(
    logits: F32Ptr,
    rows: Int,
    vocab: Int,
    temperature: Float64,
    u: List[Float64],
    threads: Int,
) raises -> Tuple[List[Int], List[Float64], List[Int]]:
    """`eb_row` for each of `rows` rows, in parallel; each row is computed
    alone, so the thread count cannot change a result."""
    var arg_buf = Bytes(8 * max(rows, 1))
    var ent_buf = Bytes(8 * max(rows, 1))
    var sam_buf = Bytes(8 * max(rows, 1))
    var ap = Pointer[Int64, MutAnyOrigin](unsafe_from_address=arg_buf.addr)
    var ep = Pointer[Float64, MutAnyOrigin](unsafe_from_address=ent_buf.addr)
    var sp = Pointer[Int64, MutAnyOrigin](unsafe_from_address=sam_buf.addr)

    def work(j: Int) {imm}:
        var r = eb_row(
            logits.unsafe_offset(j * vocab), vocab, temperature, u[j]
        )
        ap[unsafe_offset=j] = Int64(r.argmax)
        ep[unsafe_offset=j] = r.entropy
        sp[unsafe_offset=j] = Int64(r.sampled)

    parallel_for(work, rows, threads)
    var arg = List[Int](capacity=rows)
    var ent = List[Float64](capacity=rows)
    var sam = List[Int](capacity=rows)
    for j in range(rows):
        arg.append(Int(ap[unsafe_offset=j]))
        ent.append(ep[unsafe_offset=j])
        sam.append(Int(sp[unsafe_offset=j]))
    _ = arg_buf^
    _ = ent_buf^
    _ = sam_buf^
    return (arg^, ent^, sam^)


def _eb_flag_error(flag: String, rule: String, got: String) -> Error:
    return Error("--" + flag + " " + rule + ", got " + got)


def _generate_diffusion_gemma[
    M: DenoisingModel, O: StepObserver
](
    mut model: M, prompt: List[Int], cfg: GenConfig, mut obs: O
) raises -> GenResult:
    """DiffusionGemma's entropy-bounded denoiser, after
    `diffusion_generate_entropy_bound` in llama.cpp PR 24423 (2e715c0639) and
    transformers' `EntropyBoundSampler`.

        canvas = random ids over the whole vocabulary
        for step 0 .. S - 1:
            t = eb_temperature(step)
            logits = model([prompt | canvas])           # canvas rows only
            per position: argmax, entropy of softmax(logits / t),
                          one draw from that softmax
            accept positions by `eb_accept`
            canvas = draw where accepted, a fresh random id elsewhere
            output = argmax canvas
            stop if the argmax canvas held for `eb_stability` steps and the
                mean entropy is below `eb_confidence`

    The result's tokens are the prompt and the argmax canvas of the last step.
    Randomness is this module's PRNG, drawn in the reference's order but
    not its values: the canvas first, then each step's (u, renoise id) per
    position, before the rows are scored, so a seed fixes the run whatever the
    thread count.
    """
    if cfg.gen_length <= 0:
        raise Error("generation length must be positive")
    if cfg.eb_steps < 1:
        raise _eb_flag_error(
            "eb-steps", "must be at least 1", String(cfg.eb_steps)
        )
    # The reference accepts t_min = 0: every step still runs at a positive
    # temperature, the last at t_min + (t_max - t_min) / steps.
    if not isfinite(cfg.eb_t_min) or cfg.eb_t_min < 0:
        raise _eb_flag_error(
            "eb-t-min", "must be at least 0", String(cfg.eb_t_min)
        )
    if not isfinite(cfg.eb_t_max) or cfg.eb_t_max <= 0:
        raise _eb_flag_error(
            "eb-t-max", "must be a number above 0", String(cfg.eb_t_max)
        )
    if not isfinite(cfg.eb_bound) or cfg.eb_bound < 0:
        raise _eb_flag_error(
            "eb-bound", "must be at least 0", String(cfg.eb_bound)
        )
    if cfg.eb_stability < 0:
        raise _eb_flag_error(
            "eb-stability", "must be at least 0", String(cfg.eb_stability)
        )
    if not isfinite(cfg.eb_confidence) or cfg.eb_confidence < 0:
        raise _eb_flag_error(
            "eb-confidence", "must be at least 0", String(cfg.eb_confidence)
        )
    var P = len(prompt)
    if P == 0:
        raise Error("DiffusionGemma needs at least one prompt token")
    var C = cfg.gen_length
    var S = cfg.eb_steps
    var vocab = model.config().vocab
    var rng = Rng(cfg.seed)
    var canvas = eb_initial_canvas(rng, C, vocab)
    var rows = List[Int](capacity=C)
    for i in range(C):
        rows.append(model.logit_row(P + i))
    var logits = Floats(C * vocab)
    var res = GenResult()
    for t in prompt:
        res.tokens.append(t)
    for t in canvas:
        res.tokens.append(t)
    var prev = List[Int]()
    var held = 0
    var finished = False
    var u = List[Float64](length=C, fill=0.0)
    var renoise = List[Int](length=C, fill=0)
    var t_start = perf_counter_ns()
    for step in range(S):
        var t_step = perf_counter_ns()
        var temperature = eb_temperature(step, S, cfg.eb_t_min, cfg.eb_t_max)
        var feed = List[Int](capacity=P + C)
        for t in prompt:
            feed.append(t)
        for t in canvas:
            feed.append(t)
        model.forward(feed, rows, logits.ptr())
        res.forward_passes += 1
        for pos in range(C):
            u[pos] = rng.uniform()
            renoise[pos] = rng.below(vocab)
        var scored = _eb_rows(
            logits.ptr(), C, vocab, temperature, u, cfg.threads
        )
        var arg = scored[0].copy()
        var ent = scored[1].copy()
        var sam = scored[2].copy()
        var entropy_sum = Float64(0)
        for pos in range(C):
            if not isfinite(ent[pos]):
                raise Error(
                    "non-finite logits in canvas position "
                    + String(pos)
                    + " at step "
                    + String(step + 1)
                )
            entropy_sum += ent[pos]
        var accepted = eb_accept(ent, cfg.eb_bound)
        var committed = List[Int]()
        for pos in range(C):
            if accepted[pos]:
                canvas[pos] = sam[pos]
                committed.append(P + pos)
            else:
                canvas[pos] = renoise[pos]
            res.tokens[P + pos] = arg[pos]
        var same = len(prev) == C
        if same:
            for pos in range(C):
                if prev[pos] != arg[pos]:
                    same = False
                    break
        held = held + 1 if same else 0
        prev = arg^
        var stop = eb_should_stop(
            held, cfg.eb_stability, entropy_sum / Float64(C), cfg.eb_confidence
        )
        res.step_ms.append(Float64(perf_counter_ns() - t_step) / 1e6)
        obs.on_step(res.tokens, committed, res.forward_passes, S)
        if stop:
            finished = True
            break
    res.seconds = Float64(perf_counter_ns() - t_start) / 1e9
    res.steps_taken = res.forward_passes
    res.stop_reason = "converged" if finished else "max_steps"
    res.stopped_early = finished and res.steps_taken < S
    return res^
