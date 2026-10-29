"""Denoising loops: LLaDA's `generate.py` and Dream's `diffusion_generate`.

LLaDA (semi-autoregressive blocks, below) and Dream (one block, timestep
schedule, see `_generate_dream`) share the model call; `generate` picks the
loop from the model's architecture.

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

Only the masked positions of the current block are candidates, so only their
logits are requested from the model (`full_logits` turns that off, to measure
what the saving is worth). Probabilities are computed in f64, as the reference
does.

Seeds drive this module's own PRNG (splitmix64 + xoshiro256**); a seed here
does not reproduce a PyTorch run. At temperature 0 with low-confidence
remasking the algorithm is deterministic and token-for-token comparable with
the reference.
"""

from std.math import exp, log
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


struct GenResult(Movable):
    var tokens: List[Int]
    var step_ms: List[Float64]
    var forward_passes: Int
    var seconds: Float64
    var stopped_early: Bool

    def __init__(out self):
        self.tokens = List[Int]()
        self.step_ms = List[Float64]()
        self.forward_passes = 0
        self.seconds = 0.0
        self.stopped_early = False


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
    return res^
