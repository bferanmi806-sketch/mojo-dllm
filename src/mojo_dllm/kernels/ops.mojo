"""Transformer primitives in f32: RMSNorm, NORM RoPE, attention, SwiGLU.

These are a small share of a denoising step (the linear layers are over 98% of
the arithmetic), so they favour clarity and SIMD over cleverness.

Attention is bidirectional. A diffusion model sees the whole canvas, so there
is no causal mask and, by default, no KV cache: every step recomputes K and V
for every position, because any position may have changed. `--dual-cache`
(LLaDA) keeps a block-level cache and gives up exactness.
"""

from std.math import exp, sqrt, cos, sin

from mojo_dllm.sys.mem import F32Ptr, I32Ptr, Floats
from mojo_dllm.sys.parallel import parallel_for

comptime W = 8


def rmsnorm(
    x: F32Ptr,
    w: F32Ptr,
    y: F32Ptr,
    n_tokens: Int,
    dim: Int,
    eps: Float32,
    threads: Int,
) raises:
    """y[t] = x[t] / sqrt(mean(x[t]^2) + eps) * w."""

    def work(t: Int) {imm}:
        var row = x.unsafe_offset(t * dim)
        var acc = SIMD[DType.float32, W](0)
        var i = 0
        while i + W <= dim:
            var v = row.unsafe_load[width=W](i)
            acc += v * v
            i += W
        var ss = acc.reduce_add()
        while i < dim:
            ss += row.unsafe_load(i) * row.unsafe_load(i)
            i += 1
        var inv = 1.0 / sqrt(ss / Float32(dim) + eps)
        var out = y.unsafe_offset(t * dim)
        i = 0
        while i + W <= dim:
            out.unsafe_store(
                i, row.unsafe_load[width=W](i) * inv * w.unsafe_load[width=W](i)
            )
            i += W
        while i < dim:
            out.unsafe_store(i, row.unsafe_load(i) * inv * w.unsafe_load(i))
            i += 1

    parallel_for(work, n_tokens, threads)


def rope_table(
    n_tokens: Int,
    head_dim: Int,
    theta: Float32,
    pos0: Int,
    cos_t: F32Ptr,
    sin_t: F32Ptr,
):
    """cos/sin of p * theta^(-2i/d) for every position p and pair i, in f64.

    LLaDA computes its rotary angles in full precision (`rope_full_precision`),
    so the table is built in f64 and stored as f32.
    """
    var half = head_dim // 2
    for t in range(n_tokens):
        var p = Float64(pos0 + t)
        for i in range(half):
            var ang = p * (
                Float64(theta) ** (-2.0 * Float64(i) / Float64(head_dim))
            )
            cos_t.unsafe_store(t * half + i, Float32(cos(ang)))
            sin_t.unsafe_store(t * half + i, Float32(sin(ang)))


def rope_apply(
    x: F32Ptr,
    n_tokens: Int,
    n_heads: Int,
    head_dim: Int,
    cos_t: F32Ptr,
    sin_t: F32Ptr,
    neox: Bool,
    threads: Int,
) raises:
    """In-place rotary embedding at positions 0..n_tokens-1.

    NORM style (llama.cpp's LLaMA convention, used for LLaDA GGUFs) rotates
    adjacent pairs (2i, 2i+1); the converter un-permutes LLaDA's Q/K so this
    holds. NEOX style (Qwen2, and so Dream) rotates (i, i + d/2). Pair i always
    uses the angle p * theta^(-2i/d).
    """
    var half = head_dim // 2
    var lo_stride = 1 if neox else 2
    var hi_off = half if neox else 1

    def work(t: Int) {imm}:
        for h in range(n_heads):
            var base = x.unsafe_offset((t * n_heads + h) * head_dim)
            for i in range(half):
                var c = cos_t.unsafe_load(t * half + i)
                var s = sin_t.unsafe_load(t * half + i)
                var ia = i * lo_stride
                var a = base.unsafe_load(ia)
                var b = base.unsafe_load(ia + hi_off)
                base.unsafe_store(ia, a * c - b * s)
                base.unsafe_store(ia + hi_off, a * s + b * c)

    parallel_for(work, n_tokens, threads)


def rope_norm(
    x: F32Ptr,
    n_tokens: Int,
    n_heads: Int,
    head_dim: Int,
    theta: Float32,
    pos0: Int,
    neox: Bool = False,
) raises:
    """Convenience form that builds its own table. Tests use this."""
    var half = head_dim // 2
    var c = Floats(n_tokens * half)
    var s = Floats(n_tokens * half)
    rope_table(n_tokens, head_dim, theta, pos0, c.ptr(), s.ptr())
    rope_apply(x, n_tokens, n_heads, head_dim, c.ptr(), s.ptr(), neox, 1)


def softmax_inplace(x: F32Ptr, n: Int):
    var m = x.unsafe_load(0)
    for i in range(1, n):
        m = max(m, x.unsafe_load(i))
    var s = Float32(0)
    for i in range(n):
        var e = exp(x.unsafe_load(i) - m)
        x.unsafe_store(i, e)
        s += e
    var inv = 1.0 / s
    for i in range(n):
        x.unsafe_store(i, x.unsafe_load(i) * inv)


comptime QB = 4
"""Queries per attention work item: each loaded key row feeds this many dot products."""


def attention(
    q: F32Ptr,
    k: F32Ptr,
    v: F32Ptr,
    o: F32Ptr,
    L: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    threads: Int,
    n_keys: Int = -1,
) raises:
    """o = softmax(q k^T / sqrt(d)) v per head, no mask.

    q and o rows are [L, n_heads*d]; k and v rows are [n_keys, n_kv_heads*d],
    n_keys defaulting to L. A K/V cache passes more keys than queries: the
    rows of one block attend to the whole canvas.
    With grouped-query attention, query head h reads KV head
    h // (n_heads / n_kv_heads), as Qwen2 and llama.cpp do.

    A work item is QB queries of one head. Scores are computed key by key,
    each key row loaded once for all QB queries; the weighted sum over V keeps
    its 64-dimension half of the output in registers, so the inner loop is
    loads and FMAs with no store.
    """
    if head_dim % 64 != 0:
        raise Error("attention needs a head dimension that is a multiple of 64")
    var stride = n_heads * head_dim
    var kv_stride = n_kv_heads * head_dim
    var group = n_heads // n_kv_heads
    var scale = Float32(1.0 / sqrt(Float64(head_dim)))
    var NK = L if n_keys < 0 else n_keys
    var nqb = (L + QB - 1) // QB
    var scores = Floats(n_heads * nqb * QB * NK)
    var sp = scores.ptr()

    def work(item: Int) {imm}:
        var h = item // nqb
        var kvh = h // group
        var i0 = (item % nqb) * QB
        var nq = min(QB, L - i0)
        var rows = sp.unsafe_offset(item * QB * NK)
        for j in range(NK):
            var kj = k.unsafe_offset(j * kv_stride + kvh * head_dim)
            var acc = Array[SIMD[DType.float32, W], QB](
                fill=SIMD[DType.float32, W](0)
            )
            for d in range(0, head_dim, W):
                var kv = kj.unsafe_load[width=W](d)
                comptime for t in range(QB):
                    if t < nq:
                        acc[t] += (
                            q.unsafe_offset(
                                (i0 + t) * stride + h * head_dim
                            ).unsafe_load[width=W](d)
                            * kv
                        )
            comptime for t in range(QB):
                if t < nq:
                    rows.unsafe_store(t * NK + j, acc[t].reduce_add() * scale)
        for t in range(nq):
            softmax_inplace(rows.unsafe_offset(t * NK), NK)
        for t in range(nq):
            var row = rows.unsafe_offset(t * NK)
            var oi = o.unsafe_offset((i0 + t) * stride + h * head_dim)
            for half in range(0, head_dim, 64):
                var acc = Array[SIMD[DType.float32, W], 8](
                    fill=SIMD[DType.float32, W](0)
                )
                for j in range(NK):
                    var pj = row.unsafe_load(j)
                    var vj = v.unsafe_offset(
                        j * kv_stride + kvh * head_dim + half
                    )
                    comptime for c in range(8):
                        acc[c] += pj * vj.unsafe_load[width=W](c * W)
                comptime for c in range(8):
                    oi.unsafe_store(half + c * W, acc[c])

    parallel_for(work, n_heads * nqb, threads)
    _ = scores^


def silu_mul(g: F32Ptr, u: F32Ptr, n: Int, threads: Int) raises:
    """g = silu(g) * u, in place, eight lanes at a time."""
    var chunk = 4096
    var items = (n + chunk - 1) // chunk

    def work(c: Int) {imm}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        var i = lo
        while i + W <= hi:
            var a = g.unsafe_load[width=W](i)
            g.unsafe_store(i, a / (1.0 + exp(-a)) * u.unsafe_load[width=W](i))
            i += W
        while i < hi:
            var a = g.unsafe_load(i)
            g.unsafe_store(i, a / (1.0 + exp(-a)) * u.unsafe_load(i))
            i += 1

    parallel_for(work, items, threads)


def add_inplace(a: F32Ptr, b: F32Ptr, n: Int):
    var i = 0
    while i + W <= n:
        a.unsafe_store(i, a.unsafe_load[width=W](i) + b.unsafe_load[width=W](i))
        i += W
    while i < n:
        a.unsafe_store(i, a.unsafe_load(i) + b.unsafe_load(i))
        i += 1


def add_bias(x: F32Ptr, b: F32Ptr, rows: Int, dim: Int):
    """x[r] += b for every row r."""
    for r in range(rows):
        var row = x.unsafe_offset(r * dim)
        var i = 0
        while i + W <= dim:
            row.unsafe_store(
                i, row.unsafe_load[width=W](i) + b.unsafe_load[width=W](i)
            )
            i += W
        while i < dim:
            row.unsafe_store(i, row.unsafe_load(i) + b.unsafe_load(i))
            i += 1


def rope_table_factors(
    n_tokens: Int,
    head_dim: Int,
    theta: Float32,
    pos0: Int,
    factors: F32Ptr,
    use_factors: Bool,
    cos_t: F32Ptr,
    sin_t: F32Ptr,
):
    """`rope_table` with a per-pair divisor: angle = p * theta^(-2i/d) / factors[i].

    This is llama.cpp's `rope_freqs` (freq_factors): a factor of 1e30 leaves a
    pair unrotated. `factors` holds head_dim / 2 values and is read only when
    `use_factors` is set. Built in f64 like `rope_table`; the branch accumulates
    the angle in f32, which differs by under 1e-4 rad at position 300.
    """
    var half = head_dim // 2
    for t in range(n_tokens):
        var p = Float64(pos0 + t)
        for i in range(half):
            var ang = p * (
                Float64(theta) ** (-2.0 * Float64(i) / Float64(head_dim))
            )
            if use_factors:
                ang = ang / Float64(factors.unsafe_load(i))
            cos_t.unsafe_store(t * half + i, Float32(cos(ang)))
            sin_t.unsafe_store(t * half + i, Float32(sin(ang)))


def attention_ranges(
    q: F32Ptr,
    k: F32Ptr,
    v: F32Ptr,
    o: F32Ptr,
    lo: I32Ptr,
    hi: I32Ptr,
    n_q: Int,
    n_keys: Int,
    n_heads: Int,
    n_kv_heads: Int,
    head_dim: Int,
    scale: Float32,
    threads: Int,
) raises:
    """o = softmax(scale * q k^T) v per head, over keys [lo[i], hi[i]) of query i.

    q and o rows are [n_q, n_heads * head_dim]; k and v rows are
    [n_keys, n_kv_heads * head_dim]. Query head h reads KV head
    h // (n_heads / n_kv_heads). `scale` is the argument, not 1/sqrt(d):
    DiffusionGemma passes 1.0. Every query needs at least one key, so the
    softmax is never over nothing; the head size is a multiple of 8.
    """
    if head_dim % W != 0 or n_heads % n_kv_heads != 0:
        raise Error("attention needs a head size that is a multiple of 8")
    for i in range(n_q):
        var a = Int(lo.unsafe_load(i))
        var b = Int(hi.unsafe_load(i))
        if a < 0 or b > n_keys or a >= b:
            raise Error(
                "attention key range ["
                + String(a)
                + ", "
                + String(b)
                + ") of query "
                + String(i)
                + " is empty or outside "
                + String(n_keys)
                + " keys"
            )
    var stride = n_heads * head_dim
    var kv_stride = n_kv_heads * head_dim
    var group = n_heads // n_kv_heads
    var scores = Floats(n_heads * n_q * n_keys)
    var sp = scores.ptr()

    def work(item: Int) {imm}:
        var h = item // n_q
        var i = item % n_q
        var kvh = h // group
        var a = Int(lo.unsafe_load(i))
        var b = Int(hi.unsafe_load(i))
        var qi = q.unsafe_offset(i * stride + h * head_dim)
        var row = sp.unsafe_offset(item * n_keys)
        var mx = Float32(-3.4028234663852886e38)
        for j in range(a, b):
            var kj = k.unsafe_offset(j * kv_stride + kvh * head_dim)
            var acc = SIMD[DType.float32, W](0)
            for d in range(0, head_dim, W):
                acc += qi.unsafe_load[width=W](d) * kj.unsafe_load[width=W](d)
            var sc = acc.reduce_add() * scale
            row.unsafe_store(j, sc)
            mx = max(mx, sc)
        var z = Float32(0)
        for j in range(a, b):
            var e = exp(row.unsafe_load(j) - mx)
            row.unsafe_store(j, e)
            z += e
        var inv = 1.0 / z
        var oi = o.unsafe_offset(i * stride + h * head_dim)
        for d in range(0, head_dim, W):
            var acc = SIMD[DType.float32, W](0)
            for j in range(a, b):
                acc += (
                    row.unsafe_load(j)
                    * inv
                    * v.unsafe_load[width=W](j * kv_stride + kvh * head_dim + d)
                )
            oi.unsafe_store(d, acc)

    parallel_for(work, n_heads * n_q, threads)
    _ = scores^
