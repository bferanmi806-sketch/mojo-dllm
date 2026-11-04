"""CPU routing and grouped experts for a diffusion MoE layer.

Router weights are expert-major [experts, hidden]. Each raw expert slice
stores [2 * ffn, hidden] gate/up rows followed, in a separate tensor, by
[hidden, ffn] down rows. Gate rows precede up rows in the fused projection.
Weights are repacked once; the workspace is reused between denoising steps.
"""

from std.math import exp, sqrt, tanh

from mojo_dllm.formats.gguf import ggml_row_bytes
from mojo_dllm.kernels.packed import PackedMatrix, pgemm
from mojo_dllm.kernels.qgemm import QActs, quantize_acts
from mojo_dllm.sys.mem import Bytes, Floats, F32Ptr
from mojo_dllm.sys.parallel import parallel_for


@always_inline
def gelu_tanh(x: Float32) -> Float32:
    """PyTorch's GELU approximate='tanh', evaluated in f32."""
    return (
        0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    )


struct Routing(Movable):
    var n_tokens: Int
    var n_experts: Int
    var top_k: Int
    var ids: Bytes
    """Int32 expert ids, token-major [n_tokens, top_k]."""
    var weights: Floats
    """Corresponding probabilities, renormalized over the selected experts."""

    def __init__(out self, n_tokens: Int, n_experts: Int, top_k: Int) raises:
        if n_tokens < 0 or n_experts <= 0 or top_k <= 0 or top_k > n_experts:
            raise Error(
                "invalid MoE routing dimensions: tokens="
                + String(n_tokens)
                + ", experts="
                + String(n_experts)
                + ", top_k="
                + String(top_k)
            )
        self.n_tokens = n_tokens
        self.n_experts = n_experts
        self.top_k = top_k
        self.ids = Bytes(n_tokens * top_k * 4)
        self.weights = Floats(n_tokens * top_k)


def route_tokens(
    x: F32Ptr,
    hidden: Int,
    router_scale: F32Ptr,
    router_matrix: F32Ptr,
    eps: Float32,
    mut routing: Routing,
    threads: Int = 1,
) raises:
    """Unweighted RMSNorm / sqrt(hidden), scale, F32 projection, softmax/top-k.

    Equal probabilities select the smaller expert id. Selection is without
    replacement, so an expert receives at most one row from each token.
    """
    if hidden <= 0 or eps <= 0:
        raise Error("MoE router needs positive hidden size and RMS epsilon")
    var normalized = Floats(routing.n_tokens * hidden)
    var probabilities = Floats(routing.n_tokens * routing.n_experts)
    var xp = normalized.ptr()
    var pp = probabilities.ptr()
    var ids = routing.ids.i32()
    var wp = routing.weights.ptr()
    var experts = routing.n_experts
    var top_k = routing.top_k

    def work(t: Int) {imm}:
        var source = x.unsafe_offset(t * hidden)
        var norm = xp.unsafe_offset(t * hidden)
        var sumsq = SIMD[DType.float32, 8](0)
        var d = 0
        while d + 8 <= hidden:
            var v = source.unsafe_load[width=8](d)
            sumsq += v * v
            d += 8
        var ss = sumsq.reduce_add()
        while d < hidden:
            var v = source.unsafe_load(d)
            ss += v * v
            d += 1
        var inv = 1.0 / sqrt(ss / Float32(hidden) + eps) / sqrt(Float32(hidden))
        d = 0
        while d + 8 <= hidden:
            norm.unsafe_store(
                d,
                source.unsafe_load[width=8](d)
                * inv
                * router_scale.unsafe_load[width=8](d),
            )
            d += 8
        while d < hidden:
            norm.unsafe_store(
                d, source.unsafe_load(d) * inv * router_scale.unsafe_load(d)
            )
            d += 1
        var probs = pp.unsafe_offset(t * experts)
        var maximum = Float32(-3.4028234663852886e38)
        for e in range(experts):
            var row = router_matrix.unsafe_offset(e * hidden)
            var acc = SIMD[DType.float32, 8](0)
            d = 0
            while d + 8 <= hidden:
                acc += norm.unsafe_load[width=8](d) * row.unsafe_load[width=8](
                    d
                )
                d += 8
            var score = acc.reduce_add()
            while d < hidden:
                score += norm.unsafe_load(d) * row.unsafe_load(d)
                d += 1
            probs.unsafe_store(e, score)
            maximum = max(maximum, score)
        var total = Float32(0)
        for e in range(experts):
            var p = exp(probs.unsafe_load(e) - maximum)
            probs.unsafe_store(e, p)
            total += p
        for e in range(experts):
            probs.unsafe_store(e, probs.unsafe_load(e) / total)
        var selected_total = Float32(0)
        for s in range(top_k):
            var best = 0
            for e in range(1, experts):
                if probs.unsafe_load(e) > probs.unsafe_load(best):
                    best = e
            var p = probs.unsafe_load(best)
            ids.unsafe_store(t * top_k + s, Int32(best))
            wp.unsafe_store(t * top_k + s, p)
            selected_total += p
            probs.unsafe_store(best, -1)
        for s in range(top_k):
            var i = t * top_k + s
            wp.unsafe_store(i, wp.unsafe_load(i) / selected_total)

    parallel_for(work, routing.n_tokens, threads)
    # The pointers above have untracked origins; keep their owners alive
    # through the parallel callback, just as attention's score scratch does.
    _ = normalized^
    _ = probabilities^


struct ExpertWeights(Movable):
    var hidden: Int
    var ffn: Int
    var n_experts: Int
    var gate_up: List[PackedMatrix]
    var down: List[PackedMatrix]
    var scales: Floats

    def __init__(
        out self,
        gate_up_addr: Int,
        gate_up_kind: Int,
        down_addr: Int,
        down_kind: Int,
        hidden: Int,
        ffn: Int,
        n_experts: Int,
        scales: F32Ptr,
    ) raises:
        if hidden <= 0 or ffn <= 0 or n_experts <= 0:
            raise Error("MoE expert dimensions must be positive")
        self.hidden = hidden
        self.ffn = ffn
        self.n_experts = n_experts
        self.gate_up = List[PackedMatrix]()
        self.down = List[PackedMatrix]()
        self.scales = Floats(n_experts)
        var gate_stride = 2 * ffn * ggml_row_bytes(gate_up_kind, hidden)
        var down_stride = hidden * ggml_row_bytes(down_kind, ffn)
        for e in range(n_experts):
            self.gate_up.append(
                PackedMatrix(
                    gate_up_addr + e * gate_stride,
                    gate_up_kind,
                    2 * ffn,
                    hidden,
                )
            )
            self.down.append(
                PackedMatrix(
                    down_addr + e * down_stride, down_kind, hidden, ffn
                )
            )
            self.scales[e] = scales.unsafe_load(e)


struct MoEWorkspace(Movable):
    var max_tokens: Int
    var hidden: Int
    var ffn: Int
    var n_experts: Int
    var pairs: Bytes
    var input: Floats
    var gate_up: Floats
    var activated: Floats
    var output: Floats
    var qa_gate: QActs
    var qa_down: QActs

    def __init__(out self, n_tokens: Int, weights: ExpertWeights) raises:
        if n_tokens < 0:
            raise Error("MoE workspace token capacity must be nonnegative")
        self.max_tokens = n_tokens
        self.hidden = weights.hidden
        self.ffn = weights.ffn
        self.n_experts = weights.n_experts
        self.pairs = Bytes(n_tokens * weights.n_experts * 4)
        self.input = Floats(n_tokens * weights.hidden)
        self.gate_up = Floats(n_tokens * weights.ffn * 2)
        self.activated = Floats(n_tokens * weights.ffn)
        self.output = Floats(n_tokens * weights.hidden)
        self.qa_gate = QActs(
            n_tokens, weights.hidden, weights.gate_up[0].block_size
        )
        self.qa_down = QActs(n_tokens, weights.ffn, weights.down[0].block_size)


def grouped_experts(
    weights: ExpertWeights,
    x: F32Ptr,
    routing: Routing,
    mut ws: MoEWorkspace,
    dst: F32Ptr,
    threads: Int,
) raises:
    """Counting-sort pairs, gather, batched gate/up and down GEMMs, scatter-add.

    Empty experts launch no GEMM. Buffers hold the complete canvas for an
    expert, including the case where every token chooses the same expert.
    `dst` must not overlap `x`; it is replaced, not added to its old contents.
    """
    if (
        routing.n_experts != weights.n_experts
        or ws.n_experts != weights.n_experts
        or ws.hidden != weights.hidden
        or ws.ffn != weights.ffn
        or routing.n_tokens > ws.max_tokens
    ):
        raise Error("MoE routing or workspace does not match expert dimensions")
    var n = routing.n_tokens
    var hidden = weights.hidden
    var ffn = weights.ffn
    var experts = weights.n_experts
    var top_k = routing.top_k
    var ids = routing.ids.i32()
    var counts = List[Int]()
    for _ in range(experts):
        counts.append(0)
    for t in range(n):
        for s in range(top_k):
            var e = Int(ids.unsafe_load(t * top_k + s))
            if e < 0 or e >= experts:
                raise Error("MoE routing expert id out of range: " + String(e))
            for previous in range(s):
                if Int(ids.unsafe_load(t * top_k + previous)) == e:
                    raise Error(
                        "MoE routing repeats expert "
                        + String(e)
                        + " for token "
                        + String(t)
                    )
            counts[e] += 1
    var offsets = List[Int]()
    offsets.append(0)
    for e in range(experts):
        offsets.append(offsets[e] + counts[e])
        counts[e] = 0
    var pairs = ws.pairs.i32()
    for pair in range(n * top_k):
        var e = Int(ids.unsafe_load(pair))
        pairs.unsafe_store(offsets[e] + counts[e], Int32(pair))
        counts[e] += 1
    for i in range(n * hidden):
        dst.unsafe_store(i, 0)
    for e in range(experts):
        var rows = counts[e]
        if rows == 0:
            continue
        for r in range(rows):
            var pair = Int(pairs.unsafe_load(offsets[e] + r))
            var t = pair // top_k
            for d in range(hidden):
                ws.input[r * hidden + d] = x.unsafe_load(t * hidden + d)
        quantize_acts(ws.input.ptr(), rows, hidden, ws.qa_gate)
        pgemm(
            weights.gate_up[e],
            ws.qa_gate,
            rows,
            ws.gate_up.ptr(),
            2 * ffn,
            threads,
        )
        for r in range(rows):
            for f in range(ffn):
                ws.activated[r * ffn + f] = (
                    gelu_tanh(ws.gate_up[r * 2 * ffn + f])
                    * ws.gate_up[r * 2 * ffn + ffn + f]
                )
        quantize_acts(ws.activated.ptr(), rows, ffn, ws.qa_down)
        pgemm(
            weights.down[e], ws.qa_down, rows, ws.output.ptr(), hidden, threads
        )
        for r in range(rows):
            var pair = Int(pairs.unsafe_load(offsets[e] + r))
            var t = pair // top_k
            var scale = routing.weights[pair] * weights.scales[e]
            for d in range(hidden):
                var i = t * hidden + d
                dst.unsafe_store(
                    i, dst.unsafe_load(i) + scale * ws.output[r * hidden + d]
                )
