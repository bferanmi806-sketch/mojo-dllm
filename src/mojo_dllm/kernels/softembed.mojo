"""Self-conditioning's soft embedding: softmax of the previous logits times the table.

DiffusionGemma feeds the previous step's canvas logits back in as
softmax(logits / temperature) @ embedding, a [rows, vocab] by [vocab, hidden]
product with vocab = 262,144, as large as the LM head. The table is kept
transposed once ([hidden, vocab], f32) so both operands run along the
contiguous vocab axis. The product splits vocab into ranges; each range reads
its slice of the probabilities and of the table once, into a private partial
sum, and the partial sums are added at the end.
"""

from std.math import exp

from mojo_dllm.formats.gguf import ggml_row_bytes
from mojo_dllm.quant.kquants import dequant_row
from mojo_dllm.sys.mem import F32Ptr, Floats
from mojo_dllm.sys.parallel import parallel_for

comptime W = 8
comptime RT = 4
"""Probability rows per register tile."""
comptime ET = 4
"""Embedding dimensions per register tile."""
comptime KC = 1024
"""Vocabulary entries per cache block: the probabilities of it stay in L2."""
comptime ROWS_PER_BLOCK = 64
"""Table rows decoded together, so each transposed write is 256 contiguous bytes."""
comptime MAX_PARTS = 48


struct EmbeddingT(Movable):
    """The embedding table as [hidden, vocab] f32, built once from the file's rows.
    """

    var vocab: Int
    var hidden: Int
    var data: Floats

    def __init__(
        out self,
        addr: Int,
        ggml_type: Int,
        vocab: Int,
        hidden: Int,
        threads: Int,
    ) raises:
        if vocab <= 0 or hidden <= 0 or vocab % W != 0:
            raise Error(
                "the transposed embedding needs a vocabulary that is a"
                " multiple of 8, got "
                + String(vocab)
            )
        self.vocab = vocab
        self.hidden = hidden
        self.data = Floats(vocab * hidden)
        var row_bytes = ggml_row_bytes(ggml_type, hidden)
        var dp = self.data.ptr()
        var blocks = (vocab + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

        def work(b: Int) raises {imm}:
            var v0 = b * ROWS_PER_BLOCK
            var n = min(ROWS_PER_BLOCK, vocab - v0)
            var tmp = Floats(ROWS_PER_BLOCK * hidden)
            for i in range(n):
                dequant_row(
                    ggml_type,
                    addr + (v0 + i) * row_bytes,
                    tmp.ptr().unsafe_offset(i * hidden),
                    hidden,
                )
            for e in range(hidden):
                for i in range(n):
                    dp.unsafe_store(e * vocab + v0 + i, tmp[i * hidden + e])

        parallel_for_raising(work, blocks, threads)


def parallel_for_raising[
    F: def(Int) raises -> None
](func: F, n_items: Int, threads: Int) raises:
    """`parallel_for` for work that can raise: the first error is rethrown."""
    var failed = Floats(1)
    var fp = failed.ptr()

    def guarded(i: Int) {imm}:
        try:
            func(i)
        except:
            fp.unsafe_store(0, 1.0)

    parallel_for(guarded, n_items, threads)
    if failed[0] != 0.0:
        raise Error("a worker failed while decoding the embedding table")


def softmax_rows(
    x: F32Ptr, rows: Int, width: Int, inv_temp: Float32, threads: Int
) raises:
    """In place, each row becomes softmax(row * inv_temp)."""

    def work(r: Int) {imm}:
        var row = x.unsafe_offset(r * width)
        var mx = Float32(-3.4028234663852886e38)
        for i in range(width):
            mx = max(mx, row.unsafe_load(i) * inv_temp)
        var total = Float32(0)
        for i in range(width):
            var e = exp(row.unsafe_load(i) * inv_temp - mx)
            row.unsafe_store(i, e)
            total += e
        var inv = 1.0 / total
        for i in range(width):
            row.unsafe_store(i, row.unsafe_load(i) * inv)

    parallel_for(work, rows, threads)


def soft_embed(
    probs: F32Ptr, embt: EmbeddingT, dst: F32Ptr, rows: Int, threads: Int
) raises:
    """dst[r, e] = sum over the vocabulary of probs[r, v] * table[v, e]."""
    var vocab = embt.vocab
    var hidden = embt.hidden
    var parts = min(MAX_PARTS, vocab // W)
    var per = ((vocab + parts - 1) // parts + W - 1) // W * W
    parts = (vocab + per - 1) // per
    var partial = Floats(parts * rows * hidden)
    var pp = partial.ptr()
    var et = embt.data.ptr()

    def work(part: Int) {imm}:
        var lo = part * per
        var hi = min(lo + per, vocab)
        var acc_out = pp.unsafe_offset(part * rows * hidden)
        for i in range(rows * hidden):
            acc_out.unsafe_store(i, 0)
        var k0 = lo
        while k0 < hi:
            var k1 = min(k0 + KC, hi)
            var e0 = 0
            while e0 < hidden:
                var ne = min(ET, hidden - e0)
                var r0 = 0
                while r0 < rows:
                    var nr = min(RT, rows - r0)
                    var acc = Array[SIMD[DType.float32, W], RT * ET](
                        fill=SIMD[DType.float32, W](0)
                    )
                    for k in range(k0, k1, W):
                        var pv = Array[SIMD[DType.float32, W], RT](
                            fill=SIMD[DType.float32, W](0)
                        )
                        comptime for r in range(RT):
                            if r < nr:
                                pv[r] = probs.unsafe_load[width=W](
                                    (r0 + r) * vocab + k
                                )
                        comptime for e in range(ET):
                            if e < ne:
                                var ev = et.unsafe_load[width=W](
                                    (e0 + e) * vocab + k
                                )
                                comptime for r in range(RT):
                                    acc[r * ET + e] += pv[r] * ev
                    comptime for r in range(RT):
                        comptime for e in range(ET):
                            if r < nr and e < ne:
                                var at = (r0 + r) * hidden + e0 + e
                                acc_out.unsafe_store(
                                    at,
                                    acc_out.unsafe_load(at)
                                    + acc[r * ET + e].reduce_add(),
                                )
                    r0 += RT
                e0 += ET
            k0 = k1

    parallel_for(work, parts, threads)
    for i in range(rows * hidden):
        var total = Float32(0)
        for p in range(parts):
            total += pp.unsafe_load(p * rows * hidden + i)
        dst.unsafe_store(i, total)
    _ = partial^
