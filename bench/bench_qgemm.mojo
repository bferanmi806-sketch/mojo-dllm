"""Throughput of the quantized GEMM on LLaDA-8B layer shapes.

Usage: mojo run -I src bench/bench_qgemm.mojo [threads] [tokens]

Weight bytes are synthetic (timing does not depend on their values). Reports
effective GOP/s counting one multiply-add as two operations.
"""

from std.sys import argv
from std.time import perf_counter_ns

from mojo_dllm.formats.gguf import (
    GGML_Q4_K,
    GGML_Q6_K,
    ggml_row_bytes,
    ggml_type_name,
)
from mojo_dllm.kernels.qgemm import QActs, quantize_acts, qgemm
from mojo_dllm.kernels.packed import PackedMatrix, pgemm
from mojo_dllm.sys.mem import Bytes, Floats


def bench(N: Int, K: Int, w_type: Int, L: Int, threads: Int) raises:
    var w = Bytes(N * ggml_row_bytes(w_type, K))
    var p = w.u8()
    for i in range(w.size):
        p.unsafe_store(i, UInt8((i * 131 + 7) % 251))
    var x = Floats(L * K)
    for i in range(L * K):
        x[i] = Float32((i * 37) % 101 - 50) / 50.0
    var qa = QActs(L, K)
    var out = Floats(L * N)
    var pm = PackedMatrix(w.addr, w.size, w_type, N, K)
    quantize_acts(x.ptr(), L, K, qa)
    pgemm(pm, qa, L, out.ptr(), N, threads)
    var best = Float64(1e30)
    for _ in range(5):
        var t0 = perf_counter_ns()
        quantize_acts(x.ptr(), L, K, qa)
        pgemm(pm, qa, L, out.ptr(), N, threads)
        var dt = Float64(perf_counter_ns() - t0) / 1e9
        best = min(best, dt)
    var gops = 2.0 * Float64(N) * Float64(K) * Float64(L) / best / 1e9
    print(
        ggml_type_name(w_type),
        " N=",
        N,
        " K=",
        K,
        " L=",
        L,
        " threads=",
        threads,
        "  ",
        best * 1000.0,
        " ms  ",
        gops,
        " GOP/s",
        sep="",
    )


def main() raises:
    var args = argv()
    var threads = 8
    var L = 128
    if len(args) > 1:
        threads = Int(args[1])
    if len(args) > 2:
        L = Int(args[2])
    bench(4096, 4096, GGML_Q4_K, L, threads)
    bench(12288, 4096, GGML_Q4_K, L, threads)
    bench(4096, 12288, GGML_Q6_K, L, threads)
