"""A full synthetic MoE layer, including routing and grouped expert GEMMs.

Usage: build/bench_moe THREADS TOKENS REPS DOWN_TYPE [HIDDEN FFN EXPERTS TOP_K]
DOWN_TYPE is 6 (Q5_0) or 8 (Q8_0). The default dimensions are 2816, 704,
128 and 8; bench/run_moe.py supplies the 256-token measurement defaults.
Weight construction and packing are outside the measured region.
"""

from std.math import isfinite
from std.sys import argv
from std.time import perf_counter_ns

from mojo_dllm.formats.gguf import (
    GGML_Q4_K,
    GGML_Q5_0,
    GGML_Q8_0,
    ggml_row_bytes,
)
from mojo_dllm.kernels.moe import (
    Routing,
    ExpertWeights,
    MoEWorkspace,
    route_tokens,
    grouped_experts,
)
from mojo_dllm.sys.mem import Bytes, Floats


def _next(mut state: UInt64) -> UInt64:
    state = state * 6364136223846793005 + 1442695040888963407
    return state


def _weights(
    kind: Int, rows: Int, width: Int, experts: Int, seed: Int
) raises -> Bytes:
    var block_weights = 256 if kind == GGML_Q4_K else 32
    var block_bytes = ggml_row_bytes(kind, block_weights)
    var blocks = experts * rows * (width // block_weights)
    var raw = Bytes(blocks * block_bytes, zero=False)
    var data = raw.u8()
    var state = UInt64(seed)
    for b in range(blocks):
        var start = b * block_bytes
        for i in range(block_bytes):
            data.unsafe_store(start + i, UInt8(_next(state) >> 56))
        # Finite, nonzero f16 scales. Random bytes must never become NaNs.
        data.unsafe_store(start, 0)
        if kind == GGML_Q4_K:
            data.unsafe_store(start + 1, 0x0C)  # d = 2^-12
            data.unsafe_store(start + 2, 0)
            data.unsafe_store(start + 3, 0x0C)  # dmin = 2^-12
        elif kind == GGML_Q5_0:
            data.unsafe_store(start + 1, 0x1C)  # d = 2^-8
        else:
            data.unsafe_store(start + 1, 0x14)  # d = 2^-10
    return raw^


def main() raises:
    var args = argv()
    if len(args) != 5 and len(args) != 9:
        raise Error(
            "usage: bench_moe THREADS TOKENS REPS DOWN_TYPE [HIDDEN FFN EXPERTS"
            " TOP_K]"
        )
    var threads = Int(args[1])
    var tokens = Int(args[2])
    var reps = Int(args[3])
    var down_kind = Int(args[4])
    var hidden = 2816
    var ffn = 704
    var experts = 128
    var top_k = 8
    if len(args) == 9:
        hidden = Int(args[5])
        ffn = Int(args[6])
        experts = Int(args[7])
        top_k = Int(args[8])
    if threads <= 0 or tokens <= 0 or reps <= 0:
        raise Error("threads, tokens and reps must be positive")
    if down_kind != GGML_Q5_0 and down_kind != GGML_Q8_0:
        raise Error("DOWN_TYPE must be 6 (Q5_0) or 8 (Q8_0)")
    if hidden <= 0 or hidden % 256 != 0 or ffn <= 0 or ffn % 32 != 0:
        raise Error("HIDDEN must be a positive multiple of 256 and FFN of 32")
    if experts <= 0 or top_k <= 0 or top_k > experts:
        raise Error("EXPERTS and TOP_K must be positive, with TOP_K <= EXPERTS")

    var setup_start = perf_counter_ns()
    var gate_up = _weights(GGML_Q4_K, 2 * ffn, hidden, experts, 630001)
    var down = _weights(down_kind, hidden, ffn, experts, 630002)
    var scales = Floats(experts)
    for e in range(experts):
        scales[e] = 0.75 + Float32(e % 17) / 32.0
    var weights = ExpertWeights(
        gate_up.addr,
        GGML_Q4_K,
        down.addr,
        down_kind,
        hidden,
        ffn,
        experts,
        scales.ptr(),
    )
    var routing = Routing(tokens, experts, top_k)
    var workspace = MoEWorkspace(tokens, weights)
    var x = Floats(tokens * hidden)
    var router_scale = Floats(hidden)
    var router_matrix = Floats(experts * hidden)
    var out = Floats(tokens * hidden)
    var state = UInt64(630003)
    for i in range(x.n):
        x[i] = Float32(Int(_next(state) >> 40) - 8388608) / 8388608.0
    for d in range(hidden):
        router_scale[d] = 0.875 + Float32(d % 13) / 64.0
    for i in range(router_matrix.n):
        router_matrix[i] = (
            Float32(Int(_next(state) >> 40) - 8388608) / 8388608.0
        )
    var setup_seconds = Float64(perf_counter_ns() - setup_start) / 1e9
    var packed_weight_bytes = 0
    for e in range(experts):
        packed_weight_bytes += (
            weights.gate_up[e].data.size + weights.down[e].data.size
        )
    print(
        '{"kind":"setup","threads":',
        threads,
        ',"tokens":',
        tokens,
        ',"hidden":',
        hidden,
        ',"ffn":',
        ffn,
        ',"experts":',
        experts,
        ',"top_k":',
        top_k,
        ',"down_type":',
        down_kind,
        ',"setup_seconds":',
        setup_seconds,
        ',"raw_weight_bytes":',
        gate_up.size + down.size,
        ',"packed_weight_bytes":',
        packed_weight_bytes,
        "}",
        sep="",
    )
    # One full warmup, then every measured sample; do not report only the best.
    for rep in range(-1, reps):
        var start = perf_counter_ns()
        route_tokens(
            x.ptr(),
            hidden,
            router_scale.ptr(),
            router_matrix.ptr(),
            1e-6,
            routing,
            threads,
        )
        grouped_experts(
            weights, x.ptr(), routing, workspace, out.ptr(), threads
        )
        var seconds = Float64(perf_counter_ns() - start) / 1e9
        var checksum = Float64(0)
        var max_abs = Float32(0)
        for i in range(out.n):
            if not isfinite(out[i]):
                raise Error("non-finite MoE output at " + String(i))
            checksum += Float64(out[i])
            max_abs = max(max_abs, abs(out[i]))
        print(
            '{"kind":"sample","index":',
            rep,
            ',"warmup":',
            "true" if rep == -1 else "false",
            ',"seconds":',
            seconds,
            ',"checksum":',
            checksum,
            ',"max_abs":',
            max_abs,
            "}",
            sep="",
        )
    var counts = List[Int]()
    for _ in range(experts):
        counts.append(0)
    for i in range(tokens * top_k):
        counts[Int(routing.ids.i32().unsafe_load(i))] += 1
    var active = 0
    var max_tokens = 0
    for count in counts:
        active += Int(count > 0)
        max_tokens = max(max_tokens, count)
    print(
        '{"kind":"routing","active_experts":',
        active,
        ',"max_tokens_per_expert":',
        max_tokens,
        ',"token_expert_pairs":',
        tokens * top_k,
        "}",
        sep="",
    )
    # These owners back untracked pointers passed into the layer. Keep them
    # alive through the final call, including its parallel callbacks.
    _ = gate_up^
    _ = down^
    _ = scales^
    _ = x^
    _ = router_scale^
    _ = router_matrix^
    _ = out^
    _ = routing^
