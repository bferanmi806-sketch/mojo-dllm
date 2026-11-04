"""Actual Q6_K weights: original tile dispatch versus device-aware dispatch.

Build with scripts/build.sh bench/bench_gpu_q6_dispatch.mojo build/bench-q6.
Run build/bench-q6 MODEL.gguf TOKENS on an idle GPU. JSON lines include
interleaved per-shape timings and output agreement. Round zero is warmup.
Timings include host enqueue and synchronization, not just GPU kernel time.
"""
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from mojo_dllm.sys.mem import Floats, Bytes, F32Ptr, U8Ptr
from mojo_dllm.formats.gguf import GGUFFile, GGML_Q6_K
from mojo_dllm.gpu.kernels import (
    GpuQActs,
    dev_f32,
    dev_i8,
    dev_u8,
    stage_matrix,
    staged_matrix_bytes,
    quantize_gpu,
    gemm_gpu,
)
from mojo_dllm.gpu.kernels import _launch, _pick_bn


def measure(
    ctx: DeviceContext,
    w: U8Ptr,
    N: Int,
    K: Int,
    qa: GpuQActs,
    T: Int,
    o: F32Ptr,
    reps: Int,
    candidate: Bool,
) raises -> Float64:
    ctx.synchronize()
    var start = perf_counter_ns()
    for _ in range(reps):
        if candidate:
            gemm_gpu(ctx, GGML_Q6_K, w, N, K, qa, T, o, N)
        elif _pick_bn(T) == 32:
            _launch[32](ctx, GGML_Q6_K, w, N, K, qa, T, o, N)
        elif _pick_bn(T) == 64:
            _launch[64](ctx, GGML_Q6_K, w, N, K, qa, T, o, N)
        else:
            _launch[80](ctx, GGML_Q6_K, w, N, K, qa, T, o, N)
    ctx.synchronize()
    return Float64(perf_counter_ns() - start) / 1e6 / Float64(reps)


def main() raises:
    var a = argv()
    var gguf = GGUFFile(a[1])
    var T = Int(a[2])
    var ctx = DeviceContext()
    var seen = List[String]()
    for li in range(gguf.get_int(gguf.architecture() + ".block_count") + 1):
        var last = li == gguf.get_int(gguf.architecture() + ".block_count")
        for name in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ]:
            if last and name != "attn_q":
                continue
            var tensor = gguf.tensor(
                "output.weight" if last else "blk."
                + String(li)
                + "."
                + name
                + ".weight"
            )
            if tensor.ggml_type != GGML_Q6_K:
                continue
            var N = tensor.n_rows()
            var K = tensor.ne0
            var rows = 32 if last else T
            var key = String(N) + "x" + String(K) + "@" + String(rows)
            var exists = False
            for old in seen:
                if old == key:
                    exists = True
            if exists:
                continue
            seen.append(key)
            var packed = Bytes(staged_matrix_bytes(GGML_Q6_K, N, K))
            stage_matrix(gguf.data_addr(tensor), GGML_Q6_K, N, K, packed.u8())
            var w = ctx.enqueue_create_buffer[DType.uint8](packed.size)
            ctx.enqueue_copy(w, packed.u8())
            var x = Floats(rows * K)
            for i in range(x.n):
                x[i] = Float32((i * 37) % 101 - 50) / 50.0
            var dx = ctx.enqueue_create_buffer[DType.float32](x.n)
            ctx.enqueue_copy(dx, x.ptr())
            var qa = GpuQActs(ctx, rows, K)
            quantize_gpu(ctx, dev_f32(dx), rows, K, qa)
            var out = ctx.enqueue_create_buffer[DType.float32](rows * N)
            gemm_gpu(ctx, GGML_Q6_K, dev_u8(w), N, K, qa, rows, dev_f32(out), N)
            var reference = Floats(rows * N)
            var actual = Floats(rows * N)
            ctx.enqueue_copy(reference.ptr(), out)
            ctx.synchronize()
            for round in range(5):
                for order in range(2):
                    var candidate = (order + round) % 2 == 1
                    _ = measure(
                        ctx,
                        dev_u8(w),
                        N,
                        K,
                        qa,
                        rows,
                        dev_f32(out),
                        5,
                        candidate,
                    )
                    var ms = measure(
                        ctx,
                        dev_u8(w),
                        N,
                        K,
                        qa,
                        rows,
                        dev_f32(out),
                        30,
                        candidate,
                    )
                    ctx.enqueue_copy(actual.ptr(), out)
                    ctx.synchronize()
                    var error = Float32(0)
                    for i in range(actual.n):
                        error = max(error, abs(actual[i] - reference[i]))
                    print(
                        '{"N":',
                        N,
                        ',"K":',
                        K,
                        ',"T":',
                        rows,
                        ',"candidate":',
                        Int(candidate),
                        ',"round":',
                        round,
                        ',"ms":',
                        ms,
                        ',"max_abs":',
                        error,
                        "}",
                    )
