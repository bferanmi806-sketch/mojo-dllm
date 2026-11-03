"""Exploratory tile timings. Same inputs, interleaved tile widths; not a correctness check."""
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from mojo_dllm.gpu.kernels import (
    GpuQActs,
    dev_f32,
    dev_u8,
    quantize_gpu,
    staged_matrix_bytes,
    _launch,
)
from mojo_dllm.sys.mem import Bytes, Floats, U8Ptr, F32Ptr
from mojo_dllm.formats.gguf import GGML_Q4_K, GGML_Q6_K


def measure[
    BN: Int
](
    ctx: DeviceContext,
    typ: Int,
    w: U8Ptr,
    N: Int,
    K: Int,
    qa: GpuQActs,
    T: Int,
    dst: F32Ptr,
    reps: Int,
) raises -> Float64:
    ctx.synchronize()
    var start = perf_counter_ns()
    for _ in range(reps):
        _launch[BN](ctx, typ, w, N, K, qa, T, dst, N)
    ctx.synchronize()
    return Float64(perf_counter_ns() - start) / 1e6 / Float64(reps)


def main() raises:
    var args = argv()
    var T = Int(args[1]) if len(args) > 1 else 148
    var ctx = DeviceContext()
    for typ in [GGML_Q4_K, GGML_Q6_K]:
        for shape in range(5):
            var N = 4096
            var K = 4096
            if shape == 1:
                N = 12288
            elif shape == 2:
                K = 12288
            elif shape == 3:
                N = 1024
            elif shape == 4:
                N = 126464
            var rows = 32 if shape == 4 else T
            var size = staged_matrix_bytes(typ, N, K)
            var host = Bytes(size)
            for i in range(size):
                host.u8()[unsafe_offset=i] = UInt8((i * 131 + 7) % 251)
            var w = ctx.enqueue_create_buffer[DType.uint8](size)
            ctx.enqueue_copy(w, host.u8())
            var x = Floats(rows * K)
            for i in range(rows * K):
                x[i] = Float32((i * 37) % 101 - 50) / 50.0
            var dx = ctx.enqueue_create_buffer[DType.float32](rows * K)
            ctx.enqueue_copy(dx, x.ptr())
            var qa = GpuQActs(ctx, rows, K)
            quantize_gpu(ctx, dev_f32(dx), rows, K, qa)
            var out = ctx.enqueue_create_buffer[DType.float32](rows * N)
            for round in range(5):
                for order in range(3):
                    var mode = (order + round) % 3
                    var ms = Float64(0)
                    var bn = 32
                    if mode == 0:
                        _ = measure[32](
                            ctx, typ, dev_u8(w), N, K, qa, rows, dev_f32(out), 5
                        )
                        ms = measure[32](
                            ctx,
                            typ,
                            dev_u8(w),
                            N,
                            K,
                            qa,
                            rows,
                            dev_f32(out),
                            30,
                        )
                    elif mode == 1:
                        bn = 64
                        _ = measure[64](
                            ctx, typ, dev_u8(w), N, K, qa, rows, dev_f32(out), 5
                        )
                        ms = measure[64](
                            ctx,
                            typ,
                            dev_u8(w),
                            N,
                            K,
                            qa,
                            rows,
                            dev_f32(out),
                            30,
                        )
                    else:
                        bn = 80
                        _ = measure[80](
                            ctx, typ, dev_u8(w), N, K, qa, rows, dev_f32(out), 5
                        )
                        ms = measure[80](
                            ctx,
                            typ,
                            dev_u8(w),
                            N,
                            K,
                            qa,
                            rows,
                            dev_f32(out),
                            30,
                        )
                    print(
                        '{"type":',
                        typ,
                        ',"N":',
                        N,
                        ',"K":',
                        K,
                        ',"T":',
                        rows,
                        ',"BN":',
                        bn,
                        ',"round":',
                        round,
                        ',"ms":',
                        ms,
                        "}",
                    )
