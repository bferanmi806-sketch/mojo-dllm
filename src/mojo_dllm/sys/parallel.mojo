"""A parallel-for that balances work across unequal cores.

`max.algorithm.parallelize` hands each worker a fixed share of the items. On a
hybrid CPU that makes every parallel region as slow as its slowest core: on an
i5-13420H one Gracemont E-core runs the quantized GEMM at a third of a
Golden Cove P-core's rate, so 8 workers measured barely faster than one P-core.

Here each worker claims the next item from a shared atomic counter until none
are left, so fast cores take more items. The counter lives on its own
64-byte line in a heap block, because an atomic that moves with a Mojo local is
two separate counters.
"""

from std.atomic import Atomic
from std.sys import num_logical_cores
from max.algorithm import parallelize

from mojo_dllm.sys.mem import alloc_bytes, free_bytes


comptime AtomicPtr = Pointer[Atomic[Int64], MutAnyOrigin]


def worker_count(threads: Int) -> Int:
    """The worker count a kernel runs with: `threads` when it is positive and
    every logical core when it is 0, the default of the kernels that tools and
    benchmarks call without a `--threads` setting. The model paths always pass
    the setting; tests/test_threads.mojo reads the sources to check that."""
    return threads if threads > 0 else num_logical_cores()


def parallel_for_counted[
    F: def(Int) -> None
](func: F, n_items: Int, threads: Int) raises -> Int:
    """Run func(i) for every i in [0, n_items), dynamically scheduled, and
    return the number of workers that ran it (0 when there was nothing to do).

    The count lets a test see that a `--threads` setting reached the loop."""
    if n_items <= 0:
        return 0
    var workers = max(1, min(threads, n_items))
    if workers == 1:
        for i in range(n_items):
            func(i)
        return 1
    var cell = alloc_bytes(64)
    var counter = AtomicPtr(unsafe_from_address=cell)

    def worker(w: Int) {imm}:
        while True:
            var i = Int(counter[].fetch_add(1))
            if i >= n_items:
                break
            func(i)

    parallelize(worker, workers, workers)
    free_bytes(cell)
    return workers


def parallel_for[
    F: def(Int) -> None
](func: F, n_items: Int, threads: Int) raises:
    """Run func(i) for every i in [0, n_items), dynamically scheduled."""
    _ = parallel_for_counted(func, n_items, threads)
