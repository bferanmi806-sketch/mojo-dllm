"""Aligned heap buffers addressed by integer.

Struct fields in Mojo 1.1 cannot hold a pointer with an untracked origin, so a
buffer keeps its address as an `Int` and hands out pointers on demand. The
buffer owns the block and frees it in `__deinit__`.

Allocation goes through libc `aligned_alloc` with 64-byte alignment, so every
row a kernel touches starts on a cache line and 256-bit loads never split one.
"""

from std.ffi import c_int, c_size_t, external_call
from std.memory import unsafe_memset

comptime ALIGN = 64

comptime F32Ptr = Pointer[Float32, MutAnyOrigin]
comptime I8Ptr = Pointer[Int8, MutAnyOrigin]
comptime U8Ptr = Pointer[UInt8, MutAnyOrigin]
comptime ConstU8Ptr = Pointer[UInt8, ImmutAnyOrigin]
comptime I32Ptr = Pointer[Int32, MutAnyOrigin]
comptime F64Ptr = Pointer[Float64, MutAnyOrigin]
comptime I16Ptr = Pointer[Int16, MutAnyOrigin]


comptime HUGE = 2 * 1024 * 1024
comptime MADV_HUGEPAGE = 14


def alloc_bytes(n_bytes: Int, zero: Bool = True) raises -> Int:
    """A 64-byte aligned block, zeroed unless the caller overwrites all of it.

    Blocks of 2 MiB and more are aligned to 2 MiB and advised as huge-page
    candidates: a repacked model is several gigabytes, and 4 KiB pages would cost
    over a million page faults to populate it.
    """
    if n_bytes < 0:
        raise Error(
            "cannot allocate a negative size: " + String(n_bytes) + " bytes"
        )
    if n_bytes > Int.MAX - HUGE:
        # Rounding the size up to the alignment would wrap.
        raise Error(
            "cannot allocate "
            + String(n_bytes)
            + " bytes: the size is beyond what the allocator can round up"
        )
    var align = HUGE if n_bytes >= HUGE else ALIGN
    var size = ((max(n_bytes, 1) + align - 1) // align) * align
    var addr = Int(
        external_call["aligned_alloc", Int](c_size_t(align), c_size_t(size))
    )
    if addr == 0:
        raise Error(
            "out of memory: could not allocate " + String(size) + " bytes"
        )
    if align == HUGE:
        _ = external_call["madvise", c_int](
            addr, c_size_t(size), c_int(MADV_HUGEPAGE)
        )
    if zero:
        unsafe_memset(U8Ptr(unsafe_from_address=addr), 0, size)
    return addr


def free_bytes(addr: Int):
    external_call["free", NoneType](addr)


struct Bytes(Movable):
    """An owned, aligned byte block, zeroed unless `zero` is False."""

    var addr: Int
    var size: Int

    def __init__(out self, size: Int, zero: Bool = True) raises:
        self.size = size
        self.addr = alloc_bytes(size, zero)

    def __deinit__(deinit self):
        free_bytes(self.addr)

    def u8(self) -> U8Ptr:
        return U8Ptr(unsafe_from_address=self.addr)

    def i8(self) -> I8Ptr:
        return I8Ptr(unsafe_from_address=self.addr)

    def f32(self) -> F32Ptr:
        return F32Ptr(unsafe_from_address=self.addr)

    def i32(self) -> I32Ptr:
        return I32Ptr(unsafe_from_address=self.addr)

    def f64(self) -> F64Ptr:
        return F64Ptr(unsafe_from_address=self.addr)

    def i16(self) -> I16Ptr:
        return I16Ptr(unsafe_from_address=self.addr)


struct Floats(Movable):
    """An owned f32 array of `n` elements, zero-initialised unless `zero` is
    False, in which case the caller writes every element before it reads one."""

    var addr: Int
    var n: Int

    def __init__(out self, n: Int, zero: Bool = True) raises:
        if n > Int.MAX // 4:
            # n * 4 wraps: 2^62 floats would be asked of the allocator as 0.
            raise Error(
                "cannot allocate "
                + String(n)
                + " floats: the size in bytes does not fit in an Int"
            )
        self.n = n
        self.addr = alloc_bytes(n * 4, zero)

    def __deinit__(deinit self):
        free_bytes(self.addr)

    def ptr(self) -> F32Ptr:
        return F32Ptr(unsafe_from_address=self.addr)

    def __getitem__(self, i: Int) -> Float32:
        return self.ptr().unsafe_load(i)

    def __setitem__(mut self, i: Int, v: Float32):
        self.ptr().unsafe_store(i, v)

    def zero(mut self):
        unsafe_memset(U8Ptr(unsafe_from_address=self.addr), 0, self.n * 4)
