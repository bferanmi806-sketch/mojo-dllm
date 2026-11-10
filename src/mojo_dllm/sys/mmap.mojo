"""Read-only memory maps of model files.

A 5 GB GGUF file is mapped, never read into the heap. The kernel faults pages
in as the first forward pass touches them, and quantized weights stay in the
page cache in their on-disk layout.

`mmap` reports failure as (void *)-1, so the result is compared as an integer
before anything turns it into a pointer.
"""

from std.ffi import ErrNo, c_int, c_size_t, external_call, get_errno

from mojo_dllm.sys.mem import Bytes, ConstU8Ptr

comptime O_RDONLY = 0
comptime PROT_READ = 1
comptime MAP_PRIVATE = 2
comptime SEEK_END = 2
comptime MADV_WILLNEED = 3
comptime MADV_DONTNEED = 4
comptime MADV_POPULATE_READ = 22
comptime PAGE = 4096


def drop_range(addr: Int, size: Int) -> Tuple[Int, Int]:
    """Whole pages contained in [addr, addr+size); empty when hi <= lo."""
    return ((addr + PAGE - 1) // PAGE * PAGE, (addr + size) // PAGE * PAGE)


def populate_range(addr: Int, size: Int) -> Tuple[Int, Int]:
    """Whole pages covering [addr, addr+size)."""
    return (addr // PAGE * PAGE, (addr + size + PAGE - 1) // PAGE * PAGE)


struct Mapping(Movable):
    """A whole file mapped read-only. Unmapped when dropped."""

    var addr: Int
    var size: Int
    var path: String

    def __init__(out self, path: String) raises:
        self.path = path
        var cpath = path
        var fd = external_call["open", c_int, num_fixed_args=2](
            cpath.as_c_string_span(), c_int(O_RDONLY), c_int(0)
        )
        var failure = get_errno()
        _ = cpath^
        if Int(fd) < 0:
            raise Error("cannot open file: " + path + ": " + String(failure))
        # A directory opens read-only, then lseek and mmap answer differently
        # per filesystem. pread() fails with EISDIR on every one of them.
        var probe = Bytes(1)
        var got = Int(
            external_call["pread", Int](fd, probe.addr, c_size_t(1), Int(0))
        )
        if got < 0 and get_errno() == ErrNo.EISDIR:
            _ = external_call["close", c_int](fd)
            raise Error("is a directory: " + path)
        var end = Int(external_call["lseek", Int](fd, Int(0), c_int(SEEK_END)))
        var seek_failure = get_errno()
        if end < 0:
            _ = external_call["close", c_int](fd)
            raise Error(
                "cannot get the size of " + path + ": " + String(seek_failure)
            )
        if end == 0:
            _ = external_call["close", c_int](fd)
            raise Error("file is empty: " + path)
        var addr = Int(
            external_call["mmap", Int](
                Int(0),
                c_size_t(end),
                c_int(PROT_READ),
                c_int(MAP_PRIVATE),
                fd,
                Int(0),
            )
        )
        var map_failure = get_errno()
        _ = external_call["close", c_int](fd)
        if addr == -1 or addr == 0:
            raise Error("mmap failed for " + path + ": " + String(map_failure))
        self.addr = addr
        self.size = end

    def __deinit__(deinit self):
        _ = external_call["munmap", c_int](self.addr, c_size_t(self.size))

    def ptr(self) -> ConstU8Ptr:
        """The mapped bytes. The pages are `PROT_READ`, so the pointer cannot store.
        """
        return ConstU8Ptr(unsafe_from_address=self.addr)

    def will_need(self):
        """Ask the kernel to start reading the whole file ahead of use."""
        _ = external_call["madvise", c_int](
            self.addr, c_size_t(self.size), c_int(MADV_WILLNEED)
        )

    def _check_range(self, what: String, offset: Int, size: Int) raises:
        """The range [offset, offset + size) lies inside the mapping.

        The end is compared with `self.size` by subtraction, so a size near
        2^63 cannot wrap past the test. `MADV_DONTNEED` outside the mapping
        zero-fills private anonymous pages, which is heap corruption.
        """
        # An offset past the end leaves `self.size - offset` negative, so the
        # last test refuses it for every size.
        if offset < 0 or size < 0 or size > self.size - offset:
            raise Error(
                what
                + " range ["
                + String(offset)
                + ", +"
                + String(size)
                + ") is outside the "
                + String(self.size)
                + "-byte mapping of "
                + self.path
            )

    def drop(self, offset: Int, size: Int) raises:
        """Release the resident pages of [offset, offset+size) of the file,
        shrunk to whole pages.

        Used after a tensor has been repacked into the heap: the mapped bytes
        will not be read again, and leaving them resident would double-count
        the model in RSS. A later read faults the page back in from the file.
        The range must lie inside the mapping.
        """
        self._check_range("drop", offset, size)
        var lo, hi = drop_range(self.addr + offset, size)
        if hi > lo:
            _ = external_call["madvise", c_int](
                lo, c_size_t(hi - lo), c_int(MADV_DONTNEED)
            )

    def populate(self, offset: Int, size: Int) raises:
        """Fault in [offset, offset+size) of the file with one call (Linux 5.14+).

        Repacking reads every byte of a tensor once. Without this the kernel
        takes one page fault per 4 KiB page; MADV_POPULATE_READ reads the
        range in bulk. Failure (an older kernel) only costs speed. The range
        must lie inside the mapping.
        """
        self._check_range("populate", offset, size)
        var lo, hi = populate_range(self.addr + offset, size)
        if hi > lo:
            _ = external_call["madvise", c_int](
                lo, c_size_t(hi - lo), c_int(MADV_POPULATE_READ)
            )
