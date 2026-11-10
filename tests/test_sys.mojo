from std.ffi import c_int, external_call
from std.os import remove
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.sys.mem import Bytes, Floats, alloc_bytes, free_bytes
from mojo_dllm.sys.mmap import Mapping, drop_range, populate_range
from mojo_dllm.sys.parallel import parallel_for, AtomicPtr


def test_byte_alignment_and_zero_initialization() raises:
    var blocks = List[Bytes]()
    for i in range(16):
        blocks.append(Bytes(1 + i * 17))
    for i in range(len(blocks)):
        assert_equal(blocks[i].addr % 64, 0)
        for j in range(blocks[i].size):
            assert_equal(blocks[i].u8().unsafe_load(j), 0)
    var empty = Bytes(0)
    assert_equal(empty.size, 0)
    assert_true(empty.addr != 0)


def test_huge_page_alignment() raises:
    for size in List[Int]([2 * 1024 * 1024, 2 * 1024 * 1024 + 1]):
        var b = Bytes(size)
        assert_equal(b.addr % (2 * 1024 * 1024), 0)
        for i in range(size):
            assert_equal(b.u8().unsafe_load(i), 0)


def test_negative_sizes_raise() raises:
    with assert_raises(contains="-1"):
        var addr = alloc_bytes(-1)
        free_bytes(addr)
    with assert_raises(contains="-20"):
        _ = Floats(-5)


def test_impossible_allocation_raises() raises:
    # Exceeds the virtual address space on x86-64 Linux; never touch pages.
    with assert_raises(contains="out of memory: could not allocate"):
        var addr = alloc_bytes(1 << 50, zero=False)
        free_bytes(addr)


def test_sizes_that_overflow_raise_with_a_name() raises:
    # n * 4 wraps to 0 for 2^62: the old code returned a 64-byte block for it.
    with assert_raises(contains="cannot allocate 4611686018427387904 floats"):
        _ = Floats(1 << 62)
    with assert_raises(contains="cannot allocate 2305843009213693952 floats"):
        _ = Floats(Int.MAX // 4 + 1)
    with assert_raises(contains="cannot allocate 9223372036854775807 bytes"):
        var addr = alloc_bytes(Int.MAX, zero=False)
        free_bytes(addr)


def test_floats_initialization_roundtrip_and_zero() raises:
    var f = Floats(17)
    for i in range(f.n):
        assert_equal(f[i], Float32(0))
        f[i] = Float32(i) - 0.5
    for i in range(f.n):
        assert_equal(f[i], Float32(i) - 0.5)
    f.zero()
    for i in range(f.n):
        assert_equal(f[i], Float32(0))


def test_mapping_reads_bytes() raises:
    var m = Mapping("tests/fixtures/bad-magic.gguf")
    assert_equal(m.size, 64)
    var expected = "GGUX".as_bytes()
    for i in range(4):
        assert_equal(m.ptr().unsafe_load(i), expected[i])


def _open_error(path: String) -> String:
    try:
        _ = Mapping(path)
    except e:
        return String(e)
    return String("opened")


def test_mapping_missing_path_names_file_and_reason() raises:
    assert_equal(
        _open_error("tests/fixtures/does-not-exist.gguf"),
        (
            "cannot open file: tests/fixtures/does-not-exist.gguf: No such"
            " file or directory"
        ),
    )


def test_mapping_open_reason_survives_quotes_in_the_path() raises:
    assert_equal(
        _open_error("/nonexistent/x': y"),
        "cannot open file: /nonexistent/x': y: No such file or directory",
    )


def test_mapping_reports_the_errno_of_the_failure() raises:
    assert_equal(
        _open_error("tests/fixtures/bad-magic.gguf/x"),
        "cannot open file: tests/fixtures/bad-magic.gguf/x: Not a directory",
    )


def test_mapping_empty_file() raises:
    with assert_raises(
        contains="file is empty: tests/fixtures/malformed/empty.gguf"
    ):
        _ = Mapping("tests/fixtures/malformed/empty.gguf")


def test_mapping_reports_the_errno_when_mmap_fails() raises:
    # A sysfs attribute reports a size of 4096 and cannot be mapped: mmap
    # answers ENODEV, which the message must carry.
    var path = "/sys/kernel/uevent_seqnum"
    if not exists(path):
        return
    with assert_raises(contains="mmap failed for " + path + ": No such device"):
        _ = Mapping(path)


def test_mapping_reports_the_errno_when_the_size_cannot_be_read() raises:
    # A FIFO opens and answers pread and lseek with ESPIPE. An open writer
    # keeps the read-only open from blocking.
    var path = String("build/tests/seek.fifo")
    if exists(path):
        remove(path)
    var cpath = path
    if (
        external_call["mkfifo", c_int](cpath.as_c_string_span(), c_int(0o600))
        != 0
    ):
        return
    var keeper = open(path, "rw")
    with assert_raises(
        contains="cannot get the size of " + path + ": Illegal seek"
    ):
        _ = Mapping(path)
    keeper.close()
    remove(path)


def test_mapping_rejects_directory() raises:
    # ext4 answers lseek on a directory with a huge offset and tmpfs fails it,
    # so the filesystem must not decide the message.
    with assert_raises(contains="is a directory: tests/fixtures"):
        _ = Mapping("tests/fixtures")


def test_page_ranges() raises:
    var lo, hi = drop_range(4097, 8192)
    assert_equal(lo, 8192)
    assert_equal(hi, 12288)
    var outer_lo, outer_hi = populate_range(4097, 8192)
    assert_equal(outer_lo, 4096)
    assert_equal(outer_hi, 16384)
    var empty_lo, empty_hi = drop_range(4097, 100)
    assert_true(empty_hi <= empty_lo)
    var aligned_lo, aligned_hi = drop_range(4096, 4096)
    assert_equal(aligned_lo, 4096)
    assert_equal(aligned_hi, 8192)
    # Read-back after madvise cannot establish whether a page was dropped:
    # file-backed mappings read the same bytes again. Test the range directly.


def _refuses_advice(offset: Int, size: Int) raises:
    # Every size here is under a page, so even a guard that is mutated away
    # only reaches `madvise` with an empty range of whole pages for `drop`.
    var m = Mapping("tests/fixtures/bad-magic.gguf")
    assert_equal(m.size, 64)
    var want = "outside the 64-byte mapping of tests/fixtures/bad-magic.gguf"
    with assert_raises(
        contains="drop range ["
        + String(offset)
        + ", +"
        + String(size)
        + ") is "
        + want
    ):
        m.drop(offset, size)
    with assert_raises(
        contains="populate range ["
        + String(offset)
        + ", +"
        + String(size)
        + ") is "
        + want
    ):
        m.populate(offset, size)


def test_mapping_advice_refuses_a_range_below_zero() raises:
    _refuses_advice(-1, 8)
    _refuses_advice(0, -1)
    _refuses_advice(-64, 64)


def test_mapping_advice_refuses_a_range_past_the_end() raises:
    _refuses_advice(0, 65)
    _refuses_advice(1, 64)
    _refuses_advice(65, 0)
    _refuses_advice(64, 1)


def test_mapping_advice_refuses_a_range_that_wraps() raises:
    # offset + size wraps negative, so a check written as a sum accepts these.
    _refuses_advice(8, Int.MAX)
    _refuses_advice(Int.MAX, 8)
    _refuses_advice(Int.MAX - 3, 8)


def test_mapping_advice_accepts_ranges_inside_the_map() raises:
    var m = Mapping("tests/fixtures/bad-magic.gguf")
    m.populate(0, 64)
    m.populate(32, 32)
    m.populate(64, 0)
    m.populate(0, 0)
    m.drop(0, 64)
    m.drop(32, 32)
    m.drop(64, 0)
    # The bytes are still readable after the advice.
    var expected = "GGUX".as_bytes()
    for i in range(4):
        assert_equal(m.ptr().unsafe_load(i), expected[i])


def check_parallel(n: Int, threads: Int) raises:
    var counts = Bytes((max(n, 0) + 1) * 8)
    var ptr = AtomicPtr(unsafe_from_address=counts.addr)

    def work(i: Int) {imm}:
        _ = ptr[unsafe_offset=i].fetch_add(1)

    parallel_for(work, n, threads)
    for i in range(max(n, 0)):
        assert_equal(ptr[unsafe_offset=i].load(), Int64(1))
    assert_equal(ptr[unsafe_offset=max(n, 0)].load(), Int64(0))


def test_parallel_visits_each_item_once_and_preserves_guard() raises:
    for n in List[Int]([0, 1, 7, 64, 1000]):
        for threads in List[Int]([1, 2, 4, 64]):
            check_parallel(n, threads)


def test_parallel_nonpositive_items_and_threads() raises:
    for threads in List[Int]([-3, 0, 1, 4]):
        check_parallel(0, threads)
        check_parallel(-4, threads)
        check_parallel(7, threads)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
