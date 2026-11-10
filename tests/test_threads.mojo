"""`--threads` reaches every parallel loop of the CPU path.

`quantize_acts` and the weight repack used to run on every core whatever
`--threads` said, so a `--threads 1` timing included all-core work. Each token
of `quantize_acts` and each 8-row group of the repack is independent, so the
worker count cannot change a byte: the first two tests prove that. The last
test reads the sources, because the bug was a call that did not pass the count
and the outputs cannot show it.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojo_dllm.formats.gguf import GGML_Q4_K, GGML_Q8_0, ggml_row_bytes
from mojo_dllm.kernels.packed import PackedMatrix
from mojo_dllm.kernels.qgemm import (
    QActs,
    quantize_acts,
    quantize_acts_counted,
)
from mojo_dllm.sys.mem import Bytes, Floats


def _acts(tokens: Int, K: Int) raises -> Floats:
    var x = Floats(tokens * K)
    var state = UInt64(4242)
    for i in range(x.n):
        state = state * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int((state >> 33) % 4001) - 2000) / 311.0
    return x^


def _same_bytes(a: Bytes, b: Bytes, what: String) raises:
    assert_equal(a.size, b.size, what + ": size")
    var differ = 0
    for i in range(a.size):
        if a.u8().unsafe_load(i) != b.u8().unsafe_load(i):
            differ += 1
    assert_equal(differ, 0, what + ": differing bytes")


def test_quantize_acts_is_bit_identical_for_every_worker_count() raises:
    for block in List[Int]([32, 256]):
        var tokens = 37
        var K = 512
        var x = _acts(tokens, K)
        var one = QActs(tokens, K, block)
        quantize_acts(x.ptr(), tokens, K, one, threads=1)
        for workers in List[Int]([2, 3, 8]):
            var many = QActs(tokens, K, block)
            quantize_acts(x.ptr(), tokens, K, many, threads=workers)
            _same_bytes(one.q, many.q, "q, block " + String(block))
            var differ = 0
            for i in range(one.d.n):
                if one.d[i] != many.d[i]:
                    differ += 1
            for i in range(one.s.n):
                if one.s[i] != many.s[i]:
                    differ += 1
            assert_equal(differ, 0, "scales, workers " + String(workers))


def test_the_worker_count_reaches_the_quantization_loop() raises:
    var tokens = 10
    var K = 256
    var x = _acts(tokens, K)
    for block in List[Int]([32, 256]):
        var qa = QActs(tokens, K, block)
        var want = String("block ") + String(block)
        assert_equal(
            quantize_acts_counted(x.ptr(), tokens, K, qa, threads=1), 1, want
        )
        assert_equal(
            quantize_acts_counted(x.ptr(), tokens, K, qa, threads=3), 3, want
        )
        # The loop never starts more workers than it has tokens.
        assert_equal(
            quantize_acts_counted(x.ptr(), tokens, K, qa, threads=64), 10, want
        )
        assert_equal(
            quantize_acts_counted(x.ptr(), 0, K, qa, threads=4), 0, want
        )


def _raw(kind: Int, N: Int, K: Int) raises -> Bytes:
    var raw = Bytes(N * ggml_row_bytes(kind, K))
    var state = UInt64(99)
    for i in range(raw.size):
        state = state * 6364136223846793005 + 1442695040888963407
        raw.u8().unsafe_store(i, UInt8((state >> 40) & 255))
    # A finite f16 scale in every Q8_0 block keeps the expected bytes free of
    # NaN payload games; Q4_K blocks take whatever bytes the stream gave.
    if kind == GGML_Q8_0:
        for b in range(raw.size // 34):
            raw.u8().unsafe_store(b * 34, UInt8(0))
            raw.u8().unsafe_store(b * 34 + 1, UInt8(0x28))
    return raw^


def test_repack_is_bit_identical_for_every_worker_count() raises:
    for kind in List[Int]([GGML_Q8_0, GGML_Q4_K]):
        var N = 64
        var K = 512
        var raw = _raw(kind, N, K)
        var one = PackedMatrix(raw.addr, raw.size, kind, N, K, threads=1)
        for workers in List[Int]([2, 3, 8]):
            var many = PackedMatrix(
                raw.addr, raw.size, kind, N, K, threads=workers
            )
            _same_bytes(one.data, many.data, "repack, kind " + String(kind))


def test_the_worker_count_reaches_the_repack_loop() raises:
    var N = 64
    var K = 256
    var raw = _raw(GGML_Q8_0, N, K)
    var one = PackedMatrix(raw.addr, raw.size, GGML_Q8_0, N, K, threads=1)
    assert_equal(one.repack_workers, 1)
    var three = PackedMatrix(raw.addr, raw.size, GGML_Q8_0, N, K, threads=3)
    assert_equal(three.repack_workers, 3)
    # 64 rows are 8 groups: no more than 8 workers can have work.
    var many = PackedMatrix(raw.addr, raw.size, GGML_Q8_0, N, K, threads=64)
    assert_equal(many.repack_workers, 8)


def _call_text(text: String, at: Int) -> String:
    """The text of the call whose name starts at `at`: up to its closing paren.
    """
    var open = text.find("(", at)
    var depth = 0
    var i = open
    var bytes = text.as_bytes()
    while i < len(bytes):
        if bytes[i] == 40:
            depth += 1
        elif bytes[i] == 41:
            depth -= 1
            if depth == 0:
                return String(text[byte = at : i + 1])
        i += 1
    return String(text[byte=at:])


def _calls_without_threads(
    path: String, names: List[String]
) raises -> List[String]:
    var text: String
    with open(path, "r") as f:
        text = f.read()
    var bad = List[String]()
    for name in names:
        var at = 0
        while True:
            var hit = text.find(name + "(", at)
            if hit < 0:
                break
            at = hit + 1
            # A definition (`def name(`, `struct Name(`) or a method call on
            # another name (`x.name(`) is not a call of the function.
            if hit >= 4 and String(text[byte = hit - 4 : hit]) == "def ":
                continue
            if hit >= 7 and String(text[byte = hit - 7 : hit]) == "struct ":
                continue
            if hit >= 1 and text.as_bytes()[hit - 1] == 46:
                continue
            if hit >= 1 and (
                text.as_bytes()[hit - 1] == 95
                or (
                    text.as_bytes()[hit - 1] >= 97
                    and text.as_bytes()[hit - 1] <= 122
                )
            ):
                continue
            var call = _call_text(text, hit)
            if not call.__contains__("threads"):
                bad.append(path + ": " + call)
    return bad^


def test_every_production_site_passes_the_worker_count() raises:
    var names = List[String](
        ["quantize_acts", "PackedMatrix", "_load_matrix", "ExpertWeights"]
    )
    var bad = List[String]()
    for path in List[String](
        [
            "src/mojo_dllm/models/transformer.mojo",
            "src/mojo_dllm/models/diffusion_gemma.mojo",
            "src/mojo_dllm/kernels/moe.mojo",
        ]
    ):
        for b in _calls_without_threads(path, names):
            bad.append(b)
    var report = String("")
    for b in bad:
        report += b + "\n"
    assert_equal(len(bad), 0, "calls without the worker count:\n" + report)
    # The scan must find calls: each of these files has at least one.
    var text: String
    with open("src/mojo_dllm/models/transformer.mojo", "r") as f:
        text = f.read()
    assert_true(text.find("quantize_acts(") > 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
