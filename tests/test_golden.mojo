"""Pin current sampler behavior, not independent correctness.

These tokens come from this implementation. Numpy forward-pass tests and
real-model parity runs remain the independent correctness evidence.
"""

from std.os import getenv
from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from cli.args import parse_csv_ints
from mojo_dllm.diffusion.sampler import (
    GenConfig,
    StepObserver,
    generate_observed,
)
from mojo_dllm.models.transformer import DiffusionLM

comptime GOLDEN = "tests/fixtures/golden-generation.tsv"
comptime REGENERATE = "regenerate with UPDATE_GOLDEN=1 build/tests/test_golden"


def _names() -> List[String]:
    return [
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
        "L7",
        "L8",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
    ]


struct Recorder(StepObserver):
    var counts: List[Int]

    def __init__(out self):
        self.counts = List[Int]()

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        self.counts.append(len(committed))


def _csv(values: List[Int]) -> String:
    var out = String()
    for i in range(len(values)):
        if i:
            out += ","
        out += String(values[i])
    return out^


def _config(name: String) -> GenConfig:
    var c = GenConfig()
    c.gen_length = 16
    c.block_length = 8
    c.steps = 8
    c.mask_id = 511
    c.threads = 1
    c.early_stop = False
    c.threshold = 0.0
    c.dual_cache = False
    if name == "L2":
        c.block_length = 16
    elif name == "L3":
        c.gen_length = 12
        c.block_length = 4
        c.steps = 9
    elif name == "L4":
        c.gen_length = 8
    elif name == "L5":
        c.remasking = "random"
        c.seed = 1
    elif name == "L6" or name == "L7":
        c.temperature = 0.7
        c.seed = 1234
        if name == "L7":
            c.remasking = "random"
    elif name == "L8":
        c.early_stop = True
        # Fixed from the middle of L1's baseline output, not derived at runtime.
        c.stop_ids = [347]
    if name.startswith("D"):
        c.mask_id = 510
        c.block_length = 16
        if name == "D2":
            c.remasking = "maskgit_plus"
            c.eps = 0.2
        elif name == "D3":
            c.remasking = "topk_margin"
            c.gen_length = 12
        elif name == "D4":
            c.steps = 5
            c.remasking = "entropy"
            c.eps = 0.1
        elif name == "D5":
            c.eps = 0.3
        elif name == "D6":
            c.early_stop = True
            # Fixed from the middle of D1's baseline output.
            c.stop_ids = [49]
    return c^


def _run(name: String) raises -> String:
    var arch = "dream" if name.startswith("D") else "llada"
    var model = DiffusionLM(
        "tests/fixtures/tiny-" + arch + ".gguf", threads=2, max_tokens=24
    )
    var prompt: List[Int] = [5, 17, 300]
    # The issue's original prompt gives all-49 Dream outputs for both eps
    # settings. This fixed prompt makes D1/D5 sensitive to that schedule.
    if name.startswith("D"):
        prompt = [448, 50, 134]
    var recorder = Recorder()
    var result = generate_observed(model, prompt, _config(name), recorder)
    return (
        name
        + "\t"
        + String(result.forward_passes)
        + "\t"
        + String(result.stopped_early)
        + "\t"
        + _csv(recorder.counts)
        + "\t"
        + _csv(result.tokens)
    )


def _rows(text: String) raises -> List[String]:
    var rows = List[String]()
    var seen = List[String]()
    var names = _names()
    for line in text.split("\n"):
        var row = String(line)
        if row == "":
            continue
        var fields = row.split("\t")
        assert_equal(
            len(fields), 5, "golden: expected five fields; " + REGENERATE
        )
        var name = String(fields[0])
        assert_true(
            name in names, "golden: unknown case " + name + "; " + REGENERATE
        )
        assert_true(
            name not in seen,
            "golden: duplicate case " + name + "; " + REGENERATE,
        )
        seen.append(name)
        assert_true(
            String(fields[2]) == "True" or String(fields[2]) == "False",
            name + ": invalid stopped_early; " + REGENERATE,
        )
        var counts = parse_csv_ints(String(fields[3]))
        var tokens = parse_csv_ints(String(fields[4]))
        assert_true(
            len(counts) > 0 and len(tokens) >= 4,
            name + ": empty generation; " + REGENERATE,
        )
        assert_equal(
            Int(String(fields[1])),
            len(counts),
            name + ": forward_passes/counts; " + REGENERATE,
        )
        rows.append(row)
    assert_equal(
        len(rows),
        len(names),
        "golden: missing cases or empty file; " + REGENERATE,
    )
    assert_true(len(rows) >= 14, "golden: too few cases; " + REGENERATE)
    return rows^


def _load() raises -> List[String]:
    var text: String
    try:
        with open(GOLDEN, "r") as f:
            text = f.read()
    except:
        raise Error("missing or unreadable " + GOLDEN + "; " + REGENERATE)
    return _rows(text)


def _find(rows: List[String], name: String) raises -> String:
    for row in rows:
        if row.startswith(name + "\t"):
            return row.copy()
    raise Error("golden: missing case " + name + "; " + REGENERATE)


def _compare(name: String, want: String, got: String) raises:
    if want == got:
        return
    var expected = want.split("\t")
    var actual = got.split("\t")
    var a = parse_csv_ints(String(expected[4]))
    var b = parse_csv_ints(String(actual[4]))
    var position = 0
    while position < min(len(a), len(b)) and a[position] == b[position]:
        position += 1
    var difference = "none (metadata or schedule differs)"
    if position < max(len(a), len(b)):
        difference = String(position)
    raise Error(
        name
        + ": first differing token position (zero-based, prompt included): "
        + difference
        + "\nexpected tokens: "
        + String(expected[4])
        + "\nactual tokens: "
        + String(actual[4])
        + "\nexpected row: "
        + want
        + "\nactual row: "
        + got
        + "\n"
        + REGENERATE
    )


def _check(name: String) raises:
    var rows = _load()
    _compare(name, _find(rows, name), _run(name))


def test_golden_L1() raises:
    _check("L1")


def test_golden_L2() raises:
    _check("L2")


def test_golden_L3() raises:
    _check("L3")


def test_golden_L4() raises:
    _check("L4")


def test_golden_L5() raises:
    _check("L5")


def test_golden_L6() raises:
    _check("L6")


def test_golden_L7() raises:
    _check("L7")


def test_golden_L8() raises:
    _check("L8")


def test_golden_D1() raises:
    _check("D1")


def test_golden_D2() raises:
    _check("D2")


def test_golden_D3() raises:
    _check("D3")


def test_golden_D4() raises:
    _check("D4")


def test_golden_D5() raises:
    _check("D5")


def test_golden_D6() raises:
    _check("D6")


def test_golden_case_set_and_variation() raises:
    var rows = _load()
    for i in range(len(rows)):
        for j in range(i):
            assert_true(
                rows[i].split("\t")[4] != rows[j].split("\t")[4],
                "golden token lists must be distinct: "
                + String(rows[i].split("\t")[0])
                + " / "
                + String(rows[j].split("\t")[0]),
            )
    var l1 = _find(rows, "L1")
    var l2 = _find(rows, "L2")
    var d1 = _find(rows, "D1")
    var d5 = _find(rows, "D5")
    assert_true(l1.split("\t")[4] != l2.split("\t")[4], "L1 and L2 must differ")
    assert_true(d1.split("\t")[4] != d5.split("\t")[4], "D1 and D5 must differ")
    for name in ["L8", "D6"]:
        assert_equal(
            String(_find(rows, name).split("\t")[2]),
            "True",
            name + " must stop early",
        )


def test_golden_rejects_empty_or_missing_cases() raises:
    with assert_raises(contains="UPDATE_GOLDEN=1"):
        _ = _rows("")
    with assert_raises(contains="missing cases"):
        _ = _rows("L1\t1\tFalse\t1\t5,17,300,42\n")


def test_golden_rejects_duplicate_and_unknown_cases() raises:
    var row = "L1\t1\tFalse\t1\t5,17,300,42\n"
    with assert_raises(contains="duplicate case L1"):
        _ = _rows(row + row.copy())
    with assert_raises(contains="unknown case X"):
        _ = _rows("X\t1\tFalse\t1\t5,17,300,42\n")


def test_golden_mismatch_names_position_and_schedule() raises:
    var row = "L1\t1\tFalse\t1\t5,17,300,42"
    with assert_raises(contains="prompt included): 3"):
        _compare("L1", row, "L1\t1\tFalse\t1\t5,17,300,43")
    with assert_raises(contains="metadata or schedule differs"):
        _compare("L1", row, "L1\t2\tFalse\t0,1\t5,17,300,42")


def main() raises:
    if getenv("UPDATE_GOLDEN") == "1":
        var text = String()
        for name in _names():
            text += _run(name) + "\n"
        _ = _rows(text)
        with open(GOLDEN, "w") as f:
            f.write(text)
    TestSuite.discover_tests[__functions_in_module()]().run()
