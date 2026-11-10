"""The LLaDA and Dream loops refuse logits that are not finite.

A NaN logit makes `x > mx` false, so the argmax falls back to token 0 and the
confidence is NaN; the loop then commits token 0 everywhere and ends with the
stop reason "complete". A scripted model hands the loops rows chosen so that
one row is bad and every other row is one-hot at token 2, on the CPU path and
on the device-statistics path.
"""

from std.math import exp, inf, nan
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.diffusion.sampler import GenConfig, GenResult, generate
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.models.transformer import (
    DenoisingModel,
    ModelConfig,
    RowStats,
    Timings,
)
from mojo_dllm.sys.mem import F32Ptr

comptime VOCAB = 8
comptime MASK = 4
comptime GOOD = 2
comptime NAN_ROW = 0
comptime POS_INF_ROW = 1
comptime NEG_INF_ROW = 2
comptime ONE_NEG_INF = 3
"""A single -inf logit among finite ones: probability 0, a legal masked token."""
comptime NONE = -1


struct BadRowModel(DenoisingModel, Movable):
    """Every requested row is one-hot at token 2 except the `bad`-th, which is
    damaged as `kind` says. With `stats` the model reports row statistics the
    way the GPU does and never fills a logits buffer."""

    var cfg: ModelConfig
    var bad: Int
    var kind: Int
    var stats_path: Bool
    var first_call: List[Int]
    """The tokens of the first forward call: the canvas the loop built."""

    def __init__(
        out self,
        arch: String,
        bad: Int,
        kind: Int,
        stats_path: Bool,
        mask: Int = MASK,
    ) raises:
        self.cfg = ModelConfig(GGUFFile("tests/fixtures/tiny-llada.gguf"))
        self.cfg.arch = arch
        self.cfg.vocab = VOCAB
        self.cfg.shift_logits = arch == "dream"
        self.cfg.mask_id = mask
        self.first_call = List[Int]()
        self.bad = bad
        self.kind = kind
        self.stats_path = stats_path

    def config(self) -> ModelConfig:
        return self.cfg

    def logit_row(self, position: Int) -> Int:
        return position - 1 if self.cfg.shift_logits else position

    def _row(self, j: Int) -> List[Float32]:
        var r = List[Float32](length=VOCAB, fill=Float32(0))
        r[GOOD] = 20.0
        if j == self.bad:
            if self.kind == NAN_ROW:
                r[3] = nan[DType.float32]()
            elif self.kind == POS_INF_ROW:
                r[3] = inf[DType.float32]()
            elif self.kind == NEG_INF_ROW:
                for v in range(VOCAB):
                    r[v] = -inf[DType.float32]()
            else:
                r[5] = -inf[DType.float32]()
        return r^

    def forward(
        mut self, tokens: List[Int], out_rows: List[Int], logits: F32Ptr
    ) raises:
        if len(self.first_call) == 0:
            self.first_call = tokens.copy()
        for j in range(len(out_rows)):
            var r = self._row(j)
            for v in range(VOCAB):
                logits[unsafe_offset=j * VOCAB + v] = r[v]

    def forward_window(
        mut self,
        tokens: List[Int],
        lo: Int,
        hi: Int,
        out_rows: List[Int],
        logits: F32Ptr,
    ) raises:
        raise Error("no window")

    def device_row_stats(self) -> Bool:
        return self.stats_path

    def forward_stats(
        mut self, tokens: List[Int], out_rows: List[Int], mut out: RowStats
    ) raises:
        """The statistics the device would reduce from the same rows: the
        first argmax, the maximum, z = sum exp(x - max), and the other two."""
        out = RowStats()
        for j in range(len(out_rows)):
            var r = self._row(j)
            var mx = Float64(r[0])
            var best = 0
            for v in range(1, VOCAB):
                if Float64(r[v]) > mx:
                    mx = Float64(r[v])
                    best = v
            var z = Float64(0)
            var second = Float64(-1e300)
            var weighted = Float64(0)
            for v in range(VOCAB):
                var x = Float64(r[v])
                var e = exp(x - mx)
                z += e
                weighted += e * (x - mx)
                if v != best and x > second:
                    second = x
            out.tok.append(best)
            out.mx.append(mx)
            out.z.append(z)
            out.second.append(second)
            out.weighted.append(weighted)

    def model_name(self) raises -> String:
        return "bad-row"

    def load_time(self) -> Float64:
        return 0.0

    def stats(self) -> Timings:
        return Timings()


def _config(arch: String) -> GenConfig:
    var c = GenConfig()
    c.gen_length = 4
    c.block_length = 4
    c.steps = 4
    c.early_stop = False
    return c^


def _prompt() -> List[Int]:
    return [1, 3]


comptime LLADA_GREEDY = 0
comptime LLADA_RANDOM = 1
comptime LLADA_GUMBEL = 2
comptime LLADA_STATS = 3
comptime DREAM_CPU = 4
comptime DREAM_STATS = 5
comptime DREAM_MASKGIT = 6
comptime DREAM_MARGIN = 7
comptime DREAM_MASKGIT_STATS = 8
comptime PATHS = 9


def _run(path: Int, bad: Int, kind: Int) raises -> GenResult:
    var arch = "dream" if path >= DREAM_CPU else "llada"
    var model = BadRowModel(
        arch,
        bad,
        kind,
        path == LLADA_STATS
        or path == DREAM_STATS
        or path == DREAM_MASKGIT_STATS,
    )
    var cfg = _config(arch)
    if path == DREAM_MASKGIT or path == DREAM_MASKGIT_STATS:
        cfg.remasking = "maskgit_plus"
    if path == DREAM_MARGIN:
        cfg.remasking = "topk_margin"
    if path == LLADA_RANDOM:
        cfg.remasking = "random"
    if path == LLADA_GUMBEL:
        cfg.temperature = 0.7
    return generate(model, _prompt(), cfg)


def _names() -> List[String]:
    return [
        "llada greedy",
        "llada random remasking",
        "llada gumbel",
        "llada device statistics",
        "dream",
        "dream device statistics",
        "dream maskgit_plus",
        "dream topk_margin",
        "dream maskgit_plus device statistics",
    ]


def test_a_non_finite_row_is_refused_with_its_position() raises:
    var names = _names()
    for path in range(PATHS):
        for kind in range(3):
            var raised = False
            try:
                _ = _run(path, 1, kind)
            except e:
                raised = True
                assert_equal(
                    String(e),
                    "non-finite logits at canvas position 1, step 1",
                    names[path] + ", kind " + String(kind),
                )
            assert_true(
                raised, names[path] + ", kind " + String(kind) + ": no error"
            )


def test_the_position_counts_from_the_canvas_start() raises:
    # Row 3 is the last of the four canvas positions, the prompt has two tokens.
    for path in range(PATHS):
        with assert_raises(
            contains="non-finite logits at canvas position 3, step 1"
        ):
            _ = _run(path, 3, NAN_ROW)


def test_finite_rows_complete_and_commit_the_scripted_token() raises:
    for path in range(PATHS):
        var res = _run(path, NONE, NAN_ROW)
        assert_equal(res.stop_reason, "complete")
        for i in range(len(_prompt()), len(res.tokens)):
            assert_equal(res.tokens[i], GOOD)


def test_a_single_negative_infinity_follows_the_confidence_it_leaves() raises:
    # exp(-inf) is 0, so LLaDA's confidence 1 / z stays a number and the token
    # is simply never chosen. Dream's default confidence is the negative
    # entropy, whose term for that logit is 0 * -inf = NaN: the row has no
    # confidence, and the loop says so.
    for path in range(PATHS):
        if path == DREAM_CPU or path == DREAM_STATS:
            with assert_raises(
                contains="non-finite logits at canvas position 1, step 1"
            ):
                _ = _run(path, 1, ONE_NEG_INF)
            continue
        var res = _run(path, 1, ONE_NEG_INF)
        assert_equal(res.stop_reason, "complete")
        for i in range(len(_prompt()), len(res.tokens)):
            assert_equal(res.tokens[i], GOOD)


def test_the_loops_take_the_mask_id_from_the_model() raises:
    # A mask id of 7 sits in the model's config and nowhere else: the config
    # of the run no longer carries one (it defaulted to LLaDA's 126336 for
    # every architecture).
    for arch in List[String](["llada", "dream"]):
        var model = BadRowModel(arch, NONE, NAN_ROW, False, mask=7)
        var res = generate(model, _prompt(), _config(arch))
        assert_equal(len(model.first_call), 6)
        for i in range(len(_prompt())):
            assert_equal(model.first_call[i], _prompt()[i])
        for i in range(len(_prompt()), 6):
            assert_equal(model.first_call[i], 7, arch + ": canvas position")
        for i in range(len(_prompt()), len(res.tokens)):
            assert_equal(res.tokens[i], GOOD, arch + ": resolved token")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
