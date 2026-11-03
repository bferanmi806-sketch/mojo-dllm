from std.ffi import c_int, c_size_t, external_call
from std.testing import assert_equal, assert_true, TestSuite

from cli.args import Args
from cli.visual import CanvasPrinter, frame_style, render_frame
from main import _run
from mojo_dllm.diffusion.sampler import (
    GenConfig,
    StepObserver,
    generate_observed,
)
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.models.transformer import DiffusionLM
from mojo_dllm.sys.mem import Bytes
from mojo_dllm.tokenizer.bpe import Tokenizer

comptime FIX = "tests/fixtures/"


def _tokenizer() raises -> Tokenizer:
    return Tokenizer(GGUFFile(FIX + "llada-tokenizer.gguf"))


def _canvas(mut tok: Tokenizer) raises -> List[Int]:
    var ids = tok.encode("Hello world")
    var tokens = List[Int]()
    tokens.append(tok.mask_id)
    tokens.append(ids[0])
    tokens.append(ids[1])
    tokens.append(tok.eot_id)
    return tokens^


struct _OutputCapture(Movable):
    var read_fd: Int
    var write_fd: Int
    var saved_stdout: Int

    def __init__(out self) raises:
        var fds = Bytes(8)
        if external_call["pipe", c_int](fds.addr) != 0:
            raise Error("pipe failed")
        self.read_fd = Int(fds.i32().unsafe_load(0))
        self.write_fd = Int(fds.i32().unsafe_load(1))
        self.saved_stdout = Int(external_call["dup", c_int](c_int(1)))
        if self.saved_stdout < 0:
            raise Error("dup stdout failed")
        if external_call["dup2", c_int](c_int(self.write_fd), c_int(1)) < 0:
            raise Error("redirect stdout failed")

    def finish(mut self) raises -> String:
        _ = external_call["fflush", c_int](Int(0))
        _ = external_call["dup2", c_int](c_int(self.saved_stdout), c_int(1))
        _ = external_call["close", c_int](c_int(self.saved_stdout))
        _ = external_call["close", c_int](c_int(self.write_fd))
        var data = Bytes(16384)
        var n = Int(
            external_call["read", Int](
                c_int(self.read_fd), data.addr, c_size_t(data.size)
            )
        )
        _ = external_call["close", c_int](c_int(self.read_fd))
        if n < 0:
            raise Error("read captured stdout failed")
        return String(
            StringSpan(unsafe_from_utf8=Span(unsafe_ptr=data.u8(), length=n))
        )


def _capture(
    mut printer: CanvasPrinter,
    tokens: List[Int],
    committed: List[Int],
    step: Int,
    total: Int,
    finish: Bool = False,
) raises -> String:
    var output = _OutputCapture()
    printer.on_step(tokens, committed, step, total)
    if finish:
        printer.finish(tokens)
    return output.finish()


def _capture_steps(
    mut printer: CanvasPrinter,
    tokens: List[Int],
    total: Int,
) raises -> String:
    """Capture a complete observer run through the requested total step."""
    var output = _OutputCapture()
    for step in range(1, total + 1):
        printer.on_step(tokens, [], step, total)
    return output.finish()


def _capture_finish(
    mut printer: CanvasPrinter, tokens: List[Int]
) raises -> String:
    var output = _OutputCapture()
    printer.finish(tokens)
    return output.finish()


def _count(text: String, needle: String) -> Int:
    var found = 0
    var at = 0
    while True:
        var next = text.find(needle, at)
        if next < 0:
            break
        found += 1
        at = next + needle.byte_length()
    return found


def _printer(
    var tok: Tokenizer,
    interval: Int = 1,
    is_tty: Bool = True,
    no_color: String = "",
) -> CanvasPrinter:
    return CanvasPrinter(tok^, 0, tok.mask_id, True, interval, is_tty, no_color)


struct _FirstStep(StepObserver):
    var tokens: List[Int]
    var mask_id: Int
    var first_resolved_step: Int

    def __init__(out self, mask_id: Int):
        self.tokens = List[Int]()
        self.mask_id = mask_id
        self.first_resolved_step = 0

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        if self.first_resolved_step == 0 and tokens[3] != self.mask_id:
            self.tokens = tokens.copy()
            self.first_resolved_step = step


def test_golden_terminal_frame_matches_old_canvas_printer() raises:
    var tok = _tokenizer()
    var tokens = _canvas(tok)
    var got = render_frame(tok, tokens, [1], 3, 8, 0, tok.mask_id, True, True)
    var want = (
        "\u001b[2J\u001b[H\u001b[38;5;80mmojo-dllm\u001b[0m"
        "  denoising step 3 / 8\n\n"
        "\u001b[38;5;30m\u2591\u001b[0m"
        "\u001b[1;38;5;208mHello\u001b[0m world"
        "\u001b[38;5;240m\u00b7\u001b[0m\n\n"
    )
    assert_equal(got, want)
    var printer = _printer(tok^)
    assert_equal(_capture(printer, tokens, [1], 3, 8), want)


def test_no_color_terminal_keeps_clear_and_plain_cells() raises:
    var tok = _tokenizer()
    var tokens = _canvas(tok)
    var got = render_frame(tok, tokens, [1], 3, 8, 0, tok.mask_id, False, True)
    assert_true(got.startswith("\u001b[2J\u001b[H"))
    assert_equal(_count(got, "\u001b["), 2)
    assert_true(got.find("\u001b[38;") < 0)
    assert_true(got.find("\u001b[1;") < 0)
    assert_true(got.endswith("\n\n"))
    var printer = _printer(tok^, 1, True, "1")
    assert_equal(_capture(printer, tokens, [1], 3, 8), got)


def test_non_tty_frame_is_plain_and_ends_with_blank_line() raises:
    var tok = _tokenizer()
    var tokens = _canvas(tok)
    var got = render_frame(tok, tokens, [1], 3, 8, 0, tok.mask_id, False, False)
    assert_equal(got.find("\u001b"), -1)
    assert_true(got.find("denoising step 3 / 8") >= 0)
    assert_true(got.endswith("\n\n"))
    var printer = _printer(tok^, 1, False)
    var captured = _capture(printer, tokens, [1], 3, 8)
    assert_equal(captured, got)


def test_frame_style_respects_tty_and_no_color() raises:
    var tty_color = frame_style(True, "")
    assert_true(tty_color[0])
    assert_true(tty_color[1])
    var tty_no_color = frame_style(True, "1")
    assert_true(not tty_no_color[0])
    assert_true(tty_no_color[1])
    var redirected = frame_style(False, "")
    assert_true(not redirected[0])
    assert_true(not redirected[1])
    var redirected_no_color = frame_style(False, "1")
    assert_true(not redirected_no_color[0])
    assert_true(not redirected_no_color[1])
    var empty_is_unset = frame_style(True, "")
    assert_true(empty_is_unset[0])


def test_disabled_printer_does_not_write() raises:
    var tok = _tokenizer()
    var tokens = _canvas(tok)
    var printer = CanvasPrinter(tok^, 0, tok.mask_id, False)
    assert_equal(_capture(printer, tokens, [], 1, 8).byte_length(), 0)


def test_interval_draw_count_includes_final_step() raises:
    for pair in [(1, 8), (3, 3), (4, 2), (8, 1), (9, 1)]:
        var tok = _tokenizer()
        var tokens = _canvas(tok)
        var printer = _printer(tok^, pair[0])
        var captured = _capture_steps(printer, tokens, 8)
        assert_equal(_count(captured, "\u001b[2J"), pair[1])


def test_finish_draws_skipped_final_canvas_once() raises:
    var tok = _tokenizer()
    var tokens = _canvas(tok)
    var bang = tok.encode("!")[0]
    var printer = _printer(tok^, 4)
    for step in range(1, 4):
        var skipped = _capture(printer, tokens, [], step, 8)
        assert_equal(_count(skipped, "\u001b[2J"), 0)
    var first = _capture(printer, tokens, [1], 4, 8)
    tokens[0] = bang
    var final = _capture(printer, tokens, [], 5, 8, True)
    assert_equal(_count(first, "\u001b[2J"), 1)
    assert_equal(_count(final, "\u001b[2J"), 1)
    assert_true(final.find("denoising step 5 / 8") >= 0)
    assert_true(final.find("!") >= 0)

    var last_draw = _printer(_tokenizer(), 3)
    var completed = _capture(last_draw, tokens, [1], 8, 8)
    assert_equal(_count(completed, "\u001b[2J"), 1)
    assert_equal(_count(_capture_finish(last_draw, tokens), "\u001b[2J"), 0)


def test_run_finishes_when_early_stop_skips_interval() raises:
    var tok = _tokenizer()
    var model = DiffusionLM(FIX + "tiny-llada.gguf", threads=1, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var cfg = GenConfig()
    cfg.gen_length = 8
    cfg.block_length = 8
    cfg.steps = 8
    cfg.mask_id = 511
    cfg.early_stop = False
    cfg.threshold = 1e-30
    var first = _FirstStep(cfg.mask_id)
    _ = generate_observed(model, prompt, cfg, first)
    assert_equal(first.first_resolved_step, 1)
    tok.eot_id = first.tokens[3]

    var args = Args(
        [
            "mojo-dllm",
            "run",
            "--visual",
            "--visual-interval",
            "8",
            "--json",
        ],
        ["visual-interval"],
        ["visual", "json"],
    )
    var run_cfg = GenConfig()
    run_cfg.gen_length = 8
    run_cfg.block_length = 8
    run_cfg.steps = 8
    run_cfg.mask_id = 511
    run_cfg.threshold = 1e-30
    var output = _OutputCapture()
    _run(model, args, run_cfg^, prompt, tok^, 0.0, "cpu", "cpu", 1)
    var text = output.finish()
    var frame_at = text.find("denoising step 1 / 8")
    var json_at = text.find('{"runtime": "mojo-dllm"')
    assert_true(
        frame_at >= 0 and json_at > frame_at,
        "_run must finish a skipped frame before printing JSON",
    )
    assert_true(text.find('"stopped_early": true') > json_at)


def test_non_tty_interval_frames_have_no_escapes_or_clears() raises:
    for pair in [(1, 8), (3, 3), (4, 2), (8, 1), (9, 1)]:
        var tok = _tokenizer()
        var tokens = _canvas(tok)
        var printer = _printer(tok^, pair[0], False)
        var captured = _capture_steps(printer, tokens, 8)
        assert_equal(_count(captured, "\u001b[2J"), 0)
        assert_equal(_count(captured, "denoising step"), pair[1])
        assert_equal(captured.find("\u001b"), -1)
        assert_equal(_count(captured, "\n\n"), pair[1] * 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
