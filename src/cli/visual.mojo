"""Byte rendering and redraw policy for --visual."""

from mojo_dllm.diffusion.sampler import StepObserver
from mojo_dllm.tokenizer.bpe import Tokenizer


def frame_style(is_tty: Bool, no_color: String) -> Tuple[Bool, Bool]:
    return (is_tty and no_color == "", is_tty)


def render_frame(
    tok: Tokenizer,
    tokens: List[Int],
    committed: List[Int],
    step: Int,
    total: Int,
    start: Int,
    mask_id: Int,
    color: Bool,
    clear: Bool,
) raises -> String:
    var esc = chr(27)
    var out = esc + "[2J" + esc + "[H" if clear else String("")
    out += (
        esc
        + "[38;5;80mmojo-dllm"
        + esc
        + "[0m" if color else String("mojo-dllm")
    )
    out += "  denoising step " + String(step) + " / " + String(total) + "\n\n"
    for p in range(start, len(tokens)):
        var t = tokens[p]
        if t == mask_id:
            out += esc + "[38;5;30m░" + esc + "[0m" if color else String("░")
            continue
        if t == tok.eos_id or t == tok.eot_id:
            out += esc + "[38;5;240m·" + esc + "[0m" if color else String("·")
            continue
        var piece = tok.decode([t])
        var fresh = False
        for c in committed:
            if c == p:
                fresh = True
        if color and fresh:
            out += esc + "[1;38;5;208m" + piece + esc + "[0m"
        else:
            out += piece
    return out + "\n\n"


struct CanvasPrinter(StepObserver):
    """Owns the decoder and remembers the last step for an early-stop frame."""

    var tok: Tokenizer
    var start: Int
    var mask_id: Int
    var enabled: Bool
    var interval: Int
    var color: Bool
    var clear: Bool
    var tokens: List[Int]
    var committed: List[Int]
    var step: Int
    var total: Int
    var drawn: Bool

    def __init__(
        out self,
        var tok: Tokenizer,
        start: Int,
        mask_id: Int,
        enabled: Bool,
        interval: Int = 1,
        is_tty: Bool = True,
        no_color: String = "",
    ):
        self.tok = tok^
        self.start = start
        self.mask_id = mask_id
        self.enabled = enabled
        self.interval = interval
        var style = frame_style(is_tty, no_color)
        self.color = style[0]
        self.clear = style[1]
        self.tokens = List[Int]()
        self.committed = List[Int]()
        self.step = 0
        self.total = 0
        self.drawn = False

    def _draw(mut self) raises:
        print(
            render_frame(
                self.tok,
                self.tokens,
                self.committed,
                self.step,
                self.total,
                self.start,
                self.mask_id,
                self.color,
                self.clear,
            ),
            end="",
            flush=True,
        )
        self.drawn = True

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        if not self.enabled:
            return
        self.tokens = tokens.copy()
        self.committed = committed.copy()
        self.step = step
        self.total = total
        self.drawn = False
        if step % self.interval == 0 or step == total:
            self._draw()

    def finish(mut self, tokens: List[Int]) raises:
        if self.enabled and self.step > 0 and not self.drawn:
            self.tokens = tokens.copy()
            self._draw()
