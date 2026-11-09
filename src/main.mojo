"""Command line for mojo-dllm.

    mojo-dllm run      --model M --prompt TEXT [--max-tokens N] [--steps N] [--block-length N]
                       [--seed N] [--temperature T] [--threads N] [--device cpu|gpu]
                       [--remasking S] [--threshold P] [--dual-cache]
                       [--no-chat] [--full-logits] [--no-early-stop] [--no-suppress-mask]
                       [--visual] [--visual-interval N] [--verbose]
                       [--dump-step-stats] [--json]
                       [--eb-steps N] [--eb-t-min T] [--eb-t-max T] [--eb-bound B]
                       [--eb-stability N] [--eb-confidence C]
    mojo-dllm tokenize --model M --prompt TEXT [--no-chat]
    mojo-dllm inspect  MODEL.gguf
    mojo-dllm logits   --model M --tokens 1,2,3 --rows 0,2 --out logits.f32 [--threads N] [--device cpu|gpu]
                       [--path unified|cached] [--prev FILE --prev-temp-inv X]
"""

from std.ffi import ErrNo, external_call, c_int
from std.math import isfinite
from std.os import getenv
from std.sys import argv, exit, num_logical_cores
from std.time import perf_counter_ns

from cli.args import Args, parse_csv_ints, unknown_flag_error
from cli.visual import CanvasPrinter, prints_answer
from mojo_dllm.formats.gguf import GGUFFile, GGUF_TYPE_ARRAY
from mojo_dllm.diffusion.sampler import (
    GenConfig,
    GenResult,
    generate_observed,
)
from mojo_dllm.gpu.model import GpuDiffusionLM
from mojo_dllm.models.diffusion_gemma import DiffusionGemma
from mojo_dllm.models.transformer import (
    DenoisingModel,
    DiffusionLM,
    ModelConfig,
    GEMMA_ARCH,
)
from mojo_dllm.sys.mem import Bytes, F32Ptr, Floats
from mojo_dllm.sys.mmap import Mapping
from mojo_dllm.tokenizer.bpe import Tokenizer
from mojo_dllm.tokenizer.gemma4 import gemma4_chat_prompt

comptime VERSION = "0.1.0"


struct CommandSpec(Movable):
    """The flags one command declares, and what it requires."""

    var values: List[String]
    var booleans: List[String]
    var required: List[String]
    var takes_path: Bool

    def __init__(
        out self,
        var values: List[String],
        var booleans: List[String],
        var required: List[String],
        takes_path: Bool = False,
    ):
        self.values = values^
        self.booleans = booleans^
        self.required = required^
        self.takes_path = takes_path

    def declared(self) -> List[String]:
        var names = List[String]()
        for v in self.values:
            names.append(v)
        for b in self.booleans:
            names.append(b)
        return names^


def command_spec(command: String) raises -> CommandSpec:
    if command == "run":
        return CommandSpec(
            [
                "model",
                "prompt",
                "max-tokens",
                "steps",
                "block-length",
                "temperature",
                "remasking",
                "threshold",
                "seed",
                "threads",
                "device",
                "visual-interval",
                "eb-steps",
                "eb-t-min",
                "eb-t-max",
                "eb-bound",
                "eb-stability",
                "eb-confidence",
            ],
            boolean_flags(),
            ["model", "prompt"],
        )
    if command == "tokenize":
        return CommandSpec(
            ["model", "prompt"], ["no-chat"], ["model", "prompt"]
        )
    if command == "inspect":
        return CommandSpec([], [], [], takes_path=True)
    if command == "logits":
        return CommandSpec(
            [
                "model",
                "tokens",
                "rows",
                "out",
                "threads",
                "device",
                "path",
                "prev",
                "prev-temp-inv",
            ],
            [],
            ["model", "tokens", "rows", "out"],
        )
    raise Error("unknown command: " + command)


def command_usage(command: String) raises -> String:
    """The usage lines of one command, without a trailing newline."""
    if command == "run":
        return (
            "  mojo-dllm run --model M --prompt TEXT [--max-tokens 128]"
            " [--steps 128] [--block-length 32]\n"
            + "                [--seed 42] [--temperature 0] [--threads N]"
            " [--device cpu|gpu]\n"
            + "                [--remasking S] [--threshold P] [--dual-cache]\n"
            + "                [--no-chat] [--full-logits] [--no-early-stop]"
            " [--no-suppress-mask] [--visual] [--visual-interval N] [--verbose]"
            " [--dump-step-stats] [--json]\n"
            + "                [--eb-steps 48] [--eb-t-min 0.4] [--eb-t-max"
            " 0.8] [--eb-bound 0.1]\n"
            + "                [--eb-stability 1] [--eb-confidence 0.005]"
        )
    if command == "tokenize":
        return "  mojo-dllm tokenize --model M --prompt TEXT [--no-chat]"
    if command == "inspect":
        return "  mojo-dllm inspect MODEL.gguf"
    if command == "logits":
        return (
            "  mojo-dllm logits --model M --tokens IDS --rows POS --out FILE"
            " [--threads N] [--device cpu|gpu]\n"
            + "                   [--path unified|cached] [--prev FILE"
            " --prev-temp-inv X]"
        )
    raise Error("unknown command: " + command)


def usage_for(command: String) raises -> String:
    return "usage:\n" + command_usage(command)


def version_line() -> String:
    """The one line `mojo-dllm --version` prints."""
    return "mojo-dllm " + VERSION


def usage() raises -> String:
    return (
        "mojo-dllm "
        + VERSION
        + ": diffusion language model inference in Mojo\n\n"
        + "usage:\n"
        + command_usage("run")
        + "\n"
        + command_usage("tokenize")
        + "\n"
        + command_usage("inspect")
        + "\n"
        + command_usage("logits")
        + "\n"
        + "\nrun mojo-dllm COMMAND --help for one command's usage\n"
        + "run mojo-dllm --version to print the version\n"
    )


def is_top_level_help(command: String) -> Bool:
    return (
        command == ""
        or command == "help"
        or command == "--help"
        or command == "-h"
    )


def command_known(command: String) -> Bool:
    try:
        _ = command_spec(command)
        return True
    except:
        return False


def _equals_advice(a: Args, spec: CommandSpec) -> String:
    """`--model=x` is an unknown flag; say how a value is given instead."""
    var name = a.unknown[0]
    var at = name.find("=")
    if at < 0:
        return String("")
    var flag = String(name[byte=:at])
    for v in spec.values:
        if v == flag:
            return (
                "unknown flag --"
                + name
                + " for '"
                + a.command
                + "'\nflags take their value as the next argument: --"
                + flag
                + " "
                + String(name[byte = at + 1 :])
            )
    return String("")


def _number_problems(a: Args) raises -> String:
    """The first numeric flag whose value can never be valid, or "".

    Run before any model is mapped: `--steps abc` and `--threads 0` are
    wrong whatever the model is.
    """
    for flag in [
        "max-tokens",
        "steps",
        "block-length",
        "threads",
        "seed",
        "eb-steps",
        "eb-stability",
    ]:
        if not a.has(flag):
            continue
        var raw = a.values[flag]
        var n: Int
        try:
            n = Int(raw)
        except:
            return "flag --" + flag + " needs an integer, got '" + raw + "'"
        if flag == "eb-stability":
            if n < 0:
                return "flag --" + flag + " must be at least 0, got " + raw
        elif flag != "seed" and n < 1:
            return "flag --" + flag + " must be at least 1, got " + raw
    for flag in ["eb-t-min", "eb-t-max", "eb-bound", "eb-confidence"]:
        if not a.has(flag):
            continue
        var raw = a.values[flag]
        var x = Float64(0)
        var parsed = True
        try:
            x = Float64(raw)
        except:
            parsed = False
        if not parsed or not isfinite(x):
            return (
                "flag --" + flag + " needs a finite number, got '" + raw + "'"
            )
        if flag == "eb-t-max":
            if x <= 0:
                return "flag --" + flag + " must be above 0, got " + raw
        elif x < 0:
            return "flag --" + flag + " must be at least 0, got " + raw
    if a.has("temperature"):
        var raw = a.values["temperature"]
        var t = Float64(0)
        var parsed = True
        try:
            t = Float64(raw)
        except:
            parsed = False
        if not parsed or not isfinite(t):
            return "flag --temperature needs a finite number, got '" + raw + "'"
        if t < 0:
            return "flag --temperature must be at least 0, got " + raw
    for flag in ["tokens", "rows"]:
        if not a.has(flag):
            continue
        var raw = a.values[flag]
        try:
            _ = parse_csv_ints(raw)
        except:
            return (
                "flag --"
                + flag
                + " needs comma-separated integers, got '"
                + raw
                + "'"
            )
    return String("")


def check_args(a: Args, spec: CommandSpec) raises -> String:
    """The usage error for the parsed arguments, or "" when they are valid."""
    var see = "\nsee: mojo-dllm " + a.command + " --help"
    if a.missing_value.byte_length() > 0:
        return "flag --" + a.missing_value + " needs a value" + see
    if len(a.unknown) > 0:
        var advice = _equals_advice(a, spec)
        if advice.byte_length() > 0:
            return advice + see
        return unknown_flag_error(a.command, a.unknown[0], spec.declared())
    var extra = 1 if spec.takes_path else 0
    if len(a.positional) > extra:
        return (
            "unexpected argument '"
            + a.positional[extra]
            + "' for '"
            + a.command
            + "'"
            + see
        )
    for name in spec.required:
        if not a.has(name):
            return (
                "missing required flag --"
                + name
                + " for '"
                + a.command
                + "'\n"
                + usage_for(a.command)
            )
    if spec.takes_path and len(a.positional) == 0:
        return (
            "missing model path for '"
            + a.command
            + "'\n"
            + usage_for(a.command)
        )
    var number = _number_problems(a)
    if number.byte_length() > 0:
        return number + see
    return String("")


def model_error(message: String) -> String:
    """Adds the next step to a model-file error. Other errors pass through."""
    var cannot_open = "cannot open file: "
    if message.startswith(cannot_open):
        var shown = "cannot open model file " + String(
            message[byte = cannot_open.byte_length() :]
        )
        # The message ends in the reason, which never contains ": ", so the
        # suffix test cannot be fooled by the path.
        if message.endswith(": " + String(ErrNo.ENOENT)):
            return shown + "\nto download a model, run: ./mojo-dllm pull"
        return shown
    var bad_magic = "not a GGUF file (bad magic): "
    if message.startswith(bad_magic):
        var path = String(message[byte = bad_magic.byte_length() :])
        return (
            path
            + " is not a GGUF file (bad magic).\na failed download leaves an"
            " HTML error page with a .gguf name; delete the file and run"
            " ./mojo-dllm pull again"
        )
    return message


def peak_rss_bytes() raises -> Int:
    """Peak resident set size in bytes (getrusage ru_maxrss, which Linux reports in KiB).
    """
    var buf = Bytes(256)
    _ = external_call["getrusage", c_int](c_int(0), buf.addr)
    # struct rusage: two struct timeval (16 bytes each), then long ru_maxrss.
    return Int(buf.u8().unsafe_bitcast[Int64]().unsafe_load(4)) * 1024


def chat_prompt(arch: String, text: String) raises -> String:
    """One user turn in the model's chat template (its tokenizer.chat_template).

    LLaDA-8B-Instruct: Llama-3 style headers after <|startoftext|>.
    Dream-v0-Instruct: Qwen2 ChatML with the default system message.
    DiffusionGemma: `<bos>`, a user turn and an open model turn; the Jinja
    template in the file is reproduced by `gemma4_chat_prompt`.
    """
    if arch == "diffusion-gemma":
        return gemma4_chat_prompt(["user"], [text])
    if arch == "dream":
        return (
            "<|im_start|>system\nYou are a helpful"
            " assistant.<|im_end|>\n<|im_start|>user\n"
            + String(text.strip())
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    return (
        "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n"
        + String(text.strip())
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def json_str(s: String) -> String:
    """A JSON string literal. Multi-byte UTF-8 passes through unchanged."""
    var out = List[UInt8]()
    out.append(34)
    var hexd = "0123456789abcdef".as_bytes()
    for b in s.as_bytes():
        var c = Int(b)
        if c == 34 or c == 92:
            out.append(92)
            out.append(b)
        elif c == 10:
            out.append(92)
            out.append(110)
        elif c == 13:
            out.append(92)
            out.append(114)
        elif c == 9:
            out.append(92)
            out.append(116)
        elif c < 32:
            for q in "\\u00".as_bytes():
                out.append(q)
            out.append(hexd[c // 16])
            out.append(hexd[c % 16])
        else:
            out.append(b)
    out.append(34)
    return String(StringSpan(unsafe_from_utf8=Span(out)))


def encode_prompt(
    mut tok: Tokenizer, arch: String, a: Args
) raises -> List[Int]:
    var text = a.get("prompt")
    if a.has("no-chat"):
        return tok.encode(text, add_bos=tok.add_bos)
    return tok.encode(chat_prompt(arch, text))


def cmd_tokenize(a: Args) raises:
    var g = GGUFFile(a.get("model"))
    var tok = Tokenizer(g)
    var ids = encode_prompt(tok, g.architecture(), a)
    var s = String("")
    for i in range(len(ids)):
        s += (", " if i else "") + String(ids[i])
    print("[" + s + "]")


def eb_flags() -> List[String]:
    """The flags only DiffusionGemma's entropy-bounded denoiser reads."""
    return [
        "eb-steps",
        "eb-t-min",
        "eb-t-max",
        "eb-bound",
        "eb-stability",
        "eb-confidence",
    ]


def check_arch_flags(a: Args, arch: String) raises:
    """Refuse a flag the model's architecture ignores: a run that quietly
    takes another number of steps than asked for is a wrong answer."""
    if arch == "diffusion-gemma":
        if a.has("steps"):
            raise Error(
                "--steps does not apply to diffusion-gemma; use --eb-steps"
            )
        return
    for flag in eb_flags():
        if a.has(flag):
            raise Error(
                "--" + flag + " applies to diffusion-gemma only, not " + arch
            )


def stop_json(res: GenResult) -> String:
    """The JSON members that say how many steps ran and why the run ended."""
    return (
        ', "steps_taken": '
        + String(res.steps_taken)
        + ', "stop_reason": "'
        + res.stop_reason
        + '"'
    )


def cmd_run(a: Args) raises:
    var interval: Int
    try:
        interval = a.int_or("visual-interval", 1)
    except:
        raise Error("--visual-interval needs an integer")
    if interval < 1:
        raise Error("--visual-interval must be at least 1")
    if a.has("visual-interval") and not a.has("visual"):
        raise Error("--visual-interval needs --visual")
    var threads = a.int_or("threads", num_logical_cores())
    var cfg = GenConfig()
    cfg.gen_length = a.int_or("max-tokens", 128)
    cfg.steps = a.int_or("steps", cfg.gen_length)
    cfg.block_length = a.int_or("block-length", min(32, cfg.gen_length))
    cfg.seed = a.int_or("seed", 42)
    cfg.temperature = a.float_or("temperature", 0.0)
    cfg.remasking = a.get_or("remasking", "low_confidence")
    cfg.threshold = a.float_or("threshold", 0.0)
    cfg.full_logits = a.has("full-logits")
    cfg.dual_cache = a.has("dual-cache")
    cfg.suppress_mask = not a.has("no-suppress-mask")
    cfg.threads = threads
    cfg.eb_steps = a.int_or("eb-steps", cfg.eb_steps)
    cfg.eb_t_min = a.float_or("eb-t-min", cfg.eb_t_min)
    cfg.eb_t_max = a.float_or("eb-t-max", cfg.eb_t_max)
    cfg.eb_bound = a.float_or("eb-bound", cfg.eb_bound)
    cfg.eb_stability = a.int_or("eb-stability", cfg.eb_stability)
    cfg.eb_confidence = a.float_or("eb-confidence", cfg.eb_confidence)
    var t0 = perf_counter_ns()
    var g = GGUFFile(a.get("model"))
    check_arch_flags(a, g.architecture())
    if g.architecture() == "diffusion-gemma":
        cfg.steps = cfg.eb_steps
    var tok = Tokenizer(g)
    var prompt = encode_prompt(tok, g.architecture(), a)
    _ = g^
    var tok_s = Float64(perf_counter_ns() - t0) / 1e9
    var device = a.get_or("device", "cpu")
    var max_tokens = len(prompt) + cfg.gen_length
    if device == "gpu":
        var gm = GpuDiffusionLM(
            a.get("model"), max_tokens=max_tokens, kv_cache=cfg.dual_cache
        )
        _run(
            gm,
            a,
            cfg^,
            prompt,
            tok^,
            tok_s,
            device,
            "gpu (" + gm.ctx.name() + ")",
            threads,
        )
    elif device == "cpu":
        var cm = DiffusionLM(
            a.get("model"),
            threads=threads,
            max_tokens=max_tokens,
            kv_cache=cfg.dual_cache,
        )
        _run(
            cm,
            a,
            cfg^,
            prompt,
            tok^,
            tok_s,
            device,
            "cpu   threads: " + String(threads),
            threads,
        )
    else:
        raise Error("unknown device: " + device + " (expected cpu or gpu)")


def _run[
    M: DenoisingModel
](
    mut model: M,
    a: Args,
    var cfg: GenConfig,
    prompt: List[Int],
    var tok: Tokenizer,
    tok_s: Float64,
    device: String,
    device_line: String,
    threads: Int,
) raises:
    var verbose = a.has("verbose")
    cfg.mask_id = model.config().mask_id
    if verbose:
        print(
            "model:",
            model.model_name(),
            "(" + model.config().arch + ")",
        )
        print("device:", device_line)
        print(
            "prompt tokens:",
            len(prompt),
            "  generation tokens:",
            cfg.gen_length,
        )
        if model.config().arch == "dream":
            print(
                "diffusion steps:",
                cfg.steps,
                "  schedule: Dream timesteps, eps",
                cfg.eps,
            )
        elif model.config().arch == "diffusion-gemma":
            print(
                "diffusion steps: at most",
                cfg.eb_steps,
                "  entropy bound:",
                cfg.eb_bound,
                "  temperature:",
                cfg.eb_t_max,
                "down to",
                cfg.eb_t_min,
                "  stop: stable for",
                cfg.eb_stability,
                "and mean entropy below",
                cfg.eb_confidence,
            )
        else:
            print(
                "diffusion steps:",
                cfg.steps,
                "  block length:",
                cfg.block_length,
                "  remasking:",
                cfg.remasking,
            )
        print("load time:", model.load_time(), "s (tokenizer", tok_s, "s)")
    cfg.early_stop = not a.has("no-early-stop")
    cfg.stop_ids = tok.eog_ids()
    var is_tty = external_call["isatty", c_int](c_int(1)) != 0
    var printer = CanvasPrinter(
        tok^,
        len(prompt),
        model.config().mask_id,
        a.has("visual"),
        a.int_or("visual-interval", 1),
        is_tty,
        getenv("NO_COLOR"),
    )
    var res = generate_observed(model, prompt, cfg, printer)
    printer.finish(res.tokens)
    var gen = List[Int]()
    for i in range(len(prompt), len(res.tokens)):
        var t = res.tokens[i]
        if printer.tok.is_eog(t):
            break
        gen.append(t)
    var text = printer.tok.decode(gen)
    var tm = model.stats()
    var fw = Float64(max(tm.forwards, 1))
    if a.has("dump-step-stats"):
        for i in range(len(res.step_ms)):
            print("step", i + 1, ":", res.step_ms[i], "ms")
    if a.has("json"):
        var steps_json = String("[")
        for i in range(len(res.step_ms)):
            steps_json += (", " if i else "") + String(res.step_ms[i])
        steps_json += "]"
        print(
            "{"
            + '"runtime": "mojo-dllm", "version": "'
            + VERSION
            + '", "arch": "'
            + model.config().arch
            + '", "device": "'
            + device
            + '"'
            + ', "threads": '
            + String(threads)
            + ', "prompt_tokens": '
            + String(len(prompt))
            + ', "gen_length": '
            + String(cfg.gen_length)
            + ', "steps": '
            + String(cfg.steps)
            + ', "block_length": '
            + String(cfg.block_length)
            + ', "full_logits": '
            + ("true" if cfg.full_logits else "false")
            + ', "stopped_early": '
            + ("true" if res.stopped_early else "false")
            + ', "forward_passes": '
            + String(res.forward_passes)
            + stop_json(res)
            + ', "load_s": '
            + String(model.load_time())
            + ', "tokenizer_s": '
            + String(tok_s)
            + ', "generate_s": '
            + String(res.seconds)
            + ', "ms_per_step": '
            + String(res.seconds * 1000.0 / Float64(max(res.forward_passes, 1)))
            + ', "forward_ms": '
            + String(Float64(tm.total_ns) / 1e6 / fw)
            + ', "gemm_ms_per_forward": '
            + String(Float64(tm.gemm_ns) / 1e6 / fw)
            + ', "attention_ms_per_forward": '
            + String(Float64(tm.attn_ns) / 1e6 / fw)
            + ', "lm_head_ms_per_forward": '
            + String(Float64(tm.head_ns) / 1e6 / fw)
            + ', "peak_rss_bytes": '
            + String(peak_rss_bytes())
            + ', "step_ms": '
            + steps_json
            + ', "tokens": ['
            + _csv(res.tokens)
            + "]"
            + ', "text": '
            + json_str(text)
            + "}"
        )
        return
    if prints_answer(a.has("visual"), is_tty):
        print(text)
    if verbose:
        print()
        print(
            "generation:",
            res.seconds,
            "s for",
            res.forward_passes,
            "denoising steps",
        )
        print(
            "per step:",
            res.seconds * 1000.0 / Float64(max(res.forward_passes, 1)),
            "ms   (forward pass",
            Float64(tm.total_ns) / 1e6 / fw,
            "ms)",
        )
        print(
            "  gemm",
            Float64(tm.gemm_ns) / 1e6 / fw,
            "ms   attention",
            Float64(tm.attn_ns) / 1e6 / fw,
            "ms   lm head",
            Float64(tm.head_ns) / 1e6 / fw,
            "ms",
        )
        print("tokens/s:", Float64(cfg.gen_length) / res.seconds)
        print("peak RSS:", Float64(peak_rss_bytes()) / 1e9, "GB")


def _csv(xs: List[Int]) -> String:
    var s = String("")
    for i in range(len(xs)):
        s += ("," if i else "") + String(xs[i])
    return s


def _gemma_config_text(g: GGUFFile) -> String:
    """The `config:` block of `inspect` for a diffusion-gemma file: what the
    reader took from it, or the reason it refuses the file. `inspect` lists a
    file either way."""
    try:
        var c = ModelConfig(g)
        var sliding = 0
        var global_layers = String("")
        for i in range(c.n_layers):
            if c.is_sliding(i):
                sliding += 1
            else:
                if global_layers.byte_length() > 0:
                    global_layers += ", "
                global_layers += String(i)
        if global_layers.byte_length() == 0:
            global_layers = "none"
        var text = String("config:\n")
        text += (
            "  blocks: "
            + String(c.n_layers)
            + " (sliding "
            + String(sliding)
            + ", global "
            + String(c.n_layers - sliding)
            + ")\n"
        )
        text += "  global layers: " + global_layers + "\n"
        text += "  hidden size: " + String(c.hidden) + "\n"
        text += "  heads: " + String(c.n_heads) + "\n"
        text += (
            "  kv heads: "
            + String(c.n_kv_heads_swa)
            + " sliding, "
            + String(c.n_kv_heads)
            + " global\n"
        )
        text += (
            "  head size: "
            + String(c.head_dim_swa)
            + " sliding, "
            + String(c.head_dim)
            + " global\n"
        )
        text += (
            "  rope base: "
            + String(c.theta_swa)
            + " sliding, "
            + String(c.theta)
            + " global\n"
        )
        text += "  sliding window: " + String(c.sliding_window) + "\n"
        text += "  dense ffn: " + String(c.ffn) + "\n"
        text += (
            "  experts: "
            + String(c.n_experts)
            + " ("
            + String(c.n_experts_used)
            + " used), ffn "
            + String(c.expert_ffn)
            + "\n"
        )
        text += (
            "  vocabulary: "
            + String(c.vocab)
            + ", mask token "
            + String(c.mask_id)
            + "\n"
        )
        text += "  final logit softcap: " + String(c.softcap) + "\n"
        text += "  canvas length: " + String(c.canvas) + "\n"
        text += "  context length: " + String(c.ctx) + "\n"
        return text
    except e:
        return "config: refused: " + String(e) + "\n"


def inspect_text(path: String) raises -> String:
    """What `inspect` prints, as one string."""
    var g = GGUFFile(path)
    var text = String("")
    text += "file: " + path + "\n"
    text += "gguf version: " + String(g.version) + "\n"
    text += "architecture: " + g.get_str_or("general.architecture", "?") + "\n"
    text += "name: " + g.get_str_or("general.name", "?") + "\n"
    text += "tensors: " + String(g.n_tensors()) + "\n"
    text += "metadata keys: " + String(len(g.kvs)) + "\n"
    if g.get_str_or("general.architecture", "?") == GEMMA_ARCH:
        text += "\n" + _gemma_config_text(g)
    text += "\nmetadata:\n"
    for i in range(len(g.kvs)):
        text += "  " + g.kvs[i].key + " = " + g.value_str(i) + "\n"
    var counts = Dict[String, Int]()
    for i in range(g.n_tensors()):
        var tn = g.tensors[i].type_name()
        counts[tn] = counts.get(tn, 0) + 1
    text += "\ntensor types:\n"
    for e in counts.items():
        text += "  " + e.key + ": " + String(e.value) + "\n"
    text += "\ntensors:\n"
    for i in range(g.n_tensors()):
        var t = g.tensors[i]
        text += (
            "  " + t.name + "  " + t.type_name() + "  " + t.shape_str() + "\n"
        )
    return text


def cmd_inspect(a: Args) raises:
    print(inspect_text(a.positional[0]), end="")


def cmd_logits(a: Args) raises:
    var threads = a.int_or("threads", num_logical_cores())
    var tokens = parse_csv_ints(a.get("tokens"))
    var rows = parse_csv_ints(a.get("rows"))
    var device = a.get_or("device", "cpu")
    var arch = GGUFFile(a.get("model")).architecture()
    if arch == GEMMA_ARCH:
        if device != "cpu":
            raise Error(
                GEMMA_ARCH
                + " runs on the CPU only: there is no GPU forward pass yet"
            )
        var inv = Float32(a.float_or("prev-temp-inv", 0.0))
        _logits_gemma(
            a.get("model"),
            threads,
            tokens,
            rows,
            a.get("out"),
            a.get_or("path", "unified"),
            a.get_or("prev", ""),
            inv,
        )
    elif gemma_only_flags(a).byte_length() > 0:
        raise Error(gemma_only_flags(a) + " apply to " + GEMMA_ARCH + " only")
    elif device == "gpu":
        var gm = GpuDiffusionLM(a.get("model"), max_tokens=len(tokens))
        _logits(gm, tokens, rows, a.get("out"))
    elif device == "cpu":
        var cm = DiffusionLM(
            a.get("model"), threads=threads, max_tokens=len(tokens)
        )
        _logits(cm, tokens, rows, a.get("out"))
    else:
        raise Error("unknown device: " + device + " (expected cpu or gpu)")


def gemma_only_flags(a: Args) -> String:
    """The `logits` flags that only a diffusion-gemma model takes, as given."""
    var given = String("")
    for name in ["path", "prev", "prev-temp-inv"]:
        if a.has(name):
            given += ("" if given.byte_length() == 0 else ", ") + "--" + name
    return given


def _logits_gemma(
    path: String,
    threads: Int,
    tokens: List[Int],
    rows: List[Int],
    out_path: String,
    path_kind: String = "unified",
    prev_path: String = "",
    prev_temp_inv: Float32 = 0.0,
) raises:
    """`logits` for diffusion-gemma: one pass over [prompt | canvas].

    --tokens is the prompt followed by the canvas, so the canvas is the last
    `diffusion.canvas_length` tokens and the prompt is whatever precedes it.
    The model scores only the canvas rows, so --rows must name exactly those
    positions, and the file holds [canvas, vocab] f32. `--path unified` runs
    the prompt and canvas in one pass, `cached` prefills the prompt and steps
    the canvas against its K/V. `--prev` names the previous step's canvas
    logits (a file of [canvas, vocab] f32) and `--prev-temp-inv` is 1 / the
    previous temperature: both turn self-conditioning on.
    """
    if (prev_path.byte_length() > 0) != (prev_temp_inv != 0.0):
        raise Error("--prev and --prev-temp-inv go together")
    if path_kind != "unified" and path_kind != "cached":
        raise Error("--path is unified or cached, got '" + path_kind + "'")
    var m = DiffusionGemma(path, threads=threads, max_tokens=len(tokens))
    var c = m.config()
    var first = len(tokens) - c.canvas
    var expected = String("")
    for i in range(c.canvas):
        expected += ("," if i > 0 else "") + String(first + i)
    var given = String("")
    for i in range(len(rows)):
        given += ("," if i > 0 else "") + String(rows[i])
    if first < 0 or given != expected:
        raise Error(
            GEMMA_ARCH
            + " scores the "
            + String(c.canvas)
            + " canvas positions, the last of the tokens: --rows must be "
            + (expected if first >= 0 else "positions inside the tokens")
            + ", got "
            + given
        )
    var out = Floats(c.canvas * c.vocab)
    var n = c.canvas * c.vocab
    var conditioned = prev_path.byte_length() > 0
    if conditioned:
        var prev = Mapping(prev_path)
        if prev.size != n * 4:
            raise Error(
                prev_path
                + " holds "
                + String(prev.size)
                + " bytes, expected "
                + String(n * 4)
                + " (canvas "
                + String(c.canvas)
                + " x vocab "
                + String(c.vocab)
                + " f32)"
            )
        var src = F32Ptr(unsafe_from_address=prev.addr)
        for i in range(n):
            out[i] = src.unsafe_load(i)
    if path_kind == "cached":
        var prompt = List[Int]()
        var canvas = List[Int]()
        for i in range(len(tokens)):
            if i < first:
                prompt.append(tokens[i])
            else:
                canvas.append(tokens[i])
        m.prefill(prompt)
        if conditioned:
            m.step_sc(canvas, out.ptr(), prev_temp_inv)
        else:
            m.step(canvas, out.ptr())
    elif conditioned:
        m.forward_sc(tokens, out.ptr(), prev_temp_inv)
    else:
        m.forward(tokens, out.ptr())
    with open(out_path, "w") as f:
        var p = out.ptr().unsafe_bitcast[UInt8]()
        f.write_bytes(Span(unsafe_ptr=p, length=c.canvas * c.vocab * 4))
    print("wrote", c.canvas, "rows x", c.vocab, "logits")


def _logits[
    M: DenoisingModel
](mut m: M, tokens: List[Int], rows: List[Int], path: String) raises:
    var vocab = m.config().vocab
    var out = Floats(len(rows) * vocab)
    m.forward(tokens, rows, out.ptr())
    with open(path, "w") as f:
        var p = out.ptr().unsafe_bitcast[UInt8]()
        f.write_bytes(Span(unsafe_ptr=p, length=len(rows) * vocab * 4))
    print(
        "wrote",
        len(rows),
        "rows x",
        vocab,
        "logits; load",
        m.load_time(),
        "s",
    )


def boolean_flags() -> List[String]:
    return [
        "verbose",
        "full-logits",
        "no-chat",
        "dump-step-stats",
        "json",
        "visual",
        "no-early-stop",
        "no-suppress-mask",
        "dual-cache",
    ]


struct Decision(Movable):
    """What the arguments call for, before anything is loaded or printed.

    `code` is the exit status: 0 for help or a run to start, 2 for a usage
    error (`is_error`, and `text` is the message). `proceed` means the
    arguments are valid and the command should run.
    """

    var code: Int
    var text: String
    var is_error: Bool
    var proceed: Bool

    def __init__(
        out self, code: Int, text: String, is_error: Bool, proceed: Bool
    ):
        self.code = code
        self.text = text
        self.is_error = is_error
        self.proceed = proceed


def decide(raw: List[String]) raises -> Decision:
    """Help, usage errors and valid arguments, in that order of precedence."""
    var command = raw[1] if len(raw) > 1 else String("")
    if is_top_level_help(command):
        return Decision(0, usage(), False, False)
    if command == "--version":
        return Decision(0, version_line(), False, False)
    if not command_known(command):
        return Decision(
            2,
            "unknown command: " + command + "\nsee: mojo-dllm --help",
            True,
            False,
        )
    var spec = command_spec(command)
    var a = Args(raw, spec.values, spec.booleans)
    if a.has("help"):
        return Decision(0, usage_for(command), False, False)
    var problem = check_args(a, spec)
    if problem.byte_length() > 0:
        return Decision(2, problem, True, False)
    return Decision(0, String(""), False, True)


def execute(raw: List[String]) -> Int:
    """Runs the command line and returns the exit status: 0, 1 or 2."""
    try:
        var d = decide(raw)
        if d.is_error:
            print("ERROR:", d.text)
            return d.code
        if not d.proceed:
            print(d.text)
            return d.code
        var command = raw[1]
        var spec = command_spec(command)
        var a = Args(raw, spec.values, spec.booleans)
        if command == "run":
            cmd_run(a)
        elif command == "tokenize":
            cmd_tokenize(a)
        elif command == "inspect":
            cmd_inspect(a)
        else:
            cmd_logits(a)
        return 0
    except e:
        print("ERROR:", model_error(String(e)))
        return 1


def main():
    var raw = List[String]()
    for s in argv():
        raw.append(String(s))
    var code = execute(raw)
    if code != 0:
        exit(code)
