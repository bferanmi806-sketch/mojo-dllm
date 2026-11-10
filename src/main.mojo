"""Command line for mojo-dllm.

    mojo-dllm run      --model M --prompt TEXT [--max-tokens N] [--steps N] [--block-length N]
                       [--seed N] [--temperature T] [--threads N] [--device cpu|gpu]
                       [--remasking low_confidence|random] [--threshold P] [--dual-cache]
                       [--no-chat] [--full-logits] [--no-early-stop] [--visual] [--visual-interval N] [--verbose]
                       [--dump-step-stats] [--json]
                       [--eb-steps N] [--eb-t-min T] [--eb-t-max T] [--eb-bound B]
                       [--eb-stability N] [--eb-confidence C]
    mojo-dllm tokenize --model M --prompt TEXT [--no-chat]
    mojo-dllm inspect  MODEL.gguf
    mojo-dllm logits   --model M --tokens 1,2,3 --rows 0,2 --out logits.f32 [--threads N] [--device cpu|gpu]
                       [--path unified|cached] [--prev FILE --prev-temp-inv X]
"""

from std.ffi import ErrNo, external_call, c_int, get_errno
from std.math import isfinite
from std.os import getenv, remove
from std.os.path import exists, getsize, isdir, isfile
from std.sys import argv, exit, num_logical_cores, stderr
from std.time import perf_counter_ns

from cli.args import Args, parse_csv_ints, unknown_flag_error
from cli.visual import CanvasPrinter, prints_answer
from mojo_dllm.formats.gguf import GGUFFile, GGUF_TYPE_ARRAY
from mojo_dllm.formats.utf8 import utf8_first_invalid
from mojo_dllm.diffusion.sampler import (
    GenConfig,
    GenResult,
    float_flag_rule,
    generate_observed,
    int_flag_rule,
)
from mojo_dllm.gpu.model import GpuDiffusionLM
from mojo_dllm.models.diffusion_gemma import DiffusionGemma
from mojo_dllm.models.transformer import (
    DenoisingModel,
    DiffusionLM,
    ModelConfig,
    Timings,
    GEMMA_ARCH,
    gemma_run_refusal,
)
from mojo_dllm.sys.mem import Bytes, F32Ptr, Floats, U8Ptr
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
            + "                [--remasking low_confidence|random] [--threshold"
            " P] [--dual-cache]\n"
            + "                [--no-chat] [--full-logits] [--no-early-stop]"
            " [--visual] [--visual-interval N] [--verbose]"
            " [--dump-step-stats] [--json]\n"
            + "                [--eb-steps 48] [--eb-t-min 0.4] [--eb-t-max"
            " 0.8] [--eb-bound 0.1]\n"
            + "                [--eb-stability 1] [--eb-confidence 0.005]\n"
            + "                (the eb-* flags have no effect until run accepts"
            " diffusion-gemma (#277))"
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


def _given(a: Args, flag: String) raises -> Bool:
    """True when the command declares `flag` and the command line sets it. The
    numeric checks run for every command, so a flag only some commands declare
    must not be looked up on the others."""
    return a.declares(flag) and a.has(flag)


def remasking_names() -> List[String]:
    """Every `--remasking` value some architecture takes: LLaDA the first two,
    Dream the last three (`low_confidence` means `entropy` there)."""
    return [
        "low_confidence",
        "random",
        "entropy",
        "maskgit_plus",
        "topk_margin",
    ]


def schedule_problem(a: Args) raises -> String:
    """Why the block length or the steps cannot split `--max-tokens`, or "".

    LLaDA decodes `--max-tokens` in blocks of `--block-length`, and spreads
    `--steps` evenly over the blocks. The sampler refuses a split that does
    not divide, but only after the model is loaded; the numbers are known
    from the flags (and the defaults) alone.
    """
    var gen = Int(a.get_or("max-tokens", "128"))
    var explicit = a.has("block-length")
    var block = Int(a.get("block-length")) if explicit else min(32, gen)
    var steps = Int(a.get_or("steps", String(gen)))
    if gen % block != 0:
        if explicit:
            return (
                "flag --max-tokens "
                + String(gen)
                + " is not a multiple of --block-length "
                + String(block)
            )
        return (
            "flag --max-tokens "
            + String(gen)
            + " is not a multiple of the default block length "
            + String(block)
            + "; pass --block-length with a divisor of it"
        )
    var blocks = gen // block
    if steps % blocks != 0:
        return (
            "flag --steps "
            + String(steps)
            + " is not a multiple of the number of blocks "
            + String(blocks)
            + " (--max-tokens "
            + String(gen)
            + " / block length "
            + String(block)
            + ")"
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
        if not _given(a, flag):
            continue
        var raw = a.get(flag)
        var n: Int
        try:
            n = Int(raw)
        except:
            return "flag --" + flag + " needs an integer, got '" + raw + "'"
        var rule = int_flag_rule(flag, n)
        if rule.byte_length() > 0:
            return "flag --" + flag + " " + rule + ", got " + raw
    for flag in [
        "eb-t-min",
        "eb-t-max",
        "eb-bound",
        "eb-confidence",
        "temperature",
        "threshold",
        "prev-temp-inv",
    ]:
        if not _given(a, flag):
            continue
        var raw = a.get(flag)
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
        var rule = float_flag_rule(flag, x)
        if rule.byte_length() > 0:
            return "flag --" + flag + " " + rule + ", got " + raw
    if _given(a, "device"):
        var device = a.get("device")
        if device != "cpu" and device != "gpu":
            return "flag --device must be cpu or gpu, got '" + device + "'"
    if _given(a, "path"):
        var kind = a.get("path")
        if kind != "unified" and kind != "cached":
            return "flag --path must be unified or cached, got '" + kind + "'"
    if _given(a, "remasking"):
        var name = a.get("remasking")
        var known = False
        for candidate in remasking_names():
            if candidate == name:
                known = True
        if not known:
            return (
                "flag --remasking must be one of "
                + ", ".join(remasking_names())
                + ", got '"
                + name
                + "'"
            )
        if name == "random" and a.has("threshold"):
            if Float64(a.get("threshold")) > 0:
                return "flag --threshold needs low_confidence remasking"
    for flag in ["tokens", "rows"]:
        if not _given(a, flag):
            continue
        var raw = a.get(flag)
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
    if a.flag_value.byte_length() > 0:
        return (
            "flag --"
            + a.flag_value
            + " needs a value, got the flag "
            + a.flag_value_text
            + see
        )
    if a.repeated.byte_length() > 0:
        return "flag --" + a.repeated + " is given twice" + see
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
    if _given(a, "json"):
        # Each of these prints to stdout before the JSON object, so a consumer
        # that parses stdout as JSON would fail on it.
        for flag in ["visual", "verbose", "dump-step-stats"]:
            if _given(a, flag):
                return "--json cannot be combined with --" + flag + see
    var number = _number_problems(a)
    if number.byte_length() > 0:
        return number + see
    if (a.command == "run" or a.command == "tokenize") and _given(a, "prompt"):
        var bad = utf8_first_invalid(a.get("prompt").as_bytes())
        if bad >= 0:
            return (
                "flag --prompt is not valid UTF-8: invalid byte at offset "
                + String(bad)
                + see
            )
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


def peak_rss_bytes(who: Int = 0) raises -> Int:
    """Peak resident set size in bytes (getrusage ru_maxrss, which Linux reports in KiB).

    `who` is the `getrusage` selector, 0 for this process. A failing call
    raises with the system's reason: reading the buffer it never filled would
    report a peak of 0, which looks like a measurement.
    """
    var buf = Bytes(256)
    var rc = external_call["getrusage", c_int](c_int(who), buf.addr)
    var failure = get_errno()
    if Int(rc) != 0:
        raise Error("getrusage failed: " + String(failure))
    # struct rusage: two struct timeval (16 bytes each), then long ru_maxrss.
    return Int(buf.u8().unsafe_bitcast[Int64]().unsafe_load(4)) * 1024


def peak_rss_json(who: Int = 0) -> String:
    """The `peak_rss_bytes` member's value: the number, or `null` when the
    measurement failed."""
    try:
        return String(peak_rss_bytes(who))
    except:
        return "null"


def peak_rss_text(who: Int = 0) -> String:
    """The `--verbose` line for the peak: gigabytes, or why it is missing."""
    try:
        return String(Float64(peak_rss_bytes(who)) / 1e9) + " GB"
    except e:
        return "unavailable (" + String(e) + ")"


def chat_prompt(arch: String, text: String) raises -> String:
    """One user turn in the model's chat template (its tokenizer.chat_template).

    LLaDA-8B-Instruct: Llama-3 style headers after <|startoftext|>.
    Dream-v0-Instruct: Qwen2 ChatML with the default system message.
    DiffusionGemma: `<bos>`, a user turn and an open model turn; the Jinja
    template in the file is reproduced by `gemma4_chat_prompt`.
    Any other architecture has no template here: a guessed one would give ids
    for text the model never saw.
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
    if arch == "llada":
        return (
            "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n"
            + String(text.strip())
            + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    raise Error(
        "no chat template for architecture '"
        + arch
        + "' (llada, dream and diffusion-gemma have one); pass --no-chat to"
        " tokenize the text as it is"
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


def llada_only_flags() -> List[String]:
    """The flags Dream accepts on the command line and never reads."""
    return ["full-logits", "threshold", "dual-cache", "block-length"]


def sampler_json(arch: String, cfg: GenConfig) -> String:
    """The JSON members that describe the sampler the run used.

    LLaDA reports its remasking strategy, block length and whether it asked for
    every row's logits. Dream has no block and no full-logits mode, so those
    two would describe settings it ignored; it reports the confidence
    algorithm that ran, `entropy` when the flag says `low_confidence`.
    """
    if arch == "dream":
        var algorithm = cfg.remasking
        if algorithm == "low_confidence":
            algorithm = "entropy"
        return ', "remasking": "' + algorithm + '"'
    return (
        ', "remasking": "'
        + cfg.remasking
        + '", "block_length": '
        + String(cfg.block_length)
        + ', "full_logits": '
        + ("true" if cfg.full_logits else "false")
    )


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
    var remasking = a.get_or("remasking", "low_confidence")
    if arch == "llada":
        if remasking != "low_confidence" and remasking != "random":
            raise Error(
                "--remasking "
                + remasking
                + " does not apply to llada (low_confidence or random)"
            )
        var schedule = schedule_problem(a)
        if schedule.byte_length() > 0:
            raise Error(schedule)
    elif arch == "dream":
        if remasking == "random":
            raise Error(
                "--remasking random does not apply to dream (entropy,"
                " maskgit_plus or topk_margin; low_confidence means entropy)"
            )
        for flag in llada_only_flags():
            if a.has(flag):
                raise Error("--" + flag + " applies to llada only, not " + arch)


def stop_json(res: GenResult) -> String:
    """The JSON members that say how many steps ran and why the run ended."""
    return (
        ', "steps_taken": '
        + String(res.steps_taken())
        + ', "stop_reason": "'
        + res.stop_reason
        + '"'
    )


def check_tokenizer_matches_model(g: GGUFFile, tok: Tokenizer) raises:
    """The tokenizer and the model agree on the vocabulary size. A model that
    can produce an id the tokenizer lacks fails only when `decode` raises after
    the whole generation."""
    var vocab = ModelConfig(g).vocab
    if tok.vocab_size() != vocab:
        raise Error(
            "the tokenizer has "
            + String(tok.vocab_size())
            + " tokens but the model's vocabulary is "
            + String(vocab)
        )


def check_context_length(
    prompt_tokens: Int, gen_length: Int, context: Int, arch: String
) raises:
    """Refuse a canvas longer than the context length the file declares.

    The canvas is the prompt plus `--max-tokens`. Every activation buffer and
    the rope table are sized from it, and the angles past the context length
    are ones the model never trained on. The test is written as a subtraction:
    `prompt_tokens + gen_length` wraps for a huge `--max-tokens`, and a wrapped
    sum is how a size of 2^62 once reached the allocator as a small one.
    """
    if gen_length > context - prompt_tokens:
        var total = "more than " + String(Int.MAX)
        if gen_length <= Int.MAX - prompt_tokens:
            total = String(prompt_tokens + gen_length)
        raise Error(
            "prompt "
            + String(prompt_tokens)
            + " + --max-tokens "
            + String(gen_length)
            + " = "
            + total
            + " exceeds the context length "
            + String(context)
            + " ("
            + arch
            + ".context_length)"
        )


def gen_config(a: Args) raises -> GenConfig:
    """The sampler settings the `run` flags ask for: each flag to its field,
    with the defaults that depend on another flag (`--steps` is `--max-tokens`,
    `--block-length` is `--max-tokens` capped at 32). The stop and canvas
    settings that need the tokenizer are added by `_run`."""
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
    cfg.threads = a.int_or("threads", num_logical_cores())
    cfg.eb_steps = a.int_or("eb-steps", cfg.eb_steps)
    cfg.eb_t_min = a.float_or("eb-t-min", cfg.eb_t_min)
    cfg.eb_t_max = a.float_or("eb-t-max", cfg.eb_t_max)
    cfg.eb_bound = a.float_or("eb-bound", cfg.eb_bound)
    cfg.eb_stability = a.int_or("eb-stability", cfg.eb_stability)
    cfg.eb_confidence = a.float_or("eb-confidence", cfg.eb_confidence)
    return cfg^


def check_device(device: String) raises:
    """Refuse a device `run` and `logits` do not have."""
    if device != "cpu" and device != "gpu":
        raise Error("unknown device: " + device + " (expected cpu or gpu)")


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
    var device = a.get_or("device", "cpu")
    check_device(device)
    var cfg = gen_config(a)
    var threads = cfg.threads
    var t0 = perf_counter_ns()
    var g = GGUFFile(a.get("model"))
    check_arch_flags(a, g.architecture())
    if g.architecture() == GEMMA_ARCH:
        # Before the tokenizer, the prompt and the model: DiffusionGemma has no
        # DenoisingModel yet, so no work done here could end in a run.
        raise Error(gemma_run_refusal())
    var tok = Tokenizer(g)
    check_tokenizer_matches_model(g, tok)
    var prompt = encode_prompt(tok, g.architecture(), a)
    check_context_length(
        len(prompt), cfg.gen_length, ModelConfig(g).ctx, g.architecture()
    )
    _ = g^
    var tok_s = Float64(perf_counter_ns() - t0) / 1e9
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
        )
    else:
        # `check_device` has refused every other value.
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
        )


def run_json(
    arch: String,
    device: String,
    cfg: GenConfig,
    prompt_tokens: Int,
    res: GenResult,
    tm: Timings,
    load_s: Float64,
    tok_s: Float64,
    text: String,
) -> String:
    """The one line `run --json` prints. Its members come in a fixed order that
    `bench/run_bench.py` and `bench/quality.py` read by name."""
    var steps_json = String("[")
    for i in range(len(res.step_ms)):
        steps_json += (", " if i else "") + String(res.step_ms[i])
    steps_json += "]"
    var fw = Float64(max(tm.forwards, 1))
    return (
        "{"
        + '"runtime": "mojo-dllm", "version": "'
        + VERSION
        + '", "arch": "'
        + arch
        + '", "device": "'
        + device
        + '"'
        + ', "threads": '
        + String(cfg.threads)
        + ', "prompt_tokens": '
        + String(prompt_tokens)
        + ', "gen_length": '
        + String(cfg.gen_length)
        + ', "steps": '
        + String(cfg.steps)
        + sampler_json(arch, cfg)
        + ', "stopped_early": '
        + ("true" if res.stopped_early else "false")
        + ', "forward_passes": '
        + String(res.forward_passes)
        + stop_json(res)
        + ', "load_s": '
        + String(load_s)
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
        + peak_rss_json()
        + ', "step_ms": '
        + steps_json
        + ', "tokens": ['
        + _csv(res.tokens)
        + "]"
        + ', "text": '
        + json_str(text)
        + "}"
    )


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
) raises:
    var verbose = a.has("verbose")
    if verbose:
        print(
            "model:",
            model.model_name(),
            "(" + model.config().arch + ")",
        )
        print("device:", device_line)
        print(_defaults_text(model.config(), ""), end="")
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
        print(
            run_json(
                model.config().arch,
                device,
                cfg,
                len(prompt),
                res,
                tm,
                model.load_time(),
                tok_s,
                text,
            )
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
        print("peak RSS:", peak_rss_text())


def _csv(xs: List[Int]) -> String:
    var s = String("")
    for i in range(len(xs)):
        s += ("," if i else "") + String(xs[i])
    return s


def _defaults_text(c: ModelConfig, indent: String) -> String:
    """The metadata keys the file lacks and the reader replaced by a default,
    one per line under a heading; empty when there are none."""
    if c.defaults.byte_length() == 0:
        return ""
    var text = indent + "defaults (key absent from the file):\n"
    for line in c.defaults.split("\n"):
        if line.byte_length() > 0:
            text += indent + "  " + String(line) + "\n"
    return text


def _dense_config_text(g: GGUFFile) -> String:
    """What `inspect` adds for a llada or dream file: the keys that took a
    default, or the reason the config is refused. Empty when the file sets every
    key."""
    try:
        return _defaults_text(ModelConfig(g), "")
    except e:
        return "config: refused: " + String(e) + "\n"


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
        return text + _defaults_text(c, "  ")
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
    var arch = g.get_str_or("general.architecture", "?")
    if arch == GEMMA_ARCH:
        text += "\n" + _gemma_config_text(g)
    elif arch == "llada" or arch == "dream":
        var block = _dense_config_text(g)
        if block.byte_length() > 0:
            text += "\n" + block
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


def gemma_only_flags(a: Args) raises -> String:
    """The `logits` flags that only a diffusion-gemma model takes, as given."""
    var given = String("")
    for name in ["path", "prev", "prev-temp-inv"]:
        if a.has(name):
            given += ("" if given.byte_length() == 0 else ", ") + "--" + name
    return given


def write_bytes_checked(path: String, data: U8Ptr, n_bytes: Int) raises:
    """Writes `n_bytes` to `path` and checks that the file holds them.

    `FileHandle.write_bytes` takes no error path: in Mojo 1.1.0 it calls
    `abort()` when `write()` fails (stdlib `std/io/file.mojo`, tag mojo/v1.1.0),
    so a full disk ends the process with SIGABRT and a partial file. `write_all`
    raises instead, `close` raises for a failure that only shows there (a quota
    or NFS), and the size check after the close catches a device that took the
    bytes and kept none of them.
    """
    var f = open(path, "w")
    f.write_all(Span(unsafe_ptr=data, length=n_bytes))
    f.close()
    var size = getsize(path)
    if size != n_bytes:
        raise Error(
            path
            + " holds "
            + String(size)
            + " bytes after the write, expected "
            + String(n_bytes)
        )


def write_f32_file(path: String, data: F32Ptr, n_floats: Int) raises:
    """Writes `n_floats` f32 values to `path`, or leaves `path` as it was.

    The bytes go to `<path>.tmp`, are checked there, and only then renamed to
    `path`; any error removes the temporary file first. A consumer such as
    bench/parity.py then never sees a short file under the final name. The
    rename replaces a symlink at `path` (the link goes, its target stays) and a
    read-only file; a path that exists and is not a regular file (a device, a
    pipe; not a directory) is written directly instead.
    """
    if exists(path) and not isfile(path) and not isdir(path):
        # A device, a pipe or a terminal (`/dev/null`, `/dev/stdout`): there is
        # no name to rename a temporary file to and no size to check, so the
        # bytes go straight to it.
        try:
            var f = open(path, "w")
            f.write_all(
                Span(
                    unsafe_ptr=data.unsafe_bitcast[UInt8](), length=n_floats * 4
                )
            )
            f.close()
        except e:
            raise Error("cannot write " + path + ": " + String(e))
        return
    var tmp = path + ".tmp"
    try:
        write_bytes_checked(tmp, data.unsafe_bitcast[UInt8](), n_floats * 4)
        var from_path = tmp
        var to_path = path
        var rc = external_call["rename", c_int](
            from_path.as_c_string_span(), to_path.as_c_string_span()
        )
        var failure = get_errno()
        _ = from_path^
        _ = to_path^
        if Int(rc) != 0:
            raise Error(
                "cannot rename " + tmp + " to " + path + ": " + String(failure)
            )
    except e:
        try:
            remove(tmp)
        except:
            pass
        raise Error("cannot write " + path + ": " + String(e))


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
    var header = GGUFFile(path)
    var c = ModelConfig(header)
    _ = header^
    var first = len(tokens) - c.canvas
    if first < 0:
        raise Error(
            "the token count "
            + String(len(tokens))
            + " is below the canvas length "
            + String(c.canvas)
        )
    var expected = String("")
    for i in range(c.canvas):
        expected += ("," if i > 0 else "") + String(first + i)
    var given = String("")
    for i in range(len(rows)):
        given += ("," if i > 0 else "") + String(rows[i])
    if given != expected:
        raise Error(
            GEMMA_ARCH
            + " scores the "
            + String(c.canvas)
            + " canvas positions, the last of the tokens: --rows must be "
            + expected
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
        _ = prev^
    var m = DiffusionGemma(path, threads=threads, max_tokens=len(tokens))
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
    write_f32_file(out_path, out.ptr(), c.canvas * c.vocab)
    print("wrote", c.canvas, "rows x", c.vocab, "logits")


def _logits[
    M: DenoisingModel
](mut m: M, tokens: List[Int], rows: List[Int], path: String) raises:
    var vocab = m.config().vocab
    var out = Floats(len(rows) * vocab)
    m.forward(tokens, rows, out.ptr())
    write_f32_file(path, out.ptr(), len(rows) * vocab)
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
        "dual-cache",
    ]


struct Decision(Movable):
    """What the arguments call for, before anything is loaded or printed.

    Three kinds, built by `help`, `error` and `run`, and nothing else: the exit
    status follows from the kind (0 for help and a run to start, 2 for a usage
    error), so no field combination can contradict it. `text` is the help text
    or the error message and is empty for a run.
    """

    comptime HELP = 0
    comptime ERROR = 1
    comptime RUN = 2

    var kind: Int
    var text: String

    def __init__(out self, kind: Int, text: String):
        self.kind = kind
        self.text = text

    @staticmethod
    def help(text: String) -> Self:
        return Self(Self.HELP, text)

    @staticmethod
    def error(text: String) -> Self:
        return Self(Self.ERROR, text)

    @staticmethod
    def run() -> Self:
        return Self(Self.RUN, String(""))

    def code(self) -> Int:
        """The exit status: 2 for a usage error, 0 otherwise."""
        return 2 if self.kind == Self.ERROR else 0

    def is_error(self) -> Bool:
        return self.kind == Self.ERROR

    def proceeds(self) -> Bool:
        """The arguments are valid and the command should run."""
        return self.kind == Self.RUN


def decide(raw: List[String]) raises -> Decision:
    """Help, usage errors and valid arguments, in that order of precedence."""
    var command = raw[1] if len(raw) > 1 else String("")
    if is_top_level_help(command):
        return Decision.help(usage())
    if command == "--version":
        return Decision.help(version_line())
    if not command_known(command):
        return Decision.error(
            "unknown command: " + command + "\nsee: mojo-dllm --help"
        )
    var spec = command_spec(command)
    var a = Args(raw, spec.values, spec.booleans)
    if a.has("help"):
        return Decision.help(usage_for(command))
    var problem = check_args(a, spec)
    if problem.byte_length() > 0:
        return Decision.error(problem)
    return Decision.run()


def execute(raw: List[String]) -> Int:
    """Runs the command line and returns the exit status: 0, 1 or 2."""
    try:
        var d = decide(raw)
        if d.is_error():
            print("ERROR:", d.text, file=stderr)
            return d.code()
        if not d.proceeds():
            print(d.text)
            return d.code()
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
        print("ERROR:", model_error(String(e)), file=stderr)
        return 1


def main():
    var raw = List[String]()
    for s in argv():
        raw.append(String(s))
    var code = execute(raw)
    if code != 0:
        exit(code)
