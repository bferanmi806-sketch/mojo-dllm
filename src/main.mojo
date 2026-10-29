"""Command line for mojo-dllm.

    mojo-dllm run      --model M --prompt TEXT [--max-tokens N] [--steps N] [--block-length N]
                       [--seed N] [--temperature T] [--threads N] [--device cpu|gpu]
                       [--remasking low_confidence|random] [--threshold P] [--dual-cache]
                       [--no-chat] [--full-logits] [--no-early-stop] [--visual] [--visual-interval N] [--verbose]
                       [--dump-step-stats] [--json]
    mojo-dllm tokenize --model M --prompt TEXT [--no-chat]
    mojo-dllm inspect  MODEL.gguf
    mojo-dllm logits   --model M --tokens 1,2,3 --rows 0,2 --out logits.f32 [--threads N] [--device cpu|gpu]
"""

from std.ffi import ErrNo, external_call, c_int
from std.math import isfinite
from std.os import getenv
from std.sys import argv, exit, num_logical_cores
from std.time import perf_counter_ns

from cli.args import Args, parse_csv_ints, unknown_flag_error
from cli.visual import CanvasPrinter
from mojo_dllm.formats.gguf import GGUFFile, GGUF_TYPE_ARRAY
from mojo_dllm.diffusion.sampler import (
    GenConfig,
    generate_observed,
)
from mojo_dllm.gpu.model import GpuDiffusionLM
from mojo_dllm.models.transformer import DenoisingModel, DiffusionLM
from mojo_dllm.sys.mem import Bytes, Floats
from mojo_dllm.tokenizer.bpe import Tokenizer

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
            ["model", "tokens", "rows", "out", "threads", "device"],
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
            " [--dump-step-stats] [--json]"
        )
    if command == "tokenize":
        return "  mojo-dllm tokenize --model M --prompt TEXT [--no-chat]"
    if command == "inspect":
        return "  mojo-dllm inspect MODEL.gguf"
    if command == "logits":
        return (
            "  mojo-dllm logits --model M --tokens IDS --rows POS --out FILE"
            " [--threads N] [--device cpu|gpu]"
        )
    raise Error("unknown command: " + command)


def usage_for(command: String) raises -> String:
    return "usage:\n" + command_usage(command)


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
    for flag in ["max-tokens", "steps", "block-length", "threads", "seed"]:
        if not a.has(flag):
            continue
        var raw = a.values[flag]
        var n: Int
        try:
            n = Int(raw)
        except:
            return "flag --" + flag + " needs an integer, got '" + raw + "'"
        if flag != "seed" and n < 1:
            return "flag --" + flag + " must be at least 1, got " + raw
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


def chat_prompt(arch: String, text: String) -> String:
    """One user turn in the model's chat template (its tokenizer.chat_template).

    LLaDA-8B-Instruct: Llama-3 style headers after <|startoftext|>.
    Dream-v0-Instruct: Qwen2 ChatML with the default system message.
    """
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
    cfg.threads = threads
    var t0 = perf_counter_ns()
    var g = GGUFFile(a.get("model"))
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
    cfg.stop_ids = List[Int]()
    for s in [tok.eos_id, tok.eot_id]:
        if s >= 0:
            cfg.stop_ids.append(s)
    var printer = CanvasPrinter(
        tok^,
        len(prompt),
        model.config().mask_id,
        a.has("visual"),
        a.int_or("visual-interval", 1),
        external_call["isatty", c_int](c_int(1)) != 0,
        getenv("NO_COLOR"),
    )
    var res = generate_observed(model, prompt, cfg, printer)
    printer.finish(res.tokens)
    var gen = List[Int]()
    for i in range(len(prompt), len(res.tokens)):
        var t = res.tokens[i]
        if t == printer.tok.eos_id or t == printer.tok.eot_id:
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


def cmd_inspect(a: Args) raises:
    var g = GGUFFile(a.positional[0])
    print("file:", a.positional[0])
    print("gguf version:", g.version)
    print("architecture:", g.get_str_or("general.architecture", "?"))
    print("name:", g.get_str_or("general.name", "?"))
    print("tensors:", g.n_tensors())
    print("metadata keys:", len(g.kvs))
    print()
    print("metadata:")
    for i in range(len(g.kvs)):
        print("  ", g.kvs[i].key, " = ", g.value_str(i), sep="")
    var counts = Dict[String, Int]()
    for i in range(g.n_tensors()):
        var tn = g.tensors[i].type_name()
        counts[tn] = counts.get(tn, 0) + 1
    print()
    print("tensor types:")
    for e in counts.items():
        print("  ", e.key, ": ", e.value, sep="")
    print()
    print("tensors:")
    for i in range(g.n_tensors()):
        var t = g.tensors[i]
        print("  ", t.name, "  ", t.type_name(), "  ", t.shape_str(), sep="")


def cmd_logits(a: Args) raises:
    var threads = a.int_or("threads", num_logical_cores())
    var tokens = parse_csv_ints(a.get("tokens"))
    var rows = parse_csv_ints(a.get("rows"))
    var device = a.get_or("device", "cpu")
    if device == "gpu":
        var gm = GpuDiffusionLM(a.get("model"), max_tokens=len(tokens))
        _logits(gm, tokens, rows, a.get("out"))
    elif device == "cpu":
        var cm = DiffusionLM(
            a.get("model"), threads=threads, max_tokens=len(tokens)
        )
        _logits(cm, tokens, rows, a.get("out"))
    else:
        raise Error("unknown device: " + device + " (expected cpu or gpu)")


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
