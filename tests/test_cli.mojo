from std.ffi import c_int, external_call
from std.os import mkdir, remove, rmdir
from std.os.path import exists
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from cli.args import Args, env_flags
from mojo_dllm.diffusion.sampler import GenConfig, GenResult
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.models.diffusion_gemma import DiffusionGemma
from mojo_dllm.models.transformer import DiffusionLM, ModelConfig, Timings
from mojo_dllm.sys.mem import Bytes, F32Ptr, Floats
from mojo_dllm.sys.mmap import Mapping
from mojo_dllm.tokenizer.bpe import Tokenizer
from main import (
    Decision,
    boolean_flags,
    chat_prompt,
    check_args,
    check_arch_flags,
    command_spec,
    decide,
    encode_prompt,
    execute,
    inspect_text,
    check_tokenizer_matches_model,
    check_context_length,
    check_device,
    cmd_logits,
    cmd_run,
    cmd_tokenize,
    gen_config,
    run_json,
    json_str,
    _csv,
    _logits_gemma,
    gemma_only_flags,
    is_top_level_help,
    model_error,
    version_line,
    sampler_json,
    peak_rss_bytes,
    peak_rss_json,
    peak_rss_text,
    stop_json,
    write_bytes_checked,
    write_f32_file,
    usage,
    usage_for,
)


def test_documented_booleans_are_parsed_without_values() raises:
    var flags = boolean_flags()
    var count = 0
    var doc: String
    with open("docs/cli.md", "r") as f:
        doc = f.read()
    for line in doc.split("\n"):
        if not line.startswith("| `--"):
            continue
        var cells = line.split("|")
        if String(cells[2].strip()) != "off":
            continue
        var spelling = String(cells[1].strip())
        var flag = String(spelling[byte = 3 : spelling.byte_length() - 1])
        var found = False
        for candidate in flags:
            if candidate == flag:
                found = True
        assert_true(found, "documented boolean missing: " + flag)
        var spec = command_spec("run")
        var a = Args(["x", "run", "--" + flag], spec.values, spec.booleans)
        assert_true(a.has(flag))
        assert_equal(len(a.unknown), 0)
        count += 1
    assert_equal(count, 8)
    assert_equal(len(flags), count)


def test_chat_templates_strip_user_text() raises:
    var llada = "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    var dream = (
        "<|im_start|>system\nYou are a helpful"
        " assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )
    assert_equal(chat_prompt("llada", " \thi\n"), llada)
    assert_equal(chat_prompt("dream", " \thi\n"), dream)


def test_chat_prompt_refuses_an_architecture_without_a_template() raises:
    # `chat_prompt` fell through to the LLaDA header template for every
    # architecture except Dream, so the ids belonged to text the model never saw.
    with assert_raises(
        contains=(
            "no chat template for architecture 'qwen2' (llada, dream and"
            " diffusion-gemma have one); pass --no-chat"
        )
    ):
        _ = chat_prompt("qwen2", "hi")
    with assert_raises(contains="architecture '?'"):
        _ = chat_prompt("?", "hi")


def test_tokenize_refuses_a_chat_prompt_for_another_architecture() raises:
    var spec = command_spec("tokenize")
    var path = "tests/fixtures/gpt2-tiny/other-arch.gguf"
    var chat = Args(
        ["x", "tokenize", "--model", path, "--prompt", "ab"],
        spec.values,
        spec.booleans,
    )
    with assert_raises(contains="no chat template for architecture 'qwen2'"):
        cmd_tokenize(chat)
    # The text as it is still tokenizes: --no-chat asks for no template.
    var plain = Args(
        ["x", "tokenize", "--model", path, "--prompt", "ab", "--no-chat"],
        spec.values,
        spec.booleans,
    )
    cmd_tokenize(plain)
    assert_equal(
        execute(["mojo-dllm", "tokenize", "--model", path, "--prompt", "ab"]), 1
    )


def test_chat_template_for_diffusion_gemma() raises:
    # tokenizer.chat_template: <bos>, a user turn, then an open model turn.
    assert_equal(
        chat_prompt("diffusion-gemma", " \thi\n"),
        "<bos><|turn>user\nhi<turn|>\n<|turn>model\n",
    )


def test_run_prompt_for_diffusion_gemma_has_one_bos() raises:
    var tok = Tokenizer(GGUFFile("tests/fixtures/gemma4-tokenizer.gguf"))
    var spec = command_spec("run")
    var chat = Args(["x", "run", "--prompt", "Hi"], spec.values, spec.booleans)
    var ids = encode_prompt(tok, "diffusion-gemma", chat)
    # <bos> from the template is the only BOS; <|turn> opens the user turn
    # and the model turn, and <turn|> closes the user turn.
    assert_equal(ids[0], 2)
    assert_equal(ids[1], 105)
    var bos = 0
    var opens = 0
    var closes = 0
    for t in ids:
        bos += 1 if t == 2 else 0
        opens += 1 if t == 105 else 0
        closes += 1 if t == 106 else 0
    assert_equal(bos, 1)
    assert_equal(opens, 2)
    assert_equal(closes, 1)
    var raw = Args(
        ["x", "run", "--prompt", "Hi", "--no-chat"], spec.values, spec.booleans
    )
    var plain = encode_prompt(tok, "diffusion-gemma", raw)
    assert_equal(len(plain), 2)
    assert_equal(plain[0], 2)
    assert_equal(tok.decode([plain[1]], skip_special=False), "Hi")


def test_json_escapes_and_preserves_unicode() raises:
    assert_equal(json_str('"\\\n\r\t'), '"\\"\\\\\\n\\r\\t"')
    assert_equal(json_str("\x00\x08\x0b\x1f"), '"\\u0000\\u0008\\u000b\\u001f"')
    assert_equal(json_str("héllo 世界"), '"héllo 世界"')
    assert_equal(json_str(""), '""')


def test_csv_format() raises:
    assert_equal(_csv([]), "")
    assert_equal(_csv([7]), "7")
    assert_equal(_csv([1, -2, 3]), "1,-2,3")


def test_usage_lists_all_commands() raises:
    var text = usage()
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        assert_true(text.__contains__("mojo-dllm " + command + " "))


def _parse(command: String, rest: List[String]) raises -> Args:
    var spec = command_spec(command)
    var argv = List[String](["mojo-dllm", command])
    for r in rest:
        argv.append(r)
    return Args(argv, spec.values, spec.booleans)


def _check(command: String, rest: List[String]) raises -> String:
    return check_args(_parse(command, rest), command_spec(command))


def test_every_command_declares_its_flags() raises:
    var run = command_spec("run")
    assert_equal(len(run.values), 18)
    assert_equal(len(run.booleans), 8)
    var tokenize = command_spec("tokenize")
    assert_equal(len(tokenize.values), 2)
    assert_equal(len(tokenize.booleans), 1)
    var inspect = command_spec("inspect")
    assert_equal(len(inspect.values), 0)
    assert_equal(len(inspect.booleans), 0)
    assert_true(inspect.takes_path)
    var logits = command_spec("logits")
    assert_equal(len(logits.values), 9)
    assert_equal(len(logits.booleans), 0)
    with assert_raises(contains="unknown command: frobnicate"):
        _ = command_spec("frobnicate")


def test_declared_booleans_come_from_the_single_list() raises:
    var all_booleans = boolean_flags()
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        for flag in command_spec(command).booleans:
            var found = False
            for candidate in all_booleans:
                if candidate == flag:
                    found = True
            assert_true(found, command + " declares a boolean not in the list")


def test_env_flags_are_value_flags_of_run() raises:
    # Every name the environment can fill must be a declared value flag of
    # `run` and must not be a boolean: a boolean takes no value, so an
    # inserted `--flag value` pair would leave the value as a stray
    # positional, and `run` refuses stray positionals.
    var run = command_spec("run")
    for name in env_flags():
        var in_values = False
        for flag in run.values:
            if flag == name:
                in_values = True
        assert_true(in_values, name + " is not a value flag of run")
        for flag in run.booleans:
            assert_true(flag != name, name + " is a boolean flag of run")


def test_a_flag_of_another_command_is_unknown() raises:
    var message = _check(
        "tokenize", ["--model", "m", "--prompt", "p", "--json"]
    )
    assert_equal(
        message,
        "unknown flag --json for 'tokenize'\nsee: mojo-dllm tokenize --help",
    )


def test_misspelled_flag_suggests_the_declared_one() raises:
    var message = _check("run", ["--stpes", "3"])
    assert_equal(
        message,
        (
            "unknown flag --stpes for 'run' (did you mean --steps?)\nsee:"
            " mojo-dllm run --help"
        ),
    )


def test_unknown_flag_is_reported_before_a_missing_one() raises:
    var message = _check("run", ["--stpes", "3"])
    assert_true(not message.__contains__("missing required"))


def test_missing_required_flag_prints_usage() raises:
    var message = _check("run", ["--prompt", "x"])
    assert_equal(
        message,
        "missing required flag --model for 'run'\n" + usage_for("run"),
    )
    assert_true(message.__contains__("usage:\n  mojo-dllm run --model M"))
    var second = _check("run", ["--model", "m"])
    assert_true(second.startswith("missing required flag --prompt for 'run'\n"))
    var logits = _check(
        "logits", ["--model", "m", "--tokens", "1", "--rows", "0"]
    )
    assert_true(logits.startswith("missing required flag --out for 'logits'\n"))


def test_inspect_needs_a_path_and_nothing_else() raises:
    var missing = _check("inspect", [])
    assert_equal(
        missing, "missing model path for 'inspect'\n" + usage_for("inspect")
    )
    assert_equal(_check("inspect", ["m.gguf"]), "")
    var extra = _check("run", ["--model", "m", "--prompt", "p", "stray"])
    assert_equal(
        extra,
        "unexpected argument 'stray' for 'run'\nsee: mojo-dllm run --help",
    )


def test_complete_commands_pass_the_check() raises:
    assert_equal(
        _check(
            "run", ["--model", "m", "--prompt", "p", "--json", "--steps", "4"]
        ),
        "",
    )
    assert_equal(_check("tokenize", ["--model", "m", "--prompt", "p"]), "")
    assert_equal(
        _check(
            "logits",
            ["--model", "m", "--tokens", "1", "--rows", "0", "--out", "o"],
        ),
        "",
    )


def test_usage_for_one_command() raises:
    var run = usage_for("run")
    assert_true(
        run.startswith("usage:\n  mojo-dllm run --model M --prompt TEXT")
    )
    assert_true(not run.__contains__("mojo-dllm tokenize"))
    assert_true(not run.endswith("\n"))
    var tokenize = usage_for("tokenize")
    assert_equal(
        tokenize,
        "usage:\n  mojo-dllm tokenize --model M --prompt TEXT [--no-chat]",
    )
    with assert_raises(contains="unknown command: nope"):
        _ = usage_for("nope")


def test_usage_for_names_exactly_the_declared_flags() raises:
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        var spec = command_spec(command)
        var text = usage_for(command)
        var declared = spec.declared()
        for flag in declared:
            assert_true(
                text.__contains__("--" + flag),
                command + " usage omits --" + flag,
            )
        var cleaned = (
            text.replace("\n", " ").replace("]", " ").replace("|", " ")
        )
        for token in cleaned.split(" "):
            var word = String(token)
            if not word.startswith("--"):
                continue
            var name = String(word[byte=2:])
            var found = False
            for flag in declared:
                if flag == name:
                    found = True
            assert_true(found, command + " usage names undeclared --" + name)


def test_docs_table_lists_only_declared_run_flags() raises:
    var doc: String
    with open("docs/cli.md", "r") as f:
        doc = f.read()
    var declared = command_spec("run").declared()
    var seen = 0
    for line in doc.split("\n"):
        if not line.startswith("| `--"):
            continue
        var cell = String(line.split("|")[1].strip()).replace("`", "")
        var name = String(String(cell.split(" ")[0])[byte=2:])
        var found = False
        for flag in declared:
            if flag == name:
                found = True
        assert_true(found, "docs/cli.md lists undeclared flag: " + name)
        seen += 1
    assert_equal(seen, len(declared))


def test_help_routing() raises:
    for command in List[String](["", "help", "--help", "-h"]):
        assert_true(is_top_level_help(command))
    for command in List[String](["run", "tokenize", "frobnicate"]):
        assert_true(not is_top_level_help(command))
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        assert_true(_parse(command, ["--help"]).has("help"))
        assert_true(_parse(command, ["-h"]).has("help"))
    # help wins over every other problem, because check_args runs after it.
    var a = _parse("run", ["--stpes", "--help"])
    assert_true(a.has("help"))


def _decide(command: String, rest: List[String]) raises -> Decision:
    var argv = List[String](["mojo-dllm", command])
    for r in rest:
        argv.append(r)
    return decide(argv)


def test_unknown_command_is_a_usage_error() raises:
    var d = _decide("bogus", [])
    assert_equal(d.code(), 2)
    assert_true(d.is_error())
    assert_equal(d.text, "unknown command: bogus\nsee: mojo-dllm --help")
    assert_equal(execute(["mojo-dllm", "bogus"]), 2)


def test_help_exits_zero_and_wins_over_other_problems() raises:
    for command in List[String](["", "help", "--help", "-h"]):
        var d = _decide(command, [])
        assert_equal(d.code(), 0)
        assert_true(not d.is_error())
        assert_equal(d.text, usage())
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        var d = _decide(command, ["--stpes", "--steps", "abc", "--help"])
        assert_equal(d.code(), 0)
        assert_equal(d.text, usage_for(command))
        assert_true(not d.proceeds())
    # --help before a value flag that has no value: help, not "needs a value".
    var late = _decide("run", ["--help", "--prompt"])
    assert_equal(late.code(), 0)
    assert_equal(late.text, usage_for("run"))
    assert_equal(execute(["mojo-dllm", "run", "--help", "--prompt"]), 0)
    assert_equal(execute(["mojo-dllm", "--help"]), 0)


def test_version_flag_prints_one_line_and_exits_zero() raises:
    var d = _decide("--version", [])
    assert_equal(d.code(), 0)
    assert_true(not d.is_error())
    assert_true(not d.proceeds())
    assert_equal(d.text, version_line())
    assert_true(not d.text.__contains__("\n"))
    assert_true(usage().startswith(version_line() + ": "))
    assert_equal(execute(["mojo-dllm", "--version"]), 0)


def test_usage_mentions_the_version_flag() raises:
    assert_true(usage().__contains__("mojo-dllm --version"))


def test_usage_errors_exit_two() raises:
    var unknown = _decide("run", ["--stpes", "3"])
    assert_equal(unknown.code(), 2)
    assert_true(unknown.is_error())
    var missing = _decide("run", ["--model", "m", "--prompt"])
    assert_equal(missing.code(), 2)
    assert_equal(
        missing.text,
        "flag --prompt needs a value\nsee: mojo-dllm run --help",
    )
    var number = _decide(
        "run", ["--model", "m", "--prompt", "p", "--steps", "abc"]
    )
    assert_equal(number.code(), 2)
    assert_equal(
        number.text,
        "flag --steps needs an integer, got 'abc'\nsee: mojo-dllm run --help",
    )
    assert_equal(execute(["mojo-dllm", "run", "--stpes", "3"]), 2)
    assert_equal(execute(["mojo-dllm", "run", "--prompt"]), 2)
    assert_equal(
        execute(
            [
                "mojo-dllm",
                "run",
                "--model",
                "m",
                "--prompt",
                "p",
                "--steps",
                "x",
            ]
        ),
        2,
    )


def test_decisions_derive_the_exit_code_from_their_kind() raises:
    var help = Decision.help("usage text")
    assert_equal(help.code(), 0)
    assert_equal(help.text, "usage text")
    assert_true(not help.is_error())
    assert_true(not help.proceeds())
    var error = Decision.error("bad flag")
    assert_equal(error.code(), 2)
    assert_equal(error.text, "bad flag")
    assert_true(error.is_error())
    assert_true(not error.proceeds())
    var run = Decision.run()
    assert_equal(run.code(), 0)
    assert_equal(run.text, "")
    assert_true(not run.is_error())
    assert_true(run.proceeds())


struct _Capture(Movable):
    """Redirects one file descriptor (1 for stdout, 2 for stderr) into a pipe
    until `finish` hands back what was written to it."""

    var read_fd: Int
    var write_fd: Int
    var saved: Int
    var target: Int

    def __init__(out self, target: Int) raises:
        var fds = Bytes(8)
        if external_call["pipe", c_int](fds.addr) != 0:
            raise Error("pipe failed")
        self.target = target
        self.read_fd = Int(fds.i32().unsafe_load(0))
        self.write_fd = Int(fds.i32().unsafe_load(1))
        self.saved = Int(external_call["dup", c_int](c_int(target)))
        if self.saved < 0:
            raise Error("dup failed")
        if (
            external_call["dup2", c_int](c_int(self.write_fd), c_int(target))
            < 0
        ):
            raise Error("redirect failed")

    def finish(mut self) raises -> String:
        _ = external_call["fflush", c_int](Int(0))
        _ = external_call["dup2", c_int](c_int(self.saved), c_int(self.target))
        _ = external_call["close", c_int](c_int(self.saved))
        _ = external_call["close", c_int](c_int(self.write_fd))
        var data = Bytes(16384)
        var n = Int(
            external_call["read", Int](self.read_fd, data.u8(), data.size)
        )
        _ = external_call["close", c_int](c_int(self.read_fd))
        if n < 0:
            raise Error("read of the captured stream failed")
        return String(
            StringSpan(unsafe_from_utf8=Span(unsafe_ptr=data.u8(), length=n))
        )


def _streams(argv: List[String]) raises -> Tuple[Int, String, String]:
    """Runs `execute` and returns its status, its stdout and its stderr."""
    var out = _Capture(1)
    var err = _Capture(2)
    var code = execute(argv)
    var err_text = err.finish()
    var out_text = out.finish()
    return (code, out_text^, err_text^)


def test_json_cannot_be_combined_with_the_flags_that_print_before_it() raises:
    # Each of these writes to stdout before the JSON line, so a consumer that
    # calls json.loads on stdout would fail. Maintainer decision on #110: refuse.
    for flag in List[String](["--visual", "--verbose", "--dump-step-stats"]):
        var rest = List[String](
            ["--model", "m", "--prompt", "p", "--json", flag]
        )
        assert_equal(
            _check("run", rest),
            "--json cannot be combined with "
            + flag
            + "\nsee: mojo-dllm run --help",
        )
        var d = _decide("run", rest)
        assert_equal(d.code(), 2)
        assert_true(d.is_error())
    for alone in List[String](
        ["--json", "--visual", "--verbose", "--dump-step-stats"]
    ):
        assert_equal(
            _check("run", ["--model", "m", "--prompt", "p", alone]), ""
        )


def test_usage_errors_and_runtime_errors_go_to_stderr() raises:
    var usage = _streams(
        [
            "mojo-dllm",
            "run",
            "--model",
            "m",
            "--prompt",
            "p",
            "--json",
            "--visual",
        ]
    )
    assert_equal(usage[0], 2)
    assert_equal(usage[1], "")
    assert_equal(
        usage[2],
        (
            "ERROR: --json cannot be combined with --visual\nsee: mojo-dllm run"
            " --help\n"
        ),
    )
    var runtime = _streams(
        [
            "mojo-dllm",
            "run",
            "--model",
            "/nonexistent/model.gguf",
            "--prompt",
            "hi",
        ]
    )
    assert_equal(runtime[0], 1)
    assert_equal(runtime[1], "")
    assert_true(runtime[2].startswith("ERROR: cannot open model file"))
    var unknown = _streams(["mojo-dllm", "bogus"])
    assert_equal(unknown[0], 2)
    assert_equal(unknown[1], "")
    assert_true(unknown[2].startswith("ERROR: unknown command: bogus"))


def test_help_and_version_stay_on_stdout() raises:
    var help = _streams(["mojo-dllm", "run", "--help"])
    assert_equal(help[0], 0)
    assert_true(help[1].startswith("usage:\n  mojo-dllm run"))
    assert_equal(help[2], "")
    var version = _streams(["mojo-dllm", "--version"])
    assert_equal(version[0], 0)
    assert_equal(version[1], version_line() + "\n")
    assert_equal(version[2], "")


def test_valid_arguments_proceed() raises:
    var d = _decide("run", ["--model", "m", "--prompt", "p"])
    assert_equal(d.code(), 0)
    assert_true(d.proceeds())
    assert_true(not d.is_error())


def test_runtime_failure_exits_one() raises:
    assert_equal(
        execute(
            [
                "mojo-dllm",
                "run",
                "--model",
                "/nonexistent/model.gguf",
                "--prompt",
                "hi",
            ]
        ),
        1,
    )
    assert_equal(execute(["mojo-dllm", "inspect", "tests/fixtures"]), 1)


def test_integer_flags_name_the_flag_and_the_value() raises:
    var base = List[String](["--model", "m", "--prompt", "p"])
    for flag in List[String](
        ["max-tokens", "steps", "block-length", "seed", "threads"]
    ):
        for bad in List[String](["abc", "", "12.5", "99999999999999999999"]):
            var rest = base.copy()
            rest.append("--" + flag)
            rest.append(bad)
            assert_equal(
                _check("run", rest),
                "flag --"
                + flag
                + " needs an integer, got '"
                + bad
                + "'\nsee: mojo-dllm run --help",
            )
    var logits = _check(
        "logits",
        [
            "--model",
            "m",
            "--tokens",
            "1",
            "--rows",
            "0",
            "--out",
            "o",
            "--threads",
            "many",
        ],
    )
    assert_equal(
        logits,
        (
            "flag --threads needs an integer, got 'many'\nsee: mojo-dllm logits"
            " --help"
        ),
    )


def test_values_that_can_never_be_valid_are_refused() raises:
    var base = List[String](["--model", "m", "--prompt", "p"])
    for flag in List[String](
        ["max-tokens", "steps", "block-length", "threads"]
    ):
        for bad in List[String](["0", "-3"]):
            var rest = base.copy()
            rest.append("--" + flag)
            rest.append(bad)
            assert_equal(
                _check("run", rest),
                "flag --"
                + flag
                + " must be at least 1, got "
                + bad
                + "\nsee: mojo-dllm run --help",
            )
    var rest = base.copy()
    rest.append("--temperature")
    rest.append("-0.5")
    assert_equal(
        _check("run", rest),
        (
            "flag --temperature must be at least 0, got -0.5\nsee: mojo-dllm"
            " run --help"
        ),
    )
    for bad in List[String](["warm", "nan", "inf", ""]):
        var r = base.copy()
        r.append("--temperature")
        r.append(bad)
        assert_equal(
            _check("run", r),
            "flag --temperature needs a finite number, got '"
            + bad
            + "'\nsee: mojo-dllm run --help",
        )
    # Negative seeds and zero temperature stay valid.
    var ok = base.copy()
    ok.append("--seed")
    ok.append("-5")
    ok.append("--temperature")
    ok.append("0")
    assert_equal(_check("run", ok), "")


def test_entropy_bound_flags_are_checked_before_the_model_loads() raises:
    var base = List[String](["--model", "m", "--prompt", "p"])
    for flag in List[String](["eb-steps", "eb-stability"]):
        var r = base.copy()
        r.append("--" + flag)
        r.append("x")
        assert_equal(
            _check("run", r),
            "flag --"
            + flag
            + " needs an integer, got 'x'\nsee: mojo-dllm run --help",
        )
    var low = base.copy()
    low.append("--eb-steps")
    low.append("0")
    assert_equal(
        _check("run", low),
        "flag --eb-steps must be at least 1, got 0\nsee: mojo-dllm run --help",
    )
    var stab = base.copy()
    stab.append("--eb-stability")
    stab.append("-1")
    assert_equal(
        _check("run", stab),
        (
            "flag --eb-stability must be at least 0, got -1\nsee: mojo-dllm"
            " run --help"
        ),
    )
    for flag in List[String](
        ["eb-t-min", "eb-t-max", "eb-bound", "eb-confidence"]
    ):
        for bad in List[String](["warm", "nan", "inf", ""]):
            var r = base.copy()
            r.append("--" + flag)
            r.append(bad)
            assert_equal(
                _check("run", r),
                "flag --"
                + flag
                + " needs a finite number, got '"
                + bad
                + "'\nsee: mojo-dllm run --help",
            )
    var zero_max = base.copy()
    zero_max.append("--eb-t-max")
    zero_max.append("0")
    assert_equal(
        _check("run", zero_max),
        "flag --eb-t-max must be above 0, got 0\nsee: mojo-dllm run --help",
    )
    for flag in List[String](["eb-t-min", "eb-bound", "eb-confidence"]):
        var r = base.copy()
        r.append("--" + flag)
        r.append("-0.5")
        assert_equal(
            _check("run", r),
            "flag --"
            + flag
            + " must be at least 0, got -0.5\nsee: mojo-dllm run --help",
        )
    # The branch's own defaults, and the boundary values, are valid.
    var ok = base.copy()
    for pair in List[String](
        [
            "--eb-steps",
            "48",
            "--eb-t-min",
            "0",
            "--eb-t-max",
            "0.8",
            "--eb-bound",
            "0",
            "--eb-stability",
            "0",
            "--eb-confidence",
            "0",
        ]
    ):
        ok.append(pair)
    assert_equal(_check("run", ok), "")


def test_entropy_bound_flags_belong_to_diffusion_gemma() raises:
    var spec = command_spec("run")
    var with_eb = Args(
        ["x", "run", "--eb-bound", "0.2"], spec.values, spec.booleans
    )
    with assert_raises(contains="--eb-bound applies to diffusion-gemma only"):
        check_arch_flags(with_eb, "llada")
    with assert_raises(contains="not dream"):
        check_arch_flags(with_eb, "dream")
    check_arch_flags(with_eb, "diffusion-gemma")
    var plain = Args(["x", "run"], spec.values, spec.booleans)
    check_arch_flags(plain, "llada")
    var steps = Args(["x", "run", "--steps", "8"], spec.values, spec.booleans)
    with assert_raises(contains="use --eb-steps"):
        check_arch_flags(steps, "diffusion-gemma")
    check_arch_flags(steps, "llada")


def test_json_reports_steps_taken_and_the_stop_reason() raises:
    var res = GenResult()
    assert_equal(
        stop_json(res), ', "steps_taken": 0, "stop_reason": "complete"'
    )
    res.forward_passes = 17
    res.stop_reason = "converged"
    assert_equal(
        stop_json(res), ', "steps_taken": 17, "stop_reason": "converged"'
    )


def test_csv_flags_of_logits_are_checked() raises:
    var message = _check(
        "logits",
        ["--model", "m", "--tokens", "1,x", "--rows", "0", "--out", "o"],
    )
    assert_equal(
        message,
        (
            "flag --tokens needs comma-separated integers, got '1,x'\nsee:"
            " mojo-dllm logits --help"
        ),
    )


def test_an_empty_csv_part_is_a_usage_error_naming_the_flag() raises:
    var message = _check(
        "logits",
        ["--model", "m", "--tokens", "1,2", "--rows", "0,,2", "--out", "o"],
    )
    assert_equal(
        message,
        (
            "flag --rows needs comma-separated integers, got '0,,2'\nsee:"
            " mojo-dllm logits --help"
        ),
    )
    assert_equal(
        _check(
            "logits",
            [
                "--model",
                "m",
                "--tokens",
                "1,2,",
                "--rows",
                "0,1,",
                "--out",
                "o",
            ],
        ),
        "",
    )


def test_a_flag_name_as_a_value_is_a_usage_error() raises:
    assert_equal(
        _check("run", ["--model", "m", "--prompt", "--verbose"]),
        (
            "flag --prompt needs a value, got the flag --verbose\nsee:"
            " mojo-dllm run --help"
        ),
    )
    assert_equal(
        _check("run", ["--model", "--prompt", "p"]),
        (
            "flag --model needs a value, got the flag --prompt\nsee:"
            " mojo-dllm run --help"
        ),
    )
    # --help is not a declared flag name: it stays a legal prompt, as docs say.
    assert_equal(_check("run", ["--model", "m", "--prompt", "--help"]), "")


def test_a_repeated_value_flag_is_a_usage_error() raises:
    assert_equal(
        _check(
            "run",
            ["--model", "m", "--prompt", "p", "--steps", "4", "--steps", "5"],
        ),
        "flag --steps is given twice\nsee: mojo-dllm run --help",
    )
    # A boolean flag given twice says the same thing twice and stays legal.
    assert_equal(
        _check("run", ["--model", "m", "--prompt", "p", "--json", "--json"]), ""
    )


def test_equals_form_says_how_values_are_given() raises:
    assert_equal(
        _check("run", ["--model=x"]),
        (
            "unknown flag --model=x for 'run'\nflags take their value as the"
            " next argument: --model x\nsee: mojo-dllm run --help"
        ),
    )
    # A boolean has no value to give, and an undeclared name gets no advice.
    assert_equal(
        _check("run", ["--json=1"]),
        (
            "unknown flag --json=1 for 'run' (did you mean --json?)\nsee:"
            " mojo-dllm run --help"
        ),
    )


def _joined(lines: List[String]) -> String:
    var text = String("")
    for line in lines:
        text += line + "\n"
    return text


def test_inspect_pins_the_diffusion_gemma_fixture_output() raises:
    var path = "tests/fixtures/tiny-diffusion-gemma.gguf"
    var text = inspect_text(path)
    var head = _joined(
        [
            "file: " + path,
            "gguf version: 3",
            "architecture: diffusion-gemma",
            "name: tiny-diffusion-gemma",
            "tensors: 52",
            "metadata keys: 28",
            "",
            "config:",
            "  blocks: 2 (sliding 1, global 1)",
            "  global layers: 1",
            "  hidden size: 256",
            "  heads: 8",
            "  kv heads: 4 sliding, 1 global",
            "  head size: 64 sliding, 128 global",
            "  rope base: 10000.0 sliding, 1000000.0 global",
            "  sliding window: 128",
            "  dense ffn: 512",
            "  experts: 4 (2 used), ffn 64",
            "  vocabulary: 512, mask token 4",
            "  final logit softcap: 30.0",
            "  canvas length: 16",
            "  context length: 8192",
            "",
            "metadata:",
            '  general.architecture = "diffusion-gemma"',
            '  general.name = "tiny-diffusion-gemma"',
            "  diffusion-gemma.block_count = 2",
            "  diffusion-gemma.context_length = 8192",
            "  diffusion-gemma.embedding_length = 256",
            "  diffusion-gemma.feed_forward_length = 512",
            "  diffusion-gemma.attention.head_count = 8",
            "  diffusion-gemma.attention.head_count_kv = <array of 2>",
            "  diffusion-gemma.rope.freq_base = 1000000.0",
            "  diffusion-gemma.rope.freq_base_swa = 10000.0",
            "  diffusion-gemma.attention.layer_norm_rms_epsilon = 1e-06",
            "  diffusion-gemma.expert_count = 4",
            "  diffusion-gemma.expert_used_count = 2",
            "  diffusion-gemma.attention.key_length = 128",
            "  diffusion-gemma.attention.value_length = 128",
            "  diffusion-gemma.final_logit_softcapping = 30.0",
            "  diffusion-gemma.attention.sliding_window = 128",
            "  diffusion-gemma.attention.shared_kv_layers = 0",
            "  diffusion-gemma.embedding_length_per_layer_input = 0",
            "  diffusion-gemma.attention.sliding_window_pattern = <array of 2>",
            "  diffusion-gemma.attention.key_length_swa = 64",
            "  diffusion-gemma.attention.value_length_swa = 64",
            "  diffusion-gemma.expert_feed_forward_length = 64",
            "  diffusion-gemma.rope.dimension_count = 128",
            "  diffusion-gemma.rope.dimension_count_swa = 64",
            "  diffusion-gemma.attention.causal = false",
            "  diffusion.canvas_length = 16",
            "  tokenizer.ggml.mask_token_id = 4",
            "",
            "tensor types:",
            "  Q6_K: 2",
            "  F32: 31",
            "  Q4_K: 14",
            "  Q5_0: 3",
            "  Q8_0: 2",
            "",
            "tensors:",
        ]
    )
    assert_true(text.startswith(head), "the inspect output starts differently")
    assert_true(
        "  blk.1.ffn_gate_up_exps.weight  Q4_K  [256, 128, 4]\n" in text,
        "a 3-D expert tensor is listed with its shape",
    )
    assert_true(
        "  blk.0.ffn_down_exps.weight  Q5_0  [64, 256, 4]\n" in text,
        "a Q5_0 expert tensor is listed with its type",
    )
    assert_true("blk.1.attn_v.weight" not in text, "a global layer has no V")


def test_inspect_says_why_a_diffusion_gemma_config_is_refused() raises:
    var text = inspect_text(
        "tests/fixtures/diffusion-gemma/shared-kv-layers.gguf"
    )
    assert_true("architecture: diffusion-gemma\n" in text)
    assert_true(
        "config: refused: diffusion-gemma.attention.shared_kv_layers must be 0,"
        " got 2"
        in text,
        "inspect names the refused key",
    )
    assert_true(
        "  diffusion-gemma.attention.shared_kv_layers = 2\n" in text,
        "inspect still lists the metadata",
    )


def test_inspect_lists_every_global_layer() raises:
    var two = inspect_text("tests/fixtures/diffusion-gemma/all-global.gguf")
    assert_true("  blocks: 2 (sliding 0, global 2)\n" in two)
    assert_true("  global layers: 0, 1\n" in two)
    var none = inspect_text("tests/fixtures/diffusion-gemma/all-sliding.gguf")
    assert_true("  blocks: 2 (sliding 2, global 0)\n" in none)
    assert_true("  global layers: none\n" in none)


def test_inspect_prints_no_config_for_the_other_architectures() raises:
    var text = inspect_text("tests/fixtures/tiny-llada.gguf")
    assert_true("architecture: llada\n" in text)
    assert_true("config:" not in text)


def _lines_with(text: String, *needles: String) -> List[String]:
    """The lines of `text` that hold every needle."""
    var out = List[String]()
    for line in text.split("\n"):
        var all_there = True
        for n in needles:
            if n not in line:
                all_there = False
        if all_there:
            out.append(String(line))
    return out^


def test_inspect_lists_every_key_that_took_a_default() raises:
    var text = inspect_text("tests/fixtures/metadata/defaults.gguf")
    assert_true("\ndefaults (key absent from the file):\n" in text)
    assert_true(
        "  llada.attention.head_count_kv = 4 (default, key absent)\n" in text
    )
    assert_true("  llada.context_length = 4096 (default, key absent)\n" in text)
    assert_true(
        "  llada.rope.freq_base = 10000 (default, key absent)\n" in text
    )
    assert_true(
        "  diffusion.shift_logits = false (default, key absent)\n" in text
    )


def test_inspect_lists_no_default_for_a_key_the_file_sets() raises:
    var set_all = inspect_text("tests/fixtures/metadata/keys-set.gguf")
    assert_true("default" not in set_all)
    assert_equal(len(_lines_with(set_all, "rope.freq_base")), 1)
    # tiny-llada sets the rope base and the context length but not the rest.
    var tiny = inspect_text("tests/fixtures/tiny-llada.gguf")
    assert_equal(len(_lines_with(tiny, "rope.freq_base", "default")), 0)
    assert_equal(len(_lines_with(tiny, "context_length", "default")), 0)
    assert_equal(len(_lines_with(tiny, "head_count_kv", "default")), 1)


def test_inspect_lists_the_defaults_of_a_diffusion_gemma_file() raises:
    var text = inspect_text("tests/fixtures/diffusion-gemma/ones.gguf")
    assert_true("  defaults (key absent from the file):\n" in text)
    assert_true(
        "    diffusion-gemma.context_length = 4096 (default, key absent)\n"
        in text
    )
    assert_true(
        "    diffusion-gemma.final_logit_softcapping = 30 (default, key"
        " absent)\n"
        in text
    )
    var full = inspect_text("tests/fixtures/tiny-diffusion-gemma.gguf")
    assert_true("default" not in full)


def test_inspect_says_why_a_llada_config_is_refused() raises:
    var text = inspect_text("tests/fixtures/metadata/kv-heads.gguf")
    assert_true(
        "\nconfig: refused: query heads are not a multiple of KV heads\n"
        in text
    )


def test_run_refuses_a_tokenizer_that_disagrees_with_the_model_vocabulary() raises:
    var ok = GGUFFile("tests/fixtures/gpt2-tiny/with-model.gguf")
    check_tokenizer_matches_model(ok, Tokenizer(ok))
    var bad = GGUFFile("tests/fixtures/gpt2-tiny/with-model-vocab-512.gguf")
    with assert_raises(
        contains="the tokenizer has 6 tokens but the model's vocabulary is 512"
    ):
        check_tokenizer_matches_model(bad, Tokenizer(bad))


def test_inspect_lists_a_value_that_is_not_utf8_without_reading_it() raises:
    var text = inspect_text(
        "tests/fixtures/malformed/string-invalid-value.gguf"
    )
    assert_true(
        "  test.str = <string, 4 bytes, not valid UTF-8 at byte 3>\n" in text
    )
    assert_true("  tokenizer.ggml.tokens = <array of 3>\n" in text)


def test_inspect_rejects_a_second_path() raises:
    assert_equal(
        _check("inspect", ["a", "b"]),
        "unexpected argument 'b' for 'inspect'\nsee: mojo-dllm inspect --help",
    )


def _mapping_error(path: String) -> String:
    try:
        _ = Mapping(path)
    except e:
        return model_error(String(e))
    return String("opened")


def test_missing_model_file_hint() raises:
    var path = "/nonexistent/model.gguf"
    assert_equal(
        _mapping_error(path),
        "cannot open model file "
        + path
        + ": No such file or directory\nto download a model, run: ./mojo-dllm"
        " pull",
    )


def test_missing_model_file_hint_with_quote_colon_in_path() raises:
    var path = "/nonexistent/x': y"
    assert_equal(
        _mapping_error(path),
        "cannot open model file "
        + path
        + ": No such file or directory\nto download a model, run: ./mojo-dllm"
        " pull",
    )


def test_other_open_failures_get_no_pull_hint() raises:
    # ENOTDIR: a path below a regular file. EACCES takes the same branch.
    var path = "tests/fixtures/bad-magic.gguf/x"
    assert_equal(
        _mapping_error(path),
        "cannot open model file " + path + ": Not a directory",
    )
    assert_equal(
        model_error("cannot open file: m.gguf: Permission denied"),
        "cannot open model file m.gguf: Permission denied",
    )


def test_non_gguf_file_hint() raises:
    var path = "build/tests/not-a-gguf.gguf"
    with open(path, "w") as f:
        f.write("<html>404 Not Found</html>")
    var message = String("")
    try:
        _ = GGUFFile(path)
    except e:
        message = model_error(String(e))
    remove(path)
    assert_equal(
        message,
        path
        + " is not a GGUF file (bad magic).\na failed download leaves an HTML"
        " error page with a .gguf name; delete the file and run ./mojo-dllm"
        " pull again",
    )


def test_other_errors_pass_through_unchanged() raises:
    assert_equal(
        model_error("unsupported model architecture: qwen2"),
        "unsupported model architecture: qwen2",
    )
    assert_equal(
        model_error("file is empty or unreadable: m.gguf"),
        "file is empty or unreadable: m.gguf",
    )


comptime DG_TINY = "tests/fixtures/tiny-diffusion-gemma.gguf"


def _gemma_tokens(n: Int) -> List[Int]:
    var tokens = List[Int]()
    for i in range(n):
        tokens.append((i * 37) % 500 + 5)
    return tokens^


def test_logits_runs_diffusion_gemma_over_prompt_and_canvas() raises:
    # 25 tokens: a prompt of 9 and the canvas of 16 the tiny model declares.
    var tokens = _gemma_tokens(25)
    var rows = List[Int]()
    for r in range(9, 25):
        rows.append(r)
    var path = String("build/tests/dg-logits-cli.f32")
    _logits_gemma(DG_TINY, 2, tokens, rows, path)
    var m = DiffusionGemma(DG_TINY, threads=2, max_tokens=25)
    var want = Floats(16 * 512)
    m.forward(tokens, want.ptr())
    var mm = Mapping(path)
    var got = F32Ptr(unsafe_from_address=mm.addr)
    var differ = 0
    for i in range(16 * 512):
        if got.unsafe_load(i) != want[i]:
            differ += 1
    remove(path)
    assert_equal(differ, 0)


def test_logits_refuses_rows_that_are_not_the_canvas() raises:
    var tokens = _gemma_tokens(25)
    var path = String("build/tests/dg-logits-refused.f32")
    var rows = List[Int]()
    for r in range(8, 24):
        rows.append(r)
    with assert_raises(
        contains=(
            "scores the 16 canvas positions, the last of the tokens: --rows"
            " must be 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24, got 8,"
        )
    ):
        _logits_gemma(DG_TINY, 1, tokens, rows, path)
    var all_rows = List[Int]()
    for r in range(25):
        all_rows.append(r)
    with assert_raises(contains="--rows must be 9,10"):
        _logits_gemma(DG_TINY, 1, tokens, all_rows, path)
    with assert_raises(contains="below the canvas length 16"):
        _logits_gemma(DG_TINY, 1, _gemma_tokens(15), List[Int](), path)


def _read_all(path: String) raises -> List[Float32]:
    var mm = Mapping(path)
    var p = F32Ptr(unsafe_from_address=mm.addr)
    var out = List[Float32]()
    for i in range(mm.size // 4):
        out.append(p.unsafe_load(i))
    return out^


def _rows(first: Int) -> List[Int]:
    var rows = List[Int]()
    for r in range(first, first + 16):
        rows.append(r)
    return rows^


def test_logits_cached_path_writes_the_unified_logits() raises:
    var tokens = _gemma_tokens(25)
    var a = String("build/tests/dg-cli-unified.f32")
    var b = String("build/tests/dg-cli-cached.f32")
    _logits_gemma(DG_TINY, 2, tokens, _rows(9), a)
    _logits_gemma(DG_TINY, 2, tokens, _rows(9), b, "cached")
    var x = _read_all(a)
    var y = _read_all(b)
    remove(a)
    remove(b)
    assert_equal(len(x), 16 * 512)
    var differ = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            differ += 1
    assert_equal(differ, 0)


def test_logits_prev_conditions_on_the_previous_logits() raises:
    var tokens = _gemma_tokens(25)
    var prev = String("build/tests/dg-cli-prev.f32")
    var plain = String("build/tests/dg-cli-plain.f32")
    var sc = String("build/tests/dg-cli-sc.f32")
    var sc_cached = String("build/tests/dg-cli-sc-cached.f32")
    _logits_gemma(DG_TINY, 2, tokens, _rows(9), prev)
    var other = _gemma_tokens(25)
    for i in range(9, 25):
        other[i] = (i * 53 + 7) % 500 + 5
    _logits_gemma(DG_TINY, 2, other, _rows(9), plain)
    _logits_gemma(DG_TINY, 2, other, _rows(9), sc, "unified", prev, 1.25)
    _logits_gemma(DG_TINY, 2, other, _rows(9), sc_cached, "cached", prev, 1.25)
    var m = DiffusionGemma(DG_TINY, threads=2, max_tokens=25)
    var want = Floats(16 * 512)
    var src = _read_all(prev)
    for i in range(16 * 512):
        want[i] = src[i]
    m.forward_sc(other, want.ptr(), 1.25)
    var got = _read_all(sc)
    var cached = _read_all(sc_cached)
    var base = _read_all(plain)
    remove(prev)
    remove(plain)
    remove(sc)
    remove(sc_cached)
    var differ = 0
    var moved = 0
    for i in range(16 * 512):
        if got[i] != want[i] or cached[i] != want[i]:
            differ += 1
        if got[i] != base[i]:
            moved += 1
    assert_equal(differ, 0)
    assert_true(moved > 16 * 512 // 2, "--prev changed only " + String(moved))


def test_logits_flag_errors() raises:
    var tokens = _gemma_tokens(25)
    var path = String("build/tests/dg-cli-never.f32")
    with assert_raises(contains="--path is unified or cached, got 'warm'"):
        _logits_gemma(DG_TINY, 1, tokens, _rows(9), path, "warm")
    with assert_raises(contains="--prev and --prev-temp-inv go together"):
        _logits_gemma(DG_TINY, 1, tokens, _rows(9), path, "unified", "x.f32")
    with assert_raises(contains="--prev and --prev-temp-inv go together"):
        _logits_gemma(DG_TINY, 1, tokens, _rows(9), path, "unified", "", 1.25)
    var short = String("build/tests/dg-cli-short.f32")
    with open(short, "w") as f:
        f.write("0123")
    with assert_raises(contains="holds 4 bytes, expected 32768"):
        _logits_gemma(
            DG_TINY, 1, tokens, _rows(9), path, "unified", short, 1.25
        )
    remove(short)


def _same(command: String, rest: List[String], want: String) raises:
    assert_equal(
        _check(command, rest), want + "\nsee: mojo-dllm " + command + " --help"
    )


def test_run_option_values_are_checked_before_the_model_loads() raises:
    var base = List[String](["--model", "m", "--prompt", "p"])
    var cases = List[List[String]]()
    cases.append(["--threshold", "abc"])
    cases.append(["--threshold", "nan"])
    cases.append(["--threshold", "inf"])
    cases.append(["--threshold", "-1"])
    cases.append(["--threshold", "1.5"])
    cases.append(["--device", "foo"])
    cases.append(["--remasking", "foo"])
    var wants = List[String]()
    wants.append("flag --threshold needs a finite number, got 'abc'")
    wants.append("flag --threshold needs a finite number, got 'nan'")
    wants.append("flag --threshold needs a finite number, got 'inf'")
    wants.append("flag --threshold must be between 0 and 1, got -1")
    wants.append("flag --threshold must be between 0 and 1, got 1.5")
    wants.append("flag --device must be cpu or gpu, got 'foo'")
    wants.append(
        "flag --remasking must be one of low_confidence, random, entropy,"
        " maskgit_plus, topk_margin, got 'foo'"
    )
    for i in range(len(cases)):
        var rest = base.copy()
        for x in cases[i]:
            rest.append(x)
        _same("run", rest, wants[i])
    var ok = base.copy()
    for x in List[String](["--threshold", "0.9", "--device", "gpu"]):
        ok.append(x)
    assert_equal(_check("run", ok), "")
    var random = base.copy()
    for x in List[String](["--remasking", "random", "--threshold", "0.5"]):
        random.append(x)
    _same("run", random, "flag --threshold needs low_confidence remasking")


def _arch_run_args(extra: List[String]) raises -> Args:
    var spec = command_spec("run")
    var argv = List[String](["x", "run", "--prompt", "p"])
    for x in extra:
        argv.append(x)
    return Args(argv, spec.values, spec.booleans)


def test_the_block_and_step_split_is_checked_for_llada_only() raises:
    # The split needs the architecture, which only the file says, so the flag
    # check lets it through and `check_arch_flags` applies it to LLaDA.
    var bad_block = _arch_run_args(
        ["--max-tokens", "100", "--block-length", "32"]
    )
    assert_equal(
        _check(
            "run",
            [
                "--model",
                "m",
                "--prompt",
                "p",
                "--max-tokens",
                "100",
                "--block-length",
                "32",
            ],
        ),
        "",
    )
    with assert_raises(
        contains="flag --max-tokens 100 is not a multiple of --block-length 32"
    ):
        check_arch_flags(bad_block, "llada")
    var bad_steps = _arch_run_args(
        ["--max-tokens", "64", "--block-length", "32", "--steps", "3"]
    )
    with assert_raises(
        contains=(
            "flag --steps 3 is not a multiple of the number of blocks 2"
            " (--max-tokens 64 / block length 32)"
        )
    ):
        check_arch_flags(bad_steps, "llada")
    check_arch_flags(
        _arch_run_args(
            ["--max-tokens", "64", "--block-length", "32", "--steps", "4"]
        ),
        "llada",
    )
    # Dream has no blocks: the flag check lets --block-length through (it is
    # the architecture's job) and `check_arch_flags` refuses it for Dream.
    var dream = _arch_run_args(
        ["--max-tokens", "100", "--steps", "100", "--block-length", "32"]
    )
    assert_equal(
        _check(
            "run",
            [
                "--model",
                "m",
                "--prompt",
                "p",
                "--max-tokens",
                "100",
                "--steps",
                "100",
                "--block-length",
                "32",
            ],
        ),
        "",
    )
    with assert_raises(
        contains="--block-length applies to llada only, not dream"
    ):
        check_arch_flags(dream, "dream")
    # No --block-length: the default 32 depends on the architecture.
    with assert_raises(
        contains=(
            "flag --max-tokens 100 is not a multiple of the default block"
            " length 32; pass --block-length"
        )
    ):
        check_arch_flags(_arch_run_args(["--max-tokens", "100"]), "llada")
    check_arch_flags(_arch_run_args(["--max-tokens", "100"]), "dream")


def _latin1_text() -> String:
    # "caf" then 0xE9 (a Latin-1 e acute), " au lait": not UTF-8.
    var raw = List[UInt8]()
    for b in "caf".as_bytes():
        raw.append(b)
    raw.append(0xE9)
    for b in " au lait".as_bytes():
        raw.append(b)
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def test_run_and_tokenize_refuse_a_prompt_that_is_not_utf8() raises:
    # Measured: the Mojo runtime does not refuse such an argument, so the
    # command does, with the offset of the first bad byte.
    for command in ["run", "tokenize"]:
        var rest = List[String](["--model", "m", "--prompt", _latin1_text()])
        _same(
            String(command),
            rest,
            "flag --prompt is not valid UTF-8: invalid byte at offset 3",
        )
    var fine = List[String](["--model", "m", "--prompt", "caf\u00e9 au lait"])
    assert_equal(_check("run", fine), "")
    assert_equal(_check("tokenize", fine), "")
    # The first bad byte of a truncated sequence is its lead byte.
    var raw = List[UInt8]()
    for b in "ab".as_bytes():
        raw.append(b)
    raw.append(0xE2)
    raw.append(0x82)
    var cut = String(StringSpan(unsafe_from_utf8=Span(raw)))
    _same(
        "run",
        List[String](["--model", "m", "--prompt", cut]),
        "flag --prompt is not valid UTF-8: invalid byte at offset 2",
    )


def test_remasking_must_suit_the_architecture_before_the_model_loads() raises:
    var spec = command_spec("run")
    var entropy = Args(
        ["x", "run", "--prompt", "p", "--remasking", "entropy"],
        spec.values,
        spec.booleans,
    )
    with assert_raises(contains="--remasking entropy does not apply to llada"):
        check_arch_flags(entropy, "llada")
    check_arch_flags(entropy, "dream")
    var random = Args(
        ["x", "run", "--prompt", "p", "--remasking", "random"],
        spec.values,
        spec.booleans,
    )
    with assert_raises(contains="--remasking random does not apply to dream"):
        check_arch_flags(random, "dream")
    check_arch_flags(random, "llada")


def test_logits_option_values_are_checked_before_the_model_loads() raises:
    var base = List[String](
        ["--model", "m", "--tokens", "1", "--rows", "0", "--out", "o"]
    )
    for bad in List[String](["0", "-1"]):
        var rest = base.copy()
        rest.append("--prev-temp-inv")
        rest.append(bad)
        _same(
            "logits", rest, "flag --prev-temp-inv must be above 0, got " + bad
        )
    for bad in List[String](["abc", "nan", "inf"]):
        var rest = base.copy()
        rest.append("--prev-temp-inv")
        rest.append(bad)
        _same(
            "logits",
            rest,
            "flag --prev-temp-inv needs a finite number, got '" + bad + "'",
        )
    var path = base.copy()
    path.append("--path")
    path.append("warm")
    _same("logits", path, "flag --path must be unified or cached, got 'warm'")
    var device = base.copy()
    device.append("--device")
    device.append("tpu")
    _same("logits", device, "flag --device must be cpu or gpu, got 'tpu'")


def test_a_bad_option_exits_two_without_opening_the_model() raises:
    # The model path does not exist: an exit of 1 would mean the open ran.
    var head = List[String](
        ["mojo-dllm", "run", "--model", "/nonexistent/model.gguf", "--prompt"]
    )
    for flag in List[String](["--device", "--remasking", "--threshold"]):
        var argv = head.copy()
        argv.append("hi")
        argv.append(flag)
        argv.append("nan" if flag == "--threshold" else "foo")
        assert_equal(execute(argv), 2)


def test_logits_checks_rows_and_prev_before_the_weights_load() raises:
    # all-sliding.gguf is a thin file with no norm or attention tensors, so constructing the model fails
    # with "tensor ... not found"; a check made first reports the flag instead.
    var bad = "tests/fixtures/diffusion-gemma/all-sliding.gguf"
    var path = String("build/tests/dg-order.f32")
    with assert_raises(contains="--rows must be 9,10"):
        _logits_gemma(bad, 1, _gemma_tokens(25), _rows(8), path)
    var short = String("build/tests/dg-order-short.f32")
    with open(short, "w") as f:
        f.write("0123")
    with assert_raises(contains="holds 4 bytes, expected 32768"):
        _logits_gemma(
            bad, 1, _gemma_tokens(25), _rows(9), path, "unified", short, 1.25
        )
    remove(short)
    with assert_raises(contains="below the canvas length 16"):
        _logits_gemma(bad, 1, _gemma_tokens(15), List[Int](), path)


def _run_args(max_tokens: String, extra: List[String]) raises -> Args:
    var spec = command_spec("run")
    var argv = List[String](
        [
            "mojo-dllm",
            "run",
            "--model",
            "tests/fixtures/gpt2-tiny/with-model.gguf",
            "--prompt",
            "ab",
            "--no-chat",
            "--threads",
            "1",
            "--max-tokens",
            max_tokens,
        ]
    )
    for x in extra:
        argv.append(x)
    return Args(argv, spec.values, spec.booleans)


def test_run_refuses_a_canvas_longer_than_the_context_length() raises:
    # with-model.gguf has no llada.context_length, so the default 4096 holds. One block of 4096 tokens
    # and one step is the cheapest run that is one token too long once the
    # prompt is counted.
    var a = _run_args("4096", ["--block-length", "4096", "--steps", "1"])
    with assert_raises(
        contains="exceeds the context length 4096 (llada.context_length)"
    ):
        cmd_run(a)


def test_run_refuses_a_generation_length_that_overflows_the_allocations() raises:
    # 2^62 passes the integer checks and the block rule, so only the context
    # length stands between it and an allocation of 2^62 * hidden floats.
    var a = _run_args("4611686018427387904", [])
    with assert_raises(contains="exceeds the context length 4096"):
        cmd_run(a)
    var b = _run_args("9223372036854775807", ["--block-length", "1"])
    with assert_raises(contains="exceeds the context length 4096"):
        cmd_run(b)


def test_run_refuses_a_diffusion_gemma_file_before_the_tokenizer_is_built() raises:
    # The tiny file has no tokenizer: any error about one would mean the
    # parse work started before the architecture was refused.
    var spec = command_spec("run")
    var a = Args(
        ["x", "run", "--model", DG_TINY, "--prompt", "hi"],
        spec.values,
        spec.booleans,
    )
    with assert_raises(
        contains=(
            "diffusion-gemma does not run on this model: `run` waits for the"
            " model to be wired into the generation loop (#277), so use"
            " `logits` (CPU) until then"
        )
    ):
        cmd_run(a)


def test_both_refusals_of_a_diffusion_gemma_run_say_the_same() raises:
    # `run` stops in main before any parse work, and `check_runnable` stops a
    # model that is built anyway (the GPU path). One text, so a reader sees
    # one issue number.
    var spec = command_spec("run")
    var a = Args(
        ["x", "run", "--model", DG_TINY, "--prompt", "hi"],
        spec.values,
        spec.booleans,
    )
    var from_run = String("")
    try:
        cmd_run(a)
    except e:
        from_run = String(e)
    var from_config = String("")
    try:
        ModelConfig(GGUFFile(DG_TINY)).check_runnable()
    except e:
        from_config = String(e)
    assert_true(from_run.byte_length() > 0)
    assert_equal(from_run, from_config)


def test_run_help_says_the_entropy_bound_flags_have_no_effect_yet() raises:
    var text = usage_for("run")
    assert_true(
        text.__contains__(
            "the eb-* flags have no effect until run accepts"
            " diffusion-gemma (#277)"
        )
    )


def test_dream_refuses_the_flags_that_only_llada_reads() raises:
    var spec = command_spec("run")
    var cases = List[List[String]]()
    cases.append(["--full-logits"])
    cases.append(["--threshold", "0.5"])
    cases.append(["--dual-cache"])
    cases.append(["--block-length", "32"])
    for given in cases:
        var argv = List[String](["x", "run", "--prompt", "p"])
        for g in given:
            argv.append(g)
        var a = Args(argv, spec.values, spec.booleans)
        var flag = given[0]
        with assert_raises(contains=flag + " applies to llada only, not dream"):
            check_arch_flags(a, "dream")
        check_arch_flags(a, "llada")


def test_json_reports_the_remasking_the_run_used() raises:
    var cfg = GenConfig()
    var llada = sampler_json("llada", cfg)
    assert_equal(
        llada,
        (
            ', "remasking": "low_confidence", "block_length": 32,'
            ' "full_logits": false'
        ),
    )
    cfg.remasking = "random"
    cfg.full_logits = True
    cfg.block_length = 8
    assert_equal(
        sampler_json("llada", cfg),
        ', "remasking": "random", "block_length": 8, "full_logits": true',
    )
    # Dream maps low_confidence to entropy and has neither a block nor a
    # full-logits mode, so its JSON names the algorithm and omits both.
    var dream = GenConfig()
    assert_equal(sampler_json("dream", dream), ', "remasking": "entropy"')
    dream.remasking = "topk_margin"
    assert_equal(sampler_json("dream", dream), ', "remasking": "topk_margin"')


def test_the_context_length_bound_is_exact() raises:
    check_context_length(10, 4086, 4096, "llada")  # 4096 tokens fit
    with assert_raises(
        contains=(
            "prompt 10 + --max-tokens 4087 = 4097 exceeds the context length"
            " 4096 (llada.context_length)"
        )
    ):
        check_context_length(10, 4087, 4096, "llada")
    with assert_raises(contains="prompt 5000 + --max-tokens 1 = 5001 exceeds"):
        check_context_length(5000, 1, 4096, "dream")
    # The sum wraps for a huge length; the message must not print the wrap.
    with assert_raises(
        contains=(
            "--max-tokens 9223372036854775807 = more than 9223372036854775807"
        )
    ):
        check_context_length(10, Int.MAX, 4096, "llada")


def _floats(n: Int) raises -> Floats:
    var f = Floats(n)
    for i in range(n):
        f[i] = Float32(i) * 0.25 - 3.0
    return f^


def test_a_full_device_raises_instead_of_aborting() raises:
    # `write_bytes` aborts the process on ENOSPC (stdlib file.mojo:598); the
    # checked writer raises with the system's reason. /dev/full always answers
    # ENOSPC to a write.
    if not exists("/dev/full"):
        return
    var data = _floats(1024)
    with assert_raises(contains="No space left on device"):
        write_bytes_checked(
            "/dev/full", data.ptr().unsafe_bitcast[UInt8](), 4096
        )


def test_a_device_that_keeps_nothing_is_caught_by_the_size_check() raises:
    # /dev/null accepts every byte and holds none of them.
    if not exists("/dev/null"):
        return
    var data = _floats(1024)
    with assert_raises(
        contains="/dev/null holds 0 bytes after the write, expected 4096"
    ):
        write_bytes_checked(
            "/dev/null", data.ptr().unsafe_bitcast[UInt8](), 4096
        )


def test_f32_file_is_written_whole_under_its_final_name_only() raises:
    var path = String("build/tests/f32-whole.f32")
    var data = _floats(300)
    write_f32_file(path, data.ptr(), 300)
    assert_equal(len(_read_all(path)), 300)
    var back = _read_all(path)
    for i in range(300):
        assert_equal(back[i], data[i])
    assert_true(not exists(path + ".tmp"), "no temporary file is left behind")
    remove(path)


def test_f32_file_is_written_straight_to_a_special_file() raises:
    # /dev/null is not a regular file: a temporary name beside it cannot be
    # made, and a size check of the target would read 0. The bytes go to it
    # directly and the call succeeds.
    var data = _floats(8)
    write_f32_file("/dev/null", data.ptr(), 8)
    assert_true(exists("/dev/null"))
    assert_true(not exists("/dev/null.tmp"))


def test_f32_file_replaces_a_symlink_and_a_read_only_regular_file() raises:
    # Regular files (and a symlink to one) keep the temporary name and the
    # rename: the link or the read-only file is replaced, not written through.
    var real = String("build/tests/f32-link-real.f32")
    var link = String("build/tests/f32-link.f32")
    if exists(link):
        remove(link)
    with open(real, "w") as f:
        f.write("old")
    var data = _floats(4)
    var target = real
    var name = link
    var rc = external_call["symlink", c_int](
        target.as_c_string_span(), name.as_c_string_span()
    )
    assert_equal(Int(rc), 0)
    write_f32_file(link, data.ptr(), 4)
    assert_equal(len(_read_all(link)), 4)
    with open(real, "r") as f:
        assert_equal(f.read(), "old")
    remove(link)
    remove(real)


def test_f32_file_in_a_directory_that_cannot_exist_leaves_nothing() raises:
    # Mojo's open() makes missing parent directories, and /proc takes none.
    var path = String("/proc/mojo-dllm-no-such-dir/out.f32")
    var data = _floats(8)
    with assert_raises(contains="cannot write " + path):
        write_f32_file(path, data.ptr(), 8)
    assert_true(not exists(path))
    assert_true(not exists(path + ".tmp"))


def test_a_failed_rename_removes_the_temporary_file_and_keeps_the_old_one() raises:
    # The final name is a non-empty directory, so the rename fails after the
    # temporary file was written and checked.
    var dir = String("build/tests/f32-rename-dir")
    if exists(dir + "/keep"):
        remove(dir + "/keep")
    if exists(dir):
        rmdir(dir)
    mkdir(dir)
    with open(dir + "/keep", "w") as f:
        f.write("old")
    var data = _floats(8)
    with assert_raises(contains="cannot rename " + dir + ".tmp to " + dir):
        write_f32_file(dir, data.ptr(), 8)
    assert_true(not exists(dir + ".tmp"), "the temporary file was removed")
    with open(dir + "/keep", "r") as f:
        assert_equal(f.read(), "old")
    remove(dir + "/keep")
    rmdir(dir)


def test_the_logits_command_writes_every_requested_llada_row() raises:
    # The generic writer, not DiffusionGemma's: 2 rows of the tiny model's 512
    # logits, byte for byte what a direct forward pass gives.
    var path = String("build/tests/llada-logits-command.f32")
    var model = "tests/fixtures/tiny-llada.gguf"
    assert_equal(
        execute(
            [
                "mojo-dllm",
                "logits",
                "--model",
                model,
                "--tokens",
                "5,17,300,42",
                "--rows",
                "1,3",
                "--out",
                path,
                "--threads",
                "1",
            ]
        ),
        0,
    )
    var m = DiffusionLM(model, threads=1, max_tokens=4)
    var want = Floats(2 * 512)
    m.forward([5, 17, 300, 42], [1, 3], want.ptr())
    var got = _read_all(path)
    assert_equal(len(got), 2 * 512)
    var differ = 0
    for i in range(2 * 512):
        if got[i] != want[i]:
            differ += 1
    assert_equal(differ, 0)
    assert_true(not exists(path + ".tmp"))
    remove(path)


def test_the_logits_command_writes_nothing_when_the_output_cannot_be_made() raises:
    var rows = List[Int]()
    for r in range(9, 25):
        rows.append(r)
    var path = String("/proc/mojo-dllm-no-such-dir/dg-logits.f32")
    assert_equal(execute(_logits_args(_csv(rows), path, "cpu")), 1)
    assert_true(not exists(path))


def test_peak_rss_is_a_measurement_or_an_error_never_a_zero() raises:
    var peak = peak_rss_bytes()
    assert_true(peak > 1_000_000, "this process holds more than 1 MB")
    assert_true(Int(peak_rss_json()) >= peak)
    assert_true(peak_rss_text().endswith(" GB"))
    # An invalid selector makes getrusage fail with EINVAL; the old code read
    # the buffer it never filled and reported a peak of 0.
    with assert_raises(contains="getrusage failed: Invalid argument"):
        _ = peak_rss_bytes(99)
    # The two report lines never print the failed figure as a number.
    assert_equal(peak_rss_json(99), "null")
    assert_equal(
        peak_rss_text(99), "unavailable (getrusage failed: Invalid argument)"
    )


def test_the_gemma_only_flags_are_named_for_other_models() raises:
    var spec = command_spec("logits")
    var none = Args(["x", "logits", "--model", "m"], spec.values, spec.booleans)
    assert_equal(gemma_only_flags(none), "")
    var some = Args(
        ["x", "logits", "--path", "cached", "--prev-temp-inv", "1"],
        spec.values,
        spec.booleans,
    )
    assert_equal(gemma_only_flags(some), "--path, --prev-temp-inv")


def _logits_args(rows: String, path: String, device: String) -> List[String]:
    return [
        "mojo-dllm",
        "logits",
        "--model",
        DG_TINY,
        "--tokens",
        _csv(_gemma_tokens(25)),
        "--rows",
        rows,
        "--out",
        path,
        "--threads",
        "1",
        "--device",
        device,
    ]


def test_the_logits_command_runs_diffusion_gemma_on_the_cpu() raises:
    var rows = List[Int]()
    for r in range(9, 25):
        rows.append(r)
    var path = String("build/tests/dg-logits-command.f32")
    assert_equal(execute(_logits_args(_csv(rows), path, "cpu")), 0)
    var m = DiffusionGemma(DG_TINY, threads=1, max_tokens=25)
    var want = Floats(16 * 512)
    m.forward(_gemma_tokens(25), want.ptr())
    var mm = Mapping(path)
    var got = F32Ptr(unsafe_from_address=mm.addr)
    var differ = 0
    for i in range(16 * 512):
        if got.unsafe_load(i) != want[i]:
            differ += 1
    remove(path)
    assert_equal(differ, 0)


def test_the_logits_command_refuses_the_gpu_for_diffusion_gemma() raises:
    var rows = List[Int]()
    for r in range(9, 25):
        rows.append(r)
    var path = String("build/tests/dg-logits-command-gpu.f32")
    if exists(path):
        remove(path)
    assert_equal(execute(_logits_args(_csv(rows), path, "gpu")), 1)
    assert_true(not exists(path))


# ---- the run flags reach the sampler (#228) ---------------------------------


def _run_args(argv: List[String]) raises -> Args:
    var spec = command_spec("run")
    return Args(argv, spec.values, spec.booleans)


def test_gen_config_maps_every_run_flag_to_its_field() raises:
    # Every flag gets a value that differs from its default and from every
    # other flag's value, so a swap of two fields in the mapping shows.
    var a = _run_args(
        [
            "x",
            "run",
            "--max-tokens",
            "24",
            "--steps",
            "12",
            "--block-length",
            "6",
            "--seed",
            "7",
            "--temperature",
            "0.5",
            "--remasking",
            "random",
            "--threshold",
            "0.9",
            "--full-logits",
            "--dual-cache",
            "--threads",
            "3",
            "--eb-steps",
            "10",
            "--eb-t-min",
            "0.3",
            "--eb-t-max",
            "0.7",
            "--eb-bound",
            "0.2",
            "--eb-stability",
            "2",
            "--eb-confidence",
            "0.01",
        ]
    )
    var c = gen_config(a)
    assert_equal(c.gen_length, 24)
    assert_equal(c.steps, 12)
    assert_equal(c.block_length, 6)
    assert_equal(c.seed, 7)
    assert_equal(c.temperature, 0.5)
    assert_equal(c.remasking, "random")
    assert_equal(c.threshold, 0.9)
    assert_true(c.full_logits)
    assert_true(c.dual_cache)
    assert_equal(c.threads, 3)
    assert_equal(c.eb_steps, 10)
    assert_equal(c.eb_t_min, 0.3)
    assert_equal(c.eb_t_max, 0.7)
    assert_equal(c.eb_bound, 0.2)
    assert_equal(c.eb_stability, 2)
    assert_equal(c.eb_confidence, 0.01)


def test_gen_config_reads_each_boolean_flag_on_its_own() raises:
    # Given together, two flags cannot tell a field that reads the other's.
    var full = gen_config(_run_args(["x", "run", "--full-logits"]))
    assert_true(full.full_logits and not full.dual_cache)
    var dual = gen_config(_run_args(["x", "run", "--dual-cache"]))
    assert_true(dual.dual_cache and not dual.full_logits)


def test_gen_config_defaults() raises:
    var d = GenConfig()
    var c = gen_config(_run_args(["x", "run", "--threads", "2"]))
    assert_equal(c.gen_length, 128)
    assert_equal(c.steps, 128)
    assert_equal(c.block_length, 32)
    assert_equal(c.seed, 42)
    assert_equal(c.temperature, 0.0)
    assert_equal(c.remasking, "low_confidence")
    assert_equal(c.threshold, 0.0)
    assert_true(not c.full_logits)
    assert_true(not c.dual_cache)
    assert_equal(c.eb_steps, d.eb_steps)
    assert_equal(c.eb_t_min, d.eb_t_min)
    assert_equal(c.eb_t_max, d.eb_t_max)
    assert_equal(c.eb_bound, d.eb_bound)
    assert_equal(c.eb_stability, d.eb_stability)
    assert_equal(c.eb_confidence, d.eb_confidence)


def test_gen_config_derives_steps_and_block_length_from_max_tokens() raises:
    var short = gen_config(
        _run_args(["x", "run", "--max-tokens", "16", "--threads", "2"])
    )
    assert_equal(short.steps, 16)
    assert_equal(short.block_length, 16)
    var long = gen_config(
        _run_args(["x", "run", "--max-tokens", "100", "--threads", "2"])
    )
    assert_equal(long.steps, 100)
    assert_equal(long.block_length, 32)
    var given = gen_config(
        _run_args(
            [
                "x",
                "run",
                "--max-tokens",
                "100",
                "--steps",
                "10",
                "--threads",
                "2",
            ]
        )
    )
    assert_equal(given.steps, 10)
    assert_equal(given.block_length, 32)


def test_run_refuses_a_bad_visual_interval_before_it_loads_anything() raises:
    with assert_raises(contains="--visual-interval needs an integer"):
        cmd_run(_run_args(["x", "run", "--visual-interval", "x", "--visual"]))
    with assert_raises(contains="--visual-interval must be at least 1"):
        cmd_run(_run_args(["x", "run", "--visual-interval", "0", "--visual"]))
    with assert_raises(contains="--visual-interval needs --visual"):
        cmd_run(_run_args(["x", "run", "--visual-interval", "3"]))


def test_run_refuses_an_unknown_device_before_it_loads_anything() raises:
    # `check_args` refuses a bad --device first on the command line; this is
    # the command's own check, reached with no model to load.
    with assert_raises(contains="unknown device: tpu (expected cpu or gpu)"):
        cmd_run(_run_args(["x", "run", "--device", "tpu"]))
    with assert_raises(contains="unknown device: tpu (expected cpu or gpu)"):
        check_device("tpu")
    check_device("cpu")
    check_device("gpu")


def test_logits_refuses_an_unknown_device() raises:
    var spec = command_spec("logits")
    var a = Args(
        [
            "x",
            "logits",
            "--model",
            "tests/fixtures/tiny-llada.gguf",
            "--tokens",
            "1,2",
            "--rows",
            "0",
            "--out",
            "build/never-written.f32",
            "--device",
            "tpu",
        ],
        spec.values,
        spec.booleans,
    )
    with assert_raises(contains="unknown device: tpu (expected cpu or gpu)"):
        cmd_logits(a)
    assert_true(not exists("build/never-written.f32"))


def _json_keys(s: String) -> List[String]:
    """The keys of the top-level object of `s`, in order of appearance."""
    var keys = List[String]()
    var depth = 0
    var in_string = False
    var escaped = False
    var current = String("")
    var last_string = String("")
    var after_string = False
    for ch in s.codepoints():
        var c = String(ch)
        if in_string:
            if escaped:
                escaped = False
                current += c
            elif c == "\\":
                escaped = True
                current += c
            elif c == '"':
                in_string = False
                last_string = current
                after_string = True
            else:
                current += c
            continue
        if c == '"':
            in_string = True
            current = String("")
            continue
        if c == ":" and after_string and depth == 1:
            keys.append(last_string)
        after_string = False
        if c == "{" or c == "[":
            depth += 1
        elif c == "}" or c == "]":
            depth -= 1
    return keys^


def _expected_keys(arch: String) -> List[String]:
    var keys: List[String] = ["runtime", "version", "arch", "device", "threads"]
    keys.append("prompt_tokens")
    keys.append("gen_length")
    keys.append("steps")
    keys.append("remasking")
    if arch != "dream":
        keys.append("block_length")
        keys.append("full_logits")
    for k in [
        "stopped_early",
        "forward_passes",
        "steps_taken",
        "stop_reason",
        "load_s",
        "tokenizer_s",
        "generate_s",
        "ms_per_step",
        "forward_ms",
        "gemm_ms_per_forward",
        "attention_ms_per_forward",
        "lm_head_ms_per_forward",
        "peak_rss_bytes",
        "step_ms",
        "tokens",
        "text",
    ]:
        keys.append(String(k))
    return keys^


def _check_json_keys(arch: String) raises:
    var cfg = GenConfig()
    cfg.threads = 2
    var res = GenResult()
    res.tokens = [1, 2, 3]
    res.step_ms = [1.5, 2.5]
    res.forward_passes = 2
    var line = run_json(
        arch, "cpu", cfg, 1, res, Timings(), 0.25, 0.125, 'a "quoted" {text}'
    )
    var got = _json_keys(line)
    var want = _expected_keys(arch)
    assert_equal(len(got), len(want), arch + ": " + line)
    for i in range(len(want)):
        assert_equal(got[i], want[i], arch + " key " + String(i))
    assert_true(line.startswith("{") and line.endswith("}"))


def test_run_json_has_every_key_once_in_order_llada() raises:
    _check_json_keys("llada")


def test_run_json_has_every_key_once_in_order_dream() raises:
    _check_json_keys("dream")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
