from std.os import remove
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from cli.args import Args
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.sys.mmap import Mapping
from main import (
    Decision,
    boolean_flags,
    chat_prompt,
    check_args,
    command_spec,
    decide,
    execute,
    json_str,
    _csv,
    is_top_level_help,
    model_error,
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
    assert_equal(chat_prompt("unknown", "hi"), llada)


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
    assert_equal(len(run.values), 12)
    assert_equal(len(run.booleans), 8)
    var tokenize = command_spec("tokenize")
    assert_equal(len(tokenize.values), 2)
    assert_equal(len(tokenize.booleans), 1)
    var inspect = command_spec("inspect")
    assert_equal(len(inspect.values), 0)
    assert_equal(len(inspect.booleans), 0)
    assert_true(inspect.takes_path)
    var logits = command_spec("logits")
    assert_equal(len(logits.values), 6)
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
    assert_equal(d.code, 2)
    assert_true(d.is_error)
    assert_equal(d.text, "unknown command: bogus\nsee: mojo-dllm --help")
    assert_equal(execute(["mojo-dllm", "bogus"]), 2)


def test_help_exits_zero_and_wins_over_other_problems() raises:
    for command in List[String](["", "help", "--help", "-h"]):
        var d = _decide(command, [])
        assert_equal(d.code, 0)
        assert_true(not d.is_error)
        assert_equal(d.text, usage())
    for command in List[String](["run", "tokenize", "inspect", "logits"]):
        var d = _decide(command, ["--stpes", "--steps", "abc", "--help"])
        assert_equal(d.code, 0)
        assert_equal(d.text, usage_for(command))
        assert_true(not d.proceed)
    # --help before a value flag that has no value: help, not "needs a value".
    var late = _decide("run", ["--help", "--prompt"])
    assert_equal(late.code, 0)
    assert_equal(late.text, usage_for("run"))
    assert_equal(execute(["mojo-dllm", "run", "--help", "--prompt"]), 0)
    assert_equal(execute(["mojo-dllm", "--help"]), 0)


def test_usage_errors_exit_two() raises:
    var unknown = _decide("run", ["--stpes", "3"])
    assert_equal(unknown.code, 2)
    assert_true(unknown.is_error)
    var missing = _decide("run", ["--model", "m", "--prompt"])
    assert_equal(missing.code, 2)
    assert_equal(
        missing.text,
        "flag --prompt needs a value\nsee: mojo-dllm run --help",
    )
    var number = _decide(
        "run", ["--model", "m", "--prompt", "p", "--steps", "abc"]
    )
    assert_equal(number.code, 2)
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


def test_valid_arguments_proceed() raises:
    var d = _decide("run", ["--model", "m", "--prompt", "p"])
    assert_equal(d.code, 0)
    assert_true(d.proceed)
    assert_true(not d.is_error)


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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
