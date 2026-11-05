"""The gemma4 tokenizer on DiffusionGemma's real vocabulary (issue #62).

`tests/fixtures/gemma4-tokenizer.gguf` holds the 262,144 tokens and the 472
merges (of 514,906) that apply to those strings, from the Q4_K_M file, and `gemma4-tokenizer-golden.tsv` the ids that
`llama-tokenize` from the ggml-org/llama.cpp#24423 branch gives for 54 strings
on the full model (`gemma4-tokenizer-golden.provenance.json` has the commit
and the command). The tiny vocabulary in `tests/test_gemma4_tiny.mojo` covers
the rules a hand-written vocabulary can pin down exactly.
"""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from cli.args import parse_csv_ints
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.tokenizer.bpe import Tokenizer
from mojo_dllm.tokenizer.gemma4 import gemma4_chat_prompt

comptime FIX = "tests/fixtures/"
comptime BLOCK = "▁"


def _hex_to_string(h: String) raises -> String:
    var b = h.as_bytes()
    var out = List[UInt8]()
    for i in range(0, len(b), 2):
        var hi = Int(b[i])
        var lo = Int(b[i + 1])
        hi = hi - 48 if hi <= 57 else hi - 87
        lo = lo - 48 if lo <= 57 else lo - 87
        out.append(UInt8(hi * 16 + lo))
    return String(StringSpan(unsafe_from_utf8=Span(out)))


def _cases() raises -> List[Tuple[String, List[Int]]]:
    var out = List[Tuple[String, List[Int]]]()
    with open(FIX + "gemma4-tokenizer-golden.tsv", "r") as f:
        var text = f.read()
        for line in text.split("\n"):
            var l = String(line)
            if l.byte_length() == 0:
                continue
            var parts = l.split("\t")
            out.append(
                (
                    _hex_to_string(String(parts[0])),
                    parse_csv_ints(String(parts[1])),
                )
            )
    return out^


def _show(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        s += (", " if i else "") + String(ids[i])
    return s + "]"


def _same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _hex2(b: Int) -> String:
    var d = "0123456789ABCDEF".as_bytes()
    var out = List[UInt8]()
    out.append(d[b // 16])
    out.append(d[b % 16])
    return String(StringSpan(unsafe_from_utf8=Span(out)))


def _reference_encode(tok: Tokenizer, text: String) raises -> List[Int]:
    """Quadratic BPE over one word: no heap, every pair rescanned after a merge.
    """
    var raw = String(text.replace(" ", BLOCK))
    var b = raw.as_bytes()
    var syms = List[String]()
    var i = 0
    while i < len(b):
        var lead = Int(b[i]) >> 4
        var n = 1
        if lead == 12 or lead == 13:
            n = 2
        elif lead == 14:
            n = 3
        elif lead == 15:
            n = 4
        n = min(n, len(b) - i)
        syms.append(String(StringSpan(unsafe_from_utf8=b[i : i + n])))
        i += n
    while len(syms) > 1:
        var best = -1
        var best_rank = Int.MAX
        for q in range(len(syms) - 1):
            var r = tok.pair_rank(syms[q], syms[q + 1])
            if r >= 0 and r < best_rank:
                best_rank = r
                best = q
        if best < 0:
            break
        var left = syms[best].copy()
        var right = syms[best + 1].copy()
        syms[best] = left + right
        _ = syms.pop(best + 1)
    var out = List[Int]()
    for s in syms:
        var id = tok.token_id(s)
        if id >= 0:
            out.append(id)
        else:
            for byte in s.as_bytes():
                out.append(tok.token_id("<0x" + _hex2(Int(byte)) + ">"))
    return out^


def _real() raises -> Tokenizer:
    return Tokenizer(GGUFFile(FIX + "gemma4-tokenizer.gguf"))


def _golden_ids(text: String) raises -> List[Int]:
    """The reference's ids for a case that must be in the golden file."""
    for c in _cases():
        if c[0] == text:
            return c[1].copy()
    raise Error("no golden case for " + repr(text))


def test_gemma4_golden_ids_match_llama_cpp() raises:
    var tok = _real()
    var cases = _cases()
    assert_equal(len(cases), 54)
    var failures = 0
    for c in cases:
        var got = tok.encode(c[0])
        if not _same(got, c[1]):
            failures += 1
            print(
                "MISMATCH", repr(c[0]), "got", _show(got), "want", _show(c[1])
            )
    assert_equal(failures, 0)


def test_gemma4_golden_ids_decode_back_to_the_text() raises:
    # Decoding turns every U+2581 into a space (llama_unescape_whitespace), so
    # the one case that holds a literal U+2581 cannot round-trip. Every other
    # case must, special tokens included.
    var tok = _real()
    var exceptions = 0
    for c in _cases():
        var back = tok.decode(c[1], skip_special=False)
        if BLOCK in c[0]:
            exceptions += 1
            assert_equal(back, String(c[0].replace(BLOCK, " ")))
        else:
            assert_equal(back, c[0])
    assert_equal(exceptions, 1)


def test_gemma4_encode_then_decode_round_trips() raises:
    var tok = _real()
    var exceptions = 0
    for c in _cases():
        var back = tok.decode(tok.encode(c[0]), skip_special=False)
        if back != c[0]:
            exceptions += 1
            assert_true(BLOCK in c[0], "round trip failed for " + repr(c[0]))
    assert_equal(exceptions, 1)


def test_gemma4_bos_is_added_whatever_the_file_says() raises:
    var tok = _real()
    assert_true(tok.add_bos)
    assert_equal(tok.bos_id, 2)
    for c in _cases():
        var want = List[Int]()
        want.append(2)
        for t in c[1]:
            want.append(t)
        var got = tok.encode(c[0], add_bos=tok.add_bos)
        assert_true(_same(got, want), "BOS rule, " + repr(c[0]))
    # Without add_bos the ids are the goldens exactly: no BOS of its own.
    assert_true(_same(tok.encode("Hello world"), _golden_ids("Hello world")))


def test_gemma4_special_ids() raises:
    var tok = _real()
    assert_equal(tok.vocab_size(), 262144)
    assert_equal(tok.bos_id, 2)
    assert_equal(tok.eos_id, 1)
    assert_equal(tok.pad_id, 0)
    assert_equal(tok.unk_id, 3)
    assert_equal(tok.mask_id, 4)
    assert_equal(tok.eot_id, 106)  # <turn|>
    assert_true(tok.is_eog(1))
    assert_true(tok.is_eog(106))
    assert_true(tok.is_eog(50))  # <|tool_response>, which llama.cpp ends on too
    assert_true(not tok.is_eog(2))
    assert_true(not tok.is_eog(105))


def test_gemma4_spaces_become_lower_blocks_without_a_prefix() raises:
    var tok = _real()
    # add_space_prefix is false: "Hello world" starts with "Hello", not "▁Hello".
    var hello = tok.encode("Hello world")
    assert_equal(len(hello), 2)
    assert_equal(tok.decode([hello[0]]), "Hello")
    assert_equal(tok.decode([hello[1]]), " world")
    # A lone space is the single token "▁", and a run of 16 is one token too.
    assert_equal(len(tok.encode(" ")), 1)
    assert_equal(tok.decode(tok.encode(" ")), " ")
    assert_equal(len(tok.encode("                ")), 1)
    # A literal U+2581 in the text is the same character the normalizer writes.
    assert_true(_same(tok.encode(BLOCK + "x"), tok.encode(" x")))


def test_gemma4_newline_runs_are_split_off_and_kept_whole() raises:
    var tok = _real()
    assert_true(_same(tok.encode("\n"), [107]))
    assert_true(_same(tok.encode("\n\n"), [108]))
    assert_true(_same(tok.encode("\n\n\n\n\n\n\n\n"), [114]))
    # No merge crosses a newline: "x  \n  y" is x, ▁▁, \n, ▁▁, y in five pieces.
    assert_true(
        _same(tok.encode("x  \n  y"), _golden_ids("x  \n  y")), "x  \\n  y"
    )
    assert_equal(len(tok.encode("x  \n  y")), 5)


def test_gemma4_byte_tokens_are_ids_238_to_493() raises:
    var tok = _real()
    # <0x00> is id 238, <0xFF> is 493, and each decodes to its own byte.
    for b in range(256):
        var out = tok.decode([238 + b], skip_special=False)
        if b < 128:
            assert_equal(out.byte_length(), 1)
            assert_equal(Int(out.as_bytes()[0]), b)
        else:
            # A lone continuation or lead byte is not valid text: U+FFFD.
            assert_equal(out, "�")
    # Three bytes of one character decode to that character.
    assert_equal(tok.decode([238 + 0xE2, 238 + 0x96, 238 + 0x81]), BLOCK)


def test_gemma4_byte_fallback_for_characters_without_a_token() raises:
    var tok = _real()
    # U+10FFFF is F4 8F BF BF: no merge or token covers it.
    var ids = tok.encode("\U0010ffff")
    assert_true(_same(ids, [238 + 0xF4, 238 + 0x8F, 238 + 0xBF, 238 + 0xBF]))
    # U+0378 (unassigned) is CD B8, also without a token.
    assert_true(_same(tok.encode("\u0378"), [238 + 0xCD, 238 + 0xB8]))
    # A character that does have a token is not split into bytes: U+E000 is
    # token 245863, and the control character 0x01 is token 249732.
    assert_true(_same(tok.encode("\ue000"), [245863]))
    assert_true(_same(tok.encode("\x01"), [249732]))
    # The text "<0x41>" is ordinary text, not the byte token for "A".
    assert_true(not _same(tok.encode("<0x41>"), [238 + 0x41]))


def test_gemma4_chat_markers_are_single_tokens() raises:
    var tok = _real()
    var ids = tok.encode("<|turn>user\nHi<turn|>\n<|turn>model\n")
    assert_equal(ids[0], 105)
    assert_equal(len(ids), 9)
    assert_equal(ids[4], 106)
    assert_equal(ids[6], 105)
    # <bos> typed into the text is token 2, on top of the BOS added by flag.
    assert_true(_same(tok.encode("<bos><bos>"), [2, 2]))
    assert_true(_same(tok.encode("<bos>", add_bos=True), [2, 2]))
    # The other markers named in the goldens.
    assert_true(_same(tok.encode("<mask><eos><pad><unk>"), [4, 1, 0, 3]))
    assert_true(_same(tok.encode("<|channel><channel|>"), [100, 101]))


def test_gemma4_decode_special_tokens() raises:
    var tok = _real()
    # Control tokens are dropped unless asked for.
    assert_equal(tok.decode([105]), "")
    assert_equal(tok.decode([105], skip_special=False), "<|turn>")
    assert_equal(tok.decode([2, 105, 106]), "")
    # User-defined tokens always print, as in llama.cpp.
    assert_equal(tok.decode([100]), "<|channel>")
    assert_equal(tok.decode([101]), "<channel|>")
    # <|tool_response> is user-defined in the file but llama.cpp forces it to
    # control (it ends generation), so decoding drops it.
    assert_equal(tok.decode([50]), "")
    assert_equal(tok.decode([50], skip_special=False), "<|tool_response>")
    with assert_raises(contains="outside the vocabulary"):
        _ = tok.decode([262144])
    with assert_raises(contains="outside the vocabulary"):
        _ = tok.decode([-1])


def test_gemma4_vocabulary_has_no_other_name_llama_cpp_forces_to_control() raises:
    # llama.cpp turns every token whose text is one of these into a control
    # token. The three it matches in this vocabulary are handled by name;
    # a vocabulary holding any other would need the full list.
    var other = [
        "<|eot_id|>",
        "<|im_end|>",
        "<|end|>",
        "<end_of_turn>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<EOT>",
        "_<EOT>",
        "[EOT]",
        "<｜end▁of▁sentence｜>",
        "<end_of_utterance>",
        "<|eom_id|>",
        "<|fim_prefix|>",
        "<fim-prefix>",
        "<fim_prefix>",
        "<｜fim▁begin｜>",
        "<PRE>",
        "▁<PRE>",
        "<|code_prefix|>",
        "<|prefix|>",
        "<|plamo:fim_prefix|>",
        "<|fim_suffix|>",
        "<fim-suffix>",
        "<fim_suffix>",
        "<｜fim▁hole｜>",
        "<SUF>",
        "▁<SUF>",
        "<|code_suffix|>",
        "<|suffix|>",
        "<|plamo:fim_suffix|>",
        "<|fim_middle|>",
        "<fim-middle>",
        "<fim_middle>",
        "<｜fim▁end｜>",
        "<MID>",
        "▁<MID>",
        "<|code_middle|>",
        "<|middle|>",
        "<|plamo:fim_middle|>",
        "<|fim_pad|>",
        "<fim-pad>",
        "<fim_pad>",
        "<PAD>",
        "[PAD]",
        "<|fim_repo|>",
        "<|repo_name|>",
        "<fim-repo>",
        "<REPO>",
        "<reponame>",
        "<|file_sep|>",
        "<|plamo:file_separator|>",
        "<|eot|>",
        "<|return|>",
        "<|call|>",
        "<|flush|>",
        "<|calls|>",
        "[EOS]",
        "<|user|>",
        "<|observation|>",
        "[e~[",
        "<|ifm|im_end|>",
    ]
    var tok = _real()
    for name in other:
        assert_equal(tok.token_id(name), -1)
    # And </s> exists as a plain token, which llama.cpp keeps normal because
    # the vocabulary also has <|tool_response>.
    assert_equal(tok.token_id("</s>"), 212)
    assert_equal(tok.decode([212]), "</s>")


def test_gemma4_merges_by_rank_match_a_quadratic_reference() raises:
    # A brute-force merge (lowest rank, leftmost on ties, rescanning every
    # pair after each merge) must agree with the priority queue on text that
    # has no newline and no special token. The text is drawn from letters,
    # spaces, punctuation, and characters that need a multi-byte symbol.
    var tok = _real()
    var alphabet = [
        "a",
        "b",
        "e",
        "i",
        "n",
        "o",
        "r",
        "s",
        "t",
        " ",
        " ",
        ",",
        ".",
        "é",
        "中",
        "😀",
        " ",
        "1",
        "0",
        "(",
        ")",
        "_",
    ]
    var seed = 20261104
    for ci in range(150):
        seed = (seed * 1103515245 + 12345) % 2147483648
        var n = 1 + (seed // 65536) % 400
        var text = String("")
        for _ in range(n):
            seed = (seed * 1103515245 + 12345) % 2147483648
            text += alphabet[(seed // 65536) % len(alphabet)]
        var got = tok.encode(text)
        var want = _reference_encode(tok, text)
        assert_true(
            _same(got, want),
            "case "
            + String(ci)
            + " "
            + repr(text)
            + " got "
            + _show(got)
            + " want "
            + _show(want),
        )


def test_gemma4_a_long_line_is_one_word_and_round_trips() raises:
    # One line, no newline: the whole text is a single BPE word, which is
    # where a quadratic merge loop would fall over. 99,000 bytes here.
    var tok = _real()
    var line = String("")
    for i in range(2000):
        line += (
            "The quick brown fox " + String(i) + " jumps over the lazy dog. "
        )
    var ids = tok.encode(line)
    assert_true(len(ids) > 10000)
    assert_equal(tok.decode(ids), line)


def test_gemma4_end_of_generation_ids() raises:
    var tok = _real()
    assert_equal(_show(tok.eog_ids()), "[1, 106, 50]")
    # The byte-level tokenizers keep the pair they always had.
    var llada = Tokenizer(GGUFFile(FIX + "llada-tokenizer.gguf"))
    assert_equal(_show(llada.eog_ids()), "[126081, 126348]")
    assert_true(llada.is_eog(126348) and llada.is_eog(126081))
    assert_true(not llada.is_eog(126080))


def test_gemma4_no_special_token_text_contains_another() raises:
    # The partition cuts one special token at a time, longest first. With no
    # text inside another, the order among them cannot change the ids.
    var tok = _real()
    var sp = tok.g4.specials.copy()
    assert_equal(len(sp), 24)
    for a in sp:
        for b in sp:
            if a != b:
                var outer = tok.g4.tokens[a].copy()
                var inner = tok.g4.tokens[b].copy()
                assert_true(inner not in outer, inner + " is inside " + outer)


def test_gemma4_fixture_holds_the_tokens_and_the_merges_the_goldens_use() raises:
    # scripts/gen_tokenizer_fixture.py copies the tokens whole, keeps only the
    # merges that fire for the golden cases, and leaves the scores out.
    var g = GGUFFile(FIX + "gemma4-tokenizer.gguf")
    assert_equal(g.get_str("tokenizer.ggml.model"), "gemma4")
    assert_equal(g.array_len("tokenizer.ggml.tokens"), 262144)
    assert_equal(g.array_len("tokenizer.ggml.token_type"), 262144)
    assert_equal(g.array_len("tokenizer.ggml.merges"), 472)
    assert_true(not g.has_key("tokenizer.ggml.scores"))
    assert_equal(g.n_tensors(), 0)
    assert_true("<|turn>" in g.get_str("tokenizer.chat_template"))


def test_gemma4_chat_prompt_matches_the_chat_template() raises:
    # The strings are what tokenizer.chat_template renders (the first three
    # golden cases were made that way); the ids are the reference's.
    var tok = _real()
    var one = (
        "<bos><|turn>user\nWhat is paged attention?<turn|>\n<|turn>model\n"
    )
    assert_equal(
        gemma4_chat_prompt(["user"], ["What is paged attention?"]), one
    )
    var two = (
        "<bos><|turn>system\nYou are a helpful"
        " assistant.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
    )
    assert_equal(
        gemma4_chat_prompt(
            ["system", "user"], ["You are a helpful assistant.", "Hi"]
        ),
        two,
    )
    var three = (
        "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nExplain"
        " speculative decoding in two sentences.<turn|>\n<|turn>model\n"
    )
    assert_equal(
        gemma4_chat_prompt(
            ["user", "assistant", "user"],
            ["Hi", "Hello!", "Explain speculative decoding in two sentences."],
        ),
        three,
    )
    for text in [one, two, three]:
        assert_true(
            _same(tok.encode(text), _golden_ids(text)), "ids for " + repr(text)
        )


def test_gemma4_chat_prompt_follows_the_template_on_edge_cases() raises:
    # |trim strips ASCII whitespace (llama.cpp's engine uses isspace on bytes).
    assert_equal(
        gemma4_chat_prompt(["user"], ["  hi \n\t"]),
        "<bos><|turn>user\nhi<turn|>\n<|turn>model\n",
    )
    assert_equal(
        gemma4_chat_prompt(["system", "user"], [" \tsys \n", "u"], False),
        "<bos><|turn>system\nsys<turn|>\n<|turn>user\nu<turn|>\n",
    )
    # A no-break space is not stripped: isspace is false for its bytes.
    assert_equal(
        gemma4_chat_prompt(["user"], [" hi"]),
        "<bos><|turn>user\n hi<turn|>\n<|turn>model\n",
    )
    # An assistant turn drops its thinking channel, then trims.
    assert_equal(
        gemma4_chat_prompt(
            ["user", "assistant", "user"],
            ["a", "<|channel>thought\nplan<channel|> answer ", "b"],
        ),
        (
            "<bos><|turn>user\na<turn|>\n<|turn>model\nanswer<turn|>\n"
            "<|turn>user\nb<turn|>\n<|turn>model\n"
        ),
    )
    # Two assistant messages in a row share one model turn header.
    assert_equal(
        gemma4_chat_prompt(["user", "assistant", "assistant"], ["a", "x", "y"]),
        (
            "<bos><|turn>user\na<turn|>\n<|turn>model\nx<turn|>\ny<turn|>\n"
            "<|turn>model\n"
        ),
    )
    # A developer message first is the system turn; without the generation
    # prompt nothing follows the last turn.
    assert_equal(
        gemma4_chat_prompt(["developer"], ["be brief"], False),
        "<bos><|turn>system\nbe brief<turn|>\n",
    )
    with assert_raises(contains="at least one message"):
        _ = gemma4_chat_prompt(List[String](), List[String]())
    with assert_raises(contains="tool"):
        _ = gemma4_chat_prompt(["user", "tool"], ["a", "b"])


def test_gemma4_chat_prompt_trims_exactly_the_c_locale_spaces() raises:
    # The template's |trim strips the bytes C isspace accepts in the C locale:
    # space, and 9 to 13 (tab, newline, vertical tab, form feed, carriage
    # return). Every other ASCII byte stays, one step either side of the range
    # included (8 and 14).
    var spaces = [9, 10, 11, 12, 13, 32]
    for c in range(1, 128):
        var ch = chr(c)
        var kept = ch + "x" + ch
        var want = String("x")
        var is_space = False
        for s in spaces:
            if s == c:
                is_space = True
        if not is_space:
            want = kept
        assert_equal(
            gemma4_chat_prompt(["user"], [kept]),
            "<bos><|turn>user\n" + want + "<turn|>\n<|turn>model\n",
            "byte " + String(c),
        )


def test_gemma4_chat_prompt_refuses_roles_and_contents_of_different_length() raises:
    with assert_raises(contains="chat messages: 1 roles for 2 contents"):
        _ = gemma4_chat_prompt(["user"], ["a", "b"])
    with assert_raises(contains="chat messages: 2 roles for 1 contents"):
        _ = gemma4_chat_prompt(["user", "assistant"], ["a"])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
