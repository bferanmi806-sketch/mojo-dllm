"""The gemma4 tokenizer rules on a vocabulary small enough to work out by hand.

`scripts/gen_fixtures.py` writes `tests/fixtures/gemma4-tiny/`: 279 tokens (the
five control tokens, `<|turn>`, `<turn|>`, `<|channel>` and `<|tool_response>`,
the 256 byte tokens `<0xNN>` at id 9 + NN, and a handful of text tokens) and
eight merges chosen so that each rule changes the ids when it is broken. The
real vocabulary's goldens are in `tests/test_tokenizer_gemma4.mojo`.

Ids: 265 `▁`, 266 a, 267 b, 268 c, 269 ab, 270 bc, 271 abc, 272 `▁a`, 273 `\n`,
274 `\n\n`, 275 `\n\n\n`, 276 `a\n`, 277 `▁▁`, 278 é.
Merges by rank: 0 `a b`, 1 `b c`, 2 `ab c`, 3 `▁ a`, 4 `\n \n`, 5 `a \n`,
6 `▁ ▁`, 7 `a b` again.
"""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.tokenizer.bpe import Tokenizer

comptime DIR = "tests/fixtures/gemma4-tiny/"
comptime BYTE0 = 9


def _show(ids: List[Int]) -> String:
    var s = String("[")
    for i in range(len(ids)):
        s += (", " if i else "") + String(ids[i])
    return s + "]"


def _check(mut tok: Tokenizer, text: String, want: List[Int]) raises:
    var got = tok.encode(text)
    var same = len(got) == len(want)
    if same:
        for i in range(len(got)):
            if got[i] != want[i]:
                same = False
    assert_true(
        same,
        "encode(" + repr(text) + ") = " + _show(got) + ", want " + _show(want),
    )


def _tok(name: String) raises -> Tokenizer:
    return Tokenizer(GGUFFile(DIR + name))


def test_tiny_bos_is_added_although_the_file_says_false() raises:
    var tok = _tok("base.gguf")
    assert_true(tok.add_bos)
    _check_with_bos(tok, "a", [2, 266])
    _check(tok, "a", [266])


def _check_with_bos(mut tok: Tokenizer, text: String, want: List[Int]) raises:
    var got = tok.encode(text, add_bos=tok.add_bos)
    assert_equal(_show(got), _show(want))


def test_tiny_eos_is_added_only_when_the_file_asks() raises:
    var tok = _tok("eos.gguf")
    _check_with_bos(tok, "a", [2, 266, 1])
    _check(tok, "a", [266])
    var base = _tok("base.gguf")
    _check_with_bos(base, "a", [2, 266])


def test_tiny_merges_apply_lowest_rank_first() raises:
    var tok = _tok("base.gguf")
    # a b (rank 0) and b c (rank 1) compete: ab first, then ab c (rank 2).
    _check(tok, "abc", [271])
    _check(tok, "ab", [269])
    _check(tok, "bc", [270])
    _check(tok, "abab", [269, 269])


def test_tiny_equal_ranks_merge_leftmost_first() raises:
    var tok = _tok("base.gguf")
    # ▁ ▁ at positions 0-1 and 1-2 share rank 6; the left one goes first.
    _check(tok, "   ", [277, 265])


def test_tiny_the_first_of_two_equal_merges_keeps_its_rank() raises:
    var tok = _tok("base.gguf")
    assert_equal(tok.pair_rank("a", "b"), 0)
    assert_equal(tok.pair_rank("b", "c"), 1)
    assert_equal(tok.pair_rank("a", "c"), -1)
    # With "a b" at rank 7 instead, b c would go first and abc would not form.
    _check(tok, "abc", [271])


def test_tiny_spaces_become_lower_blocks() raises:
    var tok = _tok("base.gguf")
    _check(tok, "a b", [266, 265, 267])
    # No space is added in front: "a" is "a", not "▁a".
    _check(tok, "a", [266])
    _check(tok, " a", [272])
    # ▁ ▁ (rank 6) loses to ▁ a (rank 3): the space run splits before the a.
    _check(tok, "  a", [265, 272])


def test_tiny_a_newline_run_that_is_a_token_stays_whole() raises:
    var tok = _tok("base.gguf")
    # \n\n\n is a token. Merging would give \n\n then \n (leftmost first).
    _check(tok, "\n\n\n", [275])
    _check(tok, "\n\n", [274])
    _check(tok, "\n", [273])


def test_tiny_a_newline_run_that_is_not_a_token_is_merged() raises:
    var tok = _tok("base.gguf")
    # Four newlines are no token: \n\n and \n\n, leftmost merge first.
    _check(tok, "\n\n\n\n", [274, 274])


def test_tiny_no_merge_crosses_a_newline() raises:
    var tok = _tok("base.gguf")
    # "a \n" (rank 5) exists, but the pre-split puts a and \n in separate words.
    _check(tok, "a\n", [266, 273])
    _check(tok, "a\nb", [266, 273, 267])
    # A carriage return is not a newline for the split: it travels with its
    # word, and here it has no token, so it is a byte.
    _check(tok, "\r\n", [BYTE0 + 13, 273])


def test_tiny_characters_without_a_token_become_byte_tokens() raises:
    var tok = _tok("base.gguf")
    _check(tok, "x", [BYTE0 + 0x78])
    # ü is C3 BC, and has no token; é has one.
    _check(tok, "ü", [BYTE0 + 0xC3, BYTE0 + 0xBC])
    _check(tok, "é", [278])
    _check(tok, "éx", [278, BYTE0 + 0x78])
    assert_equal(tok.decode([BYTE0 + 0xC3, BYTE0 + 0xBC]), "ü")


def test_tiny_special_tokens_in_the_text_become_their_ids() raises:
    var tok = _tok("base.gguf")
    _check(tok, "<|turn>a<turn|>", [5, 266, 6])
    _check(tok, "<bos>", [2])
    _check(tok, "<|channel>", [7])
    _check(tok, "a<mask>b", [266, 4, 267])
    # Not a marker: it is cut into characters and merged like any text.
    assert_true(len(tok.encode("<|turn")) > 1)


def test_tiny_decode_follows_the_token_attributes() raises:
    var tok = _tok("base.gguf")
    assert_equal(tok.decode([5, 266, 6]), "a")
    assert_equal(tok.decode([5, 266, 6], skip_special=False), "<|turn>a<turn|>")
    # User-defined <|channel> prints; <|tool_response> was made control.
    assert_equal(tok.decode([7, 266]), "<|channel>a")
    assert_equal(tok.decode([8, 266]), "a")
    assert_equal(tok.decode([8, 266], skip_special=False), "<|tool_response>a")
    assert_equal(tok.decode([265, 266, 277]), " a  ")
    assert_equal(tok.eot_id, 6)
    assert_true(tok.is_eog(1) and tok.is_eog(6) and tok.is_eog(8))
    assert_true(not tok.is_eog(7))


def test_tiny_a_missing_byte_token_is_named() raises:
    with assert_raises(contains="no byte token <0x78>"):
        _ = _tok("no-byte-token.gguf")


def test_tiny_a_byte_token_with_the_wrong_text_is_named() raises:
    with assert_raises(contains="token 129 has type byte"):
        _ = _tok("bad-byte-text.gguf")


def test_tiny_a_merge_without_a_space_is_named() raises:
    with assert_raises(contains="tokenizer.ggml.merges[3]"):
        _ = _tok("merge-without-space.gguf")


def test_tiny_token_and_type_counts_must_agree() raises:
    with assert_raises(
        contains="tokenizer.ggml.token_type has 278 entries but"
    ):
        _ = _tok("short-types.gguf")


def test_tiny_a_missing_bos_id_is_named() raises:
    with assert_raises(contains="tokenizer.ggml.bos_token_id is required"):
        _ = _tok("no-bos-id.gguf")


def test_tiny_a_special_id_outside_the_vocabulary_is_named() raises:
    with assert_raises(
        contains=(
            "tokenizer.ggml.bos_token_id is 293, outside the vocabulary of"
            " 279 tokens"
        )
    ):
        _ = _tok("bos-out-of-range.gguf")


def test_tiny_add_eos_without_an_eos_id_is_named() raises:
    with assert_raises(contains="add_eos_token is true but"):
        _ = _tok("eos-flag-without-id.gguf")


def test_tiny_only_the_never_prepend_scheme_is_supported() raises:
    with assert_raises(contains="prepend_scheme is 'first'"):
        _ = _tok("metaspace-first.gguf")
    var tok = _tok("metaspace-never.gguf")
    _check(tok, "a", [266])


def test_tiny_two_tokens_with_one_text_are_named() raises:
    with assert_raises(contains="'a' at ids 266 and 267"):
        _ = _tok("duplicate-token.gguf")


def test_tiny_an_empty_token_gets_llama_cpps_placeholder_name() raises:
    var tok = _tok("empty-token.gguf")
    assert_equal(tok.token_id("[EMPTY_278]"), 278)
    assert_equal(tok.decode([278]), "[EMPTY_278]")


def _from_bytes(*bytes: Int) -> String:
    var raw = List[UInt8]()
    for b in bytes:
        raw.append(UInt8(b))
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def test_tiny_text_that_is_not_utf8_falls_back_to_bytes() raises:
    # llama.cpp throws on such text; here every stray byte is a byte token,
    # and a lead byte with too few bytes after it cannot read past the end.
    var tok = _tok("base.gguf")
    _check(tok, _from_bytes(0x61, 0xE2), [266, BYTE0 + 0xE2])
    _check(tok, _from_bytes(0xE2, 0x96), [BYTE0 + 0xE2, BYTE0 + 0x96])
    # A lead byte takes the bytes its length announces as one symbol, valid or
    # not, so the a after 0xFF is part of it and neither is a token.
    _check(tok, _from_bytes(0xFF, 0x61), [BYTE0 + 0xFF, BYTE0 + 0x61])


def test_tiny_end_of_generation_ids() raises:
    var tok = _tok("base.gguf")
    var eog = tok.eog_ids()
    assert_equal(_show(eog), "[1, 6, 8]")


def test_tiny_the_first_and_last_byte_tokens_are_required() raises:
    # The byte loop covers 0 to 255 whole: a vocabulary without <0x00> or
    # <0xFF> is refused as one without <0x78> is.
    with assert_raises(contains="no byte token <0x00>"):
        _ = _tok("no-byte-00.gguf")
    with assert_raises(contains="no byte token <0xFF>"):
        _ = _tok("no-byte-ff.gguf")


def test_tiny_a_merge_whose_only_space_is_its_first_byte_is_named() raises:
    # llama.cpp splits a merge at the first space after its first byte, so " ab"
    # has none, although it has a space.
    with assert_raises(
        contains="tokenizer.ggml.merges[3] = ' ab' has no space"
    ):
        _ = _tok("merge-space-first.gguf")


def test_tiny_a_special_id_equal_to_the_vocabulary_size_is_named() raises:
    # 279 tokens have ids 0 to 278; bos-out-of-range holds 293, this one 279.
    with assert_raises(
        contains=(
            "tokenizer.ggml.bos_token_id is 279, outside the vocabulary of 279"
            " tokens"
        )
    ):
        _ = _tok("bos-at-vocab-size.gguf")


def test_tiny_the_longest_special_token_is_cut_first() raises:
    # Id 279 is the user-defined text "<|turn><turn|>", which holds the ids 5 and
    # 6. Cut longest first it is one token; cut shortest first, or in id order,
    # the two inside it are found first and it never matches.
    var tok = _tok("nested-special.gguf")
    assert_equal(tok.vocab_size(), 280)
    _check(tok, "<|turn><turn|>", [279])
    _check(tok, "a<|turn><turn|>b", [266, 279, 267])
    # Where the long text is not there, the short ones are cut as before.
    _check(tok, "<|turn>a<turn|>", [5, 266, 6])
    _check(tok, "<turn|><|turn>", [6, 5])


def test_tiny_an_unknown_token_is_cut_from_the_text() raises:
    # <unk> has token type 2 here, and llama.cpp cuts control, user-defined and
    # unknown tokens out of the text alike.
    var tok = _tok("unknown-type.gguf")
    assert_equal(tok.unk_id, 3)
    _check(tok, "<unk>", [3])
    _check(tok, "a<unk>b", [266, 3, 267])


def test_tiny_an_unknown_token_is_dropped_when_special_tokens_are_skipped() raises:
    # token_to_piece drops unknown and control tokens unless asked to print them.
    var tok = _tok("unknown-type.gguf")
    assert_equal(tok.decode([3]), "")
    assert_equal(tok.decode([266, 3, 267]), "ab")
    assert_equal(tok.decode([3], skip_special=False), "<unk>")
    assert_equal(tok.decode([266, 3, 267], skip_special=False), "a<unk>b")


def test_tiny_eos_is_a_control_token_whatever_the_file_says() raises:
    # The file types <eos> as a normal token. llama.cpp overrides it to control
    # (it ends generation): it is cut from the text and not printed.
    var tok = _tok("forced-control.gguf")
    _check(tok, "<eos>", [1])
    assert_equal(tok.decode([1]), "")
    assert_equal(tok.decode([1], skip_special=False), "<eos>")


def test_tiny_end_of_turn_is_a_control_token_whatever_the_file_says() raises:
    # The same for <turn|>, which the file also types as normal.
    var tok = _tok("forced-control.gguf")
    _check(tok, "<turn|>", [6])
    assert_equal(tok.decode([6]), "")
    assert_equal(tok.decode([6], skip_special=False), "<turn|>")


def test_tiny_end_of_generation_ids_are_listed_once() raises:
    # EOS is <turn|> itself here, so the list is [6, 8], not [6, 6, 8].
    var tok = _tok("eos-is-end-of-turn.gguf")
    assert_equal(tok.eos_id, 6)
    assert_equal(tok.eot_id, 6)
    assert_equal(_show(tok.eog_ids()), "[6, 8]")
    assert_true(tok.is_eog(6) and tok.is_eog(8))
    assert_true(not tok.is_eog(1))


def test_tiny_end_of_generation_skips_tokens_the_vocabulary_lacks() raises:
    # Neither <turn|> nor <|tool_response> is in this vocabulary: only EOS ends
    # generation, and -1 (no token) does not.
    var tok = _tok("no-end-of-turn.gguf")
    assert_equal(tok.eot_id, -1)
    assert_equal(_show(tok.eog_ids()), "[1]")
    assert_true(tok.is_eog(1))
    assert_true(not tok.is_eog(-1))
    assert_true(not tok.is_eog(6) and not tok.is_eog(8))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
