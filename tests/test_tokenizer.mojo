from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from cli.args import parse_csv_ints
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.tokenizer.bpe import (
    Tokenizer,
    UnicodeClasses,
    pretokenize,
    utf8_decode,
    C_LETTER,
    C_NUMBER,
    C_SPACE,
    C_OTHER,
)

comptime FIX = "tests/fixtures/"


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


def _cases(prefix: String) raises -> List[Tuple[String, List[Int]]]:
    var out = List[Tuple[String, List[Int]]]()
    with open(FIX + prefix + "-tokenizer-golden.tsv", "r") as f:
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


def _check_golden(prefix: String) raises:
    var g = GGUFFile(FIX + prefix + "-tokenizer.gguf")
    var tok = Tokenizer(g)
    var cases = _cases(prefix)
    assert_true(
        len(cases) >= 30,
        "golden corpus is unexpectedly small: " + String(len(cases)),
    )
    var failures = 0
    for c in cases:
        var got = tok.encode(c[0])
        var want = c[1].copy()
        var same = len(got) == len(want)
        if same:
            for i in range(len(got)):
                if got[i] != want[i]:
                    same = False
        if not same:
            failures += 1
            var gs = String("")
            for t in got:
                gs += String(t) + ","
            print("MISMATCH", prefix, repr(c[0]), "got", gs)
        assert_equal(tok.decode(got, skip_special=False), c[0])
    assert_equal(failures, 0)


def test_llada_golden_ids_match_llama_cpp() raises:
    _check_golden("llada")


def test_dream_golden_ids_match_llama_cpp() raises:
    _check_golden("dream")


def test_dream_special_ids() raises:
    var tok = Tokenizer(GGUFFile(FIX + "dream-tokenizer.gguf"))
    assert_equal(tok.eos_id, 151643)
    assert_equal(tok.mask_id, 151666)
    assert_equal(tok.eot_id, tok.encode("<|im_end|>")[0])


def test_decode_skips_special_tokens_by_default() raises:
    var g = GGUFFile(FIX + "llada-tokenizer.gguf")
    var tok = Tokenizer(g)
    var ids = tok.encode("<|eot_id|>Hi")
    assert_equal(ids[0], 126348)
    assert_equal(tok.decode(ids), "Hi")


def test_special_ids() raises:
    var g = GGUFFile(FIX + "llada-tokenizer.gguf")
    var tok = Tokenizer(g)
    assert_equal(tok.bos_id, 126080)
    assert_equal(tok.eos_id, 126081)
    assert_equal(tok.eot_id, 126348)
    assert_equal(tok.mask_id, 126336)
    assert_equal(tok.vocab_size(), 126464)


def test_classify() raises:
    var uc = UnicodeClasses()
    assert_equal(uc.classify(65), C_LETTER)
    assert_equal(uc.classify(0x4E2D), C_LETTER)  # 中
    assert_equal(uc.classify(0x0661), C_NUMBER)  # Arabic-Indic one
    assert_equal(uc.classify(0x00A0), C_SPACE)
    assert_equal(
        uc.classify(0x0301), C_OTHER
    )  # combining acute: not a letter in llama.cpp's tables


def test_pretokenize_newline_runs() raises:
    # "\s*[\r\n]" ends at the last newline of a whitespace run.
    var cps = utf8_decode("x  \n  y")
    var p = pretokenize(cps, UnicodeClasses())
    assert_equal(len(p), 4)
    assert_equal(p[1][0], 1)
    assert_equal(p[1][1], 4)


def test_rejects_other_tokenizers() raises:
    var g = GGUFFile(FIX + "tiny-llada.gguf")
    with assert_raises(contains="tokenizer.ggml.model"):
        _ = Tokenizer(g)


def test_byte_level_pad_and_unknown_ids_come_from_the_file() raises:
    var llada = Tokenizer(GGUFFile(FIX + "llada-tokenizer.gguf"))
    assert_equal(llada.pad_id, 126081)
    assert_equal(llada.unk_id, -1)  # the file has no unknown_token_id
    var dream = Tokenizer(GGUFFile(FIX + "dream-tokenizer.gguf"))
    assert_equal(dream.pad_id, 151643)
    assert_equal(dream.unk_id, -1)
    # gpt2-tiny/ids.gguf names both: padding 2, unknown 3.
    var tiny = Tokenizer(GGUFFile(FIX + "gpt2-tiny/ids.gguf"))
    assert_equal(tiny.pad_id, 2)
    assert_equal(tiny.unk_id, 3)


def test_byte_level_end_of_generation_ignores_a_token_the_file_lacks() raises:
    # gpt2-tiny/ids.gguf has EOS (id 4) and neither <|eot_id|> nor <|im_end|>.
    var tiny = Tokenizer(GGUFFile(FIX + "gpt2-tiny/ids.gguf"))
    assert_equal(tiny.eos_id, 4)
    assert_equal(tiny.eot_id, -1)
    var eog = tiny.eog_ids()
    assert_equal(len(eog), 1)
    assert_equal(eog[0], 4)
    assert_true(tiny.is_eog(4))
    assert_true(not tiny.is_eog(-1))
    assert_true(not tiny.is_eog(2))


def _old_stop_rule(tok: Tokenizer, id: Int) -> Bool:
    """What `main.mojo` tested before gemma4: the id is EOS or end-of-turn."""
    return id == tok.eos_id or id == tok.eot_id


def test_byte_level_end_of_generation_is_the_rule_llada_and_dream_always_had() raises:
    for name in ["llada", "dream"]:
        var tok = Tokenizer(GGUFFile(FIX + name + "-tokenizer.gguf"))
        var old = List[Int]()
        for s in [tok.eos_id, tok.eot_id]:
            if s >= 0:
                old.append(s)
        var eog = tok.eog_ids()
        assert_equal(len(eog), len(old))
        for i in range(len(old)):
            assert_equal(eog[i], old[i])
        for id in [-1, 0, tok.bos_id, tok.eos_id, tok.eot_id, tok.mask_id]:
            assert_equal(tok.is_eog(id), _old_stop_rule(tok, id))
        assert_true(tok.is_eog(tok.eos_id) and tok.is_eog(tok.eot_id))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
