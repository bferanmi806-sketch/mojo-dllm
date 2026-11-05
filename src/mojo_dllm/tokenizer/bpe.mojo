"""Tokenizers read from GGUF metadata: byte-level BPE (`tokenizer.ggml.model = gpt2`)
and DiffusionGemma's `gemma4` (see `gemma4.mojo`; `Tokenizer` picks one from the model key).

Byte-level BPE, which LLaDA and Dream use:

Encoding, in llama.cpp's order:

1. Partition the text on special tokens (control and user-defined entries),
   longest first. The chat template's `<|start_header_id|>` etc. become single
   ids instead of being split into punctuation.
2. Split each remaining fragment with the model's pre-tokenizer. LLaDA uses
   `bailingmoe`; its regex is reproduced by a hand-written scanner below, with
   code point classes taken from llama.cpp's own Unicode tables so that
   non-ASCII text splits exactly as llama.cpp splits it.
3. Map each byte of a piece to the GPT-2 printable code point and apply the
   ranked merges, lowest rank first.

The regex (llama.cpp `LLAMA_VOCAB_PRE_TYPE_BAILINGMOE`, ECMAScript semantics):

    '(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])
    | [^\\r\\n\\p{L}\\p{N}]?\\p{L}+
    | \\p{N}
    |  ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*
    | \\s*[\\r\\n]
    | \\s+(?!\\S)
    | \\s+
"""

from std.collections import Dict

from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.tokenizer.gemma4 import Gemma4Vocab
from mojo_dllm.tokenizer.unicode_tables import (
    LETTER_RANGES,
    NUMBER_RANGES,
    WHITESPACE_HIGH,
)

comptime C_OTHER = 0
comptime C_LETTER = 1
comptime C_NUMBER = 2
comptime C_SPACE = 3
"""Whitespace other than CR and LF."""
comptime C_NEWLINE = 4
"""CR or LF."""

comptime TOKEN_NORMAL = 1
comptime TOKEN_CONTROL = 3
comptime TOKEN_USER_DEFINED = 4
comptime TOKEN_BYTE = 6


def _in_ranges(cp: Int, table: List[Int]) -> Bool:
    var lo = 0
    var hi = len(table) // 2 - 1
    while lo <= hi:
        var mid = (lo + hi) // 2
        if cp < table[2 * mid]:
            hi = mid - 1
        elif cp > table[2 * mid + 1]:
            lo = mid + 1
        else:
            return True
    return False


struct UnicodeClasses(Movable):
    """llama.cpp's code point classes, materialized once."""

    var letters: List[Int]
    var numbers: List[Int]
    var spaces: List[Int]

    def __init__(out self):
        self.letters = materialize[LETTER_RANGES]()
        self.numbers = materialize[NUMBER_RANGES]()
        self.spaces = materialize[WHITESPACE_HIGH]()

    def classify(self, cp: Int) -> Int:
        if cp < 128:
            if cp == 10 or cp == 13:
                return C_NEWLINE
            if cp == 32 or cp == 9 or cp == 11 or cp == 12:
                return C_SPACE
            if (cp >= 65 and cp <= 90) or (cp >= 97 and cp <= 122):
                return C_LETTER
            if cp >= 48 and cp <= 57:
                return C_NUMBER
            return C_OTHER
        for w in self.spaces:
            if w == cp:
                return C_SPACE
        if _in_ranges(cp, self.letters):
            return C_LETTER
        if _in_ranges(cp, self.numbers):
            return C_NUMBER
        return C_OTHER


@always_inline
def _is_ws(c: Int) -> Bool:
    return c == C_SPACE or c == C_NEWLINE


@always_inline
def _lower(cp: Int) -> Int:
    if cp >= 65 and cp <= 90:
        return cp + 32
    return cp


def utf8_decode(s: String) -> List[Int]:
    """Code points of `s`. Invalid bytes become U+FFFD."""
    var out = List[Int]()
    var b = s.as_bytes()
    var i = 0
    var n = len(b)
    while i < n:
        var c = Int(b[i])
        if c < 0x80:
            out.append(c)
            i += 1
        elif c >= 0xC0 and c < 0xE0 and i + 1 < n:
            out.append(((c & 0x1F) << 6) | (Int(b[i + 1]) & 0x3F))
            i += 2
        elif c >= 0xE0 and c < 0xF0 and i + 2 < n:
            out.append(
                ((c & 0x0F) << 12)
                | ((Int(b[i + 1]) & 0x3F) << 6)
                | (Int(b[i + 2]) & 0x3F)
            )
            i += 3
        elif c >= 0xF0 and i + 3 < n:
            out.append(
                ((c & 0x07) << 18)
                | ((Int(b[i + 1]) & 0x3F) << 12)
                | ((Int(b[i + 2]) & 0x3F) << 6)
                | (Int(b[i + 3]) & 0x3F)
            )
            i += 4
        else:
            out.append(0xFFFD)
            i += 1
    return out^


def utf8_append(mut out: List[UInt8], cp: Int):
    if cp < 0x80:
        out.append(UInt8(cp))
    elif cp < 0x800:
        out.append(UInt8(0xC0 | (cp >> 6)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    elif cp < 0x10000:
        out.append(UInt8(0xE0 | (cp >> 12)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))
    else:
        out.append(UInt8(0xF0 | (cp >> 18)))
        out.append(UInt8(0x80 | ((cp >> 12) & 0x3F)))
        out.append(UInt8(0x80 | ((cp >> 6) & 0x3F)))
        out.append(UInt8(0x80 | (cp & 0x3F)))


def bytes_to_string(b: List[UInt8]) -> String:
    return String(StringSpan(unsafe_from_utf8=Span(b)))


def pretokenize(cps: List[Int], uc: UnicodeClasses) -> List[Tuple[Int, Int]]:
    """Split code points into pieces, returned as [start, end) index pairs."""
    var n = len(cps)
    var cls = List[Int](capacity=n)
    for c in cps:
        cls.append(uc.classify(c))
    var out = List[Tuple[Int, Int]]()
    var i = 0
    while i < n:
        var e = _match_at(cps, cls, i)
        out.append((i, e))
        i = e
    return out^


def _match_at(cps: List[Int], cls: List[Int], i: Int) -> Int:
    var n = len(cps)
    var c = cps[i]
    # 1. contractions
    if c == 39 and i + 1 < n:
        var d = _lower(cps[i + 1])
        if d == 115 or d == 100 or d == 109 or d == 116:  # s d m t
            return i + 2
        if i + 2 < n:
            var d2 = _lower(cps[i + 2])
            if (
                (d == 108 and d2 == 108)
                or (d == 118 and d2 == 101)
                or (d == 114 and d2 == 101)
            ):
                return i + 3
    # 2. [^\r\n\p{L}\p{N}]?\p{L}+
    var k = cls[i]
    if (
        k != C_LETTER
        and k != C_NUMBER
        and k != C_NEWLINE
        and i + 1 < n
        and cls[i + 1] == C_LETTER
    ):
        var j = i + 1
        while j < n and cls[j] == C_LETTER:
            j += 1
        return j
    if k == C_LETTER:
        var j = i
        while j < n and cls[j] == C_LETTER:
            j += 1
        return j
    # 3. \p{N}
    if k == C_NUMBER:
        return i + 1
    # 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
    var j = i
    if c == 32 and j + 1 < n and cls[j + 1] == C_OTHER:
        j += 1
    if cls[j] == C_OTHER:
        while j < n and cls[j] == C_OTHER:
            j += 1
        while j < n and cls[j] == C_NEWLINE:
            j += 1
        return j
    # 5-7 need whitespace at i
    if _is_ws(k):
        var e = i
        while e < n and _is_ws(cls[e]):
            e += 1
        # 5. \s*[\r\n]: the last CR/LF inside the run ends the match
        var last_nl = -1
        for q in range(i, e):
            if cls[q] == C_NEWLINE:
                last_nl = q
        if last_nl >= 0:
            return last_nl + 1
        # 6. \s+(?!\S)
        if e == n:
            return e
        if e - i >= 2:
            return e - 1
        # 7. \s+
        return e
    return i + 1


struct Tokenizer(Movable):
    """The tokenizer of a GGUF file: `gpt2` (byte-level BPE) or `gemma4`.

    The fields below `g4` belong to the byte-level BPE and stay empty for a
    gemma4 file, whose vocabulary lives in `g4`. The id fields are filled for both.
    """

    var tokens: List[String]
    var types: List[Int]
    var ids: Dict[String, Int]
    var ranks: Dict[String, Int]
    var specials: List[Int]
    """Ids of control/user-defined tokens, longest text first."""
    var byte_to_cp: List[Int]
    var cp_to_byte: Dict[Int, Int]
    var bos_id: Int
    var eos_id: Int
    var eot_id: Int
    var mask_id: Int
    var pad_id: Int
    var unk_id: Int
    var add_bos: Bool
    """Whether to add the BOS token when encoding a prompt that has none of its own.
    Always true for gemma4: llama.cpp forces it, whatever the file says."""
    var cache: Dict[String, List[Int]]
    var uc: UnicodeClasses
    var is_gemma4: Bool
    var g4: Gemma4Vocab

    def __init__(out self, g: GGUFFile) raises:
        var model = g.get_str("tokenizer.ggml.model")
        self.is_gemma4 = model == "gemma4"
        self.g4 = Gemma4Vocab()
        self.tokens = List[String]()
        self.types = List[Int]()
        self.ids = Dict[String, Int]()
        self.ranks = Dict[String, Int]()
        self.specials = List[Int]()
        self.byte_to_cp = List[Int]()
        self.cp_to_byte = Dict[Int, Int]()
        self.cache = Dict[String, List[Int]]()
        self.uc = UnicodeClasses()
        self.pad_id = -1
        self.unk_id = -1
        if self.is_gemma4:
            self.g4 = Gemma4Vocab(g)
            self.bos_id = self.g4.bos_id
            self.eos_id = self.g4.eos_id
            self.eot_id = self.g4.eot_id
            self.mask_id = self.g4.mask_id
            self.pad_id = self.g4.pad_id
            self.unk_id = self.g4.unk_id
            self.add_bos = self.g4.add_bos
            return
        if model != "gpt2":
            raise Error(
                "unsupported tokenizer model: "
                + model
                + " (supported: gpt2, gemma4)"
            )
        # LLaDA ships `bailingmoe`, Dream (Qwen2) ships `qwen2`. llama.cpp's
        # regexes for the two split text identically: the same contraction
        # set, single-digit numbers, and `\s*[\r\n]+` ending at the last
        # newline exactly as `\s*[\r\n]` does. Golden ids check both.
        var pre = g.get_str_or("tokenizer.ggml.pre", "default")
        if pre != "bailingmoe" and pre != "qwen2":
            raise Error("unsupported pre-tokenizer: " + pre)
        self.tokens = g.get_str_array("tokenizer.ggml.tokens")
        self.types = g.get_int_array("tokenizer.ggml.token_type")
        var merges = g.get_str_array("tokenizer.ggml.merges")
        self.ids = Dict[String, Int]()
        for i in range(len(self.tokens)):
            self.ids[self.tokens[i]] = i
        self.ranks = Dict[String, Int]()
        for i in range(len(merges)):
            self.ranks[merges[i]] = i
        self.specials = List[Int]()
        for i in range(len(self.types)):
            if (
                self.types[i] == TOKEN_CONTROL
                or self.types[i] == TOKEN_USER_DEFINED
            ):
                if self.tokens[i].byte_length() > 0:
                    self.specials.append(i)
        # Longest first, so "<|eot_id|>" never loses to a shorter prefix.
        for a in range(1, len(self.specials)):
            var v = self.specials[a]
            var b = a - 1
            while (
                b >= 0
                and self.tokens[self.specials[b]].byte_length()
                < self.tokens[v].byte_length()
            ):
                self.specials[b + 1] = self.specials[b]
                b -= 1
            self.specials[b + 1] = v
        self.byte_to_cp = List[Int](length=256, fill=0)
        self.cp_to_byte = Dict[Int, Int]()
        var extra = 0
        for b in range(256):
            var printable = (
                (b >= 33 and b <= 126)
                or (b >= 161 and b <= 172)
                or (b >= 174 and b <= 255)
            )
            var cp = b if printable else 256 + extra
            if not printable:
                extra += 1
            self.byte_to_cp[b] = cp
            self.cp_to_byte[cp] = b
        self.bos_id = g.get_int_or("tokenizer.ggml.bos_token_id", -1)
        self.eos_id = g.get_int_or("tokenizer.ggml.eos_token_id", -1)
        self.mask_id = g.get_int_or("tokenizer.ggml.mask_token_id", -1)
        self.pad_id = g.get_int_or("tokenizer.ggml.padding_token_id", -1)
        self.unk_id = g.get_int_or("tokenizer.ggml.unknown_token_id", -1)
        self.eot_id = self.ids.get("<|eot_id|>", self.ids.get("<|im_end|>", -1))
        self.add_bos = g.get_bool("tokenizer.ggml.add_bos_token") if g.has_key(
            "tokenizer.ggml.add_bos_token"
        ) else False
        self.cache = Dict[String, List[Int]]()
        self.uc = UnicodeClasses()

    def vocab_size(self) -> Int:
        if self.is_gemma4:
            return self.g4.vocab_size()
        return len(self.tokens)

    def token_id(self, text: String) -> Int:
        """The id of the token with exactly this text, or -1."""
        if self.is_gemma4:
            return self.g4.token_id(text)
        return self.ids.get(text, -1)

    def pair_rank(self, left: String, right: String) -> Int:
        """The rank of the merge of two symbols (gemma4 only), or -1."""
        return self.g4.pair_rank(left, right)

    def is_eog(self, id: Int) -> Bool:
        """Whether generation ends at this token: EOS and the end-of-turn token.
        """
        if self.is_gemma4:
            return self.g4.is_eog(id)
        return id >= 0 and (id == self.eos_id or id == self.eot_id)

    def eog_ids(self) -> List[Int]:
        """The ids that end generation, in the order `is_eog` tests them."""
        if self.is_gemma4:
            return self.g4.eog.copy()
        var out = List[Int]()
        if self.eos_id >= 0:
            out.append(self.eos_id)
        if self.eot_id >= 0:
            out.append(self.eot_id)
        return out^

    def encode(
        mut self, text: String, add_bos: Bool = False
    ) raises -> List[Int]:
        """Token ids of `text`, special-token text parsed into its ids.

        `add_bos` adds what llama.cpp adds for a prompt: the BOS token, and
        for a gemma4 file whose `add_eos_token` is true the EOS token too.
        """
        if self.is_gemma4:
            return self.g4.encode(text, add_bos)
        var out = List[Int]()
        if add_bos and self.bos_id >= 0:
            out.append(self.bos_id)
        var b = text.as_bytes()
        var n = len(b)
        var start = 0
        var i = 0
        while i < n:
            var hit = -1
            for sid in self.specials:
                var st = self.tokens[sid].as_bytes()
                var m = len(st)
                if m <= n - i and st[0] == b[i]:
                    var ok = True
                    for q in range(1, m):
                        if st[q] != b[i + q]:
                            ok = False
                            break
                    if ok:
                        hit = sid
                        break
            if hit >= 0:
                if i > start:
                    self._encode_fragment(
                        String(StringSpan(unsafe_from_utf8=b[start:i])), out
                    )
                out.append(hit)
                i += self.tokens[hit].byte_length()
                start = i
            else:
                i += 1
        if start < n:
            self._encode_fragment(
                String(StringSpan(unsafe_from_utf8=b[start:n])), out
            )
        return out^

    def _encode_fragment(mut self, text: String, mut out: List[Int]) raises:
        var cps = utf8_decode(text)
        var pieces = pretokenize(cps, self.uc)
        for p in pieces:
            var buf = List[UInt8]()
            for q in range(p[0], p[1]):
                utf8_append(buf, cps[q])
            var word = bytes_to_string(buf)
            if word in self.cache:
                for t in self.cache[word]:
                    out.append(t)
                continue
            var ids = self._bpe(buf)
            for t in ids:
                out.append(t)
            self.cache[word] = ids^

    def _bpe(self, raw: List[UInt8]) raises -> List[Int]:
        var syms = List[String]()
        for byte in raw:
            var cpb = List[UInt8]()
            utf8_append(cpb, self.byte_to_cp[Int(byte)])
            syms.append(bytes_to_string(cpb))
        while len(syms) > 1:
            var best = -1
            var best_rank = Int.MAX
            for q in range(len(syms) - 1):
                var key = syms[q] + " " + syms[q + 1]
                var r = self.ranks.get(key, -1)
                if r >= 0 and r < best_rank:
                    best_rank = r
                    best = q
            if best < 0:
                break
            var left = syms[best].copy()
            var right = syms[best + 1].copy()
            syms[best] = left + right
            _ = syms.pop(best + 1)
        var ids = List[Int]()
        for s in syms:
            if s in self.ids:
                ids.append(self.ids[s])
            else:
                # Not in the vocabulary as a whole: fall back to its bytes.
                for cp in utf8_decode(s):
                    var cb = List[UInt8]()
                    utf8_append(cb, cp)
                    var key = bytes_to_string(cb)
                    if key not in self.ids:
                        raise Error("byte symbol missing from vocabulary")
                    ids.append(self.ids[key])
        return ids^

    def decode(
        self, ids: List[Int], skip_special: Bool = True
    ) raises -> String:
        """Text of `ids`. Special tokens are dropped when `skip_special`; which
        tokens count as special differs by tokenizer (see `Gemma4Vocab.decode`).
        """
        if self.is_gemma4:
            return self.g4.decode(ids, skip_special)
        var out = List[UInt8]()
        for id in ids:
            if id < 0 or id >= len(self.tokens):
                raise Error(
                    "token id " + String(id) + " is outside the vocabulary"
                )
            var t = self.types[id]
            if t == TOKEN_CONTROL or t == TOKEN_USER_DEFINED:
                if not skip_special:
                    for byte in self.tokens[id].as_bytes():
                        out.append(byte)
                continue
            for cp in utf8_decode(self.tokens[id]):
                if cp in self.cp_to_byte:
                    out.append(UInt8(self.cp_to_byte[cp]))
                else:
                    utf8_append(out, cp)
        # Generation can stop inside a multi-byte character; keep the text valid.
        return String(from_utf8_lossy=Span(out))
