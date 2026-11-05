"""The `gemma4` tokenizer of DiffusionGemma (`tokenizer.ggml.model = gemma4`).

It is a different algorithm from the byte-level BPE in `bpe.mojo`. The rules
below are read from llama.cpp's `src/llama-vocab.cpp` at the commit of
ggml-org/llama.cpp#24423 that `scripts/build_dg_ref.sh` pins
(`LLAMA_VOCAB_PRE_TYPE_GEMMA4`, with `llm_tokenizer_bpe_session::tokenize`):

1. Special tokens. Every token whose type is control, user-defined or unknown
   is cut out of the text, one token at a time, longest text first. The ids of
   control tokens in the text become tokens, so `<|turn>` is one id.
2. Normalizer. Each remaining fragment has every space (0x20) replaced by
   U+2581. Tabs and newlines stay. No space is added at the start
   (`add_space_prefix` is not read by the BPE path).
3. Pre-split. The only split is `[^\\n]+|[\\n]+`: runs of newlines, and runs of
   everything else. A run of newlines that is itself a token stays whole
   (ggml-org/llama.cpp#21343); otherwise it is merged like any other word.
4. Merges. The word is cut into UTF-8 characters (the length comes from the
   lead byte, as `unicode_len_utf8` does it) and merged by rank, lowest rank
   first and leftmost on a tie, with a priority queue: a long line of text is
   one word, so a rescan of every pair after each merge would be quadratic.
5. A symbol that is not a token is written as its bytes, each the token
   `<0xNN>` (uppercase hex, ids 238 to 493, token type 6).
6. BOS is added at the start whatever `tokenizer.ggml.add_bos_token` says
   (llama.cpp forces it for Gemma 4); EOS only if `add_eos_token` is true.

Decoding follows `llama_vocab::impl::token_to_piece`: control and unknown tokens
are dropped when special tokens are skipped; user-defined tokens always print;
a normal token has U+2581 turned back into a space; a byte token is its byte.
That turns every U+2581 in the text into a space on the way back, which is
llama.cpp's behaviour too, so a text that holds U+2581 does not round-trip.
Bytes are joined before the text is made, so a character split over byte
tokens comes back whole.

The vocabulary is held in two `Dict[String, Int]` (262,144 tokens and 514,906
merges). Reading the file and building both takes about 0.12 s on the real
model file, which is measured in the pull request that added this module.
"""

from std.collections import Dict

from mojo_dllm.formats.gguf import GGUFFile

comptime ATTR_UNKNOWN = 1
comptime ATTR_UNUSED = 2
comptime ATTR_NORMAL = 4
comptime ATTR_CONTROL = 8
comptime ATTR_USER_DEFINED = 16
comptime ATTR_BYTE = 32

comptime _NEWLINE = 10
comptime _SPACE = 32
comptime _FIRST_BLOCK_BYTE = 0xE2
"""U+2581 is E2 96 81 in UTF-8."""


def _bytes_string(b: Span[UInt8, _]) -> String:
    return String(StringSpan(unsafe_from_utf8=b))


def _attr_of_type(t: Int) -> Int:
    """llama.cpp's mapping from `tokenizer.ggml.token_type` to an attribute."""
    if t == 1:
        return ATTR_NORMAL
    if t == 2:
        return ATTR_UNKNOWN
    if t == 3:
        return ATTR_CONTROL
    if t == 4:
        return ATTR_USER_DEFINED
    if t == 5:
        return ATTR_UNUSED
    if t == 6:
        return ATTR_BYTE
    return 0


def _hex_digit(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 65 and c <= 70:
        return c - 55
    if c >= 97 and c <= 102:
        return c - 87
    return -1


def _byte_value(text: String) -> Int:
    """The byte named by a `<0xNN>` token text, or -1 when it is not one."""
    var b = text.as_bytes()
    if len(b) != 6 or b[0] != 60 or b[1] != 48 or b[2] != 120 or b[5] != 62:
        return -1
    var hi = _hex_digit(Int(b[3]))
    var lo = _hex_digit(Int(b[4]))
    if hi < 0 or lo < 0:
        return -1
    return hi * 16 + lo


def _byte_token_text(v: Int) -> String:
    var d = "0123456789ABCDEF".as_bytes()
    var out = List[UInt8]()
    out.append(60)
    out.append(48)
    out.append(120)
    out.append(d[v // 16])
    out.append(d[v % 16])
    out.append(62)
    return _bytes_string(Span(out))


def _utf8_len(lead: Int) -> Int:
    """`unicode_len_utf8`: the length a lead byte announces (1 for a stray byte).
    """
    var h = lead >> 4
    if h == 12 or h == 13:
        return 2
    if h == 14:
        return 3
    if h == 15:
        return 4
    return 1


struct _Bigram(Copyable, ImplicitlyCopyable, Movable):
    """A candidate merge of two neighbouring symbols, ordered by (rank, left).
    """

    var rank: Int
    var left: Int
    var right: Int
    var size: Int
    """Byte length of both symbols when the candidate was made; a merge since
    then changes it, which marks the candidate stale."""

    def __init__(out self, rank: Int, left: Int, right: Int, size: Int):
        self.rank = rank
        self.left = left
        self.right = right
        self.size = size


@always_inline
def _before(a: _Bigram, b: _Bigram) -> Bool:
    return a.rank < b.rank or (a.rank == b.rank and a.left < b.left)


def _heap_push(mut h: List[_Bigram], x: _Bigram):
    h.append(x)
    var i = len(h) - 1
    while i > 0:
        var p = (i - 1) // 2
        if _before(h[i], h[p]):
            var t = h[i]
            h[i] = h[p]
            h[p] = t
            i = p
        else:
            break


def _heap_pop(mut h: List[_Bigram]) -> _Bigram:
    var top = h[0]
    var last = h.pop()
    var n = len(h)
    if n > 0:
        h[0] = last
        var i = 0
        while True:
            var l = 2 * i + 1
            var r = l + 1
            var m = i
            if l < n and _before(h[l], h[m]):
                m = l
            if r < n and _before(h[r], h[m]):
                m = r
            if m == i:
                break
            var t = h[i]
            h[i] = h[m]
            h[m] = t
            i = m
    return top


struct Gemma4Vocab(Movable):
    """The vocabulary, merges and special-token rules of a gemma4 tokenizer."""

    var tokens: List[String]
    var attrs: List[Int]
    var ids: Dict[String, Int]
    var ranks: Dict[String, Int]
    """Merge "left right" to its rank; the first of two equal entries wins."""
    var specials: List[Int]
    """Ids that are cut out of text, longest text first."""
    var byte_ids: List[Int]
    """`<0xNN>` token ids, indexed by NN."""
    var bos_id: Int
    var eos_id: Int
    var pad_id: Int
    var unk_id: Int
    var mask_id: Int
    var eot_id: Int
    """`<turn|>`: the token that ends a turn, or -1."""
    var add_bos: Bool
    var add_eos: Bool
    var eog: List[Int]
    """End-of-generation ids: EOS, `<turn|>` and `<|tool_response>`."""

    def __init__(out self):
        self.tokens = List[String]()
        self.attrs = List[Int]()
        self.ids = Dict[String, Int]()
        self.ranks = Dict[String, Int]()
        self.specials = List[Int]()
        self.byte_ids = List[Int]()
        self.bos_id = -1
        self.eos_id = -1
        self.pad_id = -1
        self.unk_id = -1
        self.mask_id = -1
        self.eot_id = -1
        self.add_bos = False
        self.add_eos = False
        self.eog = List[Int]()

    def __init__(out self, g: GGUFFile) raises:
        self.tokens = g.get_str_array("tokenizer.ggml.tokens")
        var types = g.get_int_array("tokenizer.ggml.token_type")
        var n = len(self.tokens)
        if len(types) != n:
            raise Error(
                "tokenizer.ggml.token_type has "
                + String(len(types))
                + " entries but tokenizer.ggml.tokens has "
                + String(n)
            )
        self.ids = Dict[String, Int]()
        self.attrs = List[Int](capacity=n)
        for i in range(n):
            if self.tokens[i].byte_length() == 0:
                # llama.cpp gives an empty token this name instead.
                self.tokens[i] = "[EMPTY_" + String(i) + "]"
            if self.tokens[i] in self.ids:
                raise Error(
                    "tokenizer.ggml.tokens has the text "
                    + repr(self.tokens[i])
                    + " at ids "
                    + String(self.ids[self.tokens[i]])
                    + " and "
                    + String(i)
                )
            self.ids[self.tokens[i]] = i
            var attr = _attr_of_type(types[i])
            # llama.cpp turns these into control tokens (they end generation).
            # The vocabulary holds no other name from its list of 65.
            if (
                self.tokens[i] == "<eos>"
                or self.tokens[i] == "<turn|>"
                or self.tokens[i] == "<|tool_response>"
            ):
                attr |= ATTR_CONTROL
            self.attrs.append(attr)
            if attr == ATTR_BYTE and _byte_value(self.tokens[i]) < 0:
                raise Error(
                    "token "
                    + String(i)
                    + " has type byte but its text is "
                    + repr(self.tokens[i])
                    + ", not <0xNN>"
                )

        self.byte_ids = List[Int](length=256, fill=-1)
        for v in range(256):
            var text = _byte_token_text(v)
            if text not in self.ids:
                raise Error(
                    "the vocabulary has no byte token "
                    + text
                    + " (a character without a token is written as bytes)"
                )
            self.byte_ids[v] = self.ids[text]

        var merges = g.get_str_array("tokenizer.ggml.merges")
        self.ranks = Dict[String, Int]()
        for i in range(len(merges)):
            var m = merges[i].as_bytes()
            var has_space = False
            # llama.cpp splits at the first space after the first byte.
            for q in range(1, len(m)):
                if m[q] == _SPACE:
                    has_space = True
                    break
            if not has_space:
                raise Error(
                    "tokenizer.ggml.merges["
                    + String(i)
                    + "] = "
                    + repr(merges[i])
                    + " has no space between its two pieces"
                )
            if merges[i] not in self.ranks:
                self.ranks[merges[i]] = i

        self.specials = List[Int]()
        for i in range(n):
            if (
                self.attrs[i]
                & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN)
            ) != 0:
                self.specials.append(i)
        # Longest text first (llama.cpp sorts by length, unstably); ties by id.
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

        self.bos_id = self._special_id(g, "tokenizer.ggml.bos_token_id")
        self.eos_id = self._special_id(g, "tokenizer.ggml.eos_token_id")
        self.pad_id = self._special_id(g, "tokenizer.ggml.padding_token_id")
        self.unk_id = self._special_id(g, "tokenizer.ggml.unknown_token_id")
        self.mask_id = self._special_id(g, "tokenizer.ggml.mask_token_id")
        if self.bos_id < 0:
            raise Error(
                "metadata key tokenizer.ggml.bos_token_id is required: a"
                " gemma4 tokenizer always adds the BOS token"
            )
        # llama.cpp overrides add_bos_token to true for gemma4 (PR 21500).
        self.add_bos = True
        self.add_eos = False
        if g.has_key("tokenizer.ggml.add_eos_token"):
            self.add_eos = g.get_bool("tokenizer.ggml.add_eos_token")
        if self.add_eos and self.eos_id < 0:
            raise Error(
                "tokenizer.ggml.add_eos_token is true but"
                " tokenizer.ggml.eos_token_id is missing"
            )
        var scheme = g.get_str_or(
            "tokenizer.ggml.metaspace.prepend_scheme", "never"
        )
        if scheme != "never":
            raise Error(
                "tokenizer.ggml.metaspace.prepend_scheme is "
                + repr(scheme)
                + "; only 'never' is supported"
            )
        self.eot_id = self.ids.get("<turn|>", -1)
        self.eog = List[Int]()
        if self.eos_id >= 0:
            self.eog.append(self.eos_id)
        for name in ["<turn|>", "<|tool_response>"]:
            var id = self.ids.get(String(name), -1)
            if id >= 0 and id != self.eos_id:
                self.eog.append(id)

    @staticmethod
    def _special_id(g: GGUFFile, key: String) raises -> Int:
        var id = g.get_int_or(key, -1)
        if id >= 0 and id >= g.array_len("tokenizer.ggml.tokens"):
            raise Error(
                "metadata key "
                + key
                + " is "
                + String(id)
                + ", outside the vocabulary of "
                + String(g.array_len("tokenizer.ggml.tokens"))
                + " tokens"
            )
        return id

    def vocab_size(self) -> Int:
        return len(self.tokens)

    def token_id(self, text: String) -> Int:
        """The id of the token with exactly this text, or -1."""
        return self.ids.get(text, -1)

    def pair_rank(self, left: String, right: String) -> Int:
        """The rank of the merge `left right`, or -1."""
        return self.ranks.get(left + " " + right, -1)

    def is_eog(self, id: Int) -> Bool:
        for e in self.eog:
            if e == id:
                return True
        return False

    # ---- encoding -------------------------------------------------------

    def encode(self, text: String, add_special: Bool) raises -> List[Int]:
        var out = List[Int]()
        if add_special and self.add_bos:
            out.append(self.bos_id)
        var b = text.as_bytes()
        var n = len(b)

        # Fragments: [start, end) of raw text, or a special token (tok >= 0).
        var fs = List[Int]()
        var fe = List[Int]()
        var ft = List[Int]()
        if n > 0:
            fs.append(0)
            fe.append(n)
            ft.append(-1)
        for sid in self.specials:
            var st = self.tokens[sid].as_bytes()
            var m = len(st)
            var ns = List[Int]()
            var ne = List[Int]()
            var nt = List[Int]()
            for k in range(len(fs)):
                if ft[k] >= 0:
                    ns.append(fs[k])
                    ne.append(fe[k])
                    nt.append(ft[k])
                    continue
                var cur = fs[k]
                var pos = fs[k]
                var stop = fe[k]
                while pos + m <= stop:
                    var hit = b[pos] == st[0]
                    if hit:
                        for q in range(1, m):
                            if b[pos + q] != st[q]:
                                hit = False
                                break
                    if hit:
                        if pos > cur:
                            ns.append(cur)
                            ne.append(pos)
                            nt.append(-1)
                        ns.append(pos)
                        ne.append(pos + m)
                        nt.append(sid)
                        pos += m
                        cur = pos
                    else:
                        pos += 1
                if cur < stop:
                    ns.append(cur)
                    ne.append(stop)
                    nt.append(-1)
            fs = ns^
            fe = ne^
            ft = nt^

        for k in range(len(fs)):
            if ft[k] >= 0:
                out.append(ft[k])
            else:
                self._encode_text(b, fs[k], fe[k], out)
        if add_special and self.add_eos:
            out.append(self.eos_id)
        return out^

    def _encode_text(
        self, b: Span[UInt8, _], start: Int, end: Int, mut out: List[Int]
    ) raises:
        """Normalize one fragment, split it at newline runs, merge each piece.
        """
        var w = List[UInt8](capacity=(end - start) * 2)
        for i in range(start, end):
            if Int(b[i]) == _SPACE:
                w.append(0xE2)
                w.append(0x96)
                w.append(0x81)
            else:
                w.append(b[i])
        var i = 0
        var n = len(w)
        while i < n:
            var is_newline = Int(w[i]) == _NEWLINE
            var j = i
            while j < n and (Int(w[j]) == _NEWLINE) == is_newline:
                j += 1
            self._encode_word(w, i, j, is_newline, out)
            i = j

    def _encode_word(
        self,
        w: List[UInt8],
        start: Int,
        end: Int,
        newlines: Bool,
        mut out: List[Int],
    ) raises:
        if newlines:
            # A run of newlines that is a token is not merged (llama.cpp#21343).
            var whole = self.ids.get(_bytes_string(Span(w)[start:end]), -1)
            if whole >= 0:
                out.append(whole)
                return
        var ss = List[Int]()
        var sl = List[Int]()
        var at = start
        while at < end:
            var m = min(_utf8_len(Int(w[at])), end - at)
            ss.append(at)
            sl.append(m)
            at += m
        var count = len(ss)
        var prev = List[Int](capacity=count)
        var next = List[Int](capacity=count)
        for k in range(count):
            prev.append(k - 1)
            next.append(k + 1 if k + 1 < count else -1)
        var heap = List[_Bigram]()
        for k in range(1, count):
            self._push(heap, w, ss, sl, k - 1, k)
        while len(heap) > 0:
            var big = _heap_pop(heap)
            if sl[big.left] == 0 or sl[big.right] == 0:
                continue
            if sl[big.left] + sl[big.right] != big.size:
                continue  # one of the two grew since: an outdated candidate
            sl[big.left] += sl[big.right]
            sl[big.right] = 0
            next[big.left] = next[big.right]
            if next[big.right] >= 0:
                prev[next[big.right]] = big.left
            self._push(heap, w, ss, sl, prev[big.left], big.left)
            self._push(heap, w, ss, sl, big.left, next[big.left])
        for k in range(count):
            if sl[k] == 0:
                continue
            var piece = Span(w)[ss[k] : ss[k] + sl[k]]
            var id = self.ids.get(_bytes_string(piece), -1)
            if id >= 0:
                out.append(id)
            else:
                for q in range(len(piece)):
                    out.append(self.byte_ids[Int(piece[q])])

    def _push(
        self,
        mut heap: List[_Bigram],
        w: List[UInt8],
        ss: List[Int],
        sl: List[Int],
        left: Int,
        right: Int,
    ):
        if left < 0 or right < 0:
            return
        var key = List[UInt8](capacity=sl[left] + sl[right] + 1)
        for q in range(ss[left], ss[left] + sl[left]):
            key.append(w[q])
        key.append(UInt8(_SPACE))
        for q in range(ss[right], ss[right] + sl[right]):
            key.append(w[q])
        var rank = self.ranks.get(_bytes_string(Span(key)), -1)
        if rank >= 0:
            _heap_push(heap, _Bigram(rank, left, right, sl[left] + sl[right]))

    # ---- decoding -------------------------------------------------------

    def decode(
        self, ids: List[Int], skip_special: Bool = True
    ) raises -> String:
        var out = List[UInt8]()
        for id in ids:
            if id < 0 or id >= len(self.tokens):
                raise Error(
                    "token id " + String(id) + " is outside the vocabulary"
                )
            var a = self.attrs[id]
            if skip_special and (a & (ATTR_UNKNOWN | ATTR_CONTROL)) != 0:
                continue
            var text = self.tokens[id].as_bytes()
            if (a & (ATTR_UNKNOWN | ATTR_CONTROL | ATTR_USER_DEFINED)) != 0:
                for byte in text:
                    out.append(byte)
            elif (a & ATTR_NORMAL) != 0:
                var q = 0
                while q < len(text):
                    if (
                        q + 2 < len(text)
                        and Int(text[q]) == _FIRST_BLOCK_BYTE
                        and Int(text[q + 1]) == 0x96
                        and Int(text[q + 2]) == 0x81
                    ):
                        out.append(UInt8(_SPACE))
                        q += 3
                    else:
                        out.append(text[q])
                        q += 1
            elif (a & ATTR_BYTE) != 0:
                out.append(UInt8(_byte_value(self.tokens[id])))
            # An unused or undefined token has no text.
        # Generation can stop inside a multi-byte character; keep the text valid.
        return String(from_utf8_lossy=Span(out))


# ---- the chat template --------------------------------------------------


def _is_space(c: Int) -> Bool:
    """C `isspace` in the C locale, which is what llama.cpp's Jinja `trim` uses.
    """
    return c == 32 or (c >= 9 and c <= 13)


def _trim(s: String) -> String:
    var b = s.as_bytes()
    var lo = 0
    var hi = len(b)
    while lo < hi and _is_space(Int(b[lo])):
        lo += 1
    while hi > lo and _is_space(Int(b[hi - 1])):
        hi -= 1
    return _bytes_string(b[lo:hi])


def _find(hay: Span[UInt8, _], needle: String, start: Int) -> Int:
    var nb = needle.as_bytes()
    var m = len(nb)
    var i = start
    while i + m <= len(hay):
        var hit = True
        for q in range(m):
            if hay[i + q] != nb[q]:
                hit = False
                break
        if hit:
            return i
        i += 1
    return -1


def _strip_thinking(text: String) -> String:
    """The template's `strip_thinking` macro: drop every `<|channel>...<channel|>`
    block that ends in the text, keep what comes before each `<|channel>`, and
    trim the result."""
    var b = text.as_bytes()
    var result = List[UInt8]()
    var start = 0
    var close = String("<channel|>")
    while True:
        var stop = _find(b, close, start)
        var part_end = len(b) if stop < 0 else stop
        var part = b[start:part_end]
        var open_at = _find(part, "<|channel>", 0)
        var keep = len(part) if open_at < 0 else open_at
        for q in range(keep):
            result.append(part[q])
        if stop < 0:
            break
        start = stop + len(close.as_bytes())
    return _trim(_bytes_string(Span(result)))


def gemma4_chat_prompt(
    roles: List[String],
    contents: List[String],
    add_generation_prompt: Bool = True,
) raises -> String:
    """What `tokenizer.chat_template` renders for plain-text messages.

    The template is Jinja in the GGUF; this reproduces its output for the
    messages `mojo-dllm` can send: strings with the roles system, developer,
    user and assistant. Tool calls, tool responses, images and the thinking
    switch are not supported and a `tool` role is refused.

    The output starts with `<bos>`; `Tokenizer.encode` turns that text into
    token 2, so the result must be encoded without adding a BOS of its own.
    Each message is trimmed (ASCII whitespace), an assistant message loses its
    thinking channel first, and the prompt ends with an open model turn.
    """
    if len(roles) != len(contents):
        raise Error(
            "chat messages: "
            + String(len(roles))
            + " roles for "
            + String(len(contents))
            + " contents"
        )
    if len(roles) == 0:
        raise Error("a chat prompt needs at least one message")
    for r in roles:
        if r == "tool":
            raise Error(
                "the tool role is not supported: tool responses need the"
                " assistant message that called the tool"
            )
    var out = String("<bos>")
    var first = 0
    if roles[0] == "system" or roles[0] == "developer":
        out += "<|turn>system\n" + _trim(contents[0]) + "<turn|>\n"
        first = 1
    for i in range(first, len(roles)):
        var role = String("model") if roles[i] == "assistant" else roles[i]
        var continues = (
            i > first
            and roles[i] == "assistant"
            and roles[i - 1] == "assistant"
        )
        if not continues:
            out += "<|turn>" + role + "\n"
        if role == "model":
            out += _strip_thinking(contents[i])
        else:
            out += _trim(contents[i])
        out += "<turn|>\n"
    if add_generation_prompt:
        out += "<|turn>model\n"
    return out
