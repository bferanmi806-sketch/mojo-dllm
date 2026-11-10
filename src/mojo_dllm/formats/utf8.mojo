"""UTF-8 checks shared by the GGUF reader and the tokenizer.

A sequence is valid when it follows the standard: one byte below 0x80, or a
lead byte from 0xC2 to 0xF4 followed by the continuation bytes (0x80 to 0xBF)
its lead byte allows. That excludes a stray continuation byte, the lead bytes
0xC0, 0xC1 and 0xF5 to 0xFF, overlong forms (0xE0 followed by less than 0xA0,
0xF0 followed by less than 0x90), surrogates (0xED followed by more than 0x9F)
and values above U+10FFFF (0xF4 followed by more than 0x8F).

A byte that does not begin a valid sequence decodes to U+FFFD and decoding goes
on at the next byte. That is the rule of llama.cpp's `unicode_cpts_from_utf8`
(src/unicode.cpp lines 1389 to 1404 at 5f490d7), with one difference: llama.cpp
accepts a surrogate, and this module does not. It differs from Python's
`bytes.decode(errors="replace")` only for a sequence that stops short after a
valid prefix: Python writes one U+FFFD for the stretch and this module one for
each byte.
"""


@always_inline
def _is_continuation(b: UInt8) -> Bool:
    return (Int(b) & 0xC0) == 0x80


def utf8_sequence_length(b: Span[UInt8, _], i: Int) -> Int:
    """The length of the valid sequence that starts at `b[i]`, or 0 when the
    bytes there are not one. `i` must lie inside `b`."""
    var n = len(b)
    var c = Int(b[i])
    if c < 0x80:
        return 1
    if c < 0xC2:
        # A continuation byte with no lead, or 0xC0 and 0xC1, whose two-byte
        # forms would be overlong.
        return 0
    if c < 0xE0:
        if i + 1 < n and _is_continuation(b[i + 1]):
            return 2
        return 0
    if c < 0xF0:
        if i + 2 >= n:
            return 0
        var second = Int(b[i + 1])
        var lowest = 0xA0 if c == 0xE0 else 0x80
        var highest = 0x9F if c == 0xED else 0xBF
        if (
            second < lowest
            or second > highest
            or not _is_continuation(b[i + 2])
        ):
            return 0
        return 3
    if c < 0xF5:
        if i + 3 >= n:
            return 0
        var second = Int(b[i + 1])
        var lowest = 0x90 if c == 0xF0 else 0x80
        var highest = 0x8F if c == 0xF4 else 0xBF
        if (
            second < lowest
            or second > highest
            or not _is_continuation(b[i + 2])
            or not _is_continuation(b[i + 3])
        ):
            return 0
        return 4
    return 0


def utf8_first_invalid(b: Span[UInt8, _]) -> Int:
    """The offset of the first byte that begins an invalid sequence, or -1 when
    `b` is valid UTF-8."""
    var i = 0
    var n = len(b)
    while i < n:
        var step = utf8_sequence_length(b, i)
        if step == 0:
            return i
        i += step
    return -1


def utf8_decode_bytes(b: Span[UInt8, _]) -> List[Int]:
    """The code points of `b`. A byte that begins an invalid sequence becomes
    U+FFFD, and decoding goes on at the next byte."""
    var out = List[Int](capacity=len(b))
    var i = 0
    var n = len(b)
    while i < n:
        var step = utf8_sequence_length(b, i)
        var c = Int(b[i])
        if step == 1:
            out.append(c)
        elif step == 2:
            out.append(((c & 0x1F) << 6) | (Int(b[i + 1]) & 0x3F))
        elif step == 3:
            out.append(
                ((c & 0x0F) << 12)
                | ((Int(b[i + 1]) & 0x3F) << 6)
                | (Int(b[i + 2]) & 0x3F)
            )
        elif step == 4:
            out.append(
                ((c & 0x07) << 18)
                | ((Int(b[i + 1]) & 0x3F) << 12)
                | ((Int(b[i + 2]) & 0x3F) << 6)
                | (Int(b[i + 3]) & 0x3F)
            )
        else:
            out.append(0xFFFD)
            i += 1
            continue
        i += step
    return out^
