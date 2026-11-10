"""GGUF reader over a memory map.

The header, the key/value block and the tensor directory are parsed once at
open. Scalar values are decoded on demand from their recorded offset, and
tensor data is never copied: a tensor is an absolute offset into the mapping.

Every read of the header, the metadata and the tensor directory goes through
`Reader`, which checks it against the bytes left in the file. A GGUF file is
untrusted input, and an unchecked length prefix on a memory map turns into a
segfault instead of an error. Integers are assembled from bytes, so no read
depends on alignment. Tensor bytes are not read here: `GGUFFile.__init__`
checks each tensor's extent against the file.
"""

from std.collections import Dict
from std.memory import bitcast

from mojo_dllm.formats.utf8 import utf8_first_invalid
from mojo_dllm.sys.mmap import Mapping
from mojo_dllm.sys.mem import U8Ptr

comptime GGUF_TYPE_UINT8 = 0
comptime GGUF_TYPE_INT8 = 1
comptime GGUF_TYPE_UINT16 = 2
comptime GGUF_TYPE_INT16 = 3
comptime GGUF_TYPE_UINT32 = 4
comptime GGUF_TYPE_INT32 = 5
comptime GGUF_TYPE_FLOAT32 = 6
comptime GGUF_TYPE_BOOL = 7
comptime GGUF_TYPE_STRING = 8
comptime GGUF_TYPE_ARRAY = 9
comptime GGUF_TYPE_UINT64 = 10
comptime GGUF_TYPE_INT64 = 11
comptime GGUF_TYPE_FLOAT64 = 12

comptime GGML_F32 = 0
comptime GGML_F16 = 1
comptime GGML_Q5_0 = 6
comptime GGML_Q8_0 = 8
comptime GGML_Q4_K = 12
comptime GGML_Q5_K = 13
comptime GGML_Q6_K = 14
comptime GGML_BF16 = 30

comptime DEFAULT_ALIGNMENT = 32


def ggml_type_name(t: Int) -> String:
    """Names as llama.cpp prints them, so error messages match its vocabulary.
    """
    if t == 0:
        return "F32"
    if t == 1:
        return "F16"
    if t == 2:
        return "Q4_0"
    if t == 3:
        return "Q4_1"
    if t == 6:
        return "Q5_0"
    if t == 7:
        return "Q5_1"
    if t == 8:
        return "Q8_0"
    if t == 9:
        return "Q8_1"
    if t == 10:
        return "Q2_K"
    if t == 11:
        return "Q3_K"
    if t == 12:
        return "Q4_K"
    if t == 13:
        return "Q5_K"
    if t == 14:
        return "Q6_K"
    if t == 15:
        return "Q8_K"
    if t == 16:
        return "IQ2_XXS"
    if t == 17:
        return "IQ2_XS"
    if t == 18:
        return "IQ3_XXS"
    if t == 19:
        return "IQ1_S"
    if t == 20:
        return "IQ4_NL"
    if t == 21:
        return "IQ3_S"
    if t == 22:
        return "IQ2_S"
    if t == 23:
        return "IQ4_XS"
    if t == 29:
        return "IQ1_M"
    if t == 30:
        return "BF16"
    return "type#" + String(t)


def ggml_block_bytes(t: Int) -> Int:
    """Bytes in one block of type `t`, or 0 for a type this reader cannot size.

    F32, F16 and BF16 have blocks of one element.
    """
    if t == GGML_F32:
        return 4
    if t == GGML_F16 or t == GGML_BF16:
        return 2
    if t == GGML_Q5_0:
        return 22
    if t == GGML_Q8_0:
        return 34
    if t == GGML_Q4_K:
        return 144
    if t == GGML_Q5_K:
        return 176
    if t == GGML_Q6_K:
        return 210
    return 0


def ggml_block_size(t: Int) -> Int:
    """Elements in one block of type `t`."""
    if t == GGML_Q5_0 or t == GGML_Q8_0:
        return 32
    if t == GGML_Q4_K or t == GGML_Q5_K or t == GGML_Q6_K:
        return 256
    return 1


def ggml_row_bytes(t: Int, n: Int) raises -> Int:
    """Bytes that `n` elements of type `t` occupy. Unsupported types raise, and
    so does an `n` that ends in a partial block or whose size does not fit an Int.
    """
    var block_bytes = ggml_block_bytes(t)
    if block_bytes == 0:
        raise Error("unsupported GGUF tensor type: " + ggml_type_name(t))
    var block = ggml_block_size(t)
    if n < 0 or n % block != 0:
        raise Error(
            "row length "
            + String(n)
            + " is not a multiple of the "
            + ggml_type_name(t)
            + " block size "
            + String(block)
        )
    var blocks = n // block
    # Divide first: multiplying an untrusted length can wrap.
    if blocks > Int.MAX // block_bytes:
        raise Error(
            "a row of "
            + String(n)
            + " "
            + ggml_type_name(t)
            + " elements is too large to size"
        )
    return blocks * block_bytes


@fieldwise_init
struct KV(Copyable, ImplicitlyCopyable, Movable):
    var key: String
    var kind: Int
    var at: Int
    """Absolute offset where the value payload starts."""
    var elem_kind: Int
    """Element type for arrays, -1 otherwise."""
    var count: Int


@fieldwise_init
struct TensorInfo(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    var n_dims: Int
    var ne0: Int
    """Innermost dimension: the length of one row."""
    var ne1: Int
    var ne2: Int
    var ne3: Int
    var ggml_type: Int
    var data_offset: Int
    """Absolute offset of the first byte in the mapping."""
    var n_bytes: Int
    """Bytes the tensor occupies; 0 when the type is not `supported`."""
    var supported: Bool
    """Whether the type has a size here (`ggml_block_bytes`). `inspect` lists a
    tensor of any other type, and every loader refuses it."""

    def n_elements(self) -> Int:
        return self.ne0 * self.ne1 * self.ne2 * self.ne3

    def n_rows(self) -> Int:
        return self.ne1 * self.ne2 * self.ne3

    def row_bytes(self) raises -> Int:
        return ggml_row_bytes(self.ggml_type, self.ne0)

    def type_name(self) -> String:
        return ggml_type_name(self.ggml_type)

    def shape_str(self) -> String:
        var s = String("[") + String(self.ne0)
        if self.n_dims > 1:
            s += ", " + String(self.ne1)
        if self.n_dims > 2:
            s += ", " + String(self.ne2)
        if self.n_dims > 3:
            s += ", " + String(self.ne3)
        return s + "]"


struct Reader(Movable):
    """Bounds-checked little-endian read head."""

    var base: Int
    var size: Int
    var at: Int

    def __init__(out self, base: Int, size: Int, at: Int = 0) raises:
        if at < 0 or at > size:
            raise Error(
                "GGUF read position "
                + String(at)
                + " is outside the file of "
                + String(size)
                + " bytes"
            )
        self.base = base
        self.size = size
        self.at = at

    def _need(self, n: Int) raises:
        # `at <= size` holds from construction on, so `size - at` cannot wrap.
        # `at + n` can: a length near 2^63 would pass a check written that way.
        if n < 0 or n > self.size - self.at:
            raise Error(
                "GGUF file truncated: wanted "
                + String(n)
                + " bytes at offset "
                + String(self.at)
                + " of "
                + String(self.size)
            )

    def _le(mut self, n: Int) raises -> UInt64:
        self._need(n)
        var p = U8Ptr(unsafe_from_address=self.base)
        var v = UInt64(0)
        for i in range(n):
            v |= UInt64(p.unsafe_load(self.at + i)) << UInt64(8 * i)
        self.at += n
        return v

    def u8(mut self) raises -> UInt64:
        return self._le(1)

    def u16(mut self) raises -> UInt64:
        return self._le(2)

    def u32(mut self) raises -> UInt64:
        return self._le(4)

    def u64(mut self) raises -> UInt64:
        return self._le(8)

    def skip(mut self, n: Int) raises:
        self._need(n)
        self.at += n

    def string_at(mut self) raises -> Tuple[Int, Int]:
        """The file offset and the length of the string at the read head, and
        moves past it. The bytes are not looked at."""
        var n = Int(self.u64())
        self._need(n)
        var start = self.at
        self.at += n
        return (start, n)

    def string(mut self) raises -> String:
        """The next string. Its bytes must be valid UTF-8: `String` operations
        assume it, and a token text or a key read from an untrusted file would
        otherwise reach `repr`, `replace` and the tokenizers unchecked. A file
        with such a string is refused and the error says where."""
        var start, n = self.string_at()
        var span = Span(
            unsafe_ptr=U8Ptr(unsafe_from_address=self.base + start), length=n
        )
        var bad = utf8_first_invalid(span)
        if bad >= 0:
            raise Error(
                "string at file offset "
                + String(start)
                + " is not valid UTF-8: byte "
                + String(bad)
                + " of "
                + String(n)
                + " is 0x"
                + _hex_byte(span[bad])
            )
        return String(StringSpan(unsafe_from_utf8=span))

    def skip_string(mut self) raises:
        var n = Int(self.u64())
        self.skip(n)


def _hex_byte(b: UInt8) -> String:
    var digits = "0123456789ABCDEF".as_bytes()
    var out = List[UInt8]()
    out.append(digits[Int(b) >> 4])
    out.append(digits[Int(b) & 15])
    return String(StringSpan(unsafe_from_utf8=Span(out)))


def _scalar_size(kind: Int) raises -> Int:
    if (
        kind == GGUF_TYPE_UINT8
        or kind == GGUF_TYPE_INT8
        or kind == GGUF_TYPE_BOOL
    ):
        return 1
    if kind == GGUF_TYPE_UINT16 or kind == GGUF_TYPE_INT16:
        return 2
    if (
        kind == GGUF_TYPE_UINT32
        or kind == GGUF_TYPE_INT32
        or kind == GGUF_TYPE_FLOAT32
    ):
        return 4
    if (
        kind == GGUF_TYPE_UINT64
        or kind == GGUF_TYPE_INT64
        or kind == GGUF_TYPE_FLOAT64
    ):
        return 8
    raise Error("GGUF value type " + String(kind) + " has no fixed size")


def _signed(v: UInt64, bytes: Int) -> Int:
    var bits = 8 * bytes
    if bits == 64:
        return Int(Int64(v))
    var sign = UInt64(1) << UInt64(bits - 1)
    if v & sign:
        return Int(v) - (1 << bits)
    return Int(v)


struct GGUFFile(Movable):
    var map: Mapping
    var version: Int
    var kvs: List[KV]
    var tensors: List[TensorInfo]
    var alignment: Int
    var data_start: Int

    def __init__(out self, path: String) raises:
        self.map = Mapping(path)
        self.kvs = List[KV]()
        self.tensors = List[TensorInfo]()
        self.alignment = DEFAULT_ALIGNMENT
        var r = Reader(self.map.addr, self.map.size)
        r._need(4)
        var p = self.map.ptr()
        if not (
            p.unsafe_load(0) == 0x47
            and p.unsafe_load(1) == 0x47
            and p.unsafe_load(2) == 0x55
            and p.unsafe_load(3) == 0x46
        ):
            raise Error("not a GGUF file (bad magic): " + path)
        r.skip(4)
        self.version = Int(r.u32())
        if self.version < 2 or self.version > 3:
            raise Error("unsupported GGUF version " + String(self.version))
        var n_tensors = Int(r.u64())
        var n_kv = Int(r.u64())
        if n_tensors < 0 or n_kv < 0 or n_tensors > 1 << 20 or n_kv > 1 << 20:
            raise Error("GGUF header counts are implausible")

        var seen_keys = Dict[String, Int]()
        for i in range(n_kv):
            var key: String
            try:
                key = r.string()
            except e:
                raise Error("metadata entry " + String(i) + ": " + String(e))
            if key in seen_keys:
                raise Error(
                    "metadata key "
                    + key
                    + " is listed twice (entries "
                    + String(seen_keys[key])
                    + " and "
                    + String(i)
                    + ")"
                )
            seen_keys[key] = i
            var kind = Int(r.u32())
            if key == "general.alignment" and kind != GGUF_TYPE_UINT32:
                raise Error(
                    key + " has type " + String(kind) + ", expected UINT32"
                )
            if kind == GGUF_TYPE_ARRAY:
                var ek = Int(r.u32())
                var raw_count = r.u64()
                var count = Int(raw_count)
                var at = r.at
                var count_error = (
                    key + " has invalid array count " + String(raw_count)
                )
                if count < 0:
                    raise Error(count_error)
                if ek == GGUF_TYPE_STRING:
                    # Even an empty string needs its uint64 length prefix.
                    if count > (self.map.size - r.at) // 8:
                        raise Error(count_error)
                    for _ in range(count):
                        r.skip_string()
                elif ek == GGUF_TYPE_ARRAY:
                    raise Error("nested GGUF arrays are not supported: " + key)
                else:
                    var element_size = _scalar_size(ek)
                    # Divide first: multiplying an untrusted count can wrap.
                    if count > (self.map.size - r.at) // element_size:
                        raise Error(count_error)
                    r.skip(count * element_size)
                self.kvs.append(KV(key, kind, at, ek, count))
            else:
                var at = r.at
                if kind == GGUF_TYPE_STRING:
                    r.skip_string()
                else:
                    r.skip(_scalar_size(kind))
                self.kvs.append(KV(key, kind, at, -1, 1))
                if key == "general.alignment":
                    var rr = Reader(self.map.addr, self.map.size, at)
                    self.alignment = Int(rr.u32())
                    if (
                        self.alignment == 0
                        or (self.alignment & (self.alignment - 1)) != 0
                    ):
                        raise Error(
                            key
                            + " has invalid value "
                            + String(self.alignment)
                            + "; expected a non-zero power of two"
                        )

        var rel = List[Int]()
        var raw_rel = List[UInt64]()
        var seen_names = Dict[String, Int]()
        for i in range(n_tensors):
            var name: String
            try:
                name = r.string()
            except e:
                raise Error("tensor entry " + String(i) + ": " + String(e))
            if name in seen_names:
                raise Error(
                    "tensor "
                    + name
                    + " is listed twice (entries "
                    + String(seen_names[name])
                    + " and "
                    + String(i)
                    + ")"
                )
            seen_names[name] = i
            var n_dims = Int(r.u32())
            if n_dims < 1 or n_dims > 4:
                raise Error(
                    "tensor " + name + " has " + String(n_dims) + " dims"
                )
            var ne = List[Int](length=4, fill=1)
            var elements = 1
            for d in range(n_dims):
                var raw_dim = r.u64()
                var dim = Int(raw_dim)
                if dim < 1:
                    raise Error(
                        "tensor "
                        + name
                        + " has dimension "
                        + String(d)
                        + " of "
                        + String(raw_dim)
                        + "; every dimension must be at least 1"
                    )
                # Divide first: multiplying untrusted dimensions can wrap.
                if dim > Int.MAX // elements:
                    raise Error("tensor " + name + " has too many elements")
                elements *= dim
                ne[d] = dim
            var t = Int(r.u32())
            var raw_off = r.u64()
            rel.append(Int(raw_off))
            raw_rel.append(raw_off)
            # A type with no size here (`supported` False) is listed by
            # `inspect` and refused by every loader; it has no byte count.
            var block_bytes = ggml_block_bytes(t)
            var supported = block_bytes > 0
            var n_bytes = 0
            if supported:
                var block = ggml_block_size(t)
                if ne[0] % block != 0:
                    raise Error(
                        "tensor "
                        + name
                        + " is "
                        + ggml_type_name(t)
                        + " with "
                        + String(ne[0])
                        + " elements per row, which is not a multiple of its"
                        " block size "
                        + String(block)
                    )
                # Every size below is checked against the whole file by
                # dividing first, so none of the products can wrap. The
                # position of the tensor inside the file comes after the
                # directory has been read.
                var extends = (
                    "tensor " + name + " extends past the end of the file"
                )
                var blocks = ne[0] // block
                if blocks > self.map.size // block_bytes:
                    raise Error(extends)
                var row_bytes = blocks * block_bytes
                var rows = ne[1] * ne[2] * ne[3]
                if rows > self.map.size // row_bytes:
                    raise Error(extends)
                n_bytes = row_bytes * rows
            self.tensors.append(
                TensorInfo(
                    name,
                    n_dims,
                    ne[0],
                    ne[1],
                    ne[2],
                    ne[3],
                    t,
                    0,
                    n_bytes,
                    supported,
                )
            )
        var a = self.alignment
        self.data_start = (r.at + a - 1) // a * a
        # Negative when the directory ends the file and padding runs past it.
        var room = self.map.size - self.data_start
        for i in range(len(self.tensors)):
            var name = self.tensors[i].name
            var off = rel[i]
            if off < 0:
                raise Error(
                    "tensor "
                    + name
                    + " has offset "
                    + String(raw_rel[i])
                    + ", which is before the data section"
                )
            if off % a != 0:
                raise Error(
                    "tensor "
                    + name
                    + " has offset "
                    + String(off)
                    + ", which is not a multiple of the alignment "
                    + String(a)
                )
            # `off` and `n_bytes` are both in [0, 2^63) and `room` is below the
            # file size, so this subtraction cannot wrap.
            if self.tensors[i].n_bytes > room - off:
                raise Error(
                    "tensor " + name + " extends past the end of the file"
                )
            self.tensors[i].data_offset = self.data_start + off

    # ---- metadata -------------------------------------------------------

    def has_key(self, key: String) -> Bool:
        for i in range(len(self.kvs)):
            if self.kvs[i].key == key:
                return True
        return False

    def _kv(self, key: String) raises -> KV:
        for i in range(len(self.kvs)):
            if self.kvs[i].key == key:
                return self.kvs[i]
        raise Error("required metadata key missing: " + key)

    def _read_scalar_int(self, kind: Int, at: Int) raises -> Int:
        var r = Reader(self.map.addr, self.map.size, at)
        if kind == GGUF_TYPE_UINT8 or kind == GGUF_TYPE_BOOL:
            return Int(r.u8())
        if kind == GGUF_TYPE_INT8:
            return _signed(r.u8(), 1)
        if kind == GGUF_TYPE_UINT16:
            return Int(r.u16())
        if kind == GGUF_TYPE_INT16:
            return _signed(r.u16(), 2)
        if kind == GGUF_TYPE_UINT32:
            return Int(r.u32())
        if kind == GGUF_TYPE_INT32:
            return _signed(r.u32(), 4)
        if kind == GGUF_TYPE_UINT64:
            return Int(r.u64())
        if kind == GGUF_TYPE_INT64:
            return _signed(r.u64(), 8)
        raise Error("GGUF value of type " + String(kind) + " is not an integer")

    def get_int(self, key: String) raises -> Int:
        var kv = self._kv(key)
        try:
            return self._read_scalar_int(kv.kind, kv.at)
        except e:
            raise Error("metadata key " + key + ": " + String(e))

    def get_int_or(self, key: String, default: Int) raises -> Int:
        if not self.has_key(key):
            return default
        return self.get_int(key)

    def get_f32(self, key: String) raises -> Float32:
        var kv = self._kv(key)
        var r = Reader(self.map.addr, self.map.size, kv.at)
        if kv.kind == GGUF_TYPE_FLOAT32:
            return bitcast[DType.float32](UInt32(r.u32()))
        if kv.kind == GGUF_TYPE_FLOAT64:
            return Float32(bitcast[DType.float64](r.u64()))
        try:
            return Float32(self._read_scalar_int(kv.kind, kv.at))
        except e:
            raise Error("metadata key " + key + ": " + String(e))

    def get_f32_or(self, key: String, default: Float32) raises -> Float32:
        if not self.has_key(key):
            return default
        return self.get_f32(key)

    def get_bool(self, key: String) raises -> Bool:
        return self.get_int(key) != 0

    def get_str(self, key: String) raises -> String:
        var kv = self._kv(key)
        if kv.kind != GGUF_TYPE_STRING:
            raise Error("metadata key " + key + " is not a string")
        var r = Reader(self.map.addr, self.map.size, kv.at)
        try:
            return r.string()
        except e:
            raise Error("metadata key " + key + ": " + String(e))

    def get_str_or(self, key: String, default: String) raises -> String:
        if not self.has_key(key):
            return default
        return self.get_str(key)

    def array_len(self, key: String) raises -> Int:
        return self._kv(key).count

    def get_str_array(self, key: String) raises -> List[String]:
        var kv = self._kv(key)
        if kv.kind != GGUF_TYPE_ARRAY or kv.elem_kind != GGUF_TYPE_STRING:
            raise Error("metadata key " + key + " is not a string array")
        var out = List[String](capacity=kv.count)
        var r = Reader(self.map.addr, self.map.size, kv.at)
        for i in range(kv.count):
            try:
                out.append(r.string())
            except e:
                raise Error(
                    "metadata key " + key + "[" + String(i) + "]: " + String(e)
                )
        return out^

    def get_int_array(self, key: String) raises -> List[Int]:
        var kv = self._kv(key)
        if kv.kind != GGUF_TYPE_ARRAY:
            raise Error("metadata key " + key + " is not an array")
        if (
            kv.elem_kind == GGUF_TYPE_FLOAT32
            or kv.elem_kind == GGUF_TYPE_FLOAT64
            or kv.elem_kind == GGUF_TYPE_STRING
            or kv.elem_kind == GGUF_TYPE_BOOL
        ):
            raise Error("metadata key " + key + " is not an integer array")
        var sz = _scalar_size(kv.elem_kind)
        var out = List[Int](capacity=kv.count)
        for i in range(kv.count):
            out.append(self._read_scalar_int(kv.elem_kind, kv.at + i * sz))
        return out^

    def get_bool_array(self, key: String) raises -> List[Bool]:
        var kv = self._kv(key)
        # elem_kind is -1 for a scalar, so this also refuses a lone bool.
        if kv.elem_kind != GGUF_TYPE_BOOL:
            raise Error("metadata key " + key + " is not a bool array")
        var out = List[Bool](capacity=kv.count)
        var r = Reader(self.map.addr, self.map.size, kv.at)
        for _ in range(kv.count):
            out.append(r.u8() != 0)
        return out^

    def get_f32_array(self, key: String) raises -> List[Float32]:
        var kv = self._kv(key)
        if kv.kind != GGUF_TYPE_ARRAY or kv.elem_kind != GGUF_TYPE_FLOAT32:
            raise Error("metadata key " + key + " is not a float32 array")
        var out = List[Float32](capacity=kv.count)
        var r = Reader(self.map.addr, self.map.size, kv.at)
        for _ in range(kv.count):
            out.append(bitcast[DType.float32](UInt32(r.u32())))
        return out^

    def architecture(self) raises -> String:
        return self.get_str("general.architecture")

    def value_str(self, i: Int) raises -> String:
        """A one-line rendering of KV `i` for `inspect`."""
        var kv = self.kvs[i]
        if kv.kind == GGUF_TYPE_STRING:
            var rd = Reader(self.map.addr, self.map.size, kv.at)
            var start, n = rd.string_at()
            var span = Span(
                unsafe_ptr=U8Ptr(unsafe_from_address=self.map.addr + start),
                length=n,
            )
            var bad = utf8_first_invalid(span)
            if bad >= 0:
                return (
                    "<string, "
                    + String(n)
                    + " bytes, not valid UTF-8 at byte "
                    + String(bad)
                    + ">"
                )
            var s = String(StringSpan(unsafe_from_utf8=span))
            if s.byte_length() > 60:
                return "<string, " + String(s.byte_length()) + " bytes>"
            return '"' + s.replace("\n", "\\n") + '"'
        if kv.kind == GGUF_TYPE_ARRAY:
            return "<array of " + String(kv.count) + ">"
        if kv.kind == GGUF_TYPE_FLOAT32 or kv.kind == GGUF_TYPE_FLOAT64:
            return String(self.get_f32(kv.key))
        if kv.kind == GGUF_TYPE_BOOL:
            return "true" if self.get_bool(kv.key) else "false"
        return String(self._read_scalar_int(kv.kind, kv.at))

    # ---- tensors --------------------------------------------------------

    def n_tensors(self) -> Int:
        return len(self.tensors)

    def has_tensor(self, name: String) -> Bool:
        for i in range(len(self.tensors)):
            if self.tensors[i].name == name:
                return True
        return False

    def tensor(self, name: String) raises -> TensorInfo:
        for i in range(len(self.tensors)):
            if self.tensors[i].name == name:
                return self.tensors[i]
        raise Error("tensor not found: " + name)

    def data_addr(self, t: TensorInfo) -> Int:
        return self.map.addr + t.data_offset
