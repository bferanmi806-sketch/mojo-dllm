"""GGUF reader over a memory map.

The header, the key/value block and the tensor directory are parsed once at
open. Scalar values are decoded on demand from their recorded offset, and
tensor data is never copied: a tensor is an absolute offset into the mapping.

Every read is bounds-checked. A GGUF file is untrusted input, and an unchecked
length prefix on a memory map turns into a segfault instead of an error.
Integers are assembled from bytes, so no read depends on alignment.
"""

from std.memory import bitcast

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


def ggml_row_bytes(t: Int, n: Int) raises -> Int:
    """Bytes that `n` elements of type `t` occupy. Unsupported types raise."""
    if t == GGML_F32:
        return n * 4
    if t == GGML_F16 or t == GGML_BF16:
        return n * 2
    if t == GGML_Q5_0:
        return n // 32 * 22
    if t == GGML_Q8_0:
        return n // 32 * 34
    if t == GGML_Q4_K:
        return n // 256 * 144
    if t == GGML_Q5_K:
        return n // 256 * 176
    if t == GGML_Q6_K:
        return n // 256 * 210
    raise Error("unsupported GGUF tensor type: " + ggml_type_name(t))


def _tensor_bytes(t: Int, ne0: Int, rows: Int) -> Int:
    try:
        return ggml_row_bytes(t, ne0) * rows
    except:
        return -1


def ggml_block_size(t: Int) -> Int:
    if t == GGML_Q5_0 or t == GGML_Q8_0:
        return 32
    if t == GGML_Q4_K or t == GGML_Q5_K or t == GGML_Q6_K:
        return 256
    return 1


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

    def __init__(out self, base: Int, size: Int, at: Int = 0):
        self.base = base
        self.size = size
        self.at = at

    def _need(self, n: Int) raises:
        if n < 0 or self.at + n > self.size:
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

    def string(mut self) raises -> String:
        var n = Int(self.u64())
        self._need(n)
        var p = U8Ptr(unsafe_from_address=self.base + self.at)
        var s = String(
            StringSpan(unsafe_from_utf8=Span(unsafe_ptr=p, length=n))
        )
        self.at += n
        return s

    def skip_string(mut self) raises:
        var n = Int(self.u64())
        self.skip(n)


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

        for _ in range(n_kv):
            var key = r.string()
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
        for _ in range(n_tensors):
            var name = r.string()
            var n_dims = Int(r.u32())
            if n_dims < 1 or n_dims > 4:
                raise Error(
                    "tensor " + name + " has " + String(n_dims) + " dims"
                )
            var ne = List[Int](length=4, fill=1)
            for d in range(n_dims):
                ne[d] = Int(r.u64())
            var t = Int(r.u32())
            var off = Int(r.u64())
            rel.append(off)
            # -1 marks an unsupported type: `inspect` lists it, loaders refuse it.
            var nb = _tensor_bytes(t, ne[0], ne[1] * ne[2] * ne[3])
            self.tensors.append(
                TensorInfo(name, n_dims, ne[0], ne[1], ne[2], ne[3], t, 0, nb)
            )
        var a = self.alignment
        self.data_start = (r.at + a - 1) // a * a
        for i in range(len(self.tensors)):
            self.tensors[i].data_offset = self.data_start + rel[i]
            var t = self.tensors[i]
            if t.n_bytes > 0 and t.data_offset + t.n_bytes > self.map.size:
                raise Error(
                    "tensor " + t.name + " extends past the end of the file"
                )

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
        return r.string()

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
        for _ in range(kv.count):
            out.append(r.string())
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
            var s = rd.string()
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
