#!/usr/bin/env python3
"""Generate the binary test fixtures under tests/fixtures/.

Run with:  uv run --only-group fixtures python scripts/gen_fixtures.py

Python is a development tool here, never a runtime dependency. Every expected
value comes from llama.cpp's own `gguf` package (its K-quant dequantizers) or
from a plain numpy forward pass over those dequantized weights, so the Mojo
code is checked against the reference implementation and not against itself.

Outputs (deterministic, seed 1234 unless noted):
  tests/fixtures/dequant-types.gguf Q8_0 / Q5_K / BF16 / Q5_0 references, separate seed 340034
  tests/fixtures/unsupported-embedding.gguf minimal IQ2_XXS embedding, no RNG
  tests/fixtures/kquants.gguf      Q4_K / Q6_K / F16 tensors + their f32 dequantization
  tests/fixtures/bad-magic.gguf    a file whose magic is wrong
  tests/fixtures/tiny-llada.gguf   2-layer LLaDA-shaped model, hidden 256, vocab 512
  tests/fixtures/tiny-llada.expected.f32
                                   numpy reference logits for TINY_TOKENS, row-major [L, vocab]
  tests/fixtures/tiny-dream.gguf   2-layer Dream-shaped model (GQA, QKV bias) and its numpy logits
  tests/fixtures/tiny-dream.expected.f32
  tests/fixtures/malformed/        one-byte-damaged GGUF headers and metadata
  tests/fixtures/metadata/         valid and invalid ModelConfig metadata
"""

from __future__ import annotations

import pathlib

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize, quantize

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "fixtures"
RNG = np.random.default_rng(1234)

QK_K = 256
Q4K_BYTES = 144
Q6K_BYTES = 210

TINY_TOKENS = [5, 17, 300, 511, 511, 42, 511, 7]
TINY: dict[str, int] = dict(
    vocab=512,
    hidden=256,
    heads=2,
    head_dim=128,
    ffn=512,
    layers=2,
    mask=511,
)
TINY_EPS = 1e-5
TINY_THETA = 500000.0


def f16_bytes(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float16).view(np.uint8)


def random_q4k(rows: int, cols: int, d_scale: float) -> np.ndarray:
    """Valid Q4_K bytes: small positive f16 d/dmin, random 6-bit scales, random nibbles."""
    nb = rows * cols // QK_K
    blocks = np.zeros((nb, Q4K_BYTES), dtype=np.uint8)
    blocks[:, 0:2] = f16_bytes(RNG.uniform(0.2, 1.0, nb) * d_scale).reshape(nb, 2)
    blocks[:, 2:4] = f16_bytes(RNG.uniform(0.2, 1.0, nb) * d_scale * 4).reshape(nb, 2)
    blocks[:, 4:16] = RNG.integers(0, 256, (nb, 12), dtype=np.uint8)
    blocks[:, 16:] = RNG.integers(0, 256, (nb, 128), dtype=np.uint8)
    return blocks.reshape(rows, cols // QK_K * Q4K_BYTES)


def random_q6k(rows: int, cols: int, d_scale: float) -> np.ndarray:
    nb = rows * cols // QK_K
    blocks = np.zeros((nb, Q6K_BYTES), dtype=np.uint8)
    blocks[:, 0:192] = RNG.integers(0, 256, (nb, 192), dtype=np.uint8)
    blocks[:, 192:208] = RNG.integers(-64, 64, (nb, 16)).astype(np.int8).view(np.uint8)
    blocks[:, 208:210] = f16_bytes(RNG.uniform(0.2, 1.0, nb) * d_scale).reshape(nb, 2)
    return blocks.reshape(rows, cols // QK_K * Q6K_BYTES)


def kquants() -> None:
    w = GGUFWriter(str(OUT / "kquants.gguf"), "test")
    w.add_uint32("test.u32", 7)
    w.add_int32("test.i32", -3)
    w.add_float32("test.f32", 0.25)
    w.add_bool("test.bool", True)
    w.add_string("test.str", "héllo")
    w.add_array("test.strs", ["a", "bc", "déf"])
    w.add_array("test.ints", [1, -2, 3])

    q4 = random_q4k(4, 512, 0.01)
    q6 = random_q6k(4, 512, 0.01)
    f16 = RNG.standard_normal((2, 64)).astype(np.float16)
    w.add_tensor("q4k", q4, raw_shape=q4.shape, raw_dtype=Q.Q4_K)
    w.add_tensor("q6k", q6, raw_shape=q6.shape, raw_dtype=Q.Q6_K)
    w.add_tensor("f16", f16)
    w.add_tensor("q4k.expected", dequantize(q4, Q.Q4_K).astype(np.float32))
    w.add_tensor("q6k.expected", dequantize(q6, Q.Q6_K).astype(np.float32))
    w.add_tensor("f16.expected", f16.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    (OUT / "bad-magic.gguf").write_bytes(b"GGUX" + bytes(60))


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope_neox(x: np.ndarray, theta: float) -> np.ndarray:
    """llama.cpp NEOX rope (Qwen2, Dream): rotate (i, i + d/2)."""
    seq, heads, dim = x.shape
    half = dim // 2
    pos = np.arange(seq, dtype=np.float64)[:, None]
    inv = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    ang = pos * inv[None, :]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    x0 = x[..., :half].astype(np.float64)
    x1 = x[..., half:].astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:] = x0 * sin + x1 * cos
    return out.astype(np.float32)


def rope_norm(x: np.ndarray, theta: float) -> np.ndarray:
    """llama.cpp NORM rope: rotate adjacent pairs (2i, 2i+1)."""
    seq, heads, dim = x.shape
    pos = np.arange(seq, dtype=np.float64)[:, None]
    inv = theta ** (-np.arange(0, dim, 2, dtype=np.float64) / dim)
    ang = pos * inv[None, :]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    x0 = x[..., 0::2].astype(np.float64)
    x1 = x[..., 1::2].astype(np.float64)
    out = np.empty_like(x, dtype=np.float64)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out.astype(np.float32)


def tiny_llada() -> None:
    c = TINY
    h, f, v = c["hidden"], c["ffn"], c["vocab"]
    w = GGUFWriter(str(OUT / "tiny-llada.gguf"), "llada")
    w.add_uint32("llada.block_count", c["layers"])
    w.add_uint32("llada.context_length", 4096)
    w.add_uint32("llada.embedding_length", h)
    w.add_uint32("llada.feed_forward_length", f)
    w.add_uint32("llada.attention.head_count", c["heads"])
    w.add_float32("llada.attention.layer_norm_rms_epsilon", TINY_EPS)
    w.add_float32("llada.rope.freq_base", TINY_THETA)
    w.add_uint32("llada.rope.dimension_count", c["head_dim"])
    w.add_uint32("llada.vocab_size", v)
    w.add_bool("llada.attention.causal", False)
    w.add_uint32("tokenizer.ggml.mask_token_id", c["mask"])

    weights: dict[str, np.ndarray] = {}

    def q4(name: str, rows: int, cols: int, scale: float) -> None:
        raw = random_q4k(rows, cols, scale)
        w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=Q.Q4_K)
        weights[name] = dequantize(raw, Q.Q4_K).astype(np.float32)

    def q6(name: str, rows: int, cols: int, scale: float) -> None:
        raw = random_q6k(rows, cols, scale)
        w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=Q.Q6_K)
        weights[name] = dequantize(raw, Q.Q6_K).astype(np.float32)

    def f32(name: str, n: int) -> None:
        arr = RNG.uniform(0.8, 1.2, n).astype(np.float32)
        w.add_tensor(name, arr)
        weights[name] = arr

    q4("token_embd.weight", v, h, 0.02)
    for i in range(c["layers"]):
        p = f"blk.{i}."
        f32(p + "attn_norm.weight", h)
        q4(p + "attn_q.weight", h, h, 0.004)
        q4(p + "attn_k.weight", h, h, 0.004)
        q6(p + "attn_v.weight", h, h, 0.0004)
        q4(p + "attn_output.weight", h, h, 0.004)
        f32(p + "ffn_norm.weight", h)
        q4(p + "ffn_gate.weight", f, h, 0.004)
        q4(p + "ffn_up.weight", f, h, 0.004)
        q6(p + "ffn_down.weight", h, f, 0.0003)
    f32("output_norm.weight", h)
    q6("output.weight", v, h, 0.0005)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # Reference forward in float32/float64 numpy, no activation quantization.
    x = weights["token_embd.weight"][TINY_TOKENS]
    seq = len(TINY_TOKENS)
    nh, hd = c["heads"], c["head_dim"]
    for i in range(c["layers"]):
        p = f"blk.{i}."
        hn = rmsnorm(x, weights[p + "attn_norm.weight"], TINY_EPS)
        q = (hn @ weights[p + "attn_q.weight"].T).reshape(seq, nh, hd)
        k = (hn @ weights[p + "attn_k.weight"].T).reshape(seq, nh, hd)
        vv = (hn @ weights[p + "attn_v.weight"].T).reshape(seq, nh, hd)
        q = rope_norm(q, TINY_THETA)
        k = rope_norm(k, TINY_THETA)
        att = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", att, vv).reshape(seq, h)
        x = x + o @ weights[p + "attn_output.weight"].T
        hn = rmsnorm(x, weights[p + "ffn_norm.weight"], TINY_EPS)
        g = hn @ weights[p + "ffn_gate.weight"].T
        u = hn @ weights[p + "ffn_up.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ weights[p + "ffn_down.weight"].T
    x = rmsnorm(x, weights["output_norm.weight"], TINY_EPS)
    logits = (x @ weights["output.weight"].T).astype(np.float32)
    logits.tofile(OUT / "tiny-llada.expected.f32")
    print("tiny logits std", float(logits.std()), "argmax", logits.argmax(-1).tolist())


DREAM: dict[str, int] = dict(vocab=512, hidden=256, heads=4, kv_heads=2, head_dim=64, ffn=512, layers=2, mask=510)
DREAM_EPS = 1e-6
DREAM_THETA = 1000000.0


def tiny_dream() -> None:
    """Dream/Qwen2 layout: GQA, Q/K/V bias, NEOX rope, shifted logits."""
    c = DREAM
    h, f, v = c["hidden"], c["ffn"], c["vocab"]
    kvd = c["kv_heads"] * c["head_dim"]
    w = GGUFWriter(str(OUT / "tiny-dream.gguf"), "dream")
    w.add_uint32("dream.block_count", c["layers"])
    w.add_uint32("dream.context_length", 4096)
    w.add_uint32("dream.embedding_length", h)
    w.add_uint32("dream.feed_forward_length", f)
    w.add_uint32("dream.attention.head_count", c["heads"])
    w.add_uint32("dream.attention.head_count_kv", c["kv_heads"])
    w.add_float32("dream.attention.layer_norm_rms_epsilon", DREAM_EPS)
    w.add_float32("dream.rope.freq_base", DREAM_THETA)
    w.add_uint32("dream.vocab_size", v)
    w.add_bool("diffusion.shift_logits", True)
    w.add_uint32("tokenizer.ggml.mask_token_id", c["mask"])
    weights: dict[str, np.ndarray] = {}

    def q4(name: str, rows: int, cols: int, scale: float) -> None:
        raw = random_q4k(rows, cols, scale)
        w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=Q.Q4_K)
        weights[name] = dequantize(raw, Q.Q4_K).astype(np.float32)

    def q6(name: str, rows: int, cols: int, scale: float) -> None:
        raw = random_q6k(rows, cols, scale)
        w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=Q.Q6_K)
        weights[name] = dequantize(raw, Q.Q6_K).astype(np.float32)

    def f32(name: str, n: int, lo: float = 0.8, hi: float = 1.2) -> None:
        arr = RNG.uniform(lo, hi, n).astype(np.float32)
        w.add_tensor(name, arr)
        weights[name] = arr

    q4("token_embd.weight", v, h, 0.02)
    for i in range(c["layers"]):
        p = f"blk.{i}."
        f32(p + "attn_norm.weight", h)
        q4(p + "attn_q.weight", h, h, 0.004)
        # Biases large enough that dropping them moves the logits well past
        # the test tolerance; at +-0.5 a missing bias went unnoticed.
        f32(p + "attn_q.bias", h, -3.0, 3.0)
        q4(p + "attn_k.weight", kvd, h, 0.004)
        f32(p + "attn_k.bias", kvd, -3.0, 3.0)
        q6(p + "attn_v.weight", kvd, h, 0.0004)
        f32(p + "attn_v.bias", kvd, -2.0, 2.0)
        q4(p + "attn_output.weight", h, h, 0.004)
        f32(p + "ffn_norm.weight", h)
        q4(p + "ffn_gate.weight", f, h, 0.004)
        q4(p + "ffn_up.weight", f, h, 0.004)
        q6(p + "ffn_down.weight", h, f, 0.0003)
    f32("output_norm.weight", h)
    q6("output.weight", v, h, 0.0005)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    x = weights["token_embd.weight"][TINY_TOKENS]
    seq = len(TINY_TOKENS)
    nh, nkv, hd = c["heads"], c["kv_heads"], c["head_dim"]
    for i in range(c["layers"]):
        p = f"blk.{i}."
        hn = rmsnorm(x, weights[p + "attn_norm.weight"], DREAM_EPS)
        q = (hn @ weights[p + "attn_q.weight"].T + weights[p + "attn_q.bias"]).reshape(seq, nh, hd)
        k = (hn @ weights[p + "attn_k.weight"].T + weights[p + "attn_k.bias"]).reshape(seq, nkv, hd)
        vv = (hn @ weights[p + "attn_v.weight"].T + weights[p + "attn_v.bias"]).reshape(seq, nkv, hd)
        q = rope_neox(q, DREAM_THETA)
        k = rope_neox(k, DREAM_THETA)
        k = np.repeat(k, nh // nkv, axis=1)
        vv = np.repeat(vv, nh // nkv, axis=1)
        att = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd)
        att = np.exp(att - att.max(-1, keepdims=True))
        att /= att.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", att, vv).reshape(seq, h)
        x = x + o @ weights[p + "attn_output.weight"].T
        hn = rmsnorm(x, weights[p + "ffn_norm.weight"], DREAM_EPS)
        g = hn @ weights[p + "ffn_gate.weight"].T
        u = hn @ weights[p + "ffn_up.weight"].T
        x = x + ((g / (1 + np.exp(-g))) * u) @ weights[p + "ffn_down.weight"].T
    x = rmsnorm(x, weights["output_norm.weight"], DREAM_EPS)
    logits = (x @ weights["output.weight"].T).astype(np.float32)
    logits.tofile(OUT / "tiny-dream.expected.f32")
    print("tiny dream logits std", float(logits.std()), "argmax", logits.argmax(-1).tolist())


def malformed() -> None:
    """Small, reproducible rejection inputs; offsets come from gguf's reader."""
    out = OUT / "malformed"
    out.mkdir(parents=True, exist_ok=True)
    base_path = out / "base.gguf"
    w = GGUFWriter(str(base_path), "test")
    w.add_array("test.array", [1, 2, 3])
    w.add_tensor("f32", np.arange(8, dtype=np.float32))
    # A separate stream keeps every existing fixture byte-for-byte unchanged.
    rng = np.random.default_rng(330032)
    q4 = rng.integers(0, 256, (1, Q4K_BYTES), dtype=np.uint8)
    q4[:, :4] = f16_bytes(np.array([0.01, 0.02]))
    w.add_tensor("q4k", q4, raw_shape=q4.shape, raw_dtype=Q.Q4_K)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    base = base_path.read_bytes()
    reader = GGUFReader(str(base_path))

    def part_offset(field, index: int) -> int:
        return int(field.offset + sum(p.nbytes for p in field.parts[:index]))

    def patch(name: str, offset: int, width: int, value: int, old: int) -> None:
        data = bytearray(base)
        assert 0 <= offset <= len(data) - width
        assert int.from_bytes(data[offset:offset + width], "little") == old
        data[offset:offset + width] = value.to_bytes(width, "little")
        assert data[:offset] == base[:offset]
        assert data[offset + width:] == base[offset + width:]
        (out / (name + ".gguf")).write_bytes(data)

    for name, cut in (("empty", 0), ("magic-cut", 3), ("header-12", 12),
                      ("header-20", 20), ("metadata-cut", 60)):
        (out / (name + ".gguf")).write_bytes(base[:cut])
    for version in (1, 4):
        patch(f"version-{version}", 4, 4, version, 3)
    for name, offset in (("tensors", 8), ("kv", 16)):
        for label, value in (("large", 1 << 21), ("negative", (1 << 64) - 1)):
            patch(f"{name}-{label}", offset, 8, value, 2)
    scalar = reader.fields["general.architecture"]
    array = reader.fields["test.array"]
    assert int(scalar.parts[2][0]) == 8  # GGUF string
    assert int(array.parts[2][0]) == 9  # GGUF array
    assert int(array.parts[3][0]) == 5  # int32 elements
    patch("key-length", int(scalar.offset), 8, 1 << 20, len("general.architecture"))
    patch("key-length-negative", int(scalar.offset), 8, 1 << 63, len("general.architecture"))
    patch("scalar-type", part_offset(scalar, 2), 4, 99, 8)
    patch("array-type", part_offset(array, 3), 4, 99, 5)
    patch("nested-array", part_offset(array, 3), 4, 9, 5)
    patch("array-count-wrapped", part_offset(array, 4), 8, (1 << 62) + 2, 3)

    for label, alignment in (("zero", 0), ("nonpower", 24), ("valid", 64),
                             ("type", 32), ("array", 32)):
        w = GGUFWriter(str(out / f"alignment-{label}.gguf"), "test")
        if label == "type":
            w.add_uint8("general.alignment", alignment)
        elif label == "array":
            w.add_array("general.alignment", [alignment])
        else:
            w.add_uint32("general.alignment", alignment)
        if label == "valid":
            w.data_alignment = alignment
        w.add_tensor("f32", np.arange(8, dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

    strings_path = out / "string-array.gguf"
    w = GGUFWriter(str(strings_path), "test")
    w.add_array("test.strings", ["a", "b"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    strings = strings_path.read_bytes()
    string_field = GGUFReader(str(strings_path)).fields["test.strings"]
    count_offset = part_offset(string_field, 4)
    assert int.from_bytes(strings[count_offset:count_offset + 8], "little") == 2
    for label, count in (("negative", (1 << 64) - 1), ("large", 1 << 62)):
        data = bytearray(strings)
        data[count_offset:count_offset + 8] = count.to_bytes(8, "little")
        (out / f"string-count-{label}.gguf").write_bytes(data)
    tensor = reader.tensors[0]
    assert tensor.name == "f32"
    assert int(tensor.field.parts[2][0]) == 1
    for dims in (0, 5):
        patch(f"dims-{dims}", part_offset(tensor.field, 2), 4, dims, 1)
    last = reader.tensors[-1]
    assert last.name == "q4k"
    cut = int(last.data_offset + last.n_bytes - 1)
    assert cut > reader.data_offset and cut < len(base)
    (out / "tensor-cut.gguf").write_bytes(base[:cut])
    # Valid empty directory: the last uint64 read ends exactly at EOF, no padding.
    (out / "exact-eof.gguf").write_bytes(b"GGUF" + (3).to_bytes(4, "little") + bytes(16))


def dequant_types() -> None:
    """Reference-only types, independent of the existing fixture RNG stream."""
    rng = np.random.default_rng(340034)
    w = GGUFWriter(str(OUT / "dequant-types.gguf"), "test")
    q8 = quantize(rng.standard_normal((4, 96)).astype(np.float32), Q.Q8_0)
    bf16 = quantize(rng.standard_normal((3, 40)).astype(np.float32), Q.BF16)
    q5 = rng.integers(0, 256, (8, 176), dtype=np.uint8)
    q5[:, :2] = f16_bytes(rng.uniform(0.002, 0.01, 8)).reshape(8, 2)
    q5[:, 2:4] = f16_bytes(rng.uniform(0.002, 0.01, 8)).reshape(8, 2)
    q5 = q5.reshape(4, 352)
    # Drawn last so the tensors above keep the bytes they had before Q5_0 joined.
    q50 = quantize(rng.standard_normal((4, 96)).astype(np.float32), Q.Q5_0)
    for name, data, kind in (("q8_0", q8, Q.Q8_0), ("bf16", bf16, Q.BF16),
                             ("q5k", q5, Q.Q5_K), ("q5_0", q50, Q.Q5_0)):
        w.add_tensor(name, data, raw_shape=data.shape, raw_dtype=kind)
        w.add_tensor(name + ".expected", dequantize(data, kind).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def metadata_cases() -> None:
    """Metadata-only ModelConfig inputs; no model weights or random draws."""
    out = OUT / "metadata"
    out.mkdir(parents=True, exist_ok=True)
    required = {
        "llada.block_count": 2,
        "llada.embedding_length": 512,
        "llada.attention.head_count": 4,
        "llada.feed_forward_length": 1024,
        "llada.attention.layer_norm_rms_epsilon": 1e-5,
        "tokenizer.ggml.mask_token_id": 511,
    }

    def write(name: str, updates=None, omit=None) -> None:
        fields = {**required, "llada.vocab_size": 512, **(updates or {})}
        if omit:
            del fields[omit]
        w = GGUFWriter(str(out / (name + ".gguf")), "llada")
        for key, value in fields.items():
            if isinstance(value, str):
                w.add_string(key, value)
            elif isinstance(value, float):
                w.add_float32(key, value)
            else:
                w.add_uint32(key, value)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

    for key in required:
        write("missing-" + key, omit=key)
    write("kv-heads", {"llada.attention.head_count_kv": 3})
    write("hidden-size", {"llada.embedding_length": 513})
    write("rope-width", {"llada.rope.dimension_count": 64})
    write("key-length", {"llada.attention.key_length": 64})
    write("rope-type", {"llada.rope.scaling.type": "yarn"})
    write("rope-factor", {"llada.rope.scaling.factor": 4.0})
    write("rope-width-type", {"llada.rope.dimension_count": "128"})
    write("rope-factor-type", {"llada.rope.scaling.factor": "1.0"})
    write("defaults")
    for kind in ("none", "linear"):
        write("compatible-" + kind, {"llada.rope.dimension_count": 128,
                                    "llada.attention.key_length": 128,
                                    "llada.rope.scaling.type": kind,
                                    "llada.rope.scaling.factor": 1.0})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kquants()
    tiny_llada()
    tiny_dream()
    malformed()
    dequant_types()
    unsupported_embedding()
    metadata_cases()
    for p in sorted(OUT.iterdir()):
        print(f"{p.stat().st_size:>9}  {p.relative_to(ROOT)}")


def unsupported_embedding() -> None:
    """Minimal model directory: construction must reject the embedding first."""
    w = GGUFWriter(str(OUT / "unsupported-embedding.gguf"), "llada")
    for key, value in {"block_count": 1, "embedding_length": 256,
                       "attention.head_count": 2, "feed_forward_length": 256,
                       "vocab_size": 1}.items():
        w.add_uint32("llada." + key, value)
    w.add_float32("llada.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_uint32("tokenizer.ggml.mask_token_id", 0)
    data = np.zeros((1, 66), dtype=np.uint8)
    w.add_tensor("token_embd.weight", data, raw_shape=data.shape, raw_dtype=Q.IQ2_XXS)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
