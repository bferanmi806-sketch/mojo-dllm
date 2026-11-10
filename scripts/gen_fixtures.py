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
  tests/fixtures/malformed/        one-byte-damaged GGUF headers and metadata, separate seed 330032
  tests/fixtures/metadata/         valid and invalid ModelConfig metadata
  tests/fixtures/entropy-rows.tsv  logits rows with numpy f64 argmax, entropy and inverse-CDF draws, seed 65065
  tests/fixtures/tiny-diffusion-gemma.gguf
                                   2-layer diffusion-gemma model (one sliding, one global, 4 experts), seed 610061
  tests/fixtures/diffusion-gemma/  that model's metadata and tensor directory with one thing wrong per file
  tests/fixtures/tiny-diffusion-gemma.*.case
                                   tokens and the numpy logits of scripts/dg_numpy.py for the forward pass
  tests/fixtures/gemma4-tiny/      a 279-token gemma4 vocabulary, a well-formed variant with EOS, and files with one thing wrong each (issue #62)
  tests/fixtures/gpt2-tiny/        a six-token byte-level BPE vocabulary with the ids a real one leaves out (issue #62)
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader, GGUFValueType, GGUFWriter
from gguf.quants import dequantize, quantize

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import spec  # noqa: E402  (bench/spec.py)

OUT = ROOT / "tests" / "fixtures"
SEED = 1234
# These three fixtures once drew from one generator seeded with SEED, in the order below, so each one's values
# depend on how many numbers the earlier ones drew. Each now starts its own generator at the position (in 64-bit
# draws) where the shared stream stood when it began, which reproduces every committed byte and no longer depends
# on the order the functions run in. The positions were found by stepping the generator until its state matched
# the state recorded after the previous function; a test regenerates the three in reverse order and compares.
KQUANTS_SKIP = 0
TINY_LLADA_SKIP = 548
TINY_DREAM_SKIP = 149284


def stream(skip: int) -> np.random.Generator:
    """A generator on the SEED stream, `skip` draws in."""
    gen = np.random.default_rng(SEED)
    gen.bit_generator.advance(skip)
    return gen


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


def random_q4k(rows: int, cols: int, d_scale: float, gen: np.random.Generator) -> np.ndarray:
    """Valid Q4_K bytes: small positive f16 d/dmin, random 6-bit scales, random nibbles, drawn from `gen`."""
    nb = rows * cols // QK_K
    blocks = np.zeros((nb, Q4K_BYTES), dtype=np.uint8)
    blocks[:, 0:2] = f16_bytes(gen.uniform(0.2, 1.0, nb) * d_scale).reshape(nb, 2)
    blocks[:, 2:4] = f16_bytes(gen.uniform(0.2, 1.0, nb) * d_scale * 4).reshape(nb, 2)
    blocks[:, 4:16] = gen.integers(0, 256, (nb, 12), dtype=np.uint8)
    blocks[:, 16:] = gen.integers(0, 256, (nb, 128), dtype=np.uint8)
    return blocks.reshape(rows, cols // QK_K * Q4K_BYTES)


def random_q6k(rows: int, cols: int, d_scale: float, gen: np.random.Generator) -> np.ndarray:
    nb = rows * cols // QK_K
    blocks = np.zeros((nb, Q6K_BYTES), dtype=np.uint8)
    blocks[:, 0:192] = gen.integers(0, 256, (nb, 192), dtype=np.uint8)
    blocks[:, 192:208] = gen.integers(-64, 64, (nb, 16)).astype(np.int8).view(np.uint8)
    blocks[:, 208:210] = f16_bytes(gen.uniform(0.2, 1.0, nb) * d_scale).reshape(nb, 2)
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

    rng = stream(KQUANTS_SKIP)
    q4 = random_q4k(4, 512, 0.01, rng)
    q6 = random_q6k(4, 512, 0.01, rng)
    f16 = rng.standard_normal((2, 64)).astype(np.float16)
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
    rng = stream(TINY_LLADA_SKIP)
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
        raw = random_q4k(rows, cols, scale, rng)
        w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=Q.Q4_K)
        weights[name] = dequantize(raw, Q.Q4_K).astype(np.float32)

    def q6(name: str, rows: int, cols: int, scale: float) -> None:
        raw = random_q6k(rows, cols, scale, rng)
        w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=Q.Q6_K)
        weights[name] = dequantize(raw, Q.Q6_K).astype(np.float32)

    def f32(name: str, n: int) -> None:
        arr = rng.uniform(0.8, 1.2, n).astype(np.float32)
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
    rng = stream(TINY_DREAM_SKIP)
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
        raw = random_q4k(rows, cols, scale, rng)
        w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=Q.Q4_K)
        weights[name] = dequantize(raw, Q.Q4_K).astype(np.float32)

    def q6(name: str, rows: int, cols: int, scale: float) -> None:
        raw = random_q6k(rows, cols, scale, rng)
        w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=Q.Q6_K)
        weights[name] = dequantize(raw, Q.Q6_K).astype(np.float32)

    def f32(name: str, n: int, lo: float = 0.8, hi: float = 1.2) -> None:
        arr = rng.uniform(lo, hi, n).astype(np.float32)
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
    # A length just under 2^63: the read position plus the length wraps negative (#194).
    patch("key-length-int64max", int(scalar.offset), 8, 0x7FFFFFFFFFFFFFF0, len("general.architecture"))
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
    tensor_directory_cases(out)


GGML_F32, GGML_Q8_0, GGML_Q4_K, GGML_IQ2_XXS = 0, 8, 12, 16
GGUF_STRING, GGUF_UINT32 = 8, 4


def raw_gguf(tensors, kvs=(), data=b"", alignment=32) -> bytes:
    """A GGUF v3 file written field by field, for directories that GGUFWriter cannot produce: dimensions and
    offsets that are absurd or negative as an Int, and repeated names. `tensors` holds (name, dims, ggml type,
    offset) and `kvs` holds (key, GGUF value type, payload bytes); the directory is followed by padding to
    `alignment` and `data`."""

    def string(text: str | bytes) -> bytes:
        raw = text if isinstance(text, bytes) else text.encode()
        return len(raw).to_bytes(8, "little") + raw

    out = bytearray(b"GGUF" + (3).to_bytes(4, "little"))
    out += len(tensors).to_bytes(8, "little") + len(kvs).to_bytes(8, "little")
    for key, kind, payload in kvs:
        out += string(key) + kind.to_bytes(4, "little") + payload
    for name, dims, ggml_type, offset in tensors:
        out += string(name) + len(dims).to_bytes(4, "little")
        for dim in dims:
            out += dim.to_bytes(8, "little")
        out += ggml_type.to_bytes(4, "little") + offset.to_bytes(8, "little")
    out += bytes(-len(out) % alignment)
    return bytes(out + data)


def string_array_payload(items: list[bytes]) -> bytes:
    """The payload of a GGUF array of strings: element type, count, then each string with its length."""
    out = bytearray(GGUF_STRING.to_bytes(4, "little") + len(items).to_bytes(8, "little"))
    for item in items:
        out += len(item).to_bytes(8, "little") + item
    return bytes(out)


def string_cases(out: pathlib.Path) -> None:
    """GGUF strings that are not valid UTF-8, in a key, a value, a token text and a tensor name (#225), and a
    control with strings of every width. The lone byte is 0xE9, a Latin-1 e acute: it begins a three-byte sequence
    that the next byte does not complete."""
    arch = ("general.architecture", GGUF_STRING, (4).to_bytes(8, "little") + b"test")
    f32 = ("f32", [8], GGML_F32, 0)
    data = np.arange(8, dtype="<f4").tobytes()
    tokens = "tokenizer.ggml.tokens"
    bad = b"caf\xe9"

    def text(value: bytes) -> bytes:
        return len(value).to_bytes(8, "little") + value

    def write(name: str, tensors=(f32,), kvs=(arch,)) -> None:
        (out / (name + ".gguf")).write_bytes(raw_gguf(tensors, kvs, data))

    # The first entry is the architecture, so the bad key is entry 1 and its bytes start at file offset 76.
    write("string-invalid-key", kvs=(arch, (bad, GGUF_STRING, text(b"x"))))
    write("string-invalid-value", kvs=(arch, ("test.str", GGUF_STRING, text(bad)),
                                       (tokens, 9, string_array_payload([b"a", b"b", b"c"]))))
    write("string-invalid-token", kvs=(arch, (tokens, 9, string_array_payload([b"a", b"b" + bad, b"c"]))))
    write("string-invalid-tensor-name", tensors=((b"w\xe9ight", [8], GGML_F32, 0),))
    # Valid strings of one, two, three and four bytes per character, in the same four places.
    wide = "a \u00e9 \u20ac \U0001f600"
    write("string-valid", tensors=((wide, [8], GGML_F32, 0),),
          kvs=(arch, (wide, GGUF_STRING, text(wide.encode())), ("test.str", GGUF_STRING, text(wide.encode())),
               (tokens, 9, string_array_payload([wide.encode(), "\u00e9".encode(), b"c"]))))


def tensor_directory_cases(out: pathlib.Path) -> None:
    """One file per refusal of the tensor directory (#195), each differing from `tensors-control` in one field."""
    arch = ("general.architecture", GGUF_STRING, (4).to_bytes(8, "little") + b"test")
    f32_data = np.arange(8, dtype="<f4").tobytes()  # the 32 bytes of an F32 tensor of 8 elements
    f32 = ("f32", [8], GGML_F32, 0)

    def write(name: str, tensors, data=f32_data, kvs=(arch,)) -> None:
        (out / (name + ".gguf")).write_bytes(raw_gguf(tensors, kvs, data))

    # The control is read back by the reference reader, so a refusal below cannot come from a bad writer.
    write("tensors-control", [f32])
    control = out / "tensors-control.gguf"
    ref = GGUFReader(str(control))
    assert [t.name for t in ref.tensors] == ["f32"] and ref.tensors[0].data.tolist() == list(range(8))
    assert ref.alignment == 32 and ref.data_offset % 32 == 0

    # Offsets. The values are chosen so that each guard is the only thing that refuses its file: -32 is a
    # multiple of the alignment and lands inside the file, 2^63 - 32 is a multiple of it too.
    write("tensor-offset-negative", [("f32", [8], GGML_F32, (1 << 64) - 32)])
    write("tensor-offset-int64max", [("f32", [8], GGML_F32, (1 << 63) - 32)])
    write("tensor-offset-unaligned", [("f32", [8], GGML_F32, 16)])
    # Dimensions and byte counts.
    write("tensor-dim-zero", [("f32", [8, 0], GGML_F32, 0)])
    write("tensor-dim-negative", [("f32", [8, 1 << 63], GGML_F32, 0)])
    write("tensor-dim-wraps", [("q4k", [4096, 1 << 61], GGML_Q4_K, 0)])  # 2304 bytes per row x 2^61 = 0 mod 2^64
    # The sizes wrap to 32 bytes, the data the file holds, so a check that multiplies first accepts them.
    write("tensor-bytes-wrap", [("f32", [(1 << 62) + 8], GGML_F32, 0)])  # 4 bytes x (2^62 + 8) = 32 mod 2^64
    write("tensor-rows-wrap", [("f32", [2, (1 << 61) + 4], GGML_F32, 0)])  # 8 bytes x (2^61 + 4) rows = 32 mod 2^64
    write("tensor-row-partial-block", [("q8", [16, 2], GGML_Q8_0, 0)], data=bytes(68))  # 16 of 32 per block
    # A type this reader cannot decode stays listed, with no byte count, but its offset is still checked.
    iq2 = bytes(66)
    write("tensor-unsupported-type", [("iq2", [256], GGML_IQ2_XXS, 0)], data=iq2)
    write("tensor-unsupported-past-end", [("iq2", [256], GGML_IQ2_XXS, 1 << 20)], data=iq2)
    # Repeats: the first of two equal names or keys used to win without a word.
    write("tensor-name-duplicate", [f32, ("f32", [8], GGML_F32, 32)], data=f32_data * 2)
    two = (1).to_bytes(4, "little")
    write("kv-key-duplicate", [f32], kvs=(arch, ("test.value", GGUF_UINT32, two), ("test.value", GGUF_UINT32, two)))
    write("kv-keys-distinct", [f32], kvs=(arch, ("test.value", GGUF_UINT32, two), ("test.other", GGUF_UINT32, two)))
    string_cases(out)


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
            elif isinstance(value, bool):
                w.add_bool(key, value)
            elif value < 0:
                w.add_int32(key, value)
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
    # Every optional key that has a default is set, so `inspect` lists none (#204).
    write("keys-set", {"llada.attention.head_count_kv": 4, "llada.context_length": 4096,
                       "llada.rope.freq_base": 500000.0, "diffusion.shift_logits": False})
    # Hyperparameters that are zero, negative or not finite (#199), one file per key, named as the
    # DiffusionGemma ones are.
    for key in ("llada.block_count", "llada.embedding_length", "llada.attention.head_count",
                "llada.attention.head_count_kv", "llada.feed_forward_length", "llada.vocab_size"):
        write("zero-" + key, {key: 0})
    write("negative-llada.attention.head_count", {"llada.attention.head_count": -2})
    write("negative-llada.attention.head_count_kv", {"llada.attention.head_count_kv": -1})
    for key in ("llada.attention.layer_norm_rms_epsilon", "llada.rope.freq_base"):
        for label, value in (("zero", 0.0), ("negative", -1e-5), ("nan", float("nan")), ("inf", float("inf"))):
            write(f"{label}-{key}", {key: value})
    for kind in ("none", "linear"):
        write("compatible-" + kind, {"llada.rope.dimension_count": 128,
                                    "llada.attention.key_length": 128,
                                    "llada.rope.scaling.type": kind,
                                    "llada.rope.scaling.factor": 1.0})


def llada_shape_cases() -> None:
    """LLaDA files whose token_embd or output tensor disagrees with the vocabulary or the hidden size (#198).

    Hidden size 256, vocabulary 8, one layer; the two tensors hold zero Q4_K blocks and no layer tensors follow,
    because the check must refuse the file before any loader asks for one.
    """
    out = OUT / "metadata"
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, embd: tuple[int, int], output: tuple[int, int] | None, vocab_key: bool = True) -> None:
        w = GGUFWriter(str(out / (name + ".gguf")), "llada")
        w.add_uint32("llada.block_count", 1)
        w.add_uint32("llada.embedding_length", 256)
        w.add_uint32("llada.attention.head_count", 2)
        w.add_uint32("llada.feed_forward_length", 256)
        w.add_float32("llada.attention.layer_norm_rms_epsilon", 1e-5)
        w.add_uint32("tokenizer.ggml.mask_token_id", 7)
        if vocab_key:
            w.add_uint32("llada.vocab_size", 8)
        for tensor, shape in (("token_embd.weight", embd), ("output.weight", output)):
            if shape is None:
                continue
            width, rows = shape
            raw = np.zeros((rows, width // QK_K * Q4K_BYTES), dtype=np.uint8)
            w.add_tensor(tensor, raw, raw_shape=raw.shape, raw_dtype=Q.Q4_K)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

    write("shapes-valid", (256, 8), (256, 8))
    write("embd-rows-short", (256, 6), (256, 8))
    write("embd-rows-long", (256, 10), (256, 8))
    write("output-rows-short", (256, 8), (256, 6))
    write("embd-width", (512, 8), (256, 8))
    write("vocab-from-output", (256, 6), (256, 8), vocab_key=False)
    write("output-absent", (256, 8), None)


GEMMA_ARCH = "diffusion-gemma"
GEMMA_KEY = GEMMA_ARCH + "."
# The real file (diffusiongemma-26B-A4B-it-Q4_K_M.gguf) has 30 blocks, hidden size 2816, 16 heads and
# global layers 5, 11, 17, 23, 29. This one keeps its structure at a size a test can load: one sliding
# and one global layer, a different K width on each, and a head size that differs by kind.
GEMMA: dict = dict(
    layers=2,
    sliding=[True, False],
    kv=[4, 1],
    hidden=256,
    heads=8,
    key_swa=64,
    key=128,
    window=128,
    ffn=512,
    experts=4,
    used=2,
    expert_ffn=64,
    vocab=512,
    mask=4,
    canvas=16,
)
GEMMA_SEED = 610061
# GGUF value kind, in the form the writer is called with, for each metadata value of a case.
GEMMA_BLOCKS = {"Q4_K": (256, Q4K_BYTES), "Q6_K": (256, Q6K_BYTES), "Q5_0": (32, 22), "Q8_0": (32, 34)}


def gemma_keys(c: dict) -> dict[str, tuple[str, object]]:
    """The metadata of the real file (44 keys) that the reader reads, as (kind, value)."""
    a = GEMMA_KEY
    return {
        a + "block_count": ("u32", c["layers"]),
        a + "context_length": ("u32", 8192),
        a + "embedding_length": ("u32", c["hidden"]),
        a + "feed_forward_length": ("u32", c["ffn"]),
        a + "attention.head_count": ("u32", c["heads"]),
        a + "attention.head_count_kv": ("i32[]", c["kv"]),
        a + "rope.freq_base": ("f32", 1000000.0),
        a + "rope.freq_base_swa": ("f32", 10000.0),
        a + "attention.layer_norm_rms_epsilon": ("f32", 1e-6),
        a + "expert_count": ("u32", c["experts"]),
        a + "expert_used_count": ("u32", c["used"]),
        a + "attention.key_length": ("u32", c["key"]),
        a + "attention.value_length": ("u32", c["key"]),
        a + "final_logit_softcapping": ("f32", 30.0),
        a + "attention.sliding_window": ("u32", c["window"]),
        a + "attention.shared_kv_layers": ("u32", 0),
        a + "embedding_length_per_layer_input": ("u32", 0),
        a + "attention.sliding_window_pattern": ("bool[]", c["sliding"]),
        a + "attention.key_length_swa": ("u32", c["key_swa"]),
        a + "attention.value_length_swa": ("u32", c["key_swa"]),
        a + "expert_feed_forward_length": ("u32", c["expert_ffn"]),
        a + "rope.dimension_count": ("u32", c["key"]),
        a + "rope.dimension_count_swa": ("u32", c["key_swa"]),
        a + "attention.causal": ("bool", False),
        "diffusion.canvas_length": ("u32", c["canvas"]),
        "tokenizer.ggml.mask_token_id": ("u32", c["mask"]),
    }


def gemma_tensors(c: dict, full: bool) -> dict[str, tuple[tuple[int, ...], str]]:
    """Tensor name -> (shape with the innermost dimension first, GGML type), as in the real file.

    `full` is every tensor of the real file. Otherwise only those the reader looks at: the embedding, the
    rope table, and each block's K, V (sliding blocks), and expert tensors."""
    h, f, ex, ef = c["hidden"], c["ffn"], c["experts"], c["expert_ffn"]
    t: dict[str, tuple[tuple[int, ...], str]] = {"token_embd.weight": ((h, c["vocab"]), "Q6_K")}
    if full:
        t["output_norm.weight"] = ((h,), "F32")
    if not all(c["sliding"]):
        t["rope_freqs.weight"] = ((c["key"] // 2,), "F32")
    if full:
        t["self_cond_pre_norm.weight"] = ((h,), "F32")
        t["self_cond_gate.weight"] = ((h, f), "Q4_K")
        t["self_cond_up.weight"] = ((h, f), "Q4_K")
        t["self_cond_down.weight"] = ((f, h), "Q5_0")
    for i in range(c["layers"]):
        p = f"blk.{i}."
        sliding, kv = c["sliding"][i], c["kv"][i]
        size = c["key_swa"] if sliding else c["key"]
        down = "Q5_0" if i % 2 == 0 else "Q8_0"
        if full:
            t[p + "attn_norm.weight"] = ((h,), "F32")
            t[p + "attn_q.weight"] = ((h, c["heads"] * size), "Q4_K")
        t[p + "attn_k.weight"] = ((h, kv * size), "Q4_K")
        if sliding:
            t[p + "attn_v.weight"] = ((h, kv * size), "Q6_K" if i % 2 == 0 else "Q4_K")
        if full:
            t[p + "attn_output.weight"] = ((c["heads"] * size, h), "Q4_K")
            t[p + "attn_q_norm.weight"] = ((size,), "F32")
            t[p + "attn_k_norm.weight"] = ((size,), "F32")
            t[p + "post_attention_norm.weight"] = ((h,), "F32")
            t[p + "layer_output_scale.weight"] = ((1,), "F32")
            t[p + "enc_layer_output_scale.weight"] = ((1,), "F32")
            t[p + "ffn_norm.weight"] = ((h,), "F32")
            t[p + "ffn_gate.weight"] = ((h, f), "Q4_K")
            t[p + "ffn_up.weight"] = ((h, f), "Q4_K")
            t[p + "ffn_down.weight"] = ((f, h), down)
            for extra in ("post_ffw_norm", "post_ffw_norm_1", "post_ffw_norm_2", "pre_ffw_norm_2"):
                t[p + extra + ".weight"] = ((h,), "F32")
            t[p + "ffn_gate_inp.weight"] = ((h, ex), "F32")
            t[p + "ffn_gate_inp.scale"] = ((h,), "F32")
        t[p + "ffn_gate_up_exps.weight"] = ((h, 2 * ef, ex), "Q4_K")
        t[p + "ffn_down_exps.weight"] = ((ef, h, ex), down)
        if full:
            t[p + "ffn_down_exps.scale"] = ((ex,), "F32")
    return t


def gemma_tensor_data(rng: np.random.Generator, ne: tuple[int, ...], kind: str, zero: bool):
    """The array and GGML type to hand to the writer for one tensor. `zero` writes all-zero bytes, which
    the reader's shape checks accept and git stores for nothing."""
    shape = tuple(reversed(ne))
    if kind == "F32":
        arr = np.zeros(shape, np.float32) if zero else rng.uniform(0.8, 1.2, shape).astype(np.float32)
        return arr, None
    block, nbytes = GEMMA_BLOCKS[kind]
    assert ne[0] % block == 0, (ne, kind)
    byte_shape = shape[:-1] + (ne[0] // block * nbytes,)
    rows = int(np.prod(shape[:-1]))
    if zero:
        return np.zeros(byte_shape, np.uint8), getattr(Q, kind)
    if kind == "Q4_K":
        return random_q4k(rows, ne[0], 0.004, rng).reshape(byte_shape), Q.Q4_K
    if kind == "Q6_K":
        return random_q6k(rows, ne[0], 0.0004, rng).reshape(byte_shape), Q.Q6_K
    return quantize(rng.standard_normal(shape).astype(np.float32), getattr(Q, kind)), getattr(Q, kind)


def write_gemma(
    path: pathlib.Path,
    *,
    c: dict | None = None,
    keys: dict | None = None,
    drop: tuple[str, ...] = (),
    shapes: dict | None = None,
    types: dict | None = None,
    omit: tuple[str, ...] = (),
    add: dict | None = None,
    tensors: str = "thin",
    rng: np.random.Generator | None = None,
) -> None:
    """One diffusion-gemma file. `keys` replaces metadata values and `drop` removes keys; `shapes` and `types`
    replace tensor shapes and GGML types, `omit` removes tensors and `add` adds (name -> (shape, type)).
    `tensors` is "full" (every
    tensor of the real file, random weights), "zeros" (every tensor, zero bytes: enough for the constructor's
    refusals, and git stores it for nothing), "thin" (what the reader looks at, zero bytes) or "none"."""
    c = c or GEMMA
    meta = {**gemma_keys(c), **(keys or {})}
    for key in drop:
        del meta[key]
    w = GGUFWriter(str(path), GEMMA_ARCH)
    w.add_name("tiny-diffusion-gemma")
    for key, (kind, value) in meta.items():
        if kind == "u32":
            w.add_uint32(key, value)
        elif kind == "f32":
            w.add_float32(key, value)
        elif kind == "bool":
            w.add_bool(key, value)
        elif kind == "str":
            w.add_string(key, value)
        elif kind == "f64[]":
            w.add_key_value(key, value, GGUFValueType.ARRAY, sub_type=GGUFValueType.FLOAT64)
        else:
            assert kind in ("i32[]", "bool[]", "f32[]", "str[]"), kind
            w.add_array(key, value)
    spec = {} if tensors == "none" else gemma_tensors(c, tensors in ("full", "zeros"))
    for name, shape in (shapes or {}).items():
        assert name in spec, name
        spec[name] = (shape, spec[name][1])
    for name, kind in (types or {}).items():
        assert name in spec, name
        spec[name] = (spec[name][0], kind)
    for name in omit:
        del spec[name]
    spec.update(add or {})
    rng = rng or np.random.default_rng(GEMMA_SEED)
    for name, (ne, kind) in spec.items():
        data, raw = gemma_tensor_data(rng, ne, kind, zero=tensors != "full")
        if raw is None:
            w.add_tensor(name, data)
        else:
            w.add_tensor(name, data, raw_shape=data.shape, raw_dtype=raw)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


# Metadata the reader requires, and the ones that must be positive.
GEMMA_REQUIRED = (
    "block_count", "embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv",
    "rope.freq_base", "rope.freq_base_swa", "attention.layer_norm_rms_epsilon", "expert_count",
    "expert_used_count", "attention.key_length", "attention.value_length", "attention.sliding_window",
    "attention.sliding_window_pattern", "attention.key_length_swa", "attention.value_length_swa",
    "expert_feed_forward_length", "rope.dimension_count", "rope.dimension_count_swa",
)
GEMMA_POSITIVE = (
    "block_count", "attention.head_count", "attention.key_length", "attention.key_length_swa",
    "attention.sliding_window", "expert_count", "expert_feed_forward_length",
)


def diffusion_gemma() -> None:
    """The DiffusionGemma reader's inputs: a tiny model that passes, and one file per refusal."""
    write_gemma(OUT / "tiny-diffusion-gemma.gguf", tensors="full", rng=np.random.default_rng(GEMMA_SEED))
    out = OUT / "diffusion-gemma"
    out.mkdir(parents=True, exist_ok=True)
    k = lambda name: GEMMA_KEY + name  # noqa: E731
    pattern, kv = k("attention.sliding_window_pattern"), k("attention.head_count_kv")

    # Metadata only: the reader refuses before it looks at a tensor.
    for name in GEMMA_REQUIRED:
        write_gemma(out / f"missing-{k(name)}.gguf", drop=(k(name),), tensors="none")
    for name in ("diffusion.canvas_length", "tokenizer.ggml.mask_token_id"):
        write_gemma(out / f"missing-{name}.gguf", drop=(name,), tensors="none")
    for name in GEMMA_POSITIVE:
        write_gemma(out / f"zero-{k(name)}.gguf", keys={k(name): ("u32", 0)}, tensors="none")
    write_gemma(out / "zero-diffusion.canvas_length.gguf", keys={"diffusion.canvas_length": ("u32", 0)}, tensors="none")
    cases: dict[str, dict] = {
        "pattern-length": dict(keys={pattern: ("bool[]", [True, False, False])}),
        "pattern-length-short": dict(keys={pattern: ("bool[]", [True])}),
        "kv-heads-length": dict(keys={kv: ("i32[]", [4, 1, 1])}),
        "kv-heads-length-short": dict(keys={kv: ("i32[]", [4])}),
        "pattern-type": dict(keys={pattern: ("i32[]", [1, 0])}),
        "kv-heads-scalar": dict(keys={kv: ("u32", 4)}),
        "kv-heads-type": dict(keys={kv: ("f32[]", [4.0, 1.0])}),
        "kv-heads-bool": dict(keys={kv: ("bool[]", [True, True])}),
        "kv-heads-f64": dict(keys={kv: ("f64[]", [4.0, 1.0])}),
        "kv-heads-strings": dict(keys={kv: ("str[]", ["4", "1"])}),
        "kv-heads-divisor": dict(keys={kv: ("i32[]", [3, 3])}),
        "kv-heads-negative": dict(keys={kv: ("i32[]", [4, -1])}),
        "kv-heads-zero": dict(keys={kv: ("i32[]", [0, 1])}),
        "shared-kv-layers": dict(keys={k("attention.shared_kv_layers"): ("u32", 2)}),
        "per-layer-input": dict(keys={k("embedding_length_per_layer_input"): ("u32", 64)}),
        "causal": dict(keys={k("attention.causal"): ("bool", True)}),
        "value-length": dict(keys={k("attention.value_length"): ("u32", 64)}),
        "value-length-wide": dict(keys={k("attention.value_length"): ("u32", 256)}),
        "value-length-swa": dict(keys={k("attention.value_length_swa"): ("u32", 32)}),
        "rope-width": dict(keys={k("rope.dimension_count"): ("u32", 64)}),
        "rope-width-swa": dict(keys={k("rope.dimension_count_swa"): ("u32", 32)}),
        "expert-used-over": dict(keys={k("expert_used_count"): ("u32", 5)}),
        "expert-used-zero": dict(keys={k("expert_used_count"): ("u32", 0)}),
        "softcap-negative": dict(keys={k("final_logit_softcapping"): ("f32", -1.0)}),
    }
    for name, case in cases.items():
        write_gemma(out / f"{name}.gguf", tensors="none", **case)
    # Three layers, so that two sliding layers can disagree.
    c3 = {**GEMMA, "layers": 3, "sliding": [True, True, False], "kv": [4, 2, 1]}
    write_gemma(out / "kv-heads-uneven.gguf", c=c3, tensors="none")
    c3 = {**GEMMA, "layers": 3, "sliding": [True, False, False], "kv": [4, 1, 2]}
    write_gemma(out / "kv-heads-uneven-global.gguf", c=c3, tensors="none")

    # Tensor directory checks. Each file differs from the passing control in one tensor.
    write_gemma(out / "thin-valid.gguf")
    # Every positive value at its smallest, and the optional keys absent: they must still pass.
    ones = {**GEMMA, "layers": 1, "sliding": [True], "kv": [1], "heads": 1, "key_swa": 1, "key": 1, "window": 1,
            "experts": 1, "used": 1, "expert_ffn": 1, "canvas": 1}
    write_gemma(
        out / "ones.gguf",
        c=ones,
        drop=(k("context_length"), k("final_logit_softcapping"), k("attention.shared_kv_layers"),
              k("embedding_length_per_layer_input"), k("attention.causal")),
        types={"blk.0.ffn_down_exps.weight": "F32"},  # one expert row of 1 is no Q5_0 block
    )
    # Files with one kind of layer pass: no rope table without a global layer, no attn_v without a sliding one.
    # Self-conditioning decodes the embedding table in groups of 8 rows (#227).
    gate = {"self_cond_gate.weight": ((GEMMA["hidden"], GEMMA["ffn"]), "Q4_K")}
    write_gemma(out / "vocab-500-self-cond.gguf", c={**GEMMA, "vocab": 500}, add=gate)
    write_gemma(out / "vocab-504-self-cond.gguf", c={**GEMMA, "vocab": 504}, add=gate)
    write_gemma(out / "vocab-500-no-self-cond.gguf", c={**GEMMA, "vocab": 500})
    write_gemma(out / "all-sliding.gguf", c={**GEMMA, "sliding": [True, True], "kv": [4, 4]})

    # The constructor's own refusals (#209): every tensor is present and the config accepts the file, so the
    # error comes from a loader or from the block-size check. Zero bytes are enough, the loaders read no weights.
    zeros = {"tensors": "zeros"}
    h = GEMMA["hidden"]
    write_gemma(out / "load-attn-q-q5_0.gguf", types={"blk.0.attn_q.weight": "Q5_0"}, **zeros)
    write_gemma(out / "load-attn-q-shrunk.gguf", shapes={"blk.0.attn_q.weight": (h, 256)}, **zeros)
    write_gemma(out / "load-attn-norm-short.gguf", shapes={"blk.0.attn_norm.weight": (h // 2,)}, **zeros)
    write_gemma(out / "load-attn-output-narrow.gguf", shapes={"blk.1.attn_output.weight": (256, h)}, **zeros)
    write_gemma(out / "load-ffn-down-q4_k.gguf", types={"blk.0.ffn_down.weight": "Q4_K"}, **zeros)
    write_gemma(out / "load-experts-gate-up-q5_0.gguf", types={"blk.1.ffn_gate_up_exps.weight": "Q5_0"}, **zeros)
    write_gemma(out / "load-self-cond-gate-q5_0.gguf", types={"self_cond_gate.weight": "Q5_0"}, **zeros)
    write_gemma(out / "load-self-cond-up-q5_0.gguf", types={"self_cond_up.weight": "Q5_0"}, **zeros)
    write_gemma(out / "load-no-self-cond.gguf", omit=("self_cond_pre_norm.weight", "self_cond_gate.weight", "self_cond_up.weight", "self_cond_down.weight"), **zeros)
    write_gemma(out / "load-self-cond-down-q4_k.gguf", types={"self_cond_down.weight": "Q4_K"}, **zeros)
    write_gemma(out / "load-self-cond-up-short.gguf", shapes={"self_cond_up.weight": (h, 256)}, **zeros)
    write_gemma(out / "load-output-norm-short.gguf", shapes={"output_norm.weight": (h // 2,)}, **zeros)
    write_gemma(out / "all-global.gguf", c={**GEMMA, "sliding": [False, False], "kv": [1, 1]})
    h, ex, ef = GEMMA["hidden"], GEMMA["experts"], GEMMA["expert_ffn"]
    tensor_cases: dict[str, dict] = {
        "global-with-attn-v": dict(add={"blk.1.attn_v.weight": ((h, 128), "Q4_K")}),
        "sliding-without-attn-v": dict(omit=("blk.0.attn_v.weight",)),
        "k-width-sliding": dict(shapes={"blk.0.attn_k.weight": (h, 512)}),
        "k-width-global": dict(shapes={"blk.1.attn_k.weight": (h, 256)}),
        "k-hidden": dict(shapes={"blk.0.attn_k.weight": (512, 256)}),
        "k-dims": dict(shapes={"blk.1.attn_k.weight": (h, 128, 2)}),
        "k-missing": dict(omit=("blk.1.attn_k.weight",)),
        "rope-freqs-missing": dict(omit=("rope_freqs.weight",)),
        "rope-freqs-width": dict(shapes={"rope_freqs.weight": (128,)}),
        "exps-gate-up": dict(shapes={"blk.0.ffn_gate_up_exps.weight": (h, ef, ex)}),
        "exps-gate-up-dims": dict(shapes={"blk.0.ffn_gate_up_exps.weight": (h, 2 * ef, ex, 1)}),
        "exps-down-count": dict(shapes={"blk.1.ffn_down_exps.weight": (ef, h, 8)}),
        "exps-down-width": dict(shapes={"blk.1.ffn_down_exps.weight": (32, h, ex)}),
        "exps-missing": dict(omit=("blk.1.ffn_gate_up_exps.weight",)),
        "embd-width": dict(shapes={"token_embd.weight": (512, 512)}),
    }
    for name, case in tensor_cases.items():
        write_gemma(out / f"{name}.gguf", **case)
    dg_cases()


# Cases the Mojo forward pass is checked on: the numpy pass of scripts/dg_numpy.py over the tiny model.
# File layout, little-endian 4-byte words: n_tokens, n_prompt, self-conditioning (0 or 1), 1 / previous
# temperature (f32), the number of traced layers, the tokens (i32), the previous canvas logits
# [C, vocab] (f32, only when self-conditioning is on), the expected canvas logits [C, vocab] (f32) and
# each traced layer's output rows [n_tokens, hidden] (f32).
DG_CASES = {
    # Prompt of 9: the sliding window (128) covers it. Prompt of 150: it does not, so the window clips
    # prompt rows and the canvas reaches only the last 127 prompt tokens in the sliding layer.
    "p9": dict(prompt=9, seed=7),
    "p150": dict(prompt=150, seed=8),
    # Prompt of 9 with self-conditioning: the previous step's logits are those of another canvas over the
    # same prompt, taken one denoising step back at temperature 0.8.
    "p9sc": dict(prompt=9, seed=7, prev_seed=99, prev_temp_inv=1.0 / spec.DG_FIRST_STEP_TEMP),
    # Self-conditioning with no prompt (the canvas alone) and past the sliding window (#229).
    "p0sc": dict(prompt=0, seed=9, prev_seed=98, prev_temp_inv=1.0 / spec.DG_FIRST_STEP_TEMP),
    "p150sc": dict(prompt=150, seed=8, prev_seed=99, prev_temp_inv=1.0 / spec.DG_FIRST_STEP_TEMP),
}


def dg_cases() -> None:
    import dg_numpy

    model = dg_numpy.Model(OUT / "tiny-diffusion-gemma.gguf")
    for name, case in DG_CASES.items():
        tokens = dg_numpy.tiny_tokens(model, case["prompt"], case["seed"])
        prev = None
        temp_inv = np.float32(1.0)
        if "prev_seed" in case:
            earlier = dg_numpy.tiny_tokens(model, case["prompt"], case["prev_seed"])
            prev = dg_numpy.forward(model, earlier, case["prompt"], quant=True)
            temp_inv = np.float32(case["prev_temp_inv"])
        trace: list = []
        expected = dg_numpy.forward(
            model, tokens, case["prompt"], quant=True, trace=trace, prev_logits=prev,
            prev_temp_inv=float(temp_inv),
        )
        keep = trace if case["prompt"] < model.window else []  # the long case checks the logits only
        blob = [np.array([len(tokens), case["prompt"], 0 if prev is None else 1], np.int32).tobytes(),
                temp_inv.tobytes(), np.array([len(keep)], np.int32).tobytes(),
                np.array(tokens, np.int32).tobytes()]
        if prev is not None:
            blob.append(prev.astype("<f4").tobytes())
        blob.append(expected.astype("<f4").tobytes())
        blob += [layer.astype("<f4").tobytes() for layer in keep]
        (OUT / f"tiny-diffusion-gemma.{name}.case").write_bytes(b"".join(blob))


def generate_all() -> None:
    """Every fixture, in dependency order: gemma4_tiny and gpt2_tiny come last because they read nothing earlier."""
    kquants()
    tiny_llada()
    tiny_dream()
    malformed()
    dequant_types()
    unsupported_embedding()
    metadata_cases()
    llada_shape_cases()
    utf8_cases()
    entropy_rows()
    diffusion_gemma()
    gemma4_tiny()
    gpt2_tiny()


MANIFEST = "gen_fixtures.manifest"


def sync(staging: pathlib.Path, final: pathlib.Path) -> list[str]:
    """Make `final` hold the files of `staging`, and remove the ones an earlier run made that this one does not.

    The files a run makes are listed in `final/gen_fixtures.manifest`. Other generators write into the same folder
    (scripts/gen_tokenizer_fixture.py, scripts/gen_moe_fixture.py), so only a file named by the previous manifest
    and absent from this run is removed. A file whose bytes already match is not touched, so a regeneration that
    changes nothing leaves no trace in `git status`. Returns the relative names it removed.
    """
    made = sorted(str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file() and p.name != MANIFEST)
    manifest = final / MANIFEST
    previous = manifest.read_text().splitlines() if manifest.is_file() else []
    removed = []
    for rel in previous:
        target = final / rel
        if rel not in made and target.is_file():
            target.unlink()
            removed.append(rel)
            for parent in target.parents:
                if parent == final or any(parent.iterdir()):
                    break
                parent.rmdir()
    for rel in made:
        target = final / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (staging / rel).read_bytes()
        if not target.is_file() or target.read_bytes() != data:
            target.write_bytes(data)
    text = "\n".join(made) + "\n"
    if not manifest.is_file() or manifest.read_text() != text:
        manifest.write_text(text)
    return removed


def main() -> None:
    """Generate into a staging folder and make OUT match it, so a fixture that is no longer made does not linger."""
    global OUT
    final = OUT
    final.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fixtures-", dir=final.parent) as tmp:
        OUT = pathlib.Path(tmp) / final.name
        OUT.mkdir()
        try:
            generate_all()
        finally:
            staging, OUT = OUT, final
        removed = sync(staging, final)
    for name in removed:
        print(f"removed stale {os.path.relpath(final / name, ROOT)}")
    for p in sorted(final.iterdir()):
        print(f"{p.stat().st_size:>9}  {os.path.relpath(p, ROOT)}")


GEMMA4_TINY_BYTE0 = 9
"""Id of <0x00> in the tiny gemma4 vocabulary: <0xNN> is GEMMA4_TINY_BYTE0 + NN."""


def gemma4_tiny() -> None:
    """A small gemma4 vocabulary whose every encoding is derivable by hand (see tests/test_gemma4_tiny.mojo).

    It differs from the real one in ids only: `add_bos_token` is false, so a test can see the BOS forced on, and the
    merges are chosen so that a wrong rank order, a missing newline split and a missing whole-run rule each change
    the ids. `base.gguf` and `eos.gguf` are well formed (the second adds an EOS); every other file breaks one thing, named
    by its file name.
    """
    block = "\u2581"
    controls = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|turn>", "<turn|>"]
    tokens = controls + ["<|channel>", "<|tool_response>"]
    types = [3] * len(controls) + [4, 4]
    assert len(tokens) == GEMMA4_TINY_BYTE0
    tokens += [f"<0x{b:02X}>" for b in range(256)]
    types += [6] * 256
    plain = [block, "a", "b", "c", "ab", "bc", "abc", block + "a", "\n", "\n\n", "\n\n\n", "a\n", block * 2, "\u00e9"]
    plain_start = len(tokens)
    tokens += plain
    types += [1] * len(plain)
    merges = [
        "a b",  # 0: ab
        "b c",  # 1: bc
        "ab c",  # 2: abc
        f"{block} a",  # 3: ▁a
        "\n \n",  # 4: a run of two newlines
        "a \n",  # 5: would join a letter to a newline if the pre-split did not stop it
        f"{block} {block}",  # 6: ▁▁
        "a b",  # 7: the same pair again; llama.cpp keeps the first rank, so this one is never used
    ]

    def write(
        name: str,
        add_bos: bool,
        *,
        add_eos: bool | None = None,
        tokens=tokens,
        types=types,
        merges=merges,
        omit: tuple[str, ...] = (),
        ids: dict[str, int] | None = None,
        scheme: str | None = None,
    ):
        w = GGUFWriter(str(OUT / "gemma4-tiny" / name), "gemma4")
        w.add_string("tokenizer.ggml.model", "gemma4")
        w.add_array("tokenizer.ggml.tokens", list(tokens))
        w.add_array("tokenizer.ggml.token_type", list(types))
        w.add_array("tokenizer.ggml.merges", list(merges))
        special = {"bos": 2, "eos": 1, "unknown": 3, "padding": 0, "mask": 4, **(ids or {})}
        for key, value in special.items():
            if key in omit:
                continue
            if value < 0:
                w.add_int32(f"tokenizer.ggml.{key}_token_id", value)
            else:
                w.add_uint32(f"tokenizer.ggml.{key}_token_id", value)
        w.add_bool("tokenizer.ggml.add_space_prefix", False)
        w.add_bool("tokenizer.ggml.add_bos_token", add_bos)
        if add_eos is not None:
            w.add_bool("tokenizer.ggml.add_eos_token", add_eos)
        if scheme is not None:
            w.add_string("tokenizer.ggml.metaspace.prepend_scheme", scheme)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

    (OUT / "gemma4-tiny").mkdir(exist_ok=True)
    write("base.gguf", add_bos=False)
    write("eos.gguf", add_bos=True, add_eos=True)
    no_byte = list(tokens)
    no_byte[GEMMA4_TINY_BYTE0 + 0x78] = "[0x78]"
    no_byte_types = list(types)
    no_byte_types[GEMMA4_TINY_BYTE0 + 0x78] = 1
    write("no-byte-token.gguf", add_bos=False, tokens=no_byte, types=no_byte_types)
    write("bad-byte-text.gguf", add_bos=False, tokens=no_byte)
    write("merge-without-space.gguf", add_bos=False, merges=merges[:3] + ["ab"] + merges[3:])
    write("short-types.gguf", add_bos=False, types=types[:-1])
    write("no-bos-id.gguf", add_bos=True, omit=("bos",))
    write("bos-out-of-range.gguf", add_bos=True, ids={"bos": len(tokens) + 14})
    write("eos-flag-without-id.gguf", add_bos=True, add_eos=True, omit=("eos",))
    write("metaspace-first.gguf", add_bos=True, scheme="first")
    write("metaspace-never.gguf", add_bos=True, scheme="never")
    duplicate = list(tokens)
    duplicate[plain_start + 2] = duplicate[plain_start + 1]
    write("duplicate-token.gguf", add_bos=True, tokens=duplicate)
    empty = list(tokens)
    empty[-1] = ""
    write("empty-token.gguf", add_bos=True, tokens=empty)

    def with_entries(*changes: tuple[int, str | None, int | None]):
        """The tokens and types with some entries replaced: (index, new text or None, new type or None)."""
        changed_tokens, changed_types = list(tokens), list(types)
        for index, text, kind in changes:
            if text is not None:
                changed_tokens[index] = text
            if kind is not None:
                changed_types[index] = kind
        return changed_tokens, changed_types

    # Unusual rather than wrong, each read by a test in tests/test_gemma4_tiny.mojo.
    for byte in (0x00, 0xFF):
        # The ends of the byte range: a loop that starts at 1 or stops at 254 would not see them missing.
        t, k = with_entries((GEMMA4_TINY_BYTE0 + byte, f"[0x{byte:02X}]", 1))
        write(f"no-byte-{byte:02x}.gguf", add_bos=False, tokens=t, types=k)
    # llama.cpp splits a merge at the first space after its first byte, so " ab" has no space to split at.
    write("merge-space-first.gguf", add_bos=False, merges=merges[:3] + [" ab"] + merges[3:])
    # An id equal to the vocabulary size is one past the last token.
    write("bos-at-vocab-size.gguf", add_bos=True, ids={"bos": len(tokens)})
    # A user-defined token whose text holds two other special tokens: only the longest-first order finds it.
    write("nested-special.gguf", add_bos=False, tokens=tokens + ["<|turn><turn|>"], types=types + [4])
    # <unk> typed unknown (2) instead of control (3).
    t, k = with_entries((3, None, 2))
    write("unknown-type.gguf", add_bos=False, tokens=t, types=k)
    # <eos> and <turn|> typed normal (1): llama.cpp overrides them to control because they end generation.
    t, k = with_entries((1, None, 1), (6, None, 1))
    write("forced-control.gguf", add_bos=False, tokens=t, types=k)
    # A token type outside 0 to 6 (#227), a negative special id other than -1, and -1 written out.
    t, k = with_entries((plain_start, None, 9))
    write("unknown-token-type.gguf", add_bos=False, tokens=t, types=k)
    t, k = with_entries((plain_start, None, -1))
    write("negative-token-type.gguf", add_bos=False, tokens=t, types=k)
    write("special-negative.gguf", add_bos=False, ids={"padding": -2})
    write("special-minus-one.gguf", add_bos=False, ids={"padding": -1})
    # EOS is the end-of-turn token itself: the end-of-generation list must hold its id once.
    write("eos-is-end-of-turn.gguf", add_bos=False, ids={"eos": 6})
    # No <turn|> and no <|tool_response>: nothing for the end-of-generation list to add beyond EOS.
    t, k = with_entries((6, "<end>", None), (8, "<tool_result>", None))
    write("no-end-of-turn.gguf", add_bos=False, tokens=t, types=k)


def gpt2_tiny() -> None:
    """A byte-level BPE vocabulary with the ids that LLaDA's and Dream's files leave out (see tests/test_tokenizer.mojo).

    Tokenizer keys only: padding and unknown ids, an EOS id, and neither `<|eot_id|>` nor `<|im_end|>`, so the
    end-of-turn id is -1. The loader maps bytes to code points itself and needs no byte tokens, so six tokens do.
    """
    (OUT / "gpt2-tiny").mkdir(exist_ok=True)
    base_tokens = ["a", "b", "<pad>", "<unk>", "<eos>", "ab"]
    base_types = [1, 1, 3, 2, 3, 1]

    def write(name, *, tokens=base_tokens, types=base_types, pre="qwen2", ids=None, add_bos=None, model_vocab=None,
              arch="llada"):
        w = GGUFWriter(str(OUT / "gpt2-tiny" / name), arch)
        w.add_string("tokenizer.ggml.model", "gpt2")
        w.add_string("tokenizer.ggml.pre", pre)
        w.add_array("tokenizer.ggml.tokens", list(tokens))
        w.add_array("tokenizer.ggml.token_type", list(types))
        w.add_array("tokenizer.ggml.merges", ["a b"])
        special = {"padding": 2, "unknown": 3, "eos": 4, **(ids or {})}
        for key, value in special.items():
            if value is None:
                continue
            if value < 0:
                w.add_int32(f"tokenizer.ggml.{key}_token_id", value)
            else:
                w.add_uint32(f"tokenizer.ggml.{key}_token_id", value)
        if add_bos is not None:
            w.add_bool("tokenizer.ggml.add_bos_token", add_bos)
        if model_vocab is not None:
            # The keys ModelConfig needs, so a test can compare the two vocabulary sizes (#200).
            w.add_uint32("llada.block_count", 1)
            w.add_uint32("llada.embedding_length", 256)
            w.add_uint32("llada.attention.head_count", 2)
            w.add_uint32("llada.feed_forward_length", 256)
            w.add_float32("llada.attention.layer_norm_rms_epsilon", 1e-5)
            w.add_uint32("llada.vocab_size", model_vocab)
            w.add_uint32("tokenizer.ggml.mask_token_id", 5)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()

    write("ids.gguf")
    write("short-types.gguf", types=base_types[:-1])
    write("pre-unknown.gguf", pre="mystery")
    write("special-out-of-range.gguf", ids={"bos": 99})
    write("special-negative.gguf", ids={"bos": -5})
    write("special-minus-one.gguf", ids={"padding": -1})
    write("duplicate-token.gguf", tokens=["a", "b", "<pad>", "<unk>", "<eos>", "a"])
    write("no-eos.gguf", ids={"eos": None})
    write("add-bos-without-id.gguf", add_bos=True)
    write("with-model.gguf", model_vocab=6)
    write("other-arch.gguf", arch="qwen2")  # a tokenizer of an architecture with no chat template here (#207)
    write("with-model-vocab-512.gguf", model_vocab=512)


UTF8_CASES = [
    # (name, bytes). Valid input first, then one case or more for each way a sequence can be wrong.
    ("empty", b""),
    ("ascii", b"abc 123"),
    ("two-byte", "caf\u00e9".encode()),
    ("three-byte", "\u20ac".encode()),
    ("four-byte", "\U0001f600".encode()),
    ("lowest-two-byte", bytes([0xC2, 0x80])),
    ("highest-two-byte", bytes([0xDF, 0xBF])),
    ("lowest-three-byte", bytes([0xE0, 0xA0, 0x80])),
    ("before-surrogates", bytes([0xED, 0x9F, 0xBF])),
    ("after-surrogates", bytes([0xEE, 0x80, 0x80])),
    ("highest-three-byte", bytes([0xEF, 0xBF, 0xBF])),
    ("lowest-four-byte", bytes([0xF0, 0x90, 0x80, 0x80])),
    ("highest-code-point", bytes([0xF4, 0x8F, 0xBF, 0xBF])),
    ("latin1-e9", bytes([0x63, 0x61, 0x66, 0xE9, 0x20, 0x61])),  # "caf\xe9 a": the case of issue #197
    ("latin1-e9-last", b"caf\xe9"),
    ("stray-continuation-first", bytes([0x80])),
    ("stray-continuation-middle", bytes([0x61, 0xBF, 0x62])),
    ("overlong-c0", bytes([0xC0, 0x80])),
    ("overlong-c1", bytes([0xC1, 0xBF])),
    ("overlong-three-byte", bytes([0xE0, 0x80, 0x80])),
    ("overlong-three-byte-high", bytes([0xE0, 0x9F, 0xBF])),
    ("overlong-four-byte", bytes([0xF0, 0x80, 0x80, 0x80])),
    ("overlong-four-byte-high", bytes([0xF0, 0x8F, 0xBF, 0xBF])),
    ("surrogate-low", bytes([0xED, 0xA0, 0x80])),
    ("surrogate-high", bytes([0xED, 0xBF, 0xBF])),
    ("above-10ffff", bytes([0xF4, 0x90, 0x80, 0x80])),
    ("lead-f5", bytes([0xF5, 0x80, 0x80, 0x80])),
    ("lead-ff", bytes([0xFF])),
    ("lead-fe-then-ascii", bytes([0xFE, 0x41])),
    ("two-byte-lead-then-ascii", bytes([0xC3, 0x28])),
    ("three-byte-then-ascii-second", bytes([0xE2, 0x28, 0xA1])),
    ("three-byte-then-ascii-third", bytes([0xE2, 0x82, 0x28])),
    ("four-byte-then-ascii-fourth", bytes([0xF0, 0x9F, 0x98, 0x28])),
    ("four-byte-then-ascii-third", bytes([0xF0, 0x9F, 0x28, 0x80])),
    ("lead-then-lead", bytes([0xC3, 0xC3, 0xA9])),
    ("truncated-two-byte", bytes([0x61, 0xC3])),
    ("truncated-three-byte", bytes([0xE2, 0x82])),
    ("truncated-four-byte", bytes([0xF0, 0x9F, 0x98])),
    ("truncated-three-byte-first-only", bytes([0xE2])),
    ("two-invalid-in-a-row", bytes([0xE9, 0xE9, 0x61])),
]


def lead_byte_decode(data: bytes) -> list[int]:
    """What `utf8_decode` must return: llama.cpp's `unicode_cpts_from_utf8` (src/unicode.cpp lines 30 to 75 and 1389
    to 1404 at 5f490d7), which turns the lead byte of an invalid sequence into U+FFFD and goes on at the next byte,
    except that a surrogate is also invalid here (llama.cpp's check stops at overlong forms and values above
    U+10FFFF)."""
    out: list[int] = []
    i = 0
    while i < len(data):
        c = data[i]
        need = 0
        if c < 0x80:
            out.append(c)
            i += 1
            continue
        if 0xC2 <= c <= 0xDF:
            need = 1
        elif 0xE0 <= c <= 0xEF:
            need = 2
        elif 0xF0 <= c <= 0xF4:
            need = 3
        tail = data[i + 1:i + 1 + need]
        ok = need > 0 and len(tail) == need and all(0x80 <= t <= 0xBF for t in tail)
        cp = 0
        if ok:
            cp = (c & (0x3F >> need)) if need else c
            for t in tail:
                cp = (cp << 6) | (t & 0x3F)
            ok = cp >= (0x80, 0x800, 0x10000)[need - 1] and cp <= 0x10FFFF and not 0xD800 <= cp <= 0xDFFF
        if ok:
            out.append(cp)
            i += 1 + need
        else:
            out.append(0xFFFD)
            i += 1
    return out


def utf8_cases() -> None:
    """tests/fixtures/utf8-cases.tsv: byte strings, the code points `utf8_decode` must return, and the offset of
    the first byte that begins an invalid sequence (-1 when the input is valid).

    Python's own decoder is the check on this table. The first invalid offset is `UnicodeDecodeError.start`. The
    code points agree with `bytes.decode(errors="replace")` in every case but one kind: a sequence that stops short
    after a valid prefix (a lead byte and some good continuation bytes), where Python writes one U+FFFD for the
    whole stretch and the lead-byte rule writes one for each byte."""
    lines = []
    differs = []
    for name, data in UTF8_CASES:
        expected = lead_byte_decode(data)
        try:
            data.decode("utf-8")
            first = -1
            assert expected == [ord(ch) for ch in data.decode("utf-8")], name
        except UnicodeDecodeError as err:
            first = err.start
            assert expected[first] == 0xFFFD, name
        python = [ord(ch) for ch in data.decode("utf-8", errors="replace")]
        if python != expected:
            differs.append(name)
        lines.append("\t".join((name, data.hex() or "-", " ".join(f"{cp:x}" for cp in expected) or "-", str(first))))
    # The five cases where Python and the lead-byte rule part are the sequences that stop short after a valid
    # prefix (a lead byte and one or more good continuation bytes): Python writes one U+FFFD for the stretch,
    # the lead-byte rule one for each byte. No other case parts, so a change in either decoder shows here.
    assert sorted(differs) == [
        "four-byte-then-ascii-fourth", "four-byte-then-ascii-third", "three-byte-then-ascii-third", "truncated-four-byte", "truncated-three-byte",
    ], differs
    (OUT / "utf8-cases.tsv").write_text("\n".join(lines) + "\n")


def entropy_rows() -> None:
    """Logits rows with the entropy-bounded denoiser's per-row answers in f64.

    One line per case: name, vocab, temperature, draw u, argmax of the raw row,
    entropy of softmax(row / temperature) in nats, the token an inverse-CDF
    draw at u picks (first index whose cumulative probability reaches u), and
    the row as float32 values written with enough digits to read back exactly.
    A separate seed keeps every other fixture's bytes where they were.
    """
    rng = np.random.default_rng(65065)
    cases = []

    def add(name, row, temperature, u):
        cases.append((name, row.astype(np.float32), temperature, u))

    add("mixed40", rng.normal(0, 3, 40), 0.8, 0.37)
    tie = rng.normal(0, 1, 12)
    tie[3] = tie[9] = 5.0
    add("ties", tie, 0.5, 0.9)
    peaked = rng.normal(0, 0.5, 64)
    peaked[41] = 25.0
    add("peaked", peaked, 0.4 + 0.4 / 48, 0.5)
    add("flat30", np.zeros(30), 0.8, 0.55)
    wide = rng.normal(0, 4, 2500)
    wide[2300] = 14.0
    add("wide2500", wide, 0.6, 0.81)
    add("ascending3000", np.linspace(-10, 10, 3000), 0.7, 0.2)
    add("exact1024", rng.normal(0, 2, 1024), 0.8, 0.5)
    add("large-negative", -1000.0 + rng.normal(0, 3, 50), 0.8, 0.66)
    add("u-low", rng.normal(0, 3, 40), 0.8, 1e-9)
    add("u-high", rng.normal(0, 3, 40), 0.8, 0.999999)
    lines = []
    for name, row, t, u in cases:
        x = row.astype(np.float64) / t
        p = np.exp(x - x.max())
        p /= p.sum()
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        cum = np.cumsum(p)
        sampled = int(np.searchsorted(cum, u, side="left"))
        # An answer that sits within rounding of a boundary would test rounding.
        assert np.abs(cum - u).min() > 1e-7, name
        argmax = int(np.argmax(row))
        values = " ".join(repr(float(v)) for v in row)
        lines.append(f"{name}\t{len(row)}\t{t!r}\t{u!r}\t{argmax}\t{entropy!r}\t{sampled}\t{values}")
    (OUT / "entropy-rows.tsv").write_text("\n".join(lines) + "\n")


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
