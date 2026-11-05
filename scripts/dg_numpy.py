#!/usr/bin/env python3
"""A numpy forward pass for diffusion-gemma, written from the llama.cpp branch source.

Source: ggml-org/llama.cpp#24423 at 2e715c0639c6f07c57c943f8d5a7919953af8f7c, files
`src/models/diffusion-gemma.cpp` and `src/models/gemma4-common.h`. It shares no code with
the Mojo model: every rule below was read from that source, and the tests compare the two.

Python is a development tool here, never a runtime dependency. Run through the fixtures group:

    uv run --only-group fixtures python scripts/dg_numpy.py FILE.gguf [--prompt N] [--no-quant]

`forward` returns the canvas logits [C, vocab] of one pass over [prompt | canvas].

Activation quantization. mojo-dllm (like llama.cpp) multiplies int8 activations with the
quantized weights, so its logits differ from a float forward pass by the rounding of the
activations. `quant=True` repeats that rounding: before each matrix product the input row is
cut into blocks of 256 (Q4_K, Q6_K weights) or 32 (Q5_0, Q8_0 weights), each block scaled by
amax / 127 and rounded half away from zero. The integer dot product is exact, so the float
product of the dequantized weights and the dequantized activations equals it. `quant=False`
is the plain float32 pass, which agrees with the quantized one to a percent or so.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader
from gguf.quants import dequantize

ARCH = "diffusion-gemma."
BLOCK = {Q.Q4_K: 256, Q.Q6_K: 256, Q.Q5_0: 32, Q.Q8_0: 32}


class Weight:
    """A matrix [rows, cols] with the activation block size its quantization implies."""

    def __init__(self, w: np.ndarray, block: int | None):
        self.w = w.astype(np.float32)
        self.block = block


class Model:
    def __init__(self, path: str | pathlib.Path):
        r = GGUFReader(str(path))
        self.kv = {}
        for f in r.fields.values():
            self.kv[f.name] = f.contents()
        self.t: dict[str, Weight] = {}
        for t in r.tensors:
            if t.tensor_type in (Q.F32, Q.F16):
                self.t[t.name] = Weight(np.asarray(t.data), None)
            else:
                self.t[t.name] = Weight(dequantize(t.data, t.tensor_type), BLOCK[t.tensor_type])
        k = lambda n: self.kv[ARCH + n]  # noqa: E731
        self.layers = int(k("block_count"))
        self.hidden = int(k("embedding_length"))
        self.heads = int(k("attention.head_count"))
        self.kv_heads = [int(x) for x in k("attention.head_count_kv")]
        self.sliding = [bool(x) for x in k("attention.sliding_window_pattern")]
        self.window = int(k("attention.sliding_window"))
        self.head = [int(k("attention.key_length_swa")) if s else int(k("attention.key_length")) for s in self.sliding]
        self.base = [float(k("rope.freq_base_swa")) if s else float(k("rope.freq_base")) for s in self.sliding]
        self.eps = float(k("attention.layer_norm_rms_epsilon"))
        self.experts = int(k("expert_count"))
        self.used = int(k("expert_used_count"))
        self.expert_ffn = int(k("expert_feed_forward_length"))
        self.softcap = float(k("final_logit_softcapping"))
        self.canvas = int(self.kv["diffusion.canvas_length"])
        self.vocab = self.t["token_embd.weight"].w.shape[0]


def rms(x: np.ndarray, eps: float, w: np.ndarray | None = None) -> np.ndarray:
    x = x.astype(np.float32)
    y = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps).astype(np.float32)
    return (y * w if w is not None else y).astype(np.float32)


def round_half_away(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def quantize_rows(x: np.ndarray, block: int) -> np.ndarray:
    """What the int8 matrix kernels see: x rounded blockwise to amax/127 steps."""
    x = x.astype(np.float32)
    rows, cols = x.shape
    b = x.reshape(rows, cols // block, block)
    amax = np.abs(b).max(-1, keepdims=True)
    d = (amax / np.float32(127.0)).astype(np.float32)
    inv = np.where(amax > 0, np.float32(127.0) / np.where(amax > 0, amax, 1), 0).astype(np.float32)
    q = round_half_away((b * inv).astype(np.float32))
    return (q * d).reshape(rows, cols).astype(np.float32)


def mm(x: np.ndarray, w: Weight, quant: bool) -> np.ndarray:
    """x [n, K] times the matrix [N, K] transposed."""
    if quant and w.block is not None:
        x = quantize_rows(x, w.block)
    return (x.astype(np.float64) @ w.w.astype(np.float64).T).astype(np.float32)


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU, tanh approximation (PyTorch approximate='tanh'). ggml_gelu is the same formula."""
    x = x.astype(np.float64)
    return (0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))).astype(np.float32)


def rope_neox(x: np.ndarray, base: float, factors: np.ndarray | None) -> np.ndarray:
    """[T, heads, d]. Pair i is (i, i + d/2) with angle pos * base^(-2i/d) / factor_i (NEOX)."""
    t, _, d = x.shape
    half = d // 2
    inv = base ** (-2.0 * np.arange(half, dtype=np.float64) / d)
    if factors is not None:
        inv = inv / factors.astype(np.float64)
    ang = np.arange(t, dtype=np.float64)[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :half].astype(np.float64), x[..., half:].astype(np.float64)
    out = np.empty(x.shape, np.float64)
    out[..., :half] = a * c - b * s
    out[..., half:] = a * s + b * c
    return out.astype(np.float32)


def allowed(q: int, k: int, p: int, window: int, sliding: bool) -> bool:
    """The branch's unified mask (`llm_graph_input_attn_diffusion::set_input`), one cell."""
    q_canvas, k_canvas = q >= p, k >= p
    if q_canvas:
        return (k_canvas or k >= p - window + 1) if sliding else True
    ok = (not k_canvas) and k <= q
    if ok and sliding and q - k >= window:  # is_masked_swa, LLAMA_SWA_TYPE_STANDARD
        ok = False
    return ok


def forward(
    m: Model,
    tokens: list[int],
    n_prompt: int,
    *,
    quant: bool = True,
    prev_logits: np.ndarray | None = None,
    prev_temp_inv: float = 1.0,
    trace: list | None = None,
) -> np.ndarray:
    """Canvas logits [C, vocab] of one pass over [prompt | canvas]. `prev_logits` [C, vocab] turns on
    self-conditioning with the previous step's raw logits and 1 / (the previous step's temperature).
    `trace`, a list, receives every layer's output rows [T, hidden] (after the region scalar)."""
    t = len(tokens)
    p = n_prompt
    c = t - p
    h = m.hidden
    w = m.t
    eps = m.eps
    x = w["token_embd.weight"].w[tokens] * np.float32(np.sqrt(h))
    canvas = x[p:]
    if prev_logits is not None:
        z = prev_logits.astype(np.float64) * prev_temp_inv
        z = np.exp(z - z.max(-1, keepdims=True))
        probs = (z / z.sum(-1, keepdims=True)).astype(np.float32)
        soft = (probs.astype(np.float64) @ w["token_embd.weight"].w.astype(np.float64)).astype(np.float32)
        soft = soft * np.float32(np.sqrt(h))
        nrm = rms(soft, eps, w["self_cond_pre_norm.weight"].w)
        g = gelu(mm(nrm, w["self_cond_gate.weight"], quant))
        u = mm(nrm, w["self_cond_up.weight"], quant)
        canvas = canvas + mm(g * u, w["self_cond_down.weight"], quant)
    x = np.concatenate([x[:p], rms(canvas, eps)], 0)

    for i in range(m.layers):
        pre = f"blk.{i}."
        nh, nkv, d = m.heads, m.kv_heads[i], m.head[i]
        sl = m.sliding[i]
        cur = rms(x, eps, w[pre + "attn_norm.weight"].w)
        q = mm(cur, w[pre + "attn_q.weight"], quant).reshape(t, nh, d)
        q = rms(q, eps, w[pre + "attn_q_norm.weight"].w)
        k_raw = mm(cur, w[pre + "attn_k.weight"], quant)
        v_raw = mm(cur, w[pre + "attn_v.weight"], quant) if pre + "attn_v.weight" in w else k_raw
        k_raw, v_raw = k_raw.reshape(t, nkv, d), v_raw.reshape(t, nkv, d)
        k = rms(k_raw, eps, w[pre + "attn_k_norm.weight"].w)
        v = rms(v_raw, eps)  # no weight; on global layers v_raw is the K projection before its norm and rope
        fac = None if sl else w["rope_freqs.weight"].w
        q = rope_neox(q, m.base[i], fac)
        k = rope_neox(k, m.base[i], fac)
        att = np.empty((t, nh, d), np.float32)
        group = nh // nkv
        mask = np.array([[allowed(a, b, p, m.window, sl) for b in range(t)] for a in range(t)])
        for hh in range(nh):
            s = q[:, hh, :].astype(np.float64) @ k[:, hh // group, :].astype(np.float64).T  # scale 1.0
            s = np.where(mask, s, -np.inf)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            att[:, hh, :] = (s @ v[:, hh // group, :].astype(np.float64)).astype(np.float32)
        o = mm(att.reshape(t, nh * d), w[pre + "attn_output.weight"], quant)
        x = rms(o, eps, w[pre + "post_attention_norm.weight"].w) + x

        mlp = rms(x, eps, w[pre + "ffn_norm.weight"].w)
        mlp = gelu(mm(mlp, w[pre + "ffn_gate.weight"], quant)) * mm(mlp, w[pre + "ffn_up.weight"], quant)
        mlp = rms(mm(mlp, w[pre + "ffn_down.weight"], quant), eps, w[pre + "post_ffw_norm_1.weight"].w)

        moe_in = rms(x, eps, w[pre + "pre_ffw_norm_2.weight"].w)
        route = rms(x, eps) * np.float32(1.0 / np.sqrt(h)) * w[pre + "ffn_gate_inp.scale"].w
        logits = route.astype(np.float64) @ w[pre + "ffn_gate_inp.weight"].w.astype(np.float64).T
        pr = np.exp(logits - logits.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        gu = w[pre + "ffn_gate_up_exps.weight"]
        dn = w[pre + "ffn_down_exps.weight"]
        scale = w[pre + "ffn_down_exps.scale"].w
        moe = np.zeros((t, h), np.float64)
        for tok in range(t):
            top = np.argsort(-pr[tok], kind="stable")[: m.used]
            wts = pr[tok, top] / pr[tok, top].sum()
            for e, we in zip(top, wts):
                gw = Weight(gu.w[e], gu.block)
                dw = Weight(dn.w[e], dn.block)
                y = mm(moe_in[tok : tok + 1], gw, quant)[0]
                a = gelu(y[: m.expert_ffn]) * y[m.expert_ffn :]
                out = mm(a[None, :], dw, quant)[0]
                moe[tok] += we * scale[e] * out
        moe = rms(moe.astype(np.float32), eps, w[pre + "post_ffw_norm_2.weight"].w)
        x = rms(mlp + moe, eps, w[pre + "post_ffw_norm.weight"].w) + x

        enc, dec = w[pre + "enc_layer_output_scale.weight"].w[0], w[pre + "layer_output_scale.weight"].w[0]
        x = np.concatenate([x[:p] * enc, x[p:] * dec], 0)
        if trace is not None:
            trace.append(x.copy())

    x = rms(x[p:], eps, w["output_norm.weight"].w)
    out = mm(x, w["token_embd.weight"], quant)
    if m.softcap:
        out = (np.float32(m.softcap) * np.tanh(out * np.float32(1.0 / m.softcap))).astype(np.float32)
    assert out.shape == (c, m.vocab)
    return out


def tiny_tokens(m: Model, prompt: int, seed: int) -> list[int]:
    """A prompt of `prompt` tokens and a canvas of random tokens, the way the sampler starts."""
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(5, m.vocab, prompt + m.canvas)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--prompt", type=int, default=9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-quant", action="store_true")
    a = ap.parse_args()
    m = Model(a.gguf)
    toks = tiny_tokens(m, a.prompt, a.seed)
    out = forward(m, toks, a.prompt, quant=not a.no_quant)
    print("logits", out.shape, "std", float(out.std()), "argmax", out.argmax(-1).tolist()[:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
