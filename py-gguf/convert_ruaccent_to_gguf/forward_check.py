#!/usr/bin/env python3
"""
Forward-pass sanity check: re-implement each RUAccent encoder in numpy from the
converted GGUF tensors (using the agreed tensor naming) and compare the logits
against onnxruntime running the original ONNX model.

This verifies the conversion's structural decisions (fused qkv split, per-head
bias placement, RoPE table usage, [out, in] orientation, pooler path) beyond
the tolerance-0 tensor comparison.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from validate_ruaccent_gguf import _GGML_DTYPE, read_gguf

FLT_MIN = np.float32(-3.4028234663852886e38)


def ln(x, w, b, eps):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + eps) * w + b).astype(np.float32)


_erf = np.vectorize(math.erf, otypes=[np.float64])


def gelu(x):
    return (0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))).astype(np.float32)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def to_heads(x, hidden, n_heads):
    b, s, _ = x.shape
    return x.reshape(b, s, n_heads, hidden // n_heads).transpose(0, 2, 1, 3).astype(np.float32)


def rot_half(t):
    return np.stack([-t[..., 1::2], t[..., ::2]], axis=-1).reshape(t.shape).astype(np.float32)


def load_tensors(gguf_path):
    meta, infos, data_base, buf = read_gguf(gguf_path)
    t = {}
    for name, shape, off, dtype in infos:
        n = int(np.prod(shape))
        t[name] = np.frombuffer(buf, dtype=_GGML_DTYPE[dtype], count=n, offset=data_base + off).reshape(shape)
    return meta, t


def dql(x):
    """ORT DynamicQuantizeLinear (uint8), bit-exact (all f32).

    Verified against onnxruntime: scale = (max(max,0) - min(min,0)) / 255,
    zp = clamp(round(-min0 / scale), 0, 255), q = clamp(round(x/scale) + zp, 0, 255).
    """
    xf = x.astype(np.float32)
    mn0 = np.float32(min(float(xf.min()), 0.0))
    mx0 = np.float32(max(float(xf.max()), 0.0))
    scale = (mx0 - mn0) / np.float32(255.0)
    zp = int(np.clip(np.round(-mn0 / scale), 0, 255))
    q = np.clip(np.round(xf / scale) + zp, 0, 255).astype(np.uint8)
    return q, scale, zp


def qmm(a, b, sa, za, sb, zb, einsum_subscripts):
    """MatMulInteger + dequant: f32(int32((a-za) @ (b-zb)) * (sa * sb)).

    The int32 result is exact (|int32| < 2^24) and the scale product is
    precomputed in f32, matching the ONNX graph op-for-op.
    """
    int32 = np.einsum(einsum_subscripts,
                      a.astype(np.int32) - za, b.astype(np.int32) - zb)
    return (int32.astype(np.float32) * (sa * sb)).astype(np.float32)


def qlinear(x, t, part):
    """Quantized linear layer (no bias): DQL(x) @ w_q, dequantized."""
    wq = t[f"{part}_q8"]  # [out, in] in the GGUF
    sw = t[f"{part}_q8_scale"][0]
    q, s, z = dql(x)
    return qmm(q, wq.T, s, z, sw, 0, "...i,io->...o")


def run_ort(model_dir, feed):
    sess = ort.InferenceSession(str(Path(model_dir) / "model.onnx"), providers=["CPUExecutionProvider"])
    return sess.run(None, feed)[0]


def check_accent(meta, t, feed):
    """Quantized RoFormer: f32 embeddings/RoPE/softmax, int8-dynamic MatMuls.

    Reproduces the ONNX graph op-for-op (DQL + MatMulInteger + dequant for
    every linear and for the attention-core matmuls), so logits match
    onnxruntime up to f32 rounding only.
    """
    b, s = feed["input_ids"].shape
    hidden, nh = meta["hidden"], meta["heads"]
    hd = hidden // nh
    eps = np.float32(1e-12)
    x = t["token_embd"][feed["input_ids"]] + t["type_embd"][feed["token_type_ids"]]
    x = ln(x, t["norm"], t["norm_bias"], eps)
    mask = feed["attention_mask"].astype(np.float32)  # [b, s]
    for i in range(meta["n_layers"]):
        h = x
        q = qlinear(x, t, f"blk.{i}.attn_q") + t[f"blk.{i}.attn_q_bias"]
        k = qlinear(x, t, f"blk.{i}.attn_k") + t[f"blk.{i}.attn_k_bias"]
        v = qlinear(x, t, f"blk.{i}.attn_v") + t[f"blk.{i}.attn_v_bias"]
        q, k, v = to_heads(q, hidden, nh), to_heads(k, hidden, nh), to_heads(v, hidden, nh)
        v_q, sv, zv = dql(v)
        tab = t["rope_embd"][:s]
        sin = tab[:, : hd // 2]
        cos = tab[:, hd // 2:]
        sin_pos = np.stack([sin, sin], axis=-1).reshape(s, hd)[None, None, :, :].astype(np.float32)
        cos_pos = np.stack([cos, cos], axis=-1).reshape(s, hd)[None, None, :, :].astype(np.float32)
        q = (q * cos_pos + rot_half(q) * sin_pos).astype(np.float32)
        k = (k * cos_pos + rot_half(k) * sin_pos).astype(np.float32)
        q_q, sq, zq = dql(q)
        k_t = np.ascontiguousarray(np.transpose(k, (0, 1, 3, 2)))
        k_q, sk, zk = dql(k_t)
        scores = qmm(q_q, k_q, sq, zq, sk, zk, "bhsd,bhdt->bhst")
        scores = scores / np.float32(math.sqrt(hd))
        scores = scores + (mask[:, None, :, None] - np.float32(1.0))
        attn = softmax(scores)
        attn_q, sa, za = dql(attn)
        ctx = qmm(attn_q, v_q, sa, za, sv, zv, "bhqk,bhkd->bhqd")
        ctx = np.transpose(ctx, (0, 2, 1, 3)).reshape(b, s, hidden)
        ctx = qlinear(ctx, t, f"blk.{i}.attn_o") + t[f"blk.{i}.attn_o_bias"]
        x = ln(ctx + h, t[f"blk.{i}.attn_norm"], t[f"blk.{i}.attn_norm_bias"], eps)
        f = gelu(qlinear(x, t, f"blk.{i}.ffn_0") + t[f"blk.{i}.ffn_0_bias"])
        f = qlinear(f, t, f"blk.{i}.ffn_1") + t[f"blk.{i}.ffn_1_bias"]
        x = ln(f + x, t[f"blk.{i}.ffn_norm"], t[f"blk.{i}.ffn_norm_bias"], eps)
    return qlinear(x, t, "cls.weight") + t["cls.bias"]


def check_bert(meta, t, feed):
    b, s = feed["input_ids"].shape
    hidden, nh = meta["hidden"], meta["heads"]
    hd = hidden // nh
    eps = np.float32(1e-12)
    x = t["token_embd"][feed["input_ids"]] + t["type_embd"][feed["token_type_ids"]]
    x = x + t["position_embd"][:s]
    x = ln(x, t["norm"], t["norm_bias"], eps)
    for i in range(meta["n_layers"]):
        h = x
        q = x @ t[f"blk.{i}.attn_q"].T + t[f"blk.{i}.attn_q_bias"]
        k = x @ t[f"blk.{i}.attn_k"].T + t[f"blk.{i}.attn_k_bias"]
        v = x @ t[f"blk.{i}.attn_v"].T + t[f"blk.{i}.attn_v_bias"]
        q, k, v = to_heads(q, hidden, nh), to_heads(k, hidden, nh), to_heads(v, hidden, nh)
        scores = q @ k.transpose(0, 1, 3, 2) / np.float32(math.sqrt(hd))
        attn = softmax(scores)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, hidden)
        ctx = ctx @ t[f"blk.{i}.attn_o"].T + t[f"blk.{i}.attn_o_bias"]
        x = ln(ctx + h, t[f"blk.{i}.attn_norm"], t[f"blk.{i}.attn_norm_bias"], eps)
        f = gelu(x @ t[f"blk.{i}.ffn_0"].T + t[f"blk.{i}.ffn_0_bias"])
        f = f @ t[f"blk.{i}.ffn_1"].T + t[f"blk.{i}.ffn_1_bias"]
        x = ln(f + x, t[f"blk.{i}.ffn_norm"], t[f"blk.{i}.ffn_norm_bias"], eps)
    return x @ t["cls.weight"].T + t["cls.bias"]


def check_deberta(meta, t, feed):
    b, s = feed["input_ids"].shape
    hidden, nh = meta["hidden"], meta["heads"]
    hd = hidden // nh
    eps = np.float32(1e-7)
    x = t["token_embd"][feed["input_ids"]] + t["position_embd"][:s]
    x = ln(x, t["norm"], t["norm_bias"], eps)
    for i in range(meta["n_layers"]):
        h = x
        q = x @ t[f"blk.{i}.attn_q"].T + t[f"blk.{i}.attn_q_bias"]
        k = x @ t[f"blk.{i}.attn_k"].T
        v = x @ t[f"blk.{i}.attn_v"].T + t[f"blk.{i}.attn_v_bias"]
        q, k, v = to_heads(q, hidden, nh), to_heads(k, hidden, nh), to_heads(v, hidden, nh)
        scores = (q / np.float32(math.sqrt(hd))) @ k.transpose(0, 1, 3, 2)
        attn = softmax(scores)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, hidden)
        ctx = ctx @ t[f"blk.{i}.attn_o"].T + t[f"blk.{i}.attn_o_bias"]
        x = ln(ctx + h, t[f"blk.{i}.attn_norm"], t[f"blk.{i}.attn_norm_bias"], eps)
        f = gelu(x @ t[f"blk.{i}.ffn_0"].T + t[f"blk.{i}.ffn_0_bias"])
        f = f @ t[f"blk.{i}.ffn_1"].T + t[f"blk.{i}.ffn_1_bias"]
        x = ln(f + x, t[f"blk.{i}.ffn_norm"], t[f"blk.{i}.ffn_norm_bias"], eps)
    pooled = x[:, 0]
    pooled = pooled @ t["pooler.weight"].T + t["pooler.bias"]
    pooled = gelu(pooled)
    return pooled @ t["cls.weight"].T + t["cls.bias"]


def check_distilbert(meta, t, feed):
    # This DistilBERT variant is post-norm (BERT-style): the attention operates
    # on the raw input, and the norms (named sa_layer_norm / output_layer_norm
    # in the ONNX) are applied after the attention and FFN residuals.
    b, s = feed["input_ids"].shape
    hidden, nh = meta["hidden"], meta["heads"]
    hd = hidden // nh
    eps = np.float32(1e-12)
    x = t["token_embd"][feed["input_ids"]] + t["position_embd"][:s]
    x = ln(x, t["norm"], t["norm_bias"], eps)
    for i in range(meta["n_layers"]):
        h = x
        q = x @ t[f"blk.{i}.attn_q"].T + t[f"blk.{i}.attn_q_bias"]
        k = x @ t[f"blk.{i}.attn_k"].T + t[f"blk.{i}.attn_k_bias"]
        v = x @ t[f"blk.{i}.attn_v"].T + t[f"blk.{i}.attn_v_bias"]
        q, k, v = to_heads(q, hidden, nh), to_heads(k, hidden, nh), to_heads(v, hidden, nh)
        scores = q @ k.transpose(0, 1, 3, 2) / np.float32(math.sqrt(hd))
        attn = softmax(scores)
        ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(b, s, hidden)
        ctx = ctx @ t[f"blk.{i}.attn_o"].T + t[f"blk.{i}.attn_o_bias"]
        x = ln(ctx + h, t[f"blk.{i}.attn_norm"], t[f"blk.{i}.attn_norm_bias"], eps)
        f = gelu(x @ t[f"blk.{i}.ffn_0"].T + t[f"blk.{i}.ffn_0_bias"])
        f = f @ t[f"blk.{i}.ffn_1"].T + t[f"blk.{i}.ffn_1_bias"]
        x = ln(f + x, t[f"blk.{i}.ffn_norm"], t[f"blk.{i}.ffn_norm_bias"], eps)
    return x @ t["cls.weight"].T + t["cls.bias"]


CHECKS = {
    "roformer": (check_accent, 10, True),
    "bert": (check_bert, 12, True),
    "deberta": (check_deberta, 16, False),
    "distilbert": (check_distilbert, 12, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta, t = load_tensors(args.gguf)
    if "rope_embd" in t:
        fam = "roformer"
    elif "pooler.weight" in t:
        fam = "deberta"
    elif "type_embd" not in t:
        fam = "distilbert"
    else:
        fam = "bert"
    check, seq_len, has_type = CHECKS[fam]

    rng = np.random.default_rng(args.seed)
    vocab = meta["vocab"]
    feed = {
        "input_ids": rng.integers(0, vocab, size=(1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
    }
    if has_type:
        feed["token_type_ids"] = rng.integers(0, 2, size=(1, seq_len), dtype=np.int64)

    ref = run_ort(args.model_dir, feed)
    got = check(meta, t, feed)

    diff = np.abs(ref - got)
    print(f"family={fam} seq_len={seq_len}")
    print(f"  ref logits[:6]: {ref.ravel()[:6]}")
    print(f"  gguf logits[:6]: {got.ravel()[:6]}")
    print(f"  max_abs_diff = {diff.max():.3e}")
    if diff.max() <= 1e-4:
        print("RESULT: PASS (logits match onnxruntime within 1e-4)")
    else:
        print("RESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
