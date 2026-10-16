#!/usr/bin/env python3
"""
Convert RUAccent ONNX models to GGUF (architecture "ruaccent.encoder").

Supported model families (auto-detected from the ONNX initializer name prefix):
  roformer    nn_accent (int8 dynamic-quantized MatMuls, RoPE via a learned table)
  bert        nn_stress_usage_predictor
  deberta     nn_omograph/turbo3.1 (relative_attention=false: plain encoder with a
              fused qkv in_proj, per-head q/v biases, and a GELU pooler in the
              sequence-classification path)
  distilbert  nn_yo_homograph_resolver

Usage:
  python convert_ruaccent_to_gguf.py --model-dir /path/to/nn_accent --save-path /out/accent.gguf

Every model dir must contain model.onnx; config.json (if present) supplies the
metadata cross-checked against the actual tensors.

Tensor naming (GGUF):
  token_embd             [vocab, hidden]
  position_embd          [max_pos, hidden]        (absent for roformer: it uses RoPE)
  type_embd              [2, hidden]              (absent when has_type_embd=false)
  rope_embd              [rope_max_pos, head_dim] (roformer only, learned RoPE table)
  norm / norm_bias       [hidden]
  blk.{i}.attn_{q,k,v,o}        [hidden, hidden]  (linear weights, [out, in] layout)
  blk.{i}.attn_{q,k,v,o}_bias   [hidden]          (absent when the model has no such bias)
  blk.{i}.attn_norm(_bias)      [hidden]
   blk.{i}.ffn_0(_bias)          [hidden, 4*hidden] / [4*hidden]
   blk.{i}.ffn_1(_bias)          [4*hidden, hidden] / [hidden]
   blk.{i}.ffn_norm(_bias)       [hidden]
   pooler.weight / pooler.bias   [hidden, hidden] / [hidden] (deberta only)
   cls.weight / cls.bias         [n_labels, hidden] / [n_labels]
   <part>_q8 / <part>_q8_scale   int8 weight + scale (roformer only, see above)

Orientation: ONNX MatMul weights are stored as [in, out] and are transposed to
[out, in] here (the ggml MUL_MAT / TTS.cpp convention, same as the kokoro and
t5-encoder converters). ONNX Gemm weights (deberta pooler/classifier) are
already [out, in] and are stored as-is.

roformer (nn_accent/model.onnx) is dynamically quantized: every linear (q, k,
v, o, ffn_0, ffn_1 per layer + the classifier) is a MatMulInteger with a
static int8 weight (per-tensor scale, zero point 0) and a dynamically
quantized activation. Both the dequantized float32 weight and the raw int8
weight (+ scale) are stored, so C++ can reproduce the quantized computation
exactly:

  <part>            [out, in] f32   dequantized weight (q - zp) * scale
  <part>_q8         [out, in] i8    raw int8 weight (transposed like the f32 one)
  <part>_q8_scale   [1] f32         per-tensor weight scale

The runtime quantization (verified bit-exact against onnxruntime, all f32):
  mn0 = min(min(x), 0); mx0 = max(max(x), 0)          # global over the tensor
  scale = (mx0 - mn0) / 255
  zp    = clamp(round(-mn0 / scale), 0, 255)
  q     = clamp(round(x / scale) + zp, 0, 255)        # uint8
  int32 = (q - zp) @ w_q                               # exact (|int32| < 2^24)
  out   = f32(int32) * (scale * scale_w)               # product precomputed in f32

The attention core is quantized too: RoPE is applied in f32, then q_rot and
k_rot^T are DQL-quantized and multiplied as int32 (dequantized with
scale_q * scale_k, then divided by sqrt(head_dim)); the softmax output and v
are DQL-quantized and multiplied as int32 (dequantized with scale_a * scale_v).
The attention mask is added as (mask - 1.0) on the query dimension after the
1/sqrt(head_dim) scaling.

Skipped (non-learned) initializers:
  {family}.embeddings.position_ids - an arange(0, max_pos) index constant used
  only to gather position embeddings; not a learned weight.

All other initializers must be mapped to a GGUF tensor; the converter fails if
any initializer is left unaccounted for.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import gguf
import numpy as np
import onnx

ARCHITECTURE = "ruaccent.encoder"
F32 = np.float32

FAMILIES = ("roformer", "bert", "deberta", "distilbert")

# Default labels per family, used when config.json has no id2label.
DEFAULT_LABELS = {
    "roformer": ["NO", "STRESS_PRIMARY", "STRESS_SECONDARY"],
    "bert": ["NO_STRESS", "PUNCT", "STRESS"],
    "deberta": ["FALSE", "TRUE"],
    "distilbert": ["NO_YO", "PUNCT", "YO"],
}

logger = logging.getLogger("convert_ruaccent_to_gguf")


def _const_value(node):
    for a in node.attribute:
        if a.name == "value" and a.t is not None:
            try:
                return onnx.numpy_helper.to_array(a.t)
            except TypeError:
                return None
    return None


class ModelGraph:
    """An ONNX graph with initializer access and name->node lookups."""

    def __init__(self, path):
        m = onnx.load(str(path))
        self.graph = m.graph
        self.inits = {i.name: onnx.numpy_helper.to_array(i) for i in self.graph.initializer}
        self.nodes = self.graph.node
        self.consumers = {}
        self.producers = {}
        for n in self.nodes:
            for o in n.output:
                if o:
                    self.producers[o] = n
            for i in n.input:
                if i:
                    self.consumers.setdefault(i, []).append(n)
        self.const_values = {}
        for n in self.nodes:
            if n.op_type == "Constant":
                v = _const_value(n)
                if v is not None:
                    self.const_values[n.output[0]] = v

    def family(self):
        for name in self.inits:
            for fam in FAMILIES:
                if name.startswith(fam + "."):
                    return fam
        raise ValueError(f"unknown model family in {self.inits.keys()}")


def _linear_target(node_name):
    """Map an ONNX MatMul/MatMulInteger node name to (layer, gguf part).

    Node names look like '/bert/encoder/layer.0/attention/self/query/MatMul' or
    '/distilbert/transformer/layer.0/ffn/lin1/MatMul' (slashes, not dots).
    """
    m = re.search(r"layer\.(\d+)/attention/self/(query|key|value)/MatMul", node_name)
    if m:
        return int(m.group(1)), {"query": "attn_q", "key": "attn_k", "value": "attn_v"}[m.group(2)]
    m = re.search(r"layer\.(\d+)/attention/self/in_proj/MatMul", node_name)
    if m:
        return int(m.group(1)), "in_proj"
    m = re.search(r"layer\.(\d+)/attention/output/dense/MatMul", node_name)
    if m:
        return int(m.group(1)), "attn_o"
    m = re.search(r"layer\.(\d+)/intermediate/dense/MatMul", node_name)
    if m:
        return int(m.group(1)), "ffn_0"
    m = re.search(r"layer\.(\d+)/output/dense/MatMul", node_name)
    if m:
        return int(m.group(1)), "ffn_1"
    m = re.search(r"layer\.(\d+)/attention/(q|k|v|out)_lin/MatMul", node_name)
    if m:
        return int(m.group(1)), {"q": "attn_q", "k": "attn_k", "v": "attn_v", "out": "attn_o"}[m.group(2)]
    m = re.search(r"layer\.(\d+)/ffn/(lin1|lin2)/MatMul", node_name)
    if m:
        return int(m.group(1)), {"lin1": "ffn_0", "lin2": "ffn_1"}[m.group(2)]
    if re.search(r"classifier/MatMul", node_name):
        return None, "cls.weight"
    return None, None


def _split_in_proj(w, heads):
    """Split a fused [in, 3*hidden] qkv weight into (q, k, v) [in, hidden] each.

    The ONNX graph reshapes the fused output to [bs, seq, heads, 3*head_dim] and
    slices it, so the output layout is per-head interleaved:
    out_index = h * 3 * head_dim + part * head_dim + d.
    """
    in_dim, out_dim = w.shape
    head_dim = in_dim // heads
    assert out_dim == 3 * in_dim, f"fused in_proj has unexpected shape {w.shape}"
    w3 = w.reshape(in_dim, heads, 3, head_dim)
    return [w3[:, :, p, :].reshape(in_dim, in_dim) for p in range(3)]


def build_mapping(g: ModelGraph):
    """Map every ONNX initializer to a GGUF tensor.

    Returns (tensors, skipped) where tensors is {gguf_name: float32 ndarray} and
    skipped is {init_name: reason}.
    """
    fam = g.family()
    tensors = {}
    skipped = {}
    used = set()

    def take(init_name, gguf_name, transpose=False):
        arr = g.inits[init_name].astype(F32, copy=False)
        if transpose:
            arr = np.ascontiguousarray(arr.T)
        tensors[gguf_name] = arr
        used.add(init_name)
        logger.debug(f"  {init_name} -> {gguf_name} {arr.shape}")

    p = fam + "."

    # --- embeddings (common to all families) ---
    take(p + "embeddings.word_embeddings.weight", "token_embd")
    if p + "embeddings.token_type_embeddings.weight" in g.inits:
        take(p + "embeddings.token_type_embeddings.weight", "type_embd")
    if p + "embeddings.position_embeddings.weight" in g.inits:
        take(p + "embeddings.position_embeddings.weight", "position_embd")
    if p + "embeddings.position_ids" in g.inits:
        skipped[p + "embeddings.position_ids"] = "arange position index constant, not learned"
        used.add(p + "embeddings.position_ids")
    take(p + "embeddings.LayerNorm.weight", "norm")
    take(p + "embeddings.LayerNorm.bias", "norm_bias")

    # --- per-family named parameters ---
    if fam == "roformer":
        take(p + "encoder.embed_positions.weight", "rope_embd")
        part_map = {
            "attention.self.query.bias": "attn_q_bias",
            "attention.self.key.bias": "attn_k_bias",
            "attention.self.value.bias": "attn_v_bias",
            "attention.output.dense.bias": "attn_o_bias",
            "attention.output.LayerNorm.weight": "attn_norm",
            "attention.output.LayerNorm.bias": "attn_norm_bias",
            "intermediate.dense.bias": "ffn_0_bias",
            "output.dense.bias": "ffn_1_bias",
            "output.LayerNorm.weight": "ffn_norm",
            "output.LayerNorm.bias": "ffn_norm_bias",
        }
        layer_re = re.compile(p + r"encoder\.layer\.(\d+)\.([\w.]+)")
    elif fam == "bert":
        part_map = {
            "attention.self.query.bias": "attn_q_bias",
            "attention.self.key.bias": "attn_k_bias",
            "attention.self.value.bias": "attn_v_bias",
            "attention.output.dense.bias": "attn_o_bias",
            "attention.output.LayerNorm.weight": "attn_norm",
            "attention.output.LayerNorm.bias": "attn_norm_bias",
            "intermediate.dense.bias": "ffn_0_bias",
            "output.dense.bias": "ffn_1_bias",
            "output.LayerNorm.weight": "ffn_norm",
            "output.LayerNorm.bias": "ffn_norm_bias",
        }
        layer_re = re.compile(p + r"encoder\.layer\.(\d+)\.([\w.]+)")
    elif fam == "deberta":
        part_map = {
            "attention.output.dense.bias": "attn_o_bias",
            "attention.output.LayerNorm.weight": "attn_norm",
            "attention.output.LayerNorm.bias": "attn_norm_bias",
            "intermediate.dense.bias": "ffn_0_bias",
            "output.dense.bias": "ffn_1_bias",
            "output.LayerNorm.weight": "ffn_norm",
            "output.LayerNorm.bias": "ffn_norm_bias",
        }
        layer_re = re.compile(p + r"encoder\.layer\.(\d+)\.([\w.]+)")
        take("pooler.dense.weight", "pooler.weight")
        take("pooler.dense.bias", "pooler.bias")
        take("classifier.weight", "cls.weight")
    else:  # distilbert
        part_map = {
            "attention.q_lin.bias": "attn_q_bias",
            "attention.k_lin.bias": "attn_k_bias",
            "attention.v_lin.bias": "attn_v_bias",
            "attention.out_lin.bias": "attn_o_bias",
            "sa_layer_norm.weight": "attn_norm",
            "sa_layer_norm.bias": "attn_norm_bias",
            "ffn.lin1.bias": "ffn_0_bias",
            "ffn.lin2.bias": "ffn_1_bias",
            "output_layer_norm.weight": "ffn_norm",
            "output_layer_norm.bias": "ffn_norm_bias",
        }
        layer_re = re.compile(p + r"transformer\.layer\.(\d+)\.([\w.]+)")

    for name in g.inits:
        m = layer_re.match(name)
        if m and m.group(2) in part_map:
            take(name, f"blk.{m.group(1)}.{part_map[m.group(2)]}")

    # --- classifier bias (named in all families) ---
    if "classifier.bias" in g.inits:
        take("classifier.bias", "cls.bias")

    # --- anonymous linear weights (onnx::MatMul_NNN or quantized triples) ---
    bases = {}
    for name in g.inits:
        if name.startswith("onnx::MatMul_"):
            base = name.rsplit("_", 1)[0] if name.endswith(("_quantized", "_scale", "_zero_point")) else name
            bases.setdefault(base, set()).add(name)

    for base, names in bases.items():
        quant = None  # (int8 weight [in, out], scale f32) for dynamic-quantized weights
        if base + "_quantized" in g.inits:
            # int8 dynamic-quantized weight: (q - zp) * scale
            q = g.inits[base + "_quantized"]
            scale = g.inits[base + "_scale"].astype(F32)
            zp = g.inits[base + "_zero_point"].astype(F32)
            if not np.all(zp == 0):
                raise ValueError(f"non-zero weight zero point in {base}: {zp.ravel()}")
            w = (q.astype(F32) - zp) * scale
            quant = (q, scale)
            node = next(n for n in g.consumers.get(base + "_quantized", []) if n.op_type == "MatMulInteger")
            used.update([base + "_quantized", base + "_scale", base + "_zero_point"])
        elif base in g.inits:
            w = g.inits[base].astype(F32, copy=False)
            node = next(n for n in g.consumers.get(base, []) if n.op_type == "MatMul")
            used.add(base)
        else:
            continue

        layer, part = _linear_target(node.name)
        if part is None:
            raise ValueError(f"cannot map linear weight '{base}' (node '{node.name}')")
        if part == "in_proj":
            # heads from the per-head bias initializers ([1, heads, 1, head_dim])
            bias = next(a for a in g.inits.values() if a.ndim == 4 and a.shape[0] == 1 and a.shape[2] == 1)
            heads = bias.shape[1]
            q_w, k_w, v_w = _split_in_proj(w, heads)  # w is [in, 3*hidden]
            for part2, ww in zip(("attn_q", "attn_k", "attn_v"), (q_w, k_w, v_w)):
                tensors[f"blk.{layer}.{part2}"] = np.ascontiguousarray(ww.T)
        else:
            w = np.ascontiguousarray(w.T)  # [in, out] -> [out, in]
            if part == "cls.weight":
                tensors["cls.weight"] = w
                if quant is not None:
                    tensors["cls.weight_q8"] = np.ascontiguousarray(quant[0].T.astype(np.int8))
                    tensors["cls.weight_q8_scale"] = quant[1].reshape(1).astype(F32)
            else:
                tensors[f"blk.{layer}.{part}"] = w
                if quant is not None:
                    tensors[f"blk.{layer}.{part}_q8"] = np.ascontiguousarray(quant[0].T.astype(np.int8))
                    tensors[f"blk.{layer}.{part}_q8_scale"] = quant[1].reshape(1).astype(F32)
        logger.debug(f"  {base} -> blk.{layer}.{part} (node '{node.name}')")

    # --- deberta per-head q/v biases (onnx::Add_NNN, shape [1, heads, 1, head_dim]) ---
    if fam == "deberta":
        for name in g.inits:
            if name.startswith("onnx::Add_") and g.inits[name].ndim == 4:
                add_node = next(n for n in g.consumers.get(name, []) if n.op_type == "Add")
                m = re.search(r"layer\.(\d+)/", add_node.name)
                layer = int(m.group(1)) if m else None
                out_consumers = {c.op_type for c in g.consumers.get(add_node.output[0], [])}
                if "Div" in out_consumers:
                    part = "attn_q_bias"  # added before the 1/sqrt(head_dim) scaling
                elif "MatMul" in out_consumers:
                    part = "attn_v_bias"  # added to the values before the attention-weighted sum
                else:
                    raise ValueError(f"cannot map per-head bias '{name}' (node '{add_node.name}')")
                if layer is None:
                    raise ValueError(f"cannot determine layer for per-head bias '{name}'")
                arr = g.inits[name].astype(F32, copy=False).reshape(-1)
                tensors[f"blk.{layer}.{part}"] = np.ascontiguousarray(arr)
                used.add(name)
                logger.debug(f"  {name} -> blk.{layer}.{part} {arr.shape}")

    unaccounted = set(g.inits) - used - set(skipped)
    if unaccounted:
        raise ValueError(f"unaccounted initializers: {sorted(unaccounted)}")

    return tensors, skipped


def load_config(model_dir):
    p = Path(model_dir) / "config.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _infer_heads(g: ModelGraph, hidden):
    """Infer the attention head count from the 1/sqrt(head_dim) scale constant.

    Used when config.json (e.g. distilbert) does not provide num_attention_heads.
    """
    for n in g.nodes:
        if n.op_type != "Div" or len(n.input) != 2:
            continue
        v = g.const_values.get(n.input[1])
        if v is None or v.size != 1:
            continue
        scale = float(v.ravel()[0])
        head_dim = round(scale * scale)
        if head_dim > 0 and hidden % head_dim == 0:
            return hidden // head_dim
    raise ValueError("cannot infer num_attention_heads from the graph")


def build_metadata(fam, tensors, cfg, g=None):
    """Build and cross-check the GGUF metadata for a converted model."""
    n_layers = max(int(n.split(".")[1]) for n in tensors if n.startswith("blk.")) + 1
    hidden = tensors["token_embd"].shape[1]
    vocab = tensors["token_embd"].shape[0]
    heads = cfg.get("num_attention_heads")
    if heads is None:
        if g is None:
            raise ValueError("config.json must provide num_attention_heads")
        heads = _infer_heads(g, hidden)
    heads = int(heads)
    if hidden % heads != 0:
        raise ValueError(f"hidden {hidden} not divisible by heads {heads}")
    max_length = int(cfg.get("max_position_embeddings", 0))

    if "num_hidden_layers" in cfg and int(cfg["num_hidden_layers"]) != n_layers:
        raise ValueError(f"config num_hidden_layers {cfg['num_hidden_layers']} != tensor layers {n_layers}")
    if "hidden_size" in cfg and int(cfg["hidden_size"]) != hidden:
        raise ValueError(f"config hidden_size {cfg['hidden_size']} != tensor hidden {hidden}")
    if "vocab_size" in cfg and int(cfg["vocab_size"]) != vocab:
        raise ValueError(f"config vocab_size {cfg['vocab_size']} != tensor vocab {vocab}")
    if "position_embd" in tensors:
        if max_length and tensors["position_embd"].shape[0] < max_length:
            raise ValueError("position_embd shorter than max_position_embeddings")

    if "id2label" in cfg:
        labels = [cfg["id2label"][k] for k in sorted(cfg["id2label"], key=int)]
    else:
        labels = DEFAULT_LABELS[fam]

    meta = {
        "n_layers": n_layers,
        "hidden": hidden,
        "heads": heads,
        "vocab": vocab,
        "has_type_embd": "type_embd" in tensors,
        "use_rope": "rope_embd" in tensors,
        "quantized": any(n.endswith("_q8") for n in tensors),
        "labels": labels,
        "max_length": max_length,
    }
    if "rope_embd" in tensors:
        meta["rope_max_pos"] = int(tensors["rope_embd"].shape[0])
    return meta


TOP_ORDER = ["token_embd", "position_embd", "type_embd", "rope_embd", "norm", "norm_bias",
             "pooler.weight", "pooler.bias", "cls.weight", "cls.bias"]
PART_ORDER = ["attn_q", "attn_q_bias", "attn_k", "attn_k_bias", "attn_v", "attn_v_bias",
              "attn_o", "attn_o_bias", "attn_norm", "attn_norm_bias",
              "ffn_0", "ffn_0_bias", "ffn_1", "ffn_1_bias", "ffn_norm", "ffn_norm_bias"]


def tensor_order(tensors):
    def key(n):
        if n.startswith("blk."):
            i, part = int(n.split(".")[1]), n.split(".", 2)[2]
            base, suffix = part, 0
            if base.endswith("_q8_scale"):
                base, suffix = base[: -len("_q8_scale")], 2
            elif base.endswith("_q8"):
                base, suffix = base[: -len("_q8")], 1
            return (1, i, PART_ORDER.index(base) * 10 + suffix)
        base, suffix = n, 0
        if base.endswith("_q8_scale"):
            base, suffix = base[: -len("_q8_scale")], 2
        elif base.endswith("_q8"):
            base, suffix = base[: -len("_q8")], 1
        return (0, TOP_ORDER.index(base) * 10 + suffix, 0)
    return sorted(tensors, key=key)


def convert(model_dir, save_path):
    model_dir = Path(model_dir)
    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    g = ModelGraph(onnx_path)
    fam = g.family()
    logger.info(f"converting {onnx_path} (family: {fam})")

    tensors, skipped = build_mapping(g)
    cfg = load_config(model_dir)
    meta = build_metadata(fam, tensors, cfg, g)

    writer = gguf.GGUFWriter(path=None, arch=ARCHITECTURE)
    writer.add_string("general.name", f"ruaccent-{fam}")
    for k, v in meta.items():
        if isinstance(v, bool):
            writer.add_bool(k, v)
        elif isinstance(v, int):
            writer.add_int32(k, v)
        elif isinstance(v, list):
            writer.add_array(k, v)
        else:
            writer.add_string(k, str(v))

    for name in tensor_order(tensors):
        arr = tensors[name]
        if arr.dtype == np.int8:
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.I8)
        else:
            writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)

    save_path = Path(save_path)
    os.makedirs(str(save_path.parent), exist_ok=True)
    writer.write_header_to_file(path=save_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    logger.info(f"wrote {save_path}: {len(tensors)} tensors")
    for name in tensor_order(tensors):
        logger.info(f"  {name} {list(tensors[name].shape)}")
    for name, reason in sorted(skipped.items()):
        logger.info(f"  skipped {name}: {reason}")
    return tensors, skipped, meta


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert a RUAccent ONNX model to GGUF.")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="directory containing model.onnx (and config.json)")
    parser.add_argument("--save-path", type=str, required=True, help="path of the GGUF file to write")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_arguments()
    convert(args.model_dir, args.save_path)
