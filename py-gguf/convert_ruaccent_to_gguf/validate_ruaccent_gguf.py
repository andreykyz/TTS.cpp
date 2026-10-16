#!/usr/bin/env python3
"""
Validate a converted RUAccent GGUF file against its ONNX source with tolerance 0.

Rebuilds the expected tensor mapping with convert_ruaccent_to_gguf.build_mapping
(the same logic the converter used) and compares every GGUF tensor against it:
name, shape and float32 values must match exactly. Also checks the metadata.

This script reads the GGUF binary directly with a minimal reader, because the
gguf 0.10.0 Python reader is incompatible with NumPy 2.0 (newbyteorder error).

Usage:
  python validate_ruaccent_gguf.py --gguf /path/accent.gguf --model-dir /path/to/nn_accent
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

from convert_ruaccent_to_gguf import ARCHITECTURE, ModelGraph, build_mapping, build_metadata, load_config

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

_SCALAR = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
    10: "<Q", 11: "<q", 12: "<d",
}


def _read_string(buf, off):
    # GGUF v3 string: uint64 length + raw bytes (no padding)
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    s = buf[off:off + n].decode("utf-8")
    off += n
    return s, off


def _read_value(buf, off, vtype):
    if vtype == 8:
        return _read_string(buf, off)
    if vtype == 9:
        (et,) = struct.unpack_from("<I", buf, off)
        off += 4
        (cnt,) = struct.unpack_from("<Q", buf, off)
        off += 8
        vals = []
        for _ in range(cnt):
            v, off = _read_value(buf, off, et)
            vals.append(v)
        return vals, off
    fmt = _SCALAR[vtype]
    v = struct.unpack_from(fmt, buf, off)[0]
    return v, off + struct.calcsize(fmt)


def read_gguf(path):
    """Minimal GGUF v3 reader.

    Returns (meta, [(name, shape, offset)], data_start, buf) where shape is the
    numpy (C-order) shape: GGUF stores dims in ggml order (fastest-varying first),
    i.e. reversed relative to the numpy shape. All fields are little-endian.
    """
    with open(path, "rb") as f:
        buf = f.read()
    if buf[:4] != GGUF_MAGIC:
        raise ValueError(f"{path}: bad magic {buf[:4]!r}")
    (version,) = struct.unpack_from("<I", buf, 4)
    if version != GGUF_VERSION:
        raise ValueError(f"{path}: unsupported GGUF version {version}")
    (tensor_count, kv_count) = struct.unpack_from("<QQ", buf, 8)
    off = 24
    meta = {}
    for _ in range(kv_count):
        key, off = _read_string(buf, off)
        (vt,) = struct.unpack_from("<I", buf, off)
        off += 4
        val, off = _read_value(buf, off, vt)
        meta[key] = val
    infos = []
    for _ in range(tensor_count):
        name, off = _read_string(buf, off)
        (nd,) = struct.unpack_from("<I", buf, off)
        off += 4
        ggml_dims = struct.unpack_from(f"<{nd}Q", buf, off)
        off += 8 * nd
        (dtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        (tensor_offset,) = struct.unpack_from("<Q", buf, off)
        off += 8
        if dtype not in (0, 24):  # GGML_F32, GGML_I8 (ggml type enum of gguf 0.10.0)
            raise ValueError(f"{path}: tensor '{name}' has unexpected ggml dtype {dtype}")
        infos.append((name, ggml_dims[::-1], tensor_offset, dtype))
    # The gguf 0.10.0 writer pads the absolute file position to 32 bytes before
    # the first tensor, but records tensor offsets relative to the unaligned end
    # of the tensor-info section, so the data base is align32(ti_end).
    data_base = ((off + 31) // 32) * 32
    return meta, infos, data_base, buf


_GGML_DTYPE = {0: np.float32, 24: np.int8}


def main():
    ap = argparse.ArgumentParser(description="Validate a ruaccent GGUF against its ONNX source.")
    ap.add_argument("--gguf", required=True, help="path to the GGUF file")
    ap.add_argument("--model-dir", required=True, help="directory containing model.onnx")
    args = ap.parse_args()

    g = ModelGraph(Path(args.model_dir) / "model.onnx")
    fam = g.family()
    expected, skipped = build_mapping(g)
    meta, infos, data_start, buf = read_gguf(args.gguf)

    ok = True
    by_name = {name: (dims, off, dtype) for name, dims, off, dtype in infos}
    print(f"model family : {fam}")
    print(f"gguf tensors : {len(infos)}   expected tensors: {len(expected)}")
    if len(infos) != len(expected):
        ok = False

    for name in sorted(expected):
        arr = expected[name]
        if name not in by_name:
            print(f"  MISSING      {name} (expected {arr.shape})")
            ok = False
            continue
        dims, off, dtype = by_name[name]
        if tuple(dims) != tuple(arr.shape):
            print(f"  SHAPE MISMATCH {name}: gguf={dims} onnx={arr.shape}")
            ok = False
            continue
        if _GGML_DTYPE[dtype] != arr.dtype:
            print(f"  DTYPE MISMATCH {name}: gguf={_GGML_DTYPE[dtype]} expected={arr.dtype}")
            ok = False
            continue
        n = int(np.prod(dims))
        data = np.frombuffer(buf, dtype=_GGML_DTYPE[dtype], count=n, offset=data_start + off).reshape(dims)
        if not np.array_equal(data, arr):
            diff = np.abs(data.astype(np.float64) - arr.astype(np.float64))
            print(f"  VALUE MISMATCH {name}: max_abs_diff={diff.max():.3e}")
            ok = False
            continue
        print(f"  OK           {name} {list(dims)}")

    extra = set(by_name) - set(expected)
    if extra:
        print(f"  EXTRA tensors: {sorted(extra)}")
        ok = False

    if meta.get("general.architecture") != ARCHITECTURE:
        print(f"  META MISMATCH general.architecture: {meta.get('general.architecture')!r}")
        ok = False

    exp_meta = build_metadata(fam, expected, load_config(args.model_dir), g)
    for k, v in exp_meta.items():
        got = meta.get(k)
        if got != v:
            print(f"  META MISMATCH {k}: gguf={got!r} expected={v!r}")
            ok = False

    if skipped:
        for name, reason in sorted(skipped.items()):
            print(f"  skipped init : {name} ({reason})")
    else:
        print("  skipped init : none")

    if ok:
        print(f"RESULT: PASS ({len(expected)} tensors, all shapes and values match, tolerance 0)")
    else:
        print("RESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
