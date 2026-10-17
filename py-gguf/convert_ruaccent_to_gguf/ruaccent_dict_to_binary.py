#!/usr/bin/env python3
"""Convert RUAccent json.gz dictionaries to the binary RUAC format.

Usage:
    python ruaccent_dict_to_binary.py <dict_dir> <out_dir>

Reads accents.json.gz, omographs.json.gz, yo_words.json.gz,
yo_homographs.json.gz from <dict_dir> and writes <name>.bin to <out_dir>.

Binary layout (little-endian):
    magic    b"RUAC"
    version  u32 = 1
    count    u64
    entries sorted by UTF-8 byte order:
        key_len u32, key bytes, val_len u32, value bytes
    List values (omographs) are '\n'-joined.
"""
import gzip
import json
import os
import struct
import sys

NAMES = ["accents", "omographs", "yo_words", "yo_homographs"]


def to_value(v):
    if isinstance(v, list):
        return "\n".join(str(x) for x in v).encode("utf-8")
    return str(v).encode("utf-8")


def convert(src, dst):
    with gzip.open(src, "rt", encoding="utf-8") as f:
        d = json.load(f)
    keys = sorted(d.keys())  # codepoint order == UTF-8 byte order
    with open(dst, "wb") as f:
        f.write(b"RUAC")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", len(keys)))
        for k in keys:
            kb = k.encode("utf-8")
            vb = to_value(d[k])
            f.write(struct.pack("<I", len(kb)))
            f.write(kb)
            f.write(struct.pack("<I", len(vb)))
            f.write(vb)
    print(f"{dst}: {len(keys)} entries, {os.path.getsize(dst)} bytes")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    dict_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for name in NAMES:
        convert(os.path.join(dict_dir, f"{name}.json.gz"),
                os.path.join(out_dir, f"{name}.bin"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
