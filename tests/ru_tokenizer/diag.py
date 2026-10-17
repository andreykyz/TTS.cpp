#!/usr/bin/env python3
# Diagnostic: omograph BPE tokenization of simple strings.
import os
from transformers import AutoTokenizer
RUACCENT = "/home/andrey/model/tts/venv310/lib/python3.10/site-packages/ruaccent"
NN = os.path.join(RUACCENT, "nn")
t = AutoTokenizer.from_pretrained(os.path.join(NN, "nn_omograph", "turbo3.1"))
ft = t._tokenizer

for s in ["на", "а б", "У неё", "коса", "к+оса", "<w>коса</w>", "dлинная", "hello"]:
    enc = ft.encode(s)
    toks = enc.tokens
    print("IN   :", repr(s))
    print("TOKS :", toks)
    print("IDS  :", enc.ids)
    print()
