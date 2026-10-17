#!/usr/bin/env python3
# Generates ru_tokenizer_testdata.h (golden tokenization outputs) from the
# reference Python tokenizers (venv310). Run on the server:
#   cd <repo>/tests/ru_tokenizer
#   CUDA_VISIBLE_DEVICES="" /home/andrey/model/tts/venv310/bin/python gen_golden.py
# The emitted header is committed; the C++ tokenizers must reproduce it exactly.
import importlib.util
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
RUACCENT = "/home/andrey/model/tts/venv310/lib/python3.10/site-packages/ruaccent"
NN = os.path.join(RUACCENT, "nn")


def load_char_tokenizer():
    spec = importlib.util.spec_from_file_location(
        "char_tokenizer", os.path.join(RUACCENT, "char_tokenizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CharTokenizer.from_pretrained(os.path.join(NN, "nn_accent"))


def wp_ids(t, text, want_offsets):
    if want_offsets:
        out = t(text, return_offsets_mapping=True, return_special_tokens_mask=True,
                return_tensors="np")
        off = [[int(a), int(b)] for a, b in out["offset_mapping"][0].tolist()]
        spec = [int(x) for x in out["special_tokens_mask"][0].tolist()]
    else:
        out = t(text, return_tensors="np")
        off, spec = [], []
    return {
        "ids": [int(x) for x in out["input_ids"][0].tolist()],
        "attn": [int(x) for x in out["attention_mask"][0].tolist()],
        "type": ([int(x) for x in out["token_type_ids"][0].tolist()]
                 if "token_type_ids" in out else []),
        "off": off,
        "spec": spec,
    }


def bpe_ids(t, text, hyp):
    out = t(text, hyp, max_length=512, truncation=True, return_tensors="np")
    return {
        "ids": [int(x) for x in out["input_ids"][0].tolist()],
        "attn": [int(x) for x in out["attention_mask"][0].tolist()],
        "type": ([int(x) for x in out["token_type_ids"][0].tolist()]
                 if "token_type_ids" in out else []),
        "off": [],
        "spec": [],
    }


# ---------------------------------------------------------------------------
# Corpus (diverse: ё, punctuation, numbers, long words, subwords, mixed case)
# ---------------------------------------------------------------------------
CHAR_WORDS = [
    "мама", "папа", "дом", "окно", "солнце", "красивый", "добрый", "злой",
    "большой", "маленький", "ёжик", "ёлка", "ёлочка", "ё", "а", "ы", "у",
    "обучение", "прекрасный", "исследование", "телефон", "компьютер",
    "интернет", "щавель", "щёлка", "ма-ма", "кто", "что", "как", "молоко",
    "хлеб", "вода", "огонь", "земля", "небо", "ветер", "дождь", "снег", "лёд",
]

STRESS_SENTENCES = [
    "Здравствуйте! Как ваши дела?",
    "Это было большое окно.",
    "Замок открыт, а замок на двери.",
    "Я чувствую себя очень счастливо.",
    "Солнце светит ярко сегодня.",
    "Вчера было скучно и дорого.",
    "Много людей пришло на праздник.",
    "Яичница с беконом на завтрак.",
    "Конечно, я помогу тебе.",
    "Щавель и щётка на столе.",
    "Она сказала, что это было интересно.",
    "Привет мир и земля!",
    "Ёжик ел яблоки в лесу.",
    "Кот спит на подоконнике.",
    "Мама мыла раму.",
    "У него было 3 книги и 2 тетради.",
    "Он сказал: «Привет»!",
    "Съел бы я шашлык.",
    "Щётка чистит щавель.",
    "Аист прилетел в марте.",
]

OMO_PAIRS = [
    ("У неё была <w>коса</w> длинная", "к+оса"),
    ("У неё была <w>коса</w> длинная", "кос+а"),
    ("Он <w>начал</w> работать утром", "н+ачал"),
    ("Он <w>начал</w> работать утром", "нач+ал"),
    ("Это <w>окно</w> большое", "о+кно"),
    ("Это <w>окно</w> большое", "окн+о"),
    ("Она <w>несла</w> сумку", "н+есла"),
    ("Она <w>несла</w> сумку", "несл+а"),
    ("Мы <w>пришли</w> вовремя", "пр+ишли"),
    ("Мы <w>пришли</w> вовремя", "пришл+и"),
    ("В <w>округе</w> много лесов", "о+круге"),
    ("В <w>округе</w> много лесов", "округ+е"),
]


def c_str(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def emit_case(f, c):
    def iv(v):
        return "{" + ",".join(str(x) for x in v) + "}"
    def io(v):
        return "{" + ",".join("{" + str(a) + "," + str(b) + "}" for a, b in v) + "}"
    f.write('    { "%s", "%s", %s, %s, %s, %s, %s },\n' % (
        c_str(c["in1"]), c_str(c["in2"]),
        iv(c["ids"]), iv(c["attn"]), iv(c["type"]), iv(c["spec"]), io(c["off"])))


def main():
    char_tok = load_char_tokenizer()
    stress_tok = AutoTokenizer.from_pretrained(os.path.join(NN, "nn_stress_usage_predictor"))
    yo_tok = AutoTokenizer.from_pretrained(os.path.join(NN, "nn_yo_homograph_resolver"))
    omo_tok = AutoTokenizer.from_pretrained(os.path.join(NN, "nn_omograph", "turbo3.1"))

    char_cases, stress_cases, yo_cases, omo_cases = [], [], [], []

    for w in CHAR_WORDS:
        out = char_tok(w, return_tensors="np")
        char_cases.append({
            "in1": w, "in2": "",
            "ids": [int(x) for x in out["input_ids"][0].tolist()],
            "attn": [int(x) for x in out["attention_mask"][0].tolist()],
            "type": ([int(x) for x in out["token_type_ids"][0].tolist()]
                     if "token_type_ids" in out else []),
            "off": [], "spec": [],
        })

    for s in STRESS_SENTENCES:
        c = wp_ids(stress_tok, s, want_offsets=True)
        c["in1"], c["in2"] = s, ""
        stress_cases.append(c)
        cy = wp_ids(yo_tok, s.lower(), want_offsets=True)
        cy["in1"], cy["in2"] = s.lower(), ""
        yo_cases.append(cy)

    for text, hyp in OMO_PAIRS:
        c = bpe_ids(omo_tok, text, hyp)
        c["in1"], c["in2"] = text, hyp
        omo_cases.append(c)

    out_path = os.path.join(HERE, "ru_tokenizer_testdata.h")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// AUTO-GENERATED by gen_golden.py -- do not edit by hand.\n")
        f.write("#pragma once\n#include <string>\n#include <utility>\n#include <vector>\n")
        f.write("namespace ru_tok_test {\n")
        f.write("struct TokCase {\n")
        f.write("    std::string in1, in2;\n")
        f.write("    std::vector<int> ids, attn, type, spec;\n")
        f.write("    std::vector<std::pair<int,int>> off;\n")
        f.write("};\n")
        for name, cases in [("CHAR_CASES", char_cases), ("STRESS_CASES", stress_cases),
                            ("YO_CASES", yo_cases), ("OMO_CASES", omo_cases)]:
            f.write("static const std::vector<TokCase> %s = {\n" % name)
            for c in cases:
                emit_case(f, c)
            f.write("};\n")
            f.write("static const int %s_N = %d;\n" % (name, len(cases)))
        f.write("}  // namespace ru_tok_test\n")

    print("wrote %s" % out_path)
    print("char=%d stress=%d yo=%d omo=%d" % (
        len(char_cases), len(stress_cases), len(yo_cases), len(omo_cases)))


if __name__ == "__main__":
    main()
