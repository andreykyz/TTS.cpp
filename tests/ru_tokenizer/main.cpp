// Unit test for the RUAccent tokenizers (ru_tokenizer.cpp).
// Compares each tokenizer against reference outputs generated from the
// Python tokenizers (see ru_tokenizer_testdata.h, produced by gen_golden.py).
//
// Usage: ru_tokenizer-test [data_root]
//   data_root default: /home/andrey/model/ruaccent-data
#include <iostream>
#include <string>
#include <vector>

#include "../../src/models/kokoro/ru_tokenizer.h"
#include "ru_tokenizer_testdata.h"

using namespace ru;
using namespace ru_tok_test;

static int failures = 0;

static bool vec_eq(const std::vector<int> & a, const std::vector<int> & b) { return a == b; }
static bool off_eq(const std::vector<std::pair<int, int>> & a,
                   const std::vector<std::pair<int, int>> & b) { return a == b; }

static void report(const char * name, int idx, const std::string & field,
                   const std::string & got, const std::string & exp) {
    ++failures;
    std::cout << "FAIL " << name << " [" << idx << "] " << field << "\n"
              << "  exp: " << exp << "\n  got: " << got << "\n";
}

template <typename F>
static void check_ids(const char * name, F f, const std::vector<TokCase> & data,
                      bool check_offsets) {
    int ok = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        const TokCase & c = data[i];
        std::string in = c.in2.empty() ? c.in1 : (c.in1 + " || " + c.in2);
        auto got = f(c);
        auto show = [](const std::vector<int> & v) {
            std::string s;
            for (int x : v) { s += std::to_string(x); s += " "; }
            return s;
        };
        if (!vec_eq(got.input_ids, c.ids))
            report(name, (int) i, "ids (in=" + in + ")", show(got.input_ids), show(c.ids));
        else if (!vec_eq(got.attention_mask, c.attn))
            report(name, (int) i, "attn (in=" + in + ")", show(got.attention_mask), show(c.attn));
        else if (!c.type.empty() && !vec_eq(got.token_type_ids, c.type))
            report(name, (int) i, "type (in=" + in + ")", show(got.token_type_ids), show(c.type));
        else if (check_offsets && !off_eq(got.offsets, c.off)) {
            std::string g, e;
            for (auto & p : got.offsets) { g += std::to_string(p.first) + "," + std::to_string(p.second) + " "; }
            for (auto & p : c.off) { e += std::to_string(p.first) + "," + std::to_string(p.second) + " "; }
            report(name, (int) i, "off (in=" + in + ")", g, e);
        } else if (check_offsets && !vec_eq(got.special_mask, c.spec))
            report(name, (int) i, "spec (in=" + in + ")", show(got.special_mask), show(c.spec));
        else ++ok;
    }
    std::cout << name << ": " << ok << "/" << data.size() << " passed\n";
}

int main(int argc, char ** argv) {
    std::string root = argc > 1 ? argv[1] : "/home/andrey/model/ruaccent-data";
    std::string nn = root + "/nn";

    CharTokenizer char_tok;
    WordPieceTokenizer stress_tok, yo_tok;
    BPETokenizer omo_tok;

    if (!char_tok.load(nn + "/nn_accent/vocab.txt")) { std::cout << "load char FAIL\n"; return 2; }
    if (!stress_tok.load(nn + "/nn_stress_usage_predictor/vocab.txt")) { std::cout << "load stress FAIL\n"; return 2; }
    if (!yo_tok.load(nn + "/nn_yo_homograph_resolver/vocab.txt")) { std::cout << "load yo FAIL\n"; return 2; }
    if (!omo_tok.load(nn + "/nn_omograph/turbo3.1")) { std::cout << "load omo FAIL\n"; return 2; }

    std::cout << "vocab sizes: char=" << char_tok.vocab_size()
              << " stress=" << stress_tok.vocab_size()
              << " yo=" << yo_tok.vocab_size()
              << " omo=" << omo_tok.vocab_size() << "\n";

    check_ids("char",   [&](const TokCase & c) { return char_tok.tokenize(c.in1); },
              CHAR_CASES, false);
    check_ids("stress", [&](const TokCase & c) { return stress_tok.tokenize(c.in1); },
              STRESS_CASES, true);
    check_ids("yo",     [&](const TokCase & c) { return yo_tok.tokenize(c.in1); },
              YO_CASES, true);
    check_ids("omo",    [&](const TokCase & c) { return omo_tok.tokenize_pair(c.in1, c.in2); },
              OMO_CASES, false);

    if (failures == 0) { std::cout << "ALL TESTS PASSED\n"; return 0; }
    std::cout << failures << " FAILURE(S)\n";
    return 1;
}
