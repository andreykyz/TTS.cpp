// Unit test for the Russian G2P frontend (ru_frontend.cpp).
// Compares each pure-text function against reference outputs generated from
// kokoro-ru/ru_g2p.py (see ru_frontend_testdata.h).
#include <iostream>
#include <string>
#include <utility>

#include "../../src/models/kokoro/ru_frontend.h"
#include "ru_frontend_testdata.h"

using namespace ru;
using namespace ru_test;

static int failures = 0;

template <typename F>
static void check(const char * name, F f, const std::pair<std::string, std::string> * data, int n) {
    int ok = 0;
    for (int i = 0; i < n; ++i) {
        const std::string got = f(data[i].first);
        if (got != data[i].second) {
            ++failures;
            std::cout << "FAIL " << name << " [" << i << "]\n"
                      << "  in:  " << data[i].first << "\n"
                      << "  exp: " << data[i].second << "\n"
                      << "  got: " << got << "\n";
        } else {
            ++ok;
        }
    }
    std::cout << name << ": " << ok << "/" << n << " passed\n";
}

int main() {
    check("brackets",      brackets,      BRACKETS,  BRACKETS_N);
    check("plus_to_acute", plus_to_acute, PLUS,      PLUS_N);
    check("respell",       respell,       RESPELL,   RESPELL_N);
    check("normalize",     normalize,     NORMALIZE, NORMALIZE_N);
    check("lengthen_sch",  lengthen_sch,  LENGTHEN,  LENGTHEN_N);
    check("reduce_vowels", reduce_vowels, REDUCE,    REDUCE_N);

    // collapse_ws and map_dashes are C++-only helpers (no ru_g2p.py reference),
    // so they are checked against hand-written expectations.
    {
        const std::pair<std::string, std::string> cw[] = {
            {"a\n\nb",     "a b"},
            {"a  \t\n  b", "a b"},
            {"  leading",  "leading"},
            {"trailing  ", "trailing"},
            {"a\r\nb",     "a b"},
            {"one two",    "one two"},
        };
        check("collapse_ws", collapse_ws, cw, (int)(sizeof(cw) / sizeof(cw[0])));

        const std::pair<std::string, std::string> md[] = {
            {"что-то",                 "что-то"},                  // intra-word hyphen kept
            {"1-й",                    "1-й"},                     // digit-letter hyphen kept
            {"Красно-донской",         "Красно-донской"},          // letter-letter kept
            {"- Ох",                   "— Ох"},                    // leading dialogue dash
            {"бабуля! - прокричала",   "бабуля! — прокричала"},   // spaced hyphen as dash
            {"текст – скобка",         "текст — скобка"},          // en-dash -> em-dash
            {"уже — эм",               "уже — эм"},                // em-dash unchanged
        };
        check("map_dashes", map_dashes, md, (int)(sizeof(md) / sizeof(md[0])));
    }

    // pause_sentinels: each protected mark folds to one control byte and back;
    // ordinary text passes through untouched.
    {
        int bad = 0;
        struct { const char * mark; char s; } cases[] = {
            { "\xe2\x80\x94", '\x01' },  // em-dash
            { "\"",          '\x02' },  // straight double quote
            { "\xe2\x80\xa6", '\x03' },  // ellipsis
            { "\xe2\x80\x9c", '\x04' },  // left double quote
            { "\xe2\x80\x9d", '\x05' },  // right double quote
        };
        for (const auto & c : cases) {
            const std::string e = pause_sentinels_encode(c.mark);
            if (e.size() != 1 || e[0] != c.s) ++bad;                       // encode -> sentinel
            if (pause_sentinels_decode(std::string(1, c.s)).find(c.mark) == std::string::npos) ++bad;  // decode -> real char
        }
        if (pause_sentinels_encode("привет мир") != "привет мир") ++bad;   // plain text unchanged
        if (pause_sentinels_decode("привет мир") != "привет мир") ++bad;
        if (bad) { ++failures; std::cout << "FAIL pause_sentinels (" << bad << ")\n"; }
        else     std::cout << "pause_sentinels: all passed\n";
    }

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cout << failures << " FAILURE(S)\n";
    return 1;
}
