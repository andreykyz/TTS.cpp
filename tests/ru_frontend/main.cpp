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

    if (failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cout << failures << " FAILURE(S)\n";
    return 1;
}
