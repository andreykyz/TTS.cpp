// Unit test for the RUAccent stress pipeline (ru_accent.cpp).
// Loads the 4 GGUF models + 3 dictionaries + tokenizers, runs process() on a
// fixed set of Russian sentences, and compares the stressed output against the
// Python reference (ru_accent_testdata.h, produced by gen_oracle.py).
//
// Usage: ru_accent-test [data_dir] [models_dir]
//   data_dir   default: /home/andrey/model/ruaccent-data
//   models_dir default: /home/andrey/model/ruaccent-gguf
#include <cstddef>
#include <iostream>
#include <string>

#include "../../src/models/kokoro/ru_accent.h"
#include "ru_accent_testdata.h"

using namespace ru;

int main(int argc, char ** argv) {
    std::string data_dir = argc > 1 ? argv[1] : "/home/andrey/model/ruaccent-data";
    std::string models_dir = argc > 2 ? argv[2] : "/home/andrey/model/ruaccent-gguf";

    RUAccent ra;
    if (!ra.load(data_dir, models_dir)) {
        std::cout << "FAIL load (" << data_dir << ", " << models_dir << ")\n";
        return 1;
    }

    int failures = 0;
    for (std::size_t i = 0; i < ru_accent_test::N; ++i) {
        const auto & c = ru_accent_test::CASES[i];
        std::string got = ra.process(c.in);
        bool pass = (got == c.out);
        std::cout << (pass ? "PASS" : "FAIL") << " [" << i << "] " << c.in << "\n";
        if (!pass) {
            ++failures;
            std::cout << "  exp: " << c.out << "\n";
            std::cout << "  got: " << got << "\n";
        }
    }
    if (failures) {
        std::cout << "FAILURES: " << failures << "\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED (" << ru_accent_test::N << ")\n";
    return 0;
}
