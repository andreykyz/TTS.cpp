// End-to-end test for the Russian G2P pipeline as wired into kokoro
// (model.cpp kokoro_runner::generate). Runs the full pipeline:
//
//   text -> brackets -> RUAccent.process -> plus_to_acute -> lower -> respell
//        -> espeak(IPA) -> normalize -> reduce_vowels -> lengthen_sch
//
// and compares the IPA against the Python reference (kokoro-ru/ru_g2p.py
// RuG2P.phonemize), captured in ru_g2p_testdata.h by gen_oracle.py.
//
// The espeak step must use the acute-aware Russian data (it honors U+0301
// stress marks); the default below points at the kokoro-ru espeak-data dir.
//
// Usage: ru_g2p-test [data_dir] [models_dir] [espeak_data_dir]
//   data_dir        default: /home/andrey/model/ruaccent-data
//   models_dir      default: /home/andrey/model/ruaccent-gguf
//   espeak_data_dir default: /home/andrey/model/kokoro-ru/espeak-data
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../../src/models/kokoro/ru_accent.h"
#include "../../src/models/kokoro/ru_frontend.h"
#include "../../src/models/kokoro/phonemizer.h"
#include "ru_g2p_testdata.h"

using namespace ru;

// The exact pipeline used by kokoro_runner::generate for Russian voices.
static std::string ru_g2p(const RUAccent & accent, phonemizer * phmzr,
                          const std::string & prompt) {
    std::string text = prompt;
    text = brackets(text);
    text = accent.process(text);
    text = plus_to_acute(text);
    text = lower(text);
    text = respell(text);
    text = phmzr->text_to_phonemes(text);
    text = separate_punct(text);
    return lengthen_sch(reduce_vowels(normalize(text)));
}

int main(int argc, char ** argv) {
    std::string data_dir   = argc > 1 ? argv[1] : "/home/andrey/model/ruaccent-data";
    std::string models_dir = argc > 2 ? argv[2] : "/home/andrey/model/ruaccent-gguf";
    std::string espeak_dir = argc > 3 ? argv[3] : "/home/andrey/model/kokoro-ru/espeak-data";

    // espeak is a singleton initialized on first use; the data path must be
    // set before the phonemizer is created.
    if (const char * cur = std::getenv("TTS_ESPEAK_DATA_PATH"); !cur || !*cur)
        setenv("TTS_ESPEAK_DATA_PATH", espeak_dir.c_str(), 1);

    RUAccent accent;
    if (!accent.load(data_dir, models_dir)) {
        std::cout << "FAIL load (" << data_dir << ", " << models_dir << ")\n";
        return 1;
    }

    phonemizer * phmzr = espeak_phonemizer(false, "zle/ru");

    int failures = 0;
    for (std::size_t i = 0; i < ru_g2p_test::N; ++i) {
        const auto & c = ru_g2p_test::CASES[i];
        std::string got = ru_g2p(accent, phmzr, c.in);
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
    std::cout << "ALL TESTS PASSED (" << ru_g2p_test::N << ")\n";
    return 0;
}
