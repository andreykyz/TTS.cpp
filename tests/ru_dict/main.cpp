// Unit test for the RUAccent dictionary loader (ru_dict.cpp).
// Loads the binary RUAC files (produced by ruaccent_dict_to_binary.py) and
// checks lookups against reference values from the json.gz dictionaries
// (see ru_dict_testdata.h, produced by gen_golden.py).
//
// Usage: ru_dict-test [data_root]
//   data_root default: /home/andrey/model/ruaccent-data
#include <iostream>
#include <string>
#include <vector>

#include "../../src/models/kokoro/ru_dict.h"
#include "ru_dict_testdata.h"

using namespace ru;
using namespace ru_dict_test;

static int failures = 0;

static void report(const std::string & name, const std::string & key,
                   const std::string & got, const std::string & exp) {
    ++failures;
    std::cout << "FAIL " << name << " key=" << key << "\n"
              << "  exp: " << exp << "\n  got: " << got << "\n";
}

static void join(const std::vector<std::string> & v, std::string & out) {
    out.clear();
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += "\n";
        out += v[i];
    }
}

static void check_dict(const std::string & root, const std::string & name,
                       size_t exp_count, const std::vector<Case> & cases,
                       const std::string & missing, bool as_list) {
    Dict d;
    std::string path = root + "/dictionary/" + name + ".bin";
    if (!d.load(path)) {
        ++failures;
        std::cout << "FAIL " << name << " load " << path << "\n";
        return;
    }
    if (d.size() != exp_count) {
        ++failures;
        std::cout << "FAIL " << name << " count exp=" << exp_count
                  << " got=" << d.size() << "\n";
    }
    int ok = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const Case & c = cases[i];
        std::string got;
        if (as_list) {
            std::vector<std::string> parts;
            if (!d.get_list(c.key, parts)) {
                report(name, c.key, "<missing>", c.value);
                continue;
            }
            join(parts, got);
        } else if (!d.get(c.key, got)) {
            report(name, c.key, "<missing>", c.value);
            continue;
        }
        if (got != c.value) report(name, c.key, got, c.value);
        else ++ok;
    }
    std::string got;
    if (d.get(missing, got)) {
        ++failures;
        std::cout << "FAIL " << name << " missing key found: " << missing << "\n";
    }
    std::cout << name << ": " << ok << "/" << cases.size()
              << " lookups ok, count=" << d.size() << "\n";
}

int main(int argc, char ** argv) {
    std::string root = argc > 1 ? argv[1] : "/home/andrey/model/ruaccent-data";
    check_dict(root, "accents", ACCENTS_COUNT, ACCENTS_CASES, ACCENTS_MISSING, false);
    check_dict(root, "omographs", OMOGRAPHS_COUNT, OMOGRAPHS_CASES, OMOGRAPHS_MISSING, true);
    check_dict(root, "yo_words", YO_WORDS_COUNT, YO_WORDS_CASES, YO_WORDS_MISSING, false);
    check_dict(root, "yo_homographs", YO_HOMOGRAPHS_COUNT, YO_HOMOGRAPHS_CASES,
               YO_HOMOGRAPHS_MISSING, false);
    if (failures) {
        std::cout << "FAILURES: " << failures << "\n";
        return 1;
    }
    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
