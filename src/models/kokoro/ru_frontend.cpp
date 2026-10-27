#include "ru_frontend.h"

#include <algorithm>

namespace ru {

// ------------------------------------------------------------------ UTF-8
std::vector<uint32_t> utf8_decode(const std::string & s) {
    std::vector<uint32_t> out;
    const unsigned char * p = reinterpret_cast<const unsigned char *>(s.data());
    const size_t n = s.size();
    for (size_t i = 0; i < n;) {
        const unsigned char c = p[i];
        uint32_t cp;
        int extra;
        if (c < 0x80)      { cp = c;      extra = 0; }
        else if (c < 0xE0) { cp = c&0x1F; extra = 1; }
        else if (c < 0xF0) { cp = c&0x0F; extra = 2; }
        else               { cp = c&0x07; extra = 3; }
        ++i;
        for (int k = 0; k < extra && i < n; ++k, ++i)
            cp = (cp << 6) | (p[i] & 0x3F);
        out.push_back(cp);
    }
    return out;
}

std::string utf8_encode(const std::vector<uint32_t> & cps) {
    std::string out;
    out.reserve(cps.size() * 2);
    for (const uint32_t cp : cps) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

namespace {

constexpr uint32_t ACUTE = 0x0301;

// Russian vowels (lower + upper) for plus_to_acute / acute re-seating.
const std::vector<uint32_t> & ru_vowels() {
    static const std::vector<uint32_t> v = {
        0x430,0x435,0x451,0x438,0x43E,0x443,0x44B,0x44D,0x44E,0x44F,  // аеёиоуыэюя
        0x410,0x414,0x401,0x418,0x41E,0x423,0x42B,0x42D,0x42E,0x42F,  // АЕЁИОУЫЭЮЯ
    };
    return v;
}

// Lower-case Russian vowels (for counting stressed-vowel ordinal).
const std::vector<uint32_t> & ru_vowels_lower() {
    static const std::vector<uint32_t> v = {
        0x430,0x435,0x451,0x438,0x43E,0x443,0x44B,0x44D,0x44E,0x44F,  // аеёиоуыэюя
    };
    return v;
}

bool in_set(const std::vector<uint32_t> & set, uint32_t cp) {
    return std::find(set.begin(), set.end(), cp) != set.end();
}

// Insert the acute after the N-th (1-indexed) vowel of `target`, where N is the
// ordinal of the stressed vowel in `src`. Mirrors ru_g2p._respell_word.
std::vector<uint32_t> reseat_acute(const std::vector<uint32_t> & src,
                                   const std::vector<uint32_t> & target) {
    const auto & lv = ru_vowels_lower();
    auto idx = std::find(src.begin(), src.end(), ACUTE);
    if (idx == src.end() || idx == src.begin())
        return target;
    int vowel_n = 0;
    for (auto it = src.begin(); it != idx; ++it)
        if (in_set(lv, *it)) ++vowel_n;
    std::vector<uint32_t> out;
    out.reserve(target.size() + 1);
    int n = 0;
    for (const uint32_t cp : target) {
        out.push_back(cp);
        if (in_set(lv, cp)) {
            ++n;
            if (n == vowel_n) out.push_back(ACUTE);
        }
    }
    return out;
}

std::vector<uint32_t> strip_acute(const std::vector<uint32_t> & cps) {
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    for (const uint32_t cp : cps)
        if (cp != ACUTE) out.push_back(cp);
    return out;
}

}  // namespace

// ------------------------------------------------------------------ brackets
std::string brackets(const std::string & text) {
    std::string out = text;
    for (char & ch : out) {
        if (ch == '[' || ch == '{') ch = '(';
        else if (ch == ']' || ch == '}') ch = ')';
    }
    return out;
}

// ------------------------------------------------------------------ plus_to_acute
std::string plus_to_acute(const std::string & text) {
    const std::vector<uint32_t> cps = utf8_decode(text);
    const auto & vowels = ru_vowels();
    std::vector<uint32_t> out;
    out.reserve(cps.size() + 8);
    for (size_t i = 0; i < cps.size(); ++i) {
        if (cps[i] == '+' && i + 1 < cps.size() && in_set(vowels, cps[i + 1])) {
            out.push_back(cps[i + 1]);
            out.push_back(ACUTE);
            ++i;
        } else {
            out.push_back(cps[i]);
        }
    }
    return utf8_encode(out);
}

// ------------------------------------------------------------------ lower
std::string lower(const std::string & text) {
    const std::vector<uint32_t> cps = utf8_decode(text);
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    for (const uint32_t cp : cps) {
        if (cp >= 0x410 && cp <= 0x42F)      out.push_back(cp + 0x20);  // А-Я -> а-я
        else if (cp == 0x401)                out.push_back(0x451);       // Ё -> ё
        else if (cp >= 0x41 && cp <= 0x5A)   out.push_back(cp + 0x20);   // A-Z -> a-z
        else                                 out.push_back(cp);
    }
    return utf8_encode(out);
}

// ------------------------------------------------------------------ respell
namespace {

// Whole-word respellings (bare lower-case -> respelled). Checked first.
const std::vector<std::pair<std::string, std::string>> & respell_word() {
    static const std::vector<std::pair<std::string, std::string>> m = {
        {"дорого", "дорога"},
        {"недорого", "недорога"},
        {"настрого", "настрога"},
        {"конечно", "конешно"},
        {"скучно", "скушно"},
        {"скучный", "скушный"},
        {"нарочно", "нарошно"},
        {"яичница", "яишница"},
        {"скворечник", "скворешник"},
        {"девичник", "девишник"},
        {"двоечник", "двоешник"},
        {"троечник", "троешник"},
        {"прачечная", "прашечная"},
        {"горчичник", "горчишник"},
        {"пустячный", "пустяшный"},
    };
    return m;
}

// Words where word-final -ого/-его keeps a real [g] (no -ово/-ево respell).
const std::vector<std::string> & ogo_blacklist() {
    static const std::vector<std::string> b = {
        "много", "немного", "намного", "ненамного", "строго", "нестрого",
        "настрого", "дорого", "недорого", "полого", "убого", "лего",
        "диего", "ого", "огого",
    };
    return b;
}

// Substring respellings (old -> new). Consonant-deleting only, so the stressed
// vowel ordinal survives.
const std::vector<std::pair<std::string, std::string>> & respell_substr() {
    static const std::vector<std::pair<std::string, std::string>> m = {
        {"солнц", "сонц"},
        {"чувств", "чуств"},
        {"здравств", "здраств"},
        {"счастлив", "счаслив"},
        {"завистлив", "завислив"},
        {"участлив", "учаслив"},
        {"совестлив", "совеслив"},
    };
    return m;
}

bool is_ru_letter(uint32_t cp) {
    return (cp >= 0x430 && cp <= 0x44F) || cp == 0x451;  // а-я + ё
}

// A "word" for respelling is a run of Russian letters plus the acute stress
// mark (mirrors _WORD_RE = [а-яё́]+). The acute stays inside the word so the
// whole word (with its stress) is respelled as a unit.
bool is_word_char(uint32_t cp) {
    return is_ru_letter(cp) || cp == ACUTE;
}

// _OGO_RE = ([а-яё́]*?[ое]́?)го(́?)$ : word ends in [о|е](acute?) г о (acute?).
// Returns the index of the 'г' if the rule matches, else -1.
int ogo_match(const std::vector<uint32_t> & w) {
    // locate the trailing "го" (г=0x433, о=0x43E), optionally followed by acute
    size_t g = std::string::npos;
    if (w.size() >= 2 && w[w.size() - 2] == 0x433 && w[w.size() - 1] == 0x43E)
        g = w.size() - 2;
    else if (w.size() >= 3 && w[w.size() - 3] == 0x433 && w[w.size() - 2] == 0x43E &&
             w[w.size() - 1] == ACUTE)
        g = w.size() - 3;
    else
        return -1;
    // before г there must be [о|е], optionally with an acute
    size_t oe = g;
    if (oe == 0) return -1;
    --oe;
    if (w[oe] == ACUTE) {
        if (oe == 0) return -1;
        --oe;
    }
    if (w[oe] != 0x43E && w[oe] != 0x435) return -1;
    // everything before the [о|е] must be Russian letters / acute (the *? part)
    for (size_t i = 0; i < oe; ++i)
        if (!is_ru_letter(w[i]) && w[i] != ACUTE) return -1;
    return static_cast<int>(g);
}

std::vector<uint32_t> respell_word_cp(std::vector<uint32_t> word) {
    std::vector<uint32_t> bare = strip_acute(word);
    const std::string bare_s = utf8_encode(bare);

    // 1) whole-word respelling takes precedence
    for (const auto & kv : respell_word()) {
        if (bare_s == kv.first)
            return reseat_acute(word, utf8_decode(kv.second));
    }

    // 2) word-final -ого/-его -> -ово/-ево (unless blacklisted)
    const bool blacklisted =
        std::find(ogo_blacklist().begin(), ogo_blacklist().end(), bare_s) !=
        ogo_blacklist().end();
    if (!blacklisted) {
        const int g = ogo_match(word);
        if (g != -1) {
            word[static_cast<size_t>(g)] = 0x432;  // г -> в
            bare = strip_acute(word);
        }
    }

    // 3) substring respellings (consonant-deleting)
    const std::string cur_bare = utf8_encode(bare);
    bool any = false;
    for (const auto & kv : respell_substr())
        if (cur_bare.find(kv.first) != std::string::npos) { any = true; break; }
    if (any) {
        std::string out = cur_bare;
        for (const auto & kv : respell_substr()) {
            size_t pos = 0;
            while ((pos = out.find(kv.first, pos)) != std::string::npos) {
                out.replace(pos, kv.first.size(), kv.second);
                pos += kv.second.size();
            }
        }
        std::vector<uint32_t> out_cp = utf8_decode(out);
        // re-seat the acute only if the original word had one (vowel_n > 0)
        auto idx = std::find(word.begin(), word.end(), ACUTE);
        if (idx != word.end() && idx != word.begin())
            return reseat_acute(word, out_cp);
        return out_cp;
    }

    return word;
}

}  // namespace

std::string respell(const std::string & text) {
    const std::vector<uint32_t> cps = utf8_decode(text);
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    std::vector<uint32_t> word;
    auto flush = [&]() {
        if (!word.empty()) {
            word = respell_word_cp(word);
            out.insert(out.end(), word.begin(), word.end());
            word.clear();
        }
    };
    for (const uint32_t cp : cps) {
        if (is_word_char(cp))
            word.push_back(cp);
        else {
            flush();
            out.push_back(cp);
        }
    }
    flush();
    return utf8_encode(out);
}

// ------------------------------------------------------------------ collapse_ws
// All whitespace we care about is ASCII (space, \n, \r, \t, \v, \f), so a byte
// scan is safe: UTF-8 multibyte sequences never contain bytes < 0x20, so we
// cannot split a multibyte character. Collapse runs to one space and trim ends.
std::string collapse_ws(const std::string & text) {
    auto is_ws = [](char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f';
    };
    std::string out;
    out.reserve(text.size());
    bool prev_ws = false;
    for (char c : text) {
        if (is_ws(c)) {
            if (!out.empty() && !prev_ws) out.push_back(' ');
            prev_ws = true;
        } else {
            out.push_back(c);
            prev_ws = false;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// ------------------------------------------------------------------ separate_punct
std::string separate_punct(const std::string & ipa) {
    const std::vector<uint32_t> cps = utf8_decode(ipa);
    std::vector<uint32_t> out;
    out.reserve(cps.size() + 8);
    for (size_t i = 0; i < cps.size(); ++i) {
        out.push_back(cps[i]);
        const bool is_punct =
            cps[i] == 0x21 || cps[i] == 0x3F || cps[i] == 0x2E ||  // ! ? .
            cps[i] == 0x2C || cps[i] == 0x3B || cps[i] == 0x3A;   // , ; :
        if (is_punct && i + 1 < cps.size() && cps[i + 1] != 0x20)
            out.push_back(0x20);
    }
    return utf8_encode(out);
}

// ------------------------------------------------------------------ normalize
std::string normalize(const std::string & ipa) {
    // (old codepoints, new codepoints), applied in order.
    // The first two fold espeak's two-symbol affricates (ts, tʃ) into the
    // single IPA symbols the model vocab uses (ʦ, ʧ) -- misaki's EspeakG2P
    // does this via its e2m table; TTS.cpp's espeak mode emits them untied.
    static const std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>> map = {
        {{0x74, 0x73}, {0x2A6}},      // ts -> ʦ
        {{0x74, 0x283}, {0x2A7}},     // tʃ -> ʧ
        {{0x75, 0x22}, {0x75}},       // u" -> u
        {{0x26D},      {0x6C}},       // ɭ -> l
        {{0x275},      {0x6F}},       // ɵ -> o
        {{0x291},      {0x292}},      // ʑ -> ʒ
        {{0x290},      {0x292}},      // ʐ -> ʒ
        {{0x2A7, 0x2B2}, {0x2A7}},    // ʧʲ -> ʧ
    };
    std::vector<uint32_t> cps = utf8_decode(ipa);
    for (const auto & kv : map) {
        const auto & old = kv.first;
        const auto & neu = kv.second;
        std::vector<uint32_t> next;
        next.reserve(cps.size());
        for (size_t i = 0; i < cps.size();) {
            if (i + old.size() <= cps.size() &&
                std::equal(old.begin(), old.end(), cps.begin() + i)) {
                next.insert(next.end(), neu.begin(), neu.end());
                i += old.size();
            } else {
                next.push_back(cps[i++]);
            }
        }
        cps = std::move(next);
    }
    return utf8_encode(cps);
}

// ------------------------------------------------------------------ lengthen_sch
std::string lengthen_sch(const std::string & ipa) {
    // ɕ (0x255) not followed by ː (0x2D0) -> ɕː
    const std::vector<uint32_t> cps = utf8_decode(ipa);
    std::vector<uint32_t> out;
    out.reserve(cps.size() + 4);
    for (size_t i = 0; i < cps.size(); ++i) {
        out.push_back(cps[i]);
        if (cps[i] == 0x255 && (i + 1 >= cps.size() || cps[i + 1] != 0x2D0))
            out.push_back(0x2D0);
    }
    return utf8_encode(out);
}

// ------------------------------------------------------------------ reduce_vowels
namespace {

const std::vector<uint32_t> & ipa_vowels() {
    // _VOWELS = a ɑ o e i u y ʌ ə ɪ ɐ ɛ ɨ
    static const std::vector<uint32_t> v = {
        0x61,0x251,0x6F,0x65,0x69,0x75,0x79,0x28C,0x259,0x26A,0x250,0x25B,0x268,
    };
    return v;
}

const std::vector<uint32_t> & reducible() {
    // _REDUCIBLE = a ɑ o ʌ ə
    static const std::vector<uint32_t> v = {0x61,0x251,0x6F,0x28C,0x259};
    return v;
}

std::vector<uint32_t> reduce_token(std::vector<uint32_t> chars) {
    const auto & vowels = ipa_vowels();
    const auto & red = reducible();
    std::vector<size_t> vpos;
    for (size_t i = 0; i < chars.size(); ++i)
        if (in_set(vowels, chars[i])) vpos.push_back(i);
    if (vpos.empty()) return chars;

    auto primary_it = std::find(chars.begin(), chars.end(), 0x2C8);
    const bool has_primary = primary_it != chars.end();
    const int primary = has_primary ? static_cast<int>(primary_it - chars.begin()) : -1;

    // stressed vowel positions
    std::vector<bool> is_stressed(chars.size(), false);
    for (size_t i = 0; i < chars.size(); ++i) {
        const uint32_t c = chars[i];
        if (c != 0x2C8 && c != 0x2CC) continue;
        if (c == 0x2CC && has_primary && static_cast<int>(i) > primary) continue;
        for (size_t j : vpos)
            if (static_cast<int>(j) > static_cast<int>(i)) { is_stressed[j] = true; break; }
    }
    // stressed vowel ordinals
    std::vector<int> stressed_ord;
    for (size_t k = 0; k < vpos.size(); ++k)
        if (is_stressed[vpos[k]]) stressed_ord.push_back(static_cast<int>(k));

    for (size_t ordinal = 0; ordinal < vpos.size(); ++ordinal) {
        const size_t i = vpos[ordinal];
        if (is_stressed[i] || !in_set(red, chars[i])) continue;
        if (i > 0 && (chars[i - 1] == 0x2B2 || chars[i - 1] == 0x6A)) {
            chars[i] = 0x26A;  // ɪ (unstressed я/е after palatal)
            continue;
        }
        int next_stress = -1;
        for (int s : stressed_ord)
            if (static_cast<int>(s) > static_cast<int>(ordinal)) { next_stress = s; break; }
        const bool absolute_initial = (i == 0);
        chars[i] = (next_stress == static_cast<int>(ordinal) + 1 || absolute_initial)
                       ? 0x250  // ɐ
                       : 0x259; // ə
    }

    // drop artifact secondary stresses sitting after the primary
    if (has_primary) {
        std::vector<uint32_t> kept;
        kept.reserve(chars.size());
        for (size_t k = 0; k < chars.size(); ++k)
            if (!(chars[k] == 0x2CC && static_cast<int>(k) > primary))
                kept.push_back(chars[k]);
        chars = std::move(kept);
    }
    return chars;
}

}  // namespace

std::string reduce_vowels(const std::string & ipa) {
    const std::vector<uint32_t> cps = utf8_decode(ipa);
    std::vector<uint32_t> out;
    out.reserve(cps.size());
    std::vector<uint32_t> tok;
    auto flush = [&]() {
        if (!tok.empty()) {
            tok = reduce_token(std::move(tok));
            out.insert(out.end(), tok.begin(), tok.end());
            tok.clear();
        }
    };
    for (const uint32_t cp : cps) {
        if (cp == 0x20) { flush(); out.push_back(cp); }
        else tok.push_back(cp);
    }
    flush();
    return utf8_encode(out);
}

}  // namespace ru
