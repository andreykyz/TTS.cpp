// RUAccent stress pipeline (see ru_accent.h).
//
// Faithful port of the reference Python `RUAccent.process_all_internal`
// (ruaccent 1.5.8.3). Verified against tests/ru_accent/gen_oracle.py.
#include "ru_accent.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "ru_frontend.h"   // ru::utf8_decode / ru::utf8_encode

namespace ru {
namespace {

// ---------------------------------------------------------------------------
// codepoint helpers
// ---------------------------------------------------------------------------
bool is_space_cp(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}
bool is_word_cp(uint32_t cp) {   // Python \w (letters, digits, underscore)
    if (cp == '_') return true;
    if (cp >= '0' && cp <= '9') return true;
    if (cp >= 'a' && cp <= 'z') return true;
    if (cp >= 'A' && cp <= 'Z') return true;
    if (cp >= 0x400 && cp <= 0x4FF) return true;   // Cyrillic
    if (cp >= 0xC0  && cp <= 0x24F) return true;   // Latin extended
    if (cp >= 0x370 && cp <= 0x3FF) return true;   // Greek
    return false;
}
bool is_upper_cp(uint32_t cp) {
    return (cp >= 0x410 && cp <= 0x42F) || (cp >= 0x41 && cp <= 0x5A);
}
uint32_t to_upper_cp(uint32_t cp) {
    if (cp >= 0x430 && cp <= 0x44F) cp -= 0x20;
    else if (cp >= 0x61 && cp <= 0x7A) cp -= 0x20;
    return cp;
}
uint32_t to_lower_cp(uint32_t cp) {
    if (cp >= 0x410 && cp <= 0x42F) cp += 0x20;
    else if (cp >= 0x41 && cp <= 0x5A) cp += 0x20;
    return cp;
}
std::string to_lower(const std::string & s) {
    auto cps = utf8_decode(s);
    for (auto & cp : cps) cp = to_lower_cp(cp);
    return utf8_encode(cps);
}
bool is_omograph_punct(uint32_t cp) {
    return cp == '.' || cp == ',' || cp == '!' || cp == '?' || cp == ':' || cp == ';' || cp == 0x2026;
}

void softmax_row(const float * x, int n, float * out) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, x[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) { out[i] = std::exp((double) (x[i] - mx)); sum += out[i]; }
    for (int i = 0; i < n; ++i) out[i] = (float) (out[i] / sum);
}

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

// Apply the reference '+'-position mapping: take the stress positions from the
// (lower-case) dict value and insert '+' into the original-cased word.
std::string map_stress(const std::string & word, const std::string & stressed) {
    auto wcps = utf8_decode(word);
    auto scps = utf8_decode(stressed);
    std::vector<size_t> plus_pos;
    for (size_t i = 0; i < scps.size(); ++i) if (scps[i] == '+') plus_pos.push_back(i);
    std::vector<uint32_t> wf = wcps;
    size_t j = 0;
    for (size_t p : plus_pos) {
        std::vector<uint32_t> nf;
        size_t cut = p + j;
        if (cut > wf.size()) cut = wf.size();
        for (size_t i = 0; i < cut; ++i) nf.push_back(wf[i]);
        nf.push_back('+');
        for (size_t i = p; i < wcps.size(); ++i) nf.push_back(wcps[i]);
        wf = std::move(nf);
        ++j;
    }
    return utf8_encode(wf);
}

// re.sub(r'\s+(?=(?:[,.?!:;…]))', '', text) -- drop whitespace before punctuation.
std::string preprocess_omograph_text(const std::string & text) {
    auto cps = utf8_decode(text);
    std::vector<uint32_t> out;
    for (size_t i = 0; i < cps.size(); ++i) {
        if (is_space_cp(cps[i])) {
            size_t j = i;
            while (j < cps.size() && is_space_cp(cps[j])) ++j;
            if (j < cps.size() && is_omograph_punct(cps[j])) { i = j - 1; continue; }
        }
        out.push_back(cps[i]);
    }
    return utf8_encode(out);
}

}  // namespace

RUAccent::~RUAccent() = default;

bool RUAccent::load(const std::string & data_dir, const std::string & models_dir) {
    if (!accent_.load(models_dir + "/accent.gguf"))         return false;
    if (!stress_.load(models_dir + "/stress_usage.gguf"))   return false;
    if (!omograph_.load(models_dir + "/omograph.gguf"))     return false;
    if (!yo_.load(models_dir + "/yo_homograph.gguf"))       return false;
    if (!accent_tok_.load(data_dir + "/nn/nn_accent/vocab.txt"))                    return false;
    if (!stress_tok_.load(data_dir + "/nn/nn_stress_usage_predictor/vocab.txt"))    return false;
    if (!yo_tok_.load(data_dir + "/nn/nn_yo_homograph_resolver/vocab.txt"))         return false;
    if (!omograph_tok_.load(data_dir + "/nn/nn_omograph/turbo3.1"))                 return false;
    if (!accents_.load(data_dir + "/dictionary/accents.bin"))                       return false;
    if (!omographs_.load(data_dir + "/dictionary/omographs.bin"))                   return false;
    if (!yo_words_.load(data_dir + "/dictionary/yo_words.bin"))                     return false;
    if (!yo_homographs_.load(data_dir + "/dictionary/yo_homographs.bin"))           return false;
    loaded_ = true;
    return true;
}

std::string RUAccent::process(const std::string & text) const {
    std::string t = normalize(text);
    auto sentences = split_by_sentences(t);
    std::string out;
    for (auto & sentence : sentences) {
        std::vector<std::string> words, remaining;
        split_by_words(sentence, words, remaining);
        if (words.empty()) {
            for (auto & r : remaining) out += r;
            continue;
        }
        std::vector<std::string> stress_usages = wordpiece_predict(sentence, stress_, stress_tok_);
        auto pw = process_yo(words, sentence);
        pw = process_omographs(pw);
        pw = process_accent(pw, stress_usages);
        std::string s;
        for (size_t i = 0; i < pw.size(); ++i) { s += remaining[i]; s += pw[i]; }
        s += remaining.back();
        out += delete_spaces_before_punc(s);
    }
    return out;
}

std::string RUAccent::normalize(const std::string & text) const {
    auto cps = utf8_decode(text);
    std::vector<uint32_t> out;
    for (uint32_t cp : cps) {
        bool keep = false;
        if (cp >= 'a' && cp <= 'z') keep = true;
        else if (cp >= 'A' && cp <= 'Z') keep = true;
        else if (cp >= '0' && cp <= '9') keep = true;
        else if (is_space_cp(cp)) keep = true;
        else if (cp >= 0x430 && cp <= 0x44F) keep = true;   // а-я
        else if (cp >= 0x410 && cp <= 0x42F) keep = true;   // А-Я
        else if (cp == 0x451 || cp == 0x401) keep = true;   // ё Ё
        else {
            switch (cp) {
                case 0x2014:                                   // —
                case '.': case ',': case '!': case '?': case ':': case ';':
                case '"': case '\'': case '(': case ')': case '{': case '}':
                case '[': case ']': case '-':
                case 0x00AB: case 0x00BB:                       // « »
                case 0x201E: case 0x201C: case 0x201D:         // „ " "
                case 0x2018: case 0x2019:                       // ' '
                    keep = true; break;
                default: keep = false;
            }
        }
        if (keep) out.push_back(cp);
    }
    return utf8_encode(out);
}

std::vector<std::string> RUAccent::split_by_sentences(const std::string & text) const {
    // Approximation of razdel.sentenize: split after sentence-final punctuation
    // when followed by whitespace (the whitespace starts the next sentence) or
    // at end of input.
    auto cps = utf8_decode(text);
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i < cps.size(); ++i) {
        cur += utf8_encode({cps[i]});
        bool final_punct = (cps[i] == '.' || cps[i] == '!' || cps[i] == '?' || cps[i] == 0x2026);
        if (final_punct && (i + 1 >= cps.size() || is_space_cp(cps[i + 1]))) {
            out.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

void RUAccent::split_by_words(const std::string & s_in,
                              std::vector<std::string> & words,
                              std::vector<std::string> & remaining) const {
    std::string s = replace_all(s_in, " - ", " ~ ");
    auto cps = utf8_decode(s);
    size_t n = cps.size();
    std::vector<size_t> boff(n + 1, 0);
    size_t b = 0;
    for (size_t i = 0; i < n; ++i) { boff[i] = b; b += utf8_encode({cps[i]}).size(); }
    boff[n] = b;

    // Non-empty matches of  \w*(?:\+\w+)*  |  [^\w\s]+  , in order.
    std::vector<std::pair<size_t, size_t>> spans;
    size_t i = 0;
    while (i < n) {
        uint32_t cp = cps[i];
        if (is_space_cp(cp)) { ++i; continue; }
        if (is_word_cp(cp)) {
            size_t j = i;
            while (j < n && is_word_cp(cps[j])) ++j;
            bool more = true;
            while (more) {
                more = false;
                if (j < n && cps[j] == '+') {
                    size_t k = j + 1;
                    if (k < n && is_word_cp(cps[k])) {
                        while (k < n && is_word_cp(cps[k])) ++k;
                        j = k;
                        more = true;
                    }
                }
            }
            spans.push_back({i, j});
            i = j;
        } else {
            size_t j = i;
            while (j < n && !is_space_cp(cps[j]) && !is_word_cp(cps[j])) ++j;
            spans.push_back({i, j});
            i = j;
        }
    }

    words.clear();
    if (spans.empty()) { remaining = {"", ""}; return; }
    for (auto & sp : spans)
        words.push_back(s.substr(boff[sp.first], boff[sp.second] - boff[sp.first]));

    remaining.clear();
    remaining.push_back(s.substr(0, boff[spans.front().first]));
    for (size_t k = 0; k + 1 < spans.size(); ++k)
        remaining.push_back(s.substr(boff[spans[k].second], boff[spans[k + 1].first] - boff[spans[k].second]));
    remaining.push_back(s.substr(boff[spans.back().second]));
}

std::string RUAccent::delete_spaces_before_punc(const std::string & text) const {
    std::string s = text;
    const char * punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-";
    for (const char * p = punc; *p; ++p) {
        char c = *p;
        if (c == '-') {
            s = replace_all(s, " -", "-");
            s = replace_all(s, "- ", "-");
        }
        s = replace_all(s, std::string(" ") + c, std::string(1, c));
    }
    return replace_all(s, "~", "-");
}

int RUAccent::count_vowels(const std::string & text) const {
    auto cps = utf8_decode(text);
    int cnt = 0;
    for (uint32_t cp : cps) {
        uint32_t l = to_lower_cp(cp);
        if (l == 0x430 || l == 0x435 || l == 0x451 || l == 0x438 || l == 0x43E ||
            l == 0x443 || l == 0x44B || l == 0x44C || l == 0x44E || l == 0x44F) ++cnt;
    }
    return cnt;
}

bool RUAccent::has_punctuation(const std::string & text) const {
    auto cps = utf8_decode(text);
    static const char * set = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    for (uint32_t cp : cps) {
        if (cp >= 128) continue;
        for (const char * p = set; *p; ++p)
            if (*p == (char) cp) return true;
    }
    return false;
}

std::string RUAccent::fix_capital(const std::string & source, const std::string & target) const {
    auto scps = utf8_decode(source);
    auto tcps = utf8_decode(target);
    if (scps.size() != tcps.size()) return target;
    std::vector<uint32_t> out;
    out.reserve(scps.size());
    for (size_t i = 0; i < scps.size(); ++i)
        out.push_back(is_upper_cp(scps[i]) ? to_upper_cp(tcps[i]) : to_lower_cp(tcps[i]));
    return utf8_encode(out);
}

std::vector<std::string> RUAccent::wordpiece_predict(
        const std::string & sentence, const Encoder & enc, const WordPieceTokenizer & tok) const {
    TokOut t = tok.tokenize(sentence);
    int seq = (int) t.input_ids.size();
    EncoderInput in;
    in.input_ids = t.input_ids;
    in.attention_mask = t.attention_mask;
    if (enc.has_type_embd()) in.token_type_ids = t.token_type_ids;
    in.batch = 1;
    in.seq = seq;
    EncoderOutput out = enc.forward(in);
    int n_labels = enc.n_labels();
    const auto & labels = enc.labels();

    struct Pre { std::vector<float> scores; bool is_subword; };
    std::vector<Pre> pres;
    for (int idx = 0; idx < seq; ++idx) {
        if (idx < (int) t.special_mask.size() && t.special_mask[idx]) continue;
        // A token is a subword continuation iff its vocab form carries the "##"
        // prefix (matches the reference len(token) != len(word_ref) test; an
        // adjacent punctuation token has no prefix and stays its own word).
        const std::string & tok_str = tok.id_to_token(t.input_ids[idx]);
        bool is_sub = (tok_str.size() >= 2 && tok_str[0] == '#' && tok_str[1] == '#');
        std::vector<float> scores(n_labels);
        softmax_row(&out.logits[idx * n_labels], n_labels, scores.data());
        pres.push_back({std::move(scores), is_sub});
    }

    auto aggregate = [&](size_t a, size_t b) -> std::string {
        std::vector<float> avg(n_labels, 0.0f);
        int cnt = (int) (b - a);
        for (size_t i = a; i < b; ++i)
            for (int c = 0; c < n_labels; ++c) avg[c] += pres[i].scores[c];
        for (int c = 0; c < n_labels; ++c) avg[c] /= cnt;
        int best = 0;
        for (int c = 1; c < n_labels; ++c) if (avg[c] > avg[best]) best = c;
        return labels[best];
    };

    std::vector<std::string> result;
    size_t start = 0;
    for (size_t i = 1; i <= pres.size(); ++i) {
        if (i == pres.size() || !pres[i].is_subword) {
            result.push_back(aggregate(start, i));
            start = i;
        }
    }
    return result;
}

std::vector<std::string> RUAccent::process_yo(std::vector<std::string> words,
                                              const std::string & sentence) const {
    std::string lower_sentence = to_lower(sentence);
    bool has_e_cp = false;
    {
        auto cps = utf8_decode(lower_sentence);
        for (uint32_t cp : cps) if (cp == 0x435) { has_e_cp = true; break; }   // е
    }
    std::vector<std::string> yo_predictions;
    if (has_e_cp) yo_predictions = wordpiece_predict(lower_sentence, yo_, yo_tok_);

    for (size_t i = 0; i < words.size(); ++i) {
        std::string orig = words[i];
        std::string lower_word = to_lower(orig);
        std::string yw;
        if (yo_words_.get(lower_word, yw)) words[i] = fix_capital(orig, yw);
        if (has_e_cp && i < yo_predictions.size() && yo_predictions[i] == "YO") {
            std::string yh;
            if (yo_homographs_.get(lower_word, yh)) words[i] = fix_capital(orig, yh);
        }
    }
    return words;
}

std::vector<std::string> RUAccent::process_omographs(std::vector<std::string> words) const {
    struct Found { size_t position; std::vector<std::string> variants; };
    std::vector<Found> founded;
    for (size_t i = 0; i < words.size(); ++i) {
        std::vector<std::string> variants;
        if (omographs_.get_list(words[i], variants)) founded.push_back({i, std::move(variants)});
    }
    if (founded.empty()) return words;

    std::vector<std::string> texts_batch, hypotheses_batch;
    std::vector<int> num_hypotheses;
    for (auto & f : founded) {
        std::string saved = words[f.position];
        words[f.position] = " <w>" + words[f.position] + "</w> ";
        std::string joined;
        for (size_t k = 0; k < words.size(); ++k) { if (k) joined += " "; joined += words[k]; }
        std::string pre = delete_spaces_before_punc(joined);
        for (auto & v : f.variants) { texts_batch.push_back(pre); hypotheses_batch.push_back(v); }
        words[f.position] = saved;
        num_hypotheses.push_back((int) f.variants.size());
    }

    // Per-omograph classification: pick the variant with the highest P(true).
    // Equivalent to the reference batch/NO_BATCH paths for the 2-variant case
    // (99.7% of the dict) and more robust for the rare 3/4-variant entries.
    size_t h = 0;
    for (size_t o = 0; o < founded.size(); ++o) {
        std::string text = preprocess_omograph_text(texts_batch[h]);
        int nh = num_hypotheses[o];
        float best_p = -1.0f;
        int best_i = 0;
        for (int v = 0; v < nh; ++v) {
            TokOut t = omograph_tok_.tokenize_pair(text, hypotheses_batch[h + v]);
            EncoderInput in;
            in.input_ids = t.input_ids;
            in.attention_mask = t.attention_mask;
            in.batch = 1;
            in.seq = (int) t.input_ids.size();
            EncoderOutput out = omograph_.forward(in);
            float probs[2];
            softmax_row(out.logits.data(), 2, probs);
            if (probs[1] > best_p) { best_p = probs[1]; best_i = v; }
        }
        words[founded[o].position] = hypotheses_batch[h + best_i];
        h += nh;
    }
    return words;
}

std::vector<std::string> RUAccent::process_accent(std::vector<std::string> words,
                                                  const std::vector<std::string> & stress_usages) const {
    for (size_t i = 0; i < words.size(); ++i) {
        if (words[i].find('+') != std::string::npos) continue;
        if (i >= stress_usages.size() || stress_usages[i] != "STRESS") continue;
        std::string lower_word = to_lower(words[i]);
        std::string stressed_word;
        if (!accents_.get(lower_word, stressed_word)) stressed_word = lower_word;
        if (stressed_word == lower_word && !has_punctuation(lower_word) && count_vowels(lower_word) > 1) {
            words[i] = put_accent(words[i]);
        } else {
            words[i] = map_stress(words[i], stressed_word);
        }
    }
    return words;
}

std::string RUAccent::put_accent(const std::string & word) const {
    std::string lower = to_lower(word);
    TokOut t = accent_tok_.tokenize(lower);
    int seq = (int) t.input_ids.size();
    EncoderInput in;
    in.input_ids = t.input_ids;
    in.attention_mask = t.attention_mask;
    if (accent_.has_type_embd()) in.token_type_ids = t.token_type_ids;
    in.batch = 1;
    in.seq = seq;
    EncoderOutput out = accent_.forward(in);
    int n_labels = accent_.n_labels();
    int stress_primary = -1;
    for (int c = 0; c < n_labels; ++c)
        if (accent_.labels()[c] == "STRESS_PRIMARY") stress_primary = c;

    auto wcps = utf8_decode(word);
    std::vector<uint32_t> res;
    res.reserve(wcps.size() + 1);
    for (size_t j = 0; j < wcps.size(); ++j) {
        int model_pos = (int) (j + 1);   // after [bos]
        if (stress_primary >= 0 && model_pos < seq) {
            std::vector<float> probs(n_labels);
            softmax_row(&out.logits[model_pos * n_labels], n_labels, probs.data());
            if (probs[stress_primary] >= 0.55f) res.push_back('+');
        }
        res.push_back(wcps[j]);
    }
    return utf8_encode(res);
}

}  // namespace ru
